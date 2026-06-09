import json as _json
import logging
import os
import pickle
import urllib.request
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from accidents.models import HotspotCluster
from predictions.dbscan import haversine_distance
from predictions.decision_tree import (
    bin_proximity,
    bin_speed,
    bin_time,
    predict,
)

logger = logging.getLogger(__name__)

# ── Module-level state ─────────────────────────────────────────────────────────

TREE_PATH = os.path.join(os.path.dirname(__file__), "trained_tree.pkl")
with open(TREE_PATH, "rb") as f:
    TRAINED_TREE = pickle.load(f)

CLUSTER_CENTROIDS = list(
    HotspotCluster.objects.values_list("centroid_latitude", "centroid_longitude")
)

# ── Shared helpers ─────────────────────────────────────────────────────────────

def get_nearest_cluster_distance(lat, lon, centroids):
    """Return Haversine distance in metres to the nearest cluster centroid."""
    if not centroids:
        return float("inf")
    return min(haversine_distance(lat, lon, c_lat, c_lon) for c_lat, c_lon in centroids)


def _wmo_to_weather_bin(code):
    if code in (0, 1):
        return "fine"
    if code in (2, 3, 45, 48):
        return "other"
    if 51 <= code <= 67 or 80 <= code <= 82:
        return "rain"
    if 71 <= code <= 77 or code in (85, 86):
        return "snow"
    if 95 <= code <= 99:
        return "rain"
    return "fine"


def _wmo_to_description(code):
    if code == 0:              return "Clear Sky"
    if code == 1:              return "Mainly Clear"
    if code == 2:              return "Partly Cloudy"
    if code == 3:              return "Overcast"
    if code in (45, 48):       return "Foggy"
    if code in (51, 53):       return "Light Drizzle"
    if code in (55, 56, 57):   return "Heavy Drizzle"
    if code == 61:             return "Light Rain"
    if code == 63:             return "Moderate Rain"
    if code in (65, 66, 67):   return "Heavy Rain"
    if code in (71, 73):       return "Light Snow"
    if code in (75, 77):       return "Heavy Snow"
    if code in (80, 81):       return "Light Showers"
    if code == 82:             return "Heavy Showers"
    if code in (85, 86):       return "Snow Showers"
    if 95 <= code <= 99:       return "Thunderstorm"
    return "Cloudy"


def get_realtime_weather(lat, lon):
    """Return (weather_bin, description) from Open-Meteo. Falls back to ('fine', 'Clear Sky')."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current=weathercode&timezone=auto"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read().decode())
        code = data["current"].get("weathercode") or data["current"].get("weather_code", 0)
        code = int(code)
        return _wmo_to_weather_bin(code), _wmo_to_description(code)
    except Exception as exc:
        logger.warning(
            "Open-Meteo API failed for (%.4f, %.4f): %s — falling back to 'fine'",
            lat, lon, exc,
        )
        return "fine", "Clear Sky"


# ── Route narrative helpers ────────────────────────────────────────────────────

def _build_summary(counts, risk_level):
    fatal    = counts.get("Fatal", 0)
    serious  = counts.get("Serious", 0)
    high_zones = fatal + serious

    if risk_level == "CRITICAL":
        return "High accident density detected along this route. Exercise extreme caution throughout."
    if risk_level == "HIGH":
        return (
            f"Your route passes through {high_zones} high-risk zone{'s' if high_zones != 1 else ''}. "
            "Consider an alternative route if available."
        )
    if risk_level == "MEDIUM":
        if serious > 0:
            return (
                f"Route contains {serious} serious risk segment{'s' if serious != 1 else ''}. "
                "Exercise caution near junctions."
            )
        return "Moderate risk detected along parts of this route. Drive with care."
    # LOW
    if fatal == 0 and serious == 0:
        return "Route appears mostly safe with no serious risk segments detected."
    return "Route appears mostly safe with only minor risk segments detected."


def _peak_risk_segments(segments):
    total   = len(segments) or 1
    fatals  = [s for s in segments if s["prediction"] == "Fatal"]
    serious = [s for s in segments if s["prediction"] == "Serious"]
    top = fatals + serious
    out = []
    for s in top:
        prox = s.get("proximity_bin", "no_hotspot")
        frac = s.get("seg_index", 0) / total
        position = "early" if frac < 0.33 else ("mid-route" if frac < 0.67 else "late")
        desc = f"{s['prediction']} risk · {prox} · {position} on route"
        out.append({"lat": s["lat"], "lon": s["lon"], "prediction": s["prediction"], "description": desc})
    return out


def _safety_tip(time_bin, weather_bin, hour):
    is_night        = hour >= 22 or hour <= 5
    is_morning_rush = 7 <= hour <= 9
    is_evening_rush = 16 <= hour <= 19

    if weather_bin == "snow":
        return "Snow or icy conditions detected. Reduce speed significantly and increase stopping distance."
    if weather_bin == "fog":
        return "Reduced visibility due to fog. Use fog lights and slow down on approach to junctions."
    if is_night:
        return "Night-time driving detected. Stay alert for pedestrians and cyclists in low-lit areas."
    if is_evening_rush and weather_bin == "rain":
        return "Evening rush hour with wet conditions detected. Increase following distance and reduce speed on bends."
    if is_morning_rush and weather_bin == "rain":
        return "Morning commute with rain detected. Allow extra journey time and watch for standing water."
    if is_evening_rush:
        return "Evening rush hour detected. Expect higher traffic volumes and increased pedestrian activity."
    if is_morning_rush:
        return "Morning rush hour. Exercise caution at school zones and busy junctions."
    return "Good visibility and moderate conditions. Maintain standard following distances."


def _time_context(day, hour):
    if 7 <= hour <= 9:      period = "Morning Rush"
    elif 10 <= hour <= 14:  period = "Midday"
    elif 15 <= hour <= 19:  period = "Evening Rush"
    elif 20 <= hour <= 21:  period = "Late Evening"
    else:                   period = "Night"
    return f"{day} · {hour:02d}:00 · {period}"


def _route_recommendation(routes):
    if len(routes) <= 1:
        return "Only one route available."
    scores = [r["risk_score"] for r in routes]
    if max(scores) - min(scores) < 0.15:
        return "All routes have similar risk profiles. The fastest route is recommended."
    safest  = routes[0]   # sorted ascending by risk_score
    fastest = min(routes, key=lambda r: r["duration_seconds"])
    if safest is fastest:
        return "Route 1 is both the safest and fastest option."
    fastest_idx = routes.index(fastest) + 1
    diff        = safest["duration_seconds"] - fastest["duration_seconds"]
    extra_min   = round(diff / 60)
    if extra_min > 0:
        time_str = f"{extra_min} min longer"
    elif extra_min < 0:
        time_str = f"{abs(extra_min)} min shorter"
    else:
        time_str = "same journey time"
    pct_safer = round(100 * (fastest["risk_score"] - safest["risk_score"]) / fastest["risk_score"])
    return (
        f"Route 1 is recommended — {time_str} but {pct_safer}% lower risk score "
        f"than Route {fastest_idx}."
    )


# ── Point prediction ───────────────────────────────────────────────────────────

def predict_risk(request):
    """
    GET /api/predict/?lat=&lon=&time=&day=&road=&speed=&light=
    Weather auto-detected from Open-Meteo; any 'weather' param is ignored.
    """
    try:
        lat   = float(request.GET.get("lat", 0))
        lon   = float(request.GET.get("lon", 0))
        hour  = int(request.GET.get("time", 12))
        day   = request.GET.get("day", "Monday")
        road  = request.GET.get("road", "single")
        speed = int(request.GET.get("speed", 30))
        light = request.GET.get("light", "daylight")

        proximity_m         = get_nearest_cluster_distance(lat, lon, CLUSTER_CENTROIDS)
        weather, _          = get_realtime_weather(lat, lon)

        sample = {
            "time_period":          bin_time(hour),
            "day_of_week":          day,
            "weather":              weather,
            "road_type":            road,
            "light":                light,
            "speed":                bin_speed(speed),
            "proximity_to_hotspot": bin_proximity(proximity_m),
        }

        prediction, path = predict(TRAINED_TREE, sample)

        return JsonResponse({
            "status":           "success",
            "prediction":       prediction,
            "decision_path":    path,
            "input":            sample,
            "location":         {"lat": lat, "lon": lon},
            "weather":          weather,
            "weather_source":   "realtime",
            "proximity_metres": round(proximity_m, 1) if proximity_m != float("inf") else None,
        })

    except Exception as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=400)


# ── Route risk analysis ────────────────────────────────────────────────────────

def _sample_route(coords, step_metres=200):
    """
    Walk a GeoJSON coordinate list ([lon, lat] pairs) and return (lat, lon) sample
    points spaced approximately step_metres apart using linear interpolation.
    """
    if not coords:
        return []
    if len(coords) == 1:
        return [(coords[0][1], coords[0][0])]

    sampled = []
    prev_lon, prev_lat = coords[0]
    sampled.append((prev_lat, prev_lon))
    next_threshold = step_metres
    cumulative = 0.0

    for lon, lat in coords[1:]:
        seg_len = haversine_distance(prev_lat, prev_lon, lat, lon)
        if seg_len == 0:
            prev_lat, prev_lon = lat, lon
            continue

        while cumulative + seg_len >= next_threshold:
            t = (next_threshold - cumulative) / seg_len
            s_lat = prev_lat + t * (lat - prev_lat)
            s_lon = prev_lon + t * (lon - prev_lon)
            sampled.append((s_lat, s_lon))
            next_threshold += step_metres

        cumulative += seg_len
        prev_lat, prev_lon = lat, lon

    return sampled


def _avg_score_to_risk_level(avg):
    if avg >= 2.2:
        return "CRITICAL"
    if avg >= 1.8:
        return "HIGH"
    if avg >= 1.4:
        return "MEDIUM"
    return "LOW"

def debug_centroids(request):
    from django.http import JsonResponse
    from accidents.models import HotspotCluster
    db_clusters = list(HotspotCluster.objects.values('centroid_latitude', 'centroid_longitude')[:5])
    return JsonResponse({
        'cluster_centroids_loaded': len(CLUSTER_CENTROIDS),
        'first_5_centroids': CLUSTER_CENTROIDS[:5],
        'first_5_from_db': db_clusters,
    })

@csrf_exempt
def route_risk(request):
    """
    POST /api/route-risk/
    Body: {"origin": [lat, lon], "destination": [lat, lon]}
    Returns up to 3 OSRM routes scored segment-by-segment, sorted safest first.
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"}, status=405)

    try:
        body        = _json.loads(request.body)
        orig_lat, orig_lon = body["origin"]
        dest_lat, dest_lon = body["destination"]
    except (KeyError, ValueError, TypeError) as exc:
        return JsonResponse({"status": "error", "message": f"Invalid body: {exc}"}, status=400)

    # ── Fetch routes from OSRM ─────────────────────────────────────────────────
    osrm_url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{orig_lon},{orig_lat};{dest_lon},{dest_lat}"
        f"?alternatives=true&geometries=geojson&overview=full"
    )
    try:
        req = urllib.request.Request(osrm_url, headers={"User-Agent": "traffic-hotspot-fyp"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            osrm_data = _json.loads(resp.read().decode())
    except Exception as exc:
        return JsonResponse(
            {"status": "error", "message": f"OSRM routing failed: {exc}"},
            status=502,
        )

    if osrm_data.get("code") != "Ok" or not osrm_data.get("routes"):
        return JsonResponse({"status": "error", "message": "No route found between those points"}, status=404)

    # ── Current time features ──────────────────────────────────────────────────
    now          = datetime.now()
    current_hour = now.hour
    current_day  = now.strftime("%A")
    time_bin     = bin_time(current_hour)
    light_bin    = "daylight" if 7 <= current_hour <= 19 else "dark_lit"
    analysed_at  = now.isoformat(timespec="seconds")
    time_ctx     = _time_context(current_day, current_hour)

    severity_score = {"Slight": 1, "Serious": 2, "Fatal": 3}
    routes_out = []

    for osrm_route in osrm_data["routes"][:3]:
        geometry   = osrm_route["geometry"]          # GeoJSON LineString
        coords     = geometry["coordinates"]          # [[lon, lat], ...]
        distance_m = osrm_route["distance"]
        duration_s = osrm_route["duration"]

        sampled = _sample_route(coords, step_metres=200)
        if not sampled:
            continue

        # One weather call per route — use route centroid
        c_lat = sum(p[0] for p in sampled) / len(sampled)
        c_lon = sum(p[1] for p in sampled) / len(sampled)
        weather, weather_desc = get_realtime_weather(c_lat, c_lon)

        segments  = []
        score_sum = 0.0
        counts    = {"Slight": 0, "Serious": 0, "Fatal": 0}

        for pt_lat, pt_lon in sampled:
            prox_m   = get_nearest_cluster_distance(pt_lat, pt_lon, CLUSTER_CENTROIDS)
            prox_bin = bin_proximity(prox_m)
            sample = {
                "time_period":          time_bin,
                "day_of_week":          current_day,
                "weather":              weather,
                "road_type":            "single",
                "light":                light_bin,
                "speed":                bin_speed(30),   # urban default
                "proximity_to_hotspot": prox_bin,
            }
            prediction, _ = predict(TRAINED_TREE, sample)
            counts[prediction] = counts.get(prediction, 0) + 1
            score_sum += severity_score.get(prediction, 1)
            segments.append({
                "lat": pt_lat, "lon": pt_lon,
                "prediction": prediction,
                "proximity_bin": prox_bin,
                "seg_index": len(segments),
            })

        n          = len(segments) or 1
        avg_score  = score_sum / n
        risk_level = _avg_score_to_risk_level(avg_score)

        routes_out.append({
            "risk_level":          risk_level,
            "risk_score":          round(avg_score, 3),
            "distance_metres":     round(distance_m),
            "duration_seconds":    round(duration_s),
            "geometry":            geometry,
            "segment_counts":      counts,
            "segments":            segments,
            "summary":             _build_summary(counts, risk_level),
            "peak_risk_segments":  _peak_risk_segments(segments),
            "safety_tip":          _safety_tip(time_bin, weather, current_hour),
            "weather_description": weather_desc,
        })

    if not routes_out:
        return JsonResponse({"status": "error", "message": "Route scoring produced no results"}, status=500)

    routes_out.sort(key=lambda r: r["risk_score"])

    return JsonResponse({
        "status":         "success",
        "routes":         routes_out,
        "recommendation": _route_recommendation(routes_out),
        "analysed_at":    analysed_at,
        "time_context":   time_ctx,
    })
