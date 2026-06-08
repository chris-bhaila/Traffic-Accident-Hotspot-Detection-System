import json as _json
import logging
import os
import pickle
import urllib.request

from django.http import JsonResponse

from accidents.models import HotspotCluster
from predictions.dbscan import haversine_distance
from predictions.decision_tree import (
    bin_light,
    bin_proximity,
    bin_road,
    bin_speed,
    bin_time,
    predict,
)

logger = logging.getLogger(__name__)

# Load tree once at module level
TREE_PATH = os.path.join(os.path.dirname(__file__), "trained_tree.pkl")
with open(TREE_PATH, "rb") as f:
    TRAINED_TREE = pickle.load(f)

# Load cluster centroids once at module level
CLUSTER_CENTROIDS = list(
    HotspotCluster.objects.values_list("centroid_latitude", "centroid_longitude")
)


def get_nearest_cluster_distance(lat, lon, centroids):
    """Return Haversine distance in metres to the nearest cluster centroid."""
    if not centroids:
        return float("inf")
    return min(haversine_distance(lat, lon, c_lat, c_lon) for c_lat, c_lon in centroids)


def _wmo_to_weather_bin(code):
    """Map a WMO weather code integer to an existing weather bin value."""
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


def get_realtime_weather(lat, lon):
    """Fetch current weather from Open-Meteo; return a weather bin string.

    Falls back to 'fine' on any API or network failure.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current=weathercode&timezone=auto"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read().decode())
        code = data["current"].get("weathercode") or data["current"].get("weather_code", 0)
        return _wmo_to_weather_bin(int(code))
    except Exception as exc:
        logger.warning(
            "Open-Meteo API failed for (%.4f, %.4f): %s — falling back to 'fine'",
            lat, lon, exc,
        )
        return "fine"


def predict_risk(request):
    """
    GET /api/predict/?lat=53.80&lon=-1.55&time=17&day=Friday&road=single&speed=30&light=dark_lit

    Weather is auto-detected from Open-Meteo; any 'weather' query param is ignored.
    """
    try:
        lat   = float(request.GET.get("lat", 0))
        lon   = float(request.GET.get("lon", 0))
        hour  = int(request.GET.get("time", 12))
        day   = request.GET.get("day", "Monday")
        road  = request.GET.get("road", "single")
        speed = int(request.GET.get("speed", 30))
        light = request.GET.get("light", "daylight")

        proximity_m = get_nearest_cluster_distance(lat, lon, CLUSTER_CENTROIDS)
        weather     = get_realtime_weather(lat, lon)

        sample = {
            "time_period":        bin_time(hour),
            "day_of_week":        day,
            "weather":            weather,
            "road_type":          road,
            "light":              light,
            "speed":              bin_speed(speed),
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
