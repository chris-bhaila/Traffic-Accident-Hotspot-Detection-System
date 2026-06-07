import os
import django
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings as _django_settings
_db = _django_settings.DATABASES["default"]
print(f"[DB] host={_db.get('HOST', '?')} name={_db.get('NAME', '?')}")

import numpy as np
from accidents.models import AccidentRecord, HotspotCluster
from predictions.dbscan import dbscan, haversine_distance


def compute_k_distances(points, k):
    """For each point compute its k-th nearest-neighbour Haversine distance.
    Returns a sorted numpy array of all k-th NN distances (ascending)."""
    n = len(points)
    k_distances = np.empty(n)
    for i in range(n):
        dists = np.array([
            haversine_distance(points[i, 0], points[i, 1], points[j, 0], points[j, 1])
            for j in range(n) if j != i
        ])
        dists.sort()
        k_distances[i] = dists[k - 1]
    k_distances.sort()
    return k_distances


def find_elbow(distances):
    """Find the elbow of the k-distance curve using maximum perpendicular distance
    from the line connecting the first and last points (both axes normalised to [0,1]).
    Returns (epsilon_value, index)."""
    n = len(distances)
    x = np.arange(n, dtype=float)
    y = distances.astype(float)

    x_range = x[-1] - x[0]
    y_range = y[-1] - y[0]
    x_norm = (x - x[0]) / x_range if x_range != 0 else x
    y_norm = (y - y[0]) / y_range if y_range != 0 else y

    line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    line_unit = line_vec / np.linalg.norm(line_vec)

    pts_vec = np.column_stack([x_norm - x_norm[0], y_norm - y_norm[0]])
    proj = pts_vec @ line_unit
    perp_dist = np.linalg.norm(pts_vec - np.outer(proj, line_unit), axis=1)

    elbow_idx = int(np.argmax(perp_dist))
    return float(distances[elbow_idx]), elbow_idx


MIN_SAMPLES = 5
EPSILON_MIN = 50.0
EPSILON_MAX = 400.0

# Load points with full data
accidents = list(
    AccidentRecord.objects.filter(source="UK_STATS19").values(
        "id", "latitude", "longitude", "severity", "weather_condition",
        "day_of_week", "time"
    )
)

points = np.array([[a["latitude"], a["longitude"]] for a in accidents])
print(f"Loaded {len(points)} points")

# Automatic epsilon selection via k-distance elbow method
k = MIN_SAMPLES - 1
print(f"Computing {k}-distances for {len(points)} points (this may take a moment)...")
k_dists = compute_k_distances(points, k)
epsilon, elbow_idx = find_elbow(k_dists)
total_k = len(k_dists)
print(f"Elbow method selected epsilon: {epsilon:.1f}m (at k-distance index {elbow_idx}/{total_k})")

if epsilon < EPSILON_MIN:
    print(
        f"WARNING: computed epsilon {epsilon:.1f}m is below {EPSILON_MIN}m "
        f"(dataset may be too dense) — clamping to {EPSILON_MIN}m"
    )
    epsilon = EPSILON_MIN
elif epsilon > EPSILON_MAX:
    print(
        f"WARNING: computed epsilon {epsilon:.1f}m is above {EPSILON_MAX}m "
        f"(dataset may be too sparse) — clamping to {EPSILON_MAX}m"
    )
    epsilon = EPSILON_MAX

# Run DBSCAN
labels = dbscan(points, epsilon=epsilon, min_samples=MIN_SAMPLES)

# Clear existing clusters
HotspotCluster.objects.all().delete()

# Build and save cluster summaries
unique_labels = set(labels)
unique_labels.discard(-1)

cluster_sizes = []

for cluster_id in sorted(unique_labels):
    mask = labels == cluster_id
    cluster_accidents = [accidents[i] for i in range(len(accidents)) if mask[i]]
    cluster_points = points[mask]

    # Centroid
    centroid_lat = float(cluster_points[:, 0].mean())
    centroid_lon = float(cluster_points[:, 1].mean())

    # Radius (max distance from centroid to any point in cluster)
    centroid_dists = [
        haversine_distance(centroid_lat, centroid_lon, p[0], p[1])
        for p in cluster_points
    ]
    radius = float(max(centroid_dists)) if centroid_dists else 0

    # Severity breakdown
    severities = [a["severity"] for a in cluster_accidents]
    total = len(severities)
    cluster_sizes.append(total)

    # Average severity score (Fatal=3, Serious=2, Slight=1)
    severity_map = {"Fatal": 3, "Serious": 2, "Slight": 1}
    avg_severity = np.mean([severity_map.get(s, 1) for s in severities])

    # Peak day
    days = [a["day_of_week"] for a in cluster_accidents]
    peak_day = max(set(days), key=days.count)

    # Peak time (hour)
    times = [a["time"] for a in cluster_accidents if a["time"]]
    if times:
        hours = [t.hour for t in times]
        peak_hour = max(set(hours), key=hours.count)
        peak_time = f"{peak_hour:02d}:00"
    else:
        peak_time = "Unknown"

    # Dominant weather
    weathers = [a["weather_condition"] for a in cluster_accidents]
    dominant_weather = max(set(weathers), key=weathers.count)

    # Risk level based on composite score
    if total >= 20 and avg_severity >= 2.0:
        risk_level = "CRITICAL"
    elif total >= 15 or (total >= 10 and avg_severity >= 1.8):
        risk_level = "HIGH"
    elif total >= 8 or (total >= 5 and avg_severity >= 1.5):
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    HotspotCluster.objects.create(
        centroid_latitude=centroid_lat,
        centroid_longitude=centroid_lon,
        accident_count=total,
        radius=radius,
        average_severity=float(avg_severity),
        peak_time=peak_time,
        peak_day=peak_day,
        dominant_weather=dominant_weather,
        risk_level=risk_level,
        district="Leeds",
    )

    print(f"Cluster {cluster_id}: {total} accidents, {risk_level} risk, peak: {peak_day} {peak_time}")

n_clusters = len(unique_labels)
n_noise = int(np.sum(labels == -1))
print(f"\nSaved {n_clusters} clusters to database")

# Run summary
print("\n--- Run Summary ---")
print(f"  Points loaded   : {len(points)}")
print(f"  k (NN for elbow): {k}")
print(f"  Epsilon used    : {epsilon:.1f}m")
print(f"  Clusters found  : {n_clusters}")
print(f"  Noise points    : {n_noise}")
if cluster_sizes:
    buckets = {}
    for s in cluster_sizes:
        lo = (s // 10) * 10
        label = f"{lo}-{lo + 9}"
        buckets[label] = buckets.get(label, 0) + 1
    print(
        f"  Cluster sizes   : min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
        f"mean={np.mean(cluster_sizes):.1f}, median={np.median(cluster_sizes):.1f}"
    )
    print(f"  Size distribution (by 10s): {dict(sorted(buckets.items()))}")
