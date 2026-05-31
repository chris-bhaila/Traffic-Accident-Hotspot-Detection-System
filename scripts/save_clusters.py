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
from predictions.dbscan import dbscan

# Load points with full data
accidents = list(
    AccidentRecord.objects.filter(source="KTM_SYNTHETIC").values(
        "id", "latitude", "longitude", "severity", "weather_condition",
        "day_of_week", "time"
    )
)

points = np.array([[a["latitude"], a["longitude"]] for a in accidents])
print(f"Loaded {len(points)} points")

# Run DBSCAN
labels = dbscan(points, epsilon=150, min_samples=5)

# Clear existing clusters
HotspotCluster.objects.all().delete()

# Build and save cluster summaries
unique_labels = set(labels)
unique_labels.discard(-1)

for cluster_id in sorted(unique_labels):
    mask = labels == cluster_id
    cluster_accidents = [accidents[i] for i in range(len(accidents)) if mask[i]]
    cluster_points = points[mask]

    # Centroid
    centroid_lat = float(cluster_points[:, 0].mean())
    centroid_lon = float(cluster_points[:, 1].mean())

    # Radius (max distance from centroid to any point in cluster)
    from predictions.dbscan import haversine_distance
    distances = [
        haversine_distance(centroid_lat, centroid_lon, p[0], p[1])
        for p in cluster_points
    ]
    radius = float(max(distances)) if distances else 0

    # Severity breakdown
    severities = [a["severity"] for a in cluster_accidents]
    fatal = severities.count("Fatal")
    serious = severities.count("Serious")
    slight = severities.count("Slight")
    total = len(severities)

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
    # Frequency + severity weighted
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

print(f"\nSaved {len(unique_labels)} clusters to database")