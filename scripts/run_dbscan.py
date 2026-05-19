import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

import numpy as np
from accidents.models import AccidentRecord
from predictions.dbscan import dbscan

#Load Leeds accident coordinates
accidents = AccidentRecord.objects.filter(source="UK_STATS19").values_list("latitude", "longitude")
points = np.array(list(accidents))
print(f"Loaded {len(points)} accident points")

# Run DBSCAN
# epsilon = 300 meters, min_samples=5
labels = dbscan(points, epsilon=150, min_samples=5)

#Summary
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print(f"\nResults:")
print(f" Clusters found: {n_clusters}")
print(f" Noise points: {n_noise}")
print(f" Clustered points: {len(labels) - n_noise}")

#Show top clusters by size
for cluster_id in range(min(n_clusters, 10)):
    cluster_points = points[labels == cluster_id]
    centroid_lat = cluster_points[:, 0].mean()
    centroid_lon = cluster_points[:, 1].mean()
    print(f" Cluster {cluster_id}: {len(cluster_points)} points, centroid: ({centroid_lat:.4f}, {centroid_lon:.4f})")