import os
import django
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings as _django_settings
_db = _django_settings.DATABASES["default"]
print(f"[DB] host={_db.get('HOST', '?')} name={_db.get('NAME', '?')}")

import pandas as pd
from accidents.models import AccidentRecord, HotspotCluster
from predictions.dbscan import haversine_distance
from predictions.decision_tree import prepare_features, build_tree, bin_proximity

# ── Load full UK_STATS19 dataset ───────────────────────────────────────────────
qs = AccidentRecord.objects.filter(source="UK_STATS19").values(
    "latitude", "longitude",
    "day_of_week", "weather_condition", "road_type",
    "light_condition", "speed_limit", "junction_type", "severity", "time"
)
df = pd.DataFrame(list(qs))
print(f"Loaded {len(df)} UK_STATS19 records.")

# ── Compute proximity_to_hotspot ───────────────────────────────────────────────
centroids = list(HotspotCluster.objects.values_list("centroid_latitude", "centroid_longitude"))
print(f"Loaded {len(centroids)} cluster centroids.")

def _nearest_dist(row):
    if not centroids:
        return float("inf")
    return min(haversine_distance(row["latitude"], row["longitude"], c_lat, c_lon)
               for c_lat, c_lon in centroids)

df["proximity_to_hotspot"] = df.apply(
    lambda row: bin_proximity(_nearest_dist(row)), axis=1
)

# ── Train on full dataset ──────────────────────────────────────────────────────
features, labels = prepare_features(df)
attribute_names = list(features[0].keys())

print(f"Training on {len(features)} records with features: {attribute_names}")
tree = build_tree(features, labels, attribute_names, max_depth=6, min_samples=5)

# ── Serialise ──────────────────────────────────────────────────────────────────
tree_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "predictions", "trained_tree.pkl"
)
with open(tree_path, "wb") as f:
    pickle.dump(tree, f)
print(f"Tree saved to {tree_path}")
