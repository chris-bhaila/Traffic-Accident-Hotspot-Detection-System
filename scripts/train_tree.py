import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

import pandas as pd
import numpy as np
from collections import Counter
from accidents.models import AccidentRecord, HotspotCluster
from predictions.dbscan import haversine_distance
from predictions.decision_tree import (
    prepare_features,
    build_tree,
    predict_batch,
    print_tree,
    evaluate,
    bin_proximity,
)

# ── Load data ──────────────────────────────────────────────────────────────────
qs = AccidentRecord.objects.filter(source="UK_STATS19").values(
    "latitude", "longitude",
    "day_of_week", "weather_condition", "road_type",
    "light_condition", "speed_limit", "junction_type", "severity", "time"
)
df = pd.DataFrame(list(qs))
print(f"Total records: {len(df)}")

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
print(f"Proximity distribution:\n{Counter(df['proximity_to_hotspot'].tolist())}")

# ── Prepare features ───────────────────────────────────────────────────────────
features, labels = prepare_features(df)
print(f"Features prepared: {len(features)}")
print(f"Feature keys: {list(features[0].keys())}")

# ── Bin distributions ──────────────────────────────────────────────────────────
print("\n--- Bin distributions ---")
for key in features[0].keys():
    vals = [f[key] for f in features]
    print(f"\n{key}:")
    for val, count in Counter(vals).most_common():
        print(f"  {val}: {count} ({count/len(vals)*100:.1f}%)")

# ── Train/test split (80/20) ───────────────────────────────────────────────────
np.random.seed(42)
n = len(features)
indices = np.random.permutation(n)
split = int(0.8 * n)

train_features = [features[i] for i in indices[:split]]
train_labels   = [labels[i]   for i in indices[:split]]
test_features  = [features[i] for i in indices[split:]]
test_labels    = [labels[i]   for i in indices[split:]]

print(f"\nTrain: {len(train_features)}, Test: {len(test_features)}")

# ── Build and evaluate tree ────────────────────────────────────────────────────
attribute_names = list(features[0].keys())
print("\nBuilding tree...")
tree = build_tree(train_features, train_labels, attribute_names, max_depth=6, min_samples=5)

print("\n--- Decision Tree ---")
print_tree(tree)

print("\n--- Test Set Evaluation ---")
predictions, paths = predict_batch(tree, test_features)
evaluate(test_labels, predictions)

print("\n--- Example Predictions ---")
for i in range(min(5, len(test_features))):
    print(f"\nInput: {test_features[i]}")
    print(f"True: {test_labels[i]}, Predicted: {predictions[i]}")
    print(f"Path: {paths[i]}")

# ── Build production tree on full dataset and serialise ────────────────────────
print("\nBuilding production tree on full dataset...")
full_tree = build_tree(features, labels, attribute_names, max_depth=6, min_samples=5)

tree_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "predictions", "trained_tree.pkl"
)
with open(tree_path, "wb") as f:
    pickle.dump(full_tree, f)
print(f"Tree serialised to {tree_path}")
