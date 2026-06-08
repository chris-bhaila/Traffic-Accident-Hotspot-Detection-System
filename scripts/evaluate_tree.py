import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

import pandas as pd
from accidents.models import AccidentRecord, HotspotCluster, TreeEvaluation
from predictions.dbscan import haversine_distance
from predictions.decision_tree import prepare_features, build_tree, predict_batch, bin_proximity

CLASSES = ["Slight", "Serious", "Fatal"]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading UK_STATS19 records from database...")
qs = AccidentRecord.objects.filter(source="UK_STATS19").values(
    "latitude", "longitude",
    "day_of_week", "weather_condition", "road_type",
    "light_condition", "speed_limit", "junction_type", "severity", "time",
)
df = pd.DataFrame(list(qs))
print(f"  {len(df)} records loaded.")

# Compute proximity_to_hotspot for each record
centroids = list(HotspotCluster.objects.values_list("centroid_latitude", "centroid_longitude"))
print(f"  {len(centroids)} cluster centroids loaded.")

def _nearest_dist(row):
    if not centroids:
        return float("inf")
    return min(haversine_distance(row["latitude"], row["longitude"], c_lat, c_lon)
               for c_lat, c_lon in centroids)

df["proximity_to_hotspot"] = df.apply(
    lambda row: bin_proximity(_nearest_dist(row)), axis=1
)

features, labels = prepare_features(df)
print(f"  {len(features)} samples prepared.")

# ── Stratified 80/20 split ─────────────────────────────────────────────────────
# Group indices by severity class, shuffle each group independently, then take
# 80% train / 20% test per class so rare Fatal records are represented in both.
random.seed(42)

train_features, train_labels = [], []
test_features,  test_labels  = [], []

print("\nStratified 80/20 split (seed=42):")
for cls in CLASSES:
    indices = [i for i, lbl in enumerate(labels) if lbl == cls]
    random.shuffle(indices)
    n_train = int(0.8 * len(indices))
    for i in indices[:n_train]:
        train_features.append(features[i])
        train_labels.append(labels[i])
    for i in indices[n_train:]:
        test_features.append(features[i])
        test_labels.append(labels[i])
    n_test = len(indices) - n_train
    print(f"  {cls:<8}: {len(indices):>4} total → {n_train} train, {n_test} test")

print(f"\n  Total: {len(train_labels)} train, {len(test_labels)} test")

# ── Train fresh tree ───────────────────────────────────────────────────────────
attribute_names = list(features[0].keys())
print("\nBuilding ID3 tree on training set (max_depth=6, min_samples=5)...")
tree = build_tree(train_features, train_labels, attribute_names, max_depth=6, min_samples=5)
print("  Tree built.")

# ── Predict on test set ────────────────────────────────────────────────────────
print("\nRunning predictions on test set...")
predictions, _ = predict_batch(tree, test_features)

# ── Compute confusion matrix ───────────────────────────────────────────────────
confusion = {a: {p: 0 for p in CLASSES} for a in CLASSES}
for true, pred in zip(test_labels, predictions):
    if true in confusion and pred in confusion[true]:
        confusion[true][pred] += 1

# ── Overall accuracy ───────────────────────────────────────────────────────────
total   = len(test_labels)
correct = sum(1 for t, p in zip(test_labels, predictions) if t == p)
accuracy = correct / total if total > 0 else 0.0

# ── Per-class precision, recall, F1 ───────────────────────────────────────────
per_class = {}
f1_scores = []

for cls in CLASSES:
    support = sum(confusion[cls].values())
    tp = confusion[cls][cls]
    fp = sum(confusion[other][cls] for other in CLASSES if other != cls)
    fn = sum(confusion[cls][other] for other in CLASSES if other != cls)

    if support == 0:
        print(f"  WARNING: {cls} has zero test samples — metrics undefined.")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    if (tp + fp) == 0:
        print(f"  WARNING: {cls} has zero predictions — precision undefined (set to 0).")

    per_class[cls] = {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "support":   support,
        "tp": tp, "fp": fp, "fn": fn,
    }
    f1_scores.append(f1)

f1_macro = sum(f1_scores) / len(f1_scores)

# ── Print results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Overall Accuracy : {accuracy:.4f}  ({correct}/{total})")
print(f"Macro F1 Score   : {f1_macro:.4f}")
print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 50)
for cls in CLASSES:
    m = per_class[cls]
    print(f"{cls:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")

print(f"\nConfusion Matrix (rows = Actual, cols = Predicted):")
print(f"{'':>12}", end="")
for cls in CLASSES:
    print(f"{cls:>10}", end="")
print()
for actual in CLASSES:
    print(f"{actual:>12}", end="")
    for pred in CLASSES:
        print(f"{confusion[actual][pred]:>10}", end="")
    print()
print("=" * 60)

# ── Save to database ───────────────────────────────────────────────────────────
print("\nSaving results to database...")
record = TreeEvaluation(
    accuracy=accuracy,
    f1_macro=f1_macro,
    confusion_matrix_json=json.dumps(confusion),
    per_class_metrics_json=json.dumps(per_class),
    train_size=len(train_labels),
    test_size=len(test_labels),
)
record.save()
print(f"  Saved TreeEvaluation (id={record.id}).")
print("Done.")
