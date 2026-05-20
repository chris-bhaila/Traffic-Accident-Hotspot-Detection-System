import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

import pandas as pd
import numpy as np
import json
from accidents.models import AccidentRecord
from predictions.decision_tree import (
    prepare_features,
    build_tree,
    predict_batch,
    print_tree,
    evaluate,
)

# Load data
qs = AccidentRecord.objects.filter(source="UK_STATS19").values(
    "day_of_week", "weather_condition", "road_type",
    "light_condition", "speed_limit", "junction_type", "severity", "time"
)
df = pd.DataFrame(list(qs))
print(f"Total records: {len(df)}")

# Prepare features
features, labels = prepare_features(df)
print(f"Features prepared: {len(features)}")
print(f"Feature keys: {list(features[0].keys())}")

# Check bin distributions
print("\n--- Bin distributions ---")
for key in features[0].keys():
    vals = [f[key] for f in features]
    from collections import Counter
    print(f"\n{key}:")
    for val, count in Counter(vals).most_common():
        print(f"  {val}: {count} ({count/len(vals)*100:.1f}%)")

# Train/test split (80/20)
np.random.seed(42)
n = len(features)
indices = np.random.permutation(n)
split = int(0.8 * n)

train_features = [features[i] for i in indices[:split]]
train_labels = [labels[i] for i in indices[:split]]
test_features = [features[i] for i in indices[split:]]
test_labels = [labels[i] for i in indices[split:]]

print(f"\nTrain: {len(train_features)}, Test: {len(test_features)}")

# Build tree
attribute_names = list(features[0].keys())
print("\nBuilding tree...")
tree = build_tree(train_features, train_labels, attribute_names, max_depth=6, min_samples=5)

# Print tree structure
print("\n--- Decision Tree ---")
print_tree(tree)

# Evaluate on test set
print("\n--- Test Set Evaluation ---")
predictions, paths = predict_batch(tree, test_features)
accuracy = evaluate(test_labels, predictions)

# Show a few example predictions with paths
print("\n--- Example Predictions ---")
for i in range(min(5, len(test_features))):
    print(f"\nInput: {test_features[i]}")
    print(f"True: {test_labels[i]}, Predicted: {predictions[i]}")
    print(f"Path: {paths[i]}")