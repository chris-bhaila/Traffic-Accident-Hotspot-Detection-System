import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

import pandas as pd
import numpy as np
import pickle
from accidents.models import AccidentRecord
from predictions.decision_tree import prepare_features, build_tree

# Load all data (train on full dataset for production)
qs = AccidentRecord.objects.filter(source="KTM_SYNTHETIC").values(
    "day_of_week", "weather_condition", "road_type",
    "light_condition", "speed_limit", "junction_type", "severity", "time"
)
df = pd.DataFrame(list(qs))

features, labels = prepare_features(df)
attribute_names = list(features[0].keys())

print(f"Training on {len(features)} records...")
tree = build_tree(features, labels, attribute_names, max_depth=6, min_samples=5)

# Save tree to file
tree_path = "predictions/trained_tree.pkl"
with open(tree_path, "wb") as f:
    pickle.dump(tree, f)

print(f"Tree saved to {tree_path}")