import numpy as np
from collections import Counter


def bin_time(hour):
    """Bin hour into time periods."""
    if 7 <= hour <= 9:
        return "morning_rush"
    elif 10 <= hour <= 15:
        return "midday"
    elif 16 <= hour <= 19:
        return "evening_rush"
    else:
        return "night"


def bin_speed(speed):
    """Bin speed limit into categories."""
    if speed <= 20:
        return "low"
    elif speed <= 30:
        return "urban"
    elif speed <= 50:
        return "suburban"
    else:
        return "fast"


def bin_weather(weather):
    """Simplify weather into fewer categories."""
    if "Rain" in weather:
        return "rain"
    elif "Snow" in weather:
        return "snow"
    elif "Fog" in weather:
        return "fog"
    elif "Fine" in weather:
        return "fine"
    else:
        return "other"


def bin_light(light):
    """Simplify light conditions."""
    if light == "Daylight":
        return "daylight"
    elif "lit" in light.lower() or "lights lit" in light.lower():
        return "dark_lit"
    else:
        return "dark_unlit"


def bin_road(road):
    """Simplify road type."""
    if road == "Single carriageway":
        return "single"
    elif road == "Dual carriageway":
        return "dual"
    elif road == "Roundabout":
        return "roundabout"
    else:
        return "other"


def prepare_features(df):
    """
    Convert raw accident data into binned categorical features.
    Returns feature matrix (list of dicts) and labels.
    """
    features = []
    labels = []

    for _, row in df.iterrows():
        hour = row["time"].hour if row["time"] else 12

        feature = {
            "time_period": bin_time(hour),
            "day_of_week": row["day_of_week"],
            "weather": bin_weather(row["weather_condition"]),
            "road_type": bin_road(row["road_type"]),
            "light": bin_light(row["light_condition"]),
            "speed": bin_speed(row["speed_limit"] if row["speed_limit"] else 30),
        }
        features.append(feature)
        labels.append(row["severity"])

    return features, labels


# ── Decision Tree Implementation ──


def entropy(labels):
    """Calculate entropy of a label list."""
    n = len(labels)
    if n == 0:
        return 0

    counts = Counter(labels)
    ent = 0
    for count in counts.values():
        p = count / n
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def information_gain(features, labels, attribute):
    """Calculate information gain for splitting on a given attribute."""
    total_entropy = entropy(labels)
    n = len(labels)

    # Get unique values for this attribute
    values = set(f[attribute] for f in features)

    weighted_entropy = 0
    for value in values:
        # Subset where attribute == value
        subset_labels = [
            labels[i] for i in range(n) if features[i][attribute] == value
        ]
        weight = len(subset_labels) / n
        weighted_entropy += weight * entropy(subset_labels)

    return total_entropy - weighted_entropy


class DecisionTreeNode:
    """A node in the decision tree."""

    def __init__(self):
        self.attribute = None  # Feature to split on
        self.children = {}  # {value: child_node}
        self.label = None  # Predicted class (leaf nodes only)
        self.is_leaf = False
        self.samples = 0  # Number of training samples at this node
        self.distribution = {}  # Class distribution at this node


def build_tree(features, labels, attributes, depth=0, max_depth=6, min_samples=5):
    """
    Recursively build the decision tree using ID3 algorithm.

    Args:
        features: list of dicts (each dict is one sample's features)
        labels: list of class labels
        attributes: list of attribute names still available to split on
        depth: current depth
        max_depth: maximum tree depth
        min_samples: minimum samples to attempt a split
    """
    node = DecisionTreeNode()
    node.samples = len(labels)
    node.distribution = dict(Counter(labels))

    # Base cases
    # 1. All labels are the same
    if len(set(labels)) == 1:
        node.is_leaf = True
        node.label = labels[0]
        return node

    # 2. No attributes left to split on
    if not attributes:
        node.is_leaf = True
        node.label = Counter(labels).most_common(1)[0][0]
        return node

    # 3. Max depth reached
    if depth >= max_depth:
        node.is_leaf = True
        node.label = Counter(labels).most_common(1)[0][0]
        return node

    # 4. Too few samples
    if len(labels) < min_samples:
        node.is_leaf = True
        node.label = Counter(labels).most_common(1)[0][0]
        return node

    # Find best attribute to split on
    best_attribute = None
    best_gain = -1

    for attr in attributes:
        gain = information_gain(features, labels, attr)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attr

    # If no information gain, make leaf
    if best_gain <= 0:
        node.is_leaf = True
        node.label = Counter(labels).most_common(1)[0][0]
        return node

    node.attribute = best_attribute
    remaining_attributes = [a for a in attributes if a != best_attribute]

    # Split on best attribute
    values = set(f[best_attribute] for f in features)
    for value in values:
        # Get subset for this value
        indices = [i for i in range(len(features)) if features[i][best_attribute] == value]
        subset_features = [features[i] for i in indices]
        subset_labels = [labels[i] for i in indices]

        if subset_labels:
            child = build_tree(
                subset_features,
                subset_labels,
                remaining_attributes,
                depth + 1,
                max_depth,
                min_samples,
            )
            node.children[value] = child
        else:
            # Empty subset — make leaf with parent's majority class
            child = DecisionTreeNode()
            child.is_leaf = True
            child.label = Counter(labels).most_common(1)[0][0]
            child.samples = 0
            child.distribution = {}
            node.children[value] = child

    return node


def predict(node, sample):
    """
    Predict class for a single sample.
    Returns (predicted_label, decision_path)
    """
    path = []

    while not node.is_leaf:
        attr = node.attribute
        value = sample.get(attr, None)
        path.append(f"{attr} = {value}")

        if value in node.children:
            node = node.children[value]
        else:
            # Unseen value — return majority class at this node
            path.append("(unseen value — using majority)")
            return node.distribution and Counter(node.distribution).most_common(1)[0][0], " → ".join(path)

    path.append(f"PREDICT: {node.label}")
    return node.label, " → ".join(path)


def predict_batch(node, features):
    """Predict for multiple samples. Returns (predictions, paths)."""
    predictions = []
    paths = []
    for f in features:
        pred, path = predict(node, f)
        predictions.append(pred)
        paths.append(path)
    return predictions, paths


def print_tree(node, indent=""):
    """Print the tree structure."""
    if node.is_leaf:
        print(f"{indent}→ {node.label} (samples: {node.samples}, dist: {node.distribution})")
        return

    print(f"{indent}[{node.attribute}] (samples: {node.samples}, dist: {node.distribution})")
    for value, child in sorted(node.children.items(), key=lambda x: str(x[0])):
        print(f"{indent}  {node.attribute} = {value}:")
        print_tree(child, indent + "    ")


def evaluate(true_labels, predicted_labels):
    """Calculate accuracy, and per-class precision, recall, F1."""
    classes = sorted(set(true_labels + predicted_labels))
    n = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    accuracy = correct / n if n > 0 else 0

    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{n})")
    print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 58)

    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == cls and p != cls)
        support = sum(1 for t in true_labels if t == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{cls:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':>12}", end="")
    for cls in classes:
        print(f"{cls:>12}", end="")
    print()
    for true_cls in classes:
        print(f"{true_cls:>12}", end="")
        for pred_cls in classes:
            count = sum(
                1 for t, p in zip(true_labels, predicted_labels)
                if t == true_cls and p == pred_cls
            )
            print(f"{count:>12}", end="")
        print()

    return accuracy