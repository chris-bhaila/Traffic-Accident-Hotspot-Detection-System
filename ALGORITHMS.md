# Algorithm Design

This document covers the mathematical and computational reasoning behind the two core algorithms in this project: DBSCAN for hotspot clustering and the ID3 decision tree for accident severity prediction. Both are implemented from scratch in `predictions/dbscan.py` and `predictions/decision_tree.py` without scikit-learn.

---

## 1. DBSCAN — Density-Based Spatial Clustering of Applications with Noise

### Why DBSCAN?

DBSCAN was chosen over alternatives (k-means, hierarchical clustering) for three reasons specific to this problem:

**No fixed cluster count.** Road accident hotspots are not evenly distributed — some areas have many, some have none. K-means requires specifying k in advance, which makes no sense when you don't know how many hotspots exist. DBSCAN discovers the number of clusters from the data.

**Handles noise natively.** Not every accident belongs to a hotspot. Isolated accidents on rural roads are not hotspots — they're noise. DBSCAN labels these as noise points (label `-1`) rather than forcing them into a cluster.

**Finds clusters of arbitrary shape.** Road accident hotspots follow road geometry — they're not spherical blobs. An accident-prone ring road or motorway junction produces an elongated cluster that k-means would split incorrectly. DBSCAN finds these naturally.

---

### 1.1 Haversine Distance Formula

All distance calculations use the Haversine formula rather than Euclidean distance. This is essential for geographic coordinates.

**Why not Euclidean?** Latitude and longitude are angular measurements on a sphere, not coordinates on a flat plane. At Leeds's latitude (~53.8°N), one degree of longitude is approximately 66km, while one degree of latitude is approximately 111km. Euclidean distance treats these as equal, producing distorted ellipses instead of circles. At small scales (under ~50km) the error is small but non-trivial for a system where 400m precision matters.

**The formula:**

Given two points (φ₁, λ₁) and (φ₂, λ₂) in radians:

```
a = sin²((φ₂ - φ₁) / 2) + cos(φ₁) · cos(φ₂) · sin²((λ₂ - λ₁) / 2)
c = 2 · atan2(√a, √(1 - a))
d = R · c
```

where R = 6,371,000 metres (mean Earth radius).

This gives the great-circle distance — the shortest path between two points on a sphere. For distances under 500km, the error from treating Earth as a perfect sphere is under 0.3%.

**Implementation note:** Coordinates are stored as degrees in the database. The Haversine function converts to radians internally with `math.radians()`.

---

### 1.2 Epsilon-Neighbourhood Search

For a point p, its ε-neighbourhood N(p) is the set of all points within Haversine distance ε:

```
N(p) = { q ∈ D | haversine(p, q) ≤ ε }
```

The implementation iterates over all points for each query point — O(n) per query, O(n²) overall. For 1498 points this is ~1.1 million Haversine computations, which runs in a few seconds. For datasets above ~50,000 points a spatial index (ball tree or k-d tree) would be needed.

---

### 1.3 Core Point Identification

A point p is a **core point** if it has at least `min_samples` points in its ε-neighbourhood (including itself):

```
|N(p)| ≥ min_samples
```

`min_samples = 5` in this implementation. This means a location needs at least 5 accidents within 400m to be considered a genuine hotspot rather than a random cluster.

**Non-core points** that fall within a core point's neighbourhood are **border points** — they belong to the cluster but cannot expand it.

**Noise points** are neither core points nor reachable from any core point. They receive label `-1`.

---

### 1.4 Cluster Expansion via BFS

Once a core point is identified, the cluster is grown by Breadth-First Search (BFS) rather than Depth-First Search (DFS).

**Why BFS and not DFS?** Both produce the same final clusters — the choice is a practical one. BFS uses a queue (FIFO) which processes points layer by layer outward from the seed. DFS uses a stack (LIFO) and can recurse deeply before backtracking. For this application the difference is negligible, but BFS was chosen because it's easier to reason about and avoids Python's recursion limit on large datasets.

**The expansion algorithm:**

```
function expand_cluster(point p, cluster_id):
    queue = [p]
    assign p → cluster_id
    while queue is not empty:
        current = queue.pop_left()
        neighbours = find_neighbours(current, ε)
        if |neighbours| ≥ min_samples:          # current is a core point
            for each neighbour q in neighbours:
                if q is unvisited:
                    mark q as visited
                    queue.append(q)
                if q has no cluster:
                    assign q → cluster_id
```

This ensures every density-reachable point from the seed core point ends up in the same cluster.

---

### 1.5 Noise Point Labelling

After all clusters are formed, any point that was never assigned a cluster ID remains labelled `-1`. These are accidents that don't belong to any hotspot — isolated incidents on roads without sufficient surrounding accident density. In the Leeds dataset, 581 of 1498 points (38.8%) are noise.

---

### 1.6 Epsilon Selection via K-Distance Elbow Method

The original implementation used a hardcoded ε = 150m (later revised to 400m). The current implementation selects epsilon automatically using the k-distance graph elbow method.

**The method:**

1. For each point p, compute the distance to its k-th nearest neighbour, where k = `min_samples - 1` = 4.
2. Collect all n such distances into a sorted array (ascending order).
3. Plot this array — it shows a characteristic "elbow" shape: flat for well-separated points, then a sharp rise where the density drops off. The elbow is the optimal epsilon — below it, most points are noise; above it, clusters merge into one blob.

**Finding the elbow algorithmically:**

The elbow is the point of maximum curvature, found by the perpendicular distance method:
1. Normalise both axes (index and distance) to [0, 1].
2. Draw a straight line from the first point (0, 0) to the last point (1, 1).
3. For each point on the curve, compute its perpendicular distance from this line.
4. The index of maximum perpendicular distance is the elbow.

**Sanity clamp:** If the computed epsilon is < 50m (dataset too dense, likely a data quality issue) or > 400m (dataset too spread out, clusters would be meaninglessly large), the value is clamped and a warning is logged. The Leeds dataset produces an elbow at ~914m, which is clamped to 400m — indicating the dataset is too geographically dispersed for the elbow method to find a natural fine-scale boundary.

---

### 1.7 Post-Clustering: Cluster Summarisation

After DBSCAN assigns cluster labels, each cluster is summarised into a `HotspotCluster` record:

| Field | Computation |
|---|---|
| `centroid_latitude/longitude` | Mean lat/lon of all member points |
| `radius` | Max Haversine distance from centroid to any member |
| `accident_count` | Number of member points |
| `average_severity` | Mean of (Fatal→3, Serious→2, Slight→1) |
| `peak_time` | Mode of `time` field across all members |
| `peak_day` | Mode of `day_of_week` field |
| `dominant_weather` | Mode of `weather_condition` field |
| `risk_level` | Composite rule (see below) |

**Risk level assignment:**
```
CRITICAL : count ≥ 20  AND  avg_severity ≥ 2.0
HIGH     : count ≥ 15  OR  (count ≥ 10 AND avg_severity ≥ 1.8)
MEDIUM   : count ≥ 8   OR  (count ≥ 5  AND avg_severity ≥ 1.5)
LOW      : everything else
```

These thresholds are heuristic — chosen by inspection of the cluster distribution rather than derived statistically.

---

## 2. ID3 Decision Tree Classifier

### Why a Decision Tree?

A decision tree was chosen for the severity prediction task because:

**Interpretability.** The prediction panel shows the decision path step-by-step ("speed = urban → weather = rain → light = dark_unlit → Serious"). This transparency is important for a safety-critical application and for academic demonstration.

**No feature scaling required.** DBSCAN needs Haversine distances; the decision tree works directly on categorical bins without normalisation.

**Handles categorical features naturally.** All input features (time-of-day bin, weather category, road type, etc.) are categorical after binning. Decision trees split on categories directly; algorithms like logistic regression would require one-hot encoding.

**Why ID3 and not C4.5 or CART?**

ID3 is the simplest entropy-based decision tree algorithm. It was chosen for academic clarity. Its limitations compared to successors:

| Feature | ID3 | C4.5 | CART |
|---|---|---|---|
| Split criterion | Information gain | Gain ratio | Gini impurity |
| Handles continuous features | No (bins required) | Yes | Yes |
| Handles missing values | No | Yes | Yes |
| Pruning | No | Yes | Yes |
| Multi-way splits | Yes | Yes | No (binary only) |

C4.5 improves on ID3 by normalising information gain by the split's intrinsic information (gain ratio), preventing bias toward attributes with many distinct values. CART uses Gini impurity instead of entropy. Neither was implemented because the goal was academic demonstration of the core concept, not production-grade accuracy.

---

### 2.1 Entropy

Entropy H measures the uncertainty or impurity in a set of labels. For a set S with classes c₁, c₂, ..., cₖ:

```
H(S) = -∑ p(cᵢ) · log₂(p(cᵢ))
```

where p(cᵢ) is the proportion of samples with class cᵢ.

**Interpretation:**
- H = 0: all samples have the same class (perfectly pure — predictable)
- H = 1: equal split between two classes (maximum uncertainty for binary)
- H = log₂(k): equal split across k classes (maximum uncertainty)

For this project with three classes (Slight, Serious, Fatal), maximum entropy is log₂(3) ≈ 1.585 bits.

**Example:** If a set has 60% Slight, 35% Serious, 5% Fatal:
```
H = -(0.60 · log₂(0.60)) - (0.35 · log₂(0.35)) - (0.05 · log₂(0.05))
  = 0.442 + 0.530 + 0.216
  = 1.188 bits
```

---

### 2.2 Information Gain

Information Gain IG(S, A) measures how much a feature A reduces entropy when used to split set S:

```
IG(S, A) = H(S) - ∑ (|Sᵥ| / |S|) · H(Sᵥ)
```

where the sum is over each distinct value v of attribute A, and Sᵥ is the subset of S where A = v.

**Interpretation:** IG is the entropy of the parent set minus the weighted average entropy of the child sets after splitting. Higher IG means the split produces purer children — the attribute is more useful for classification.

**Example:** Splitting on `speed_limit` (urban/suburban/fast/low):
- If urban accidents are mostly Slight, suburban are mixed, fast are mostly Serious/Fatal, the split produces three relatively pure groups → high IG.
- If all speed limit groups have the same severity distribution, splitting on speed changes nothing → IG = 0.

The algorithm selects the attribute with the highest IG at each node.

---

### 2.3 Feature Binning

Raw numeric/string fields are binned into categorical values before tree training and at prediction time. The same `bin_*` functions are used in both `train_tree.py` and `predictions/views.py` — consistent binning is critical; a mismatch would silently corrupt predictions.

| Raw field | Bins |
|---|---|
| `time` (hour 0–23) | `morning_rush` (7–9), `evening_rush` (16–19), `midday` (10–15), `night` (otherwise) |
| `speed_limit` (mph) | `low` (≤20), `urban` (≤30), `suburban` (≤50), `fast` (>50) |
| `weather_condition` | `fine`, `rain`, `snow`, `fog`, `other` |
| `light_condition` | `daylight`, `dark_lit`, `dark_unlit` |
| `road_type` | `single`, `dual`, `roundabout`, `other` |
| `day_of_week` | Monday–Sunday (unchanged, already categorical) |

Binning trades information loss for generalisability — the tree learns "fast roads are more dangerous" rather than overfitting to specific speed values like 60mph vs 70mph.

---

### 2.4 Recursive Tree Building

The tree is built top-down by the ID3 algorithm:

```
function build_tree(samples, attributes, depth):
    if all samples have same label:
        return LeafNode(label)
    if no attributes remain OR depth == max_depth OR |samples| < min_samples:
        return LeafNode(majority_class(samples))
    
    best_attr = argmax_attr IG(samples, attr) for attr in attributes
    node = InternalNode(attribute=best_attr)
    
    for each value v of best_attr:
        subset = samples where best_attr == v
        if subset is empty:
            node.children[v] = LeafNode(majority_class(samples))
        else:
            node.children[v] = build_tree(subset, attributes - {best_attr}, depth + 1)
    
    return node
```

**Stopping conditions:**
1. **Pure node** — all samples in the subset have the same label. No further splitting needed.
2. **No attributes remain** — all features have been used on this path. Return majority class.
3. **Max depth reached** (`max_depth = 6`) — prevents overfitting on deep paths.
4. **Min samples** (`min_samples = 5`) — prevents splits on tiny subsets that don't generalise.

**Attribute removal:** Once an attribute is used at a node, it is removed from the available set for all descendants of that node. This is an ID3 property — each attribute is used at most once per root-to-leaf path. C4.5 and CART allow reuse.

---

### 2.5 Prediction (Tree Traversal)

To predict the severity for a new sample:

```
function predict(node, sample):
    if node is leaf:
        return node.label
    
    value = sample[node.attribute]
    
    if value not in node.children:
        return node.majority_class    # unseen value — fallback
    
    return predict(node.children[value], sample)
```

The traversal starts at the root and follows the branch matching the sample's value for each node's attribute until a leaf is reached. The leaf's label is the predicted severity.

**Unseen value handling:** If a sample has a value for an attribute that wasn't seen during training (e.g. a new weather category), the tree returns the majority class at that node rather than crashing. This is a basic form of graceful degradation.

**Decision path construction:** As the tree is traversed, each split is recorded as a string: `"speed = urban → weather = rain → Serious"`. This is returned in the API response and displayed in the prediction panel.

---

### 2.6 Serialisation

The trained tree is serialised to `predictions/trained_tree.pkl` using Python's `pickle` module. It is loaded once at module import in `predictions/views.py` and reused for all requests — no retraining at runtime.

**Security note:** Pickle files can execute arbitrary code on load. The file is committed to version control for convenience in this student project, but in production it should be regenerated as part of the deployment pipeline rather than stored in git.

---

## 3. Algorithm Interaction

The two algorithms serve different purposes and operate on different data:

| | DBSCAN | ID3 Decision Tree |
|---|---|---|
| **Purpose** | Discover where accidents cluster geographically | Predict severity given contextual conditions |
| **Input** | (lat, lon) coordinates of accident records | Categorical feature vector (time, weather, road, etc.) |
| **Output** | Cluster assignments + noise labels | Severity prediction (Slight/Serious/Fatal) |
| **Training data** | UK STATS19 (Leeds, 1498 records) — geographic density | UK STATS19 (Leeds, 1498 records) — feature labels |
| **Runtime use** | Offline only — clusters pre-computed | Online — tree walked per API request |
| **Location awareness** | Yes — purely geographic | No — location not a feature |

The two algorithms are intentionally decoupled. DBSCAN identifies where hotspots are; the decision tree predicts how dangerous a given set of conditions is, independent of location. A future improvement would be to connect them — incorporating proximity to a known hotspot as a feature in the decision tree.