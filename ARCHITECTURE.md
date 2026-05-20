# Traffic Hotspot Prediction System — Architecture

## Project Overview

A Django web application that visualises traffic accident hotspots across the Kathmandu Valley on an interactive map and predicts accident risk for any user-clicked location. Built as a final-year project.

**Stack:** Django 6, PostgreSQL + PostGIS, Leaflet.js, Tailwind CSS (CDN)  
**Location:** Kathmandu Valley, Nepal  
**Key constraint:** All ML algorithms (DBSCAN, decision tree) are implemented from scratch — no scikit-learn.

---

## Directory Structure

```
traffic-hotspot/
├── config/               # Django project config
│   ├── settings.py
│   ├── urls.py           # Root URL routing
│   └── wsgi.py
│
├── accidents/            # Core data models
│   ├── models.py         # AccidentRecord, HotspotCluster, DataUpload
│   └── migrations/
│
├── dashboard/            # Map and analytics views
│   ├── views.py          # map_view — serves the main map page
│   ├── urls.py
│   └── templates/dashboard/map.html   # The entire frontend
│
├── predictions/          # ML algorithms + prediction API
│   ├── dbscan.py         # DBSCAN from scratch (Haversine distance)
│   ├── decision_tree.py  # ID3 decision tree from scratch
│   ├── views.py          # predict_risk API endpoint
│   ├── urls.py           # /api/predict/
│   └── trained_tree.pkl  # Serialised trained tree (pickle)
│
└── scripts/              # Offline data pipeline (run manually)
    ├── generate_ktm_data.py   # Generates 2000 synthetic accident records
    ├── scrape_accidents.py    # Scrapes Kathmandu Post for real accidents
    ├── save_clusters.py       # Runs DBSCAN and saves HotspotCluster rows
    ├── save_tree.py           # Serialises the trained tree to .pkl
    └── train_tree.py          # Trains decision tree on UK STATS19 data
```

---

## Database Models (`accidents/models.py`)

### `AccidentRecord`
Stores individual accident events. Has a `source` field discriminating between three datasets:
- `KTM_SYNTHETIC` — 2000 procedurally generated records based on real hotspot geography
- `KTM_SCRAPED` — records extracted from Kathmandu Post news articles
- `UK_STATS19` — real UK government road accident statistics (used only for training the decision tree)

Key fields: `latitude`, `longitude`, `location` (PostGIS PointField), `date`, `time`, `day_of_week`, `weather_condition`, `road_type`, `light_condition`, `speed_limit`, `severity` (Slight/Serious/Fatal), `number_of_casualties`, `number_of_deaths`, `vehicle_type`, `accident_type`, `description`, `source_url`, `location_name`.

### `HotspotCluster`
Stores pre-computed DBSCAN cluster summaries. Written by `save_clusters.py`, read at runtime. Never computed on-demand.

Key fields: `centroid_latitude`, `centroid_longitude`, `centroid_location` (PostGIS), `accident_count`, `radius` (metres — max Haversine distance from centroid to any cluster member), `average_severity` (numeric: Fatal=3, Serious=2, Slight=1), `peak_time`, `peak_day`, `dominant_weather`, `risk_level` (LOW/MEDIUM/HIGH/CRITICAL), `district`.

### `DataUpload`
Tracks file upload history. Not currently used by the main map flow.

---

## URL Routing

```
/               → dashboard.views.map_view       (main map page)
/analytics/     → dashboard.views.analytics_view
/api/predict/   → predictions.views.predict_risk (JSON API)
/admin/         → Django admin
```

---

## Offline Data Pipeline

These scripts are run manually once (or whenever the dataset needs refreshing). They write to the database. The web application only reads.

### 1. Synthetic Data Generation (`scripts/generate_ktm_data.py`)

Generates 2000 `AccidentRecord` rows with `source="KTM_SYNTHETIC"`.

Defines ~55 named hotspots across Kathmandu, Lalitpur, and Bhaktapur districts. Each hotspot has `(lat, lon, radius_metres, weight)`. Weight reflects real relative accident frequency from traffic police reports (Kalanki Chowk = 50, Chapagaun = 5).

For each record:
1. Picks a hotspot by weighted random selection.
2. Places the point randomly within that hotspot's radius using lat/lon offset conversion.
3. Picks a date (2023–2025), samples an hour from `HOUR_WEIGHTS` (peak at 17:00 = weight 10), derives day of week from the actual date.
4. Samples weather from `WEATHER_BY_MONTH` — monsoon months (Jun–Sep) skew heavily to rain.
5. Samples severity: Fatal 4%, Serious 35%, Slight 61%.
6. Fills road type, light condition (Daylight if 6–17, dark variant otherwise), speed limit, vehicle type, accident type.
7. Generates a human-readable description from templates.

### 2. News Scraper (`scripts/scrape_accidents.py`)

Scrapes Kathmandu Post category pages for accident articles. Resolves location by scanning article text against a hardcoded dictionary of ~100 Kathmandu Valley place names → coordinates (longest-match-first to avoid "baneshwor" matching before "new baneshwor"). Extracts severity, time, and weather from keyword lookup. Saves as `source="KTM_SCRAPED"`.

> **Note:** Scraped records are stored in the database but the current `map_view` filters to `KTM_SYNTHETIC` only. Scraped data is unused on the live map.

### 3. DBSCAN Clustering (`scripts/save_clusters.py`)

Loads all `KTM_SYNTHETIC` accident points as a numpy `(n, 2)` array of `[lat, lon]`.

Calls `predictions/dbscan.py` with `epsilon=150` metres, `min_samples=5`.

**DBSCAN algorithm (custom, `predictions/dbscan.py`):**
- Distance metric: Haversine formula (great-circle distance on a sphere). Essential for geographic coordinates — Euclidean distance is wrong here.
- For each unvisited point: find all neighbours within 150m.
- If ≥5 neighbours → core point → assign new cluster ID, BFS-expand through all reachable core points.
- Points that never reach 5 neighbours remain label `-1` (noise).

After clustering, for each cluster:
- Centroid = mean lat/lon.
- Radius = max Haversine distance from centroid to any member.
- Average severity = mean of (Fatal→3, Serious→2, Slight→1).
- Peak day, peak time, dominant weather = mode of each attribute.
- Risk level assigned by composite rule:
  - **CRITICAL**: count ≥ 20 AND avg_severity ≥ 2.0
  - **HIGH**: count ≥ 15, OR (count ≥ 10 AND avg_severity ≥ 1.8)
  - **MEDIUM**: count ≥ 8, OR (count ≥ 5 AND avg_severity ≥ 1.5)
  - **LOW**: everything else

Results saved as `HotspotCluster` rows. Existing clusters are deleted before each run.

### 4. Decision Tree Training (`scripts/train_tree.py`)

Loads `UK_STATS19` records from the database. Converts raw fields to binned categories via `prepare_features()`:

| Raw field | Bins |
|---|---|
| time (hour) | morning_rush (7–9), midday (10–15), evening_rush (16–19), night |
| speed_limit | low (≤20), urban (≤30), suburban (≤50), fast (>50) |
| weather_condition | fine / rain / snow / fog / other |
| light_condition | daylight / dark_lit / dark_unlit |
| road_type | single / dual / roundabout / other |
| day_of_week | Monday … Sunday (unchanged) |

Target label: `severity` (Slight / Serious / Fatal).

**ID3 algorithm (custom, `predictions/decision_tree.py`):**
- At each node, compute information gain (entropy reduction) for every remaining attribute.
- Split on the highest-gain attribute.
- Recurse into each branch, removing the used attribute from the available set.
- Stopping conditions: all labels identical, no attributes remain, `max_depth=6` reached, or fewer than `min_samples=5` samples.
- Handles unseen values at prediction time: returns the majority class at that node.

Serialised to `predictions/trained_tree.pkl` with `pickle`. Loaded once at module import in `predictions/views.py`.

---

## Web Application (Runtime)

### Map Page (`GET /`)

`dashboard/views.py → map_view`:
1. Reads all `HotspotCluster` rows (pre-computed, fast).
2. Reads all `AccidentRecord` where `source="KTM_SYNTHETIC"`.
3. Serialises both to JSON via `json.dumps(..., default=str)`.
4. Renders `dashboard/map.html` with `clusters_json`, `accidents_json`, `n_clusters`, `n_total` injected into the template.

No ML, no clustering, no calculation at request time.

### Frontend (`dashboard/templates/dashboard/map.html`)

Single-file frontend. Both JSON datasets are embedded directly in the HTML as JavaScript variables at render time — no AJAX for the initial data load.

**Leaflet layer architecture:**
- Four `L.layerGroup` instances: `riskLayers.CRITICAL`, `.HIGH`, `.MEDIUM`, `.LOW` — all added to the map by default.
- For each cluster: an `L.circle` (halo, geographic radius in metres, min 300m) and an `L.circleMarker` (centroid dot, screen-space pixels, radius `min(10 + count/3, 24)`) go into the matching risk layer.
- `accidentLayer` holds individual accident `circleMarker`s. Added/removed from the map on `zoomend` — only visible at zoom ≥ 15.
- Risk level toggles call `map.addLayer`/`map.removeLayer` on the per-risk group. Both halo and centroid are in the same group, so filtering removes everything for that level.
- A "Hide All / Show All" button and per-level count badges (e.g. "High (45)") are computed client-side from the clusters array after render.

**Prediction panel:**
- User clicks map → `map.on("click")` fires → places a `L.marker`, shows the prediction panel with coordinate display.
- Selecting inputs and hitting "Predict Risk" fires a `fetch` to `/api/predict/?lat=...&day=...&weather=...` etc.
- Response `{ prediction, decision_path }` populates the risk badge, colours the result card, and renders the decision path as a step-by-step list.
- An ✕ button in the panel header removes the pin marker and resets the panel to the click-prompt state.

### Prediction API (`GET /api/predict/`)

`predictions/views.py → predict_risk`:
1. Parses query parameters: `lat`, `lon`, `time` (hour int), `day`, `weather`, `road`, `speed`, `light`.
2. Bins each value using the same functions used during training (`bin_time`, `bin_speed`, `bin_weather`, `bin_light`, `bin_road`).
3. Calls `predict(TRAINED_TREE, sample)` — walks the tree, at each node reads `node.attribute`, looks up the sample's value for that attribute, follows the matching child. Builds a `decision_path` string as it goes.
4. Returns JSON: `{ status, prediction, decision_path, input, location }`.

`lat`/`lon` are passed through but **not used by the tree** — the model predicts from contextual features only, not location. They are returned in the response for potential future use.

---

## Data Flow Diagram

```
Real hotspot knowledge (police data, news)
          ↓ (encoded in HOTSPOTS dict)
generate_ktm_data.py ──────────────────────────────────→ AccidentRecord (KTM_SYNTHETIC)
scrape_accidents.py  ──────────────────────────────────→ AccidentRecord (KTM_SCRAPED)
                                                                  │
                                                    save_clusters.py
                                                    [DBSCAN ε=150m, minPts=5]
                                                                  │
                                                                  ↓
                                                          HotspotCluster table

UK STATS19 data (in DB)
          ↓
train_tree.py [ID3, max_depth=6]
          ↓
trained_tree.pkl

──────────────────────── runtime ────────────────────────

Browser  GET /
          ↓
map_view → SELECT * FROM HotspotCluster
         → SELECT * FROM AccidentRecord WHERE source='KTM_SYNTHETIC'
         → JSON embed → render map.html
          ↓
Leaflet renders circles + dots client-side (no server calls)

Browser  GET /api/predict/?lat=...&time=17&weather=rain...
          ↓
predict_risk → bin inputs → walk trained_tree.pkl
          ↓
{ prediction: "Serious", decision_path: "speed = urban → ..." }
```

---

## Key Design Decisions

- **Clustering is offline.** Adding new accident records to the DB does not update the map until `save_clusters.py` is re-run. The map always shows the last saved cluster state.
- **Tree trained on UK data, used for Kathmandu.** The decision tree is trained on UK STATS19 (which has large, labelled datasets) because comparable Kathmandu data isn't available at scale. The feature bins are domain-agnostic (time-of-day, weather, road type) so the model transfers reasonably. Location is not a feature.
- **No scikit-learn.** Both DBSCAN and the ID3 decision tree are hand-rolled in Python/numpy for academic demonstration purposes.
- **All frontend data is server-side rendered.** Both the clusters and accident points are embedded as JS variables in the initial HTML response. There is no separate REST endpoint for fetching map data — only for predictions.
- **PostGIS is included but largely unused.** `PointField` is on both `AccidentRecord` and `HotspotCluster`, but all distance calculations go through the custom Haversine function rather than PostGIS spatial queries.
