# Traffic Hotspot Prediction System — Architecture

## Project Overview

A Django web application that visualises traffic accident hotspots on an interactive map and predicts accident risk for any user-clicked location. Built as a final-year project.

**Stack:** Django 6, PostgreSQL (plain, no PostGIS), Leaflet.js, Tailwind CSS (CDN)  
**Clustering dataset:** UK STATS19 (Leeds district, 1498 records)  
**Prediction dataset:** UK STATS19 (same dataset, decision tree trained on it)  
**Scraped dataset:** Kathmandu Valley accident articles (OnlineKhabar Nepali, Kathmandu Post English)  
**Key constraint:** All ML algorithms (DBSCAN, decision tree) are implemented from scratch — no scikit-learn.  
**Deployment:** Render (web service + PostgreSQL), local dev on Fedora Linux.

---

## Directory Structure

```
traffic-hotspot/
├── config/                      # Django project config
│   ├── settings.py              # Env-var based config, dj-database-url
│   ├── urls.py                  # Root URL routing
│   └── wsgi.py
│
├── accidents/                   # Core data models
│   ├── models.py                # AccidentRecord, HotspotCluster, DataUpload
│   └── migrations/
│
├── dashboard/                   # Map and analytics views
│   ├── views.py                 # map_view, analytics_view
│   ├── urls.py
│   └── templates/dashboard/map.html   # Entire frontend (single file)
│
├── predictions/                 # ML algorithms + prediction API
│   ├── dbscan.py                # DBSCAN from scratch (Haversine distance)
│   ├── decision_tree.py         # ID3 decision tree from scratch
│   ├── views.py                 # predict_risk API endpoint
│   ├── urls.py                  # /api/predict/
│   └── trained_tree.pkl         # Serialised trained tree (pickle)
│
└── scripts/                     # Offline data pipeline (run manually)
    ├── generate_ktm_data.py     # Generates 2000 synthetic KTM accident records
    ├── scrape_accidents.py      # Scrapes Kathmandu Post (English) for accidents
    ├── scrape_onlinekhabar.py   # Scrapes OnlineKhabar (Nepali) for accidents
    ├── save_clusters.py         # Runs DBSCAN with elbow-method epsilon, saves HotspotCluster rows
    ├── save_tree.py             # Serialises the trained tree to .pkl
    └── train_tree.py            # Trains decision tree on UK_STATS19 records
```

---

## Database Models (`accidents/models.py`)

### `AccidentRecord`
Stores individual accident events. The `source` field discriminates between datasets:

| Value | Description |
|---|---|
| `UK_STATS19` | Real UK government road accident statistics. Used for clustering and decision tree training. |
| `KTM_SYNTHETIC` | 2000 procedurally generated Kathmandu Valley records based on real hotspot geography. Currently archived as `KTM_SYNTHETIC_ARCH` to exclude from live map. |
| `KTM_SYNTHETIC_ARCH` | Archived synthetic data — excluded from map but retained for demo restoration. |
| `KTM_SCRAPED` | Records extracted from Kathmandu Post (English) and OnlineKhabar (Nepali) news articles. |

Key fields: `latitude`, `longitude`, `date`, `time`, `day_of_week`, `weather_condition`, `road_type` (nullable), `light_condition` (nullable), `speed_limit` (nullable), `junction_type` (nullable), `severity` (Slight/Serious/Fatal), `number_of_casualties`, `number_of_deaths`, `vehicle_type`, `accident_type`, `description`, `source_url`, `location_name`, `source`.

> Note: `road_type`, `light_condition`, `speed_limit`, and `junction_type` are nullable to accommodate scraped records where these fields are unavailable in news article text.

### `HotspotCluster`
Stores pre-computed DBSCAN cluster summaries. Written by `save_clusters.py`, read at runtime. Never computed on-demand.

Key fields: `centroid_latitude`, `centroid_longitude`, `accident_count`, `radius` (metres — max Haversine distance from centroid to any member), `average_severity` (numeric: Fatal=3, Serious=2, Slight=1), `peak_time`, `peak_day`, `dominant_weather`, `risk_level` (LOW/MEDIUM/HIGH/CRITICAL), `district`.

### `DataUpload`
Tracks file upload history. Not currently used by the main map flow.

---

## URL Routing

```
/               → dashboard.views.map_view
/analytics/     → dashboard.views.analytics_view
/api/predict/   → predictions.views.predict_risk
/admin/         → Django admin
```

---

## Offline Data Pipeline

These scripts are run manually once (or whenever the dataset needs refreshing). They write to the database; the web application only reads.

### 1. Synthetic Data Generation (`scripts/generate_ktm_data.py`)

Generates 2000 `AccidentRecord` rows with `source="KTM_SYNTHETIC"`. Defines ~55 named hotspots across Kathmandu, Lalitpur, and Bhaktapur districts with weighted random selection based on traffic police reports. Currently the synthetic data is archived (`KTM_SYNTHETIC_ARCH`) on the live system.

To restore for a demo:
```bash
python -c "from accidents.models import AccidentRecord; AccidentRecord.objects.filter(source='KTM_SYNTHETIC_ARCH').update(source='KTM_SYNTHETIC')"
```

To re-archive:
```bash
python -c "from accidents.models import AccidentRecord; AccidentRecord.objects.filter(source='KTM_SYNTHETIC').update(source='KTM_SYNTHETIC_ARCH')"
```

### 2. English Scraper (`scripts/scrape_accidents.py`)

Scrapes Kathmandu Post category pages and article pages for Kathmandu Valley road accident reports. Resolves location against a dictionary of ~100 place names → coordinates (longest-match-first). Extracts severity, time, weather from English keyword dictionaries. Saves as `source="KTM_SCRAPED"`.

### 3. Nepali Scraper (`scripts/scrape_onlinekhabar.py`)

Scrapes OnlineKhabar (Nepal's largest online Nepali-language news portal) for Kathmandu Valley accident articles. Key differences from the English scraper:

- Uses Devanagari keyword dictionaries for severity (`मृत्यु`, `घाइते`), time (`बिहान`, `साँझ`), weather (`वर्षा`, `कुहिरो`), and vehicle type (`मोटरसाइकल`, `माइक्रोबस`)
- Location dictionary maps Devanagari place names → coordinates
- Date extraction via WordPress REST API (`/wp-json/wp/v2/posts/{id}`) — more reliable than HTML meta tags
- Accident article filter requires accident keywords in the title AND rejects policy/infrastructure articles via negative keyword list
- Merges with existing `KTM_SCRAPED` records (deduplicates by `source_url`) rather than clearing and replacing

Usage:
```bash
python scripts/scrape_onlinekhabar.py --dry-run --max-pages 3   # test
python scripts/scrape_onlinekhabar.py --max-pages 15            # full run
```

### 4. DBSCAN Clustering (`scripts/save_clusters.py`)

Loads all `UK_STATS19` accident points as a numpy `(n, 2)` array of `[lat, lon]`.

**Epsilon selection via k-distance elbow method:**
1. For each point, compute the Haversine distance to its k-th nearest neighbour (k = `min_samples - 1` = 4).
2. Sort all k-distances in ascending order.
3. Normalise both axes to [0, 1] and find the point of maximum perpendicular distance from the line connecting the first and last points (maximum curvature = elbow).
4. Use that distance as epsilon.
5. Sanity clamp: if computed epsilon < 50m or > 400m, clamp and log a warning.

Current results on Leeds UK_STATS19 (1498 records): elbow exceeds 400m cap → epsilon clamped to **400m**, producing **43 clusters**, 581 noise points.

**DBSCAN algorithm (custom, `predictions/dbscan.py`):**
- Distance metric: Haversine formula (great-circle distance). Euclidean distance is wrong for geographic coordinates.
- For each unvisited point: find all neighbours within epsilon.
- If ≥ `min_samples` (5) neighbours → core point → assign new cluster ID, BFS-expand.
- Points never reaching `min_samples` neighbours remain label `-1` (noise).

After clustering, for each cluster:
- Centroid = mean lat/lon of all member points.
- Radius = max Haversine distance from centroid to any member.
- Average severity = mean of (Fatal→3, Serious→2, Slight→1).
- Peak day, peak time, dominant weather = mode of each attribute.
- Risk level assigned by composite rule:
  - **CRITICAL**: count ≥ 20 AND avg_severity ≥ 2.0
  - **HIGH**: count ≥ 15, OR (count ≥ 10 AND avg_severity ≥ 1.8)
  - **MEDIUM**: count ≥ 8, OR (count ≥ 5 AND avg_severity ≥ 1.5)
  - **LOW**: everything else

Existing clusters are deleted before each run.

### 5. Decision Tree Training (`scripts/train_tree.py`)

Loads `UK_STATS19` records from the database. Converts raw fields to binned categories:

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
- Recurse, removing used attributes from the available set.
- Stopping conditions: all labels identical, no attributes remain, `max_depth=6`, or fewer than `min_samples=5` samples.
- Handles unseen values at prediction time: returns majority class at that node.

Serialised to `predictions/trained_tree.pkl` with `pickle`. Loaded once at module import in `predictions/views.py`.

---

## Web Application (Runtime)

### Settings & Environment (`config/settings.py`)

Credentials and environment-specific config are read from environment variables via `python-dotenv` (local) and Render's environment tab (production):

```
SECRET_KEY, DEBUG, ALLOWED_HOSTS
DATABASE_URL         # Render internal URL (production) or external URL (local → Render)
DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT   # local dev fallback
```

`dj-database-url` parses `DATABASE_URL` if set; otherwise falls back to individual `DB_*` vars. `load_dotenv()` is only called when not running on Render (detected via `os.environ.get('RENDER')`).

### Map Page (`GET /`)

`dashboard/views.py → map_view`:
1. Reads all `HotspotCluster` rows.
2. Reads all `AccidentRecord` where `source__in=["KTM_SYNTHETIC", "KTM_SCRAPED"]`.
3. Serialises both to JSON.
4. Renders `dashboard/map.html`.

No ML, no clustering, no calculation at request time.

### Frontend (`dashboard/templates/dashboard/map.html`)

Single-file frontend. Data embedded as JSON in the HTML response — no AJAX for initial load.

**Basemap:** CartoDB Positron (`light_all`) and CartoDB Dark Matter (`dark_all`) — clean minimal tiles with no POI clutter, toggled by a dark mode button.

**Leaflet layer architecture:**
- Four `L.layerGroup` instances: `riskLayers.CRITICAL/HIGH/MEDIUM/LOW` — all visible by default.
- Per cluster: an `L.circle` (geographic halo, min 300m radius) and an `L.circleMarker` (centroid dot) in the matching risk layer.
- `accidentLayer` holds individual accident dots, visible at all zoom levels.
- Risk level toggles call `map.addLayer`/`map.removeLayer` on the per-risk group.
- Per-level count badges computed client-side from clusters array.

**Prediction panel:**
- User clicks map → pin placed, prediction panel shown.
- User selects day/time/weather/road/speed/light and hits "Predict Risk".
- `fetch` to `/api/predict/?lat=...&day=...` etc.
- Response `{ prediction, decision_path }` populates risk badge and step-by-step decision path list.
- ✕ button removes pin and resets panel.

### Prediction API (`GET /api/predict/`)

`predictions/views.py → predict_risk`:
1. Parses query params: `lat`, `lon`, `time`, `day`, `weather`, `road`, `speed`, `light`.
2. Bins each value using the same functions as training.
3. Walks `TRAINED_TREE`, building a `decision_path` string at each node.
4. Returns `{ status, prediction, decision_path, input, location }`.

`lat`/`lon` are passed through but not used by the tree — the model predicts from contextual features only, not location.

---

## Deployment

**Platform:** Render (free tier)
- Web service: auto-deploys from GitHub `main` branch on push.
- Build command: `pip install -r requirements.txt && python manage.py migrate`
- Start command: `gunicorn config.wsgi:application`
- PostgreSQL: Render managed database (plain PostgreSQL, no PostGIS).
- Environment variables set in Render dashboard.

**Local development:** Fedora Linux, PostgreSQL installed natively, pgAdmin via Docker (`docker-compose.pgadmin.yml`).

---

## Data Flow

```
UK STATS19 records (in DB, source="UK_STATS19")
          │
          ├─→ save_clusters.py
          │     [k-distance elbow → ε=400m cap, minPts=5]
          │     [DBSCAN from scratch, Haversine]
          │                 ↓
          │         HotspotCluster table (43 clusters)
          │
          └─→ train_tree.py [ID3, max_depth=6]
                            ↓
                    trained_tree.pkl

Kathmandu Post articles → scrape_accidents.py ──→ AccidentRecord (KTM_SCRAPED)
OnlineKhabar articles  → scrape_onlinekhabar.py ─→ AccidentRecord (KTM_SCRAPED)
generate_ktm_data.py   ─────────────────────────→ AccidentRecord (KTM_SYNTHETIC_ARCH)

──────────────────────── runtime ────────────────────────

Browser  GET /
          ↓
map_view → SELECT * FROM HotspotCluster
         → SELECT * FROM AccidentRecord WHERE source IN ('KTM_SYNTHETIC','KTM_SCRAPED')
         → JSON embed → render map.html
          ↓
Leaflet renders clusters + accident dots client-side

Browser  GET /api/predict/?lat=...&time=17&weather=rain...
          ↓
predict_risk → bin inputs → walk trained_tree.pkl
          ↓
{ prediction: "Serious", decision_path: "speed = urban → ..." }
```

---

## Key Design Decisions

- **PostGIS removed.** Originally included for spatial queries but unused — all distance calculations go through the custom Haversine function. Removed to enable deployment on PythonAnywhere/Render which don't support PostGIS.
- **Epsilon selected automatically.** k-distance elbow method replaces hardcoded epsilon. More defensible academically and adapts to dataset density.
- **Clustering uses UK STATS19, not Kathmandu data.** No large labelled Kathmandu accident dataset exists publicly. UK STATS19 (Leeds) provides real geographic clustering. Scraped Kathmandu data (9 records) is too sparse for DBSCAN.
- **Decision tree trained on UK data, used for Kathmandu predictions.** Feature bins are domain-agnostic (time-of-day, weather, road type) so the model transfers reasonably. Location is not a feature.
- **No scikit-learn.** DBSCAN and ID3 decision tree are hand-rolled in Python/numpy for academic demonstration.
- **Synthetic data archived, not deleted.** `KTM_SYNTHETIC_ARCH` source preserves 2000 synthetic records for demo restoration without cluttering the live map.
- **Nepali scraper uses rule-based extraction, not NLP.** Nepali accident reporting is formulaic — Devanagari keyword dictionaries are sufficient and avoid model training/API dependencies.

---

## Known Limitations

- **UK→Kathmandu transfer gap.** The decision tree trained on UK data predicts for Kathmandu contexts. UK and Kathmandu road infrastructure, speed limits, and reporting conventions differ significantly. A locally trained model would be more accurate.
- **Scraped data is sparse.** 9 Kathmandu Valley records from news articles is insufficient for clustering. Clustering currently runs on Leeds UK STATS19 data.
- **DBSCAN epsilon clamped.** The elbow method selects >800m; clamped to 400m. The Leeds dataset is geographically spread, so the natural cluster density doesn't produce a clean elbow at fine scales.
- **Prediction ignores location.** Clicking different map locations with the same contextual inputs gives identical predictions. Incorporating proximity to known hotspots as a feature would improve this.
- **`trained_tree.pkl` in version control.** Acceptable for a student project but a security risk in production — pickle files can execute arbitrary code if tampered with.

---

## Planned Features (Unbuilt)

### Analytics Page (`/analytics/`)
Route exists in `config/urls.py` and `dashboard/urls.py` but the view returns an empty response. Planned content:
- Accident frequency by hour of day (bar chart)
- Severity distribution (Fatal/Serious/Slight) by road type
- Weather condition breakdown
- Month-over-month trend if date range spans multiple years
- Cluster risk level distribution
- Top 10 hotspot locations by accident count

### Nearest-Cluster Proximity as Prediction Feature
The prediction API currently ignores `lat`/`lon` entirely — clicking different map locations with identical dropdown inputs returns the same prediction. The fix is to compute the distance from the clicked point to the nearest `HotspotCluster` centroid and bin it (e.g. `within_hotspot` / `near_hotspot` / `outside_hotspot`), then retrain the decision tree with this as an additional feature. This makes location meaningful in predictions.

### Weather Enrichment via Open-Meteo API
Historical weather data could be backfilled for `UK_STATS19` records missing `weather_condition` by querying the Open-Meteo historical API (`https://archive-api.open-meteo.com/v1/archive`) with each record's date, latitude, and longitude. The API is free and requires no key. This would improve the decision tree's weather feature quality. Not implemented due to API call volume (one call per record = ~1500 calls for the Leeds dataset).

### Extended Nepali Scraping
The Nepali scraper currently only targets OnlineKhabar. eKantipur (same Kantipur Media Group as Kathmandu Post, largest circulation) and Ratopati (high local accident coverage) are planned additions using the same Devanagari keyword architecture. eKantipur blocks automated requests (bot detection); Ratopati has not been tested yet.

### Decision Tree Model Evaluation
No train/test split or cross-validation has been performed. The tree is trained on the full `UK_STATS19` dataset and accuracy against a held-out test set has never been measured. A proper evaluation with a confusion matrix (Slight/Serious/Fatal) would strengthen the project academically.

---

## Unoptimized Components

### DBSCAN Epsilon Selection
The k-distance elbow method selects >800m for the Leeds dataset, which is clamped to 400m. The clamp is a workaround — the dataset is too geographically spread for the elbow method to find a natural fine-scale density boundary. A better approach would be to run the elbow method on a filtered subset (e.g. city centre only) or tune `min_samples` and `k` to get a natural elbow below the cap.

### DBSCAN `min_samples` Is Hardcoded
`min_samples=5` was chosen without justification. The standard heuristic is `min_samples ≥ dimensionality + 1`, which gives 3 for 2D data. A higher value (e.g. 10) would produce fewer, denser clusters; a lower value would produce more, noisier clusters. This parameter was never tuned against the actual dataset.

### Decision Tree `max_depth` Is Hardcoded
`max_depth=6` was set without cross-validation. Deeper trees overfit; shallower trees underfit. The optimal depth should be selected by evaluating accuracy on a held-out validation set across multiple depths.

### Risk Level Thresholds Are Arbitrary
The composite rules for CRITICAL/HIGH/MEDIUM/LOW risk assignment were chosen by inspection:
- CRITICAL: count ≥ 20 AND avg_severity ≥ 2.0
- HIGH: count ≥ 15, OR (count ≥ 10 AND avg_severity ≥ 1.8)
- MEDIUM: count ≥ 8, OR (count ≥ 5 AND avg_severity ≥ 1.5)

These were not derived from any statistical analysis. A principled approach would use percentile thresholds on the actual cluster distribution.

### All Accident Points Embedded in HTML
`map_view` embeds all accident records as a JSON blob in the initial HTML response. At 1498 records this is ~200KB of inline JSON. At larger scale this would cause unacceptable page load times. A GeoJSON API endpoint with bounding-box filtering would be the correct solution.

### `DataUpload` Model Is Unused
The `DataUpload` model was intended to track CSV upload history but was never integrated into any view or pipeline. It exists in the schema but serves no purpose.

### `trained_tree.pkl` in Version Control
Pickle files are not safe to store in version control — they can execute arbitrary code on load if tampered with. The correct approach is to regenerate the tree as part of deployment (run `save_tree.py` after `migrate` in the build command) rather than committing the binary.

---

## Future Work

- Nepali-language scraping extended to eKantipur and Ratopati.
- Incorporate nearest-cluster proximity as a prediction feature.
- Build the analytics page with accident distribution charts.
- Perform proper train/test evaluation of the decision tree.
- Retrain decision tree on Kathmandu data if sufficient labelled records become available.
- Replace inline JSON embedding with a GeoJSON API endpoint for scalability.