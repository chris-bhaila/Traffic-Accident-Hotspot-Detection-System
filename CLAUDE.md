# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

All commands must be run inside the virtual environment:

```bash
source .venv/bin/activate
```

**Run dev server**
```bash
python manage.py runserver
```

**Migrations**
```bash
python manage.py makemigrations
python manage.py migrate
```

**Formatting**
```bash
black .
```

There are no automated tests. Verification is done by running the app and inspecting output.

### Offline ML pipeline (run manually, in order)

These scripts write to the database. Run from the project root with the venv active.

```bash
# 1. Re-cluster accidents and save HotspotCluster rows
python scripts/save_clusters.py

# 2. Retrain decision tree on full UK_STATS19 dataset and save trained_tree.pkl
python scripts/save_tree.py

# 3. Evaluate tree on 80/20 stratified split and save TreeEvaluation record to DB
python scripts/evaluate_tree.py
```

`train_tree.py` is for exploration (prints tree structure + metrics, then saves the full-dataset pkl). `save_tree.py` is the production serialiser.

---

## Architecture

### Two-phase design

**Offline phase** (`scripts/`) writes to the database; the **web application only reads**. No ML runs at request time — `HotspotCluster` rows are pre-computed by DBSCAN; the decision tree is loaded from `predictions/trained_tree.pkl` once at module import.

### ML constraint

Both DBSCAN and the ID3 decision tree are **implemented from scratch** — no scikit-learn. Do not add scikit-learn imports.

### Bin function symmetry

The `bin_*` functions in `predictions/decision_tree.py` (`bin_time`, `bin_speed`, `bin_weather`, `bin_light`, `bin_road`, `bin_proximity`) are used identically during training (`scripts/`) and at prediction time (`predictions/views.py`). Any change to a bin function must be applied in both places and followed by retraining (`save_tree.py`) and re-evaluation (`evaluate_tree.py`).

### Haversine

All geographic distance calculations use `haversine_distance` from `predictions/dbscan.py`. Do not rewrite or duplicate it.

### Database models (`accidents/models.py`)

- **`AccidentRecord`** — individual accident events. `source` field distinguishes datasets: `UK_STATS19` (real Leeds data, used for all ML), `KTM_SCRAPED` (news-scraped Kathmandu), `KTM_SYNTHETIC_ARCH` (archived synthetic data — excluded from live map).
- **`HotspotCluster`** — pre-computed DBSCAN cluster summaries. Written by `save_clusters.py`, read at runtime.
- **`TreeEvaluation`** — evaluation metrics written by `evaluate_tree.py`, displayed on the analytics page.
- **`DataUpload`** — unused; ignore.

### URL routing

```
/               → dashboard.views.map_view
/analytics/     → dashboard.views.analytics_view
/api/predict/   → predictions.views.predict_risk   (GET)
/admin/         → Django admin
```

### Frontend

The entire frontend is a **single template file**: `dashboard/templates/dashboard/map.html`. Data is embedded as JSON in the HTML response — no AJAX for initial load. Tailwind CSS via CDN, Leaflet.js for the map, `leaflet.heat` for heatmap mode.

The map has three modes (toggled by the sidebar button group):
- **Clusters** (default) — shows `HotspotCluster` circles and `UK_STATS19` accident dots
- **Heatmap** — `L.heatLayer` built from the same accidents array, weighted by severity
- **Route** — placeholder, disabled

The single-point prediction panel is only active in Cluster mode. It POSTs to `/api/predict/`, which fetches current weather from Open-Meteo and computes proximity to the nearest `HotspotCluster` centroid — both are invisible to the user.

### predictions/views.py module-level state

`TRAINED_TREE` (pickle) and `CLUSTER_CENTROIDS` (list of `(lat, lon)` tuples from `HotspotCluster`) are both loaded once at module import. Changes to `HotspotCluster` data require a server restart to take effect.

### Environment / deployment

Local `.env` is loaded via `python-dotenv` (skipped on Render where `RENDER` env var is set). Database is configured via `DATABASE_URL` (Render) or individual `DB_*` vars (local). Deployed on Render — auto-deploys from `main` branch; build command runs `migrate`.

`trained_tree.pkl` is committed to version control (acceptable for this student project).
