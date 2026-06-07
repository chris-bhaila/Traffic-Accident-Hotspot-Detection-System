import json
from collections import defaultdict

from django.db.models import Count
from django.db.models.functions import ExtractHour
from django.shortcuts import render

from accidents.models import AccidentRecord, HotspotCluster, TreeEvaluation


def map_view(request):
    # Load clusters from database (instant, no DBSCAN needed)
    clusters = list(
        HotspotCluster.objects.all().values(
            "id",
            "centroid_latitude",
            "centroid_longitude",
            "accident_count",
            "radius",
            "average_severity",
            "peak_time",
            "peak_day",
            "dominant_weather",
            "risk_level",
            "district",
        )
    )

    # Load accident points
    accidents = list(
        AccidentRecord.objects.filter(source="UK_STATS19").values(
            "latitude",
            "longitude",
            "severity",
            "vehicle_type",
            "accident_type",
            "description",
            "location_name",
            "date",
            "source_url",
            "number_of_casualties",
            "number_of_vehicles",
        )
    )

    context = {
        "clusters_json": json.dumps(clusters, default=str),
        "accidents": accidents,
        "n_clusters": len(clusters),
        "n_total": len(accidents),
    }
    return render(request, "dashboard/map.html", context)


def analytics_view(request):
    uk_qs = AccidentRecord.objects.filter(source="UK_STATS19")

    # Summary counts
    total = uk_qs.count()
    fatal_count = uk_qs.filter(severity="Fatal").count()
    serious_count = uk_qs.filter(severity="Serious").count()
    slight_count = uk_qs.filter(severity="Slight").count()

    # Cluster stats
    clusters_qs = HotspotCluster.objects.all()
    total_clusters = clusters_qs.count()
    risk_raw = dict(
        clusters_qs.values("risk_level")
        .annotate(n=Count("id"))
        .values_list("risk_level", "n")
    )
    risk_breakdown = {lvl: risk_raw.get(lvl, 0) for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]}

    # Accidents by hour (0–23)
    hour_qs = (
        uk_qs.exclude(time=None)
        .annotate(hr=ExtractHour("time"))
        .values("hr")
        .annotate(n=Count("id"))
        .values_list("hr", "n")
    )
    hours = [0] * 24
    for hr, n in hour_qs:
        if hr is not None:
            hours[hr] = n

    peak_h = hours.index(max(hours)) if total > 0 else 0
    if 7 <= peak_h <= 9:
        period = "Morning Rush"
    elif 10 <= peak_h <= 15:
        period = "Midday"
    elif 16 <= peak_h <= 19:
        period = "Evening Rush"
    else:
        period = "Night"
    most_dangerous_hour_label = f"{peak_h:02d}:00 – {period}"

    # Accidents by day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_raw = dict(
        uk_qs.values("day_of_week")
        .annotate(n=Count("id"))
        .values_list("day_of_week", "n")
    )

    # Weather condition breakdown
    weather_qs = list(
        uk_qs.values("weather_condition")
        .annotate(n=Count("id"))
        .order_by("-n")
        .values_list("weather_condition", "n")
    )

    # Road type breakdown (ordered by count desc for consistent chart ordering)
    road_qs = list(
        uk_qs.exclude(road_type=None)
        .exclude(road_type="")
        .values("road_type")
        .annotate(n=Count("id"))
        .order_by("-n")
        .values_list("road_type", "n")
    )
    road_labels = [r for r, _ in road_qs]

    # Road type vs severity matrix
    road_sev_qs = (
        uk_qs.exclude(road_type=None)
        .exclude(road_type="")
        .values("road_type", "severity")
        .annotate(n=Count("id"))
    )
    road_sev = defaultdict(lambda: {"Fatal": 0, "Serious": 0, "Slight": 0})
    for row in road_sev_qs:
        road_sev[row["road_type"]][row["severity"]] = row["n"]

    # Light condition breakdown
    light_qs = list(
        uk_qs.exclude(light_condition=None)
        .exclude(light_condition="")
        .values("light_condition")
        .annotate(n=Count("id"))
        .order_by("-n")
        .values_list("light_condition", "n")
    )

    # Top 10 clusters by accident count
    top_clusters = list(
        clusters_qs.order_by("-accident_count")[:10].values(
            "centroid_latitude",
            "centroid_longitude",
            "accident_count",
            "risk_level",
            "peak_time",
            "peak_day",
            "district",
        )
    )

    chart_data = {
        "hours": hours,
        "day_labels": day_order,
        "day_counts": [day_raw.get(d, 0) for d in day_order],
        "weather_labels": [w for w, _ in weather_qs],
        "weather_values": [n for _, n in weather_qs],
        "road_labels": road_labels,
        "road_sev_fatal": [road_sev[r]["Fatal"] for r in road_labels],
        "road_sev_serious": [road_sev[r]["Serious"] for r in road_labels],
        "road_sev_slight": [road_sev[r]["Slight"] for r in road_labels],
        "light_labels": [lbl for lbl, _ in light_qs],
        "light_values": [n for _, n in light_qs],
        "top_clusters": top_clusters,
        "severity_labels": ["Fatal", "Serious", "Slight"],
        "severity_values": [fatal_count, serious_count, slight_count],
    }

    # Tree evaluation — latest record from the database
    _CLASSES = ["Slight", "Serious", "Fatal"]
    _SEV_COLORS = {"Slight": "#3498db", "Serious": "#f39c12", "Fatal": "#e74c3c"}

    eval_record = TreeEvaluation.objects.order_by("-evaluated_at").first()
    if eval_record:
        cm = json.loads(eval_record.confusion_matrix_json)
        per_class_raw = json.loads(eval_record.per_class_metrics_json)

        max_off_diag = max(
            (cm.get(a, {}).get(p, 0) for a in _CLASSES for p in _CLASSES if a != p),
            default=1,
        ) or 1

        cm_rows = []
        for actual in _CLASSES:
            row_total = sum(cm.get(actual, {}).values()) or 1
            cells = []
            for pred in _CLASSES:
                count = cm.get(actual, {}).get(pred, 0)
                if actual == pred:
                    opacity = 0.12 + (count / row_total) * 0.55
                    bg = f"rgba(34,197,94,{opacity:.2f})"
                    text_color = "#166534"
                else:
                    opacity = (count / max_off_diag) * 0.55 if count > 0 else 0
                    bg = f"rgba(239,68,68,{opacity:.2f})" if count > 0 else "#f8fafc"
                    text_color = "#991b1b" if count > 0 else "#94a3b8"
                cells.append({"count": count, "bg": bg, "text_color": text_color})
            cm_rows.append({"actual": actual, "cells": cells})

        per_class_list = [
            {
                "name": cls,
                "color": _SEV_COLORS.get(cls, "#94a3b8"),
                "precision_pct": f"{per_class_raw.get(cls, {}).get('precision', 0) * 100:.1f}",
                "recall_pct":    f"{per_class_raw.get(cls, {}).get('recall',    0) * 100:.1f}",
                "f1_pct":        f"{per_class_raw.get(cls, {}).get('f1',        0) * 100:.1f}",
                "support":       per_class_raw.get(cls, {}).get("support", 0),
            }
            for cls in _CLASSES
        ]

        tree_eval = {
            "accuracy_pct":  f"{eval_record.accuracy * 100:.1f}",
            "f1_macro_pct":  f"{eval_record.f1_macro * 100:.1f}",
            "train_size":    eval_record.train_size,
            "test_size":     eval_record.test_size,
            "evaluated_at":  eval_record.evaluated_at,
            "classes":       _CLASSES,
            "cm_rows":       cm_rows,
            "per_class":     per_class_list,
        }
    else:
        tree_eval = None

    context = {
        "total": total,
        "fatal_count": fatal_count,
        "serious_count": serious_count,
        "slight_count": slight_count,
        "total_clusters": total_clusters,
        "risk_breakdown": risk_breakdown,
        "most_dangerous_hour_label": most_dangerous_hour_label,
        "chart_data": chart_data,
        "tree_eval": tree_eval,
    }
    return render(request, "dashboard/analytics.html", context)
