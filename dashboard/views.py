import json
from django.shortcuts import render
from accidents.models import AccidentRecord, HotspotCluster


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
