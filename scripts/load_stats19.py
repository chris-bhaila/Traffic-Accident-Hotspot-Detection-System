"""
Load STATS19 collision data into the database.
Run with: python manage.py shell < scripts/load_stats19.py
Or:       python manage.py runscript load_stats19  (if using django-extensions)
"""

import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

import pandas as pd
from datetime import datetime
from accidents.models import AccidentRecord

# ── STATS19 Code Lookups ──

DAY_OF_WEEK = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}

WEATHER_CONDITIONS = {
    1: "Fine no high winds",
    2: "Raining no high winds",
    3: "Snowing no high winds",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
    -1: "Data missing",
}

ROAD_TYPE = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
    12: "One way street/slip road",
    -1: "Data missing",
}

LIGHT_CONDITIONS = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
    -1: "Data missing",
}

JUNCTION_DETAIL = {
    0: "Not at junction",
    1: "Roundabout",
    2: "Mini-roundabout",
    3: "T or staggered junction",
    5: "Slip road",
    6: "Crossroads",
    7: "More than 4 arms (not roundabout)",
    8: "Private drive or entrance",
    9: "Other junction",
    10: "Pedestrian crossing",
    11: "Level crossing",
    19: "Unknown junction type",
    -1: "Data missing",
}

SEVERITY = {
    1: "Fatal",
    2: "Serious",
    3: "Slight",
}


def load_data(csv_path, city_lat_range=None, city_lon_range=None, limit=None):
    """
    Load STATS19 data from CSV into the database.

    Args:
        csv_path: Path to the collisions CSV file
        city_lat_range: Tuple (min_lat, max_lat) to filter by city
        city_lon_range: Tuple (min_lon, max_lon) to filter by city
        limit: Max number of records to load (None for all)
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total records in file: {len(df)}")

    # Drop rows with missing coordinates
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"] != 0]
    print(f"Records with valid coordinates: {len(df)}")

    # Filter by city bounding box if provided
    if city_lat_range and city_lon_range:
        df = df[
            (df["latitude"] >= city_lat_range[0])
            & (df["latitude"] <= city_lat_range[1])
            & (df["longitude"] >= city_lon_range[0])
            & (df["longitude"] <= city_lon_range[1])
        ]
        print(f"Records in city bounding box: {len(df)}")

    if limit:
        df = df.head(limit)
        print(f"Limiting to {limit} records")

    # Map codes to labels
    df["day_of_week_label"] = df["day_of_week"].map(DAY_OF_WEEK).fillna("Unknown")
    df["weather_label"] = df["weather_conditions"].map(WEATHER_CONDITIONS).fillna("Unknown")
    df["road_type_label"] = df["road_type"].map(ROAD_TYPE).fillna("Unknown")
    df["light_label"] = df["light_conditions"].map(LIGHT_CONDITIONS).fillna("Unknown")
    df["junction_label"] = df["junction_detail"].map(JUNCTION_DETAIL).fillna("Unknown")
    df["severity_label"] = df["collision_severity"].map(SEVERITY).fillna("Unknown")

    # Clear existing UK data
    deleted, _ = AccidentRecord.objects.filter(source="UK_STATS19").delete()
    if deleted:
        print(f"Cleared {deleted} existing UK_STATS19 records")

    # Bulk create records
    records = []
    errors = 0
    for _, row in df.iterrows():
        try:
            # Parse date
            date_str = str(row["date"])
            try:
                date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
            except ValueError:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

            # Parse time
            time_obj = None
            if pd.notna(row["time"]):
                try:
                    time_obj = datetime.strptime(str(row["time"]), "%H:%M").time()
                except ValueError:
                    time_obj = None

            records.append(
                AccidentRecord(
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    date=date_obj,
                    time=time_obj,
                    day_of_week=row["day_of_week_label"],
                    weather_condition=row["weather_label"],
                    road_type=row["road_type_label"],
                    light_condition=row["light_label"],
                    speed_limit=int(row["speed_limit"]) if pd.notna(row["speed_limit"]) else None,
                    junction_type=row["junction_label"],
                    severity=row["severity_label"],
                    number_of_vehicles=int(row["number_of_vehicles"]) if pd.notna(row["number_of_vehicles"]) else 1,
                    number_of_casualties=int(row["number_of_casualties"]) if pd.notna(row["number_of_casualties"]) else 0,
                    source="UK_STATS19",
                )
            )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error on row: {e}")

    # Bulk insert in batches of 5000
    batch_size = 5000
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        AccidentRecord.objects.bulk_create(batch)
        print(f"Inserted {min(i + batch_size, total)}/{total} records...")

    print(f"\nDone! Loaded {total} records. Errors: {errors}")


if __name__ == "__main__":
    CSV_PATH = "data/dft-road-casualty-statistics-collision-2023.csv"

    # Leeds bounding box - medium city, good for testing
    # Similar density/complexity to Kathmandu
    LEEDS_LAT = (53.70, 53.87)
    LEEDS_LON = (-1.70, -1.40)

    load_data(
        csv_path=CSV_PATH,
        city_lat_range=LEEDS_LAT,
        city_lon_range=LEEDS_LON,
    )
