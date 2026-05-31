import sys
import os
import django

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings as _django_settings
_db = _django_settings.DATABASES["default"]
print(f"[DB] host={_db.get('HOST', '?')} name={_db.get('NAME', '?')}")

import numpy as np
import random
from datetime import datetime, timedelta, time
from accidents.models import AccidentRecord

random.seed(42)
np.random.seed(42)

# ── Kathmandu Valley Accident Hotspot Definitions ──
# Based on real data from Metropolitan Traffic Police Division
# and Kathmandu Post reporting

HOTSPOTS = {
    # (lat, lon, radius_meters, weight, name)
    # Weight reflects relative accident frequency from real reports
    # Kalanki-Koteshwor Ring Road (highest accident density in valley)
    "Kalanki Chowk": (27.6933, 85.2814, 200, 50),
    "Balkhu Bridge": (27.6870, 85.3000, 150, 30),
    "Dhobighat": (27.6800, 85.3040, 150, 20),
    "Ekantakuna": (27.6680, 85.3100, 200, 35),
    "Sanepa Chowk": (27.6850, 85.3100, 150, 30),
    "Nakkhu": (27.6600, 85.3150, 150, 20),
    "Satdobato": (27.6580, 85.3220, 200, 40),
    "Mahalaxmisthan": (27.6650, 85.3180, 150, 25),
    "Gwarko": (27.6660, 85.3300, 150, 25),
    "Balkumari": (27.6700, 85.3450, 200, 35),
    "B&B Hospital Area": (27.6620, 85.3250, 200, 45),
    "Koteshwor": (27.6780, 85.3492, 250, 45),
    # Other major Kathmandu hotspots
    "Chabahil": (27.7172, 85.3411, 200, 25),
    "Gongabu Bus Park": (27.7350, 85.3150, 200, 35),
    "Balaju": (27.7270, 85.3020, 150, 20),
    "Maharajgunj": (27.7370, 85.3280, 150, 20),
    "Baneshwor": (27.6910, 85.3380, 200, 30),
    "Tinkune": (27.6850, 85.3450, 150, 25),
    "Gaushala": (27.7100, 85.3380, 150, 15),
    "Kalimati": (27.6970, 85.3030, 150, 20),
    "Samakhusi": (27.7300, 85.3120, 150, 15),
    "Machapokhari": (27.7250, 85.3050, 150, 15),
    "Basundhara": (27.7350, 85.3300, 150, 15),
    "Jorpati": (27.7270, 85.3610, 150, 10),
    "Bouddha": (27.7210, 85.3620, 150, 15),
    "Kapan": (27.7400, 85.3490, 150, 10),
    "Tokha": (27.7470, 85.3140, 150, 10),
    "Jadibuti": (27.6740, 85.3530, 150, 20),
    "Lokanthali": (27.6760, 85.3580, 150, 15),
    "Sinamangal": (27.6900, 85.3500, 150, 15),
    "Tripureshwor": (27.6980, 85.3140, 150, 15),
    "Putalisadak": (27.7050, 85.3210, 150, 10),
    "Thapathali": (27.6920, 85.3220, 150, 15),
    "Thankot": (27.6880, 85.2540, 200, 20),
    "Kirtipur": (27.6790, 85.2780, 150, 10),
    "Swoyambhu": (27.7150, 85.2900, 150, 10),
    "Kupondole": (27.6880, 85.3150, 100, 10),
    "Pulchowk": (27.6810, 85.3180, 100, 10),
    # Bhaktapur hotspots
    "Thimi": (27.6730, 85.3870, 200, 30),
    "Bhaktapur Durbar": (27.6710, 85.4280, 150, 15),
    "Suryabinayak": (27.6650, 85.4450, 150, 15),
    "Sallaghari": (27.6700, 85.4100, 150, 15),
    "Gatthaghar": (27.6750, 85.3960, 150, 15),
    "Radhe Radhe": (27.6720, 85.3800, 150, 10),
    "Jagati": (27.6670, 85.4070, 100, 10),
    "Changunarayan": (27.7100, 85.4270, 100, 8),
    "Duwakot": (27.6950, 85.4050, 100, 8),
    # Lalitpur hotspots
    "Jawalakhel": (27.6720, 85.3130, 150, 20),
    "Lagankhel": (27.6670, 85.3230, 150, 15),
    "Mangalbazar Patan": (27.6720, 85.3250, 100, 10),
    "Imadol": (27.6600, 85.3370, 100, 10),
    "Lubhu": (27.6490, 85.3400, 100, 8),
    "Kusunti": (27.6730, 85.3080, 100, 10),
    "Chapagaun": (27.6250, 85.3580, 100, 5),
    "Godavari": (27.5970, 85.3840, 100, 5),
}

# ── Time Distribution (based on real traffic police data) ──
# Evening rush and night have highest accident rates
HOUR_WEIGHTS = {
    0: 3,
    1: 2,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 3,
    7: 6,
    8: 8,
    9: 5,
    10: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 7,
    15: 8,
    16: 9,
    17: 10,
    18: 9,
    19: 7,
    20: 6,
    21: 5,
    22: 4,
    23: 3,
}

# ── Weather Distribution (Nepal context) ──
# Monsoon months (June-Sep) have rain, rest mostly fine
WEATHER_BY_MONTH = {
    1: [("Fine no high winds", 0.80), ("Fog or mist", 0.15), ("Other", 0.05)],
    2: [("Fine no high winds", 0.85), ("Fog or mist", 0.10), ("Other", 0.05)],
    3: [("Fine no high winds", 0.85), ("Fine + high winds", 0.10), ("Other", 0.05)],
    4: [
        ("Fine no high winds", 0.70),
        ("Raining no high winds", 0.20),
        ("Fine + high winds", 0.10),
    ],
    5: [
        ("Fine no high winds", 0.55),
        ("Raining no high winds", 0.35),
        ("Raining + high winds", 0.10),
    ],
    6: [
        ("Fine no high winds", 0.30),
        ("Raining no high winds", 0.45),
        ("Raining + high winds", 0.20),
        ("Other", 0.05),
    ],
    7: [
        ("Fine no high winds", 0.20),
        ("Raining no high winds", 0.50),
        ("Raining + high winds", 0.25),
        ("Other", 0.05),
    ],
    8: [
        ("Fine no high winds", 0.25),
        ("Raining no high winds", 0.45),
        ("Raining + high winds", 0.25),
        ("Other", 0.05),
    ],
    9: [
        ("Fine no high winds", 0.35),
        ("Raining no high winds", 0.40),
        ("Raining + high winds", 0.20),
        ("Other", 0.05),
    ],
    10: [
        ("Fine no high winds", 0.75),
        ("Raining no high winds", 0.15),
        ("Fog or mist", 0.05),
        ("Other", 0.05),
    ],
    11: [("Fine no high winds", 0.80), ("Fog or mist", 0.10), ("Other", 0.10)],
    12: [("Fine no high winds", 0.80), ("Fog or mist", 0.15), ("Other", 0.05)],
}

# ── Severity Distribution (from real data: ~3-4% fatal, ~35% serious, ~62% slight) ──
SEVERITY_WEIGHTS = [("Fatal", 0.04), ("Serious", 0.35), ("Slight", 0.61)]

# ── Road Type Distribution ──
ROAD_TYPE_WEIGHTS = [
    ("Single carriageway", 0.55),
    ("Dual carriageway", 0.30),
    ("Roundabout", 0.05),
    ("One way street", 0.05),
    ("Other", 0.05),
]

# ── Vehicle Types (70% two-wheeler based on real data) ──
VEHICLE_TYPES = [
    ("Motorcycle", 0.45),
    ("Scooter", 0.25),
    ("Bus", 0.08),
    ("Truck", 0.06),
    ("Car", 0.05),
    ("Microbus", 0.04),
    ("Tempo", 0.03),
    ("Bicycle", 0.02),
    ("Pedestrian", 0.02),
]

# ── Accident Types ──
ACCIDENT_TYPES = [
    ("Vehicle-pedestrian collision", 0.25),
    ("Two-wheeler collision with vehicle", 0.20),
    ("Head-on collision", 0.12),
    ("Rear-end collision", 0.10),
    ("Side collision at junction", 0.10),
    ("Vehicle lost control", 0.08),
    ("Hit and run", 0.07),
    ("Single vehicle accident", 0.05),
    ("Multiple vehicle pileup", 0.03),
]

# ── Description Templates ──
DESCRIPTIONS = {
    "Vehicle-pedestrian collision": [
        "A {vehicle} struck a pedestrian crossing the road near {location}.",
        "Pedestrian hit by a {vehicle} while attempting to cross at {location}.",
        "A speeding {vehicle} hit a pedestrian near {location}. The victim was rushed to nearby hospital.",
    ],
    "Two-wheeler collision with vehicle": [
        "A {vehicle} collided with a {vehicle2} near {location}.",
        "Collision between a {vehicle} and a {vehicle2} at {location} intersection.",
        "A {vehicle2} hit a {vehicle} from behind near {location}.",
    ],
    "Head-on collision": [
        "Head-on collision between a {vehicle} and a {vehicle2} near {location}.",
        "Two vehicles collided head-on at {location} after one veered into oncoming lane.",
    ],
    "Rear-end collision": [
        "A {vehicle} rear-ended a {vehicle2} near {location} during heavy traffic.",
        "Rear-end collision involving a {vehicle} and {vehicle2} at {location}.",
    ],
    "Side collision at junction": [
        "Side collision at {location} junction between a {vehicle} and a {vehicle2}.",
        "A {vehicle} failed to yield at {location} junction, colliding with a {vehicle2}.",
    ],
    "Vehicle lost control": [
        "A {vehicle} lost control near {location} and crashed into the roadside barrier.",
        "Driver of a {vehicle} lost control near {location}, possibly due to overspeeding.",
    ],
    "Hit and run": [
        "Hit and run incident near {location}. An unidentified vehicle struck a {vehicle} and fled.",
        "A pedestrian was hit by an unidentified vehicle near {location}. Police investigating.",
    ],
    "Single vehicle accident": [
        "A {vehicle} skidded off the road near {location}.",
        "Single vehicle accident involving a {vehicle} near {location}. Driver lost control.",
    ],
    "Multiple vehicle pileup": [
        "Multiple vehicles involved in a pileup near {location} during {weather} conditions.",
        "Chain collision involving several vehicles near {location}.",
    ],
}


# ── Light Condition by Hour ──
def get_light_condition(hour):
    if 6 <= hour <= 17:
        return "Daylight"
    elif hour >= 18 or hour <= 5:
        choices = [
            ("Darkness - lights lit", 0.70),
            ("Darkness - no lighting", 0.15),
            ("Darkness - lighting unknown", 0.10),
            ("Darkness - lights unlit", 0.05),
        ]
        labels, weights = zip(*choices)
        return random.choices(labels, weights=weights, k=1)[0]


def weighted_choice(choices):
    """Pick from list of (value, weight) tuples."""
    labels, weights = zip(*choices)
    return random.choices(labels, weights=weights, k=1)[0]


def random_point_near(lat, lon, radius_meters):
    """Generate a random point within radius_meters of (lat, lon)."""
    # Convert radius to approximate degrees
    lat_offset = radius_meters / 111000  # ~111km per degree latitude
    lon_offset = radius_meters / (111000 * np.cos(np.radians(lat)))

    new_lat = lat + random.uniform(-lat_offset, lat_offset)
    new_lon = lon + random.uniform(-lon_offset, lon_offset)
    return new_lat, new_lon


def get_district(lat, lon):
    """Rough district assignment based on coordinates."""
    if lon > 85.37:
        return "Bhaktapur"
    elif lat < 27.68 and lon < 85.33:
        return "Lalitpur"
    else:
        return "Kathmandu"


def generate_accidents(n_records=2000, start_date="2023-01-01", end_date="2025-12-31"):
    """Generate synthetic accident records based on real patterns."""

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days_range = (end - start).days

    # Build weighted hotspot selection
    hotspot_names = list(HOTSPOTS.keys())
    hotspot_weights = [HOTSPOTS[h][3] for h in hotspot_names]

    # Build hour selection weights
    hours = list(HOUR_WEIGHTS.keys())
    hour_weights = list(HOUR_WEIGHTS.values())

    # Day of week weights (Friday slightly higher based on STATS19 pattern)
    day_weights = {
        "Monday": 14,
        "Tuesday": 15,
        "Wednesday": 14,
        "Thursday": 15,
        "Friday": 18,
        "Saturday": 13,
        "Sunday": 11,
    }
    day_names = list(day_weights.keys())
    day_w = list(day_weights.values())

    records = []
    speed_limits = [20, 30, 40, 50, 60]
    speed_weights = [0.05, 0.50, 0.25, 0.10, 0.10]

    for i in range(n_records):
        # Pick hotspot
        hotspot_name = random.choices(hotspot_names, weights=hotspot_weights, k=1)[0]
        h_lat, h_lon, h_radius, _ = HOTSPOTS[hotspot_name]

        # Generate point near hotspot
        lat, lon = random_point_near(h_lat, h_lon, h_radius)

        # Random date
        date = start + timedelta(days=random.randint(0, days_range))
        month = date.month

        # Pick hour (weighted)
        hour = random.choices(hours, weights=hour_weights, k=1)[0]
        minute = random.randint(0, 59)
        time_obj = time(hour, minute)

        # Day of week — use actual day from date
        day_of_week = date.strftime("%A")

        # Weather based on month
        weather = weighted_choice(WEATHER_BY_MONTH[month])

        # Severity
        severity = weighted_choice(SEVERITY_WEIGHTS)

        # Road type
        road_type = weighted_choice(ROAD_TYPE_WEIGHTS)

        # Light condition based on hour
        light = get_light_condition(hour)

        # Speed limit
        speed = random.choices(speed_limits, weights=speed_weights, k=1)[0]

        # Vehicles and casualties
        n_vehicles = random.choices([1, 2, 3], weights=[0.2, 0.6, 0.2], k=1)[0]
        n_casualties = random.choices(
            [1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03], k=1
        )[0]
        # Deaths (only for Fatal accidents)
        if severity == "Fatal":
            n_deaths = random.choices([1, 2, 3], weights=[0.75, 0.20, 0.05], k=1)[0]
        else:
            n_deaths = 0

        district = get_district(lat, lon)

        # Vehicle type
        vehicle = weighted_choice(VEHICLE_TYPES)
        vehicle2_choices = [v for v in VEHICLE_TYPES if v[0] != vehicle]
        vehicle2 = weighted_choice(vehicle2_choices)

        # Accident type
        accident_type = weighted_choice(ACCIDENT_TYPES)

        # Generate description
        templates = DESCRIPTIONS.get(accident_type, ["Accident near {location}."])
        desc_template = random.choice(templates)
        description = desc_template.format(
            vehicle=vehicle.lower(),
            vehicle2=vehicle2.lower() if "{vehicle2}" in desc_template else "",
            location=hotspot_name,
            weather=weather.lower(),
        )

        records.append({
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "date": date.date(),
            "time": time_obj,
            "day_of_week": day_of_week,
            "weather_condition": weather,
            "road_type": road_type,
            "light_condition": light,
            "speed_limit": speed,
            "junction_type": random.choice(["Not at junction", "T or staggered junction", "Crossroads", "Roundabout"]),
            "severity": severity,
            "number_of_vehicles": n_vehicles,
            "number_of_casualties": n_casualties,
            "source": "KTM_SYNTHETIC",
            "vehicle_type": vehicle,
            "accident_type": accident_type,
            "description": description,
            "location_name": hotspot_name,
        })

    return records


def save_records(records):
    """Save generated records to database."""
    deleted, _ = AccidentRecord.objects.filter(source="KTM_SYNTHETIC").delete()
    if deleted:
        print(f"Cleared {deleted} existing synthetic records")

    batch = []
    for r in records:
        batch.append(AccidentRecord(**r))

    AccidentRecord.objects.bulk_create(batch, batch_size=5000)
    print(f"Saved {len(batch)} synthetic Kathmandu Valley records")


def print_summary(records):
    """Print distribution summary."""
    from collections import Counter

    print("\n--- Generation Summary ---")
    print(f"Total records: {len(records)}")

    severities = Counter(r["severity"] for r in records)
    print(f"\nSeverity: {dict(severities)}")

    districts = Counter(get_district(r["latitude"], r["longitude"]) for r in records)
    print(f"Districts: {dict(districts)}")

    weathers = Counter(r["weather_condition"] for r in records)
    print(f"\nWeather:")
    for w, c in weathers.most_common():
        print(f"  {w}: {c} ({c/len(records)*100:.1f}%)")

    hours = Counter(r["time"].hour for r in records)
    print(f"\nPeak hours:")
    for h in sorted(hours.keys()):
        bar = "#" * (hours[h] // 5)
        print(f"  {h:02d}:00  {hours[h]:>4}  {bar}")


if __name__ == "__main__":
    records = generate_accidents(n_records=2000)
    print_summary(records)
    save_records(records)
