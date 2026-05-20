"""
Scrape Kathmandu Valley road accident reports from Kathmandu Post.

Strategy:
  1. Search-based discovery via Google (site:kathmandupost.com + accident keywords)
  2. Category page crawling (/valley/kathmandu, /valley/lalitpur, /valley/bhaktapur,
     /national with Valley filter)
  3. Hardcoded seed URLs for known good articles

For each article: fetch full text, extract location (Valley-only), severity,
time, weather, date, casualty counts. Save to AccidentRecord with source="KTM_SCRAPED".

Usage:
    python scripts/scrape_accidents.py            # full run
    python scripts/scrape_accidents.py --dry-run   # fetch + extract but don't save to DB
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from datetime import datetime
from collections import Counter

# ── Django setup ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

import requests
from bs4 import BeautifulSoup

from accidents.models import AccidentRecord

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
        "Gecko/20100101 Firefox/128.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

POLITE_DELAY = 2.0  # seconds between requests to the same domain
REQUEST_TIMEOUT = 20

# Accident-related keywords for URL/title filtering
ACCIDENT_KEYWORDS = [
    "accident", "crash", "collision", "killed", "dead", "dies",
    "injured", "fatal", "hit-and-run", "hit and run", "run over",
    "mowed down", "knocked down", "road mishap", "pile-up", "pileup",
    "plunge", "veered off", "overturned", "overturn",
]

# ────────────────────────────────────────────────────────────────────
# Kathmandu Valley Location Dictionary
# ────────────────────────────────────────────────────────────────────
# Coordinates are (lat, lon). Sorted by string length descending at
# lookup time so "new baneshwor" matches before "baneshwor".
LOCATION_COORDS = {
    # ── Kathmandu District ──
    "kalanki chowk": (27.6933, 85.2814),
    "new baneshwor": (27.6910, 85.3380),
    "old baneshwor": (27.6870, 85.3350),
    "madhyapur thimi": (27.6730, 85.3870),
    "kamal binayak": (27.6720, 85.4190),
    "radhe radhe": (27.6720, 85.3800),
    "naya bazar": (27.7120, 85.3030),
    "new road": (27.7040, 85.3120),
    "ring road": (27.7000, 85.3300),
    "koteshwor": (27.6780, 85.3492),
    "koteshwar": (27.6780, 85.3492),
    "kalanki": (27.6933, 85.2814),
    "chabahil": (27.7172, 85.3411),
    "gongabu": (27.7350, 85.3150),
    "balaju": (27.7270, 85.3020),
    "maharajgunj": (27.7370, 85.3280),
    "baneshwor": (27.6910, 85.3380),
    "tinkune": (27.6850, 85.3450),
    "sinamangal": (27.6900, 85.3500),
    "gaushala": (27.7100, 85.3380),
    "putalisadak": (27.7050, 85.3210),
    "ratnapark": (27.7060, 85.3150),
    "ratna park": (27.7060, 85.3150),
    "ason": (27.7080, 85.3110),
    "indrachowk": (27.7060, 85.3100),
    "thamel": (27.7150, 85.3120),
    "lazimpat": (27.7200, 85.3230),
    "naxal": (27.7170, 85.3270),
    "durbarmarg": (27.7130, 85.3190),
    "durbar marg": (27.7130, 85.3190),
    "jawalakhel": (27.6720, 85.3130),
    "satdobato": (27.6580, 85.3220),
    "balkhu": (27.6870, 85.3000),
    "kalimati": (27.6970, 85.3030),
    "swoyambhu": (27.7150, 85.2900),
    "swayambhu": (27.7150, 85.2900),
    "kirtipur": (27.6790, 85.2780),
    "thankot": (27.6880, 85.2540),
    "jadibuti": (27.6740, 85.3530),
    "lokanthali": (27.6760, 85.3580),
    "thapathali": (27.6920, 85.3220),
    "tripureshwor": (27.6980, 85.3140),
    "tripureshwar": (27.6980, 85.3140),
    "maitighar": (27.6960, 85.3230),
    "singhadurbar": (27.7000, 85.3220),
    "singha durbar": (27.7000, 85.3220),
    "babarmahal": (27.6930, 85.3290),
    "minbhawan": (27.6870, 85.3420),
    "machapokhari": (27.7250, 85.3050),
    "machhapokhari": (27.7250, 85.3050),
    "samakhusi": (27.7300, 85.3120),
    "tokha": (27.7470, 85.3140),
    "budhanilkantha": (27.7680, 85.3640),
    "jorpati": (27.7270, 85.3610),
    "bouddha": (27.7210, 85.3620),
    "boudha": (27.7210, 85.3620),
    "bauddha": (27.7210, 85.3620),
    "kapan": (27.7400, 85.3490),
    "basundhara": (27.7350, 85.3300),
    "bagbazar": (27.7050, 85.3170),
    "newroad": (27.7040, 85.3120),
    "sundhara": (27.7010, 85.3140),
    "anamnagar": (27.7000, 85.3250),
    "kamalpokhari": (27.7100, 85.3250),
    "dillibazar": (27.7100, 85.3300),
    "dilli bazar": (27.7100, 85.3300),
    "battisputali": (27.7090, 85.3400),
    "mitrapark": (27.7050, 85.3380),
    "mitra park": (27.7050, 85.3380),
    "sanepa": (27.6850, 85.3100),
    "ekantakuna": (27.6680, 85.3100),
    "imadol": (27.6600, 85.3370),
    "lubhu": (27.6490, 85.3400),
    "godavari": (27.5970, 85.3840),
    "chapagaun": (27.6250, 85.3580),
    "kupondole": (27.6880, 85.3150),
    "kupondol": (27.6880, 85.3150),
    "pulchowk": (27.6810, 85.3180),
    "mangalbazar": (27.6720, 85.3250),
    "mangal bazar": (27.6720, 85.3250),
    "lagankhel": (27.6670, 85.3230),
    "kusunti": (27.6730, 85.3080),
    "dhobighat": (27.6800, 85.3040),
    "patan": (27.6720, 85.3180),
    "lalitpur": (27.6720, 85.3180),
    "gwarko": (27.6650, 85.3310),
    "balkumari": (27.6700, 85.3370),
    "mahalaxmisthan": (27.6750, 85.3100),
    "sankhu": (27.7500, 85.4000),
    "sundarijal": (27.7680, 85.4200),
    "shankharapur": (27.7550, 85.4100),
    "sitapaila": (27.7150, 85.2800),
    "sukedhara": (27.7320, 85.3250),
    "rabibhawan": (27.6980, 85.2950),
    "soltimode": (27.6960, 85.2870),
    "gwarko": (27.6660, 85.3300),
    # ── Bhaktapur District ──
    "thimi": (27.6730, 85.3870),
    "bhaktapur": (27.6710, 85.4280),
    "suryabinayak": (27.6650, 85.4450),
    "sallaghari": (27.6700, 85.4100),
    "kamalbinayak": (27.6720, 85.4190),
    "changunarayan": (27.7100, 85.4270),
    "gatthaghar": (27.6750, 85.3960),
    "jagati": (27.6670, 85.4070),
    "dudhpati": (27.6720, 85.4300),
    "nagarkot": (27.7150, 85.5200),
    "duwakot": (27.6950, 85.4050),
    "sipadol": (27.6550, 85.4350),
    # ── Lalitpur District ──
    "hattiban": (27.6550, 85.3100),
    "tikabhairab": (27.6400, 85.3500),
    "lele": (27.6100, 85.3400),
    "bungamati": (27.6400, 85.3150),
    "khokana": (27.6450, 85.3050),
}

# Pre-sort location names by length descending for longest-match-first
_SORTED_LOCATIONS = sorted(LOCATION_COORDS.keys(), key=len, reverse=True)

# ── Severity Keywords (ordered by priority) ──
_FATAL_KEYWORDS = [
    "killed", "dead", "died", "death", "fatal", "succumbed",
    "pronounced dead", "lost his life", "lost her life",
    "claimed the life", "lost their lives",
]
_SERIOUS_KEYWORDS = [
    "seriously injured", "serious injuries", "critical condition",
    "critically injured", "grievous", "critical",
]

# ── Time Keywords ──
TIME_KEYWORDS = {
    "early morning": "06:00",
    "late night": "23:00",
    "morning": "08:00",
    "forenoon": "10:00",
    "noon": "12:00",
    "midday": "12:00",
    "afternoon": "14:00",
    "evening": "18:00",
    "night": "21:00",
    "midnight": "00:00",
    "dawn": "05:30",
    "dusk": "18:30",
}
_SORTED_TIME_KW = sorted(TIME_KEYWORDS.keys(), key=len, reverse=True)

# ── Weather Keywords ──
WEATHER_KEYWORDS = {
    "heavy rain": "Raining + high winds",
    "monsoon": "Raining no high winds",
    "rainy": "Raining no high winds",
    "rain": "Raining no high winds",
    "wet road": "Raining no high winds",
    "foggy": "Fog or mist",
    "fog": "Fog or mist",
    "misty": "Fog or mist",
    "mist": "Fog or mist",
    "snow": "Snowing no high winds",
}
_SORTED_WEATHER_KW = sorted(WEATHER_KEYWORDS.keys(), key=len, reverse=True)


# ════════════════════════════════════════════════════════════════════
# EXTRACTION HELPERS
# ════════════════════════════════════════════════════════════════════


def extract_location(title: str, body: str):
    """
    Extract the most relevant Kathmandu Valley location from the article.

    Priority: title match first (accident headlines usually name the place),
    then first match in the *first 3 sentences* of the body (the lede).
    This avoids matching hospital names or police station locations mentioned
    later in the article.

    Returns (location_name, (lat, lon)) or (None, None).
    """
    # Try title first — most specific signal
    title_lower = title.lower()
    for loc in _SORTED_LOCATIONS:
        if loc in title_lower:
            return loc, LOCATION_COORDS[loc]

    # Try first ~3 sentences of body (the lede paragraph)
    body_lower = body.lower()
    lede = body_lower[:600]  # roughly 3 sentences
    for loc in _SORTED_LOCATIONS:
        if loc in lede:
            return loc, LOCATION_COORDS[loc]

    # Fallback: anywhere in body, but this is lower confidence
    for loc in _SORTED_LOCATIONS:
        if loc in body_lower:
            return loc, LOCATION_COORDS[loc]

    return None, None


def extract_severity(text: str) -> str:
    """Return highest severity found: Fatal > Serious > Slight."""
    text_lower = text.lower()
    for kw in _FATAL_KEYWORDS:
        if kw in text_lower:
            return "Fatal"
    for kw in _SERIOUS_KEYWORDS:
        if kw in text_lower:
            return "Serious"
    if "injured" in text_lower:
        return "Slight"
    return "Slight"


def extract_time(text: str):
    """
    Extract time from text. Tries explicit patterns first (10:15 am),
    then keyword matching. Returns "HH:MM" string or None.
    """
    text_lower = text.lower()

    # Explicit time: "10:15 am", "4:50pm", "12.30 am"
    m = re.search(r"(\d{1,2})[:\.](\d{2})\s*(am|pm)", text_lower)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        ampm = m.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    # "around 4 pm", "at 6 am"
    m = re.search(r"(?:around|at|about)\s+(\d{1,2})\s*(am|pm)", text_lower)
    if m:
        hour = int(m.group(1))
        ampm = m.group(2)
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:00"

    # Keyword matching
    for kw in _SORTED_TIME_KW:
        if kw in text_lower:
            return TIME_KEYWORDS[kw]

    return None


def extract_weather(text: str) -> str:
    text_lower = text.lower()
    for kw in _SORTED_WEATHER_KW:
        if kw in text_lower:
            return WEATHER_KEYWORDS[kw]
    return "Fine no high winds"


def extract_casualties(text: str) -> dict:
    """
    Try to extract actual casualty/death numbers from text.
    Returns {"deaths": int or None, "casualties": int or None}.
    """
    text_lower = text.lower()
    deaths = None
    casualties = None

    # "three killed", "3 dead", "two died"
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12,
    }
    num_pattern = r"(\d+|" + "|".join(number_words.keys()) + r")"

    for pattern in [
        rf"{num_pattern}\s+(?:people\s+)?(?:killed|dead|died|deaths?)",
        rf"(?:killed|dead|died)\s+{num_pattern}",
        rf"claim(?:ed|s)?\s+{num_pattern}\s+lives?",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            val = m.group(1)
            deaths = number_words.get(val, None) or int(val) if val.isdigit() else number_words.get(val)
            break

    for pattern in [
        rf"{num_pattern}\s+(?:people\s+)?(?:injured|hurt|wounded)",
        rf"(?:injuring)\s+{num_pattern}",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            val = m.group(1)
            casualties = number_words.get(val, None) or int(val) if val.isdigit() else number_words.get(val)
            break

    return {"deaths": deaths, "casualties": casualties}


def extract_vehicle_type(text: str) -> str:
    """Best-effort vehicle type extraction."""
    text_lower = text.lower()
    # Check longest/most-specific first
    vehicle_map = [
        ("microbus", "Bus"),
        ("bus", "Bus"),
        ("truck", "Goods vehicle"),
        ("tipper", "Goods vehicle"),
        ("lorry", "Goods vehicle"),
        ("motorcycle", "Motorcycle"),
        ("motorbike", "Motorcycle"),
        ("bike", "Motorcycle"),
        ("scooter", "Motorcycle"),
        ("car", "Car"),
        ("jeep", "Car"),
        ("taxi", "Taxi"),
        ("auto rickshaw", "Other"),
        ("tempo", "Other"),
        ("bicycle", "Pedal cycle"),
        ("pedestrian", "Pedestrian"),
    ]
    for keyword, vtype in vehicle_map:
        if keyword in text_lower:
            return vtype
    return "Unknown"


# ════════════════════════════════════════════════════════════════════
# DATE PARSING
# ════════════════════════════════════════════════════════════════════

DATE_FORMATS = [
    "%B %d, %Y",   # November 28, 2024
    "%b %d, %Y",   # Nov 28, 2024
    "%Y-%m-%d",    # 2024-11-28
    "%d/%m/%Y",    # 28/11/2024
]


def parse_date(date_str: str):
    """Parse a date string, return datetime.date or None."""
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


# ════════════════════════════════════════════════════════════════════
# ARTICLE FETCHING & PARSING
# ════════════════════════════════════════════════════════════════════


def fetch_html(url: str, session: requests.Session) -> str | None:
    """Fetch a URL and return raw HTML, or None on failure."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.text
        log.warning("HTTP %d for %s", resp.status_code, url)
        return None
    except requests.RequestException as e:
        log.warning("Request failed for %s: %s", url, e)
        return None


def parse_article(html: str, url: str) -> dict | None:
    """
    Parse a Kathmandu Post article page.
    Returns {"title", "body", "date_str", "url"} or None.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title: find meaningful h1 (skip site branding)
    title = ""
    for h1 in soup.find_all("h1"):
        text = h1.get_text(strip=True)
        if text and len(text) > 10 and "Without Fear" not in text:
            title = text
            break

    # Body: story-section paragraphs
    story = soup.find("section", class_="story-section")
    if not story:
        # Fallback: look for article tag or main content div
        story = soup.find("article") or soup.find("div", class_="main-content")

    if story:
        paragraphs = story.find_all("p")
        body = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    else:
        body = ""

    if not body and not title:
        return None

    # Date: "Published at : November 28, 2024"
    date_str = ""
    for div in soup.find_all("div", class_="updated-time"):
        text = div.get_text(strip=True)
        if "Published at" in text:
            date_str = re.sub(r"Published at\s*:?\s*", "", text).strip()
            break

    # Fallback: try meta og:url which often has date in path
    if not date_str:
        og_url = soup.find("meta", property="og:url")
        if og_url:
            m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", og_url.get("content", ""))
            if m:
                date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return {
        "title": title,
        "body": body,
        "date_str": date_str,
        "url": url,
    }


# ════════════════════════════════════════════════════════════════════
# URL DISCOVERY
# ════════════════════════════════════════════════════════════════════


def _is_accident_link(href: str, anchor_text: str) -> bool:
    """Check if a link likely points to an accident article."""
    combined = (href + " " + anchor_text).lower()
    return any(kw in combined for kw in ACCIDENT_KEYWORDS)


def discover_from_category_pages(
    session: requests.Session,
    max_pages_per_category: int = 15,
) -> set[str]:
    """
    Crawl Kathmandu Post category/section pages for accident article URLs.
    Focuses on Valley sub-sections and national news.
    """
    categories = [
        # Valley sections — highest signal for Kathmandu Valley accidents
        "https://kathmandupost.com/valley",
        "https://kathmandupost.com/valley/kathmandu",
        "https://kathmandupost.com/valley/lalitpur",
        "https://kathmandupost.com/valley/bhaktapur",
        # National — covers Valley accidents too, but also rest of Nepal
        "https://kathmandupost.com/national",
        "https://kathmandupost.com/national/province-no-3",  # Bagmati Province
    ]

    urls = set()

    for cat_url in categories:
        for page_num in range(1, max_pages_per_category + 1):
            page_url = f"{cat_url}?page={page_num}" if page_num > 1 else cat_url
            log.info("Scanning %s", page_url)

            html = fetch_html(page_url, session)
            if not html:
                break  # likely hit last page or got blocked

            soup = BeautifulSoup(html, "html.parser")
            found_on_page = 0

            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True)
                if _is_accident_link(href, text):
                    full = href if href.startswith("http") else "https://kathmandupost.com" + href
                    if full not in urls:
                        urls.add(full)
                        found_on_page += 1

            log.info("  Found %d accident links on page %d", found_on_page, page_num)

            # If no accident links found for 2 consecutive pages, move on
            if found_on_page == 0 and page_num > 2:
                break

            time.sleep(POLITE_DELAY)

    return urls


def get_seed_urls() -> set[str]:
    """
    Manually curated seed URLs known to be Kathmandu Valley accident articles.
    Expand this list as you find more confirmed articles.
    """
    return {
        "https://kathmandupost.com/national/2024/11/28/one-dead-in-bhaktapur-road-accident",
        "https://kathmandupost.com/province-no-3/2024/03/11/traffic-police-officer-injured-in-road-accident-dies",
        "https://kathmandupost.com/national/2025/02/12/three-dead-40-injured-in-bus-accident-in-kathmandu",
        "https://kathmandupost.com/national/2024/11/04/three-dead-in-separate-road-accidents",
        "https://kathmandupost.com/valley/2019/06/05/kalanki-koteshwor-becomes-killer-road-recording-658-accidents-in-10-months",
        "https://kathmandupost.com/valley/2019/12/03/despite-traffic-police-s-assurances-pedestrians-are-dying-on-kathmandu-streets",
    }


# ════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════


def process_article(article_data: dict) -> dict | None:
    """
    Extract structured accident record from a parsed article.
    Returns a dict ready for DB insertion, or None if not a
    Valley accident or not enough data.
    """
    title = article_data["title"]
    body = article_data["body"]
    full_text = title + " " + body

    # Location extraction (Valley only)
    loc_name, coords = extract_location(title, body)
    if not coords:
        return None  # not a Kathmandu Valley article

    lat, lon = coords

    # Core fields
    severity = extract_severity(full_text)
    time_str = extract_time(full_text)
    weather = extract_weather(full_text)
    date = parse_date(article_data["date_str"])
    day_of_week = date.strftime("%A") if date else None
    vehicle_type = extract_vehicle_type(full_text)
    casualties = extract_casualties(full_text)

    return {
        "latitude": lat,
        "longitude": lon,
        "date": date,
        "time": time_str,
        "day_of_week": day_of_week,
        "weather_condition": weather,
        "severity": severity,
        "location_name": loc_name,
        "source_url": article_data["url"],
        "title": title,
        "vehicle_type": vehicle_type,
        "number_of_deaths": casualties["deaths"],
        "number_of_casualties": casualties["casualties"],
    }


def save_to_database(records: list[dict], clear_existing: bool = True):
    """Save extracted records to AccidentRecord."""
    if clear_existing:
        deleted, _ = AccidentRecord.objects.filter(source="KTM_SCRAPED").delete()
        if deleted:
            log.info("Cleared %d existing scraped records", deleted)

    saved = 0
    for r in records:
        try:
            time_obj = None
            if r["time"]:
                try:
                    time_obj = datetime.strptime(r["time"], "%H:%M").time()
                except ValueError:
                    pass

            AccidentRecord.objects.create(
                latitude=r["latitude"],
                longitude=r["longitude"],
                date=r["date"] or datetime.now().date(),
                time=time_obj,
                day_of_week=r["day_of_week"] or "Unknown",
                weather_condition=r["weather_condition"],
                # These fields are left NULL/None rather than fake defaults.
                # Your analysis code should filter on is-not-null when needed.
                road_type=None,
                light_condition=None,
                speed_limit=None,
                severity=r["severity"],
                number_of_casualties=r["number_of_casualties"] or 1,
                number_of_deaths=r["number_of_deaths"] or 0,
                vehicle_type=r.get("vehicle_type", "Unknown"),
                source="KTM_SCRAPED",
                source_url=r["source_url"],
                location_name=r["location_name"],
                description=r["title"],
            )
            saved += 1
        except Exception as e:
            log.error("Error saving record for %s: %s", r["source_url"], e)

    log.info("Saved %d / %d records to database", saved, len(records))
    return saved


def run(dry_run: bool = False, max_pages: int = 15):
    session = requests.Session()

    # ── Step 1: Discover article URLs ──
    log.info("=== Phase 1: Discovering article URLs ===")
    urls = get_seed_urls()
    log.info("Seed URLs: %d", len(urls))

    crawled = discover_from_category_pages(session, max_pages_per_category=max_pages)
    urls.update(crawled)
    log.info("Total unique URLs after crawling: %d", len(urls))

    # ── Step 2: Fetch and extract ──
    log.info("=== Phase 2: Fetching and extracting articles ===")
    records = []
    stats = Counter()

    for i, url in enumerate(sorted(urls), 1):
        log.info("[%d/%d] %s", i, len(urls), url[:90])

        html = fetch_html(url, session)
        if not html:
            stats["fetch_failed"] += 1
            continue

        article = parse_article(html, url)
        if not article or not article["body"]:
            stats["parse_failed"] += 1
            continue

        record = process_article(article)
        if not record:
            stats["no_valley_location"] += 1
            continue

        records.append(record)
        stats["extracted"] += 1
        log.info(
            "  ✓ %s | %s | %s | %s",
            record["location_name"],
            record["severity"],
            record["day_of_week"] or "no date",
            record["time"] or "no time",
        )

        time.sleep(POLITE_DELAY)

    # ── Step 3: Report ──
    log.info("=== Results ===")
    log.info("Total URLs:            %d", len(urls))
    log.info("Fetch failures:        %d", stats["fetch_failed"])
    log.info("Parse failures:        %d", stats["parse_failed"])
    log.info("No Valley location:    %d", stats["no_valley_location"])
    log.info("Usable records:        %d", stats["extracted"])

    if records:
        log.info("\n--- Location distribution ---")
        for loc, count in Counter(r["location_name"] for r in records).most_common(25):
            log.info("  %-25s %d", loc, count)

        log.info("\n--- Severity distribution ---")
        for sev, count in Counter(r["severity"] for r in records).most_common():
            log.info("  %-10s %d", sev, count)

    # ── Step 4: Save ──
    if dry_run:
        log.info("DRY RUN — not saving to database")
        # Dump to JSON for inspection
        out_path = os.path.join(os.path.dirname(__file__), "scraped_preview.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        log.info("Preview written to %s", out_path)
    else:
        if records:
            save_to_database(records)
        else:
            log.warning("No records to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape KTM Valley accident data")
    parser.add_argument("--dry-run", action="store_true", help="Extract but don't save to DB")
    parser.add_argument("--max-pages", type=int, default=15, help="Max category pages to crawl")
    args = parser.parse_args()

    run(dry_run=args.dry_run, max_pages=args.max_pages)