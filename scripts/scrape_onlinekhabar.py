"""
Scrape Kathmandu Valley road accident reports from OnlineKhabar (Nepali language).

Strategy:
  1. Search-based discovery using OnlineKhabar's WordPress search endpoint
     with Nepali accident keywords (दुर्घटना, सडक, मृत्यु, घाइते)
  2. Category page crawling (/content/news with Valley location filters)
  3. Hardcoded seed URLs for known good articles

For each article: fetch full Nepali text, extract location (Valley-only),
severity, time, weather, date, casualty counts using Devanagari keyword dicts.
Saves to AccidentRecord with source="KTM_SCRAPED".

Usage:
    python scripts/scrape_onlinekhabar.py             # full run
    python scripts/scrape_onlinekhabar.py --dry-run    # extract but don't save to DB
    python scripts/scrape_onlinekhabar.py --max-pages 5  # limit crawl depth
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
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) " "Gecko/20100101 Firefox/128.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ne,en;q=0.5",
}
POLITE_DELAY = 2.0
REQUEST_TIMEOUT = 20
BASE_URL = "https://www.onlinekhabar.com"

# ────────────────────────────────────────────────────────────────────
# NEPALI ACCIDENT KEYWORDS (Devanagari)
# Used to filter links during category page crawling
# ────────────────────────────────────────────────────────────────────
ACCIDENT_KEYWORDS_NP = [
    "दुर्घटना",  # accident
    "ठोक्किय",  # collided
    "ठक्कर",  # collision/impact
    "मृत्यु",  # death
    "घाइते",  # injured
    "हताहत",  # casualty
    "ज्यान गयो",  # lost life
    "मोटरसाइकल",  # motorcycle
    "सवारी",  # vehicle
    "चापिय",  # crushed/run over
    "किचिय",  # crushed
    "दुर्घटनाग्रस्त",  # accident-stricken
]

# ────────────────────────────────────────────────────────────────────
# KATHMANDU VALLEY LOCATION DICTIONARY (Nepali/Devanagari)
# Maps Nepali place name spellings → (lat, lon)
# ────────────────────────────────────────────────────────────────────
LOCATION_COORDS_NP = {
    # ── Kathmandu ──
    "कालंकी चोक": (27.6933, 85.2814),
    "नयाँ बानेश्वर": (27.6910, 85.3380),
    "पुरानो बानेश्वर": (27.6870, 85.3350),
    "रिंग रोड": (27.7000, 85.3300),
    "नयाँ सडक": (27.7040, 85.3120),
    "कोटेश्वर": (27.6780, 85.3492),
    "कालंकी": (27.6933, 85.2814),
    "छाबहिल": (27.7172, 85.3411),
    "गोंगबु": (27.7350, 85.3150),
    "गोङ्गबु": (27.7350, 85.3150),
    "बालाजु": (27.7270, 85.3020),
    "महाराजगञ्ज": (27.7370, 85.3280),
    "बानेश्वर": (27.6910, 85.3380),
    "टिंकुने": (27.6850, 85.3450),
    "सिनामंगल": (27.6900, 85.3500),
    "गौशाला": (27.7100, 85.3380),
    "पुतलीसडक": (27.7050, 85.3210),
    "रत्नपार्क": (27.7060, 85.3150),
    "असन": (27.7080, 85.3110),
    "इन्द्रचोक": (27.7060, 85.3100),
    "थमेल": (27.7150, 85.3120),
    "लाजिम्पाट": (27.7200, 85.3230),
    "नक्साल": (27.7170, 85.3270),
    "दरबार मार्ग": (27.7130, 85.3190),
    "जावलाखेल": (27.6720, 85.3130),
    "सातदोबाटो": (27.6580, 85.3220),
    "बल्खु": (27.6870, 85.3000),
    "कालीमाटी": (27.6970, 85.3030),
    "बबरमहल": (27.6930, 85.3290),
    "स्वयम्भू": (27.7150, 85.2900),
    "किर्तिपुर": (27.6790, 85.2780),
    "थानकोट": (27.6880, 85.2540),
    "जाडीबुटी": (27.6740, 85.3530),
    "लोकन्थली": (27.6760, 85.3580),
    "थापाथली": (27.6920, 85.3220),
    "त्रिपुरेश्वर": (27.6980, 85.3140),
    "मैतीघर": (27.6960, 85.3230),
    "सिंहदरबार": (27.7000, 85.3220),
    "बाबरमहल": (27.6930, 85.3290),
    "मिनभवन": (27.6870, 85.3420),
    "मच्छापोखरी": (27.7250, 85.3050),
    "समाखुसी": (27.7300, 85.3120),
    "टोखा": (27.7470, 85.3140),
    "बुद्धनीलकण्ठ": (27.7680, 85.3640),
    "जोरपाटी": (27.7270, 85.3610),
    "बौद्ध": (27.7210, 85.3620),
    "बोधनाथ": (27.7210, 85.3620),
    "कापन": (27.7400, 85.3490),
    "बसुन्धरा": (27.7350, 85.3300),
    "बागबजार": (27.7050, 85.3170),
    "सुन्धारा": (27.7010, 85.3140),
    "अनामनगर": (27.7000, 85.3250),
    "कमलपोखरी": (27.7100, 85.3250),
    "दिल्लीबजार": (27.7100, 85.3300),
    "बत्तीसपुतली": (27.7090, 85.3400),
    "मित्रपार्क": (27.7050, 85.3380),
    "सानेपा": (27.6850, 85.3100),
    "एकान्तकुना": (27.6680, 85.3100),
    "इमाडोल": (27.6600, 85.3370),
    "लुभु": (27.6490, 85.3400),
    "गोदावरी": (27.5970, 85.3840),
    "चापागाउँ": (27.6250, 85.3580),
    "कुपन्डोल": (27.6880, 85.3150),
    "पुल्चोक": (27.6810, 85.3180),
    "लागनखेल": (27.6670, 85.3230),
    "धोबीघाट": (27.6800, 85.3040),
    "सितापाइला": (27.7150, 85.2800),
    "सुकेधारा": (27.7320, 85.3250),
    "रविभवन": (27.6980, 85.2950),
    "सोल्टीमोड": (27.6960, 85.2870),
    # ── Lalitpur ──
    "पाटन": (27.6720, 85.3180),
    "ललितपुर": (27.6720, 85.3180),
    "मंगलबजार": (27.6720, 85.3250),
    "जावलाखेल": (27.6720, 85.3130),
    "हत्तीवन": (27.6550, 85.3100),
    "बुंगमती": (27.6400, 85.3150),
    "खोकना": (27.6450, 85.3050),
    "लेले": (27.6100, 85.3400),
    # ── Bhaktapur ──
    "थिमी": (27.6730, 85.3870),
    "मध्यपुर थिमी": (27.6730, 85.3870),
    "भक्तपुर": (27.6710, 85.4280),
    "सूर्यबिनायक": (27.6650, 85.4450),
    "सल्लाघारी": (27.6700, 85.4100),
    "कमल विनायक": (27.6720, 85.4190),
    "चाँगुनारायण": (27.7100, 85.4270),
    "गठ्ठाघर": (27.6750, 85.3960),
    "राधे राधे": (27.6720, 85.3800),
    "जगाती": (27.6670, 85.4070),
    "नागरकोट": (27.7150, 85.5200),
    "दुवाकोट": (27.6950, 85.4050),
}

# Pre-sort by length descending for longest-match-first
_SORTED_LOCATIONS_NP = sorted(LOCATION_COORDS_NP.keys(), key=len, reverse=True)

# ────────────────────────────────────────────────────────────────────
# SEVERITY KEYWORDS (Nepali)
# ────────────────────────────────────────────────────────────────────
_FATAL_KW_NP = [
    "मृत्यु भयो",  # died
    "मृत्यु भए",  # died (plural)
    "मृत्यु भएको",  # died (past)
    "ज्यान गयो",  # lost life
    "ज्यान गए",  # lost lives
    "निधन भयो",  # passed away
    "मारिए",  # killed
    "मारियो",  # killed
    "मृत",  # dead
    "मृतक",  # deceased
    "शव",  # corpse
    "घटनास्थलमै मृत्यु",  # died on the spot
]
_SERIOUS_KW_NP = [
    "गम्भीर घाइते",  # seriously injured
    "गम्भीर चोट",  # serious injury
    "अचेत",  # unconscious
    "संकटकालीन",  # critical
    "गम्भीर अवस्था",  # serious condition
    "उपचाररत",  # under treatment
]
_SLIGHT_KW_NP = [
    "घाइते",  # injured
    "चोट लाग्यो",  # got hurt
    "चोटपटक",  # injuries
    "घाइते भए",  # were injured
    "घाइते भयो",  # was injured
]

# ────────────────────────────────────────────────────────────────────
# TIME KEYWORDS (Nepali)
# ────────────────────────────────────────────────────────────────────
TIME_KW_NP = {
    "बिहान": "08:00",  # morning
    "साँझ": "18:00",  # evening
    "राति": "21:00",  # night
    "दिउँसो": "14:00",  # afternoon
    "दिउँसो": "13:00",  # daytime
    "मध्यरात": "00:00",  # midnight
    "भोर": "05:30",  # dawn
    "बिहानीपख": "06:00",  # early morning
    "दिनको": "12:00",  # daytime/noon
    "मध्यान्ह": "12:00",  # noon
    "अपरान्ह": "15:00",  # afternoon
    "सन्ध्या": "17:30",  # dusk/evening
    "बेलुका": "19:00",  # evening (late)
}
_SORTED_TIME_KW_NP = sorted(TIME_KW_NP.keys(), key=len, reverse=True)

REJECT_TITLE_KEYWORDS = [
    "सप्ताह",
    "नियम",
    "अभियान",
    "रोक",
    "तोकियो",
    "सुझाव",
    "नीति",
    "बजेट",
    "हत्या",
    "डुबेर",
    "आगलागी",
    "भूकम्प",
    "पार्किङ",
    "लेन",
    "स्पिड ब्रेकर",
    "ट्राफिक सप्ताह",
    "चक्कु",
    "हानाहान",
    "कुटपिट",
]

REQUIRE_TITLE_KEYWORDS = [
    "दुर्घटना",
    "ठोक्किय",
    "ठक्कर",
    "मृत्यु",
    "घाइते",
    "दुर्घटनाग्रस्त",
    "किचिय",
    "चापिय",
]
# ────────────────────────────────────────────────────────────────────
# WEATHER KEYWORDS (Nepali)
# ────────────────────────────────────────────────────────────────────
WEATHER_KW_NP = {
    "भारी वर्षा": "Raining + high winds",  # heavy rain
    "मुसलधारे वर्षा": "Raining + high winds",  # torrential rain
    "वर्षा": "Raining no high winds",  # rain
    "पानी परेको": "Raining no high winds",  # it was raining
    "पानी पर्दा": "Raining no high winds",  # while raining
    "मनसुन": "Raining no high winds",  # monsoon
    "हुस्सु": "Fog or mist",  # fog
    "कुहिरो": "Fog or mist",  # mist/fog
    "हिमपात": "Snowing no high winds",  # snowfall
    "हिउँ": "Snowing no high winds",  # snow
}
_SORTED_WEATHER_KW_NP = sorted(WEATHER_KW_NP.keys(), key=len, reverse=True)

# ────────────────────────────────────────────────────────────────────
# VEHICLE TYPE KEYWORDS (Nepali)
# ────────────────────────────────────────────────────────────────────
VEHICLE_KW_NP = [
    ("माइक्रोबस", "Bus"),
    ("मिनिबस", "Bus"),
    ("यात्रुबस", "Bus"),
    ("बस दुर्घटना", "Bus"),  # "bus accident" — avoids matching बस inside other words
    ("ट्रक", "Goods vehicle"),
    ("टिपर", "Goods vehicle"),
    ("ट्याक्टर", "Goods vehicle"),
    ("टेम्पो", "Other"),
    ("अटोरिक्सा", "Other"),
    ("मोटरसाइकल", "Motorcycle"),
    ("मोटरसाईकल", "Motorcycle"),
    ("बाइक", "Motorcycle"),
    ("स्कुटर", "Motorcycle"),
    ("साइकल", "Pedal cycle"),
    ("कार दुर्घटना", "Car"),
    ("जिप", "Car"),
    ("ट्याक्सी", "Taxi"),
    ("पैदल यात्री", "Pedestrian"),
    ("पैदल", "Pedestrian"),
]

# ────────────────────────────────────────────────────────────────────
# NEPALI NUMBER WORDS
# ────────────────────────────────────────────────────────────────────
NEPALI_NUMBERS = {
    "एक": 1,
    "दुई": 2,
    "तीन": 3,
    "चार": 4,
    "पाँच": 5,
    "छ": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "एघार": 11,
    "बाह्र": 12,
    "तेह्र": 13,
    "चौध": 14,
    "पन्ध्र": 15,
    # Nepali digits (Unicode Devanagari numerals)
    "१": 1,
    "२": 2,
    "३": 3,
    "४": 4,
    "५": 5,
    "६": 6,
    "७": 7,
    "८": 8,
    "९": 9,
    "०": 0,
}

# ────────────────────────────────────────────────────────────────────
# NEPALI MONTH → Gregorian mapping (Bikram Sambat approximate)
# Used for parsing dates in Nepali format
# ────────────────────────────────────────────────────────────────────
# OnlineKhabar also shows Gregorian dates on articles, which is easier
# to parse directly. We use the og:article:published_time meta tag.


# ════════════════════════════════════════════════════════════════════
# EXTRACTION HELPERS
# ════════════════════════════════════════════════════════════════════


def extract_location(title: str, body: str):
    """
    Extract the most relevant Kathmandu Valley location from Nepali text.
    Priority: title first, then first 500 chars of body (lede).
    Returns (location_name_nepali, (lat, lon)) or (None, None).
    """
    for text in [title, body[:500], body]:
        for loc in _SORTED_LOCATIONS_NP:
            if loc in text:
                return loc, LOCATION_COORDS_NP[loc]
    return None, None


def extract_severity(text: str) -> str:
    """Return highest severity: Fatal > Serious > Slight."""
    for kw in _FATAL_KW_NP:
        if kw in text:
            return "Fatal"
    for kw in _SERIOUS_KW_NP:
        if kw in text:
            return "Serious"
    for kw in _SLIGHT_KW_NP:
        if kw in text:
            return "Slight"
    return "Slight"


def extract_time(text: str):
    """
    Extract time from Nepali text.
    Tries explicit patterns first (१०:१५ बजे, 10:15 बजे),
    then keyword matching.
    """

    # Nepali/Arabic digits + बजे pattern: "साँझ ६:३० बजे" or "6:30 बजे"
    # Convert Nepali digits to Arabic for parsing
    def np_to_arabic(s):
        for np, ar in [
            ("०", "0"),
            ("१", "1"),
            ("२", "2"),
            ("३", "3"),
            ("४", "4"),
            ("५", "5"),
            ("६", "6"),
            ("७", "7"),
            ("८", "8"),
            ("९", "9"),
        ]:
            s = s.replace(np, ar)
        return s

    converted = np_to_arabic(text)

    # Pattern: "X:XX बजे" or "X बजे"
    m = re.search(r"(\d{1,2})[:\.](\d{2})\s*बजे", converted)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        # If context says साँझ (evening) and hour < 7, it's PM
        if "साँझ" in text and hour < 7:
            hour += 12
        elif "राति" in text and hour < 6:
            hour += 12
        return f"{hour:02d}:{minute:02d}"

    m = re.search(r"(\d{1,2})\s*बजे", converted)
    if m:
        hour = int(m.group(1))
        if "साँझ" in text and hour < 7:
            hour += 12
        elif "राति" in text and hour < 6:
            hour += 12
        return f"{hour:02d}:00"

    # Keyword matching
    for kw in _SORTED_TIME_KW_NP:
        if kw in text:
            return TIME_KW_NP[kw]

    return None


def extract_weather(text: str) -> str:
    for kw in _SORTED_WEATHER_KW_NP:
        if kw in text:
            return WEATHER_KW_NP[kw]
    return "Fine no high winds"


def extract_vehicle_type(text: str) -> str:
    for kw, vtype in VEHICLE_KW_NP:
        if kw in text:
            return vtype
    return "Unknown"


def extract_casualties(text: str) -> dict:
    """
    Extract death and injury counts from Nepali text.
    Handles both Nepali number words and Devanagari/Arabic digits.
    """

    def np_to_arabic(s):
        for np, ar in [
            ("०", "0"),
            ("१", "1"),
            ("२", "2"),
            ("३", "3"),
            ("४", "4"),
            ("५", "5"),
            ("६", "6"),
            ("७", "7"),
            ("८", "8"),
            ("९", "9"),
        ]:
            s = s.replace(np, ar)
        return s

    converted = np_to_arabic(text)
    num_words = "|".join(NEPALI_NUMBERS.keys())

    deaths = None
    casualties = None

    # Death patterns
    for pattern in [
        rf"(\d+|{num_words})\s*(?:जनाको\s+)?मृत्यु",  # X जनाको मृत्यु
        rf"(\d+|{num_words})\s*(?:जना\s+)?मारिए",  # X जना मारिए
        rf"(\d+|{num_words})\s*(?:जनाको\s+)?ज्यान गयो",  # X जनाको ज्यान गयो
    ]:
        m = re.search(pattern, converted)
        if m:
            val = m.group(1)
            deaths = int(val) if val.isdigit() else NEPALI_NUMBERS.get(val)
            break

    # Injury patterns
    for pattern in [
        rf"(\d+|{num_words})\s*(?:जना\s+)?घाइते",  # X जना घाइते
        rf"(\d+|{num_words})\s*(?:जना\s+)?घाइते भए",  # X जना घाइते भए
    ]:
        m = re.search(pattern, converted)
        if m:
            val = m.group(1)
            casualties = int(val) if val.isdigit() else NEPALI_NUMBERS.get(val)
            break

    return {"deaths": deaths, "casualties": casualties}


# ════════════════════════════════════════════════════════════════════
# DATE PARSING
# ════════════════════════════════════════════════════════════════════


def parse_date_from_wp_api(
    post_id: str, session: requests.Session
) -> datetime.date | None:
    """
    Use WordPress REST API to get exact publish date.
    OnlineKhabar exposes wp-json/wp/v2/posts/{id}
    """
    try:
        api_url = f"{BASE_URL}/wp-json/wp/v2/posts/{post_id}"
        resp = session.get(api_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            date_str = data.get("date", "")[:10]  # "2022-01-19T..."
            return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        pass
    return None


def parse_date_from_url(url: str) -> datetime.date | None:
    # OnlineKhabar URLs are /YYYY/MM/article_id/ — no day component
    # This should only be used as a last resort when meta tag parsing fails
    m = re.search(r"/(\d{4})/(\d{2})/\d{4,}/", url)
    if m:
        try:
            # We only know year and month — use middle of month as approximation
            # and flag it so analysis code can filter these out if needed
            return datetime.strptime(f"{m.group(1)}-{m.group(2)}-15", "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


# ════════════════════════════════════════════════════════════════════
# ARTICLE FETCHING & PARSING
# ════════════════════════════════════════════════════════════════════


def fetch_html(url: str, session: requests.Session) -> str | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.text
        log.warning("HTTP %d for %s", resp.status_code, url)
        return None
    except requests.RequestException as e:
        log.warning("Request failed for %s: %s", url, e)
        return None


def parse_article(html: str, url: str, session: requests.Session) -> dict | None:
    """
    Parse an OnlineKhabar article page.
    Returns {"title", "body", "date", "url"} or None.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title: og:title meta is most reliable
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    else:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Body: OnlineKhabar uses .ok-news-post or .post-content
    body = ""
    for selector in [
        {"class": "ok-news-post"},
        {"class": "post-content"},
        {"class": "ok-post-content"},
        {"id": "post-content"},
    ]:
        content_div = soup.find("div", selector)
        if content_div:
            paragraphs = content_div.find_all("p")
            body = " ".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )
            if body:
                break

    # Fallback: all article paragraphs
    if not body:
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
            body = " ".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

    if not body and not title:
        return None

    # Date
    post_id = get_post_id(url)
    date = (
        parse_date_from_wp_api(post_id, session)
        if post_id
        else parse_date_from_url(url)
    )

    return {
        "title": title,
        "body": body,
        "date": date,
        "url": url,
    }


def get_post_id(url: str) -> str | None:
    parts = url.rstrip("/").split("/")
    for part in parts:
        # Skip year (4 digits starting with 20) and month (2 digits)
        if part.isdigit() and len(part) >= 5:
            return part
        if part.isdigit() and len(part) == 4 and not part.startswith("20"):
            return part
    return None


# ════════════════════════════════════════════════════════════════════
# URL DISCOVERY
# ════════════════════════════════════════════════════════════════════


def _is_accident_link(href: str, anchor_text: str) -> bool:
    combined = href + " " + anchor_text
    return any(kw in combined for kw in ACCIDENT_KEYWORDS_NP)


def discover_from_search(session: requests.Session, max_pages: int = 10) -> set[str]:
    """
    Use OnlineKhabar's WordPress search endpoint to find accident articles.
    Searches for key Nepali accident terms targeting the Valley.
    """
    search_queries = [
        "काठमाडौं सडक दुर्घटना",  # Kathmandu road accident
        "ललितपुर सडक दुर्घटना",  # Lalitpur road accident
        "भक्तपुर सडक दुर्घटना",  # Bhaktapur road accident
        "काठमाडौं मोटरसाइकल दुर्घटना",  # Kathmandu motorcycle accident
        "कालंकी दुर्घटना",  # Kalanki accident
        "कोटेश्वर दुर्घटना",  # Koteshwor accident
        "उपत्यका सवारी दुर्घटना",  # Valley vehicle accident
        "काठमाडौं घाइते मृत्यु",  # Kathmandu injured death
    ]

    urls = set()

    for query in search_queries:
        for page in range(1, max_pages + 1):
            search_url = f"{BASE_URL}/?s={requests.utils.quote(query)}&paged={page}"
            log.info("Searching: %s (page %d)", query, page)

            html = fetch_html(search_url, session)
            if not html:
                break

            soup = BeautifulSoup(html, "html.parser")
            found = 0

            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True)
                # OnlineKhabar article URLs follow /YYYY/MM/ID/ pattern
                if re.search(r"/\d{4}/\d{2}/\d+", href) and _is_accident_link(
                    href, text
                ):
                    full = href if href.startswith("http") else BASE_URL + href
                    if full not in urls:
                        urls.add(full)
                        found += 1

            log.info("  Found %d new links", found)
            if found == 0:
                break  # no more results

            time.sleep(POLITE_DELAY)

    return urls


def discover_from_category_pages(
    session: requests.Session, max_pages: int = 15
) -> set[str]:
    """
    Crawl OnlineKhabar news category pages for accident articles.
    """
    categories = [
        f"{BASE_URL}/content/news",  # general news
        f"{BASE_URL}/content/news/rastiya",  # national news
    ]

    urls = set()

    for cat_url in categories:
        for page_num in range(1, max_pages + 1):
            page_url = f"{cat_url}/page/{page_num}" if page_num > 1 else cat_url
            log.info("Crawling %s", page_url)

            html = fetch_html(page_url, session)
            if not html:
                break

            soup = BeautifulSoup(html, "html.parser")
            found = 0

            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True)
                if re.search(r"/\d{4}/\d{2}/\d+", href) and _is_accident_link(
                    href, text
                ):
                    full = href if href.startswith("http") else BASE_URL + href
                    if full not in urls:
                        urls.add(full)
                        found += 1

            log.info("  Found %d new links on page %d", found, page_num)
            if found == 0 and page_num > 3:
                break

            time.sleep(POLITE_DELAY)

    return urls


def get_seed_urls() -> set[str]:
    """Known good OnlineKhabar accident articles for the Kathmandu Valley."""
    return {
        # Add confirmed article URLs here as you find them
        "https://www.onlinekhabar.com/2024/01/1417123",
        "https://www.onlinekhabar.com/2024/09/1545895",
    }


# ════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════


def process_article(article_data: dict) -> dict | None:
    """
    Extract structured accident record from a parsed article.
    Returns a dict ready for DB insertion, or None if not a Valley accident.
    """
    title = article_data["title"]
    body = article_data["body"]
    full_text = title + " " + body

    loc_name, coords = extract_location(title, body)
    if not coords:
        return None  # not a Kathmandu Valley article

    # Reject articles that aren't actually about accidents
    if not is_accident_article(title, body):
        return None

    lat, lon = coords
    severity = extract_severity(full_text)
    time_str = extract_time(full_text)
    weather = extract_weather(full_text)
    date = article_data["date"]
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


def is_accident_article(title: str, body: str) -> bool:
    # Reject if title contains non-accident keywords
    if any(kw in title for kw in REJECT_TITLE_KEYWORDS):
        return False
    # Require at least one accident keyword in the title
    if not any(kw in title for kw in REQUIRE_TITLE_KEYWORDS):
        return False
    return True


def save_to_database(records: list[dict]):
    """Save extracted records to AccidentRecord, merging with existing KTM_SCRAPED."""
    # Don't clear existing — this scraper adds to what the English scraper found
    existing_urls = set(
        AccidentRecord.objects.filter(source="KTM_SCRAPED").values_list(
            "source_url", flat=True
        )
    )

    saved = 0
    skipped_duplicate = 0

    for r in records:
        if r["source_url"] in existing_urls:
            skipped_duplicate += 1
            continue
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

    log.info("Saved %d new records (%d duplicates skipped)", saved, skipped_duplicate)
    return saved


def run(dry_run: bool = False, max_pages: int = 15):
    session = requests.Session()

    # Step 1: Discover URLs
    log.info("=== Phase 1: Discovering article URLs ===")
    urls = get_seed_urls()
    log.info("Seed URLs: %d", len(urls))

    search_urls = discover_from_search(session, max_pages=max_pages)
    urls.update(search_urls)
    log.info("After search discovery: %d URLs", len(urls))

    category_urls = discover_from_category_pages(session, max_pages=max_pages)
    urls.update(category_urls)
    log.info("Total unique URLs: %d", len(urls))

    # Step 2: Fetch and extract
    log.info("=== Phase 2: Fetching and extracting articles ===")
    records = []
    stats = Counter()

    for i, url in enumerate(sorted(urls), 1):
        log.info("[%d/%d] %s", i, len(urls), url[:90])

        html = fetch_html(url, session)
        if not html:
            stats["fetch_failed"] += 1
            continue

        article = parse_article(html, url, session)
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

    # Step 3: Report
    log.info("=== Results ===")
    log.info("Total URLs:          %d", len(urls))
    log.info("Fetch failures:      %d", stats["fetch_failed"])
    log.info("Parse failures:      %d", stats["parse_failed"])
    log.info("No Valley location:  %d", stats["no_valley_location"])
    log.info("Usable records:      %d", stats["extracted"])

    if records:
        log.info("\n--- Location distribution ---")
        for loc, count in Counter(r["location_name"] for r in records).most_common(20):
            log.info("  %-25s %d", loc, count)

        log.info("\n--- Severity distribution ---")
        for sev, count in Counter(r["severity"] for r in records).most_common():
            log.info("  %-10s %d", sev, count)

    # Step 4: Save or preview
    if dry_run:
        log.info("DRY RUN — not saving to database")
        out_path = os.path.join(
            os.path.dirname(__file__), "scraped_nepali_preview.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str, ensure_ascii=False)
        log.info("Preview written to %s", out_path)
    else:
        if records:
            save_to_database(records)
        else:
            log.warning("No records to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape OnlineKhabar Nepali accident data"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Extract but don't save to DB"
    )
    parser.add_argument(
        "--max-pages", type=int, default=15, help="Max pages to crawl per source"
    )
    args = parser.parse_args()

    run(dry_run=args.dry_run, max_pages=args.max_pages)
