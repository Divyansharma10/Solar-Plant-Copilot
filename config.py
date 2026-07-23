# config.py
"""
Solar Plant Copilot — Central Configuration
All constants, zone definitions, thresholds, and model settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ───────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Zone Definitions ───────────────────────────────────────────
BASE_LAT = 26.9124
BASE_LON = 75.7873

ZONES = {
    "Zone_A": {"lat": BASE_LAT + 0.005, "lon": BASE_LON + 0.005},
    "Zone_B": {"lat": BASE_LAT + 0.005, "lon": BASE_LON - 0.005},
    "Zone_C": {"lat": BASE_LAT - 0.005, "lon": BASE_LON + 0.005},
    "Zone_D": {"lat": BASE_LAT - 0.005, "lon": BASE_LON - 0.005},
}

# ── Data Settings ──────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "sunshine_duration",
]

TIMEZONE = "Asia/Kolkata"
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ── Panel & Plant Settings ─────────────────────────────────────
TILT = 24                        # panel tilt angle degrees
AZIMUTH = 0                      # 0 = south facing
ZONE_CAPACITY_KW = 5000          # 5 MW per zone
MODULE_EFFICIENCY = 0.19         # 19% standard c-Si
PDC0 = ZONE_CAPACITY_KW * 1000   # watts at STC
GAMMA_PDC = -0.004               # temp coefficient per °C
INVERTER_EFFICIENCY = 0.97
SYSTEM_LOSSES = 0.14
U0 = 25.0                        # Faiman model constant
U1 = 6.84                        # Faiman model wind coefficient

# ── Grid & Operations Thresholds ──────────────────────────────
GRID_CONTRACT_MIN_KW = 1500
BATTERY_RESERVE_THRESHOLD_KW = 1500
BATTERY_RESERVE_DURATION_HOURS = 2
NORMAL_PR_MIN = 0.75
NORMAL_PR_MAX = 0.85

# ── Sliding Window Settings ────────────────────────────────────
WINDOW_SIZE_HOURS = 7 * 24       # 7 days
STEP_SIZE_HOURS = 24             # slide by 1 day

# ── Model Settings ─────────────────────────────────────────────
EDGE_LLM_MODEL = "phi4-mini"
CLOUD_LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLOUD_LLM_TEMPERATURE = 0.3
EDGE_LLM_MAX_SENTENCES = 3

# ── RAG Settings ───────────────────────────────────────────────
BM25_WEIGHT = 0.3
FAISS_WEIGHT = 0.7
RAG_TOP_K = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Performance Thresholds ─────────────────────────────────────
PERFORMANCE_NORMAL = 95          # above this = Normal
PERFORMANCE_MONITOR = 85         # above this = Monitor
PERFORMANCE_WARNING = 75         # above this = Warning
                                 # below 75 = Critical

# ── Season Mapping ─────────────────────────────────────────────
SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring/Peak", 4: "Spring/Peak",
    5: "Spring/Peak", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
    9: "Monsoon", 10: "Autumn", 11: "Winter", 12: "Winter"
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# ── File Paths ─────────────────────────────────────────────────
DATA_DIR = "./data"
COMBINED_CSV = f"{DATA_DIR}/all_zones_weather_combined.csv"
ZENITH_CSV = f"{DATA_DIR}/all_zones_weather_with_zenith.csv"
DAYLIGHT_CSV = f"{DATA_DIR}/all_zones_daylight_only.csv"
OUTPUT_CSV = f"{DATA_DIR}/all_zones_with_output.csv"
WINDOW_SUMMARIES_CSV = f"{DATA_DIR}/zone_a_window_summaries.csv"
FAISS_INDEX_PATH = f"{DATA_DIR}/faiss_index"
ALL_ZONES_WINDOW_SUMMARIES_CSV = f"{DATA_DIR}/all_zones_window_summaries.csv"
