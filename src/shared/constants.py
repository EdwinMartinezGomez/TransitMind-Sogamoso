"""
TransitMind Sogamoso — Global Constants
========================================
Sogamoso-specific constants used across all layers.
"""

from typing import Dict, List, Tuple

# ============================================
# Intersection Definitions
# ============================================

INTERSECTIONS: List[str] = [
    "carrera_11_norte",
    "carrera_11_sur",
    "av_castellana_entrada",
    "av_castellana_salida",
    "calle_14_centro_historico",
    "acceso_morca",
]

# Base vehicle flow per intersection (veh/15min during normal hours)
INTERSECTION_BASE_FLOW: Dict[str, float] = {
    "carrera_11_norte": 85.0,
    "carrera_11_sur": 78.0,
    "av_castellana_entrada": 95.0,
    "av_castellana_salida": 90.0,
    "calle_14_centro_historico": 65.0,
    "acceso_morca": 45.0,
}

# Intersections near Coliseo Olímpico (affected by events)
COLISEO_NEARBY_INTERSECTIONS: List[str] = [
    "carrera_11_norte",
    "carrera_11_sur",
    "calle_14_centro_historico",
]

# ============================================
# Peak Hour Definitions
# ============================================

# (start_hour, start_min, end_hour, end_min, flow_multiplier)
PEAK_HOURS: List[Tuple[int, int, int, int, float]] = [
    (6, 30, 8, 30, 1.60),    # Morning peak: +60%
    (12, 0, 13, 30, 1.40),   # Midday peak: +40%
    (17, 30, 19, 30, 1.70),  # Evening peak: +70%
]

# ============================================
# Day-of-Week Modifiers
# ============================================

# Flow multiplier by day of week (0=Monday)
DAY_MULTIPLIERS: Dict[int, float] = {
    0: 1.00,  # Lunes
    1: 1.02,  # Martes
    2: 1.05,  # Miércoles (mercado campesino)
    3: 1.00,  # Jueves
    4: 1.08,  # Viernes (pre-fin de semana)
    5: 0.75,  # Sábado (mercado pero menor tráfico base)
    6: 0.65,  # Domingo
}

# Market days (0=Monday): Wednesday and Saturday
MARKET_DAYS: List[int] = [2, 5]

# Market day flow multiplier (additional to base)
MARKET_DAY_MULTIPLIER: float = 1.45

# ============================================
# Weather Constants
# ============================================

WEATHER_CODES: Dict[int, str] = {
    0: "despejado",
    1: "nublado",
    2: "lluvia",
    3: "niebla",
}

# Morning fog probability on weekdays (Chicamocha valley effect)
FOG_PROBABILITY_WEEKDAY: float = 0.40

# Speed reduction by weather condition
WEATHER_SPEED_MODIFIER: Dict[int, float] = {
    0: 1.00,   # Clear
    1: 0.95,   # Cloudy: -5%
    2: 0.80,   # Rain: -20%
    3: 0.80,   # Fog: -20%
}

# ============================================
# Event Constants
# ============================================

# Coliseo event probability per day
COLISEO_EVENT_PROBABILITY: float = 0.10

# Flow multiplier for nearby intersections during Coliseo events
COLISEO_EVENT_MULTIPLIER: float = 1.80

# ============================================
# Vehicle Composition (Sogamoso typical)
# ============================================

# Base heavy vehicle ratio (buses + trucks)
BASE_HEAVY_VEHICLE_RATIO: float = 0.15

# Base motorcycle ratio (common in Colombian cities)
BASE_MOTORCYCLE_RATIO: float = 0.35

# Base average speed (km/h) — urban area
BASE_AVG_SPEED: float = 35.0

# ============================================
# TimeGAN Defaults
# ============================================

DEFAULT_SEQ_LEN: int = 24
DEFAULT_PREDICTION_HORIZON: int = 4
DEFAULT_SEED: int = 42
WINDOWS_PER_DAY: int = 96  # 24 hours × 4 (15-min windows)

# ============================================
# Feature Columns (numeric only, for TimeGAN input)
# ============================================

NUMERIC_FEATURES: List[str] = [
    "hour",
    "day_of_week",
    "is_peak_hour",
    "vehicle_flow",
    "heavy_vehicle_ratio",
    "motorcycle_ratio",
    "avg_speed_kmh",
    "congestion_level",
    "weather_code",
    "event_impact",
    "is_market_day",
]

# All columns including non-numeric
ALL_COLUMNS: List[str] = [
    "timestamp",
    "hour",
    "day_of_week",
    "is_peak_hour",
    "vehicle_flow",
    "heavy_vehicle_ratio",
    "motorcycle_ratio",
    "avg_speed_kmh",
    "congestion_level",
    "weather_code",
    "event_impact",
    "is_market_day",
    "intersection_id",
]

# Target variable for TSTR evaluation
TARGET_VARIABLE: str = "congestion_level"

# ============================================
# Sogamoso City Info
# ============================================

CITY_NAME: str = "Sogamoso"
CITY_DEPARTMENT: str = "Boyacá"
CITY_COUNTRY: str = "Colombia"
CITY_POPULATION: int = 120_000
CITY_ALTITUDE_M: int = 2_569  # meters above sea level
