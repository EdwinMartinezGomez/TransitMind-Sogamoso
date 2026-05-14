"""
TransitMind Sogamoso — Data Loader (Fase 0)
=============================================
Generates synthetic seed data for TimeGAN training, simulating realistic
traffic patterns for Sogamoso, Colombia.

Functions:
    - generate_seed_data(): Creates 90 days of plausible traffic data
    - normalize_data(): MinMaxScaler normalization
    - create_sequences(): Converts to windowed numpy arrays
    - split_dataset(): Train/validation split
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.shared.constants import (
    ALL_COLUMNS,
    BASE_AVG_SPEED,
    BASE_HEAVY_VEHICLE_RATIO,
    BASE_MOTORCYCLE_RATIO,
    COLISEO_EVENT_MULTIPLIER,
    COLISEO_EVENT_PROBABILITY,
    COLISEO_NEARBY_INTERSECTIONS,
    DAY_MULTIPLIERS,
    DEFAULT_SEQ_LEN,
    FOG_PROBABILITY_WEEKDAY,
    INTERSECTION_BASE_FLOW,
    INTERSECTIONS,
    MARKET_DAY_MULTIPLIER,
    MARKET_DAYS,
    NUMERIC_FEATURES,
    PEAK_HOURS,
    WEATHER_SPEED_MODIFIER,
    WINDOWS_PER_DAY,
)
from src.shared.logger import setup_logger
from src.shared.utils import ensure_dir, resolve_path

logger = setup_logger("data_loader")


def generate_seed_data(
    n_days: int = 90,
    intersections: Optional[List[str]] = None,
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Generate synthetic seed data simulating realistic traffic patterns
    for Sogamoso, Colombia.

    Simulates:
        - Peak hours: morning (6:30-8:30, +60%), midday (12:00-13:30, +40%),
          evening (17:30-19:30, +70%)
        - Weekend reduction: -30% flow vs weekdays
        - Market days (Wed/Sat): +45% flow
        - Morning fog (40% weekday probability): -20% speed
        - Coliseo Olímpico events: +80% on nearby intersections

    Args:
        n_days: Number of days to generate (default: 90).
        intersections: List of intersection IDs. Defaults to all 6 pilot intersections.
        seed: Random seed for reproducibility.
        start_date: Start date string in ISO format.

    Returns:
        DataFrame with columns matching data_schema.yaml.
        Shape: (n_days × 96 windows × n_intersections, 13 columns)
    """
    np.random.seed(seed)

    if intersections is None:
        intersections = INTERSECTIONS

    start = datetime.strptime(start_date, "%Y-%m-%d")
    records: List[Dict] = []

    logger.info(
        "generating_seed_data",
        n_days=n_days,
        n_intersections=len(intersections),
        start_date=start_date,
    )

    for day_offset in range(n_days):
        current_date = start + timedelta(days=day_offset)
        day_of_week = current_date.weekday()  # 0=Monday
        is_market = day_of_week in MARKET_DAYS
        is_weekend = day_of_week >= 5

        # Determine if there's a Coliseo event today (random)
        has_coliseo_event = np.random.random() < COLISEO_EVENT_PROBABILITY
        # Events typically happen in the evening (17:00-22:00)
        event_start_hour = 17 if has_coliseo_event else -1

        # Determine weather for the day
        day_weather = _generate_daily_weather(day_of_week)

        for window in range(WINDOWS_PER_DAY):
            hour = window // 4
            minute = (window % 4) * 15
            timestamp = current_date.replace(hour=hour, minute=minute, second=0)

            # Check peak hour status
            is_peak, peak_multiplier = _check_peak_hour(hour, minute)

            # Current weather (fog is morning-only: 5am-9am)
            if day_weather == 3 and not (5 <= hour <= 9):
                current_weather = np.random.choice([0, 1], p=[0.6, 0.4])
            else:
                current_weather = day_weather

            # Event impact
            event_impact = 0.0
            if has_coliseo_event and event_start_hour <= hour <= 22:
                event_impact = 0.9 * np.clip(
                    np.random.normal(1.0, 0.1), 0.7, 1.0
                )

            for intersection in intersections:
                record = _generate_record(
                    timestamp=timestamp,
                    hour=hour,
                    day_of_week=day_of_week,
                    is_peak=is_peak,
                    peak_multiplier=peak_multiplier,
                    is_market=is_market,
                    is_weekend=is_weekend,
                    weather_code=current_weather,
                    event_impact=event_impact,
                    has_coliseo_event=has_coliseo_event,
                    intersection_id=intersection,
                )
                records.append(record)

    df = pd.DataFrame(records)

    # Ensure column order matches schema
    df = df[ALL_COLUMNS]

    # Log statistics
    _log_statistics(df)

    # Save to disk
    _save_seed_data(df)

    logger.info(
        "seed_data_generated",
        total_rows=len(df),
        columns=list(df.columns),
        date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}",
    )

    return df


def _generate_daily_weather(day_of_week: int) -> int:
    """
    Generate weather condition for a day based on Sogamoso climate patterns.

    Sogamoso is in the Chicamocha valley — morning fog is common on weekdays.

    Args:
        day_of_week: Day of week (0=Monday).

    Returns:
        Weather code: 0=clear, 1=cloudy, 2=rain, 3=fog.
    """
    is_weekday = day_of_week < 5

    if is_weekday and np.random.random() < FOG_PROBABILITY_WEEKDAY:
        return 3  # Morning fog
    else:
        # General weather distribution for Sogamoso
        return int(np.random.choice(
            [0, 1, 2],
            p=[0.45, 0.35, 0.20],
        ))


def _check_peak_hour(hour: int, minute: int) -> Tuple[bool, float]:
    """
    Check if the given time is within a peak hour window.

    Args:
        hour: Hour of day (0-23).
        minute: Minute of hour (0-59).

    Returns:
        Tuple of (is_peak, multiplier). Multiplier is 1.0 if not peak.
    """
    current_minutes = hour * 60 + minute

    for start_h, start_m, end_h, end_m, multiplier in PEAK_HOURS:
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if start_minutes <= current_minutes <= end_minutes:
            return True, multiplier

    return False, 1.0


def _generate_record(
    timestamp: datetime,
    hour: int,
    day_of_week: int,
    is_peak: bool,
    peak_multiplier: float,
    is_market: bool,
    is_weekend: bool,
    weather_code: int,
    event_impact: float,
    has_coliseo_event: bool,
    intersection_id: str,
) -> Dict:
    """
    Generate a single traffic observation record with realistic patterns.

    Args:
        timestamp: Observation timestamp.
        hour: Hour of day.
        day_of_week: Day of week (0=Monday).
        is_peak: Whether this is a peak hour.
        peak_multiplier: Flow multiplier for peak hours.
        is_market: Whether today is a market day.
        is_weekend: Whether today is a weekend.
        weather_code: Weather condition code.
        event_impact: Event impact level (0-1).
        has_coliseo_event: Whether there's a Coliseo event today.
        intersection_id: Intersection identifier.

    Returns:
        Dictionary with all traffic variables for one observation.
    """
    base_flow = INTERSECTION_BASE_FLOW[intersection_id]

    # ---- Vehicle Flow Calculation ----
    flow = base_flow

    # Time-of-day base pattern (sinusoidal + peak overlay)
    time_factor = 0.3 + 0.7 * np.sin(np.pi * max(0, hour - 5) / 17) if 5 <= hour <= 22 else 0.15
    flow *= time_factor

    # Peak hour multiplier
    flow *= peak_multiplier

    # Day-of-week modifier
    flow *= DAY_MULTIPLIERS.get(day_of_week, 1.0)

    # Market day boost
    if is_market:
        # Market effect is strongest 7am-2pm
        if 7 <= hour <= 14:
            flow *= MARKET_DAY_MULTIPLIER
        elif 5 <= hour <= 7 or 14 <= hour <= 16:
            flow *= 1.0 + (MARKET_DAY_MULTIPLIER - 1.0) * 0.5

    # Coliseo event boost (nearby intersections only)
    if has_coliseo_event and event_impact > 0:
        if intersection_id in COLISEO_NEARBY_INTERSECTIONS:
            flow *= 1.0 + (COLISEO_EVENT_MULTIPLIER - 1.0) * event_impact
        else:
            flow *= 1.0 + (COLISEO_EVENT_MULTIPLIER - 1.0) * event_impact * 0.3

    # Add random noise (±15%)
    flow *= np.random.uniform(0.85, 1.15)
    flow = max(0.0, min(300.0, flow))

    # ---- Average Speed ----
    speed = BASE_AVG_SPEED

    # Weather effect on speed
    speed *= WEATHER_SPEED_MODIFIER.get(weather_code, 1.0)

    # Congestion reduces speed
    flow_capacity_ratio = flow / 200.0  # rough capacity estimate
    if flow_capacity_ratio > 0.7:
        speed *= max(0.3, 1.0 - (flow_capacity_ratio - 0.7) * 1.5)

    # Peak hour speed reduction
    if is_peak:
        speed *= 0.85

    speed *= np.random.uniform(0.9, 1.1)
    speed = max(5.0, min(80.0, speed))

    # ---- Congestion Level ----
    congestion = np.clip(flow_capacity_ratio * 0.8 + (1.0 - speed / BASE_AVG_SPEED) * 0.4, 0.0, 1.0)
    congestion += np.random.uniform(-0.05, 0.05)
    congestion = np.clip(congestion, 0.0, 1.0)

    # ---- Vehicle Composition ----
    heavy_ratio = BASE_HEAVY_VEHICLE_RATIO
    # More heavy vehicles during morning (delivery) and market days
    if 6 <= hour <= 10:
        heavy_ratio *= 1.3
    if is_market:
        heavy_ratio *= 1.4
    heavy_ratio *= np.random.uniform(0.8, 1.2)
    heavy_ratio = np.clip(heavy_ratio, 0.0, 0.5)

    motorcycle_ratio = BASE_MOTORCYCLE_RATIO
    # More motorcycles during peak hours
    if is_peak:
        motorcycle_ratio *= 1.15
    # Fewer motorcycles in rain
    if weather_code == 2:
        motorcycle_ratio *= 0.7
    motorcycle_ratio *= np.random.uniform(0.85, 1.15)
    motorcycle_ratio = np.clip(motorcycle_ratio, 0.05, 0.65)

    return {
        "timestamp": timestamp,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_peak_hour": is_peak,
        "vehicle_flow": round(flow, 2),
        "heavy_vehicle_ratio": round(heavy_ratio, 4),
        "motorcycle_ratio": round(motorcycle_ratio, 4),
        "avg_speed_kmh": round(speed, 2),
        "congestion_level": round(congestion, 4),
        "weather_code": weather_code,
        "event_impact": round(event_impact, 4),
        "is_market_day": is_market,
        "intersection_id": intersection_id,
    }


def _log_statistics(df: pd.DataFrame) -> None:
    """Log descriptive statistics for generated seed data."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[numeric_cols].describe()

    for col in numeric_cols:
        logger.info(
            "variable_stats",
            variable=col,
            mean=round(stats[col]["mean"], 4),
            std=round(stats[col]["std"], 4),
            min=round(stats[col]["min"], 4),
            max=round(stats[col]["max"], 4),
        )


def _save_seed_data(df: pd.DataFrame) -> None:
    """Save seed data CSV and feature metadata JSON."""
    # Save CSV
    csv_path = resolve_path("data/processed/train_seed.csv")
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    logger.info("saved_csv", path=str(csv_path), rows=len(df))

    # Save feature metadata
    metadata = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        metadata[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "dtype": str(df[col].dtype),
        }

    meta_path = resolve_path("data/processed/feature_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("saved_metadata", path=str(meta_path))


def normalize_data(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    scaler_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize numeric features using MinMaxScaler.

    Args:
        df: Input DataFrame with traffic data.
        feature_columns: List of columns to normalize. Defaults to NUMERIC_FEATURES.
        scaler_path: Path to save the fitted scaler. Defaults to models/timegan/scaler.pkl.

    Returns:
        Tuple of (normalized DataFrame, fitted MinMaxScaler).
    """
    if feature_columns is None:
        feature_columns = NUMERIC_FEATURES

    scaler = MinMaxScaler()
    df_normalized = df.copy()

    # Convert boolean columns to int for scaling
    for col in feature_columns:
        if df_normalized[col].dtype == bool:
            df_normalized[col] = df_normalized[col].astype(int)

    df_normalized[feature_columns] = scaler.fit_transform(
        df_normalized[feature_columns].values
    )

    # Save scaler
    if scaler_path is None:
        scaler_path = str(resolve_path("models/timegan/scaler.pkl"))
    ensure_dir(Path(scaler_path).parent)
    joblib.dump(scaler, scaler_path)

    logger.info(
        "data_normalized",
        n_features=len(feature_columns),
        scaler_path=scaler_path,
    )

    return df_normalized, scaler


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = DEFAULT_SEQ_LEN,
    feature_columns: Optional[List[str]] = None,
    group_by_intersection: bool = True,
) -> np.ndarray:
    """
    Convert a DataFrame of traffic data into windowed sequences for TimeGAN.

    Args:
        df: Normalized DataFrame with traffic data.
        seq_len: Length of each sequence (number of 15-min windows).
        feature_columns: Columns to include in sequences. Defaults to NUMERIC_FEATURES.
        group_by_intersection: If True, create sequences per intersection separately.

    Returns:
        Numpy array of shape (n_samples, seq_len, n_features).
    """
    if feature_columns is None:
        feature_columns = NUMERIC_FEATURES

    all_sequences = []

    if group_by_intersection and "intersection_id" in df.columns:
        for intersection in df["intersection_id"].unique():
            intersection_df = df[df["intersection_id"] == intersection]
            values = intersection_df[feature_columns].values
            sequences = _sliding_window(values, seq_len)
            all_sequences.append(sequences)
    else:
        values = df[feature_columns].values
        sequences = _sliding_window(values, seq_len)
        all_sequences.append(sequences)

    result = np.concatenate(all_sequences, axis=0)

    logger.info(
        "sequences_created",
        shape=result.shape,
        n_samples=result.shape[0],
        seq_len=result.shape[1],
        n_features=result.shape[2],
    )

    return result


def _sliding_window(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create sliding window sequences from a 2D array.

    Args:
        data: Array of shape (n_timesteps, n_features).
        window_size: Size of each window.

    Returns:
        Array of shape (n_windows, window_size, n_features).
    """
    n_samples = len(data) - window_size + 1
    if n_samples <= 0:
        return np.empty((0, window_size, data.shape[1]))

    sequences = np.array([
        data[i: i + window_size]
        for i in range(n_samples)
    ])
    return sequences


def split_dataset(
    sequences: np.ndarray,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split sequences into training and validation sets.

    Args:
        sequences: Array of shape (n_samples, seq_len, n_features).
        train_ratio: Fraction of data for training (default: 0.8).
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of (train_sequences, val_sequences).
    """
    n_total = len(sequences)
    n_train = int(n_total * train_ratio)

    if shuffle:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)
        sequences = sequences[indices]

    train = sequences[:n_train]
    val = sequences[n_train:]

    logger.info(
        "dataset_split",
        total=n_total,
        train=len(train),
        val=len(val),
        train_ratio=train_ratio,
    )

    return train, val


def generate_validation_data(
    n_days: int = 14,
    intersections: Optional[List[str]] = None,
    seed: int = 123,
) -> None:
    """
    Generate simulated validation data (stand-in for real field data).
    Uses a different seed to simulate independent observations.

    Args:
        n_days: Number of days of validation data.
        intersections: Intersection IDs for validation.
        seed: Different seed from training data.
    """
    if intersections is None:
        intersections = ["carrera_11_norte", "av_castellana_entrada"]

    logger.info("generating_validation_data", n_days=n_days, seed=seed)

    df = generate_seed_data(
        n_days=n_days,
        intersections=intersections,
        seed=seed,
        start_date="2024-04-01",
    )

    val_dir = resolve_path("data/validation")
    ensure_dir(val_dir)

    # Split by intersection and save
    for intersection in intersections:
        idf = df[df["intersection_id"] == intersection]
        filename_map = {
            "carrera_11_norte": "pilot_carrera11.csv",
            "carrera_11_sur": "pilot_carrera11_sur.csv",
            "av_castellana_entrada": "pilot_av_castellana.csv",
            "av_castellana_salida": "pilot_av_castellana_salida.csv",
            "calle_14_centro_historico": "pilot_calle14.csv",
            "acceso_morca": "pilot_acceso_morca.csv",
        }
        filename = filename_map.get(intersection, f"pilot_{intersection}.csv")
        path = val_dir / filename
        idf.to_csv(path, index=False)
        logger.info("saved_validation_data", path=str(path), rows=len(idf))
