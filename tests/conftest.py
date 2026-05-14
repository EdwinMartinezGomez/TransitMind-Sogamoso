"""Shared test fixtures for TransitMind Sogamoso."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.shared.constants import INTERSECTIONS, NUMERIC_FEATURES


@pytest.fixture
def sample_config():
    return {
        "model": {"n_features": 11, "seq_len": 24, "hidden_dim": 32, "noise_dim": 32, "num_layers": 2, "dropout": 0.1},
        "training": {"batch_size": 8, "lr_generator": 0.001, "lr_discriminator": 0.001, "gamma": 1.0,
                      "epochs_autoencoder": 5, "epochs_supervisor": 5, "epochs_joint": 5,
                      "checkpoint_every": 100, "log_every_phase_ab": 2, "log_every_phase_c": 2},
        "data": {"seed": 42, "n_days": 3, "train_ratio": 0.8, "intersections": INTERSECTIONS[:2]},
        "paths": {"checkpoints": "models/timegan/checkpoints", "best_model": "models/timegan/best_model",
                  "scaler": "models/timegan/scaler.pkl"},
    }


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "hour": np.tile(np.repeat(range(24), 4), n // 96 + 1)[:n],
        "day_of_week": np.zeros(n, dtype=int),
        "is_peak_hour": np.random.choice([True, False], n),
        "vehicle_flow": np.random.uniform(10, 200, n),
        "heavy_vehicle_ratio": np.random.uniform(0.05, 0.3, n),
        "motorcycle_ratio": np.random.uniform(0.2, 0.5, n),
        "avg_speed_kmh": np.random.uniform(15, 50, n),
        "congestion_level": np.random.uniform(0, 1, n),
        "weather_code": np.random.choice([0, 1, 2, 3], n),
        "event_impact": np.random.uniform(0, 0.3, n),
        "is_market_day": np.random.choice([True, False], n),
        "intersection_id": np.random.choice(INTERSECTIONS[:2], n),
    })
