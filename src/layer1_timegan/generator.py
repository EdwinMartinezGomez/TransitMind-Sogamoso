"""
TransitMind Sogamoso — Synthetic Data Generator (Fase 3)
==========================================================
Public interface for generating synthetic traffic data using trained TimeGAN.
This module is the bridge between Layer 1 (TimeGAN) and Layer 3 (Multi-Agents).

Classes:
    TrafficDataGenerator: Main class for on-demand synthetic generation.

Constants:
    SOGAMOSO_SCENARIOS: Pre-defined traffic scenarios for Sogamoso.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from src.shared.constants import (
    ALL_COLUMNS,
    INTERSECTIONS,
    NUMERIC_FEATURES,
)
from src.shared.logger import setup_logger
from src.shared.schemas import AgentDataBatch, AgentTrafficData
from src.shared.utils import get_config, resolve_path

logger = setup_logger("generator")


# ============================================
# Pre-defined Sogamoso Traffic Scenarios
# ============================================

SOGAMOSO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "normal_weekday": {
        "weather": 0,
        "is_market_day": False,
        "event_impact": 0.0,
        "description": "Día laborable normal, clima despejado",
    },
    "market_day": {
        "weather": 0,
        "is_market_day": True,
        "event_impact": 0.3,
        "description": "Día de mercado campesino (miércoles o sábado)",
    },
    "morning_fog": {
        "weather": 3,
        "is_market_day": False,
        "event_impact": 0.0,
        "description": "Niebla matutina típica del valle del Chicamocha",
    },
    "coliseo_event": {
        "weather": 0,
        "is_market_day": False,
        "event_impact": 0.9,
        "description": "Evento en el Coliseo Olímpico de Sogamoso",
    },
    "rain_market": {
        "weather": 2,
        "is_market_day": True,
        "event_impact": 0.4,
        "description": "Día de mercado con lluvia",
    },
    "duitama_blockage": {
        "weather": 0,
        "is_market_day": False,
        "event_impact": 0.7,
        "description": "Bloqueo vial en la vía Sogamoso-Duitama",
    },
}


class TrafficDataGenerator:
    """
    Generates synthetic traffic data using a trained TimeGAN model.

    This class is the public API for Layer 1, designed to be called
    by the GAN Simulator agent in Layer 3.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the generator with a trained model and scaler.

        Args:
            model_path: Path to the trained model file (.pt).
                Defaults to models/timegan/best_model/timegan_best.pt.
            scaler_path: Path to the fitted MinMaxScaler (.pkl).
                Defaults to models/timegan/scaler.pkl.
            config: Configuration dictionary. Loads from YAML if None.
        """
        if config is None:
            config = get_config()
        self.config = config

        model_cfg = config.get("model", {})
        self.hidden_dim = model_cfg.get("hidden_dim", 64)
        self.noise_dim = model_cfg.get("noise_dim", 64)
        self.seq_len = model_cfg.get("seq_len", 24)
        self.n_features = model_cfg.get("n_features", 11)

        # Paths
        paths_cfg = config.get("paths", {})
        if model_path is None:
            model_path = str(
                resolve_path(paths_cfg.get("best_model", "models/timegan/best_model"))
                / "timegan_best.pt"
            )
        if scaler_path is None:
            scaler_path = str(
                resolve_path(paths_cfg.get("scaler", "models/timegan/scaler.pkl"))
            )

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model_loaded = False
        self.scaler = None
        self.generator = None
        self.supervisor = None
        self.recovery = None

        # Try to load model and scaler
        self._load_model()
        self._load_scaler()

    def _load_model(self) -> None:
        """Load the trained TimeGAN model components."""
        try:
            from src.layer1_timegan.timegan_model import build_timegan

            checkpoint = torch.load(
                self.model_path, map_location="cpu", weights_only=False
            )

            components = build_timegan(self.config)

            # Load saved weights
            components["generator"].load_state_dict(checkpoint["generator"])
            components["supervisor"].load_state_dict(checkpoint["supervisor"])
            components["recovery"].load_state_dict(checkpoint["recovery"])

            self.generator = components["generator"]
            self.supervisor = components["supervisor"]
            self.recovery = components["recovery"]

            # Set to eval mode
            self.generator.eval()
            self.supervisor.eval()
            self.recovery.eval()

            self.model_loaded = True
            logger.info("model_loaded", path=self.model_path)
        except FileNotFoundError:
            logger.warning("model_not_found", path=self.model_path)
            self.model_loaded = False
        except Exception as e:
            logger.error("model_load_error", error=str(e))
            self.model_loaded = False

    def _load_scaler(self) -> None:
        """Load the fitted MinMaxScaler."""
        try:
            self.scaler = joblib.load(self.scaler_path)
            logger.info("scaler_loaded", path=self.scaler_path)
        except FileNotFoundError:
            logger.warning("scaler_not_found", path=self.scaler_path)
            self.scaler = None
        except Exception as e:
            logger.error("scaler_load_error", error=str(e))
            self.scaler = None

    def generate(
        self,
        n_samples: int,
        intersection_id: str,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic traffic data sequences.

        Args:
            n_samples: Number of sequences to generate.
            intersection_id: Target intersection ID.
            scenario: Optional scenario parameters. Can be a dict with keys:
                - weather (int): Weather code (0-3)
                - is_market_day (bool): Market day flag
                - event_impact (float): Event impact level (0-1)
                - hour_range (list[int]): [start_hour, end_hour] filter

        Returns:
            DataFrame with denormalized traffic data columns.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Model not loaded. Train the model first or check model_path."
            )

        start_time = time.time()

        logger.info(
            "generating",
            n_samples=n_samples,
            intersection_id=intersection_id,
            scenario=scenario,
        )

        with torch.no_grad():
            # Generate random noise
            z = torch.randn(n_samples, self.seq_len, self.noise_dim)

            # Pass through Generator → Supervisor → Recovery
            h_fake = self.generator(z)
            h_supervised = self.supervisor(h_fake)
            x_synthetic = self.recovery(h_supervised)

        # Convert to numpy
        synthetic_data = x_synthetic.numpy()

        # Reshape to 2D: (n_samples * seq_len, n_features)
        n_total = n_samples * self.seq_len
        flat_data = synthetic_data.reshape(n_total, self.n_features)

        # Inverse transform (denormalize)
        if self.scaler is not None:
            flat_data = self.scaler.inverse_transform(flat_data)

        # Build DataFrame
        df = pd.DataFrame(flat_data, columns=NUMERIC_FEATURES)

        # Post-process: clip values to valid ranges
        df = self._postprocess(df, intersection_id)

        # Apply scenario constraints if provided
        if scenario:
            df = self._apply_scenario(df, scenario)

        # Add timestamp and intersection_id
        df = self._add_metadata(df, intersection_id, n_samples)

        elapsed = time.time() - start_time
        logger.info(
            "generation_complete",
            n_rows=len(df),
            duration_s=round(elapsed, 3),
        )

        return df

    def _postprocess(self, df: pd.DataFrame, intersection_id: str) -> pd.DataFrame:
        """Post-process generated data to ensure valid ranges."""
        # Guard against NaN/inf from partially-trained models
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df["hour"] = df["hour"].round().clip(0, 23).astype(int)
        df["day_of_week"] = df["day_of_week"].round().clip(0, 6).astype(int)
        df["is_peak_hour"] = (df["is_peak_hour"] > 0.5).astype(bool)
        df["vehicle_flow"] = df["vehicle_flow"].clip(0, 300).round(2)
        df["heavy_vehicle_ratio"] = df["heavy_vehicle_ratio"].clip(0, 1).round(4)
        df["motorcycle_ratio"] = df["motorcycle_ratio"].clip(0, 1).round(4)
        df["avg_speed_kmh"] = df["avg_speed_kmh"].clip(0, 80).round(2)
        df["congestion_level"] = df["congestion_level"].clip(0, 1).round(4)
        df["weather_code"] = df["weather_code"].round().clip(0, 3).astype(int)
        df["event_impact"] = df["event_impact"].clip(0, 1).round(4)
        df["is_market_day"] = (df["is_market_day"] > 0.5).astype(bool)
        return df

    def _apply_scenario(
        self, df: pd.DataFrame, scenario: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply scenario constraints to generated data."""
        if "weather" in scenario:
            df["weather_code"] = int(scenario["weather"])

        if "is_market_day" in scenario:
            df["is_market_day"] = bool(scenario["is_market_day"])

        if "event_impact" in scenario:
            df["event_impact"] = float(scenario["event_impact"])

        if "hour_range" in scenario:
            h_start, h_end = scenario["hour_range"]
            mask = (df["hour"] >= h_start) & (df["hour"] <= h_end)
            df = df[mask].reset_index(drop=True)

        return df

    def _add_metadata(
        self, df: pd.DataFrame, intersection_id: str, n_samples: int
    ) -> pd.DataFrame:
        """Add timestamp and intersection_id columns."""
        base_time = datetime(2024, 1, 1)
        timestamps = [
            base_time + timedelta(minutes=15 * i) for i in range(len(df))
        ]
        df.insert(0, "timestamp", timestamps[: len(df)])
        df["intersection_id"] = intersection_id
        return df

    def generate_batch_scenarios(
        self,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        n_per_scenario: int = 100,
        intersection_id: str = "carrera_11_norte",
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate multiple scenarios in batch.

        Args:
            scenarios: List of scenario dicts. Defaults to SOGAMOSO_SCENARIOS.
            n_per_scenario: Number of samples per scenario.
            intersection_id: Target intersection.

        Returns:
            Dictionary mapping scenario names to DataFrames.
        """
        if scenarios is None:
            scenarios_dict = SOGAMOSO_SCENARIOS
        else:
            scenarios_dict = {f"scenario_{i}": s for i, s in enumerate(scenarios)}

        results: Dict[str, pd.DataFrame] = {}

        for name, params in scenarios_dict.items():
            scenario_params = {
                k: v for k, v in params.items() if k != "description"
            }
            try:
                df = self.generate(
                    n_samples=n_per_scenario,
                    intersection_id=intersection_id,
                    scenario=scenario_params,
                )
                results[name] = df
                logger.info(
                    "batch_scenario_generated",
                    scenario=name,
                    rows=len(df),
                )
            except Exception as e:
                logger.error(
                    "batch_scenario_error",
                    scenario=name,
                    error=str(e),
                )

        return results

    def export_for_agents(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert a generated DataFrame to the JSON format expected
        by Layer 3 LangGraph agents.

        Args:
            df: Generated traffic DataFrame.

        Returns:
            Dictionary compatible with AgentDataBatch schema.

        Raises:
            ValueError: If the data fails Pydantic validation.
        """
        records = []
        for _, row in df.iterrows():
            record = AgentTrafficData(
                intersection_id=str(row.get("intersection_id", "")),
                timestamp=str(row.get("timestamp", "")),
                vehicle_flow=float(row.get("vehicle_flow", 0)),
                congestion_level=float(row.get("congestion_level", 0)),
                avg_speed_kmh=float(row.get("avg_speed_kmh", 0)),
                weather_code=int(row.get("weather_code", 0)),
                scenario_metadata=None,
            )
            records.append(record)

        batch = AgentDataBatch(
            data=records,
            scenario_name="generated",
            generated_at=datetime.now().isoformat(),
            n_samples=len(records),
            schema_version="1.0",
        )

        result = batch.model_dump()

        logger.info(
            "exported_for_agents",
            n_records=len(records),
            schema_version="1.0",
        )

        return result

    def save_synthetic_data(
        self, df: pd.DataFrame, filename: Optional[str] = None
    ) -> str:
        """
        Save generated synthetic data to disk.

        Args:
            df: Generated DataFrame.
            filename: Output filename. Auto-generated if None.

        Returns:
            Path to the saved file.
        """
        output_dir = resolve_path("data/synthetic/generated_flows")
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_{ts}.csv"

        path = output_dir / filename
        df.to_csv(path, index=False)
        logger.info("synthetic_data_saved", path=str(path), rows=len(df))
        return str(path)
