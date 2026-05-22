"""
TransitMind Sogamoso — Agent Predictor
========================================
Agent 2: Analyzes sensor data to predict which intersections
need immediate attention. Gates expensive Layer 2 LLM calls
by filtering out low-congestion intersections early.
"""

from typing import Any, Dict, List, Optional

from src.shared.logger import get_logger

logger = get_logger("layer3.agent_predictor")


class PredictorAgent:
    """Predicts congestion levels and identifies high-priority intersections."""

    def __init__(self, config: dict):
        self._window_size = config["agents"]["predictor"]["window_size"]
        self._threshold = config["agents"]["predictor"]["congestion_threshold"]

    def _predict_congestion(self, sensor_records: list) -> float:
        """
        Calculate weighted-average congestion level.

        Formula:
        1. Take the last window_size records from synthetic_data
        2. Compute weighted mean of congestion_level
           (weight[i] = (i+1) / sum(range(1, window_size+1)))
        3. Apply peak_hour factor: if is_peak_hour → * 1.15
        4. Apply weather factor: weather_code == 3 → * 1.10
        5. Clamp to [0.0, 1.0]

        Returns:
            float — predicted congestion
        """
        if not sensor_records:
            return 0.0

        # Take last window_size records
        window = sensor_records[-self._window_size:]
        n = len(window)

        if n == 0:
            return 0.0

        # Weighted mean — more recent records have higher weight
        weight_sum = sum(range(1, n + 1))
        weighted_congestion = 0.0
        peak_count = 0
        fog_count = 0

        for i, record in enumerate(window):
            weight = (i + 1) / weight_sum
            congestion = float(record.get("congestion_level", 0.0))
            weighted_congestion += congestion * weight

            if record.get("is_peak_hour", False):
                peak_count += 1
            if record.get("weather_code", 0) == 3:
                fog_count += 1

        # Apply correction factors
        if peak_count > n / 2:
            weighted_congestion *= 1.15

        if fog_count > n / 2:
            weighted_congestion *= 1.10

        # Clamp to [0, 1]
        return max(0.0, min(1.0, weighted_congestion))

    def _detect_trend(self, records: list) -> str:
        """
        Detect trend in the last 4 windows.

        Returns:
            "rising" | "falling" | "stable"
        """
        if len(records) < 3:
            return "stable"

        last_values = [
            float(r.get("congestion_level", 0.0)) for r in records[-4:]
        ]

        if len(last_values) < 3:
            return "stable"

        # Check if last 3 values are monotonically increasing
        rising = all(
            last_values[i] < last_values[i + 1]
            for i in range(len(last_values) - 3, len(last_values) - 1)
        )
        if rising:
            return "rising"

        # Check if last 3 values are monotonically decreasing
        falling = all(
            last_values[i] > last_values[i + 1]
            for i in range(len(last_values) - 3, len(last_values) - 1)
        )
        if falling:
            return "falling"

        return "stable"

    def run(self, state: dict) -> dict:
        """
        Main logic:
        1. Read state["sensor_data"]
        2. For each intersection, extract synthetic_data and predict
        3. Identify high_congestion_intersections (pred > threshold)
        4. If sensor_status == "failed": propagate error and return
        """
        sensor_status = state.get("sensor_status")
        if sensor_status == "failed":
            error_log = list(state.get("error_log", []))
            error_log.append("predictor_skipped: sensor_status=failed")
            logger.warning("predictor_skipped", reason="sensor_status=failed")
            return {
                **state,
                "predictions": {},
                "high_congestion_intersections": [],
                "predictor_status": "failed",
                "error_log": error_log,
            }

        sensor_data = state.get("sensor_data", {})
        predictions: Dict[str, float] = {}
        trends: Dict[str, str] = {}
        high_congestion: List[str] = []

        for intersection_id, data in sensor_data.items():
            records = data.get("synthetic_data", [])
            pred = self._predict_congestion(records)
            trend = self._detect_trend(records)

            predictions[intersection_id] = round(pred, 4)
            trends[intersection_id] = trend

            if pred >= self._threshold:
                high_congestion.append(intersection_id)

        logger.info(
            "predictor_complete",
            total=len(predictions),
            high_congestion=len(high_congestion),
            high_ids=high_congestion,
        )

        return {
            **state,
            "predictions": predictions,
            "high_congestion_intersections": high_congestion,
            "predictor_status": "ok",
        }


def predictor_node(state: dict) -> dict:
    """LangGraph node function for the Predictor."""
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")
    agent = PredictorAgent(config)
    return agent.run(state)
