"""
TransitMind Sogamoso — Agent GAN Simulator
============================================
Agent 3: Interface to Layer 1 TimeGAN generator.
Generates detailed synthetic traffic scenarios for
high-congestion intersections identified by the Predictor.

NOTE: The original GANSimulatorAgent class and generate_scenario()
      method are preserved. Only run() and gan_simulator_node() are added.
"""

from typing import Any, Dict, List, Optional

import requests

from src.shared.logger import get_logger

logger = get_logger("layer3.agent_gan_simulator")


class GANSimulatorAgent:
    """Calls the Layer 1 API to generate synthetic traffic scenarios."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

    def generate_scenario(
        self, intersection_id: str, n_samples: int = 100, scenario: Optional[str] = None
    ) -> Dict[str, Any]:
        response = requests.post(
            f"{self.api_url}/generate",
            json={"intersection_id": intersection_id, "n_samples": n_samples, "scenario": scenario},
        )
        response.raise_for_status()
        return response.json()

    def _infer_active_scenario(self, sensor_data: Dict[str, Any]) -> str:
        """
        Infer the active scenario from sensor data majority voting.

        Rules:
        - is_market_day=True in majority → "market_day"
        - weather_code=3 in majority → "morning_fog"
        - event_impact > 0.7 in majority → "coliseo_event"
        - None of the above → "normal_weekday"
        """
        market_count = 0
        fog_count = 0
        event_count = 0
        total_records = 0

        for _iid, data in sensor_data.items():
            records = data.get("synthetic_data", [])
            for r in records:
                total_records += 1
                if r.get("is_market_day", False):
                    market_count += 1
                if r.get("weather_code", 0) == 3:
                    fog_count += 1
                if float(r.get("event_impact", 0.0)) > 0.7:
                    event_count += 1

        if total_records == 0:
            return "normal_weekday"

        half = total_records / 2

        if market_count > half:
            return "market_day"
        if fog_count > half:
            return "morning_fog"
        if event_count > half:
            return "coliseo_event"

        return "normal_weekday"

    def run(self, state: dict) -> dict:
        """
        Generate detailed scenarios for high-congestion intersections.

        If high_congestion_intersections is empty, reuse sensor_data
        as gan_scenarios (no extra API calls needed).

        For high-congestion intersections, call Layer 1 to generate
        fresh detailed scenarios.
        """
        sensor_data = state.get("sensor_data", {})
        high_congestion = state.get("high_congestion_intersections", [])
        scenario = state.get("scenario")

        # Infer active scenario from sensor data
        active_scenario = self._infer_active_scenario(sensor_data)
        # Override with explicit scenario if provided
        if scenario and scenario != "normal_weekday":
            active_scenario = scenario

        gan_scenarios: Dict[str, Any] = {}
        error_log = list(state.get("error_log", []))

        if not high_congestion:
            # No high congestion — reuse sensor data directly
            logger.info("gan_simulator_skip", reason="no_high_congestion")
            gan_scenarios = dict(sensor_data)
        else:
            # Generate fresh scenarios for high-congestion intersections
            for iid in high_congestion:
                try:
                    result = self.generate_scenario(
                        intersection_id=iid,
                        n_samples=100,
                        scenario=active_scenario,
                    )
                    gan_scenarios[iid] = result
                except Exception as e:
                    logger.warning(
                        "gan_simulator_failed",
                        intersection=iid,
                        error=str(e),
                    )
                    error_log.append(f"gan_simulator_failed: {iid} - {str(e)}")
                    # Fallback: use sensor data if available
                    if iid in sensor_data:
                        gan_scenarios[iid] = sensor_data[iid]

            # Also include non-high-congestion intersections from sensor
            for iid, data in sensor_data.items():
                if iid not in gan_scenarios:
                    gan_scenarios[iid] = data

        logger.info(
            "gan_simulator_complete",
            scenarios_generated=len(gan_scenarios),
            active_scenario=active_scenario,
        )

        return {
            **state,
            "gan_scenarios": gan_scenarios,
            "active_scenario": active_scenario,
            "error_log": error_log,
        }


def gan_simulator_node(state: dict) -> dict:
    """LangGraph node function for the GAN Simulator."""
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")
    api_url = config["agents"]["gan_simulator"]["api_url"]
    agent = GANSimulatorAgent(api_url=api_url)
    return agent.run(state)
