"""
TransitMind Sogamoso — Agent Sensor
====================================
Agent 1: Collects synthetic traffic data from Layer 1 API
for ALL pilot intersections in parallel using async HTTP.
This is the first node in the LangGraph — without its data,
no other agent can operate.
"""

import asyncio
from typing import Any, Dict, Optional, Tuple

import httpx

from src.shared.constants import INTERSECTIONS
from src.shared.logger import get_logger

logger = get_logger("layer3.agent_sensor")


class SensorAgent:
    """Fetches synthetic traffic data from Layer 1 for all intersections."""

    def __init__(self, config: dict):
        self._layer1_url = config["layer1_api"]["base_url"]
        self._timeout = config["layer1_api"]["timeout_seconds"]
        self._n_samples = config["layer1_api"]["n_samples_per_agent"]

    async def _fetch_intersection(
        self,
        client: httpx.AsyncClient,
        intersection_id: str,
        scenario: Optional[str],
    ) -> Tuple[str, Optional[dict]]:
        """Fetch data for a single intersection. Returns (id, data|None)."""
        try:
            response = await client.post(
                f"{self._layer1_url}/generate",
                json={
                    "intersection_id": intersection_id,
                    "n_samples": self._n_samples,
                    "scenario": scenario,
                },
            )
            response.raise_for_status()
            return intersection_id, response.json()
        except Exception as e:
            logger.warning(
                "sensor_fetch_failed",
                intersection=intersection_id,
                error=str(e),
            )
            return intersection_id, None

    async def run_async(self, state: dict) -> dict:
        """
        Async implementation: fetches all intersections in parallel.
        Uses asyncio.gather() for concurrency.
        """
        scenario = state.get("scenario")
        intersections = state.get("intersections_to_analyze", list(INTERSECTIONS))

        logger.info(
            "sensor_starting",
            n_intersections=len(intersections),
            scenario=scenario,
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [
                self._fetch_intersection(client, iid, scenario)
                for iid in intersections
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        sensor_data: Dict[str, Any] = {}
        failed = []
        for intersection_id, data in results:
            if data is not None:
                sensor_data[intersection_id] = data
            else:
                failed.append(intersection_id)

        if not failed:
            status = "ok"
        elif sensor_data:
            status = "partial"
        else:
            status = "failed"

        error_log = list(state.get("error_log", []))
        if failed:
            error_log.append(f"sensor_failed: {', '.join(failed)}")

        logger.info(
            "sensor_complete",
            fetched=len(sensor_data),
            failed=len(failed),
            status=status,
        )

        return {
            **state,
            "sensor_data": sensor_data,
            "sensor_status": status,
            "error_log": error_log,
        }

    def run(self, state: dict) -> dict:
        """Sync wrapper for LangGraph node compatibility."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop (e.g. FastAPI) — use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.run_async(state))
        else:
            return asyncio.run(self.run_async(state))


def sensor_node(state: dict) -> dict:
    """
    LangGraph node function. Instantiates SensorAgent and runs it.
    LangGraph calls this function with the current state.
    """
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")
    agent = SensorAgent(config)
    return agent.run(state)
