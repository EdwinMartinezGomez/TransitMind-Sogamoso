"""
TransitMind Sogamoso — Agent Causal Analyst
=============================================
Agent 4: Bridge between LangGraph and Layer 2 API (LLM + RAG).
Only calls Layer 2 for intersections marked as high-congestion
by the Predictor. Creates minimal fallback results for low-congestion
intersections to avoid expensive LLM calls.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.shared.logger import get_logger

logger = get_logger("layer3.agent_causal_analyst")


class CausalAnalystAgent:
    """Coordinates causal analysis via Layer 2 API for high-congestion intersections."""

    def __init__(self, config: dict):
        self._layer2_url = config["layer2_api"]["base_url"]
        self._timeout = config["layer2_api"]["timeout_seconds"]
        self._skip_low = config["agents"]["causal_analyst"]["skip_if_low_congestion"]
        self._low_threshold = config["thresholds"]["low_congestion"]

    async def _analyze_intersection(
        self,
        client: httpx.AsyncClient,
        intersection_id: str,
        scenario: str,
    ) -> Tuple[str, Optional[dict]]:
        """
        Call POST /analyze-scenario on Layer 2.
        Returns (intersection_id, CausalAnalysisResult dict | None).
        """
        try:
            import time

            t0 = time.time()
            response = await client.post(
                f"{self._layer2_url}/analyze-scenario",
                json={
                    "intersection_id": intersection_id,
                    "scenario": scenario,
                    "n_samples": 50,
                },
            )
            response.raise_for_status()
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.info(
                "causal_analysis_received",
                intersection=intersection_id,
                elapsed_ms=elapsed_ms,
            )
            return intersection_id, response.json()
        except Exception as e:
            logger.warning(
                "causal_analysis_failed",
                intersection=intersection_id,
                error=str(e),
            )
            return intersection_id, None

    def _create_fallback_result(
        self,
        intersection_id: str,
        congestion_level: float,
    ) -> dict:
        """
        Create a minimal CausalAnalysisResult for intersections
        not sent to Layer 2 (low congestion or API failure).
        """
        # Map congestion to severity
        if congestion_level < 0.4:
            severity = "baja"
        elif congestion_level < 0.6:
            severity = "media"
        elif congestion_level < 0.85:
            severity = "alta"
        else:
            severity = "critica"

        return {
            "intersection_id": intersection_id,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "causal_context": {
                "primary_cause": "Datos insuficientes para análisis causal profundo",
                "secondary_causes": [],
                "causal_explanation": "Análisis basado en predicción local sin LLM",
                "severity": severity,
                "confidence": 0.5,
            },
            "traffic_forecast": {
                "congestion_level_adjusted": congestion_level,
                "peak_window": "N/A",
                "expected_delay_minutes": int(congestion_level * 30),
                "affected_intersections": [intersection_id],
            },
            "recommendations": {
                "traffic_light_adjustment": {
                    "intersection_id": intersection_id,
                    "green_phase_extension_seconds": 0,
                    "priority_direction": "rotacional",
                    "rationale": "Sin ajuste recomendado — congestión baja",
                },
                "alternative_routes": [],
                "citizen_alert": f"Tráfico normal en {intersection_id.replace('_', ' ').title()}.",
            },
            "rag_sources_used": [],
            "llm_reasoning_trace": "",
            "processing_time_ms": 0,
            "model_used": "fallback",
            "is_fallback": True,
        }

    async def run_async(self, state: dict) -> dict:
        """
        Main logic:
        1. Determine which intersections to analyze with Layer 2
        2. Create fallback results for skipped intersections
        3. Async parallel calls to Layer 2 for high-congestion ones
        4. Combine results and compute causal_status
        """
        predictions = state.get("predictions", {})
        high_congestion = state.get("high_congestion_intersections", [])
        all_intersections = state.get("intersections_to_analyze", [])
        scenario = state.get("active_scenario") or state.get("scenario") or "normal_weekday"

        # Determine which intersections to analyze deeply
        if self._skip_low:
            to_analyze = list(high_congestion)
        else:
            to_analyze = list(all_intersections)

        to_skip = [iid for iid in all_intersections if iid not in to_analyze]

        logger.info(
            "causal_analyst_starting",
            to_analyze=len(to_analyze),
            to_skip=len(to_skip),
            scenario=scenario,
        )

        causal_analyses: Dict[str, Any] = {}
        error_log = list(state.get("error_log", []))

        # Create fallback for skipped intersections
        for iid in to_skip:
            congestion = predictions.get(iid, 0.0)
            causal_analyses[iid] = self._create_fallback_result(iid, congestion)

        # Analyze high-congestion intersections via Layer 2
        if to_analyze:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                tasks = [
                    self._analyze_intersection(client, iid, scenario)
                    for iid in to_analyze
                ]
                results = await asyncio.gather(*tasks, return_exceptions=False)

            analyzed_ok = 0
            for intersection_id, result in results:
                if result is not None:
                    causal_analyses[intersection_id] = result
                    analyzed_ok += 1
                else:
                    # Create fallback for failed analysis
                    congestion = predictions.get(intersection_id, 0.5)
                    causal_analyses[intersection_id] = self._create_fallback_result(
                        intersection_id, congestion
                    )
                    error_log.append(f"causal_analysis_fallback: {intersection_id}")

            # Determine status
            if analyzed_ok == len(to_analyze):
                causal_status = "ok"
            elif analyzed_ok > 0:
                causal_status = "partial"
            else:
                causal_status = "failed"
        else:
            causal_status = "ok"  # Nothing to analyze deeply

        logger.info(
            "causal_analyst_complete",
            total_results=len(causal_analyses),
            status=causal_status,
        )

        return {
            **state,
            "causal_analyses": causal_analyses,
            "causal_status": causal_status,
            "error_log": error_log,
        }

    def run(self, state: dict) -> dict:
        """Sync wrapper for LangGraph node compatibility."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.run_async(state))
        else:
            return asyncio.run(self.run_async(state))


def causal_analyst_node(state: dict) -> dict:
    """LangGraph node function for the Causal Analyst."""
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")
    agent = CausalAnalystAgent(config)
    return agent.run(state)
