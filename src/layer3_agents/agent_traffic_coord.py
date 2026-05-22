"""
TransitMind Sogamoso — Agent Traffic Coordinator
==================================================
Agent 6: Translates causal analyses into concrete traffic light
commands. Respects cooldown between adjustments and physical
limits defined in configuration.
"""

import time
from typing import Any, Dict, List, Optional, TypedDict

from src.shared.logger import get_logger

logger = get_logger("layer3.agent_traffic_coord")

SEVERITY_ORDER = {"baja": 0, "media": 1, "alta": 2, "critica": 3}

# Extension mapping by severity (seconds)
SEVERITY_EXTENSION_MAP = {
    "baja": 5,
    "media": 12,
    "alta": 25,
    "critica": 40,
}


class TrafficCommand(TypedDict):
    """Traffic light adjustment command."""

    intersection_id: str
    green_extension_seconds: int
    priority_direction: str        # "norte_sur" | "este_oeste" | "rotacional"
    cycle_adjustment_percent: int  # 0-20%
    rationale: str
    severity: str
    estimated_tmc_reduction: float


class TrafficCoordinatorAgent:
    """Generates traffic light commands based on causal analyses."""

    def __init__(self, config: dict):
        self._max_extension = config["agents"]["traffic_coordinator"][
            "max_green_extension_seconds"
        ]
        self._min_extension = config["agents"]["traffic_coordinator"][
            "min_green_extension_seconds"
        ]
        self._cooldown = config["agents"]["traffic_coordinator"]["cooldown_seconds"]
        self._last_command_times: Dict[str, float] = {}

    def _in_cooldown(self, intersection_id: str) -> bool:
        """True if this intersection was adjusted less than cooldown_seconds ago."""
        last_time = self._last_command_times.get(intersection_id)
        if last_time is None:
            return False
        return (time.time() - last_time) < self._cooldown

    def _calculate_extension(self, analysis: dict) -> int:
        """
        Calculate optimal green phase extension.

        Based on severity and congestion_level_adjusted.
        Clamped to [min_extension, max_extension].
        """
        causal_ctx = analysis.get("causal_context", {})
        severity = causal_ctx.get("severity", "baja")
        forecast = analysis.get("traffic_forecast", {})
        congestion = forecast.get("congestion_level_adjusted", 0.0)

        # Base from severity
        base = SEVERITY_EXTENSION_MAP.get(severity, 5)

        # Scale by congestion level
        scaled = int(base * (0.7 + 0.3 * congestion))

        # Clamp
        return max(self._min_extension, min(self._max_extension, scaled))

    def _estimate_tmc_reduction(self, extension_s: int, severity: str) -> float:
        """
        Simple TMC reduction estimation model.

        Base: extension_s / max_extension * 0.35 (max 35% reduction)
        Severity factor: alta/critica have higher improvement potential.
        Returns: float between 0.0 and 0.40
        """
        base = (extension_s / max(self._max_extension, 1)) * 0.35

        severity_factor = {
            "baja": 0.6,
            "media": 0.8,
            "alta": 1.0,
            "critica": 1.15,
        }
        factor = severity_factor.get(severity, 0.8)

        return min(0.40, round(base * factor, 4))

    def run(self, state: dict) -> dict:
        """
        Main logic:
        1. For each intersection with causal_analyses:
           a. Skip if in cooldown or severity == "baja"
           b. Calculate green extension
           c. Get priority_direction from recommendations
           d. Calculate cycle_adjustment_percent and TMC reduction
        2. Calculate global tmc_reduction_estimate
        """
        causal_analyses = state.get("causal_analyses", {})
        predictions = state.get("predictions", {})

        traffic_commands: Dict[str, Any] = {}
        tmc_reductions: List[float] = []
        congestion_weights: List[float] = []
        error_log = list(state.get("error_log", []))

        for intersection_id, analysis in causal_analyses.items():
            causal_ctx = analysis.get("causal_context", {})
            severity = causal_ctx.get("severity", "baja")

            # Skip if in cooldown
            if self._in_cooldown(intersection_id):
                logger.debug(
                    "traffic_coord_cooldown",
                    intersection=intersection_id,
                )
                continue

            # Skip if low severity
            if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get("media", 1):
                continue

            # Calculate extension
            extension = self._calculate_extension(analysis)

            # Get priority direction from recommendations
            recommendations = analysis.get("recommendations", {})
            tl_adjustment = recommendations.get("traffic_light_adjustment", {})
            priority_direction = tl_adjustment.get("priority_direction", "rotacional")

            # Calculate cycle adjustment percent (capped at 20%)
            cycle_adj = min(20, int((extension / 90) * 100))

            # Estimate TMC reduction
            tmc_red = self._estimate_tmc_reduction(extension, severity)

            # Build rationale
            rationale = (
                f"Extensión de {extension}s en dirección {priority_direction} "
                f"por congestión {severity} "
                f"(nivel: {analysis.get('traffic_forecast', {}).get('congestion_level_adjusted', 'N/A')})"
            )

            command: TrafficCommand = {
                "intersection_id": intersection_id,
                "green_extension_seconds": extension,
                "priority_direction": priority_direction,
                "cycle_adjustment_percent": cycle_adj,
                "rationale": rationale,
                "severity": severity,
                "estimated_tmc_reduction": tmc_red,
            }

            traffic_commands[intersection_id] = command
            self._last_command_times[intersection_id] = time.time()

            # For weighted average
            congestion = predictions.get(intersection_id, 0.5)
            tmc_reductions.append(tmc_red)
            congestion_weights.append(congestion)

        # Calculate global weighted TMC reduction estimate
        if tmc_reductions and congestion_weights:
            total_weight = sum(congestion_weights)
            if total_weight > 0:
                tmc_reduction_estimate = sum(
                    r * w for r, w in zip(tmc_reductions, congestion_weights)
                ) / total_weight
            else:
                tmc_reduction_estimate = sum(tmc_reductions) / len(tmc_reductions)
        else:
            tmc_reduction_estimate = 0.0

        commands_applied = len(traffic_commands) > 0

        logger.info(
            "traffic_coordinator_complete",
            commands_generated=len(traffic_commands),
            tmc_reduction_estimate=round(tmc_reduction_estimate, 4),
            commands_applied=commands_applied,
        )

        return {
            **state,
            "traffic_commands": traffic_commands,
            "commands_applied": commands_applied,
            "tmc_reduction_estimate": round(tmc_reduction_estimate, 4),
            "error_log": error_log,
        }


def traffic_coordinator_node(state: dict) -> dict:
    """LangGraph node function for the Traffic Coordinator."""
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")
    agent = TrafficCoordinatorAgent(config)
    return agent.run(state)
