"""
TransitMind Sogamoso — Agent Monitor
======================================
Agent 7: Validates coherence of all decisions in the cycle,
detects anomalies, blocks LLM hallucinations, and decides
whether the cycle should be approved or retried.
"""

import time
from datetime import datetime
from typing import Any, Dict, List

from src.shared.logger import get_logger

logger = get_logger("layer3.agent_monitor")

VALID_PRIORITY_DIRECTIONS = {"norte_sur", "este_oeste", "rotacional"}


class MonitorAgent:
    """Validates decisions, detects anomalies, and approves/retries cycles."""

    def __init__(self, config: dict):
        self._anomaly_threshold = config["agents"]["monitor"]["anomaly_threshold"]
        self._max_fallback_rate = config["agents"]["monitor"]["max_fallback_rate"]
        self._block_on_anomaly = config["agents"]["monitor"]["block_on_anomaly"]
        self._tmc_target = config["thresholds"]["tmc_reduction_target"]

    def _check_prediction_coherence(
        self,
        predictions: Dict[str, float],
        causal_analyses: Dict[str, Any],
    ) -> List[str]:
        """
        Detect incoherence (possible LLM hallucination).

        For each intersection:
        - predicted = predictions[id]
        - analyzed = causal_analyses[id].traffic_forecast.congestion_level_adjusted
        - If |predicted - analyzed| > anomaly_threshold → anomaly

        Returns: list of intersection IDs with anomalies.
        """
        anomalies = []
        for iid, predicted in predictions.items():
            analysis = causal_analyses.get(iid)
            if analysis is None:
                continue

            forecast = analysis.get("traffic_forecast", {})
            analyzed = forecast.get("congestion_level_adjusted")
            if analyzed is None:
                continue

            diff = abs(float(predicted) - float(analyzed))
            if diff > self._anomaly_threshold:
                logger.warning(
                    "prediction_coherence_anomaly",
                    intersection=iid,
                    predicted=predicted,
                    analyzed=analyzed,
                    diff=round(diff, 4),
                )
                anomalies.append(iid)

        return anomalies

    def _check_fallback_rate(
        self, causal_analyses: Dict[str, Any]
    ) -> float:
        """
        Calculate proportion of analyses using fallback (is_fallback=True).
        """
        if not causal_analyses:
            return 0.0

        fallback_count = sum(
            1 for a in causal_analyses.values()
            if a.get("is_fallback", False)
        )
        rate = fallback_count / len(causal_analyses)

        if rate > self._max_fallback_rate:
            logger.warning(
                "high_fallback_rate",
                rate=round(rate, 4),
                threshold=self._max_fallback_rate,
            )

        return round(rate, 4)

    def _check_command_sanity(
        self, commands: Dict[str, Any]
    ) -> List[str]:
        """
        Validate that traffic light commands are physically possible.
        Returns: list of intersection IDs with invalid commands.
        """
        invalid = []
        for iid, cmd in commands.items():
            issues = []

            ext = cmd.get("green_extension_seconds", 0)
            if not (0 <= ext <= 60):
                issues.append(f"green_extension={ext} out of [0,60]")

            direction = cmd.get("priority_direction", "")
            if direction not in VALID_PRIORITY_DIRECTIONS:
                issues.append(f"invalid priority_direction={direction}")

            adj = cmd.get("cycle_adjustment_percent", 0)
            if not (0 <= adj <= 20):
                issues.append(f"cycle_adjustment={adj}% out of [0,20]")

            if issues:
                logger.warning(
                    "command_sanity_issue",
                    intersection=iid,
                    issues=issues,
                )
                invalid.append(iid)

        return invalid

    def _correct_invalid_commands(
        self, commands: Dict[str, Any], invalid_ids: List[str]
    ) -> Dict[str, Any]:
        """Correct invalid commands with safe defaults instead of blocking."""
        corrected = dict(commands)
        for iid in invalid_ids:
            if iid not in corrected:
                continue
            cmd = dict(corrected[iid])

            # Clamp green extension
            ext = cmd.get("green_extension_seconds", 0)
            cmd["green_extension_seconds"] = max(0, min(60, ext))

            # Fix priority direction
            if cmd.get("priority_direction") not in VALID_PRIORITY_DIRECTIONS:
                cmd["priority_direction"] = "rotacional"

            # Clamp cycle adjustment
            adj = cmd.get("cycle_adjustment_percent", 0)
            cmd["cycle_adjustment_percent"] = max(0, min(20, adj))

            corrected[iid] = cmd

        return corrected

    def _build_monitor_report(
        self,
        state: dict,
        anomalies: List[str],
        fallback_rate: float,
        invalid_commands: List[str],
        cycle_ms: int,
    ) -> Dict[str, Any]:
        """Build the final monitor validation report."""
        error_log = state.get("error_log", [])
        tmc_reduction = state.get("tmc_reduction_estimate", 0.0)

        # Count healthy agents (7 total minus critical errors)
        critical_errors = sum(
            1 for e in error_log
            if "failed" in e and "sensor" in e
        )
        agents_healthy = 7 - critical_errors

        # Hallucinations blocked (anomalies when block_on_anomaly is active)
        hallucinations_blocked = len(anomalies) if self._block_on_anomaly else 0

        # Determine cycle approval
        sensor_failed = state.get("sensor_status") == "failed"
        has_blocking_anomalies = self._block_on_anomaly and len(anomalies) > 0

        cycle_approved = not sensor_failed and not has_blocking_anomalies

        # Build warnings list
        warnings = []
        if fallback_rate > self._max_fallback_rate:
            warnings.append(
                f"Tasa de fallback alta: {fallback_rate:.1%} > {self._max_fallback_rate:.1%}"
            )
        if anomalies:
            warnings.append(
                f"Anomalías detectadas en: {', '.join(anomalies)}"
            )
        if invalid_commands:
            warnings.append(
                f"Comandos corregidos en: {', '.join(invalid_commands)}"
            )
        if tmc_reduction < self._tmc_target:
            warnings.append(
                f"Reducción TMC ({tmc_reduction:.1%}) por debajo de meta ({self._tmc_target:.1%})"
            )

        return {
            "cycle_id": state.get("cycle_id", ""),
            "cycle_duration_ms": cycle_ms,
            "agents_healthy": agents_healthy,
            "anomalies_detected": anomalies,
            "hallucinations_blocked": hallucinations_blocked,
            "fallback_rate": fallback_rate,
            "invalid_commands_corrected": invalid_commands,
            "tmc_reduction_estimate": tmc_reduction,
            "tmc_target_met": tmc_reduction >= self._tmc_target,
            "cycle_approved": cycle_approved,
            "warnings": warnings,
        }

    def run(self, state: dict) -> dict:
        """
        Main validation logic:
        1. Check prediction vs causal analysis coherence
        2. Calculate fallback rate
        3. Check and correct command sanity
        4. Calculate cycle duration
        5. Build monitor report
        6. Determine should_retry and cycle_approved
        """
        predictions = state.get("predictions", {})
        causal_analyses = state.get("causal_analyses", {})
        traffic_commands = state.get("traffic_commands", {})
        retry_count = state.get("retry_count", 0)

        # 1. Coherence check
        anomalies = self._check_prediction_coherence(predictions, causal_analyses)

        # 2. Fallback rate
        fallback_rate = self._check_fallback_rate(causal_analyses)

        # 3. Command sanity
        invalid_commands = self._check_command_sanity(traffic_commands)
        if invalid_commands:
            traffic_commands = self._correct_invalid_commands(
                traffic_commands, invalid_commands
            )

        # 4. Cycle duration
        timestamp_start = state.get("timestamp_start", "")
        try:
            start_dt = datetime.fromisoformat(timestamp_start)
            cycle_ms = int((datetime.now(start_dt.tzinfo) - start_dt).total_seconds() * 1000)
        except (ValueError, TypeError):
            cycle_ms = 0

        # 5. Build report
        monitor_report = self._build_monitor_report(
            state, anomalies, fallback_rate, invalid_commands, cycle_ms
        )

        cycle_approved = monitor_report["cycle_approved"]

        # 6. Determine retry
        should_retry = (
            self._block_on_anomaly
            and len(anomalies) > 0
            and retry_count < 2
        )

        if should_retry:
            logger.warning(
                "monitor_requesting_retry",
                anomalies=anomalies,
                retry_count=retry_count + 1,
            )

        logger.info(
            "monitor_complete",
            cycle_approved=cycle_approved,
            anomalies=len(anomalies),
            fallback_rate=fallback_rate,
            should_retry=should_retry,
            cycle_ms=cycle_ms,
        )

        return {
            **state,
            "traffic_commands": traffic_commands,
            "monitor_report": monitor_report,
            "anomalies_detected": anomalies,
            "cycle_approved": cycle_approved,
            "cycle_duration_ms": cycle_ms,
            "should_retry": should_retry,
            "retry_count": retry_count + (1 if should_retry else 0),
        }


def monitor_node(state: dict) -> dict:
    """LangGraph node function for the Monitor."""
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")
    agent = MonitorAgent(config)
    return agent.run(state)
