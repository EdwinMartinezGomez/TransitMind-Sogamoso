"""
TransitMind Sogamoso — LangGraph Orchestrator
===============================================
Main StateGraph definition with 7 agent nodes + finalize node.
Defines conditional edges (sensor failure → END, monitor → retry/finalize)
and the run_cycle() entry point for executing a complete decision cycle.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from src.layer3_agents.agent_causal_analyst import causal_analyst_node
from src.layer3_agents.agent_gan_simulator import gan_simulator_node
from src.layer3_agents.agent_monitor import monitor_node
from src.layer3_agents.agent_predictor import predictor_node
from src.layer3_agents.agent_route_planner import route_planner_node
from src.layer3_agents.agent_sensor import sensor_node
from src.layer3_agents.agent_traffic_coord import traffic_coordinator_node
from src.layer3_agents.graph_state import TransitMindState, get_initial_state
from src.shared.logger import get_logger
from src.shared.utils import get_project_root

logger = get_logger("layer3.orchestrator")

# Module-level compiled graph cache
_compiled_graph = None


def _ensure_output_dirs() -> Path:
    """Ensure output directories exist and return the decisions path."""
    root = get_project_root()
    decisions_dir = root / "data" / "layer3_outputs" / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / "data" / "layer3_outputs" / "monitor_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return decisions_dir


def finalize_node(state: dict) -> dict:
    """
    Final node: builds the final_decision JSON for Layer 4.
    Persists the decision to data/layer3_outputs/decisions/{cycle_id}.json.
    """
    cycle_id = state.get("cycle_id", "unknown")
    timestamp_start = state.get("timestamp_start", "")

    # Calculate final cycle duration
    try:
        start_dt = datetime.fromisoformat(timestamp_start)
        cycle_duration_ms = int(
            (datetime.now(start_dt.tzinfo) - start_dt).total_seconds() * 1000
        )
    except (ValueError, TypeError):
        cycle_duration_ms = state.get("cycle_duration_ms", 0)

    # Build per-intersection decisions
    causal_analyses = state.get("causal_analyses", {})
    traffic_commands = state.get("traffic_commands", {})
    route_plans = state.get("route_plans", {})
    predictions = state.get("predictions", {})

    decisions = []
    agent_trace = [
        "sensor",
        "predictor",
        "gan_simulator",
        "causal_analyst",
        "route_planner",
        "traffic_coordinator",
        "monitor",
    ]

    for iid, analysis in causal_analyses.items():
        causal_ctx = analysis.get("causal_context", {})
        forecast = analysis.get("traffic_forecast", {})
        recommendations = analysis.get("recommendations", {})

        # Get traffic command if available
        cmd = traffic_commands.get(iid, {})
        traffic_light_command = {
            "green_extension_seconds": cmd.get("green_extension_seconds", 0),
            "priority_direction": cmd.get("priority_direction", "rotacional"),
            "cycle_adjustment_percent": cmd.get("cycle_adjustment_percent", 0),
        }

        # Get routes
        active_routes = route_plans.get(iid, [])

        # Get citizen alert
        citizen_alert = recommendations.get("citizen_alert", "")

        decision = {
            "intersection_id": iid,
            "severity": causal_ctx.get("severity", "baja"),
            "congestion_level": forecast.get(
                "congestion_level_adjusted",
                predictions.get(iid, 0.0),
            ),
            "traffic_light_command": traffic_light_command,
            "active_routes": active_routes,
            "citizen_alert": citizen_alert,
            "agent_trace": agent_trace,
        }
        decisions.append(decision)

    # Build final decision
    monitor_report = state.get("monitor_report", {})
    tmc_reduction = state.get("tmc_reduction_estimate", 0.0)

    final_decision = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intersections_analyzed": len(decisions),
        "decisions": decisions,
        "global_tmc_reduction_percent": round(tmc_reduction * 100, 2),
        "monitor_report": {
            "anomalies_detected": len(monitor_report.get("anomalies_detected", [])),
            "hallucinations_blocked": monitor_report.get("hallucinations_blocked", 0),
            "cycle_duration_ms": cycle_duration_ms,
            "agents_healthy": monitor_report.get("agents_healthy", 7),
        },
    }

    # Persist to disk
    try:
        decisions_dir = _ensure_output_dirs()
        output_path = decisions_dir / f"{cycle_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_decision, f, indent=2, ensure_ascii=False, default=str)
        logger.info("decision_persisted", path=str(output_path))

        # Also persist monitor report
        root = get_project_root()
        reports_dir = root / "data" / "layer3_outputs" / "monitor_reports"
        report_path = reports_dir / f"{cycle_id}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(monitor_report, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("decision_persist_failed", error=str(e))

    logger.info(
        "cycle_finalized",
        cycle_id=cycle_id,
        intersections=len(decisions),
        tmc_reduction=f"{tmc_reduction:.1%}",
        approved=state.get("cycle_approved", False),
        duration_ms=cycle_duration_ms,
    )

    return {
        **state,
        "final_decision": final_decision,
        "cycle_duration_ms": cycle_duration_ms,
    }


def route_after_sensor(state: dict) -> str:
    """Routing after Sensor node."""
    if state.get("sensor_status") == "failed":
        return "end_error"
    return "predictor"


def route_after_monitor(state: dict) -> str:
    """Routing after Monitor node."""
    if state.get("should_retry") and state.get("retry_count", 0) < 2:
        return "retry_sensor"
    return "finalize"


def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph with all 7 agent nodes.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    workflow = StateGraph(TransitMindState)

    # Add nodes
    workflow.add_node("sensor", sensor_node)
    workflow.add_node("predictor", predictor_node)
    workflow.add_node("gan_simulator", gan_simulator_node)
    workflow.add_node("causal_analyst", causal_analyst_node)
    workflow.add_node("route_planner", route_planner_node)
    workflow.add_node("traffic_coordinator", traffic_coordinator_node)
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("finalize", finalize_node)

    # Entry point
    workflow.set_entry_point("sensor")

    # Conditional edge after sensor
    workflow.add_conditional_edges(
        "sensor",
        route_after_sensor,
        {
            "predictor": "predictor",
            "end_error": "finalize",
        },
    )

    # Linear edges: predictor → gan → causal → route → traffic → monitor
    workflow.add_edge("predictor", "gan_simulator")
    workflow.add_edge("gan_simulator", "causal_analyst")
    workflow.add_edge("causal_analyst", "route_planner")
    workflow.add_edge("route_planner", "traffic_coordinator")
    workflow.add_edge("traffic_coordinator", "monitor")

    # Conditional edge after monitor
    workflow.add_conditional_edges(
        "monitor",
        route_after_monitor,
        {
            "retry_sensor": "sensor",
            "finalize": "finalize",
        },
    )

    # Finalize → END
    workflow.add_edge("finalize", END)

    return workflow.compile()


def get_compiled_graph() -> Any:
    """Get or create the compiled graph (cached at module level)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("graph_compiled")
    return _compiled_graph


def run_cycle(
    scenario: str = "normal_weekday",
    intersections: Optional[List[str]] = None,
) -> dict:
    """
    Entry point to execute a complete decision cycle.

    Args:
        scenario: Scenario name (e.g. "market_day", "morning_fog").
        intersections: Optional list of intersection IDs. Uses all if None.

    Returns:
        final_decision dict with all decisions for Layer 4.
    """
    logger.info(
        "cycle_starting",
        scenario=scenario,
        intersections=intersections or "all",
    )

    try:
        # Create initial state
        initial_state = get_initial_state(
            scenario=scenario,
            intersections=intersections,
        )

        # Get compiled graph
        graph = get_compiled_graph()

        # Invoke the graph
        result_state = graph.invoke(initial_state)

        final_decision = result_state.get("final_decision")
        if final_decision:
            logger.info(
                "cycle_completed",
                cycle_id=final_decision.get("cycle_id"),
                decisions=final_decision.get("intersections_analyzed"),
                tmc_reduction=final_decision.get("global_tmc_reduction_percent"),
            )
            return final_decision
        else:
            logger.error("cycle_no_final_decision")
            return {
                "cycle_id": initial_state.get("cycle_id", "unknown"),
                "error": "No final decision produced",
                "error_log": result_state.get("error_log", []),
            }

    except Exception as e:
        logger.error("cycle_exception", error=str(e))
        return {
            "cycle_id": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
