"""
TransitMind Sogamoso — LangGraph Shared State
===============================================
TypedDict definition for the LangGraph StateGraph.
This is the central data structure passed between all agent nodes.
Each agent reads from and writes to this state during a decision cycle.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from src.shared.schemas import CausalAnalysisResult


class TransitMindState(TypedDict, total=False):
    """
    Shared state for the LangGraph StateGraph.

    Every agent node receives this state and returns an updated copy.
    LangGraph merges returned dicts into the running state automatically.
    All fields are optional (total=False) except those with no default.
    """

    # --- Cycle metadata ---
    cycle_id: str                              # UUID unique per decision cycle
    timestamp_start: str                       # ISO8601 cycle start
    scenario: Optional[str]                    # Active scenario (e.g. "market_day")
    intersections_to_analyze: List[str]        # List of intersection IDs

    # --- Agent Sensor output ---
    sensor_data: Optional[Dict[str, Any]]      # {intersection_id: GenerateResponse}
    sensor_status: Optional[str]               # "ok" | "partial" | "failed"

    # --- Agent Predictor output ---
    predictions: Optional[Dict[str, float]]            # {intersection_id: congestion}
    high_congestion_intersections: Optional[List[str]]  # Above threshold
    predictor_status: Optional[str]

    # --- Agent GAN Simulator output ---
    gan_scenarios: Optional[Dict[str, Any]]    # {intersection_id: GenerateResponse}
    active_scenario: Optional[str]             # Inferred scenario name

    # --- Agent Causal Analyst output ---
    causal_analyses: Optional[Dict[str, Any]]  # {intersection_id: CausalAnalysisResult dict}
    causal_status: Optional[str]

    # --- Agent Route Planner output ---
    route_plans: Optional[Dict[str, List[str]]]  # {intersection_id: [route1, ...]}
    reroute_needed: Optional[bool]

    # --- Agent Traffic Coordinator output ---
    traffic_commands: Optional[Dict[str, Any]]   # {intersection_id: TrafficCommand}
    commands_applied: Optional[bool]

    # --- Agent Monitor output ---
    monitor_report: Optional[Dict[str, Any]]     # Validation report
    anomalies_detected: Optional[List[str]]      # Intersection IDs with anomalies
    cycle_approved: Optional[bool]               # True if Monitor approves

    # --- Final cycle result ---
    final_decision: Optional[Dict[str, Any]]     # JSON for Layer 4
    tmc_reduction_estimate: Optional[float]      # % TMC reduction estimate
    cycle_duration_ms: Optional[int]

    # --- Flow control ---
    error_log: List[str]                         # Accumulated cycle errors
    retry_count: int                             # Retry counter
    should_retry: bool                           # Monitor can request retry


def get_initial_state(
    cycle_id: Optional[str] = None,
    scenario: Optional[str] = "normal_weekday",
    intersections: Optional[List[str]] = None,
) -> TransitMindState:
    """
    Create an initial TransitMindState for a new decision cycle.

    Args:
        cycle_id: Optional UUID. Generated if not provided.
        scenario: Scenario name (default: "normal_weekday").
        intersections: List of intersection IDs. Uses all if None.

    Returns:
        A fully initialized TransitMindState ready for the graph.
    """
    from src.shared.constants import INTERSECTIONS

    return TransitMindState(
        # Metadata
        cycle_id=cycle_id or str(uuid.uuid4()),
        timestamp_start=datetime.now(timezone.utc).isoformat(),
        scenario=scenario,
        intersections_to_analyze=intersections or list(INTERSECTIONS),
        # Sensor
        sensor_data=None,
        sensor_status=None,
        # Predictor
        predictions=None,
        high_congestion_intersections=None,
        predictor_status=None,
        # GAN Simulator
        gan_scenarios=None,
        active_scenario=None,
        # Causal Analyst
        causal_analyses=None,
        causal_status=None,
        # Route Planner
        route_plans=None,
        reroute_needed=None,
        # Traffic Coordinator
        traffic_commands=None,
        commands_applied=None,
        # Monitor
        monitor_report=None,
        anomalies_detected=None,
        cycle_approved=None,
        # Final
        final_decision=None,
        tmc_reduction_estimate=None,
        cycle_duration_ms=None,
        # Control flow
        error_log=[],
        retry_count=0,
        should_retry=False,
    )
