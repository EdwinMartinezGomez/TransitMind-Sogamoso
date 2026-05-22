"""
TransitMind Sogamoso — Layer 3 API
====================================
FastAPI application exposing the multi-agent decision cycle
on port 8002 for consumption by Layer 4 (bots + dashboard).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.shared.logger import get_logger
from src.shared.utils import get_project_root

logger = get_logger("layer3.api")

# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="TransitMind Sogamoso — Capa 3: Multi-Agentes LangGraph",
    description=(
        "Ecosistema de 7 agentes de IA autónomos orquestados por LangGraph "
        "para decisiones operacionales de tráfico en tiempo real en Sogamoso, Colombia."
    ),
    version="1.0.0",
)

# CORS middleware for Layer 4 dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Request / Response Models
# ============================================


class RunCycleRequest(BaseModel):
    """Request for POST /run-cycle."""

    scenario: Optional[str] = Field(
        default="normal_weekday",
        description="Scenario name (e.g. 'market_day', 'morning_fog')",
    )
    intersections: Optional[List[str]] = Field(
        default=None,
        description="List of intersection IDs. None = all 6 intersections.",
    )
    async_mode: bool = Field(
        default=False,
        description="If True, returns cycle_id immediately and runs in background.",
    )


class CycleStatusResponse(BaseModel):
    """Response for GET /cycle/{cycle_id}."""

    cycle_id: str
    status: str  # "running" | "completed" | "failed"
    result: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str  # "ok" | "degraded"
    graph_compiled: bool
    layer1_reachable: bool
    layer2_reachable: bool
    last_cycle_id: Optional[str] = None
    last_cycle_timestamp: Optional[str] = None


class AgentStatus(BaseModel):
    """Status of a single agent."""

    name: str
    enabled: bool
    last_run_ms: Optional[int] = None


class AgentsStatusResponse(BaseModel):
    """Response for GET /agents/status."""

    agents: List[AgentStatus]


# ============================================
# In-memory cycle tracking (for async mode)
# ============================================

_running_cycles: Dict[str, str] = {}  # cycle_id → "running" | "completed" | "failed"


# ============================================
# Startup Event
# ============================================


@app.on_event("startup")
async def startup_event():
    """Pre-compile the graph on startup to reduce first-cycle latency."""
    try:
        from src.layer3_agents.orchestrator import get_compiled_graph

        get_compiled_graph()
        logger.info("startup_graph_compiled")
    except Exception as e:
        logger.error("startup_graph_compilation_failed", error=str(e))


# ============================================
# Helper Functions
# ============================================


def _get_decisions_dir() -> Path:
    """Get the decisions output directory."""
    return get_project_root() / "data" / "layer3_outputs" / "decisions"


def _get_latest_decision_file() -> Optional[Path]:
    """Get the most recent decision file."""
    decisions_dir = _get_decisions_dir()
    if not decisions_dir.exists():
        return None

    json_files = list(decisions_dir.glob("*.json"))
    if not json_files:
        return None

    return max(json_files, key=lambda f: f.stat().st_mtime)


async def _check_service_health(url: str, timeout: float = 5.0) -> bool:
    """Check if a service is reachable via GET /health."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except Exception:
        return False


# ============================================
# Endpoints
# ============================================


@app.post("/run-cycle", response_model=None)
async def run_cycle_endpoint(
    request: RunCycleRequest,
    background_tasks: BackgroundTasks,
):
    """
    Execute a complete decision cycle with all 7 agents.

    If async_mode=False: waits for result (can take up to 120s).
    If async_mode=True: returns cycle_id immediately and runs in background.
    """
    from src.layer3_agents.orchestrator import run_cycle

    if request.async_mode:
        # Generate a cycle_id and run in background
        import uuid

        cycle_id = str(uuid.uuid4())
        _running_cycles[cycle_id] = "running"

        def _run_in_background():
            try:
                result = run_cycle(
                    scenario=request.scenario or "normal_weekday",
                    intersections=request.intersections,
                )
                _running_cycles[cycle_id] = "completed"
            except Exception as e:
                _running_cycles[cycle_id] = "failed"
                logger.error("async_cycle_failed", cycle_id=cycle_id, error=str(e))

        background_tasks.add_task(_run_in_background)
        return {"cycle_id": cycle_id, "status": "running", "message": "Cycle started in background"}
    else:
        # Synchronous execution
        result = run_cycle(
            scenario=request.scenario or "normal_weekday",
            intersections=request.intersections,
        )
        return result


@app.get("/cycle/{cycle_id}", response_model=CycleStatusResponse)
async def get_cycle_status(cycle_id: str):
    """
    Query the result of a cycle by ID.
    Reads from data/layer3_outputs/decisions/{cycle_id}.json.
    """
    # Check in-memory status first (for async cycles)
    if cycle_id in _running_cycles:
        status = _running_cycles[cycle_id]
        if status == "running":
            return CycleStatusResponse(
                cycle_id=cycle_id, status="running", result=None
            )

    # Try to read from disk
    decisions_dir = _get_decisions_dir()
    file_path = decisions_dir / f"{cycle_id}.json"

    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        return CycleStatusResponse(
            cycle_id=cycle_id, status="completed", result=result
        )

    raise HTTPException(
        status_code=404,
        detail=f"Cycle {cycle_id} not found",
    )


@app.get("/latest-decision")
async def get_latest_decision():
    """
    Return the most recent decision saved in data/layer3_outputs/decisions/.
    Useful for Layer 4 to query current state without running a new cycle.
    """
    latest_file = _get_latest_decision_file()
    if latest_file is None:
        raise HTTPException(
            status_code=404,
            detail="No decisions found. Run a cycle first.",
        )

    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check including Layer 1 and Layer 2 connectivity.
    """
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")
    layer1_url = config["layer1_api"]["base_url"]
    layer2_url = config["layer2_api"]["base_url"]

    # Check graph compilation
    graph_compiled = False
    try:
        from src.layer3_agents.orchestrator import get_compiled_graph

        get_compiled_graph()
        graph_compiled = True
    except Exception:
        pass

    # Check Layer 1 and Layer 2 connectivity
    layer1_ok = await _check_service_health(layer1_url)
    layer2_ok = await _check_service_health(layer2_url)

    # Get last cycle info
    latest_file = _get_latest_decision_file()
    last_cycle_id = None
    last_cycle_timestamp = None
    if latest_file:
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                last_cycle_id = data.get("cycle_id")
                last_cycle_timestamp = data.get("timestamp")
        except Exception:
            pass

    status = "ok" if (graph_compiled and layer1_ok) else "degraded"

    return HealthResponse(
        status=status,
        graph_compiled=graph_compiled,
        layer1_reachable=layer1_ok,
        layer2_reachable=layer2_ok,
        last_cycle_id=last_cycle_id,
        last_cycle_timestamp=last_cycle_timestamp,
    )


@app.get("/agents/status", response_model=AgentsStatusResponse)
async def get_agents_status():
    """
    Return status of each of the 7 agents.
    """
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")

    agent_names = [
        ("sensor", "Agente Sensor"),
        ("predictor", "Agente Predictor"),
        ("gan_simulator", "Agente GAN Simulator"),
        ("causal_analyst", "Agente Analista Causal"),
        ("route_planner", "Agente Planificador de Rutas"),
        ("traffic_coordinator", "Coordinador de Semáforos"),
        ("monitor", "Agente Monitor"),
    ]

    agents = []
    for key, default_name in agent_names:
        agent_cfg = config.get("agents", {}).get(key, {})
        agents.append(
            AgentStatus(
                name=agent_cfg.get("name", default_name),
                enabled=agent_cfg.get("enabled", False),
                last_run_ms=None,  # Populated after first cycle
            )
        )

    return AgentsStatusResponse(agents=agents)
