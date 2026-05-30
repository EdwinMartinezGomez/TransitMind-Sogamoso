"""
TransitMind Sogamoso — Layer 4 API
=====================================
FastAPI application on port 8003.
Orchestrates alert engine, social graph, and bot communication.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.shared.logger import get_logger
from src.shared.utils import load_yaml_config
from src.layer4_bots.alert_engine import AlertEngine
from src.layer4_bots.social_graph import SocialGraphModule
from src.layer4_bots.whatsapp_handler import WhatsAppHandler

logger = get_logger("layer4.api")

# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="TransitMind Sogamoso — Capa 4: Comunicación Ciudadana",
    description=(
        "API de comunicación ciudadana: alertas priorizadas por grafo social, "
        "integración con Telegram y WhatsApp, y dashboard para la Secretaría de Movilidad."
    ),
    version="1.0.0",
)

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


class AlertTriggerRequest(BaseModel):
    """Request for POST /alert/trigger."""
    scenario: Optional[str] = Field(default=None, description="Scenario name")
    dry_run: bool = Field(default=False, description="If True, return plan without sending")


class UserRegisterRequest(BaseModel):
    """Request for POST /users/register."""
    user_id: str = Field(..., description="Anonymous user ID")
    phone_hash: Optional[str] = Field(default=None, description="Hashed phone number")
    corridors: List[str] = Field(default_factory=list, description="Subscribed corridors")
    role: str = Field(default="vecino", description="User role")
    is_seed: bool = Field(default=False, description="Seed node flag")
    peak_hours: List[int] = Field(default_factory=list, description="Peak activity hours")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str
    alert_engine_running: bool
    telegram_connected: bool
    whatsapp_connected: bool
    graph_nodes: int
    layer3_reachable: bool


# ============================================
# Global State
# ============================================

_config: Optional[dict] = None
_alert_engine: Optional[AlertEngine] = None
_graph: Optional[SocialGraphModule] = None
_wa_handler: Optional[WhatsAppHandler] = None
_engine_task: Optional[asyncio.Task] = None


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = load_yaml_config("layer4_config.yaml")
    return _config


def _get_engine() -> AlertEngine:
    global _alert_engine
    if _alert_engine is None:
        _alert_engine = AlertEngine(_get_config())
    return _alert_engine


def _get_graph() -> SocialGraphModule:
    global _graph
    if _graph is None:
        _graph = _get_engine().graph
    return _graph


def _get_wa() -> WhatsAppHandler:
    global _wa_handler
    if _wa_handler is None:
        _wa_handler = WhatsAppHandler(_get_config())
    return _wa_handler


# ============================================
# Startup
# ============================================


@app.on_event("startup")
async def startup_event():
    """Initialize all components and start AlertEngine in background."""
    global _engine_task

    config = _get_config()
    engine = _get_engine()
    graph = _get_graph()

    logger.info(
        "layer4_startup",
        graph_nodes=graph._graph.number_of_nodes(),
        graph_edges=graph._graph.number_of_edges(),
    )

    # Start alert engine in background
    async def _send_callback(plan: dict):
        """Callback: log alert plans. Bots handle their own sending."""
        n = len(plan.get("alerts_to_send", []))
        logger.info("alert_callback", alerts=n, cycle_id=plan.get("cycle_id"))

    _engine_task = asyncio.create_task(engine.run_forever(_send_callback))
    logger.info("layer4_startup_complete", port=config.get("layer4_api", {}).get("port", 8003))


@app.on_event("shutdown")
async def shutdown_event():
    """Stop alert engine on shutdown."""
    global _engine_task
    engine = _get_engine()
    engine.stop()
    if _engine_task:
        _engine_task.cancel()
    logger.info("layer4_shutdown")


# ============================================
# Endpoints
# ============================================


@app.post("/alert/trigger")
async def trigger_alert(request: AlertTriggerRequest):
    """
    Force alert emission from latest Layer 3 cycle.
    If dry_run=True: return the alert plan without sending.
    """
    engine = _get_engine()
    plan = await engine.process_cycle()

    if not plan.get("alerts_to_send"):
        return {"message": "No alerts to send", "plan": plan}

    if request.dry_run:
        return {"message": "Dry run — alerts not sent", "plan": plan}

    # In non-dry-run, the plan was already processed and logged
    return {"message": f"Alert plan processed: {len(plan['alerts_to_send'])} alerts", "plan": plan}


@app.get("/alert/latest")
async def get_latest_alert():
    """Return the last executed alert plan."""
    engine = _get_engine()
    plan = engine.get_latest_plan()
    if plan is None:
        raise HTTPException(status_code=404, detail="No alert plans executed yet")
    return plan


@app.post("/users/register")
async def register_user(request: UserRegisterRequest):
    """Register a user in the SocialGraph."""
    graph = _get_graph()
    graph.register_user({
        "user_id": request.user_id,
        "corridors": request.corridors,
        "peak_hours": request.peak_hours,
        "query_count": 0,
        "is_seed": request.is_seed,
        "role": request.role,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "last_active": datetime.now(timezone.utc).isoformat(),
    })
    return {
        "message": f"User {request.user_id} registered",
        "graph_nodes": graph._graph.number_of_nodes(),
    }


@app.get("/graph/summary")
async def get_graph_summary():
    """Return SocialGraphModule.get_graph_summary()."""
    graph = _get_graph()
    return graph.get_graph_summary()


@app.post("/graph/rebuild")
async def rebuild_graph(background_tasks: BackgroundTasks):
    """Force graph reconstruction in background."""
    graph = _get_graph()

    def _rebuild():
        graph.rebuild_edges()
        graph.compute_centrality()
        logger.info("graph_rebuild_complete")

    background_tasks.add_task(_rebuild)
    return {"message": "Graph rebuild started in background"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Layer 4 health check + Layer 3 connectivity."""
    config = _get_config()
    engine = _get_engine()
    graph = _get_graph()

    # Check Layer 3
    layer3_url = config.get("layer3_api", {}).get("base_url", "http://localhost:8002")
    layer3_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{layer3_url}/health")
            layer3_ok = resp.status_code == 200
    except Exception:
        pass

    tg_enabled = config.get("telegram", {}).get("enabled", False)
    wa_enabled = config.get("whatsapp", {}).get("enabled", False)

    status = "ok" if layer3_ok else "degraded"

    return HealthResponse(
        status=status,
        alert_engine_running=engine._running,
        telegram_connected=tg_enabled,
        whatsapp_connected=wa_enabled,
        graph_nodes=graph._graph.number_of_nodes(),
        layer3_reachable=layer3_ok,
    )


# ============================================
# WhatsApp Webhook Endpoints
# ============================================


@app.get("/webhook/whatsapp")
async def whatsapp_webhook_verify(
    hub_mode: str = Query(alias="hub.mode", default=""),
    hub_token: str = Query(alias="hub.verify_token", default=""),
    hub_challenge: str = Query(alias="hub.challenge", default=""),
):
    """WhatsApp webhook verification (GET)."""
    wa = _get_wa()
    result = await wa.verify_webhook(hub_mode, hub_token, hub_challenge)
    if result:
        return int(result) if result.isdigit() else result
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook_receive(request: Request):
    """WhatsApp webhook for incoming messages (POST)."""
    wa = _get_wa()
    body = await request.json()
    return await wa.handle_webhook(body)
