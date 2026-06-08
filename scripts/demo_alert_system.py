#!/usr/bin/env python3
"""
TransitMind Sogamoso — Demo del Sistema de Alertas en Tiempo Real
===================================================================
Simulates the full alert flow: user ingestion → queries → alerts.
Designed to run alongside the Streamlit dashboard.

Usage:
    python scripts/demo_alert_system.py [--cycles N] [--seed N]
"""

import sys
import asyncio
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer4_bots.social_graph import SocialGraphModule
from src.layer4_bots.message_formatter import MessageFormatter
from src.layer4_bots.alert_engine import AlertEngine
from src.shared.utils import load_yaml_config, get_project_root
from src.shared.constants import INTERSECTIONS

DEMO_DECISIONS_BY_SCENARIO = {
    "market_day": [
        {
            "intersection_id": "carrera_11_norte",
            "severity": "alta",
            "congestion_level": 0.82,
            "traffic_light_command": {
                "green_extension_seconds": 30,
                "priority_direction": "norte_sur",
                "cycle_adjustment_percent": 18,
            },
            "active_routes": ["Avenida Industrial", "Calle 14"],
            "citizen_alert": "Congestión alta en Carrera 11 Norte por mercado campesino.",
            "agent_trace": [
                "sensor", "predictor", "gan_simulator",
                "causal_analyst", "route_planner",
                "traffic_coordinator", "monitor",
            ],
        },
        {
            "intersection_id": "calle_14_centro_historico",
            "severity": "alta",
            "congestion_level": 0.76,
            "traffic_light_command": {
                "green_extension_seconds": 22,
                "priority_direction": "este_oeste",
                "cycle_adjustment_percent": 12,
            },
            "active_routes": ["Carrera 9", "Variante Morca"],
            "citizen_alert": "Tráfico denso en Calle 14 Centro por flujo de mercado.",
            "agent_trace": [
                "sensor", "predictor", "gan_simulator",
                "causal_analyst", "route_planner",
                "traffic_coordinator", "monitor",
            ],
        },
        {
            "intersection_id": "carrera_11_sur",
            "severity": "media",
            "congestion_level": 0.55,
            "traffic_light_command": {
                "green_extension_seconds": 10,
                "priority_direction": "sur_norte",
                "cycle_adjustment_percent": 5,
            },
            "active_routes": ["Calle 16"],
            "citizen_alert": "Tráfico moderado en Carrera 11 Sur.",
            "agent_trace": ["sensor", "predictor", "monitor"],
        },
        {
            "intersection_id": "av_castellana_entrada",
            "severity": "media",
            "congestion_level": 0.48,
            "traffic_light_command": {
                "green_extension_seconds": 8,
                "priority_direction": "este_oeste",
                "cycle_adjustment_percent": 4,
            },
            "active_routes": [],
            "citizen_alert": "Flujo normal con algo de carga en Av. Castellana entrada.",
            "agent_trace": ["sensor", "predictor", "monitor"],
        },
        {
            "intersection_id": "av_castellana_salida",
            "severity": "baja",
            "congestion_level": 0.22,
            "traffic_light_command": {
                "green_extension_seconds": 0,
                "priority_direction": "",
                "cycle_adjustment_percent": 0,
            },
            "active_routes": [],
            "citizen_alert": "Tráfico normal en Av. Castellana salida.",
            "agent_trace": ["sensor", "monitor"],
        },
        {
            "intersection_id": "acceso_morca",
            "severity": "baja",
            "congestion_level": 0.18,
            "traffic_light_command": {
                "green_extension_seconds": 0,
                "priority_direction": "",
                "cycle_adjustment_percent": 0,
            },
            "active_routes": [],
            "citizen_alert": "Sin congestión en Acceso Morca.",
            "agent_trace": ["sensor", "monitor"],
        },
    ],
    "morning_fog": [
        {
            "intersection_id": "acceso_morca",
            "severity": "alta",
            "congestion_level": 0.79,
            "traffic_light_command": {
                "green_extension_seconds": 28,
                "priority_direction": "norte_sur",
                "cycle_adjustment_percent": 15,
            },
            "active_routes": [
                "Variante Sogamoso-Duitama", "Carretera alterna",
            ],
            "citizen_alert": "Niebla densa en acceso Morca. Visibilidad reducida.",
            "agent_trace": [
                "sensor", "predictor", "gan_simulator",
                "causal_analyst", "route_planner",
                "traffic_coordinator", "monitor",
            ],
        },
        {
            "intersection_id": "av_castellana_entrada",
            "severity": "media",
            "congestion_level": 0.52,
            "traffic_light_command": {
                "green_extension_seconds": 12,
                "priority_direction": "este_oeste",
                "cycle_adjustment_percent": 6,
            },
            "active_routes": ["Calle 14"],
            "citizen_alert": "Reducción de velocidad por niebla en Av. Castellana.",
            "agent_trace": ["sensor", "predictor", "traffic_coordinator", "monitor"],
        },
        {
            "intersection_id": "carrera_11_norte",
            "severity": "baja",
            "congestion_level": 0.25,
            "traffic_light_command": {
                "green_extension_seconds": 0,
                "priority_direction": "",
                "cycle_adjustment_percent": 0,
            },
            "active_routes": [],
            "citizen_alert": "Tráfico normal en Carrera 11 Norte.",
            "agent_trace": ["sensor", "monitor"],
        },
        {
            "intersection_id": "carrera_11_sur",
            "severity": "baja",
            "congestion_level": 0.20,
            "traffic_light_command": {
                "green_extension_seconds": 0,
                "priority_direction": "",
                "cycle_adjustment_percent": 0,
            },
            "active_routes": [],
            "citizen_alert": "Sin novedades en Carrera 11 Sur.",
            "agent_trace": ["sensor", "monitor"],
        },
        {
            "intersection_id": "av_castellana_salida",
            "severity": "baja",
            "congestion_level": 0.15,
            "traffic_light_command": {
                "green_extension_seconds": 0,
                "priority_direction": "",
                "cycle_adjustment_percent": 0,
            },
            "active_routes": [],
            "citizen_alert": "Flujo libre en Av. Castellana salida.",
            "agent_trace": ["sensor", "monitor"],
        },
        {
            "intersection_id": "calle_14_centro_historico",
            "severity": "baja",
            "congestion_level": 0.19,
            "traffic_light_command": {
                "green_extension_seconds": 0,
                "priority_direction": "",
                "cycle_adjustment_percent": 0,
            },
            "active_routes": [],
            "citizen_alert": "Todo tranquilo en Calle 14 Centro.",
            "agent_trace": ["sensor", "monitor"],
        },
    ],
    "coliseo_event": [
        {
            "intersection_id": "carrera_11_norte",
            "severity": "critica",
            "congestion_level": 0.91,
            "traffic_light_command": {
                "green_extension_seconds": 42,
                "priority_direction": "norte_sur",
                "cycle_adjustment_percent": 20,
            },
            "active_routes": ["Avenida Industrial"],
            "citizen_alert": "Congestión crítica por evento en Coliseo. Evite Carrera 11.",
            "agent_trace": [
                "sensor", "predictor", "gan_simulator",
                "causal_analyst", "route_planner",
                "traffic_coordinator", "monitor",
            ],
        },
        {
            "intersection_id": "calle_14_centro_historico",
            "severity": "alta",
            "congestion_level": 0.84,
            "traffic_light_command": {
                "green_extension_seconds": 32,
                "priority_direction": "este_oeste",
                "cycle_adjustment_percent": 17,
            },
            "active_routes": ["Carrera 9", "Calle 16"],
            "citizen_alert": "Alto tráfico en Calle 14 por evento masivo.",
            "agent_trace": [
                "sensor", "predictor", "gan_simulator",
                "causal_analyst", "route_planner",
                "traffic_coordinator", "monitor",
            ],
        },
        {
            "intersection_id": "carrera_11_sur",
            "severity": "media",
            "congestion_level": 0.62,
            "traffic_light_command": {
                "green_extension_seconds": 15,
                "priority_direction": "sur_norte",
                "cycle_adjustment_percent": 8,
            },
            "active_routes": ["Calle 16"],
            "citizen_alert": "Tráfico moderado en Carrera 11 Sur por dispersión del evento.",
            "agent_trace": ["sensor", "predictor", "traffic_coordinator", "monitor"],
        },
        {
            "intersection_id": "av_castellana_entrada",
            "severity": "media",
            "congestion_level": 0.50,
            "traffic_light_command": {
                "green_extension_seconds": 10,
                "priority_direction": "este_oeste",
                "cycle_adjustment_percent": 5,
            },
            "active_routes": [],
            "citizen_alert": "Algo de congestión en Av. Castellana por tráfico del evento.",
            "agent_trace": ["sensor", "predictor", "monitor"],
        },
        {
            "intersection_id": "av_castellana_salida",
            "severity": "baja",
            "congestion_level": 0.28,
            "traffic_light_command": {
                "green_extension_seconds": 0,
                "priority_direction": "",
                "cycle_adjustment_percent": 0,
            },
            "active_routes": [],
            "citizen_alert": "Tráfico normal en Av. Castellana salida.",
            "agent_trace": ["sensor", "monitor"],
        },
        {
            "intersection_id": "acceso_morca",
            "severity": "baja",
            "congestion_level": 0.15,
            "traffic_light_command": {
                "green_extension_seconds": 0,
                "priority_direction": "",
                "cycle_adjustment_percent": 0,
            },
            "active_routes": [],
            "citizen_alert": "Sin novedades en Acceso Morca.",
            "agent_trace": ["sensor", "monitor"],
        },
    ],
}


def _build_cycle_decision(scenario: str, cycle_num: int) -> dict:
    """Build a mock final_decision JSON for the given scenario."""
    decisions = DEMO_DECISIONS_BY_SCENARIO.get(scenario, [])
    now = datetime.now(timezone.utc)

    # Compute a synthetic TMC reduction
    avg_cong = sum(d["congestion_level"] for d in decisions) / len(decisions) if decisions else 0
    tmc = max(15, min(40, (avg_cong * 45) + random.uniform(-5, 5)))

    return {
        "cycle_id": f"demo_{cycle_num:04d}_{scenario}",
        "timestamp": now.isoformat(),
        "decisions": decisions,
        "global_tmc_reduction_percent": round(tmc, 1),
        "monitor_report": {
            "agents_healthy": 7,
            "anomalies_detected": 0,
            "cycle_duration_ms": random.randint(800, 2500),
        },
    }


def _save_mock_decision(decision: dict):
    """Save mock decision so the AlertEngine/dashboard can read it."""
    out_dir = get_project_root() / "data" / "layer3_outputs" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"demo_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False, default=str)
    return path


async def run_demo_cycle(
    engine: AlertEngine, scenario: str, cycle_num: int,
) -> dict:
    """Execute one demo cycle."""
    decision = _build_cycle_decision(scenario, cycle_num)

    # Save so dashboard can see it
    saved_path = _save_mock_decision(decision)

    # Build alert plan using AlertEngine's logic directly
    sg = engine.graph
    formatter = engine.formatter
    all_users = sg.get_all_user_ids()

    from src.layer4_bots.message_formatter import SEVERITY_ORDER

    alerts_to_send = []
    skipped = 0

    for dec in decision.get("decisions", []):
        sev = dec.get("severity", "baja")
        if SEVERITY_ORDER.get(sev, 0) < 1:
            skipped += 1
            continue

        iid = dec.get("intersection_id", "")
        messages = {
            "whatsapp": formatter.format_decision(dec, "whatsapp"),
            "telegram_citizen": formatter.format_decision(dec, "telegram_citizen"),
            "dashboard": formatter.format_decision(dec, "dashboard"),
        }

        affected = [iid] + dec.get("active_routes", [])
        alert_order = sg.get_alert_order(affected, all_users)

        alerts_to_send.append({
            "intersection_id": iid,
            "severity": sev,
            "messages": messages,
            "first_wave_users": alert_order["first_wave"],
            "broadcast_users": alert_order["broadcast"],
            "broadcast_delay_minutes": alert_order["broadcast_delay_minutes"],
            "expected_coverage_pct": alert_order["expected_coverage_pct"],
            "graph_stats": alert_order["graph_stats"],
        })

    plan = {
        "cycle_id": decision["cycle_id"],
        "alerts_to_send": alerts_to_send,
        "skipped": skipped,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save alert log
    alerts_dir = get_project_root() / "data" / "layer4_outputs" / "alerts_sent"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alert_path = alerts_dir / f"demo_{ts}.json"

    log_plan = {**plan}
    for a in log_plan.get("alerts_to_send", []):
        a["first_wave_count"] = len(a.get("first_wave_users", []))
        a["broadcast_count"] = len(a.get("broadcast_users", []))

    with open(alert_path, "w", encoding="utf-8") as f:
        json.dump(log_plan, f, indent=2, ensure_ascii=False, default=str)

    return plan


async def main_async(args):
    config = load_yaml_config("layer4_config.yaml")

    # Keep TG/WA disabled locally — we'll trigger via the running Layer 4 API instead
    config["telegram"]["enabled"] = False
    config["whatsapp"]["enabled"] = False

    engine = AlertEngine(config)
    sg = engine.graph

    layer4_url = config.get("layer4_api", {}).get("base_url", "http://localhost:8003")
    # Fallback: build from host/port
    if layer4_url == "http://localhost:8003":
        port = config.get("layer4_api", {}).get("port", 8003)
        layer4_url = f"http://localhost:{port}"

    scenarios = ["market_day", "morning_fog", "coliseo_event"]
    rng = random.Random(args.seed)

    print(f"\n{'=' * 60}")
    print(f"  TransitMind Sogamoso -- Demo de Alertas en Tiempo Real")
    print(f"  Ciclos: {args.cycles} | Usuarios en grafo: {sg._graph.number_of_nodes()}")
    print(f"  Alertas se envian via Layer 4 API -> Telegram bot")
    print(f"  Layer 4: {layer4_url}")
    print(f"{'=' * 60}\n")

    import httpx

    for cycle_num in range(1, args.cycles + 1):
        scenario = scenarios[(cycle_num - 1) % len(scenarios)]
        print(f"\n--- Ciclo {cycle_num}/{args.cycles} -- Escenario: {scenario} ---")

        # 1. Build + save decision file (dashboard reads from disk)
        result = await run_demo_cycle(engine, scenario, cycle_num)

        for alert in result.get("alerts_to_send", []):
            iid = alert.get("intersection_id", "")
            sev = alert.get("severity", "")
            n_fw = len(alert.get("first_wave_users", []))
            n_bc = len(alert.get("broadcast_users", []))
            msg_tg = alert.get("messages", {}).get("telegram_citizen", "")

            # Safe print for Windows console
            def safe_print(text):
                print(str(text).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
                
            safe_print(f"  [ALERTA] {iid} [{sev.upper()}]")
            safe_print(f"     1ra ola: {n_fw} propagadores | Broadcast: {n_bc}")
            safe_print(f"     TG: {msg_tg[:80]}...")

        skipped = result.get("skipped", 0)
        if skipped > 0:
            print(f"  [SKIP] {skipped} intersecciones omitidas (baja severidad)")

        # 2. POST to Layer 4 API -> triggers real TG dispatch
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{layer4_url}/alert/trigger",
                    json={"dry_run": False},
                )
                if resp.status_code == 200:
                    api_result = resp.json()
                    n_api = len(api_result.get("plan", {}).get("alerts_to_send", []))
                    print(f"  [API] Layer 4 dispatch OK -> {n_api} alertas procesadas")
                else:
                    print(f"  [API] HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"  [API] No se pudo conectar a Layer 4: {str(e)[:80]}")

        if cycle_num < args.cycles:
            delay = rng.uniform(3, 8)
            print(f"\n  Proximo ciclo en {delay:.1f}s...")
            await asyncio.sleep(delay)

    print(f"\n{'=' * 60}")
    print(f"  Demo completado. Revisa Telegram y el dashboard.")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Demo sistema de alertas TransitMind",
    )
    parser.add_argument(
        "--cycles", type=int, default=5,
        help="Numero de ciclos de demo (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

