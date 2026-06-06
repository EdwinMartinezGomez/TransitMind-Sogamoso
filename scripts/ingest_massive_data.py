#!/usr/bin/env python3
"""
TransitMind Sogamoso — Script de Ingesta Masiva de Datos
=========================================================
Populates the SocialGraphModule with synthetic users to test
the social graph, centrality computation, and SIR simulation.

Usage:
    python scripts/ingest_massive_data.py [--users N] [--queries M] [--scenario SCENARIO]
    python scripts/ingest_massive_data.py --users 500 --scenario market --export-graph
"""

import sys
import argparse
import random
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer4_bots.social_graph import SocialGraphModule
from src.shared.utils import load_yaml_config, get_project_root
from src.shared.constants import INTERSECTIONS
from src.shared.logger import get_logger

logger = get_logger("scripts.ingest_massive")

# ---- Scenario configurations ----

SCENARIOS = {
    "normal": {
        "description": "Día laboral normal",
        "peak_hour_bias": 0.5,
        "corridor_distribution": {
            "carrera_11_norte": 0.25, "carrera_11_sur": 0.20,
            "av_castellana_entrada": 0.20, "av_castellana_salida": 0.15,
            "calle_14_centro_historico": 0.12, "acceso_morca": 0.08,
        },
        "role_distribution": {
            "vecino": 0.70, "conductor": 0.20,
            "lider_barrio": 0.07, "despachador": 0.03,
        },
    },
    "market": {
        "description": "Día de mercado campesino",
        "peak_hour_bias": 0.7,
        "corridor_distribution": {
            "carrera_11_norte": 0.30, "calle_14_centro_historico": 0.28,
            "carrera_11_sur": 0.20, "av_castellana_entrada": 0.12,
            "av_castellana_salida": 0.07, "acceso_morca": 0.03,
        },
        "role_distribution": {
            "vecino": 0.60, "conductor": 0.25,
            "lider_barrio": 0.10, "despachador": 0.05,
        },
    },
    "fog": {
        "description": "Mañana con niebla del Chicamocha",
        "peak_hour_bias": 0.8,
        "corridor_distribution": {
            "acceso_morca": 0.35, "av_castellana_entrada": 0.30,
            "carrera_11_norte": 0.20, "carrera_11_sur": 0.10,
            "calle_14_centro_historico": 0.03, "av_castellana_salida": 0.02,
        },
        "role_distribution": {
            "vecino": 0.55, "conductor": 0.30,
            "lider_barrio": 0.10, "despachador": 0.05,
        },
    },
    "event": {
        "description": "Evento en Coliseo Olímpico",
        "peak_hour_bias": 0.6,
        "corridor_distribution": {
            "carrera_11_norte": 0.35, "calle_14_centro_historico": 0.30,
            "carrera_11_sur": 0.20, "av_castellana_entrada": 0.10,
            "av_castellana_salida": 0.03, "acceso_morca": 0.02,
        },
        "role_distribution": {
            "vecino": 0.75, "conductor": 0.15,
            "lider_barrio": 0.07, "despachador": 0.03,
        },
    },
    "stress": {
        "description": "Test de estrés — máxima carga",
        "peak_hour_bias": 0.9,
        "corridor_distribution": {c: 1 / 6 for c in INTERSECTIONS},
        "role_distribution": {
            "vecino": 0.65, "conductor": 0.20,
            "lider_barrio": 0.10, "despachador": 0.05,
        },
    },
}

# Peak hour sets by role
ROLE_PEAK_HOURS = {
    "despachador": [5, 6, 7, 8, 16, 17, 18, 19],
    "conductor": [6, 7, 8, 12, 17, 18],
    "lider_barrio": [7, 8, 12, 17, 18],
    "vecino": [7, 8, 12, 17, 18, 19],
}

# Number of corridors by role
ROLE_CORRIDOR_COUNT = {
    "despachador": (2, 4),
    "conductor": (1, 2),
    "lider_barrio": (1, 2),
    "vecino": (1, 1),
}


def _weighted_choice(dist: dict, rng: random.Random) -> str:
    """Sample from a weighted distribution dict."""
    items = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(items, weights=weights, k=1)[0]


def generate_user(user_id: str, scenario_cfg: dict, rng: random.Random) -> dict:
    """Generate a synthetic user node based on scenario config."""
    role = _weighted_choice(scenario_cfg["role_distribution"], rng)

    # Corridors
    min_c, max_c = ROLE_CORRIDOR_COUNT.get(role, (1, 1))
    n_corridors = rng.randint(min_c, max_c)
    corridors = []
    for _ in range(n_corridors):
        c = _weighted_choice(scenario_cfg["corridor_distribution"], rng)
        if c not in corridors:
            corridors.append(c)

    # Peak hours
    pool = ROLE_PEAK_HOURS.get(role, [7, 8, 17, 18])
    if role == "despachador":
        n_hours = rng.randint(5, len(pool))
    elif role == "conductor":
        n_hours = rng.randint(2, 3)
    elif role == "lider_barrio":
        n_hours = rng.randint(3, len(pool))
    else:
        n_hours = rng.randint(1, 2)
    peak_hours = sorted(rng.sample(pool, min(n_hours, len(pool))))

    now = datetime.now(timezone.utc).isoformat()
    return {
        "user_id": user_id,
        "role": role,
        "corridors": corridors,
        "peak_hours": peak_hours,
        "query_count": 0,
        "is_seed": False,
        "registered_at": now,
        "last_active": now,
    }


def generate_query(
    user_id: str, scenario_cfg: dict, rng: random.Random, base_time: datetime,
) -> dict:
    """Generate a synthetic bot query."""
    offset = rng.uniform(0, 8 * 3600)
    ts = base_time + timedelta(seconds=offset)
    iid = _weighted_choice(scenario_cfg["corridor_distribution"], rng)
    return {
        "user_id": user_id,
        "intersection_id": iid,
        "timestamp": ts.isoformat(),
    }


def run_ingestion(args) -> dict:
    """Execute the full ingestion pipeline."""
    scenario_cfg = SCENARIOS[args.scenario]
    rng = random.Random(args.seed)

    # Load config and instantiate graph module
    config = load_yaml_config("layer4_config.yaml")
    g = SocialGraphModule(config)

    print(f"🚀 Iniciando ingesta de {args.users} usuarios (escenario: {args.scenario})")

    # ---- Register users ----
    t0 = time.time()
    for i in range(args.users):
        uid = f"sim_{i:05d}"
        user = generate_user(uid, scenario_cfg, rng)
        g.register_user(user)
        if (i + 1) % 100 == 0:
            print(f"  ✓ {i + 1}/{args.users} usuarios registrados")
    user_time = time.time() - t0
    print(f"  ✓ {args.users} usuarios registrados en {user_time:.1f}s")

    # ---- Register queries ----
    print(f"  📝 Registrando {args.queries} consultas...")
    base_time = datetime.now(timezone.utc) - timedelta(hours=8)
    t0 = time.time()
    for i in range(args.queries):
        uid = f"sim_{rng.randint(0, args.users - 1):05d}"
        query = generate_query(uid, scenario_cfg, rng, base_time)
        g.register_query(query)
    query_time = time.time() - t0
    print(f"  ✓ {args.queries} consultas registradas en {query_time:.1f}s")

    # ---- Rebuild edges ----
    if args.users > 200:
        print("  🔄 Reconstruyendo aristas (O(V²))...")
        t0 = time.time()
        g.rebuild_edges()
        print(f"  ✓ Aristas reconstruidas en {time.time() - t0:.2f}s")

    # ---- Centrality ----
    if not args.no_centrality:
        print("  📊 Calculando centralidad (Brandes + k-shell)...")
        t0 = time.time()
        g.compute_centrality()
        print(f"  ✓ Centralidad calculada en {time.time() - t0:.2f}s")

    # ---- SIR simulation ----
    top_props = g.get_propagators_for_corridor(
        ["carrera_11_norte", "av_castellana_entrada"], k=5,
    )
    coverage = g.simulate_sir_coverage(top_props)
    print(f"  📡 Cobertura SIR desde {len(top_props)} propagadores: {coverage:.1%}")

    # ---- Report ----
    n = g._graph.number_of_nodes()
    e = g._graph.number_of_edges()
    density = (2 * e / (n * (n - 1)) * 100) if n > 1 else 0

    print(f"\n  {'─' * 50}")
    print(f"  Nodos: {n} | Aristas: {e} | Densidad: {density:.2f}%")
    print(f"  Top 5 propagadores:")

    sorted_props = sorted(g._propagator_ranking.items(), key=lambda x: x[1], reverse=True)[:5]
    for rank, (uid, score) in enumerate(sorted_props, 1):
        role = g._graph.nodes.get(uid, {}).get("role", "vecino")
        print(f"    #{rank} {uid} ({role}) — score: {score:.4f}")

    print(f"  Cobertura primera ola: {coverage:.1%}")
    tmc_est = min(35.0, coverage * 40)
    print(f"  Reducción TMC estimada (demo): {tmc_est:.1f}%")
    print(f"  {'─' * 50}")

    # ---- Export graph ----
    if args.export_graph:
        export_path = (
            get_project_root() / "data" / "layer4_outputs"
            / "social_graph" / "graph_export.json"
        )
        export_path.parent.mkdir(parents=True, exist_ok=True)

        nodes_list = []
        for uid in g._graph.nodes:
            at = g._graph.nodes[uid]
            nodes_list.append({
                "user_id": uid, "role": at.get("role", "vecino"),
                "corridors": at.get("corridors", []),
                "peak_hours": at.get("peak_hours", []),
                "score": g._propagator_ranking.get(uid, 0),
            })

        edges_list = []
        for u, v, data in g._graph.edges(data=True):
            edges_list.append({
                "source": u, "target": v, "weight": round(data.get("weight", 0), 4),
            })

        export = {
            "metadata": {
                "scenario": args.scenario,
                "n_users": args.users,
                "n_queries": args.queries,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "nodes": nodes_list,
            "edges": edges_list,
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"  📦 Grafo exportado a {export_path}")

    return {
        "nodes": n, "edges": e, "density": density,
        "top_propagators": sorted_props,
        "coverage": coverage, "tmc_estimate": tmc_est,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingesta masiva TransitMind Sogamoso",
    )
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()), default="normal",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-centrality", action="store_true")
    parser.add_argument("--export-graph", action="store_true")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  TransitMind Sogamoso — Ingesta Masiva")
    print(f"  Escenario: {SCENARIOS[args.scenario]['description']}")
    print(f"  Usuarios: {args.users} | Queries: {args.queries}")
    print(f"{'=' * 60}\n")

    metrics = run_ingestion(args)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Ingesta completada exitosamente")
    print(f"  El dashboard mostrará el grafo actualizado.")
    print(f"  Ejecuta: streamlit run src/layer4_bots/dashboard.py")
    print(f"{'=' * 60}\n")
    return metrics


if __name__ == "__main__":
    main()
