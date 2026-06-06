"""
TransitMind Sogamoso — Layer 4: Graph Visualizer
====================================================
Interactive visualization of the Social Graph G=(V,E,W).
Generates Graphviz DOT code and explanation tables for the dashboard.
"""

from __future__ import annotations

import math
from typing import Optional

import networkx as nx
import pandas as pd

from src.shared.logger import get_logger

logger = get_logger("layer4.graph_visualizer")

ROLE_COLORS = {
    "despachador": "#ef4444",
    "conductor": "#3b82f6",
    "lider_barrio": "#22c55e",
    "vecino": "#9ca3af",
}

ROLE_SHAPES = {
    "despachador": "diamond",
    "conductor": "ellipse",
    "lider_barrio": "box",
    "vecino": "circle",
}

COMMUNITY_COLORS = [
    "#ef4444", "#3b82f6", "#22c55e", "#f59e0b",
    "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
    "#6366f1", "#06b6d4", "#84cc16", "#e11d48",
]


# ---- Internal helpers ----

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def _cosine_hours(hours_a: list, hours_b: list) -> float:
    if not hours_a and not hours_b:
        return 0.0
    va = [0.0] * 24
    vb = [0.0] * 24
    for h in hours_a:
        if 0 <= h < 24:
            va[h] = 1.0
    for h in hours_b:
        if 0 <= h < 24:
            vb[h] = 1.0
    dot = sum(a * b for a, b in zip(va, vb))
    ma = math.sqrt(sum(a * a for a in va))
    mb = math.sqrt(sum(b * b for b in vb))
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)


def _dominant_label(graph, u, v):
    ua = graph.nodes.get(u, {})
    va = graph.nodes.get(v, {})
    sc = _jaccard(set(ua.get("corridors", [])), set(va.get("corridors", [])))
    st = _cosine_hours(ua.get("peak_hours", []), va.get("peak_hours", []))
    w1v = 0.5 * sc
    w2v = 0.3 * st
    ed = graph.get_edge_data(u, v) or {}
    w3v = max(0.0, ed.get("weight", 0) - w1v - w2v)
    if w1v >= w2v and w1v >= w3v:
        return "corredor"
    elif w2v >= w3v:
        return "horario"
    return "co-consulta"


def _count_communities_bridged(graph, uid):
    try:
        import community as community_louvain
        part = community_louvain.best_partition(graph)
        nbrs = list(graph.neighbors(uid))
        return max(len(set(part.get(n, 0) for n in nbrs)), 2)
    except Exception:
        return max(2, min(graph.degree(uid), 5))


# ============================================
# DOT Generation
# ============================================

def build_graphviz_dot(
    graph: nx.Graph,
    propagator_ranking: dict,
    top_n_nodes: int = 30,
    show_edge_labels: bool = True,
) -> str:
    """Generate DOT code for st.graphviz_chart()."""
    if graph.number_of_nodes() == 0:
        return ""

    sorted_nodes = sorted(propagator_ranking.items(), key=lambda x: x[1], reverse=True)
    top_ids = [uid for uid, _ in sorted_nodes[:top_n_nodes]]
    top_set = set(top_ids)
    if not top_set:
        return ""

    scores = [propagator_ranking.get(uid, 0) for uid in top_ids]
    min_s = min(scores) if scores else 0
    max_s = max(scores) if scores else 1
    sr = max_s - min_s if max_s > min_s else 1.0

    lines = [
        "graph G {",
        '  rankdir=LR;',
        '  bgcolor="transparent";',
        '  node [style=filled, fontname="Helvetica", fontsize=9];',
        '  edge [fontname="Helvetica", fontsize=7];',
        "",
    ]

    for uid in top_ids:
        attrs = graph.nodes.get(uid, {})
        role = attrs.get("role", "vecino")
        score = propagator_ranking.get(uid, 0)
        color = ROLE_COLORS.get(role, "#9ca3af")
        shape = ROLE_SHAPES.get(role, "circle")
        norm = (score - min_s) / sr
        w = 0.3 + norm * 0.9
        short = uid[:8]
        lbl = f"{short}\\n{role}\\n{score:.3f}"
        lines.append(
            f'  "{uid}" [label="{lbl}", shape={shape}, fillcolor="{color}", '
            f'fontcolor="white", width={w:.2f}, height={w:.2f}];'
        )

    lines.append("")
    rendered = set()
    for u, v, data in graph.edges(data=True):
        if u not in top_set or v not in top_set:
            continue
        ek = tuple(sorted([u, v]))
        if ek in rendered:
            continue
        rendered.add(ek)
        weight = data.get("weight", 0)
        pw = max(0.5, min(3.0, weight * 4.0))
        if weight >= 0.6:
            ec = "#ef4444"
        elif weight >= 0.4:
            ec = "#f59e0b"
        else:
            ec = "#d1d5db"
        la = ""
        if show_edge_labels:
            la = f', label="{_dominant_label(graph, u, v)}", fontcolor="#374151"'
        lines.append(f'  "{u}" -- "{v}" [penwidth={pw:.1f}, color="{ec}"{la}];')

    lines.append("")
    lines.append("  subgraph cluster_legend {")
    lines.append('    label="Leyenda"; style=dashed; fontname="Helvetica"; fontsize=10; color="#6b7280";')
    lines.append('    leg_d [label="Despachador", shape=diamond, fillcolor="#ef4444", fontcolor="white", width=0.4];')
    lines.append('    leg_c [label="Conductor", shape=ellipse, fillcolor="#3b82f6", fontcolor="white", width=0.4];')
    lines.append('    leg_l [label="Líder barrio", shape=box, fillcolor="#22c55e", fontcolor="white", width=0.4];')
    lines.append('    leg_v [label="Vecino", shape=circle, fillcolor="#9ca3af", fontcolor="white", width=0.4];')
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


# ============================================
# Edge Explanation Table
# ============================================

def build_edge_explanation_table(
    graph: nx.Graph, propagator_ranking: dict, top_n: int = 15,
) -> pd.DataFrame:
    """DataFrame explaining the top edges by weight."""
    eww = [(u, v, d.get("weight", 0)) for u, v, d in graph.edges(data=True)]
    eww.sort(key=lambda x: x[2], reverse=True)

    rows = []
    for u, v, wt in eww[:top_n]:
        ua = graph.nodes.get(u, {})
        va = graph.nodes.get(v, {})
        uc = set(ua.get("corridors", []))
        vc = set(va.get("corridors", []))
        shared = uc & vc
        sc = _jaccard(uc, vc)
        st = _cosine_hours(ua.get("peak_hours", []), va.get("peak_hours", []))
        w1, w2, w3 = 0.5, 0.3, 0.2
        cc = max(0.0, min(1.0, (wt - w1 * sc - w2 * st) / w3)) if w3 > 0 else 0.0
        comps = {"Mismo corredor": w1 * sc, "Mismo horario": w2 * st, "Co-consulta": w3 * cc}
        dom = max(comps, key=comps.get)
        if wt >= 0.6:
            intensity = "🔴 Fuerte"
        elif wt >= 0.4:
            intensity = "🟠 Media"
        else:
            intensity = "⚪ Débil"
        rows.append({
            "Usuario A": f"{u[:8]} ({ua.get('role', 'vecino')})",
            "Usuario B": f"{v[:8]} ({va.get('role', 'vecino')})",
            "Peso total": round(wt, 3),
            "Corredores compartidos": ", ".join(sorted(shared)) if shared else "—",
            "Similitud horaria": round(st, 2),
            "Co-consultas": round(cc, 2),
            "Razón principal": dom,
            "Intensidad": intensity,
        })
    return pd.DataFrame(rows)


# ============================================
# Propagator Explanation Table
# ============================================

def build_propagator_explanation_table(
    graph: nx.Graph, propagator_ranking: dict, top_n: int = 10,
) -> pd.DataFrame:
    """Explain why each top propagator has its score."""
    if graph.number_of_nodes() < 2:
        return pd.DataFrame()

    bc = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    ks = nx.core_number(graph)
    mks = max(ks.values()) if ks else 1
    if mks == 0:
        mks = 1

    sp = sorted(propagator_ranking.items(), key=lambda x: x[1], reverse=True)[:top_n]
    rows = []
    for rank, (uid, score) in enumerate(sp, 1):
        at = graph.nodes.get(uid, {})
        role = at.get("role", "vecino")
        corrs = at.get("corridors", [])
        bv = bc.get(uid, 0.0)
        kv = ks.get(uid, 0)
        kn = kv / mks
        if bv > 0.15 and kn > 0.6:
            interp = "Equilibrio entre centralidad e influencia local"
        elif bv > 0.1:
            nc = _count_communities_bridged(graph, uid)
            interp = f"Puente entre {nc} comunidades de movilidad (BC alto)"
        elif kn > 0.7:
            interp = f"Profundamente embebido en el núcleo de la red (ks={kv})"
        elif role == "despachador":
            interp = "Nodo operativo con conexiones a múltiples corredores"
        elif role == "lider_barrio":
            interp = "Influenciador local con alcance comunitario"
        else:
            interp = "Contribuye a la cobertura de su corredor vial"
        rows.append({
            "Rango": rank,
            "Usuario": f"{uid[:8]} ({role})",
            "Score total": round(score, 4),
            "Betweenness (BC)": round(bv, 4),
            "K-Shell": kv,
            "Corredores": ", ".join(corrs[:3]) if corrs else "—",
            "Interpretación": interp,
        })
    return pd.DataFrame(rows)


# ============================================
# Community Subgraph DOT
# ============================================

def build_community_subgraph_dot(
    graph: nx.Graph, propagator_ranking: dict,
) -> Optional[str]:
    """DOT with community-colored nodes via Louvain. Returns None if unavailable."""
    try:
        import community as community_louvain
    except ImportError:
        return None

    if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
        return None

    try:
        partition = community_louvain.best_partition(graph)
    except Exception:
        return None

    comms = {}
    for node, cid in partition.items():
        comms.setdefault(cid, []).append(node)

    lines = [
        "graph G {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [style=filled, fontname="Helvetica", fontsize=9];',
        '  edge [color="#d1d5db", penwidth=0.5];',
        "",
    ]
    for cid, members in sorted(comms.items()):
        color = COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)]
        for uid in members:
            at = graph.nodes.get(uid, {})
            shape = ROLE_SHAPES.get(at.get("role", "vecino"), "circle")
            score = propagator_ranking.get(uid, 0)
            w = 0.5 if score > 0.3 else 0.3
            lbl = f"{uid[:8]}\\nC{cid}"
            lines.append(
                f'  "{uid}" [label="{lbl}", shape={shape}, fillcolor="{color}", '
                f'fontcolor="white", width={w:.2f}];'
            )
    lines.append("")
    rendered = set()
    for u, v, _ in graph.edges(data=True):
        ek = tuple(sorted([u, v]))
        if ek in rendered:
            continue
        rendered.add(ek)
        sty = "dashed" if partition.get(u, 0) != partition.get(v, 0) else "solid"
        ec = "#9ca3af" if sty == "dashed" else "#d1d5db"
        lines.append(f'  "{u}" -- "{v}" [style={sty}, color="{ec}"];')
    lines.append("}")
    return "\n".join(lines)


# ============================================
# Edge Decomposition
# ============================================

def get_edge_decomposition(graph: nx.Graph, u: str, v: str, sg=None) -> dict:
    """Decompose W(u,v) into its three similarity components."""
    ua = graph.nodes.get(u, {})
    va = graph.nodes.get(v, {})
    uc = set(ua.get("corridors", []))
    vc = set(va.get("corridors", []))
    sc = _jaccard(uc, vc)
    st = _cosine_hours(ua.get("peak_hours", []), va.get("peak_hours", []))
    w1 = getattr(sg, "_w1", 0.5) if sg else 0.5
    w2 = getattr(sg, "_w2", 0.3) if sg else 0.3
    ed = graph.get_edge_data(u, v) or {}
    wt = ed.get("weight", 0.0)
    wc = max(0.0, wt - w1 * sc - w2 * st)
    comps = {"corredor": w1 * sc, "horario": w2 * st, "co-consulta": wc}
    dom = max(comps, key=comps.get)
    return {
        "w_corridor": round(w1 * sc, 4),
        "w_temporal": round(w2 * st, 4),
        "w_coconsult": round(wc, 4),
        "w_total": round(wt, 4),
        "dominant": dom,
        "corridors_shared": sorted(uc & vc),
    }
