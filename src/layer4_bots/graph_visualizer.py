"""
TransitMind Sogamoso — Layer 4: Graph Visualizer (Plotly)
==========================================================
Interactive force-directed visualization of G=(V,E,W).
Replaces static Graphviz with rich Plotly charts.
"""

from __future__ import annotations

import math
from typing import Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from src.shared.logger import get_logger

logger = get_logger("layer4.graph_visualizer")

# ---- Theme ----

ROLE_COLORS = {
    "despachador": "#ff6b6b",
    "conductor": "#4ecdc4",
    "lider_barrio": "#ffe66d",
    "vecino": "#a8b2d1",
}

ROLE_GLOW = {
    "despachador": "rgba(255,107,107,0.35)",
    "conductor": "rgba(78,205,196,0.35)",
    "lider_barrio": "rgba(255,230,109,0.35)",
    "vecino": "rgba(168,178,209,0.15)",
}

ROLE_SYMBOLS = {
    "despachador": "diamond",
    "conductor": "circle",
    "lider_barrio": "square",
    "vecino": "circle",
}

ROLE_LABELS_ES = {
    "despachador": "Despachador",
    "conductor": "Conductor",
    "lider_barrio": "Líder barrio",
    "vecino": "Vecino",
}

COMMUNITY_PALETTE = [
    "#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe",
    "#fd79a8", "#00cec9", "#fab1a0", "#74b9ff",
    "#55efc4", "#ffeaa7", "#dfe6e9", "#e17055",
]

ROLE_SHAPES_DOT = {
    "despachador": "diamond",
    "conductor": "ellipse",
    "lider_barrio": "box",
    "vecino": "circle",
}


# ---- Helpers (unchanged math) ----

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
# Plotly Interactive Graph
# ============================================

def build_plotly_graph(
    graph: nx.Graph,
    propagator_ranking: dict,
    top_n_nodes: int = 30,
    show_edge_labels: bool = False,
) -> Optional[go.Figure]:
    """Build interactive Plotly force-directed graph."""
    if graph.number_of_nodes() == 0:
        return None

    sorted_nodes = sorted(propagator_ranking.items(), key=lambda x: x[1], reverse=True)
    top_ids = [uid for uid, _ in sorted_nodes[:top_n_nodes]]
    top_set = set(top_ids)
    if not top_set:
        return None

    # Subgraph
    sub = graph.subgraph(top_set).copy()
    if sub.number_of_nodes() == 0:
        return None

    # Spring layout
    pos = nx.spring_layout(sub, k=2.5 / math.sqrt(max(sub.number_of_nodes(), 1)),
                           iterations=80, seed=42, weight="weight")

    scores = [propagator_ranking.get(uid, 0) for uid in top_ids if uid in pos]
    min_s = min(scores) if scores else 0
    max_s = max(scores) if scores else 1
    sr = max_s - min_s if max_s > min_s else 1.0

    # ---- Edges ----
    edge_traces = []
    for u, v, data in sub.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data.get("weight", 0)
        if w >= 0.6:
            ecolor = "rgba(255,107,107,0.6)"
            width = 2.5
        elif w >= 0.4:
            ecolor = "rgba(255,230,109,0.5)"
            width = 1.8
        else:
            ecolor = "rgba(168,178,209,0.2)"
            width = 0.8

        hover = ""
        if show_edge_labels:
            lbl = _dominant_label(graph, u, v)
            hover = f"{u[:8]} ↔ {v[:8]}<br>Peso: {w:.3f}<br>Razón: {lbl}"

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=ecolor),
            hoverinfo="text" if hover else "skip",
            hovertext=hover if hover else None,
            showlegend=False,
        ))

    # ---- Glow layer (larger translucent circles behind nodes) ----
    glow_x, glow_y, glow_size, glow_color = [], [], [], []
    for uid in sub.nodes():
        if uid not in pos:
            continue
        x, y = pos[uid]
        role = sub.nodes[uid].get("role", "vecino")
        score = propagator_ranking.get(uid, 0)
        norm = (score - min_s) / sr
        glow_x.append(x)
        glow_y.append(y)
        glow_size.append(25 + norm * 45)
        glow_color.append(ROLE_GLOW.get(role, "rgba(168,178,209,0.15)"))

    glow_trace = go.Scatter(
        x=glow_x, y=glow_y,
        mode="markers",
        marker=dict(size=glow_size, color=glow_color, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    )

    # ---- Nodes ----
    node_x, node_y, node_size, node_color, node_symbol = [], [], [], [], []
    node_text, node_hover = [], []
    for uid in sub.nodes():
        if uid not in pos:
            continue
        x, y = pos[uid]
        attrs = sub.nodes[uid]
        role = attrs.get("role", "vecino")
        score = propagator_ranking.get(uid, 0)
        norm = (score - min_s) / sr
        corrs = attrs.get("corridors", [])

        node_x.append(x)
        node_y.append(y)
        node_size.append(14 + norm * 28)
        node_color.append(ROLE_COLORS.get(role, "#a8b2d1"))
        node_symbol.append(ROLE_SYMBOLS.get(role, "circle"))
        node_text.append(uid[:8])
        node_hover.append(
            f"<b>{uid[:10]}</b><br>"
            f"Rol: {ROLE_LABELS_ES.get(role, role)}<br>"
            f"Score: {score:.4f}<br>"
            f"Conexiones: {sub.degree(uid)}<br>"
            f"Corredores: {', '.join(corrs[:3]) if corrs else '—'}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            symbol=node_symbol,
            line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=8, color="#ccd6f6", family="Inter, sans-serif"),
        hoverinfo="text",
        hovertext=node_hover,
        showlegend=False,
    )

    # ---- Legend traces ----
    legend_traces = []
    for role, color in ROLE_COLORS.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=color, symbol=ROLE_SYMBOLS.get(role, "circle"),
                        line=dict(width=1, color="rgba(255,255,255,0.4)")),
            name=ROLE_LABELS_ES.get(role, role),
            showlegend=True,
        ))

    # ---- Figure ----
    fig = go.Figure(
        data=edge_traces + [glow_trace, node_trace] + legend_traces
    )
    fig.update_layout(
        plot_bgcolor="#0a192f",
        paper_bgcolor="#0a192f",
        font=dict(family="Inter, sans-serif", color="#ccd6f6"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            bgcolor="rgba(10,25,47,0.8)", bordercolor="rgba(100,255,218,0.3)",
            borderwidth=1, font=dict(size=11, color="#ccd6f6"),
        ),
        hoverlabel=dict(
            bgcolor="#112240", bordercolor="rgba(100,255,218,0.5)",
            font=dict(size=12, color="#ccd6f6", family="Inter"),
        ),
        title=dict(
            text="Red de Propagadores — Grafo Social",
            font=dict(size=16, color="#64ffda"),
            x=0.5, xanchor="center",
        ),
    )
    return fig


# ============================================
# Community Plotly Graph
# ============================================

def build_community_plotly(
    graph: nx.Graph, propagator_ranking: dict,
) -> Optional[go.Figure]:
    """Community-colored Plotly graph via Louvain."""
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

    pos = nx.spring_layout(graph, k=2.0 / math.sqrt(max(graph.number_of_nodes(), 1)),
                           iterations=60, seed=42)

    # Edges
    edge_x, edge_y = [], []
    for u, v in graph.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="rgba(168,178,209,0.15)"),
        hoverinfo="skip", showlegend=False,
    )

    # Nodes by community
    comms = {}
    for node, cid in partition.items():
        comms.setdefault(cid, []).append(node)

    node_traces = []
    for cid, members in sorted(comms.items()):
        color = COMMUNITY_PALETTE[cid % len(COMMUNITY_PALETTE)]
        nx_list, ny_list, sizes, hovers, texts = [], [], [], [], []
        for uid in members:
            if uid not in pos:
                continue
            x, y = pos[uid]
            score = propagator_ranking.get(uid, 0)
            role = graph.nodes[uid].get("role", "vecino")
            nx_list.append(x)
            ny_list.append(y)
            sizes.append(8 + score * 30)
            texts.append(uid[:6])
            hovers.append(
                f"<b>{uid[:10]}</b><br>Comunidad: C{cid}<br>"
                f"Rol: {ROLE_LABELS_ES.get(role, role)}<br>Score: {score:.4f}"
            )
        node_traces.append(go.Scatter(
            x=nx_list, y=ny_list, mode="markers+text",
            marker=dict(size=sizes, color=color,
                        line=dict(width=1, color="rgba(255,255,255,0.2)")),
            text=texts, textposition="top center",
            textfont=dict(size=7, color="#8892b0"),
            hoverinfo="text", hovertext=hovers,
            name=f"Comunidad {cid} ({len(members)})",
            showlegend=True,
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        plot_bgcolor="#0a192f", paper_bgcolor="#0a192f",
        font=dict(family="Inter, sans-serif", color="#ccd6f6"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=550,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            bgcolor="rgba(10,25,47,0.8)", bordercolor="rgba(100,255,218,0.3)",
            borderwidth=1, font=dict(size=10, color="#ccd6f6"),
        ),
        hoverlabel=dict(
            bgcolor="#112240", bordercolor="rgba(100,255,218,0.5)",
            font=dict(size=12, color="#ccd6f6"),
        ),
        title=dict(
            text="Comunidades de Movilidad — Louvain",
            font=dict(size=16, color="#64ffda"), x=0.5,
        ),
    )
    return fig


# ============================================
# Graphviz DOT (legacy fallback)
# ============================================

def build_graphviz_dot(
    graph: nx.Graph, propagator_ranking: dict,
    top_n_nodes: int = 30, show_edge_labels: bool = True,
) -> str:
    """Legacy DOT fallback — kept for compatibility."""
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
        '  rankdir=LR; bgcolor="transparent";',
        '  node [style=filled, fontname="Helvetica", fontsize=9];',
        '  edge [fontname="Helvetica", fontsize=7];', "",
    ]
    for uid in top_ids:
        attrs = graph.nodes.get(uid, {})
        role = attrs.get("role", "vecino")
        score = propagator_ranking.get(uid, 0)
        color = {"despachador": "#ef4444", "conductor": "#3b82f6",
                 "lider_barrio": "#22c55e", "vecino": "#9ca3af"}.get(role, "#9ca3af")
        shape = ROLE_SHAPES_DOT.get(role, "circle")
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
        ec = "#ef4444" if weight >= 0.6 else "#f59e0b" if weight >= 0.4 else "#d1d5db"
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


def build_community_subgraph_dot(
    graph: nx.Graph, propagator_ranking: dict,
) -> Optional[str]:
    """Legacy DOT community fallback."""
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
        '  edge [color="#d1d5db", penwidth=0.5];', "",
    ]
    for cid, members in sorted(comms.items()):
        color = COMMUNITY_PALETTE[cid % len(COMMUNITY_PALETTE)]
        for uid in members:
            at = graph.nodes.get(uid, {})
            shape = ROLE_SHAPES_DOT.get(at.get("role", "vecino"), "circle")
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
