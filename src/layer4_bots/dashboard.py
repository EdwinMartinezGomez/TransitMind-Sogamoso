"""
TransitMind Sogamoso — Layer 4: Streamlit Dashboard
======================================================
Dashboard for Secretaría de Movilidad de Sogamoso.
Shows technical data + citizen context. Auto-refreshes every 30s.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import httpx
import pandas as pd

from src.shared.utils import load_yaml_config, get_project_root
from src.layer4_bots.message_formatter import MessageFormatter, SEVERITY_ORDER
from src.layer4_bots.social_graph import SocialGraphModule


# ============================================
# Helper Functions
# ============================================

def fetch_latest_decision(layer3_url: str):
    """Fetch GET /latest-decision from Layer 3."""
    try:
        resp = httpx.get(f"{layer3_url}/latest-decision", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_layer_health(urls: dict) -> dict:
    """Check /health of Layers 1, 2, 3 in parallel."""
    results = {}
    for name, url in urls.items():
        try:
            resp = httpx.get(f"{url}/health", timeout=5)
            results[name] = {"status": "ok" if resp.status_code == 200 else "error", "code": resp.status_code}
        except Exception:
            results[name] = {"status": "offline", "code": 0}
    return results


def severity_color(severity: str) -> str:
    """Return hex color for severity level."""
    return {"baja": "#22c55e", "media": "#f59e0b", "alta": "#ef4444", "critica": "#dc2626"}.get(severity, "#6b7280")


def render_intersection_card(decision: dict, message: str, config: dict):
    """Render a Streamlit card for one intersection."""
    display_names = config.get("intersection_display_names", {})
    emoji_map = config.get("severity_emoji", {})
    iid = decision.get("intersection_id", "")
    name = display_names.get(iid, iid.replace("_", " ").title())
    severity = decision.get("severity", "media")
    congestion = decision.get("congestion_level", 0.0)
    emoji = emoji_map.get(severity, "🟡")
    routes = decision.get("active_routes", [])
    tlc = decision.get("traffic_light_command", {})
    green_ext = tlc.get("green_extension_seconds", 0)
    priority_dir = tlc.get("priority_direction", "")

    color = severity_color(severity)

    with st.container():
        st.markdown(
            f"<div style='border-left: 4px solid {color}; padding: 0.5rem 1rem; "
            f"margin-bottom: 0.75rem; background: rgba(0,0,0,0.02); border-radius: 0 8px 8px 0;'>"
            f"<strong>{emoji} {name}</strong> — <em>{severity.upper()}</em></div>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Congestión", f"{int(congestion * 100)}%")
        col2.metric("Severidad", severity.capitalize())
        col3.metric("Verde +", f"{green_ext}s" if green_ext else "—")

        st.progress(min(congestion, 1.0))

        if routes:
            st.info(f"🗺️ Rutas alternativas: {', '.join(routes)}")
        if green_ext > 0 and priority_dir:
            st.caption(f"🚦 Semáforo: fase verde extendida {green_ext}s → {priority_dir.replace('_', '-')}")

        with st.expander("💬 Mensaje ciudadano generado"):
            st.text(message)

        st.divider()


def render_graph_section(graph_summary: dict, sg: SocialGraphModule, config: dict):
    """Expanded Social Graph section with interactive Plotly visualization."""
    from src.layer4_bots.graph_visualizer import (
        build_plotly_graph,
        build_edge_explanation_table,
        build_propagator_explanation_table,
        build_community_plotly,
    )

    st.markdown(
        "<div style='background: linear-gradient(135deg, #0a192f 0%, #112240 100%); "
        "padding: 1.5rem 2rem; border-radius: 16px; border: 1px solid rgba(100,255,218,0.15); "
        "margin-bottom: 1.5rem;'>"
        "<h2 style='margin:0; color:#ccd6f6;'>🕸️ Grafo Social</h2>"
        "<p style='margin:0.3rem 0 0; color:#64ffda; font-size:0.95rem;'>"
        "Red de Propagadores de Alertas — Modelo G=(V,E,W)</p></div>",
        unsafe_allow_html=True,
    )

    # --- Metrics with styled cards ---
    cols = st.columns(4)
    metrics = [
        ("👥", "Usuarios", graph_summary.get("total_users", 0), "#4ecdc4"),
        ("🔗", "Conexiones", graph_summary.get("edges", 0), "#ff6b6b"),
        ("🏘️", "Comunidades", graph_summary.get("communities_detected", 0), "#ffe66d"),
        ("⚡", "Activos 24h", graph_summary.get("active_last_24h", 0), "#a29bfe"),
    ]
    for col, (icon, label, val, accent) in zip(cols, metrics):
        col.markdown(
            f"<div style='background:#112240; border-radius:12px; padding:1rem; "
            f"text-align:center; border:1px solid {accent}33;'>"
            f"<span style='font-size:1.8rem;'>{icon}</span><br>"
            f"<span style='font-size:1.6rem; font-weight:700; color:{accent};'>{val}</span><br>"
            f"<span style='font-size:0.8rem; color:#8892b0;'>{label}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # --- Tabs ---
    tab_viz, tab_edges, tab_props, tab_comm = st.tabs([
        "🔵 Visualización Interactiva",
        "🔗 Por qué están conectados",
        "🏆 Por qué son propagadores",
        "🏘️ Comunidades de movilidad",
    ])

    with tab_viz:
        st.markdown(
            "<div style='background:#112240; padding:1rem 1.2rem; border-radius:10px; "
            "border-left:3px solid #64ffda; margin-bottom:1rem; font-size:0.9rem; color:#8892b0;'>"
            "Cada <b style='color:#4ecdc4;'>nodo</b> es un usuario del bot. "
            "Las conexiones representan similitud en corredores, horarios y co-consultas. "
            "Los nodos más grandes = mayor score propagador. "
            "<b style='color:#ff6b6b;'>Hover</b> para ver detalles.</div>",
            unsafe_allow_html=True,
        )
        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            n_nodes = st.slider(
                "Nodos a visualizar", 10, 60, 30, step=5,
                help="Muestra los N usuarios con mayor score de propagador",
            )
        with col_ctrl2:
            show_labels = st.checkbox(
                "Hover aristas", value=True,
                help="Muestra peso y razón de conexión al pasar el mouse",
            )
        fig = build_plotly_graph(
            sg._graph, sg._propagator_ranking,
            top_n_nodes=n_nodes, show_edge_labels=show_labels,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
            })
        else:
            st.info(
                "El grafo aún no tiene suficientes nodos para visualizar. "
                "Ejecuta el script de ingesta masiva primero."
            )

        with st.expander("⚙️ Parámetros del modelo (paper Sección III)"):
            sg_cfg = config.get("social_graph", {})
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("w₁ Corredor", sg_cfg.get("w1_corridor", 0.5),
                         help="Peso de similitud de corredor vial (Jaccard)")
            col_b.metric("w₂ Horario", sg_cfg.get("w2_temporal", 0.3),
                         help="Peso de similitud temporal (coseno)")
            col_c.metric("w₃ Co-consulta", sg_cfg.get("w3_coconsult", 0.2),
                         help="Peso de co-consultas simultáneas")
            col_a.metric("α betweenness", sg_cfg.get("alpha", 0.6),
                         help="Peso de centralidad de intermediación en el score")
            col_b.metric("θW umbral", sg_cfg.get("edge_threshold", 0.3),
                         help="Peso mínimo para crear arista")
            col_c.metric("K primera ola", sg_cfg.get("first_wave_k", 5),
                         help="Propagadores en la primera ola de alertas")

    with tab_edges:
        st.markdown(
            "**¿Por qué están conectados estos usuarios?**\n"
            "Cada fila explica la razón principal de una conexión: "
            "si comparten corredor vial, horario de desplazamiento, "
            "o consultan frecuentemente las mismas intersecciones."
        )
        if sg._graph.number_of_edges() > 0:
            edge_df = build_edge_explanation_table(sg._graph, sg._propagator_ranking, top_n=20)
            st.dataframe(edge_df, use_container_width=True, hide_index=True)
            if not edge_df.empty and "Razón principal" in edge_df.columns:
                reason_counts = edge_df["Razón principal"].value_counts()
                st.bar_chart(reason_counts, color="#4ecdc4", use_container_width=True)
                st.caption("Distribución de razones de conexión entre usuarios")
        else:
            st.info("Sin aristas aún. Registra más usuarios o ejecuta el script de ingesta.")

    with tab_props:
        st.markdown(
            "**¿Por qué son buenos propagadores?**\n"
            "Un buen propagador tiene alta *centralidad de intermediación* "
            "(es puente entre grupos) o está en capas *k-shell* altas "
            "(rodeado de vecinos densamente conectados).\n\n"
            "`score(v) = 0.6 × BC_norm(v) + 0.4 × ks_norm(v)`"
        )
        if sg._propagator_ranking:
            prop_df = build_propagator_explanation_table(
                sg._graph, sg._propagator_ranking, top_n=10,
            )
            st.dataframe(prop_df, use_container_width=True, hide_index=True)
            scores = list(sg._propagator_ranking.values())
            if scores:
                score_df = pd.DataFrame({"Score": sorted(scores, reverse=True)})
                st.line_chart(score_df, use_container_width=True)
                st.caption("Distribución de scores de propagadores (mayor = mejor)")
        else:
            st.info("Ejecuta compute_centrality() primero (desde el script de ingesta).")

    with tab_comm:
        st.markdown(
            "**Comunidades de movilidad detectadas**\n"
            "Agrupaciones naturales de usuarios que comparten "
            "corredores y horarios similares. Alertar a un nodo "
            "de alta centralidad en cada comunidad maximiza la cobertura."
        )
        comm_fig = build_community_plotly(sg._graph, sg._propagator_ranking)
        if comm_fig:
            st.plotly_chart(comm_fig, use_container_width=True, config={
                "displayModeBar": True, "displaylogo": False,
            })
        else:
            st.info(
                "Instala python-louvain para detección de comunidades: "
                "`pip install python-louvain`"
            )

    last_update = graph_summary.get("last_centrality_update", "")
    if last_update:
        st.caption(f"📅 Última actualización de centralidad: {last_update}")


# ============================================
# Main Dashboard
# ============================================

def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="TransitMind Sogamoso — Panel de Control",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load config
    try:
        config = load_yaml_config("layer4_config.yaml")
    except FileNotFoundError:
        st.error("⚠️ No se encontró configs/layer4_config.yaml")
        return

    layer3_url = config.get("layer3_api", {}).get("base_url", "http://localhost:8002")
    formatter = MessageFormatter(config)

    # ---- Header ----
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🏙️ TransitMind Sogamoso</h1>"
        "<p style='text-align:center; color:#888; margin-top:0;'>Panel de Control — Secretaría de Movilidad</p>",
        unsafe_allow_html=True,
    )

    # Fetch data
    data = fetch_latest_decision(layer3_url)

    if data:
        cycle_id = data.get("cycle_id", "—")[:8]
        ts = data.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_s = (datetime.now(dt.tzinfo) - dt).total_seconds() if dt.tzinfo else 0
            age_str = f"hace {int(age_s)}s" if age_s < 120 else f"hace {int(age_s / 60)} min"
        except (ValueError, TypeError):
            age_str = "—"

        st.success(f"⬤ Sistema activo | Último ciclo: {age_str} | ID: {cycle_id}")
    else:
        st.warning("⬤ Sin datos — No se pudo conectar a Capa 3")
        data = {"decisions": [], "global_tmc_reduction_percent": 0, "monitor_report": {}}

    decisions = data.get("decisions", [])
    monitor = data.get("monitor_report", {})
    tmc = data.get("global_tmc_reduction_percent", 0)

    # ---- Section 1: Executive Summary ----
    st.subheader("📊 Resumen Ejecutivo")
    n_crit = sum(1 for d in decisions if d.get("severity") in ("alta", "critica"))
    n_alerts = sum(1 for d in decisions if SEVERITY_ORDER.get(d.get("severity", "baja"), 0) >= 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Intersecciones críticas", n_crit)
    c2.metric("📉 Reducción TMC", f"{tmc:.1f}%")
    c3.metric("📢 Alertas activas", n_alerts)
    c4.metric("🤖 Agentes", f"{monitor.get('agents_healthy', 0)}/7")

    # System summary
    if decisions:
        summary = formatter.format_system_summary(data)
        st.info(summary)

    st.divider()

    # ---- Section 2: Intersection Cards by Severity ----
    st.subheader("🗺️ Estado de Intersecciones")

    if decisions:
        tab_crit, tab_med, tab_norm = st.tabs(["🔴 Críticas/Altas", "🟡 Medias", "🟢 Normales"])

        with tab_crit:
            crit_decs = [d for d in decisions if d.get("severity") in ("alta", "critica")]
            if crit_decs:
                for d in sorted(crit_decs, key=lambda x: SEVERITY_ORDER.get(x.get("severity"), 0), reverse=True):
                    msg = formatter.format_decision(d, "dashboard")
                    render_intersection_card(d, msg, config)
            else:
                st.success("✅ Sin intersecciones críticas en este momento")

        with tab_med:
            med_decs = [d for d in decisions if d.get("severity") == "media"]
            if med_decs:
                for d in med_decs:
                    msg = formatter.format_decision(d, "dashboard")
                    render_intersection_card(d, msg, config)
            else:
                st.success("✅ Sin intersecciones en severidad media")

        with tab_norm:
            norm_decs = [d for d in decisions if d.get("severity") == "baja"]
            if norm_decs:
                for d in norm_decs:
                    msg = formatter.format_decision(d, "dashboard")
                    render_intersection_card(d, msg, config)
            else:
                st.info("Sin datos de intersecciones en estado normal")
    else:
        st.info("No hay datos de intersecciones disponibles")

    st.divider()

    # ---- Section 3: Social Graph ----
    sg_enabled = config.get("social_graph", {}).get("enabled", False)
    graph_summary = {"total_users": 0}
    if sg_enabled:
        try:
            graph_module = SocialGraphModule(config)
            graph_module.maybe_update()
            graph_summary = graph_module.get_graph_summary()
            render_graph_section(graph_summary, graph_module, config)
        except Exception as e:
            st.warning(f"⚠️ Error al cargar grafo social: {e}")

        st.divider()

    # ---- Section 3B: Impact Charts ----
    from src.layer4_bots.impact_charts import render_all_impact_charts
    render_all_impact_charts(graph_summary, config)

    st.divider()

    # ---- Section 4: System Health ----
    st.subheader("🏥 Estado del Sistema")

    health_urls = {
        "Capa 1 (TimeGAN)": "http://localhost:8000",
        "Capa 2 (LLM+RAG)": "http://localhost:8001",
        "Capa 3 (Agentes)": "http://localhost:8002",
    }

    health = fetch_layer_health(health_urls)
    hcols = st.columns(len(health))
    for col, (name, info) in zip(hcols, health.items()):
        if info["status"] == "ok":
            col.success(f"✅ {name}")
        elif info["status"] == "error":
            col.warning(f"⚠️ {name} (HTTP {info['code']})")
        else:
            col.error(f"❌ {name}")

    # Alert log
    st.markdown("**📋 Últimas alertas enviadas:**")
    alerts_dir = get_project_root() / "data" / "layer4_outputs" / "alerts_sent"
    if alerts_dir.exists():
        alert_files = sorted(alerts_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
        if alert_files:
            import json
            rows = []
            for f in alert_files:
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        alert_data = json.load(fh)
                    rows.append({
                        "Timestamp": alert_data.get("timestamp", "—"),
                        "Cycle ID": str(alert_data.get("cycle_id", "—"))[:8],
                        "Alertas": len(alert_data.get("alerts_to_send", [])),
                        "Omitidas": alert_data.get("skipped", 0),
                    })
                except Exception:
                    pass
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("Sin alertas registradas aún")
        else:
            st.caption("Sin alertas registradas aún")
    else:
        st.caption("Sin directorio de alertas")

    st.divider()

    # ---- Section 5: History ----
    st.subheader("📜 Historial de Ciclos")
    decisions_dir = get_project_root() / "data" / "layer3_outputs" / "decisions"
    if decisions_dir.exists():
        import json
        dec_files = sorted(decisions_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
        if dec_files:
            rows = []
            for f in dec_files:
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        d = json.load(fh)
                    mon = d.get("monitor_report", {})
                    rows.append({
                        "Cycle ID": str(d.get("cycle_id", "—"))[:8],
                        "Timestamp": d.get("timestamp", "—"),
                        "Alertas": len(d.get("decisions", [])),
                        "TMC %": f"{d.get('global_tmc_reduction_percent', 0):.1f}",
                        "Duración (ms)": mon.get("cycle_duration_ms", 0),
                    })
                except Exception:
                    pass
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Sin ciclos registrados")
    else:
        st.caption("Sin directorio de decisiones")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Configuración")
        st.markdown(f"**Ciudad:** {config.get('message', {}).get('city', 'Sogamoso')}")
        st.markdown(f"**Severidad mínima:** {config.get('alert', {}).get('min_severity_to_alert', 'media')}")
        st.markdown(f"**Poll interval:** {config.get('layer3_api', {}).get('poll_interval_seconds', 30)}s")

        st.divider()

        if st.button("🔄 Ejecutar ciclo manual", use_container_width=True):
            with st.spinner("Ejecutando ciclo en Capa 3..."):
                try:
                    resp = httpx.post(f"{layer3_url}/run-cycle", json={"scenario": "normal_weekday"}, timeout=130)
                    if resp.status_code == 200:
                        st.success("✅ Ciclo completado")
                        st.rerun()
                    else:
                        st.error(f"Error: HTTP {resp.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)[:80]}")

        if st.button("🔃 Refrescar datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption("TransitMind Sogamoso v1.0\nUPTC — Ing. de Sistemas")

    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
