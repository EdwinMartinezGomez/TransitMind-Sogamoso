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

@st.cache_data(ttl=25)
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


def render_graph_section(graph_summary: dict):
    """Visualize social graph stats and top propagators."""
    st.subheader("🕸️ Grafo Social — Red de Usuarios")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Usuarios", graph_summary.get("total_users", 0))
    col2.metric("Aristas", graph_summary.get("edges", 0))
    col3.metric("Comunidades", graph_summary.get("communities_detected", 0))
    col4.metric("Activos 24h", graph_summary.get("active_last_24h", 0))

    # Top propagators table
    top = graph_summary.get("top_propagators", [])
    if top:
        st.markdown("**Top 5 propagadores:**")
        df = pd.DataFrame(top)
        if "corridors" in df.columns:
            df["corridors"] = df["corridors"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        if "score" in df.columns:
            df["score"] = df["score"].apply(lambda x: f"{x:.4f}")
        st.dataframe(df, use_container_width=True, hide_index=True)

    last_update = graph_summary.get("last_centrality_update", "")
    if last_update:
        st.caption(f"Última actualización de centralidad: {last_update}")


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
    if sg_enabled:
        try:
            graph_module = SocialGraphModule(config)
            graph_summary = graph_module.get_graph_summary()
            render_graph_section(graph_summary)
        except Exception as e:
            st.warning(f"⚠️ Error al cargar grafo social: {e}")

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
