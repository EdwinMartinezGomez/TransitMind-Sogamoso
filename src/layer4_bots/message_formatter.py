"""
TransitMind Sogamoso — Layer 4: Message Formatter
====================================================
Translates Layer 3 technical JSON decisions into clear,
natural-language messages for citizens using Ollama llama3:8b.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.shared.logger import get_logger

logger = get_logger("layer4.message_formatter")

# ============================================
# Constants & Lookup Tables
# ============================================

SYSTEM_PROMPT_FORMATTER = (
    "Eres el comunicador ciudadano del sistema de movilidad TransitMind Sogamoso. "
    "Tu trabajo es traducir datos técnicos de tráfico a mensajes CLAROS, CONCISOS "
    "y ACCIONABLES para ciudadanos colombianos de una ciudad intermedia.\n\n"
    "REGLAS ESTRICTAS:\n"
    "1. NUNCA uses palabras técnicas: no digas 'congestion_level', 'severity', "
    "'green_extension', 'GAN', 'LLM', 'agente' ni ningún término de sistemas.\n"
    "2. USA lenguaje cotidiano colombiano: 'trancón', 'está feo el tráfico', "
    "'mejor espérese', 'calcule más tiempo', 'tome la ruta alterna'.\n"
    "3. SÉ ESPECÍFICO con nombres de calles y tiempos estimados.\n"
    "4. El mensaje debe responder: ¿Debo salir? ¿Por dónde?\n"
    "5. Responde SOLO con el mensaje final, sin explicaciones previas."
)

SEVERITY_TO_PLAIN: Dict[str, tuple] = {
    "baja":    ("verde",        "tráfico normal",     "puede circular sin problema"),
    "media":   ("amarillo",     "algo de tráfico",    "calcule 5-10 minutos extra"),
    "alta":    ("rojo",         "trancón fuerte",     "evite si puede, calcule 15-20 min extra"),
    "critica": ("rojo intenso", "trancón crítico",    "evite completamente, hay congestión severa"),
}

CONGESTION_TO_DELAY: List[tuple] = [
    (0.0,  0.4,   0),
    (0.4,  0.6,   7),
    (0.6,  0.75, 15),
    (0.75, 0.9,  22),
    (0.9,  1.01, 35),
]

SEVERITY_ORDER = {"baja": 0, "media": 1, "alta": 2, "critica": 3}

FORBIDDEN_TERMS = [
    "congestion_level", "severity", "green_extension", "cycle_adjustment",
    "priority_direction", "agent_trace", "GAN", "LLM", "agente", "sensor",
    "predictor", "gan_simulator", "traffic_light_command", "intersection_id",
]


class MessageFormatter:
    """Translates Layer 3 technical decisions into human-readable messages."""

    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {})
        self._model = llm_cfg.get("model", "llama3:8b")
        self._temperature = float(llm_cfg.get("temperature", 0.3))
        self._max_tokens = int(llm_cfg.get("max_tokens", 300))
        self._timeout = int(llm_cfg.get("timeout_seconds", 25))
        self._config = config
        self._llm_available: Optional[bool] = None

        msg_cfg = config.get("message", {})
        self._city = msg_cfg.get("city", "Sogamoso")

        self._display_names: Dict[str, str] = config.get("intersection_display_names", {})
        self._severity_emoji: Dict[str, str] = config.get("severity_emoji", {
            "baja": "🟢", "media": "🟡", "alta": "🔴", "critica": "🚨",
        })

        alert_cfg = config.get("alert", {})
        self._min_severity = alert_cfg.get("min_severity_to_alert", "media")

    def _check_llm(self) -> bool:
        """Verify if Ollama is available (with cache)."""
        if self._llm_available is not None:
            return self._llm_available
        try:
            import ollama
            ollama.list()
            self._llm_available = True
            logger.info("llm_check_ok", model=self._model)
        except Exception as e:
            self._llm_available = False
            logger.warning("llm_unavailable", error=str(e))
        return self._llm_available

    def _get_display_name(self, intersection_id: str) -> str:
        """Get human-readable name for an intersection."""
        return self._display_names.get(
            intersection_id, intersection_id.replace("_", " ").title()
        )

    def _estimate_delay(self, congestion_level: float) -> int:
        """Convert congestion_level float to estimated delay in minutes."""
        for low, high, delay in CONGESTION_TO_DELAY:
            if low <= congestion_level < high:
                return delay
        return 35

    def _get_max_chars(self, channel: str) -> int:
        """Get max characters for a given channel."""
        msg_cfg = self._config.get("message", {})
        if channel == "whatsapp":
            return msg_cfg.get("max_length_whatsapp", 280)
        elif channel == "telegram_citizen":
            return msg_cfg.get("max_length_telegram", 400)
        elif channel == "dashboard":
            return msg_cfg.get("max_length_dashboard", 600)
        return 400

    def _build_formatter_prompt(self, decision: dict, channel: str, max_chars: int) -> str:
        """Build the LLM prompt with few-shot examples per channel."""
        name = self._get_display_name(decision.get("intersection_id", ""))
        severity = decision.get("severity", "media")
        congestion = decision.get("congestion_level", 0.5)
        routes = decision.get("active_routes", [])
        emoji = self._severity_emoji.get(severity, "🟡")
        delay = self._estimate_delay(congestion)
        routes_str = ", ".join(routes) if routes else "sin rutas alternas reportadas"

        tlc = decision.get("traffic_light_command", {})
        green_ext = tlc.get("green_extension_seconds", 0)
        priority_dir = tlc.get("priority_direction", "")
        now_str = datetime.now().strftime("%I:%M %p").lower()

        prompt = (
            f"Genera un mensaje para el canal {channel} (máximo {max_chars} caracteres).\n\n"
            f"DATOS:\n"
            f"- Intersección: {name}\n"
            f"- Estado: {severity} ({emoji})\n"
            f"- Congestión: {congestion:.0%}\n"
            f"- Demora estimada: ~{delay} min\n"
            f"- Rutas alternativas: {routes_str}\n"
            f"- Semáforo: fase verde +{green_ext}s dirección {priority_dir}\n"
            f"- Hora: {now_str}\n\n"
            f"Usa emoji {emoji} al inicio. Incluye nombre de intersección y rutas.\n"
            f"Lenguaje coloquial colombiano. SOLO el mensaje, sin comillas."
        )
        return prompt

    def format_decision(self, decision: dict, channel: str) -> str:
        """
        Main pipeline: LLM → fallback → truncate.
        Returns string ready to send to citizen.
        """
        max_chars = self._get_max_chars(channel)

        if self._check_llm():
            try:
                message = self._format_with_llm(decision, channel, max_chars)
                if message and len(message.strip()) > 10:
                    msg_lower = message.lower()
                    has_forbidden = any(t.lower() in msg_lower for t in FORBIDDEN_TERMS)
                    if not has_forbidden:
                        return self._truncate(message.strip(), max_chars)
                    logger.warning("llm_output_has_forbidden_terms", channel=channel)
            except Exception as e:
                logger.warning("llm_format_failed", error=str(e), channel=channel)

        return self._fallback_format(decision, channel)

    def _format_with_llm(self, decision: dict, channel: str, max_chars: int) -> str:
        """Format a decision using Ollama LLM."""
        import ollama

        prompt = self._build_formatter_prompt(decision, channel, max_chars)
        t0 = time.time()
        response = ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FORMATTER},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": self._temperature, "num_predict": self._max_tokens},
        )
        elapsed = time.time() - t0
        message = response.get("message", {}).get("content", "")
        logger.info("llm_format_ok", channel=channel, chars=len(message), elapsed_s=round(elapsed, 2))
        return message

    def _fallback_format(self, decision: dict, channel: str) -> str:
        """Deterministic format without LLM. Always works."""
        iid = decision.get("intersection_id", "desconocido")
        name = self._get_display_name(iid)
        severity = decision.get("severity", "media")
        congestion = decision.get("congestion_level", 0.5)
        routes = decision.get("active_routes", [])
        emoji = self._severity_emoji.get(severity, "🟡")
        delay = self._estimate_delay(congestion)
        max_chars = self._get_max_chars(channel)

        _, plain_status, plain_advice = SEVERITY_TO_PLAIN.get(
            severity, ("amarillo", "algo de tráfico", "calcule 5-10 minutos extra")
        )
        routes_str = ", ".join(routes) if routes else ""
        tlc = decision.get("traffic_light_command", {})
        green_ext = tlc.get("green_extension_seconds", 0)
        priority_dir = tlc.get("priority_direction", "")

        if channel == "whatsapp":
            msg = f"{emoji} {plain_status.capitalize()} en {name}."
            if routes_str and SEVERITY_ORDER.get(severity, 0) >= 2:
                msg += f" Rutas alternas: {routes_str}."
            if delay > 0:
                msg += f" Calcule ~{delay} min extra."
            else:
                msg += f" {plain_advice.capitalize()}."

        elif channel == "telegram_citizen":
            msg = f"{emoji} *{name} — {plain_status.capitalize()}*\n"
            msg += f"{plain_advice.capitalize()}.\n"
            if routes_str:
                msg += f"Rutas alternativas: {routes_str}.\n"
            if delay > 0:
                msg += f"⏱ Demora estimada: ~{delay} minutos."

        elif channel == "dashboard":
            cpct = int(congestion * 100)
            now_str = datetime.now().strftime("%I:%M %p").lower()
            msg = f"{emoji} {severity.upper()} — {name}\n"
            msg += f"Nivel de congestión: {cpct}%"
            msg += f" | Demora: ~{delay} min\n" if delay > 0 else "\n"
            if green_ext > 0:
                d = priority_dir.replace("_", "-") if priority_dir else ""
                msg += f"Semáforos: Fase verde extendida {green_ext}s"
                msg += f" hacia {d}\n" if d else "\n"
            if routes_str:
                msg += f"Rutas alternativas: {routes_str}\n"
            msg += f"Última actualización: {now_str}"
        else:
            msg = f"{emoji} {name}: {plain_status}. {plain_advice}."

        return self._truncate(msg.strip(), max_chars)

    def format_batch(self, decisions: list, channel: str) -> List[Dict[str, Any]]:
        """Format all decisions. Filter by severity, order critica first."""
        min_ord = SEVERITY_ORDER.get(self._min_severity, 1)
        eligible = [d for d in decisions if SEVERITY_ORDER.get(d.get("severity", "baja"), 0) >= min_ord]
        eligible.sort(key=lambda d: SEVERITY_ORDER.get(d.get("severity", "baja"), 0), reverse=True)

        results = []
        for dec in eligible:
            message = self.format_decision(dec, channel)
            results.append({
                "intersection_id": dec.get("intersection_id", ""),
                "message": message,
                "severity": dec.get("severity", "media"),
            })
        return results

    def format_system_summary(self, final_decision: dict) -> str:
        """Generate an executive summary of the cycle for the dashboard."""
        decisions = final_decision.get("decisions", [])
        tmc = final_decision.get("global_tmc_reduction_percent", 0.0)
        monitor = final_decision.get("monitor_report", {})
        agents = monitor.get("agents_healthy", 0)
        anomalies = monitor.get("anomalies_detected", 0)
        cycle_ms = monitor.get("cycle_duration_ms", 0)

        n_crit = sum(1 for d in decisions if d.get("severity") in ("alta", "critica"))
        if n_crit == 0:
            s = "Sistema activo — tráfico normal en Sogamoso"
        else:
            s = f"Sistema activo — {n_crit} punto{'s' if n_crit > 1 else ''} crítico{'s' if n_crit > 1 else ''} en Sogamoso"

        summary = f"{s}.\nOptimización de tráfico en curso. Reducción estimada: {tmc:.0f}%.\n"
        summary += f"Agentes activos: {agents}/7"
        if anomalies > 0:
            summary += f" | Anomalías: {anomalies}"
        summary += f" | Ciclo: {cycle_ms}ms"
        return summary

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate text to max_chars, adding ellipsis if needed."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3].rstrip() + "..."
