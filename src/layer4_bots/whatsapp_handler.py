"""
TransitMind Sogamoso — Layer 4: WhatsApp Business Handler
============================================================
WhatsApp Business Cloud API v18.0 handler.
Push alerts to subscribed users + webhook for incoming messages.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from src.shared.logger import get_logger
from src.layer4_bots.message_formatter import MessageFormatter
from src.layer4_bots.social_graph import SocialGraphModule

logger = get_logger("layer4.whatsapp")


class WhatsAppHandler:
    """WhatsApp Business Cloud API handler for TransitMind Sogamoso."""

    def __init__(self, config: dict):
        wa_cfg = config.get("whatsapp", {})
        self._phone_number_id = self._resolve_env(wa_cfg.get("phone_number_id", ""))
        self._access_token = self._resolve_env(wa_cfg.get("access_token", ""))
        self._verify_token = self._resolve_env(wa_cfg.get("verify_token", ""))
        self._api_url = wa_cfg.get("api_url", "https://graph.facebook.com/v18.0")
        self._formatter = MessageFormatter(config)
        self._graph = SocialGraphModule(config)
        self._enabled = wa_cfg.get("enabled", False) and bool(self._access_token)
        self._config = config
        self._layer3_url = config.get("layer3_api", {}).get("base_url", "http://localhost:8002")

        # Track active windows (phone → last_message_timestamp)
        self._active_windows: Dict[str, float] = {}

        if not self._enabled:
            logger.info("whatsapp_disabled", reason="missing credentials or disabled in config")

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve ${ENV_VAR} references."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        return value

    def _has_active_window(self, phone: str) -> bool:
        """Check if there's an active 24h messaging window for this phone."""
        import time
        ts = self._active_windows.get(phone)
        if ts is None:
            return False
        # 24h window
        return (time.time() - ts) < 86400

    async def send_text_message(self, phone: str, message: str) -> bool:
        """
        Send a text message to a WhatsApp number.
        Uses POST /{phone_number_id}/messages.
        Returns True if successful.
        """
        if not self._enabled:
            logger.debug("whatsapp_send_skip_disabled")
            return False

        url = f"{self._api_url}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": phone,
            "type": "text",
            "text": {"preview_url": False, "body": message},
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (200, 201):
                    logger.info("whatsapp_sent", phone=phone[-4:], chars=len(message))
                    return True
                else:
                    error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    logger.warning(
                        "whatsapp_send_failed",
                        status=resp.status_code,
                        error=error_data.get("error", {}).get("message", "unknown"),
                    )
                    return False
        except Exception as e:
            logger.error("whatsapp_send_error", error=str(e))
            return False

    async def send_template_message(
        self, phone: str, template_name: str, parameters: list
    ) -> bool:
        """
        Send a pre-approved template message via Meta API.
        Used when no 24h window is active.
        """
        if not self._enabled:
            return False

        url = f"{self._api_url}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        components = []
        if parameters:
            body_params = [{"type": "text", "text": str(p)} for p in parameters]
            components.append({"type": "body", "parameters": body_params})

        payload = {
            "messaging_product": "whatsapp",
            "to": phone,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": "es"},
                "components": components,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (200, 201):
                    logger.info("whatsapp_template_sent", phone=phone[-4:], template=template_name)
                    return True
                else:
                    logger.warning("whatsapp_template_failed", status=resp.status_code)
                    return False
        except Exception as e:
            logger.error("whatsapp_template_error", error=str(e))
            return False

    async def send_alert(self, alert: dict, recipients: list):
        """
        Send an alert to a list of phone numbers.
        Uses text message if 24h window active, template otherwise.
        Respects first_wave/broadcast ordering.
        """
        if not self._enabled:
            logger.info("whatsapp_disabled_skipping")
            return

        severity = alert.get("severity", "media")
        message = alert.get("messages", {}).get("whatsapp", "")

        if not message:
            return

        sent_count = 0
        failed_count = 0

        for phone in recipients:
            if self._has_active_window(phone):
                ok = await self.send_text_message(phone, message)
            else:
                # Use template based on severity
                if severity in ("alta", "critica"):
                    template = "transit_alert_high"
                    display_names = self._config.get("intersection_display_names", {})
                    iid = alert.get("intersection_id", "")
                    name = display_names.get(iid, iid.replace("_", " ").title())
                    routes = ", ".join(alert.get("active_routes", [])[:2]) or "consulte la app"
                    ok = await self.send_template_message(phone, template, [name, message[:100], routes])
                else:
                    template = "transit_alert_medium"
                    display_names = self._config.get("intersection_display_names", {})
                    iid = alert.get("intersection_id", "")
                    name = display_names.get(iid, iid.replace("_", " ").title())
                    ok = await self.send_template_message(phone, template, [name, message[:100]])

            if ok:
                sent_count += 1
            else:
                failed_count += 1

        logger.info("whatsapp_alert_batch", sent=sent_count, failed=failed_count, severity=severity)

    async def handle_webhook(self, request_body: dict) -> dict:
        """
        Process incoming webhook from Meta (received messages).
        Returns response dict.
        """
        try:
            entry = request_body.get("entry", [])
            for e in entry:
                changes = e.get("changes", [])
                for change in changes:
                    value = change.get("value", {})
                    messages = value.get("messages", [])

                    for msg in messages:
                        phone = msg.get("from", "")
                        text = msg.get("text", {}).get("body", "")
                        msg_type = msg.get("type", "")

                        if msg_type != "text" or not text:
                            continue

                        # Register active window
                        import time
                        self._active_windows[phone] = time.time()

                        # Register query in social graph
                        user_id = f"wa_{hashlib.sha256(phone.encode()).hexdigest()[:12]}"
                        self._graph.register_user({
                            "user_id": user_id,
                            "corridors": [],
                            "peak_hours": [datetime.now().hour],
                            "query_count": 1,
                            "is_seed": False,
                            "role": "vecino",
                        })

                        # Detect intent and respond
                        intent = self._detect_intent(text)
                        response_text = await self._handle_intent(intent, user_id, text)

                        if response_text:
                            await self.send_text_message(phone, response_text)

            return {"status": "ok"}
        except Exception as e:
            logger.error("webhook_processing_error", error=str(e))
            return {"status": "error", "detail": str(e)}

    async def verify_webhook(self, mode: str, token: str, challenge: str) -> Optional[str]:
        """Verify webhook at Meta Business setup. Returns challenge if valid."""
        if mode == "subscribe" and token == self._verify_token:
            logger.info("webhook_verified")
            return challenge
        logger.warning("webhook_verification_failed", mode=mode)
        return None

    def _detect_intent(self, text: str) -> str:
        """Detect basic intent from incoming message."""
        text_lower = text.lower().strip()

        estado_keywords = ["estado", "tráfico", "trafico", "cómo está", "como esta", "trancón", "trancon"]
        if any(kw in text_lower for kw in estado_keywords):
            return "estado"

        rutas_keywords = ["ruta", "alternativa", "camino", "vía", "via", "por dónde", "por donde"]
        if any(kw in text_lower for kw in rutas_keywords):
            return "rutas"

        suscribir_keywords = ["suscribir", "avisar", "alertar", "notificar"]
        if any(kw in text_lower for kw in suscribir_keywords):
            return "suscribir"

        return "help"

    async def _handle_intent(self, intent: str, user_id: str, text: str) -> str:
        """Handle a detected intent and return response text."""
        if intent == "estado":
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{self._layer3_url}/latest-decision")
                    if resp.status_code == 200:
                        data = resp.json()
                        decisions = data.get("decisions", [])
                        # Show top 2 most severe
                        from src.layer4_bots.message_formatter import SEVERITY_ORDER
                        decisions.sort(key=lambda d: SEVERITY_ORDER.get(d.get("severity", "baja"), 0), reverse=True)
                        top = decisions[:2]

                        if not top:
                            return "🟢 Todo tranquilo en Sogamoso. ¡Buen viaje!"

                        msgs = []
                        for d in top:
                            msg = self._formatter.format_decision(d, "whatsapp")
                            msgs.append(msg)

                            # Register query
                            self._graph.register_query({
                                "user_id": user_id,
                                "intersection_id": d.get("intersection_id", ""),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })

                        return "\n\n".join(msgs)
                    else:
                        return "⚠️ No hay datos de tráfico disponibles en este momento."
            except Exception:
                return "⚠️ No pude consultar el estado del tráfico. Intenta más tarde."

        elif intent == "rutas":
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{self._layer3_url}/latest-decision")
                    if resp.status_code == 200:
                        data = resp.json()
                        decisions = [d for d in data.get("decisions", []) if d.get("active_routes")]
                        if not decisions:
                            return "🟢 No hay rutas alternativas activas. ¡Todo fluye bien!"
                        msg = "🗺️ Rutas alternativas activas:\n\n"
                        display_names = self._config.get("intersection_display_names", {})
                        for d in decisions[:3]:
                            name = display_names.get(d["intersection_id"], d["intersection_id"])
                            routes = ", ".join(d.get("active_routes", []))
                            msg += f"📍 {name}: {routes}\n"
                        return msg.strip()
            except Exception:
                return "⚠️ No pude consultar las rutas. Intenta más tarde."

        elif intent == "suscribir":
            return (
                "📱 Para recibir alertas automáticas, dime el nombre de tu "
                "corredor habitual. Por ejemplo:\n"
                "• Carrera 11 Norte\n• Av. Castellana\n• Calle 14\n• Acceso Morca"
            )

        # Default: help
        return (
            "👋 ¡Hola! Soy el bot de movilidad de Sogamoso.\n\n"
            "Puedes preguntarme:\n"
            '• "¿Cómo está el tráfico?" — Estado actual\n'
            '• "Rutas alternativas" — Vías alternas\n'
            '• "Suscribir" — Recibir alertas automáticas\n\n'
            "¡Solo escríbeme lo que necesitas!"
        )
