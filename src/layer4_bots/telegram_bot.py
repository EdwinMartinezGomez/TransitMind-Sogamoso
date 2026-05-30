"""
TransitMind Sogamoso — Layer 4: Telegram Bot
================================================
Bot for citizens and operators of Secretaría de Movilidad.
Citizens: receive alerts, query road status.
Operators: full technical alerts + management commands.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import httpx

from src.shared.logger import get_logger
from src.layer4_bots.message_formatter import MessageFormatter, SEVERITY_ORDER
from src.layer4_bots.social_graph import SocialGraphModule

logger = get_logger("layer4.telegram")

# Intersection IDs for keyboard selection
INTERSECTION_OPTIONS = [
    ("carrera_11_norte", "Carrera 11 Norte"),
    ("carrera_11_sur", "Carrera 11 Sur"),
    ("av_castellana_entrada", "Av. Castellana (entrada)"),
    ("av_castellana_salida", "Av. Castellana (salida)"),
    ("calle_14_centro_historico", "Calle 14 / Centro Histórico"),
    ("acceso_morca", "Acceso Morca"),
]


class TransitMindTelegramBot:
    """Telegram bot for TransitMind Sogamoso."""

    def __init__(self, config: dict):
        tg_cfg = config.get("telegram", {})
        self._token = tg_cfg.get("token", "")
        # Resolve env var reference
        if self._token.startswith("${") and self._token.endswith("}"):
            env_var = self._token[2:-1]
            self._token = os.environ.get(env_var, "")
        self._operator_ids: Set[int] = set()
        raw_ids = tg_cfg.get("operator_chat_ids", [])
        if isinstance(raw_ids, str):
            raw_ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
        for rid in raw_ids:
            try:
                self._operator_ids.add(int(rid))
            except (ValueError, TypeError):
                pass
        # Also from env
        env_ops = os.environ.get("TELEGRAM_OPERATOR_CHAT_IDS", "")
        if env_ops:
            for x in env_ops.split(","):
                try:
                    self._operator_ids.add(int(x.strip()))
                except (ValueError, TypeError):
                    pass

        self._enabled = tg_cfg.get("enabled", True) and bool(self._token)
        self._formatter = MessageFormatter(config)
        self._graph = SocialGraphModule(config)
        self._config = config
        self._layer3_url = config.get("layer3_api", {}).get("base_url", "http://localhost:8002")
        self._app = None

        # User subscriptions: {chat_id: {"corridors": [...], "user_id": str}}
        self._user_subs: Dict[int, dict] = {}

    def _is_operator(self, chat_id: int) -> bool:
        """True if chat_id is in the operator list."""
        return chat_id in self._operator_ids

    async def _start_handler(self, update, ctx):
        """Register user in SocialGraph and show welcome with corridor buttons."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        chat_id = update.effective_chat.id
        user_id = f"tg_{chat_id}"

        # Register in graph
        self._graph.register_user({
            "user_id": user_id,
            "corridors": [],
            "peak_hours": [datetime.now().hour],
            "query_count": 0,
            "is_seed": False,
            "role": "despachador" if self._is_operator(chat_id) else "vecino",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "last_active": datetime.now(timezone.utc).isoformat(),
        })

        self._user_subs[chat_id] = {"corridors": [], "user_id": user_id}

        # Build keyboard
        keyboard = []
        for iid, display in INTERSECTION_OPTIONS:
            keyboard.append([InlineKeyboardButton(display, callback_data=f"sub_{iid}")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        welcome = (
            "¡Hola! 👋 Soy el bot de movilidad de Sogamoso 🏙️\n\n"
            "Te avisaré cuando haya trancones en tus rutas habituales.\n\n"
            "¿Cuál es tu corredor principal? Selecciona uno:"
        )
        await update.message.reply_text(welcome, reply_markup=reply_markup)
        logger.info("user_registered", chat_id=chat_id, user_id=user_id)

    async def _corridor_callback(self, update, ctx):
        """Handle corridor selection from inline keyboard."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if not data.startswith("sub_"):
            return

        iid = data[4:]
        chat_id = query.message.chat_id
        user_id = f"tg_{chat_id}"

        # Update subscription
        if chat_id not in self._user_subs:
            self._user_subs[chat_id] = {"corridors": [], "user_id": user_id}

        corrs = self._user_subs[chat_id]["corridors"]
        if iid not in corrs:
            corrs.append(iid)

        # Update graph
        self._graph.register_user({
            "user_id": user_id,
            "corridors": [iid],
            "peak_hours": [datetime.now().hour],
            "query_count": 0,
            "is_seed": False,
        })

        # Display name
        display = dict(INTERSECTION_OPTIONS).get(iid, iid)
        await query.edit_message_text(
            f"✅ ¡Listo! Te avisaré sobre el tráfico en *{display}*.\n\n"
            f"Puedes suscribirte a más corredores con /suscribir\n"
            f"Consulta el estado actual con /estado",
            parse_mode="Markdown",
        )

    async def _estado_handler(self, update, ctx):
        """Query latest-decision and show formatted status."""
        chat_id = update.effective_chat.id
        user_id = f"tg_{chat_id}"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._layer3_url}/latest-decision")
                if resp.status_code != 200:
                    await update.message.reply_text("⚠️ No hay datos disponibles en este momento. Intenta más tarde.")
                    return
                data = resp.json()
        except Exception as e:
            await update.message.reply_text("⚠️ No pude conectarme al sistema. Intenta más tarde.")
            logger.error("estado_fetch_failed", error=str(e))
            return

        decisions = data.get("decisions", [])

        # Filter by user's corridors if any
        user_corrs = set()
        if chat_id in self._user_subs:
            user_corrs = set(self._user_subs[chat_id].get("corridors", []))

        # Check if specific intersection was requested
        args = ctx.args if ctx.args else []
        if args:
            iid = args[0].lower().replace(" ", "_")
            decisions = [d for d in decisions if d.get("intersection_id") == iid]
        elif user_corrs:
            # Show only subscribed corridors, plus any alta/critica
            decisions = [
                d for d in decisions
                if d.get("intersection_id") in user_corrs
                or d.get("severity") in ("alta", "critica")
            ]

        if not decisions:
            await update.message.reply_text("🟢 Todo tranquilo en tus rutas. ¡Buen viaje!")
            return

        # Sort by severity
        decisions.sort(key=lambda d: SEVERITY_ORDER.get(d.get("severity", "baja"), 0), reverse=True)

        # Register query
        for d in decisions:
            self._graph.register_query({
                "user_id": user_id,
                "intersection_id": d.get("intersection_id", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Format and send
        for d in decisions[:5]:  # Max 5 messages
            msg = self._formatter.format_decision(d, "telegram_citizen")
            await update.message.reply_text(msg, parse_mode="Markdown")

    async def _rutas_handler(self, update, ctx):
        """Show active alternative routes for user's corridors."""
        chat_id = update.effective_chat.id

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._layer3_url}/latest-decision")
                if resp.status_code != 200:
                    await update.message.reply_text("⚠️ No hay datos disponibles.")
                    return
                data = resp.json()
        except Exception:
            await update.message.reply_text("⚠️ No pude conectarme al sistema.")
            return

        decisions = data.get("decisions", [])
        user_corrs = set()
        if chat_id in self._user_subs:
            user_corrs = set(self._user_subs[chat_id].get("corridors", []))

        # Filter
        relevant = [d for d in decisions if d.get("intersection_id") in user_corrs or not user_corrs]
        relevant = [d for d in relevant if d.get("active_routes")]

        if not relevant:
            await update.message.reply_text("🟢 No hay rutas alternativas activas. ¡Todo fluye bien!")
            return

        msg = "🗺️ *Rutas alternativas activas*\n\n"
        for d in relevant:
            display_names = self._config.get("intersection_display_names", {})
            name = display_names.get(d["intersection_id"], d["intersection_id"])
            emoji = self._config.get("severity_emoji", {}).get(d.get("severity", "media"), "🟡")
            routes = d.get("active_routes", [])
            delay = self._formatter._estimate_delay(d.get("congestion_level", 0.5))
            msg += f"{emoji} *{name}*\n"
            for i, r in enumerate(routes, 1):
                msg += f"  {i}. {r}"
                if delay > 0:
                    msg += f" (~{delay} min extra)"
                msg += "\n"
            msg += "\n"

        await update.message.reply_text(msg.strip(), parse_mode="Markdown")

    async def _suscribir_handler(self, update, ctx):
        """Add an intersection to user's subscriptions."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = []
        for iid, display in INTERSECTION_OPTIONS:
            keyboard.append([InlineKeyboardButton(display, callback_data=f"sub_{iid}")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Selecciona el corredor al que quieres suscribirte:", reply_markup=reply_markup
        )

    async def _cancelar_handler(self, update, ctx):
        """User unsubscribes from alerts."""
        chat_id = update.effective_chat.id
        if chat_id in self._user_subs:
            del self._user_subs[chat_id]
        await update.message.reply_text(
            "✅ Listo, ya no recibirás alertas de tráfico.\n"
            "Si cambias de opinión, escribe /start para volver."
        )

    async def _ciclo_handler(self, update, ctx):
        """OPERATORS ONLY: Execute POST /run-cycle."""
        chat_id = update.effective_chat.id
        if not self._is_operator(chat_id):
            await update.message.reply_text("⛔ Este comando es solo para operadores de la Secretaría de Movilidad.")
            return

        await update.message.reply_text("⏳ Ejecutando ciclo de decisión...")

        try:
            async with httpx.AsyncClient(timeout=130) as client:
                resp = await client.post(f"{self._layer3_url}/run-cycle", json={"scenario": "normal_weekday"})
                if resp.status_code == 200:
                    result = resp.json()
                    decisions = result.get("decisions", [])
                    tmc = result.get("global_tmc_reduction_percent", 0)
                    monitor = result.get("monitor_report", {})
                    agents = monitor.get("agents_healthy", 0)

                    msg = (
                        f"✅ *Ciclo completado*\n\n"
                        f"📊 Intersecciones analizadas: {len(decisions)}\n"
                        f"📉 Reducción TMC: {tmc:.1f}%\n"
                        f"🤖 Agentes activos: {agents}/7\n"
                        f"⏱ Duración: {monitor.get('cycle_duration_ms', 0)}ms"
                    )
                    await update.message.reply_text(msg, parse_mode="Markdown")
                else:
                    await update.message.reply_text(f"⚠️ Error en el ciclo (HTTP {resp.status_code})")
        except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {str(e)[:100]}")
            logger.error("ciclo_failed", error=str(e))

    async def _sistema_handler(self, update, ctx):
        """OPERATORS ONLY: Show health of all layers."""
        chat_id = update.effective_chat.id
        if not self._is_operator(chat_id):
            await update.message.reply_text("⛔ Este comando es solo para operadores.")
            return

        services = {
            "Capa 1 (TimeGAN)": "http://localhost:8000",
            "Capa 2 (LLM+RAG)": "http://localhost:8001",
            "Capa 3 (Agentes)": "http://localhost:8002",
        }

        msg = "📊 *Sistema TransitMind Sogamoso*\n\n"
        async with httpx.AsyncClient(timeout=5) as client:
            for name, url in services.items():
                try:
                    resp = await client.get(f"{url}/health")
                    if resp.status_code == 200:
                        msg += f"✅ {name}: OK\n"
                    else:
                        msg += f"⚠️ {name}: HTTP {resp.status_code}\n"
                except Exception:
                    msg += f"❌ {name}: Sin conexión\n"

        await update.message.reply_text(msg.strip(), parse_mode="Markdown")

    async def send_alert(self, alert: dict):
        """Send an alert from AlertEngine to Telegram users."""
        if not self._enabled or not self._app:
            logger.info("telegram_not_enabled_skip_alert")
            return

        bot = self._app.bot
        severity = alert.get("severity", "media")
        msg_citizen = alert.get("messages", {}).get("telegram_citizen", "")
        first_wave = alert.get("first_wave_users", [])
        broadcast = alert.get("broadcast_users", [])
        delay_min = alert.get("broadcast_delay_minutes", 2)

        # Send to operators first (always)
        msg_dashboard = alert.get("messages", {}).get("dashboard", msg_citizen)
        for op_id in self._operator_ids:
            try:
                await bot.send_message(chat_id=op_id, text=f"🔧 OPERADOR\n{msg_dashboard}")
            except Exception as e:
                logger.warning("telegram_send_operator_failed", op_id=op_id, error=str(e))

        # First wave (propagators)
        first_wave_msg = f"📢 Aviso prioritario — compartir con vecinos\n\n{msg_citizen}"
        for uid in first_wave:
            chat_id = self._resolve_chat_id(uid)
            if chat_id:
                try:
                    await bot.send_message(chat_id=chat_id, text=first_wave_msg, parse_mode="Markdown")
                except Exception as e:
                    logger.warning("telegram_first_wave_failed", user=uid, error=str(e))

        # Wait for broadcast delay
        if broadcast and delay_min > 0:
            await asyncio.sleep(delay_min * 60)

        # Broadcast
        for uid in broadcast:
            chat_id = self._resolve_chat_id(uid)
            if chat_id:
                try:
                    await bot.send_message(chat_id=chat_id, text=msg_citizen, parse_mode="Markdown")
                except Exception as e:
                    logger.warning("telegram_broadcast_failed", user=uid, error=str(e))

    def _resolve_chat_id(self, user_id: str) -> Optional[int]:
        """Resolve a user_id to a Telegram chat_id."""
        if user_id.startswith("tg_"):
            try:
                return int(user_id[3:])
            except (ValueError, TypeError):
                return None
        # Search subscriptions
        for cid, info in self._user_subs.items():
            if info.get("user_id") == user_id:
                return cid
        return None

    def setup_handlers(self):
        """Register all handlers in self._app."""
        from telegram.ext import CommandHandler, CallbackQueryHandler

        self._app.add_handler(CommandHandler("start", self._start_handler))
        self._app.add_handler(CommandHandler("estado", self._estado_handler))
        self._app.add_handler(CommandHandler("rutas", self._rutas_handler))
        self._app.add_handler(CommandHandler("suscribir", self._suscribir_handler))
        self._app.add_handler(CommandHandler("cancelar", self._cancelar_handler))
        self._app.add_handler(CommandHandler("ciclo", self._ciclo_handler))
        self._app.add_handler(CommandHandler("sistema", self._sistema_handler))
        self._app.add_handler(CallbackQueryHandler(self._corridor_callback, pattern=r"^sub_"))

    def run(self):
        """Start the bot in polling mode."""
        if not self._enabled:
            logger.warning("telegram_bot_disabled", reason="no token or disabled in config")
            return

        from telegram.ext import Application

        self._app = Application.builder().token(self._token).build()
        self.setup_handlers()
        logger.info("telegram_bot_starting")
        self._app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    from src.shared.utils import load_yaml_config
    config = load_yaml_config("layer4_config.yaml")
    bot = TransitMindTelegramBot(config)
    bot.run()
