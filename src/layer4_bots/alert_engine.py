"""
TransitMind Sogamoso — Layer 4: Alert Engine
================================================
Central orchestrator that:
1. Polls /latest-decision from Layer 3
2. Filters by severity, dedup, rate limit
3. Invokes MessageFormatter for readable messages
4. Invokes SocialGraphModule for propagator ranking
5. Coordinates: first_wave → delay → broadcast
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from src.shared.logger import get_logger
from src.shared.utils import get_project_root
from src.layer4_bots.message_formatter import MessageFormatter, SEVERITY_ORDER
from src.layer4_bots.social_graph import SocialGraphModule

logger = get_logger("layer4.alert_engine")


class AlertEngine:
    """Orchestrates alert generation and dispatch for Layer 4."""

    def __init__(self, config: dict):
        self._config = config
        self._layer3_url = config.get("layer3_api", {}).get("base_url", "http://localhost:8002")
        self._poll_interval = config.get("layer3_api", {}).get("poll_interval_seconds", 30)
        self._timeout = config.get("layer3_api", {}).get("timeout_seconds", 130)
        self._min_severity = config.get("alert", {}).get("min_severity_to_alert", "media")
        self._dedup_window = config.get("alert", {}).get("dedup_window_minutes", 15)
        self._max_per_hour = config.get("alert", {}).get("max_alerts_per_hour", 10)
        self._operator_always = config.get("alert", {}).get("operator_always_notified", True)

        self._formatter = MessageFormatter(config)
        self._graph = SocialGraphModule(config)

        # Deduplication: {intersection_id: last_alert_timestamp}
        self._last_alert: Dict[str, float] = {}
        # Rate limiter: list of timestamps in last hour
        self._alert_timestamps: List[float] = []
        # Running flag
        self._running = False
        # Last processed cycle_id
        self._last_cycle_id: Optional[str] = None
        # Last alert plan
        self._latest_plan: Optional[dict] = None

    def _should_alert(self, decision: dict) -> bool:
        """
        Decide if a decision merits an alert.
        Filters: severity >= min, dedup window, rate limit.
        """
        severity = decision.get("severity", "baja")
        iid = decision.get("intersection_id", "")

        # 1. Severity filter
        min_ord = SEVERITY_ORDER.get(self._min_severity, 1)
        if SEVERITY_ORDER.get(severity, 0) < min_ord:
            return False

        # 2. Dedup: same intersection not alerted within window
        now = time.time()
        if iid in self._last_alert:
            elapsed_min = (now - self._last_alert[iid]) / 60
            if elapsed_min < self._dedup_window:
                logger.debug("dedup_skip", intersection=iid, elapsed_min=round(elapsed_min, 1))
                return False

        # 3. Rate limit
        self._alert_timestamps = [
            ts for ts in self._alert_timestamps if (now - ts) < 3600
        ]
        if len(self._alert_timestamps) >= self._max_per_hour:
            logger.warning("rate_limit_reached", count=len(self._alert_timestamps))
            return False

        return True

    def _update_dedup(self, intersection_id: str):
        """Record that this intersection was alerted now."""
        self._last_alert[intersection_id] = time.time()
        self._alert_timestamps.append(time.time())

    def _filter_and_sort_decisions(self, decisions: list) -> list:
        """Filter decisions that pass _should_alert() and sort by severity desc."""
        filtered = [d for d in decisions if self._should_alert(d)]
        filtered.sort(
            key=lambda d: SEVERITY_ORDER.get(d.get("severity", "baja"), 0),
            reverse=True,
        )
        return filtered

    async def _fetch_latest_decision(self) -> Optional[dict]:
        """GET /latest-decision from Layer 3."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._layer3_url}/latest-decision")
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    logger.debug("no_decisions_available")
                    return None
                else:
                    logger.warning("layer3_unexpected_status", status=resp.status_code)
                    return None
        except Exception as e:
            logger.error("layer3_fetch_failed", error=str(e))
            return None

    async def process_cycle(self) -> dict:
        """
        Main per-cycle process:
        1. Fetch latest_decision from Layer 3
        2. Filter and sort decisions
        3. For each: format messages, compute propagator ranking
        4. Return structured alert_plan
        """
        result = await self._fetch_latest_decision()
        if not result:
            return {"cycle_id": None, "alerts_to_send": [], "skipped": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat()}

        cycle_id = result.get("cycle_id", "")

        # Skip if same cycle already processed
        if cycle_id and cycle_id == self._last_cycle_id:
            return {"cycle_id": cycle_id, "alerts_to_send": [], "skipped": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "cycle_already_processed"}

        decisions = result.get("decisions", [])
        filtered = self._filter_and_sort_decisions(decisions)
        skipped = len(decisions) - len(filtered)

        alerts_to_send = []
        all_users = self._graph.get_all_user_ids()

        for decision in filtered:
            iid = decision.get("intersection_id", "")
            severity = decision.get("severity", "media")

            # Format messages for each channel
            messages = {
                "whatsapp": self._formatter.format_decision(decision, "whatsapp"),
                "telegram_citizen": self._formatter.format_decision(decision, "telegram_citizen"),
                "dashboard": self._formatter.format_decision(decision, "dashboard"),
            }

            # Get propagator ranking from social graph
            affected = [iid] + decision.get("active_routes", [])
            alert_order = self._graph.get_alert_order(affected, all_users)

            alerts_to_send.append({
                "intersection_id": iid,
                "severity": severity,
                "messages": messages,
                "first_wave_users": alert_order["first_wave"],
                "broadcast_users": alert_order["broadcast"],
                "broadcast_delay_minutes": alert_order["broadcast_delay_minutes"],
                "expected_coverage_pct": alert_order["expected_coverage_pct"],
                "graph_stats": alert_order["graph_stats"],
            })

            # Update dedup
            self._update_dedup(iid)

        self._last_cycle_id = cycle_id
        plan = {
            "cycle_id": cycle_id,
            "alerts_to_send": alerts_to_send,
            "skipped": skipped,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._latest_plan = plan

        # Save alert log
        self._save_alert_log(plan)

        return plan

    def _save_alert_log(self, plan: dict):
        """Persist alert plan to disk."""
        try:
            out_dir = get_project_root() / "data" / "layer4_outputs" / "alerts_sent"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"alert_{ts}.json"

            # Remove large user lists for log
            log_plan = {**plan}
            for alert in log_plan.get("alerts_to_send", []):
                alert["first_wave_count"] = len(alert.get("first_wave_users", []))
                alert["broadcast_count"] = len(alert.get("broadcast_users", []))

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(log_plan, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("save_alert_log_failed", error=str(e))

    async def run_forever(self, send_callback: Callable):
        """
        Infinite loop: process_cycle() every poll_interval_seconds.
        send_callback(alert_plan): function bots implement to send alerts.
        """
        self._running = True
        logger.info("alert_engine_started", poll_interval=self._poll_interval)

        while self._running:
            try:
                plan = await self.process_cycle()
                if plan.get("alerts_to_send"):
                    await send_callback(plan)
                    logger.info(
                        "alerts_dispatched",
                        n=len(plan["alerts_to_send"]),
                        cycle_id=plan.get("cycle_id"),
                    )
            except Exception as e:
                logger.error("alert_engine_error", error=str(e))

            await asyncio.sleep(self._poll_interval)

    def stop(self):
        """Stop the alert engine loop."""
        self._running = False
        logger.info("alert_engine_stopped")

    def get_latest_plan(self) -> Optional[dict]:
        """Return the last executed alert plan."""
        return self._latest_plan

    @property
    def formatter(self) -> MessageFormatter:
        """Expose the formatter for external use."""
        return self._formatter

    @property
    def graph(self) -> SocialGraphModule:
        """Expose the social graph for external use."""
        return self._graph
