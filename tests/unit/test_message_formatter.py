"""
Unit tests for Layer 4 MessageFormatter.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.shared.utils import load_yaml_config
from src.layer4_bots.message_formatter import MessageFormatter, SEVERITY_ORDER, FORBIDDEN_TERMS


@pytest.fixture
def config():
    return load_yaml_config("layer4_config.yaml")


@pytest.fixture
def formatter(config):
    return MessageFormatter(config)


SAMPLE_DECISION_ALTA = {
    "intersection_id": "carrera_11_norte",
    "severity": "alta",
    "congestion_level": 0.78,
    "traffic_light_command": {
        "green_extension_seconds": 25,
        "priority_direction": "norte_sur",
        "cycle_adjustment_percent": 15,
    },
    "active_routes": ["Avenida Industrial", "Calle 14"],
    "citizen_alert": "Alerta movilidad: congestión alta.",
}

SAMPLE_DECISION_BAJA = {
    "intersection_id": "acceso_morca",
    "severity": "baja",
    "congestion_level": 0.2,
    "traffic_light_command": {"green_extension_seconds": 0, "priority_direction": "norte_sur"},
    "active_routes": [],
    "citizen_alert": "Tráfico normal.",
}


class TestMessageFormatter:
    """Tests for MessageFormatter."""

    def test_fallback_whatsapp_length(self, formatter):
        """WhatsApp fallback message should be <= 280 chars."""
        msg = formatter._fallback_format(SAMPLE_DECISION_ALTA, "whatsapp")
        assert len(msg) <= 280, f"WhatsApp message too long: {len(msg)} chars"

    def test_fallback_telegram_length(self, formatter):
        """Telegram fallback message should be <= 400 chars."""
        msg = formatter._fallback_format(SAMPLE_DECISION_ALTA, "telegram_citizen")
        assert len(msg) <= 400, f"Telegram message too long: {len(msg)} chars"

    def test_fallback_dashboard_length(self, formatter):
        """Dashboard fallback message should be <= 600 chars."""
        msg = formatter._fallback_format(SAMPLE_DECISION_ALTA, "dashboard")
        assert len(msg) <= 600, f"Dashboard message too long: {len(msg)} chars"

    def test_no_forbidden_terms_whatsapp(self, formatter):
        """WhatsApp messages should contain no technical terms."""
        msg = formatter._fallback_format(SAMPLE_DECISION_ALTA, "whatsapp")
        for term in FORBIDDEN_TERMS:
            assert term.lower() not in msg.lower(), f"Forbidden term '{term}' found in WhatsApp msg"

    def test_no_forbidden_terms_telegram(self, formatter):
        """Telegram messages should contain no technical terms."""
        msg = formatter._fallback_format(SAMPLE_DECISION_ALTA, "telegram_citizen")
        for term in FORBIDDEN_TERMS:
            assert term.lower() not in msg.lower(), f"Forbidden term '{term}' found in Telegram msg"

    def test_intersection_name_present(self, formatter):
        """Messages should contain the human-readable intersection name."""
        msg = formatter._fallback_format(SAMPLE_DECISION_ALTA, "whatsapp")
        assert "Carrera 11 Norte" in msg, f"Missing intersection name in: {msg}"

    def test_routes_present_for_alta(self, formatter):
        """Alta severity messages should include alternative routes."""
        msg = formatter._fallback_format(SAMPLE_DECISION_ALTA, "telegram_citizen")
        assert "Avenida Industrial" in msg or "Calle 14" in msg, f"Missing routes in: {msg}"

    def test_batch_excludes_baja(self, formatter):
        """format_batch should exclude 'baja' severity."""
        decisions = [SAMPLE_DECISION_ALTA, SAMPLE_DECISION_BAJA]
        batch = formatter.format_batch(decisions, "whatsapp")
        severities = [b["severity"] for b in batch]
        assert "baja" not in severities, "Baja severity should not be in batch"

    def test_batch_ordered_by_severity(self, formatter):
        """format_batch should order by severity descending."""
        decisions = [SAMPLE_DECISION_BAJA, SAMPLE_DECISION_ALTA]
        batch = formatter.format_batch(decisions, "whatsapp")
        if len(batch) > 1:
            for i in range(len(batch) - 1):
                assert SEVERITY_ORDER[batch[i]["severity"]] >= SEVERITY_ORDER[batch[i + 1]["severity"]]

    def test_estimate_delay(self, formatter):
        """Delay estimation should return reasonable values."""
        assert formatter._estimate_delay(0.2) == 0
        assert formatter._estimate_delay(0.5) == 7
        assert formatter._estimate_delay(0.7) == 15
        assert formatter._estimate_delay(0.85) == 22
        assert formatter._estimate_delay(0.95) == 35

    def test_system_summary(self, formatter):
        """System summary should mention Sogamoso and TMC reduction."""
        sample = {
            "decisions": [SAMPLE_DECISION_ALTA],
            "global_tmc_reduction_percent": 31.2,
            "monitor_report": {"agents_healthy": 7, "anomalies_detected": 0, "cycle_duration_ms": 4200},
        }
        summary = formatter.format_system_summary(sample)
        assert "Sogamoso" in summary
        assert "31" in summary
