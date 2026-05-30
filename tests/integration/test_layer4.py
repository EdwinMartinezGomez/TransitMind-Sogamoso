"""
Integration tests for Layer 4.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.shared.utils import load_yaml_config
from src.layer4_bots.message_formatter import MessageFormatter
from src.layer4_bots.social_graph import SocialGraphModule
from src.layer4_bots.alert_engine import AlertEngine


@pytest.fixture
def config():
    return load_yaml_config("layer4_config.yaml")


SAMPLE_DECISIONS = [
    {
        "intersection_id": "carrera_11_norte",
        "severity": "alta",
        "congestion_level": 0.78,
        "traffic_light_command": {"green_extension_seconds": 25, "priority_direction": "norte_sur"},
        "active_routes": ["Avenida Industrial", "Calle 14"],
        "citizen_alert": "Alerta movilidad: congestión alta.",
    },
    {
        "intersection_id": "acceso_morca",
        "severity": "baja",
        "congestion_level": 0.2,
        "traffic_light_command": {"green_extension_seconds": 0, "priority_direction": "norte_sur"},
        "active_routes": [],
        "citizen_alert": "Tráfico normal.",
    },
]


class TestLayer4Integration:
    """Integration tests for Layer 4 components working together."""

    def test_formatter_graph_pipeline(self, config):
        """MessageFormatter + SocialGraph work together."""
        formatter = MessageFormatter(config)
        graph = SocialGraphModule(config)

        # Format a decision
        msg = formatter.format_decision(SAMPLE_DECISIONS[0], "whatsapp")
        assert len(msg) > 0

        # Get alert order
        all_users = graph.get_all_user_ids()
        order = graph.get_alert_order(["carrera_11_norte"], all_users)
        assert "first_wave" in order
        assert isinstance(order["first_wave"], list)

    def test_alert_engine_filtering(self, config):
        """AlertEngine correctly filters and deduplicates."""
        engine = AlertEngine(config)

        # Should alert on alta
        assert engine._should_alert(SAMPLE_DECISIONS[0]) is True
        # Should not alert on baja
        assert engine._should_alert(SAMPLE_DECISIONS[1]) is False

        # After alerting, should dedup
        engine._update_dedup("carrera_11_norte")
        assert engine._should_alert(SAMPLE_DECISIONS[0]) is False

    def test_alert_engine_rate_limit(self, config):
        """AlertEngine enforces rate limit."""
        engine = AlertEngine(config)
        max_per_hour = config.get("alert", {}).get("max_alerts_per_hour", 10)

        # Fill up rate limit
        for _ in range(max_per_hour + 2):
            engine._alert_timestamps.append(time.time())

        dec = {"intersection_id": "test_rate", "severity": "critica", "congestion_level": 0.95}
        assert engine._should_alert(dec) is False

    def test_all_modules_import(self):
        """All Layer 4 modules should import without error."""
        from src.layer4_bots import message_formatter
        from src.layer4_bots import social_graph
        from src.layer4_bots import alert_engine
        from src.layer4_bots import telegram_bot
        from src.layer4_bots import whatsapp_handler
        from src.layer4_bots import api
        assert True
