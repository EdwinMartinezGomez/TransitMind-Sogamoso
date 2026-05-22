"""
TransitMind Sogamoso — Unit Tests for Layer 3 Agents
======================================================
Tests each agent in isolation using mocked state and configuration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from datetime import datetime, timezone

from src.shared.constants import INTERSECTIONS


@pytest.fixture
def agents_config():
    return {
        "layer1_api": {"base_url": "http://localhost:8000", "timeout_seconds": 15, "n_samples_per_agent": 10},
        "layer2_api": {"base_url": "http://localhost:8001", "timeout_seconds": 40},
        "agents": {
            "predictor": {"window_size": 4, "congestion_threshold": 0.6},
            "gan_simulator": {"api_url": "http://localhost:8000", "default_n_samples": 50},
            "causal_analyst": {"api_url": "http://localhost:8001", "skip_if_low_congestion": True},
            "route_planner": {"max_alternative_routes": 3, "min_severity_for_reroute": "media"},
            "traffic_coordinator": {"max_green_extension_seconds": 45, "min_green_extension_seconds": 5, "cooldown_seconds": 120},
            "monitor": {"anomaly_threshold": 0.25, "max_fallback_rate": 0.10, "block_on_anomaly": False},
        },
        "thresholds": {"low_congestion": 0.4, "medium_congestion": 0.6, "high_congestion": 0.75, "critical_congestion": 0.85, "tmc_reduction_target": 0.30},
    }


@pytest.fixture
def sensor_data():
    high = [{"congestion_level": v, "is_peak_hour": True, "weather_code": 0, "is_market_day": False, "event_impact": 0.0} for v in [0.8, 0.75, 0.85, 0.9]]
    low = [{"congestion_level": v, "is_peak_hour": False, "weather_code": 0, "is_market_day": False, "event_impact": 0.0} for v in [0.3, 0.25]]
    return {
        "carrera_11_norte": {"synthetic_data": high, "metadata": {"intersection_id": "carrera_11_norte"}},
        "carrera_11_sur": {"synthetic_data": low, "metadata": {"intersection_id": "carrera_11_sur"}},
    }


@pytest.fixture
def causal_analysis():
    return {
        "carrera_11_norte": {
            "intersection_id": "carrera_11_norte",
            "causal_context": {"severity": "alta", "confidence": 0.85, "primary_cause": "Flujo alto", "secondary_causes": [], "causal_explanation": ""},
            "traffic_forecast": {"congestion_level_adjusted": 0.82, "peak_window": "07:00-08:30", "expected_delay_minutes": 15, "affected_intersections": []},
            "recommendations": {"traffic_light_adjustment": {"intersection_id": "carrera_11_norte", "green_phase_extension_seconds": 20, "priority_direction": "norte_sur", "rationale": ""}, "alternative_routes": ["Avenida Industrial", "Calle 14"], "citizen_alert": "Congestión alta."},
            "is_fallback": False,
        },
        "carrera_11_sur": {
            "intersection_id": "carrera_11_sur",
            "causal_context": {"severity": "baja", "confidence": 0.5, "primary_cause": "", "secondary_causes": [], "causal_explanation": ""},
            "traffic_forecast": {"congestion_level_adjusted": 0.28, "peak_window": "N/A", "expected_delay_minutes": 5, "affected_intersections": []},
            "recommendations": {"traffic_light_adjustment": {"intersection_id": "carrera_11_sur", "green_phase_extension_seconds": 0, "priority_direction": "rotacional", "rationale": ""}, "alternative_routes": [], "citizen_alert": "Normal."},
            "is_fallback": True,
        },
    }


class TestGraphState:
    def test_get_initial_state_defaults(self):
        from src.layer3_agents.graph_state import get_initial_state
        state = get_initial_state()
        assert "cycle_id" in state
        assert state["scenario"] == "normal_weekday"
        assert len(state["intersections_to_analyze"]) == 6
        assert state["error_log"] == []
        assert state["retry_count"] == 0

    def test_get_initial_state_custom(self):
        from src.layer3_agents.graph_state import get_initial_state
        state = get_initial_state(cycle_id="test-123", scenario="market_day", intersections=["carrera_11_norte"])
        assert state["cycle_id"] == "test-123"
        assert state["scenario"] == "market_day"
        assert state["intersections_to_analyze"] == ["carrera_11_norte"]


class TestPredictorAgent:
    def test_predict_high(self, agents_config, sensor_data):
        from src.layer3_agents.agent_predictor import PredictorAgent
        agent = PredictorAgent(agents_config)
        pred = agent._predict_congestion(sensor_data["carrera_11_norte"]["synthetic_data"])
        assert pred > 0.6

    def test_predict_low(self, agents_config, sensor_data):
        from src.layer3_agents.agent_predictor import PredictorAgent
        agent = PredictorAgent(agents_config)
        pred = agent._predict_congestion(sensor_data["carrera_11_sur"]["synthetic_data"])
        assert pred < 0.6

    def test_predict_empty(self, agents_config):
        from src.layer3_agents.agent_predictor import PredictorAgent
        assert PredictorAgent(agents_config)._predict_congestion([]) == 0.0

    def test_trend_rising(self, agents_config):
        from src.layer3_agents.agent_predictor import PredictorAgent
        records = [{"congestion_level": v} for v in [0.3, 0.5, 0.7, 0.9]]
        assert PredictorAgent(agents_config)._detect_trend(records) == "rising"

    def test_trend_falling(self, agents_config):
        from src.layer3_agents.agent_predictor import PredictorAgent
        records = [{"congestion_level": v} for v in [0.9, 0.7, 0.5, 0.3]]
        assert PredictorAgent(agents_config)._detect_trend(records) == "falling"

    def test_run_identifies_high(self, agents_config, sensor_data):
        from src.layer3_agents.agent_predictor import PredictorAgent
        result = PredictorAgent(agents_config).run({"sensor_data": sensor_data, "sensor_status": "ok", "error_log": []})
        assert "carrera_11_norte" in result["high_congestion_intersections"]
        assert "carrera_11_sur" not in result["high_congestion_intersections"]

    def test_run_failed_sensor(self, agents_config):
        from src.layer3_agents.agent_predictor import PredictorAgent
        result = PredictorAgent(agents_config).run({"sensor_data": {}, "sensor_status": "failed", "error_log": []})
        assert result["predictor_status"] == "failed"


class TestGANSimulatorAgent:
    def test_infer_normal(self):
        from src.layer3_agents.agent_gan_simulator import GANSimulatorAgent
        data = {"i": {"synthetic_data": [{"is_market_day": False, "weather_code": 0, "event_impact": 0.1}] * 10}}
        assert GANSimulatorAgent()._infer_active_scenario(data) == "normal_weekday"

    def test_infer_market(self):
        from src.layer3_agents.agent_gan_simulator import GANSimulatorAgent
        data = {"i": {"synthetic_data": [{"is_market_day": True, "weather_code": 0, "event_impact": 0.1}] * 10}}
        assert GANSimulatorAgent()._infer_active_scenario(data) == "market_day"

    def test_run_no_high_reuses(self, sensor_data):
        from src.layer3_agents.agent_gan_simulator import GANSimulatorAgent
        result = GANSimulatorAgent().run({"sensor_data": sensor_data, "high_congestion_intersections": [], "scenario": "normal_weekday", "error_log": []})
        assert result["gan_scenarios"] == sensor_data


class TestRoutePlannerAgent:
    def test_severity_threshold(self, agents_config):
        from src.layer3_agents.agent_route_planner import RoutePlannerAgent
        agent = RoutePlannerAgent(agents_config)
        assert agent._severity_meets_threshold("alta")
        assert agent._severity_meets_threshold("media")
        assert not agent._severity_meets_threshold("baja")

    def test_filter_congested(self, agents_config, causal_analysis):
        from src.layer3_agents.agent_route_planner import RoutePlannerAgent
        filtered = RoutePlannerAgent(agents_config)._filter_congested_routes(["Avenida Industrial", "Carrera 11"], causal_analysis)
        assert "Avenida Industrial" in filtered
        assert "Carrera 11" not in filtered

    def test_run(self, agents_config, causal_analysis):
        from src.layer3_agents.agent_route_planner import RoutePlannerAgent
        result = RoutePlannerAgent(agents_config).run({"causal_analyses": causal_analysis, "intersections_to_analyze": list(causal_analysis.keys())})
        assert len(result["route_plans"]["carrera_11_norte"]) > 0
        assert result["route_plans"]["carrera_11_sur"] == []


class TestTrafficCoordinatorAgent:
    def test_extension_alta(self, agents_config):
        from src.layer3_agents.agent_traffic_coord import TrafficCoordinatorAgent
        ext = TrafficCoordinatorAgent(agents_config)._calculate_extension({"causal_context": {"severity": "alta"}, "traffic_forecast": {"congestion_level_adjusted": 0.8}})
        assert 5 <= ext <= 45 and ext >= 15

    def test_tmc_reduction(self, agents_config):
        from src.layer3_agents.agent_traffic_coord import TrafficCoordinatorAgent
        r = TrafficCoordinatorAgent(agents_config)._estimate_tmc_reduction(25, "alta")
        assert 0.0 <= r <= 0.40

    def test_run(self, agents_config, causal_analysis):
        from src.layer3_agents.agent_traffic_coord import TrafficCoordinatorAgent
        result = TrafficCoordinatorAgent(agents_config).run({"causal_analyses": causal_analysis, "predictions": {"carrera_11_norte": 0.82, "carrera_11_sur": 0.28}, "error_log": []})
        assert "carrera_11_norte" in result["traffic_commands"]
        assert "carrera_11_sur" not in result["traffic_commands"]


class TestMonitorAgent:
    def test_coherence_ok(self, agents_config, causal_analysis):
        from src.layer3_agents.agent_monitor import MonitorAgent
        anomalies = MonitorAgent(agents_config)._check_prediction_coherence({"carrera_11_norte": 0.82, "carrera_11_sur": 0.28}, causal_analysis)
        assert len(anomalies) == 0

    def test_coherence_anomaly(self, agents_config, causal_analysis):
        from src.layer3_agents.agent_monitor import MonitorAgent
        anomalies = MonitorAgent(agents_config)._check_prediction_coherence({"carrera_11_norte": 0.40}, causal_analysis)
        assert "carrera_11_norte" in anomalies

    def test_command_sanity_valid(self, agents_config):
        from src.layer3_agents.agent_monitor import MonitorAgent
        invalid = MonitorAgent(agents_config)._check_command_sanity({"t": {"green_extension_seconds": 20, "priority_direction": "norte_sur", "cycle_adjustment_percent": 10}})
        assert len(invalid) == 0

    def test_command_sanity_invalid(self, agents_config):
        from src.layer3_agents.agent_monitor import MonitorAgent
        invalid = MonitorAgent(agents_config)._check_command_sanity({"t": {"green_extension_seconds": 100, "priority_direction": "bad", "cycle_adjustment_percent": 30}})
        assert "t" in invalid

    def test_run_approves(self, agents_config, causal_analysis):
        from src.layer3_agents.agent_monitor import MonitorAgent
        state = {"cycle_id": "test", "timestamp_start": datetime.now(timezone.utc).isoformat(), "predictions": {"carrera_11_norte": 0.82, "carrera_11_sur": 0.28}, "causal_analyses": causal_analysis, "traffic_commands": {"carrera_11_norte": {"green_extension_seconds": 20, "priority_direction": "norte_sur", "cycle_adjustment_percent": 10}}, "sensor_status": "ok", "error_log": [], "retry_count": 0, "tmc_reduction_estimate": 0.31}
        result = MonitorAgent(agents_config).run(state)
        assert result["cycle_approved"] is True
        assert result["should_retry"] is False
