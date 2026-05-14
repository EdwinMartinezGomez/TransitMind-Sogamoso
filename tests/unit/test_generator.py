"""Unit tests for the generator module."""
import pytest
from src.layer1_timegan.generator import SOGAMOSO_SCENARIOS, TrafficDataGenerator


class TestSogamosoScenarios:
    def test_all_scenarios_defined(self):
        expected = ["normal_weekday", "market_day", "morning_fog", "coliseo_event", "rain_market", "duitama_blockage"]
        for s in expected:
            assert s in SOGAMOSO_SCENARIOS

    def test_scenario_keys(self):
        for name, params in SOGAMOSO_SCENARIOS.items():
            assert "weather" in params
            assert "is_market_day" in params
            assert "event_impact" in params


class TestTrafficDataGeneratorInit:
    def test_init_without_model(self, sample_config):
        gen = TrafficDataGenerator(model_path="nonexistent.pt", scaler_path="nonexistent.pkl", config=sample_config)
        assert gen.model_loaded is False

    def test_generate_raises_without_model(self, sample_config):
        gen = TrafficDataGenerator(model_path="nonexistent.pt", config=sample_config)
        with pytest.raises(RuntimeError):
            gen.generate(10, "carrera_11_norte")
