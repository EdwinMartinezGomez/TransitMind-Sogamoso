"""
TransitMind Sogamoso — Causal Analyst
=====================================
Rule-based causal analysis with optional LLM prompt construction.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.shared.constants import (
	COLISEO_NEARBY_INTERSECTIONS,
	INTERSECTIONS,
	WEATHER_CODES,
)
from src.shared.logger import get_logger
from src.shared.schemas import (
	CausalAnalysisResult,
	CausalContext,
	Recommendations,
	TrafficForecast,
	TrafficLightAdjustment,
)
from src.shared.utils import load_yaml_config
from src.layer2_llm.context_builder import ContextBuilder
from src.layer2_llm.rag_pipeline import RagPipeline

logger = get_logger("layer2.causal_analyst")


class CausalAnalyst:
	"""Main entry point for Layer 2 causal analysis."""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		if config is None:
			config = load_yaml_config("llm_config.yaml")

		self._config = config
		self._rag = RagPipeline(config)
		self._context_builder = ContextBuilder()
		self._intersection_names = {
			item.get("id"): item.get("name")
			for item in config.get("intersections", [])
			if isinstance(item, dict)
		}

	def analyze(self, payload: Dict[str, Any]) -> CausalAnalysisResult:
		"""
		Analyze Layer 1 payload and return structured causal analysis.

		Args:
			payload: Layer 1 response or a single synthetic record.

		Returns:
			CausalAnalysisResult
		"""
		start = time.time()

		records = self._extract_records(payload)
		if not records:
			raise ValueError("No synthetic data provided for analysis")

		avg = self._average_record(records)
		intersection_id = avg.get("intersection_id") or payload.get("intersection_id")
		if intersection_id not in INTERSECTIONS:
			intersection_id = INTERSECTIONS[0]

		rag_result = self._rag.build_context(payload, intersection_id=intersection_id)
		rag_context = rag_result.get("context", "")
		rag_sources = rag_result.get("sources", [])

		_ = self._context_builder.build_prompt(payload, rag_context)

		causal_context = self._build_causal_context(avg)
		forecast = self._build_forecast(avg, intersection_id)
		recommendations = self._build_recommendations(avg, intersection_id)

		processing_time_ms = int((time.time() - start) * 1000)

		result = CausalAnalysisResult(
			intersection_id=intersection_id,
			analysis_timestamp=datetime.utcnow(),
			causal_context=causal_context,
			traffic_forecast=forecast,
			recommendations=recommendations,
			rag_sources_used=rag_sources,
			llm_reasoning_trace="fallback_rule_based",
			processing_time_ms=processing_time_ms,
			model_used="fallback",
			is_fallback=True,
		)

		logger.info(
			"analysis_complete",
			intersection_id=intersection_id,
			severity=causal_context.severity,
			is_fallback=True,
		)
		return result

	def _extract_records(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
		if not isinstance(payload, dict):
			return []

		records = payload.get("synthetic_data", [])
		if records:
			return records

		if "congestion_level" in payload:
			return [payload]

		return []

	def _average_record(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
		if not records:
			return {}

		n = len(records)
		avg: Dict[str, Any] = {}

		numeric_fields = [
			"vehicle_flow",
			"heavy_vehicle_ratio",
			"motorcycle_ratio",
			"avg_speed_kmh",
			"congestion_level",
			"event_impact",
		]
		for field in numeric_fields:
			values = [float(r.get(field, 0)) for r in records if field in r]
			avg[field] = sum(values) / len(values) if values else 0

		avg["weather_code"] = int(records[0].get("weather_code", 0))
		avg["hour"] = int(records[0].get("hour", 12))
		avg["intersection_id"] = records[0].get("intersection_id", "")

		for bool_field in ["is_peak_hour", "is_market_day"]:
			true_count = sum(1 for r in records if r.get(bool_field, False))
			avg[bool_field] = true_count > n / 2

		return avg

	def _build_causal_context(self, avg: Dict[str, Any]) -> CausalContext:
		congestion = float(avg.get("congestion_level", 0))
		event_impact = float(avg.get("event_impact", 0))
		is_market_day = bool(avg.get("is_market_day", False))
		weather_code = int(avg.get("weather_code", 0))
		is_peak = bool(avg.get("is_peak_hour", False))
		heavy_ratio = float(avg.get("heavy_vehicle_ratio", 0))

		primary = "flujo vehicular habitual"
		secondary: List[str] = []

		if event_impact > 0.7:
			primary = "evento masivo cercano"
		elif is_market_day:
			primary = "mercado campesino"
		elif weather_code in (2, 3):
			primary = "condiciones climaticas adversas"
		elif is_peak:
			primary = "hora pico"
		elif heavy_ratio > 0.25:
			primary = "alta presencia de vehiculos pesados"

		if is_peak and primary != "hora pico":
			secondary.append("hora pico")
		if is_market_day and primary != "mercado campesino":
			secondary.append("mercado campesino")
		if weather_code in (2, 3) and primary != "condiciones climaticas adversas":
			secondary.append("clima adverso")
		if event_impact > 0.3 and primary != "evento masivo cercano":
			secondary.append("evento local")

		severity = self._severity_from_congestion(congestion, event_impact)
		confidence = self._confidence_score(event_impact, is_peak, weather_code)

		explanation = (
			f"La congestion estimada es {congestion:.2f} y el factor principal es "
			f"{primary}."
		)

		return CausalContext(
			primary_cause=primary,
			secondary_causes=secondary,
			causal_explanation=explanation,
			severity=severity,
			confidence=confidence,
		)

	def _build_forecast(
		self, avg: Dict[str, Any], intersection_id: str
	) -> TrafficForecast:
		congestion = float(avg.get("congestion_level", 0))
		event_impact = float(avg.get("event_impact", 0))
		is_market_day = bool(avg.get("is_market_day", False))
		is_peak = bool(avg.get("is_peak_hour", False))
		weather_code = int(avg.get("weather_code", 0))
		hour = int(avg.get("hour", 12))

		adjustment = 0.0
		if event_impact > 0.3:
			adjustment += min(0.2, event_impact * 0.2)
		if is_market_day:
			adjustment += 0.1
		if is_peak:
			adjustment += 0.05
		if weather_code in (2, 3):
			adjustment += 0.05

		congestion_adjusted = min(1.0, max(0.0, congestion + adjustment))
		peak_window = f"{hour:02d}:00 - {min(hour + 1, 23):02d}:00"
		expected_delay = min(120, int(congestion_adjusted * 60))

		affected = [intersection_id]
		if event_impact > 0.7 and intersection_id in COLISEO_NEARBY_INTERSECTIONS:
			affected = COLISEO_NEARBY_INTERSECTIONS

		return TrafficForecast(
			congestion_level_adjusted=congestion_adjusted,
			peak_window=peak_window,
			expected_delay_minutes=expected_delay,
			affected_intersections=affected,
		)

	def _build_recommendations(
		self, avg: Dict[str, Any], intersection_id: str
	) -> Recommendations:
		severity = self._severity_from_congestion(
			float(avg.get("congestion_level", 0)),
			float(avg.get("event_impact", 0)),
		)
		extension = {"baja": 5, "media": 10, "alta": 20, "critica": 30}.get(
			severity, 10
		)

		priority = self._priority_direction(intersection_id)

		adjustment = TrafficLightAdjustment(
			intersection_id=intersection_id,
			green_phase_extension_seconds=extension,
			priority_direction=priority,
			rationale=f"Ajuste por congestion {severity}.",
		)

		routes = self._suggest_routes(intersection_id)

		primary_cause = self._build_causal_context(avg).primary_cause
		intersection_name = self._intersection_names.get(
			intersection_id, intersection_id
		)
		alert = (
			f"Alerta movilidad: congestion {severity} en {intersection_name}. "
			f"Causa principal: {primary_cause}. Considere rutas alternas y "
			"reduzca velocidad."
		)
		if len(alert) > 280:
			alert = alert[:277].rstrip() + "..."

		return Recommendations(
			traffic_light_adjustment=adjustment,
			alternative_routes=routes,
			citizen_alert=alert,
		)

	def _severity_from_congestion(self, congestion: float, event_impact: float) -> str:
		score = congestion + (event_impact * 0.2)
		if score >= 0.85:
			return "critica"
		if score >= 0.7:
			return "alta"
		if score >= 0.5:
			return "media"
		return "baja"

	def _confidence_score(self, event_impact: float, is_peak: bool, weather: int) -> float:
		confidence = 0.55
		if event_impact > 0.5:
			confidence += 0.15
		if is_peak:
			confidence += 0.1
		if weather in (2, 3):
			confidence += 0.05
		return min(0.95, confidence)

	def _priority_direction(self, intersection_id: str) -> str:
		if intersection_id.startswith("carrera_11"):
			return "norte_sur"
		if intersection_id == "calle_14_centro_historico":
			return "este_oeste"
		return "rotacional"

	def _suggest_routes(self, intersection_id: str) -> List[str]:
		if intersection_id in ("av_castellana_entrada", "av_castellana_salida"):
			return [
				"Variante Morca",
				"Carrera 9 (via centro)",
			]
		if intersection_id in ("carrera_11_norte", "carrera_11_sur"):
			return [
				"Avenida Industrial",
				"Calle 14 (centro historico)",
			]
		if intersection_id == "calle_14_centro_historico":
			return [
				"Carrera 11",
				"Avenida Industrial",
			]
		if intersection_id == "acceso_morca":
			return [
				"Variante Morca",
				"Acceso norte por Carrera 9",
			]
		return ["Avenida Industrial"]
