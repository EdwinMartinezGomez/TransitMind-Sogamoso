"""
TransitMind Sogamoso — Causal Analyst
=====================================
LLM-powered causal analysis with rule-based fallback.
Uses Ollama (llama3:8b) for primary analysis, falls back to
deterministic rules when the LLM is unavailable.
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
from src.layer2_llm.causal.output_parser import OutputParser
from src.layer2_llm.causal.prompt_templates import (
	SYSTEM_PROMPT,
	USER_PROMPT_TEMPLATE,
	FEW_SHOT_EXAMPLES,
	REPAIR_PROMPT,
)

logger = get_logger("layer2.causal_analyst")


class CausalAnalyst:
	"""Main entry point for Layer 2 causal analysis."""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		if config is None:
			config = load_yaml_config("llm_config.yaml")

		self._config = config
		self._rag = RagPipeline(config)
		self._context_builder = ContextBuilder()
		self._output_parser = OutputParser()
		self._intersection_names = {
			item.get("id"): item.get("name")
			for item in config.get("intersections", [])
			if isinstance(item, dict)
		}

		# LLM config
		llm_cfg = config.get("llm", {})
		self._provider = llm_cfg.get("provider", "ollama")
		self._model = llm_cfg.get("model", "llama3:8b")
		self._temperature = float(llm_cfg.get("temperature", 0.1))
		self._max_tokens = int(llm_cfg.get("max_tokens", 1500))
		self._timeout = int(llm_cfg.get("timeout_seconds", 30))

		# Cache LLM availability
		self._llm_available: Optional[bool] = None

	def analyze(self, payload: Dict[str, Any]) -> CausalAnalysisResult:
		"""
		Analyze Layer 1 payload and return structured causal analysis.
		Tries LLM first, falls back to rule-based if LLM fails.

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

		# Build RAG context
		rag_result = self._rag.build_context(payload, intersection_id=intersection_id)
		rag_context = rag_result.get("context", "")
		rag_sources = rag_result.get("sources", [])

		# Try LLM analysis first
		llm_result = self._analyze_with_llm(avg, intersection_id, rag_context)

		if llm_result is not None:
			# LLM succeeded — build result from LLM response
			result = self._build_result_from_llm(
				llm_result, intersection_id, rag_sources, start
			)
			logger.info(
				"analysis_complete",
				intersection_id=intersection_id,
				severity=result.causal_context.severity,
				is_fallback=False,
				model=self._model,
			)
			return result

		# Fallback to rule-based
		logger.info("using_fallback", intersection_id=intersection_id)
		return self._fallback_analysis(avg, intersection_id, rag_sources, start)

	# ============================================
	# LLM Analysis
	# ============================================

	def _check_llm_available(self) -> bool:
		"""Check if Ollama is running and the model is available."""
		if self._llm_available is not None:
			return self._llm_available

		try:
			import ollama as ollama_client

			models = ollama_client.list()
			model_names = []
			if hasattr(models, "models"):
				model_names = [m.model for m in models.models]
			elif isinstance(models, dict):
				model_names = [m.get("name", "") for m in models.get("models", [])]

			# Check if our model is available (with or without tag)
			base_model = self._model.split(":")[0]
			available = any(
				base_model in name for name in model_names
			)

			if not available:
				logger.info("model_not_found_pulling", model=self._model)
				try:
					ollama_client.pull(self._model)
					available = True
				except Exception as pull_err:
					logger.warning("model_pull_failed", error=str(pull_err))
					available = False

			self._llm_available = available
			return available

		except ImportError:
			logger.warning("ollama_package_not_installed")
			self._llm_available = False
			return False
		except Exception as e:
			logger.warning("ollama_not_available", error=str(e))
			self._llm_available = False
			return False

	def _analyze_with_llm(
		self, avg: Dict[str, Any], intersection_id: str, rag_context: str
	) -> Optional[Dict[str, Any]]:
		"""
		Perform causal analysis using Ollama LLM.

		Returns:
			Parsed and validated dict from LLM, or None on failure.
		"""
		if not self._check_llm_available():
			return None

		try:
			import ollama as ollama_client

			# Build the user prompt
			weather_code = int(avg.get("weather_code", 0))
			weather_desc = WEATHER_CODES.get(weather_code, "desconocido")

			user_prompt = USER_PROMPT_TEMPLATE.format(
				intersection_id=intersection_id,
				intersection_name=self._intersection_names.get(
					intersection_id, intersection_id
				),
				hour=int(avg.get("hour", 12)),
				is_peak_hour=bool(avg.get("is_peak_hour", False)),
				is_market_day=bool(avg.get("is_market_day", False)),
				weather_description=weather_desc,
				weather_code=weather_code,
				vehicle_flow=float(avg.get("vehicle_flow", 0)),
				avg_speed_kmh=float(avg.get("avg_speed_kmh", 0)),
				congestion_level=float(avg.get("congestion_level", 0)),
				heavy_vehicle_ratio=float(avg.get("heavy_vehicle_ratio", 0)),
				motorcycle_ratio=float(avg.get("motorcycle_ratio", 0)),
				event_impact=float(avg.get("event_impact", 0)),
				rag_context=rag_context or "No hay contexto local disponible.",
			)

			# Build messages with few-shot examples
			messages = [
				{"role": "system", "content": SYSTEM_PROMPT},
			]
			messages.extend(FEW_SHOT_EXAMPLES)
			messages.append({"role": "user", "content": user_prompt})

			# Call Ollama
			logger.info(
				"llm_call_starting",
				model=self._model,
				intersection=intersection_id,
			)

			response = ollama_client.chat(
				model=self._model,
				messages=messages,
				options={
					"temperature": self._temperature,
					"num_predict": self._max_tokens,
				},
			)

			# Extract response text
			response_text = ""
			if hasattr(response, "message"):
				response_text = response.message.content
			elif isinstance(response, dict):
				response_text = response.get("message", {}).get("content", "")

			if not response_text:
				logger.warning("empty_llm_response")
				return None

			logger.debug(
				"llm_response_received",
				length=len(response_text),
				preview=response_text[:200],
			)

			# Parse JSON from response
			parsed = self._output_parser.parse(response_text)
			if parsed is None:
				# Try repair with LLM
				parsed = self._try_repair_json(response_text)

			if parsed is None:
				logger.warning("json_parse_failed", response_preview=response_text[:300])
				return None

			# Validate and fill defaults
			validated = self._output_parser.validate_schema(parsed)

			# Ensure intersection_id is set
			if validated.get("recommendations", {}).get("traffic_light_adjustment", {}).get("intersection_id") == "":
				validated["recommendations"]["traffic_light_adjustment"]["intersection_id"] = intersection_id

			return validated

		except Exception as e:
			logger.error("llm_analysis_failed", error=str(e))
			return None

	def _try_repair_json(self, broken_text: str) -> Optional[Dict[str, Any]]:
		"""Last resort: ask the LLM to repair its own broken JSON."""
		try:
			import ollama as ollama_client

			repair_prompt = REPAIR_PROMPT.format(broken_json=broken_text[:2000])

			response = ollama_client.chat(
				model=self._model,
				messages=[{"role": "user", "content": repair_prompt}],
				options={"temperature": 0.0, "num_predict": self._max_tokens},
			)

			response_text = ""
			if hasattr(response, "message"):
				response_text = response.message.content
			elif isinstance(response, dict):
				response_text = response.get("message", {}).get("content", "")

			if response_text:
				return self._output_parser.parse(response_text)

		except Exception as e:
			logger.warning("json_repair_failed", error=str(e))

		return None

	def _build_result_from_llm(
		self,
		llm_data: Dict[str, Any],
		intersection_id: str,
		rag_sources: List[str],
		start_time: float,
	) -> CausalAnalysisResult:
		"""Build CausalAnalysisResult from validated LLM output."""
		ctx_data = llm_data.get("causal_context", {})
		fc_data = llm_data.get("traffic_forecast", {})
		rec_data = llm_data.get("recommendations", {})
		adj_data = rec_data.get("traffic_light_adjustment", {})

		causal_context = CausalContext(
			primary_cause=ctx_data.get("primary_cause", "flujo vehicular habitual"),
			secondary_causes=ctx_data.get("secondary_causes", []),
			causal_explanation=ctx_data.get("causal_explanation", ""),
			severity=ctx_data.get("severity", "media"),
			confidence=ctx_data.get("confidence", 0.5),
		)

		forecast = TrafficForecast(
			congestion_level_adjusted=fc_data.get("congestion_level_adjusted", 0.5),
			peak_window=fc_data.get("peak_window", "07:00 - 08:00"),
			expected_delay_minutes=fc_data.get("expected_delay_minutes", 10),
			affected_intersections=fc_data.get("affected_intersections", [intersection_id]),
		)

		adjustment = TrafficLightAdjustment(
			intersection_id=adj_data.get("intersection_id", intersection_id),
			green_phase_extension_seconds=adj_data.get("green_phase_extension_seconds", 10),
			priority_direction=adj_data.get("priority_direction", "rotacional"),
			rationale=adj_data.get("rationale", "Ajuste sugerido por LLM."),
		)

		citizen_alert = rec_data.get("citizen_alert", "Revise condiciones de tráfico.")
		if len(citizen_alert) > 280:
			citizen_alert = citizen_alert[:277].rstrip() + "..."

		recommendations = Recommendations(
			traffic_light_adjustment=adjustment,
			alternative_routes=rec_data.get("alternative_routes", []),
			citizen_alert=citizen_alert,
		)

		processing_time_ms = int((time.time() - start_time) * 1000)

		return CausalAnalysisResult(
			intersection_id=intersection_id,
			analysis_timestamp=datetime.utcnow(),
			causal_context=causal_context,
			traffic_forecast=forecast,
			recommendations=recommendations,
			rag_sources_used=rag_sources,
			llm_reasoning_trace="llm_analysis",
			processing_time_ms=processing_time_ms,
			model_used=self._model,
			is_fallback=False,
		)

	# ============================================
	# Fallback (rule-based) — preserved from original
	# ============================================

	def _fallback_analysis(
		self,
		avg: Dict[str, Any],
		intersection_id: str,
		rag_sources: List[str],
		start_time: float,
	) -> CausalAnalysisResult:
		"""Rule-based fallback when LLM is unavailable."""
		_ = self._context_builder.build_prompt(
			{"synthetic_data": [avg]}, ""
		)

		causal_context = self._build_causal_context(avg)
		forecast = self._build_forecast(avg, intersection_id)
		recommendations = self._build_recommendations(avg, intersection_id)

		processing_time_ms = int((time.time() - start_time) * 1000)

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

	# ============================================
	# Helpers (unchanged from original)
	# ============================================

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
