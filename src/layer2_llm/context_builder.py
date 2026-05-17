"""
TransitMind Sogamoso — Context Builder
======================================
Builds prompt-ready context for LLM analysis from Layer 1 data and RAG context.
"""

from typing import Any, Dict, List

from src.shared.constants import WEATHER_CODES
from src.shared.logger import get_logger

logger = get_logger("layer2.context_builder")


SYSTEM_PROMPT = (
	"Eres el analista causal de trafico para TransitMind Sogamoso. "
	"Debes analizar datos sinteticos y contexto local. "
	"Responde en JSON valido y en espanol."
)


class ContextBuilder:
	"""Builds a prompt from synthetic traffic data and retrieved context."""

	def build_prompt(self, payload: Dict[str, Any], rag_context: str) -> str:
		"""
		Build a prompt string for LLM consumption.

		Args:
			payload: Layer 1 response or a single synthetic record.
			rag_context: Retrieved knowledge base context.

		Returns:
			Full prompt text.
		"""
		summary = self._summarize_payload(payload)

		prompt_parts = [
			SYSTEM_PROMPT,
			"\nDatos sinteticos (resumen):",
			summary,
		]

		if rag_context:
			prompt_parts.extend([
				"\nContexto local (RAG):",
				rag_context,
			])

		prompt_parts.append(
			"\nInstrucciones: entrega un JSON con causal_context, traffic_forecast y recommendations."
		)

		return "\n".join(prompt_parts)

	def _summarize_payload(self, payload: Dict[str, Any]) -> str:
		"""Summarize synthetic records into a compact description."""
		records = payload.get("synthetic_data", []) if isinstance(payload, dict) else []
		if not records and isinstance(payload, dict) and "congestion_level" in payload:
			records = [payload]

		if not records:
			return "Sin datos sinteticos disponibles."

		avg = self._average_record(records)

		weather_code = int(avg.get("weather_code", 0))
		weather_desc = WEATHER_CODES.get(weather_code, "desconocido")

		parts = [
			f"intersection_id: {avg.get('intersection_id', 'desconocido')}",
			f"congestion_level: {avg.get('congestion_level', 0):.2f}",
			f"avg_speed_kmh: {avg.get('avg_speed_kmh', 0):.1f}",
			f"vehicle_flow: {avg.get('vehicle_flow', 0):.1f}",
			f"weather: {weather_desc}",
			f"event_impact: {avg.get('event_impact', 0):.2f}",
			f"is_peak_hour: {avg.get('is_peak_hour', False)}",
			f"is_market_day: {avg.get('is_market_day', False)}",
		]

		return "\n".join(parts)

	def _average_record(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Compute average values from multiple records."""
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

		avg["weather_code"] = records[0].get("weather_code", 0)
		avg["hour"] = records[0].get("hour", 12)
		avg["intersection_id"] = records[0].get("intersection_id", "")

		for bool_field in ["is_peak_hour", "is_market_day"]:
			true_count = sum(1 for r in records if r.get(bool_field, False))
			avg[bool_field] = true_count > n / 2

		return avg
