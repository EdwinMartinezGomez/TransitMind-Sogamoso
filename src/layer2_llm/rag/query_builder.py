"""
TransitMind Sogamoso — Query Builder
=====================================
Converts Layer 1 numeric JSON data into semantic queries
in Spanish for RAG retrieval from the Sogamoso knowledge base.
"""

from typing import Any, Dict, List, Optional

from src.shared.logger import get_logger
from src.shared.constants import WEATHER_CODES

logger = get_logger("layer2.query_builder")


class QueryBuilder:
    """Builds semantic queries from Layer 1 synthetic data for RAG retrieval."""

    # Mapping of weather codes to search terms
    _WEATHER_QUERIES = {
        0: "",  # Clear — no weather-specific query
        1: "clima nublado Sogamoso",
        2: "lluvia calzada húmeda Sogamoso movilidad",
        3: "niebla matutina cuenca Chicamocha visibilidad reducida",
    }

    # Intersection-specific context queries
    _INTERSECTION_CONTEXT = {
        "carrera_11_norte": "Carrera 11 Norte ingreso centro Sogamoso",
        "carrera_11_sur": "Carrera 11 Sur terminal transporte Sogamoso",
        "av_castellana_entrada": "Avenida Castellana entrada buses intermunicipales",
        "av_castellana_salida": "Avenida Castellana salida norte Duitama",
        "calle_14_centro_historico": "Calle 14 centro histórico Sogamoso mercado",
        "acceso_morca": "acceso Morca vereda rural Sogamoso Chicamocha",
    }

    def build_query(self, layer1_data: Dict[str, Any]) -> str:
        """
        Convert Layer 1 JSON into a semantic query in Spanish.

        Analyzes the numeric data to determine which contextual factors
        are active and builds a query targeting relevant knowledge base docs.

        Args:
            layer1_data: Full response from Layer 1 /generate endpoint
                         or a single synthetic data record.

        Returns:
            Spanish semantic query string (max ~150 words).
        """
        query_parts: List[str] = []

        # Extract data — handle both full response and single record
        records = layer1_data.get("synthetic_data", [])
        metadata = layer1_data.get("metadata", {})
        intersection_id = layer1_data.get("intersection_id", "")

        # If no records but the data itself looks like a record
        if not records and "congestion_level" in layer1_data:
            records = [layer1_data]

        if not records:
            return "movilidad urbana Sogamoso tráfico congestión"

        # Use first record or compute averages
        sample = records[0] if len(records) == 1 else self._average_record(records)

        # Factor 1: Market day
        is_market_day = sample.get("is_market_day", False)
        if is_market_day:
            query_parts.append(
                "mercado campesino plaza central Sogamoso impacto tráfico "
                "vehículos carga rural horario"
            )

        # Factor 2: Weather
        weather_code = int(sample.get("weather_code", 0))
        weather_query = self._WEATHER_QUERIES.get(weather_code, "")
        if weather_query:
            query_parts.append(weather_query)

        # Factor 3: Event impact
        event_impact = float(sample.get("event_impact", 0))
        if event_impact > 0.7:
            query_parts.append(
                "evento masivo Coliseo Olímpico congestión intersecciones afectadas"
            )
        elif event_impact > 0.3:
            query_parts.append("evento Coliseo Olímpico impacto tráfico")

        # Factor 4: Congestion level
        congestion = float(sample.get("congestion_level", 0))
        if congestion > 0.8:
            query_parts.append(
                "congestión crítica rutas alternativas desvío Sogamoso"
            )
        elif congestion > 0.6:
            query_parts.append("congestión alta rutas alternativas")

        # Factor 5: Peak hour + intersection context
        is_peak = sample.get("is_peak_hour", False)
        hour = int(sample.get("hour", 12))
        if is_peak and hour < 9:
            query_parts.append("hora pico matutina flujo vehicular")
            if intersection_id in ("av_castellana_entrada", "av_castellana_salida"):
                query_parts.append(
                    "buses intermunicipales terminal Sogamoso hora pico mañana"
                )
        elif is_peak and hour >= 17:
            query_parts.append("hora pico vespertina congestión salida")

        # Factor 6: Heavy vehicle ratio
        heavy_ratio = float(sample.get("heavy_vehicle_ratio", 0))
        if heavy_ratio > 0.25:
            query_parts.append(
                "vehículos pesados buses intermunicipales carga"
            )

        # Add intersection-specific context
        if intersection_id in self._INTERSECTION_CONTEXT:
            query_parts.append(self._INTERSECTION_CONTEXT[intersection_id])

        # Build final query
        if not query_parts:
            query_parts.append("movilidad urbana Sogamoso tráfico")

        query = " ".join(query_parts)

        # Limit to ~150 words
        words = query.split()
        if len(words) > 150:
            query = " ".join(words[:150])

        logger.debug(
            "query_built",
            intersection=intersection_id,
            factors=len(query_parts),
            query_preview=query[:100],
        )
        return query

    def build_intersection_filter(
        self, intersection_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Build a ChromaDB metadata filter for intersection relevance.

        Args:
            intersection_id: The intersection to filter for.

        Returns:
            ChromaDB where filter dict, or None if no filter needed.
        """
        if not intersection_id:
            return None

        # Filter chunks that mention this intersection or are general
        return {
            "$or": [
                {"intersection_relevance": {"$contains": intersection_id}},
                {"intersection_relevance": "general"},
            ]
        }

    def _average_record(self, records: List[Dict]) -> Dict[str, Any]:
        """Compute average values from multiple records."""
        if not records:
            return {}

        n = len(records)
        avg = {}

        # Numeric fields to average
        numeric_fields = [
            "vehicle_flow", "heavy_vehicle_ratio", "motorcycle_ratio",
            "avg_speed_kmh", "congestion_level", "event_impact",
        ]
        for field in numeric_fields:
            values = [float(r.get(field, 0)) for r in records if field in r]
            avg[field] = sum(values) / len(values) if values else 0

        # Integer fields: use mode/first
        avg["weather_code"] = records[0].get("weather_code", 0)
        avg["hour"] = records[0].get("hour", 12)

        # Boolean fields: majority vote
        for bool_field in ["is_peak_hour", "is_market_day"]:
            true_count = sum(1 for r in records if r.get(bool_field, False))
            avg[bool_field] = true_count > n / 2

        return avg
