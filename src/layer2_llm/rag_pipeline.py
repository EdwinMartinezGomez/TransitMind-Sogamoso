"""
TransitMind Sogamoso — RAG Pipeline
===================================
Orchestrates query building and context retrieval for Layer 2.
"""

from typing import Any, Dict, Optional

from src.shared.logger import get_logger
from src.shared.utils import load_yaml_config
from src.layer2_llm.rag.document_ingestion import DocumentIngestion
from src.layer2_llm.rag.query_builder import QueryBuilder
from src.layer2_llm.rag.retriever import ContextRetriever

logger = get_logger("layer2.rag_pipeline")


class RagPipeline:
	"""High-level RAG pipeline for Layer 2 analysis."""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		if config is None:
			config = load_yaml_config("llm_config.yaml")

		self._config = config
		self._ingestion = DocumentIngestion(config)
		self._query_builder = QueryBuilder()
		self._retriever = ContextRetriever(config)

	def ingest_if_needed(self) -> Dict[str, Any]:
		"""Ensure the vector store is populated before retrieval."""
		return self._ingestion.ingest_all(force_reingest=False)

	def build_context(
		self, layer1_payload: Dict[str, Any], intersection_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Build RAG context for the given Layer 1 payload.

		Args:
			layer1_payload: Layer 1 response or a single synthetic record.
			intersection_id: Optional override for intersection filtering.

		Returns:
			Dict with query, context, sources, and retrieval metadata.
		"""
		# Make sure there is data indexed
		ingestion_result = self.ingest_if_needed()
		if ingestion_result.get("status") == "error":
			logger.warning("ingestion_failed", detail=ingestion_result)

		inferred_intersection = intersection_id or self._extract_intersection_id(
			layer1_payload
		)
		query = self._query_builder.build_query(layer1_payload)

		context, sources, raw_results = self._retriever.retrieve_with_context(
			query, intersection_id=inferred_intersection
		)

		return {
			"query": query,
			"context": context,
			"sources": sources,
			"intersection_id": inferred_intersection,
			"raw_results": raw_results,
		}

	def _extract_intersection_id(self, payload: Dict[str, Any]) -> str:
		"""Best-effort extraction of intersection_id from payload."""
		if not isinstance(payload, dict):
			return ""

		intersection_id = payload.get("intersection_id", "")
		if intersection_id:
			return intersection_id

		records = payload.get("synthetic_data", [])
		if records and isinstance(records, list):
			return str(records[0].get("intersection_id", ""))

		return ""
