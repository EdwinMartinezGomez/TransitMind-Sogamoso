"""
TransitMind Sogamoso — Context Retriever
=======================================
Retrieves relevant knowledge base context from ChromaDB for Layer 2 analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.shared.logger import get_logger
from src.shared.utils import load_yaml_config
from src.layer2_llm.rag.vector_store import VectorStoreManager
from src.layer2_llm.rag.query_builder import QueryBuilder

logger = get_logger("layer2.retriever")


class ContextRetriever:
    """Retrieves and formats knowledge base context for a given query."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_yaml_config("llm_config.yaml")

        rag_cfg = config.get("rag", {})
        self._top_k = int(rag_cfg.get("top_k_retrieval", 4))
        self._similarity_threshold = float(rag_cfg.get("similarity_threshold", 0.65))
        self._vector_store = VectorStoreManager(rag_cfg)
        self._query_builder = QueryBuilder()

    def retrieve(
        self,
        query: str,
        intersection_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from the vector store.

        Args:
            query: Semantic query text.
            intersection_id: Optional intersection filter.
            top_k: Optional override for number of results.

        Returns:
            Raw ChromaDB query results.
        """
        if not query:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        where_filter = None
        if intersection_id:
            where_filter = self._query_builder.build_intersection_filter(intersection_id)

        results = self._vector_store.query(
            query_texts=[query],
            n_results=top_k or self._top_k,
            where=where_filter,
        )
        return results

    def format_context(self, results: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Format ChromaDB results into a compact context block.

        Args:
            results: Raw ChromaDB query results.

        Returns:
            Tuple of (context_text, sources).
        """
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        if not documents:
            return "", []

        context_blocks: List[str] = []
        sources: List[str] = []

        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None

            if distance is not None:
                similarity = max(0.0, 1.0 - float(distance))
                if similarity < self._similarity_threshold:
                    continue

            filename = str(meta.get("filename", "desconocido"))
            sources.append(filename)

            preview = doc.strip()
            if len(preview) > 1200:
                preview = preview[:1200].rstrip() + "..."

            block = f"Fuente: {filename}\n{preview}"
            context_blocks.append(block)

        if not context_blocks:
            return "", []

        context_text = "\n\n".join(context_blocks)
        unique_sources = sorted(set(sources))

        logger.debug("context_formatted", blocks=len(context_blocks))
        return context_text, unique_sources

    def retrieve_with_context(
        self,
        query: str,
        intersection_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Convenience method: retrieve and immediately format context.

        Returns:
            (context_text, sources, raw_results)
        """
        results = self.retrieve(query, intersection_id=intersection_id, top_k=top_k)
        context, sources = self.format_context(results)
        return context, sources, results
