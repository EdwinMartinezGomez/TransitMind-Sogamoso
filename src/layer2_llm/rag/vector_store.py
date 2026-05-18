"""
TransitMind Sogamoso — Vector Store Manager
============================================
Manages ChromaDB vector store for document embeddings.
Handles collection creation, persistence, and queries.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from src.shared.logger import get_logger
from src.shared.utils import get_project_root, load_yaml_config

logger = get_logger("layer2.vector_store")


class VectorStoreManager:
    """Manages ChromaDB vector store for the Sogamoso mobility knowledge base."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize vector store with ChromaDB.

        Args:
            config: RAG configuration dict. Loaded from llm_config.yaml if None.
        """
        if config is None:
            full_config = load_yaml_config("llm_config.yaml")
            config = full_config.get("rag", {})

        self._config = config
        self._persist_dir = str(
            get_project_root() / config.get(
                "chroma_persist_dir",
                "src/layer2_llm/knowledge_base/chroma_db"
            )
        )
        self._collection_name = config.get("collection_name", "sogamoso_mobility")
        
        # Define the embedding function here so ChromaDB uses it directly
        model_name = config.get("embedding_model", "nomic-embed-text")
        self._embedding_function = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=model_name,
        )

        # Ensure persist directory exists
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None
        logger.info(
            "vector_store_initialized",
            persist_dir=self._persist_dir,
            collection_name=self._collection_name,
        )

    def get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create a new one."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "collection_ready",
                name=self._collection_name,
                count=self._collection.count(),
            )
        return self._collection

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            ids: Unique identifiers for each document chunk.
            documents: Text content of each chunk.
            metadatas: Metadata for each chunk.
            embeddings: Pre-computed embeddings. If None, ChromaDB will compute them.

        Returns:
            Number of documents added.
        """
        collection = self.get_or_create_collection()
        start = time.time()

        kwargs = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        collection.add(**kwargs)
        elapsed_ms = int((time.time() - start) * 1000)

        logger.info(
            "documents_added",
            count=len(ids),
            elapsed_ms=elapsed_ms,
            total_in_collection=collection.count(),
        )
        return len(ids)

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_texts: Text queries (will be embedded by ChromaDB).
            query_embeddings: Pre-computed query embeddings.
            n_results: Number of results to return.
            where: Metadata filter.

        Returns:
            ChromaDB query results dict.
        """
        collection = self.get_or_create_collection()
        start = time.time()

        kwargs = {"n_results": min(n_results, max(collection.count(), 1))}
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        if where is not None:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        elapsed_ms = int((time.time() - start) * 1000)

        logger.debug(
            "query_executed",
            elapsed_ms=elapsed_ms,
            n_results=n_results,
        )
        return results

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection."""
        try:
            collection = self.get_or_create_collection()
            return collection.count()
        except Exception:
            return 0

    def get_collections_list(self) -> List[str]:
        """Return list of collection names."""
        try:
            collections = self._client.list_collections()
            return [c.name for c in collections]
        except Exception:
            return []

    def delete_collection(self) -> None:
        """Delete the current collection (useful for re-ingestion)."""
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = None
            logger.info("collection_deleted", name=self._collection_name)
        except Exception as e:
            logger.warning("collection_delete_failed", error=str(e))

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.delete_collection()
        self.get_or_create_collection()
        logger.info("collection_reset", name=self._collection_name)
