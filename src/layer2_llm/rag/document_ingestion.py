"""
TransitMind Sogamoso — Document Ingestion
==========================================
Loads, chunks, embeds, and stores local knowledge base documents
into ChromaDB for RAG retrieval.
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.shared.logger import get_logger
from src.shared.utils import get_project_root, load_yaml_config
from src.layer2_llm.rag.vector_store import VectorStoreManager

logger = get_logger("layer2.ingestion")


# ============================================
# Data classes
# ============================================

@dataclass
class Document:
    """A loaded document with metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk of a document ready for embedding."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Mapping of intersection keywords found in documents to intersection IDs
_INTERSECTION_KEYWORDS = {
    "carrera_11_norte": [
        "carrera 11 norte", "carrera_11_norte", "carrera 11 n",
    ],
    "carrera_11_sur": [
        "carrera 11 sur", "carrera_11_sur", "carrera 11 s", "terminal",
    ],
    "av_castellana_entrada": [
        "castellana entrada", "av_castellana_entrada",
        "avenida castellana entrada", "castellana ingreso",
    ],
    "av_castellana_salida": [
        "castellana salida", "av_castellana_salida",
        "avenida castellana salida",
    ],
    "calle_14_centro_historico": [
        "calle 14", "calle_14_centro_historico", "centro histórico",
        "centro historico",
    ],
    "acceso_morca": [
        "acceso morca", "acceso_morca", "vereda morca", "morca",
    ],
}


def _detect_intersection_relevance(text: str) -> str:
    """Detect which intersections a text chunk is relevant to."""
    text_lower = text.lower()
    found = []
    for iid, keywords in _INTERSECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(iid)
                break
    if not found:
        return "general"
    return ",".join(sorted(set(found)))


class DocumentIngestion:
    """Loads, chunks, and indexes knowledge base documents into ChromaDB."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize document ingestion pipeline.

        Args:
            config: Full llm_config.yaml dict. Loaded automatically if None.
        """
        if config is None:
            config = load_yaml_config("llm_config.yaml")

        self._config = config
        rag_cfg = config.get("rag", {})
        self._chunk_size = rag_cfg.get("chunk_size", 500)
        self._chunk_overlap = rag_cfg.get("chunk_overlap", 50)
        self._embedding_model = rag_cfg.get("embedding_model", "nomic-embed-text")

        self._docs_dir = str(
            get_project_root() / "src" / "layer2_llm" / "knowledge_base" / "documents"
        )
        self._vector_store = VectorStoreManager(rag_cfg)
        self._ollama_available: Optional[bool] = None

    # ============================================
    # Step 1: Load documents
    # ============================================

    def load_documents(self, docs_dir: Optional[str] = None) -> List[Document]:
        """
        Load all .txt and .pdf files from the documents directory.

        Args:
            docs_dir: Path to documents directory. Uses default if None.

        Returns:
            List of Document objects with text and metadata.
        """
        docs_path = Path(docs_dir or self._docs_dir)
        if not docs_path.exists():
            logger.warning("docs_dir_not_found", path=str(docs_path))
            return []

        documents: List[Document] = []

        for fpath in sorted(docs_path.iterdir()):
            if fpath.suffix == ".txt":
                doc = self._load_txt(fpath)
                if doc:
                    documents.append(doc)
            elif fpath.suffix == ".pdf":
                doc = self._load_pdf(fpath)
                if doc:
                    documents.append(doc)

        logger.info("documents_loaded", count=len(documents))
        return documents

    def _load_txt(self, path: Path) -> Optional[Document]:
        """Load a text file."""
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                return None
            return Document(
                text=text,
                metadata={
                    "filename": path.name,
                    "doc_type": "txt",
                    "source_path": str(path),
                },
            )
        except Exception as e:
            logger.error("txt_load_failed", file=path.name, error=str(e))
            return None

    def _load_pdf(self, path: Path) -> Optional[Document]:
        """Load a PDF file using PyMuPDF."""
        try:
            import fitz  # pymupdf

            doc = fitz.open(str(path))
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()

            text = "\n".join(text_parts).strip()
            if not text:
                logger.warning("pdf_empty", file=path.name)
                return None

            return Document(
                text=text,
                metadata={
                    "filename": path.name,
                    "doc_type": "pdf",
                    "source_path": str(path),
                },
            )
        except ImportError:
            logger.warning("pymupdf_not_installed", file=path.name)
            return None
        except Exception as e:
            logger.error("pdf_load_failed", file=path.name, error=str(e))
            return None

    # ============================================
    # Step 2: Chunk documents
    # ============================================

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split documents into chunks preserving paragraph context.

        Args:
            documents: List of loaded documents.

        Returns:
            List of Chunk objects.
        """
        all_chunks: List[Chunk] = []

        for doc in documents:
            text = doc.text
            filename = doc.metadata.get("filename", "unknown")

            # Split by paragraphs first (double newline)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            current_chunk = ""
            chunk_idx = 0

            for para in paragraphs:
                # If adding this paragraph would exceed chunk_size
                if len(current_chunk) + len(para) + 2 > self._chunk_size and current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, filename, chunk_idx, doc.metadata
                    )
                    all_chunks.append(chunk)
                    chunk_idx += 1

                    # Keep overlap from end of current chunk
                    if self._chunk_overlap > 0 and len(current_chunk) > self._chunk_overlap:
                        current_chunk = current_chunk[-self._chunk_overlap:] + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

            # Don't forget the last chunk
            if current_chunk.strip():
                chunk = self._create_chunk(
                    current_chunk, filename, chunk_idx, doc.metadata
                )
                all_chunks.append(chunk)

        logger.info("documents_chunked", total_chunks=len(all_chunks))
        return all_chunks

    def _create_chunk(
        self, text: str, filename: str, position: int, doc_metadata: Dict
    ) -> Chunk:
        """Create a Chunk with proper ID and metadata."""
        chunk_id = hashlib.md5(
            f"{filename}_{position}_{text[:50]}".encode()
        ).hexdigest()

        intersection_relevance = _detect_intersection_relevance(text)

        return Chunk(
            chunk_id=f"{filename}_{position}_{chunk_id[:8]}",
            text=text.strip(),
            metadata={
                **doc_metadata,
                "chunk_position": position,
                "intersection_relevance": intersection_relevance,
                "char_count": len(text),
            },
        )

    # ============================================
    # Step 3: Embed and store
    # ============================================

    def embed_and_store(self, chunks: List[Chunk]) -> int:
        """
        Generate embeddings and store chunks in ChromaDB.

        Uses Ollama nomic-embed-text if available, otherwise falls back
        to ChromaDB's default embedding function.

        Args:
            chunks: List of chunks to embed and store.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            logger.warning("no_chunks_to_store")
            return 0

        start = time.time()

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Try to generate embeddings with Ollama
        embeddings = self._generate_embeddings_ollama(documents)

        if embeddings is not None:
            count = self._vector_store.add_documents(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )
        else:
            # Fallback: let ChromaDB handle embeddings with its default model
            logger.info("using_chromadb_default_embeddings")
            count = self._vector_store.add_documents(
                ids=ids, documents=documents, metadatas=metadatas
            )

        elapsed = time.time() - start
        logger.info(
            "embed_and_store_complete",
            chunks_indexed=count,
            elapsed_seconds=round(elapsed, 2),
        )
        return count

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and the embedding model is available."""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            import ollama as ollama_client
            # Try to pull the model if not present
            try:
                ollama_client.show(self._embedding_model)
            except Exception:
                logger.info("pulling_embedding_model", model=self._embedding_model)
                try:
                    ollama_client.pull(self._embedding_model)
                except Exception as pull_err:
                    logger.warning("ollama_pull_failed", error=str(pull_err))
                    self._ollama_available = False
                    return False

            self._ollama_available = True
            return True
        except ImportError:
            logger.warning("ollama_package_not_installed")
            self._ollama_available = False
            return False
        except Exception as e:
            logger.warning("ollama_not_available", error=str(e))
            self._ollama_available = False
            return False

    def _generate_embeddings_ollama(
        self, texts: List[str]
    ) -> Optional[List[List[float]]]:
        """Generate embeddings using Ollama nomic-embed-text."""
        if not self._check_ollama_available():
            return None

        try:
            import ollama as ollama_client

            embeddings = []
            for text in texts:
                resp = ollama_client.embed(model=self._embedding_model, input=text)
                # ollama.embed returns {"embeddings": [[...]]}
                if "embeddings" in resp and resp["embeddings"]:
                    embeddings.append(resp["embeddings"][0])
                else:
                    logger.warning("empty_embedding_response")
                    return None

            return embeddings
        except Exception as e:
            logger.warning("ollama_embedding_failed", error=str(e))
            return None

    # ============================================
    # Full pipeline
    # ============================================

    def ingest_all(self, force_reingest: bool = False) -> Dict[str, Any]:
        """
        Full ingestion pipeline: load → chunk → embed → store.

        Args:
            force_reingest: If True, delete existing collection first.

        Returns:
            Summary dict with ingestion statistics.
        """
        start = time.time()
        logger.info("ingestion_starting", force_reingest=force_reingest)

        if force_reingest:
            self._vector_store.reset()

        # Check if collection already has data
        existing_count = self._vector_store.get_collection_count()
        if existing_count > 0 and not force_reingest:
            logger.info("collection_already_populated", count=existing_count)
            return {
                "status": "skipped",
                "existing_chunks": existing_count,
                "message": "Collection already has data. Use force_reingest=True to re-index.",
            }

        # Step 1: Load
        documents = self.load_documents()
        if not documents:
            return {"status": "error", "message": "No documents found to ingest."}

        # Step 2: Chunk
        chunks = self.chunk_documents(documents)
        if not chunks:
            return {"status": "error", "message": "No chunks produced from documents."}

        # Step 3: Embed and store
        count = self.embed_and_store(chunks)

        elapsed = time.time() - start
        result = {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_indexed": count,
            "elapsed_seconds": round(elapsed, 2),
            "collection_name": self._vector_store._collection_name,
            "embedding_model": self._embedding_model,
            "ollama_used": self._ollama_available or False,
        }

        logger.info("ingestion_complete", **result)
        return result
