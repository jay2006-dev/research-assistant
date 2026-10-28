"""Persistent Chroma vector store module for Asterline knowledge base.

Implements:
- Cosine distance retrieval with explicit 'distance' semantics (lower = closer).
- Idempotent indexing and unique versioned staging collections.
- Single-writer build locking with process-owned cleanup.
- Atomic pointer file replacement via os.replace.
- Non-destructive rebuild (previous collections are retained for reader safety).
- Pre-flight corpus and staging validation.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, Field

from asterline.config import Settings
from asterline.corpus import CorpusChunk, ParseResult, parse_corpus
from asterline.embeddings import ChromaEmbeddingAdapter, EmbeddingFunction, LocalSentenceTransformerEmbedding

logger = logging.getLogger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────────────

class VectorStoreError(Exception):
    """Base exception for vector store operations."""


class IndexNotInitializedError(VectorStoreError):
    """Raised when querying an uninitialized or missing vector index."""


class IndexCorruptedError(VectorStoreError):
    """Raised when the active index pointer references an invalid or missing collection."""


class IndexBuildLockedError(VectorStoreError):
    """Raised when an index build is attempted while another build operation holds the lock."""


class PointerUpdateError(VectorStoreError):
    """Raised when writing or atomically replacing the active pointer file fails."""


# ── Data Models ─────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    """Retrieved search result with cosine distance semantics."""

    chunk_id: str
    doc_id: str
    title: str
    category: str
    owner: str
    source_url: str
    published: str  # Preserved string
    version: int  # Standardized integer
    content: str
    distance: float = Field(
        ...,
        description="Cosine distance (1 - cosine_similarity). Lower values indicate greater semantic relevance.",
    )


class ActivePointer(BaseModel):
    """Metadata recorded in the active pointer file."""

    collection_name: str
    created_at: str
    chunks_count: int
    corpus_sha256: str
    distance_metric: str = "cosine"


# ── Build Lock ──────────────────────────────────────────────────────────────

class SingleWriterBuildLock:
    """Exclusive filesystem lock guarding index build and rebuild operations.

    Uses atomic creation (os.O_CREAT | os.O_EXCL). Only releases the lock if this
    process instance successfully acquired it. Never automatically removes an
    existing lock held by another process.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = Path(lock_path)
        self._fd: Optional[int] = None
        self._acquired: bool = False

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Atomic creation: fails if file already exists
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            self._fd = os.open(str(self.lock_path), flags)
            payload = f"pid={os.getpid()}\ntimestamp={datetime.now(timezone.utc).isoformat()}\n"
            os.write(self._fd, payload.encode("utf-8"))
            self._acquired = True
        except FileExistsError as err:
            raise IndexBuildLockedError(
                f"An index build operation is already in progress. "
                f"Lockfile present at: {self.lock_path}. "
                "Wait for the current operation to finish or verify no build process is running."
            ) from err

    def release(self) -> None:
        if self._acquired and self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except OSError as err:
                logger.warning("Failed to remove lockfile %s: %s", self.lock_path, err)
            finally:
                self._fd = None
                self._acquired = False

    def __enter__(self) -> SingleWriterBuildLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


# ── Vector Store ────────────────────────────────────────────────────────────

class VectorStore:
    """Persistent Chroma vector store with versioned staging and safe pointer swapping."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[EmbeddingFunction] = None,
    ):
        self.settings = settings or Settings()
        self.embedder = embedder or LocalSentenceTransformerEmbedding(
            model_name=self.settings.embedding_model_name
        )
        self.persist_dir = self.settings.chroma_abs_dir
        self.pointer_path = self.settings.active_pointer_abs_path
        self.lock_path = self.settings.build_lock_abs_path
        self._chroma_client = None

    @property
    def client(self):
        if self._chroma_client is None:
            import chromadb
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._chroma_client

    def _get_active_pointer(self) -> Optional[ActivePointer]:
        """Read and validate the active collection pointer."""
        if not self.pointer_path.is_file():
            return None

        try:
            raw = self.pointer_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return ActivePointer(**data)
        except Exception as err:
            logger.error("Active pointer file at %s is corrupted: %s", self.pointer_path, err)
            raise IndexNotInitializedError(
                f"Active index pointer at {self.pointer_path} is corrupt or invalid. "
                "Run 'python -m asterline.cli index --rebuild' to repair."
            ) from err

    def _write_active_pointer_atomically(self, pointer: ActivePointer) -> None:
        """Write pointer to a temporary file and atomically replace target."""
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.pointer_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        try:
            tmp_path.write_text(pointer.model_dump_json(indent=2), encoding="utf-8")
            os.replace(str(tmp_path), str(self.pointer_path))
        except Exception as err:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise PointerUpdateError(
                f"Failed to atomically update active pointer file: {err}"
            ) from err

    def get_active_collection(self):
        """Retrieve the currently active Chroma collection.

        Raises
        ------
        IndexNotInitializedError
            If no pointer exists or pointer is corrupt.
        IndexCorruptedError
            If pointer points to a non-existent Chroma collection.
        """
        pointer = self._get_active_pointer()
        if pointer is None:
            raise IndexNotInitializedError(
                f"Index has not been initialized. No active pointer found at {self.pointer_path}. "
                "Run 'python -m asterline.cli index' to build it."
            )

        adapter = ChromaEmbeddingAdapter(self.embedder)
        try:
            return self.client.get_collection(
                name=pointer.collection_name,
                embedding_function=adapter,
            )
        except Exception as err:
            raise IndexCorruptedError(
                f"Active pointer references collection '{pointer.collection_name}', "
                f"which does not exist in Chroma. Run 'python -m asterline.cli index --rebuild' to recover."
            ) from err

    def build_index(self, rebuild: bool = False) -> ActivePointer:
        """Build or rebuild the knowledge base vector index.

        Parameters
        ----------
        rebuild : bool, default False
            If True, always stages into a fresh versioned collection and swaps pointer.
            If False and an active index exists, performs idempotent upsert into the active collection.

        Returns
        -------
        ActivePointer
            The active collection pointer after successful indexing.
        """
        import hashlib

        corpus_file = self.settings.corpus_abs_path
        if not corpus_file.is_file():
            raise FileNotFoundError(f"Corpus file not found at: {corpus_file}")

        corpus_bytes = corpus_file.read_bytes()
        corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()

        with SingleWriterBuildLock(self.lock_path):
            # Step 1: Pre-flight parse and validate complete corpus
            parse_result: ParseResult = parse_corpus(corpus_file)
            chunks: List[CorpusChunk] = parse_result.chunks

            existing_pointer = self._get_active_pointer()

            if not rebuild and existing_pointer is not None:
                # Idempotent sync into existing collection
                try:
                    collection = self.get_active_collection()
                    return self._idempotent_sync(collection, chunks, corpus_hash, existing_pointer)
                except (IndexNotInitializedError, IndexCorruptedError):
                    logger.info("Active index invalid or missing; falling back to clean versioned build.")

            # Versioned Staging Rebuild
            return self._staged_rebuild(chunks, corpus_hash)

    def _idempotent_sync(
        self,
        collection: Any,
        chunks: List[CorpusChunk],
        corpus_hash: str,
        current_pointer: ActivePointer,
    ) -> ActivePointer:
        """Idempotently sync chunks into the existing collection."""
        existing_data = collection.get()
        existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()

        chunk_ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "category": c.category,
                "owner": c.owner,
                "source_url": c.source_url,
                "published": c.published,  # Preserved as string
                "version": c.version,      # Preserved as string
            }
            for c in chunks
        ]

        # Upsert documents (idempotent: updates existing, inserts new)
        collection.upsert(
            ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
        )

        # Prune stale records if any exist
        current_chunk_ids_set = set(chunk_ids)
        stale_ids = list(existing_ids - current_chunk_ids_set)
        if stale_ids:
            collection.delete(ids=stale_ids)
            logger.info("Pruned %d stale chunks from active collection.", len(stale_ids))

        updated_pointer = ActivePointer(
            collection_name=current_pointer.collection_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            chunks_count=len(chunks),
            corpus_sha256=corpus_hash,
            distance_metric="cosine",
        )
        self._write_active_pointer_atomically(updated_pointer)
        return updated_pointer

    def _staged_rebuild(self, chunks: List[CorpusChunk], corpus_hash: str) -> ActivePointer:
        """Build into a separate versioned collection, validate, and atomically swap pointer."""
        base_name = self.settings.chroma_collection_name
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = f"{timestamp_str}_{uuid.uuid4().hex[:8]}"
        staging_collection_name = f"{base_name}_{unique_suffix}"

        adapter = ChromaEmbeddingAdapter(self.embedder)

        # Create staging collection with cosine distance metric
        staging_collection = self.client.create_collection(
            name=staging_collection_name,
            embedding_function=adapter,
            metadata={"hnsw:space": "cosine"},
        )

        staging_successful = False
        try:
            chunk_ids = [c.chunk_id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "category": c.category,
                    "owner": c.owner,
                    "source_url": c.source_url,
                    "published": c.published,
                    "version": c.version,
                }
                for c in chunks
            ]

            # Ingest all chunks
            staging_collection.add(
                ids=chunk_ids,
                documents=documents,
                metadatas=metadatas,
            )

            # Pre-flight staging validation
            staging_count = staging_collection.count()
            if staging_count != len(chunks):
                raise VectorStoreError(
                    f"Staging validation failed: expected {len(chunks)} records, found {staging_count}."
                )

            # Probe retrieval on staging collection
            sample_query = chunks[0].title
            probe_results = staging_collection.query(
                query_texts=[sample_query],
                n_results=1,
            )
            if not probe_results or not probe_results.get("ids") or not probe_results["ids"][0]:
                raise VectorStoreError("Staging validation failed: probe retrieval returned empty results.")

            staging_successful = True
        except Exception as err:
            logger.error("Staging collection population/validation failed: %s", err)
            try:
                self.client.delete_collection(name=staging_collection_name)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up staging collection: %s", cleanup_err)
            raise

        # Atomic pointer swap
        new_pointer = ActivePointer(
            collection_name=staging_collection_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            chunks_count=len(chunks),
            corpus_sha256=corpus_hash,
            distance_metric="cosine",
        )

        try:
            self._write_active_pointer_atomically(new_pointer)
        except PointerUpdateError as err:
            # Atomic swap failed: delete staging collection since it is verified inactive
            logger.error("Pointer replacement failed: %s. Cleaning up staging collection.", err)
            try:
                self.client.delete_collection(name=staging_collection_name)
            except Exception:
                pass
            raise

        # Note: We intentionally DO NOT delete the previous active collection here.
        # Active reader queries that resolved the previous pointer may still be executing.
        # Old collections are safely retained on disk.
        logger.info(
            "Successfully activated new collection '%s' with %d chunks.",
            staging_collection_name,
            len(chunks),
        )
        return new_pointer

    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Search the active index using cosine distance retrieval.

        Parameters
        ----------
        query : str
            Query string. Cannot be empty or whitespace.
        top_k : Optional[int]
            Number of top results to return. Defaults to settings.retrieval_top_k.

        Returns
        -------
        List[SearchResult]
            Ranked results with explicit 'distance' attribute (lower = closer).

        Raises
        ------
        ValueError
            If query is empty or whitespace.
        IndexNotInitializedError
            If index has not been built.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty or whitespace.")

        k = top_k or self.settings.retrieval_top_k
        collection = self.get_active_collection()

        raw = collection.query(
            query_texts=[query.strip()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        results: List[SearchResult] = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        docs = raw["documents"][0] if raw.get("documents") else [""] * len(ids)
        metas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)
        distances = raw["distances"][0] if raw.get("distances") else [0.0] * len(ids)

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    category=meta.get("category", ""),
                    owner=meta.get("owner", ""),
                    source_url=meta.get("source_url", ""),
                    published=str(meta.get("published", "")),
                    version=int(meta.get("version", 0)),
                    content=doc,
                    distance=float(dist),
                )
            )

        # Sort explicitly by distance ascending (lower distance = closer match)
        results.sort(key=lambda r: r.distance)
        return results
