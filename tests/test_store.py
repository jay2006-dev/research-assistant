"""Offline tests for Chroma vector store, indexing, locking, rebuild safety, and retrieval.

Uses FakeEmbeddingFunction for deterministic, zero-network, sub-second execution.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from asterline.config import Settings
from asterline.embeddings import FakeEmbeddingFunction
from asterline.store import (
    ActivePointer,
    IndexBuildLockedError,
    IndexNotInitializedError,
    PointerUpdateError,
    SingleWriterBuildLock,
    VectorStore,
    VectorStoreError,
)


@pytest.fixture
def test_settings(tmp_path):
    """Provides isolated Settings pointing to a temporary test directory."""
    chroma_dir = tmp_path / "chroma"
    pointer_file = tmp_path / "chroma_active_pointer.json"
    lock_file = tmp_path / "index_build.lock"

    # Use default corpus path from repo
    return Settings(
        chroma_persist_dir=str(chroma_dir),
        active_pointer_file=str(pointer_file),
        build_lock_file=str(lock_file),
        chroma_collection_name="test_kb",
    )


@pytest.fixture
def fake_store(test_settings):
    """Provides a VectorStore wired with deterministic FakeEmbeddingFunction."""
    embedder = FakeEmbeddingFunction()
    return VectorStore(settings=test_settings, embedder=embedder)


def test_uninitialized_index_raises(fake_store):
    """Querying an uninitialized index raises IndexNotInitializedError."""
    with pytest.raises(IndexNotInitializedError) as exc:
        fake_store.search("test query")
    assert "Index has not been initialized" in str(exc.value)


def test_empty_query_rejected(fake_store):
    """Empty or whitespace-only query strings are rejected before embedding."""
    with pytest.raises(ValueError) as exc:
        fake_store.search("   ")
    assert "Query text cannot be empty" in str(exc.value)


def test_build_index_idempotency(fake_store):
    """Indexing twice produces exactly 75 records without duplicates."""
    pointer_1 = fake_store.build_index(rebuild=False)
    assert pointer_1.chunks_count == 75

    collection_1 = fake_store.get_active_collection()
    assert collection_1.count() == 75

    # Run indexing a second time (idempotent sync)
    pointer_2 = fake_store.build_index(rebuild=False)
    assert pointer_2.chunks_count == 75
    assert pointer_2.collection_name == pointer_1.collection_name

    collection_2 = fake_store.get_active_collection()
    assert collection_2.count() == 75


def test_metadata_preservation_and_distance_semantics(fake_store):
    """Search results preserve all metadata fields, string types, and cosine distance."""
    fake_store.build_index()

    results = fake_store.search("Company overview", top_k=3)
    assert len(results) > 0

    first = results[0]
    assert hasattr(first, "distance")
    assert isinstance(first.distance, float)
    assert first.distance >= 0.0

    # Verify metadata fields and string standardization
    assert first.chunk_id != ""
    assert first.doc_id != ""
    assert first.title != ""
    assert first.category != ""
    assert first.owner != ""
    assert first.source_url.startswith("kb://")
    assert isinstance(first.published, str)
    assert isinstance(first.version, int)
    assert len(first.content) > 0

    # Verify results are sorted by distance ascending
    distances = [r.distance for r in results]
    assert distances == sorted(distances)


def test_single_writer_lock(test_settings):
    """SingleWriterBuildLock prevents concurrent index operations."""
    lock_file = test_settings.build_lock_abs_path

    with SingleWriterBuildLock(lock_file):
        assert lock_file.exists()
        # Second attempt must fail
        with pytest.raises(IndexBuildLockedError) as exc:
            with SingleWriterBuildLock(lock_file):
                pass
        assert "already in progress" in str(exc.value)

    # After exit, lockfile must be removed
    assert not lock_file.exists()


def test_rebuild_creates_new_collection_and_retains_old(fake_store):
    """Rebuilding creates a versioned collection, updates pointer, and retains old collection."""
    pointer_1 = fake_store.build_index(rebuild=False)
    old_collection_name = pointer_1.collection_name

    # Trigger clean staged rebuild
    pointer_2 = fake_store.build_index(rebuild=True)
    new_collection_name = pointer_2.collection_name

    assert old_collection_name != new_collection_name
    assert pointer_2.chunks_count == 75

    # Crucial safety requirement: previous collection MUST be retained for active readers
    old_collection = fake_store.client.get_collection(old_collection_name)
    assert old_collection.count() == 75

    # Active collection points to new collection
    active_collection = fake_store.get_active_collection()
    assert active_collection.name == new_collection_name


def test_failed_staging_preserves_active_index(fake_store):
    """If staging fails during rebuild, the active pointer and collection remain untouched."""
    pointer_1 = fake_store.build_index(rebuild=False)
    initial_collection_name = pointer_1.collection_name

    # Simulate failure during staging validation by mocking staging_collection.count to return wrong number
    with patch("chromadb.api.models.Collection.Collection.count", return_value=10):
        with pytest.raises(VectorStoreError):
            fake_store.build_index(rebuild=True)

    # Active pointer and collection must remain completely intact
    current_pointer = fake_store._get_active_pointer()
    assert current_pointer.collection_name == initial_collection_name
    assert fake_store.get_active_collection().name == initial_collection_name


def test_failed_pointer_update_preserves_active_index(fake_store):
    """If atomic pointer replacement fails, previous pointer is preserved and rebuild fails."""
    pointer_1 = fake_store.build_index(rebuild=False)
    initial_collection_name = pointer_1.collection_name

    # Simulate pointer atomic write failure (e.g. disk or permission error)
    with patch.object(
        fake_store,
        "_write_active_pointer_atomically",
        side_effect=PointerUpdateError("Simulated atomic pointer failure"),
    ):
        with pytest.raises(PointerUpdateError):
            fake_store.build_index(rebuild=True)

    # Previous pointer must still be active
    current_pointer = fake_store._get_active_pointer()
    assert current_pointer.collection_name == initial_collection_name


def test_corrupt_pointer_file_handling(fake_store, test_settings):
    """Corrupted JSON in pointer file raises IndexNotInitializedError with repair guidance."""
    test_settings.active_pointer_abs_path.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(IndexNotInitializedError) as exc:
        fake_store.search("query")
    assert "is corrupt or invalid" in str(exc.value)
