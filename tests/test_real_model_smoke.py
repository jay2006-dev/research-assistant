"""Real runtime smoke test for sentence-transformers/all-MiniLM-L6-v2 on Windows.

Requires model weights to be downloaded locally. Separately invoked via:
    pytest -m real_model
"""

import math

import pytest

from asterline.config import Settings
from asterline.embeddings import LocalSentenceTransformerEmbedding
from asterline.store import VectorStore


@pytest.mark.real_model
def test_real_model_loading_and_dimensions():
    """Verify sentence-transformers/all-MiniLM-L6-v2 loads and outputs 384-d normalized vectors."""
    embedder = LocalSentenceTransformerEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    test_sentences = [
        "Asterline Labs provides workflow monitoring software.",
        "Pricing for Launch plan is $59 per month.",
    ]

    embeddings = embedder.embed_documents(test_sentences)

    assert len(embeddings) == 2
    for vec in embeddings:
        assert len(vec) == 384
        # Verify L2 normalization: norm should be approximately 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-3

    query_vec = embedder.embed_query("workflow runs and monitoring")
    assert len(query_vec) == 384


@pytest.mark.real_model
def test_real_model_retrieval_ranking(tmp_path):
    """Verify real embeddings produce correct semantic ranking in a Chroma collection."""
    settings = Settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        active_pointer_file=str(tmp_path / "active_pointer.json"),
        build_lock_file=str(tmp_path / "build.lock"),
        chroma_collection_name="smoke_test_collection",
    )

    embedder = LocalSentenceTransformerEmbedding(settings.embedding_model_name)
    store = VectorStore(settings=settings, embedder=embedder)

    # Build index using full corpus
    pointer = store.build_index(rebuild=True)
    assert pointer.chunks_count == 75

    # Query specifically about Launch and Team plan pricing
    results = store.search("How much does the Launch and Team subscription cost?", top_k=3)
    assert len(results) > 0

    top_result = results[0]
    # Expect pricing:0 or pricing chunk to be ranked top
    assert "pricing" in top_result.chunk_id or "pricing" in top_result.category
    assert top_result.distance >= 0.0
    assert "Launch costs $59" in top_result.content or "pricing" in top_result.doc_id
