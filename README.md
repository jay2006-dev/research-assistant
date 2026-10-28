# Asterline Research Assistant

AI-powered research assistant built with FastAPI, Streamlit, ChromaDB, and local sentence embeddings.

## Prerequisites

| Requirement | Minimum Version | Check Command |
|---|---|---|
| Python | 3.11+ | `python --version` |
| uv | 0.4+ | `uv --version` |
| Git | any | `git --version` |

## Quick Start

```powershell
# 1. Clone the repository
git clone <repository-url>
cd asterline-research-assistant

# 2. Install dependencies (creates .venv automatically)
uv sync --extra dev

# 3. Copy the environment template
copy .env.example .env

# 4. Verify the installation
uv run python -c "from asterline.config import Settings; print(Settings().app_name)"
# → Asterline Research Assistant
```

## Vector Index & Retrieval (Milestone 2)

The assistant uses **ChromaDB** with local embeddings (**`sentence-transformers/all-MiniLM-L6-v2`**, 384 dimensions) for local semantic search.

### CLI Commands

```powershell
# Build index (idempotent upsert into active collection):
uv run python -m asterline.cli index

# Clean rebuild (stages into a fresh versioned collection, validates, and swaps pointer):
uv run python -m asterline.cli index --rebuild

# Check index status and active collection metadata:
uv run python -m asterline.cli status

# Test retrieval probe from the terminal:
uv run python -m asterline.cli search "How much does the Launch plan cost?" --top-k 3
```

### Retrieval & Distance Semantics
Search results return an explicit `distance` field using **cosine distance** (`1 - cosine_similarity`). Lower distance values indicate greater semantic relevance ($0.0$ indicates identical orientation).

### Rebuild Architecture & Safety
- **Collision-Proof Versioning:** Rebuilds stage into unique collections (`asterline_knowledge_base_<timestamp>_<uuid>`).
- **Atomic Pointer Swap:** The active collection is tracked via `data/chroma_active_pointer.json`, swapped atomically using `os.replace`.
- **Pre-Flight Validation:** If staging validation fails, the staging collection is discarded and the active index remains untouched.
- **Reader Coordination:** Old collections are retained on disk to ensure active reader queries are not disrupted.
- **Single-Writer Build Lock:** `data/index_build.lock` prevents concurrent conflicting index builds.

### Recovery Instructions
- **Active Build Lock Error (`IndexBuildLockedError`):** If an index build operation was abruptly terminated (e.g., process killed), verify no build process is currently running and remove `data/index_build.lock`.
- **Corrupted or Missing Pointer:** If `data/chroma_active_pointer.json` is missing or corrupted, run:
  ```powershell
  uv run python -m asterline.cli index --rebuild
  ```
  This creates a fresh versioned collection and writes a healthy pointer file.

## Running Tests

### 1. Routine Offline Tests (Fast, zero network access)
Uses a deterministic `FakeEmbeddingFunction` without downloading model weights:
```powershell
uv run pytest -m "not real_model" -v
```

### 2. Real-Model Windows Smoke Test (Requires model download)
Validates local weight loading, 384-dimensional vector output, and retrieval with `sentence-transformers/all-MiniLM-L6-v2`:
```powershell
uv run pytest -m real_model -v
```

## Running the Application

### FastAPI (API server)

```powershell
uv run uvicorn asterline.main:app --reload --port 8000
```

Visit [http://localhost:8000/health](http://localhost:8000/health) to verify.

### Streamlit (UI)

```powershell
uv run streamlit run src/asterline/app.py
```

## Project Structure

```
asterline-research-assistant/
├── data/
│   ├── corpus.txt                     # Assignment dataset (plain text, read-only, never modified)
│   ├── chroma/                        # ChromaDB persistent vector store
│   └── chroma_active_pointer.json     # Atomic active collection pointer
├── src/
│   └── asterline/
│       ├── __init__.py                # Package version
│       ├── config.py                  # Central configuration
│       ├── corpus.py                  # Parser, strict validator, artifact stripper
│       ├── embeddings.py              # Embedding abstractions (Fake + SentenceTransformer)
│       ├── store.py                   # Persistent Chroma store, rebuild safety, distance semantics
│       ├── cli.py                     # CLI entry point for index, rebuild, status, search
│       ├── main.py                    # FastAPI application
│       └── app.py                     # Streamlit application
├── tests/
│   ├── test_config.py                 # Milestone 1 acceptance tests (6 tests)
│   ├── test_corpus.py                 # Corpus parser and validation tests (5 tests)
│   ├── test_store.py                  # Offline vector store, locking, idempotence tests (9 tests)
│   └── test_real_model_smoke.py       # Isolated real-model smoke test
├── .env.example                       # Environment variable template
├── .gitignore                         # Git ignore rules
├── pyproject.toml                     # Project metadata and dependencies
├── uv.lock                            # Dependency lockfile
└── README.md                          # Documentation
```

## Dataset Immutability

The dataset lives at `data/corpus.txt` and is strictly **read-only**. The parser extracts 75 valid chunks and strips only the 18 confirmed `Batch 5 | Complete candidate package <N>` pagination artifacts without modifying the underlying file. SHA-256 and modification timestamps are verified in test suites.

## License

Private — for assignment use only.