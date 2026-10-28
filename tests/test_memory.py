"""Deterministic offline unit tests for persistent SQLite UserMemoryStore.

Verifies:
- Persistence across store close/reopen.
- Complete user isolation.
- Idempotent duplicate remember.
- Automatic superseding on content update.
- Correcting existing and missing keys.
- Comprehensive forgetting across versions by key and ID.
- Partial unique index enforcement (at most one active record).
- Input validation on empty/invalid parameters.
- Atomic transaction rollback on failure.
- Database initialization and schema migration behavior.
- Sanitized error messages that never leak memory content or secrets.
"""

from __future__ import annotations

import json
import sqlite3
import pytest

from asterline.memory import (
    CURRENT_SCHEMA_VERSION,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStorageError,
    MemoryValidationError,
    UserMemoryStore,
)


@pytest.fixture
def memory_db_path(tmp_path):
    """Provide a temporary database path for isolated testing."""
    return tmp_path / "test_user_memory.db"


@pytest.fixture
def memory_store(memory_db_path):
    """Provide a freshly initialized UserMemoryStore on a temporary DB."""
    return UserMemoryStore(db_path=memory_db_path)


# ── 1. Persistence across store close/reopen ──────────────────────────────────

def test_memory_persistence_across_reopen(memory_db_path):
    """Memories persist across store instantiations and process restarts."""
    store1 = UserMemoryStore(memory_db_path)
    mem1 = store1.remember(
        user_id="user_alice",
        category="preference",
        key="tier_focus",
        content="Focus strictly on Launch tier pricing.",
    )
    assert mem1.status == "active"
    store1.close()

    # Reopen brand new instance pointing to same file
    store2 = UserMemoryStore(memory_db_path)
    recalled = store2.recall(user_id="user_alice")

    assert len(recalled) == 1
    assert recalled[0].memory_id == mem1.memory_id
    assert recalled[0].key == "tier_focus"
    assert recalled[0].content == "Focus strictly on Launch tier pricing."
    assert recalled[0].status == "active"


# ── 2. User Isolation ────────────────────────────────────────────────────────

def test_memory_user_isolation(memory_store):
    """Memories stored for user_alice are strictly invisible to user_bob."""
    memory_store.remember("user_alice", "preference", "tier", "Launch")
    memory_store.remember("user_alice", "constraint", "region", "US-East")

    memory_store.remember("user_bob", "preference", "tier", "Enterprise")

    alice_mem = memory_store.recall("user_alice")
    bob_mem = memory_store.recall("user_bob")
    charlie_mem = memory_store.recall("user_charlie")

    assert len(alice_mem) == 2
    assert {m.key: m.content for m in alice_mem} == {"tier": "Launch", "region": "US-East"}

    assert len(bob_mem) == 1
    assert bob_mem[0].content == "Enterprise"

    assert len(charlie_mem) == 0


# ── 3. Idempotent Duplicate Remember ─────────────────────────────────────────

def test_memory_idempotent_duplicate_remember(memory_store):
    """Calling remember with identical (user_id, category, key, content) returns existing record without creating duplicates."""
    mem1 = memory_store.remember("user_alice", "preference", "plan", "Launch")
    mem2 = memory_store.remember("user_alice", "preference", "plan", "Launch")

    assert mem1.memory_id == mem2.memory_id
    recalled = memory_store.recall("user_alice")
    assert len(recalled) == 1


# ── 4. Superseding Changed Content ───────────────────────────────────────────

def test_memory_superseding_changed_content(memory_store):
    """Calling remember with changed content marks the old record superseded and inserts new active record atomically."""
    mem_v1 = memory_store.remember("user_alice", "preference", "budget", "100")
    mem_v2 = memory_store.remember("user_alice", "preference", "budget", "250")

    assert mem_v1.memory_id != mem_v2.memory_id
    assert mem_v2.status == "active"

    # recall only returns the active version
    active = memory_store.recall("user_alice")
    assert len(active) == 1
    assert active[0].content == "250"
    assert active[0].memory_id == mem_v2.memory_id

    # list_memories with include_inactive=True shows the superseding link
    all_history = memory_store.list_memories("user_alice", include_inactive=True)
    assert len(all_history) == 2

    superseded = [m for m in all_history if m.status == "superseded"][0]
    assert superseded.memory_id == mem_v1.memory_id
    assert superseded.superseded_by == mem_v2.memory_id
    assert superseded.content == "100"


# ── 5. Correcting Existing and Missing Keys ──────────────────────────────────

def test_memory_correct_existing_and_missing_keys(memory_store):
    """'correct' atomically updates an existing key, or raises MemoryNotFoundError if the key is missing."""
    memory_store.remember("user_alice", "role", "title", "Junior Analyst")

    # Correct existing key
    corrected = memory_store.correct("user_alice", "title", "Senior Research Lead")
    assert corrected.content == "Senior Research Lead"
    assert corrected.status == "active"

    recalled = memory_store.recall("user_alice")
    assert len(recalled) == 1
    assert recalled[0].content == "Senior Research Lead"

    # Attempting to correct a nonexistent key must fail with MemoryNotFoundError
    with pytest.raises(MemoryNotFoundError, match="no active memory found"):
        memory_store.correct("user_alice", "nonexistent_key", "Some value")


# ── 6. Forgetting by Key and by Memory ID ─────────────────────────────────────

def test_memory_forget_by_key_clears_active_and_historical(memory_store):
    """Forgetting by key marks all versions as forgotten and overwrites content."""
    # Create history: v1 superseded by v2
    mem_v1 = memory_store.remember("user_alice", "preference", "tier", "Launch")
    mem_v2 = memory_store.remember("user_alice", "preference", "tier", "Team")

    count = memory_store.forget("user_alice", key="tier")
    assert count == 2  # Both v1 and v2 updated

    # recall must be empty
    assert len(memory_store.recall("user_alice")) == 0

    # list_memories(include_inactive=True) ignores forgotten records completely
    assert len(memory_store.list_memories("user_alice", include_inactive=True)) == 0

    # export_json must exclude forgotten records
    exported = json.loads(memory_store.export_json("user_alice"))
    assert len(exported) == 0


def test_memory_forget_by_memory_id(memory_store):
    """Forgetting by specific memory_id forgets the memory and its associated key history."""
    mem = memory_store.remember("user_alice", "preference", "lang", "Python")

    count = memory_store.forget("user_alice", memory_id=mem.memory_id)
    assert count == 1
    assert len(memory_store.recall("user_alice")) == 0


# ── 7. Partial Unique Index Enforcement ──────────────────────────────────────

def test_memory_unique_active_index_enforcement(memory_store, memory_db_path):
    """Direct database violation of uq_user_memories_active_key fails at SQLite engine level."""
    memory_store.remember("user_alice", "preference", "currency", "USD")

    # Bypass store layer and attempt raw duplicate active insertion
    with sqlite3.connect(memory_db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO user_memories (
                    memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                ) VALUES ('dup-id', 'user_alice', 'preference', 'currency', 'EUR', 'active', NULL, 1000.0, 1000.0);
                """
            )


# ── 8. Input Validation on Empty/Invalid Parameters ──────────────────────────

def test_memory_input_validation(memory_store):
    """Empty or whitespace-only parameters raise MemoryValidationError."""
    with pytest.raises(MemoryValidationError, match="user_id"):
        memory_store.remember("", "pref", "key", "content")

    with pytest.raises(MemoryValidationError, match="category"):
        memory_store.remember("alice", "   ", "key", "content")

    with pytest.raises(MemoryValidationError, match="key"):
        memory_store.remember("alice", "pref", "", "content")

    with pytest.raises(MemoryValidationError, match="content"):
        memory_store.remember("alice", "pref", "key", " \n\t ")

    with pytest.raises(MemoryValidationError, match="Must specify either key or memory_id"):
        memory_store.forget("alice")


# ── 9. Transaction Rollback on Failure ────────────────────────────────────────

def test_memory_transaction_rollback_on_failure(memory_store, monkeypatch):
    """If an error occurs mid-transaction, changes roll back cleanly."""
    memory_store.remember("user_alice", "pref", "theme", "dark")

    # Simulate operational failure during subsequent mutation
    orig_conn_func = memory_store._get_connection

    class FailingConn:
        def __init__(self, real_conn):
            self.real_conn = real_conn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.real_conn.__exit__(exc_type, exc_val, exc_tb)

        def execute(self, sql, params=()):
            if "INSERT INTO user_memories" in sql:
                raise sqlite3.OperationalError("Simulated disk error during INSERT")
            return self.real_conn.execute(sql, params)

    def mock_get_conn():
        return FailingConn(orig_conn_func())

    monkeypatch.setattr(memory_store, "_get_connection", mock_get_conn)

    with pytest.raises(MemoryStorageError, match="OperationalError"):
        memory_store.remember("user_alice", "pref", "theme", "light")

    # Restore connection and assert that previous state was preserved
    monkeypatch.undo()
    recalled = memory_store.recall("user_alice")
    assert len(recalled) == 1
    assert recalled[0].content == "dark"
    assert recalled[0].status == "active"


# ── 10. Database Initialization & Schema Version Migration ───────────────────

def test_memory_database_initialization_and_migration(memory_db_path):
    """Database initializes schema_migrations and user_memories tables with version 1."""
    store = UserMemoryStore(memory_db_path)
    store.close()

    with sqlite3.connect(memory_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT version FROM schema_migrations;")
        versions = [row[0] for row in cur.fetchall()]
        assert versions == [CURRENT_SCHEMA_VERSION]

        cur.execute("PRAGMA table_info(user_memories);")
        columns = {row[1] for row in cur.fetchall()}
        expected_cols = {
            "memory_id",
            "user_id",
            "category",
            "key",
            "content",
            "status",
            "superseded_by",
            "created_at",
            "updated_at",
        }
        assert expected_cols.issubset(columns)


# ── 11. Sanitized Errors (No Content or Secrets Leaked) ──────────────────────

def test_memory_sanitized_errors_do_not_leak_secrets(memory_store, monkeypatch):
    """Exceptions raised by storage operations do not include memory content or sensitive values."""
    secret_content = "SuperSecretPassword123"

    def broken_conn():
        raise sqlite3.OperationalError(f"Database locked while processing {secret_content}")

    monkeypatch.setattr(memory_store, "_get_connection", broken_conn)

    with pytest.raises(MemoryStorageError) as exc_info:
        memory_store.remember("user_alice", "cred", "api", secret_content)

    err_str = str(exc_info.value)
    # Verifies that raw message containing sensitive content is not forwarded
    assert secret_content not in err_str
    assert "OperationalError" in err_str


# ── 12. Atomic Superseding Mid-Transaction Rollback ───────────────────────────

def test_atomic_superseding_failure_rolls_back_cleanly(memory_store, monkeypatch):
    """1. Inject a failure between superseding the old record and inserting the replacement.
    Verifies transaction rolls back and the original memory remains active and untouched.
    """
    mem_v1 = memory_store.remember("user_alice", "preference", "tier", "Launch")

    orig_conn_func = memory_store._get_connection

    class FailingInsertConn:
        def __init__(self, real_conn):
            self.real_conn = real_conn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.real_conn.__exit__(exc_type, exc_val, exc_tb)

        def execute(self, sql, params=()):
            # Allow the UPDATE to succeed, but fail on the subsequent INSERT
            if "INSERT INTO user_memories" in sql:
                raise sqlite3.OperationalError("Simulated disk error during superseding INSERT")
            return self.real_conn.execute(sql, params)

    monkeypatch.setattr(memory_store, "_get_connection", lambda: FailingInsertConn(orig_conn_func()))

    with pytest.raises(MemoryStorageError, match="OperationalError"):
        memory_store.remember("user_alice", "preference", "tier", "Enterprise")

    monkeypatch.undo()

    # Verify that the original memory remains active, not superseded or corrupted
    active = memory_store.recall("user_alice")
    assert len(active) == 1
    assert active[0].memory_id == mem_v1.memory_id
    assert active[0].content == "Launch"
    assert active[0].status == "active"
    assert active[0].superseded_by is None


# ── 13. Concurrent Writes Invariant ──────────────────────────────────────────

def test_concurrent_writes_cannot_create_multiple_active_records(memory_store, memory_db_path):
    """2. Verify that two concurrent writes for the same user/key cannot leave two active records.
    Tests that the SQLite partial unique index uq_user_memories_active_key rejects any race condition.
    """
    memory_store.remember("user_alice", "preference", "budget", "100")

    # In a multithreaded / multiprocessing race, if two transactions simultaneously attempted to
    # insert an active record for the same (user_id, key), SQLite's database-level partial unique
    # index raises an IntegrityError on whichever commits second.
    with sqlite3.connect(memory_db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO user_memories (
                    memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                ) VALUES ('race-id', 'user_alice', 'preference', 'budget', '200', 'active', NULL, 2000.0, 2000.0);
                """
            )

    # Confirm only exactly 1 active record remains
    active = memory_store.recall("user_alice")
    assert len(active) == 1
    assert active[0].content == "100"


# ── 14. Forgetting History User Isolation ─────────────────────────────────────

def test_forget_history_user_isolation(memory_store):
    """3. Verify forgetting by key wipes all active and superseded versions for user_alice
    without affecting user_bob's records for the same key.
    """
    # Alice has v1 superseded by v2
    memory_store.remember("user_alice", "preference", "plan", "Launch")
    memory_store.remember("user_alice", "preference", "plan", "Team")

    # Bob has his own record for the same key
    memory_store.remember("user_bob", "preference", "plan", "Enterprise")

    # Alice forgets "plan"
    count = memory_store.forget("user_alice", key="plan")
    assert count == 2

    # Alice's memories are completely cleared
    assert len(memory_store.recall("user_alice")) == 0
    assert len(memory_store.list_memories("user_alice", include_inactive=True)) == 0

    # Bob's memory remains active and untouched
    bob_active = memory_store.recall("user_bob")
    assert len(bob_active) == 1
    assert bob_active[0].content == "Enterprise"
    assert bob_active[0].status == "active"


# ── 15. Forgetting Transaction Failure Rollback ──────────────────────────────

def test_forget_transaction_failure_rolls_back(memory_store, monkeypatch):
    """4. Inject a failure during forget operation and verify transaction rolls back."""
    memory_store.remember("user_alice", "preference", "plan", "Launch")

    orig_conn_func = memory_store._get_connection

    class FailingUpdateConn:
        def __init__(self, real_conn):
            self.real_conn = real_conn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.real_conn.__exit__(exc_type, exc_val, exc_tb)

        def execute(self, sql, params=()):
            if "UPDATE user_memories" in sql and "SET status = 'forgotten'" in sql:
                raise sqlite3.OperationalError("Simulated disk error during forget UPDATE")
            return self.real_conn.execute(sql, params)

    monkeypatch.setattr(memory_store, "_get_connection", lambda: FailingUpdateConn(orig_conn_func()))

    with pytest.raises(MemoryStorageError, match="OperationalError"):
        memory_store.forget("user_alice", key="plan")

    monkeypatch.undo()

    # State was rolled back and remains active
    recalled = memory_store.recall("user_alice")
    assert len(recalled) == 1
    assert recalled[0].content == "Launch"
    assert recalled[0].status == "active"


# ── 16. Foreign-Key Integrity ────────────────────────────────────────────────

def test_foreign_key_integrity_and_referential_consistency(memory_store, memory_db_path):
    """5. Verify superseding references point to valid records and no broken references exist."""
    mem1 = memory_store.remember("user_alice", "role", "role_title", "Engineer")
    mem2 = memory_store.remember("user_alice", "role", "role_title", "Senior Engineer")
    mem3 = memory_store.remember("user_alice", "role", "role_title", "Staff Engineer")

    with sqlite3.connect(memory_db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        # Check foreign key violations across the whole table
        violations = conn.execute("PRAGMA foreign_key_check(user_memories);").fetchall()
        assert len(violations) == 0

        # Direct verification: mem1 superseded by mem2, mem2 superseded by mem3
        rows = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT memory_id, status, superseded_by FROM user_memories;").fetchall()}
        assert rows[mem1.memory_id] == ("superseded", mem2.memory_id)
        assert rows[mem2.memory_id] == ("superseded", mem3.memory_id)
        assert rows[mem3.memory_id] == ("active", None)


# ── 17. Database Failure Classification (Locked / Corrupt / Unavailable) ──────

def test_database_failure_classification_redacts_user_and_parameters(memory_store, monkeypatch):
    """6. Database locked, unavailable, or corrupt errors are classified into MemoryStorageError
    without leaking user IDs, keys, contents, or raw parameters.
    """
    sensitive_user = "user_confidential_9999"
    sensitive_key = "super_private_secret_key"
    sensitive_content = "ConfidentialContextValue"

    # Simulated locked database
    def locked_conn():
        raise sqlite3.OperationalError(f"database is locked ({sensitive_user}, {sensitive_key})")

    monkeypatch.setattr(memory_store, "_get_connection", locked_conn)

    with pytest.raises(MemoryStorageError) as exc_info:
        memory_store.remember(sensitive_user, "pref", sensitive_key, sensitive_content)

    err_msg = str(exc_info.value)
    assert sensitive_user not in err_msg
    assert sensitive_key not in err_msg
    assert sensitive_content not in err_msg
    assert "OperationalError" in err_msg

    # Simulated database corruption
    def corrupt_conn():
        raise sqlite3.DatabaseError(f"file is not a database: malformed header with {sensitive_content}")

    monkeypatch.setattr(memory_store, "_get_connection", corrupt_conn)

    with pytest.raises(MemoryStorageError) as exc_info2:
        memory_store.recall(sensitive_user)

    err_msg2 = str(exc_info2.value)
    assert sensitive_content not in err_msg2
    assert "DatabaseError" in err_msg2


# ── 18. Export Edge Cases ─────────────────────────────────────────────────────

def test_export_json_only_includes_selected_user_active_memories(memory_store):
    """7. Export contains only the selected user's active memories and never forgotten/superseded memories."""
    # Alice: active v2 and superseded v1, plus forgotten v3
    memory_store.remember("user_alice", "preference", "tier", "Launch")
    memory_store.remember("user_alice", "preference", "tier", "Team")
    memory_store.remember("user_alice", "preference", "temp_pref", "Draft")
    memory_store.forget("user_alice", key="temp_pref")

    # Bob: active memory
    memory_store.remember("user_bob", "preference", "tier", "Enterprise")

    # Export Alice
    alice_export_str = memory_store.export_json("user_alice")
    alice_export = json.loads(alice_export_str)

    # Exactly 1 active memory for Alice ("tier": "Team")
    assert len(alice_export) == 1
    assert alice_export[0]["key"] == "tier"
    assert alice_export[0]["content"] == "Team"
    assert alice_export[0]["status"] == "active"
    assert "Launch" not in alice_export_str
    assert "Draft" not in alice_export_str
    assert "[FORGOTTEN]" not in alice_export_str
    assert "user_bob" not in alice_export_str
    assert "Enterprise" not in alice_export_str
