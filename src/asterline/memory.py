"""Persistent SQLite user memory storage layer for Asterline Research Assistant.

Provides structured, scoped memory persistence for user preferences, constraints,
and working context.

Design & Security Invariants
-----------------------------
1. Complete isolation: Every read, write, update, and deletion is strictly scoped to an explicit ``user_id``.
2. Database-level partial unique index: Enforces at most one 'active' record per ``(user_id, key)``.
3. Atomic mutations: Idempotent duplicate remember; changed content supersedes the prior active record atomically.
4. Comprehensive forget: ``forget`` wipes and marks content 'forgotten' across all versions (active and historical).
5. Sanitized errors: Exception strings never contain memory content, API secrets, or raw SQL parameter bindings.
6. Separation of concerns: Stored memories are personal context, NEVER authoritative corporate evidence.
7. Journaling limitation notice: Soft-deletes and content overwrites remove records from database queries;
   standard SQLite delete/update does not perform cryptographic forensic sanitization of underlying OS blocks
   or WAL files without VACUUM / zero-fill.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1


class MemoryItem(BaseModel):
    """Pydantic model representing a persistent user memory record."""

    memory_id: str = Field(..., description="Unique UUID for this memory record.")
    user_id: str = Field(..., description="Namespace identity for this user.")
    category: str = Field(..., description="Category (e.g. preference, constraint, role).")
    key: str = Field(..., description="Semantic identifier key.")
    content: str = Field(..., description="Memory text content.")
    status: Literal["active", "superseded", "forgotten"] = Field(
        ..., description="Lifecycle status of this memory record."
    )
    superseded_by: Optional[str] = Field(
        default=None, description="Memory ID of the record that superseded this one."
    )
    created_at: float = Field(..., description="Unix timestamp of creation.")
    updated_at: float = Field(..., description="Unix timestamp of last status update.")


class MemoryError(Exception):
    """Base exception for memory store errors."""


class MemoryValidationError(MemoryError):
    """Raised when memory inputs are empty, invalid, or malformed."""


class MemoryNotFoundError(MemoryError):
    """Raised when an operation targets a non-existent memory key or ID."""


class MemoryStorageError(MemoryError):
    """Raised when SQLite storage operations encounter database locks or corruption."""


class UserMemoryStore:
    """Persistent SQLite user memory store with transaction management and user isolation."""

    def __init__(self, db_path: Path | str, timeout: float = 5.0) -> None:
        """Initialize the store and ensure schema migrations are applied.

        Parameters
        ----------
        db_path : Path | str
            Filesystem path to the SQLite database file.
        timeout : float
            SQLite busy timeout in seconds.
        """
        self.db_path = Path(db_path)
        self.timeout = timeout
        self._ensure_parent_dir()
        self._init_database()

    def _ensure_parent_dir(self) -> None:
        """Ensure parent directory exists for the database file."""
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Open a configured SQLite connection."""
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.timeout,
                isolation_level=None,  # Autocommit mode; we manage explicit transactions
            )
            conn.row_factory = sqlite3.Row
            # Enforce foreign key constraints and standard pragmatic settings
            # Note: DEFERRED foreign keys allow circular/superseding transactions within an immediate block
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA defer_foreign_keys = ON;")
            if str(self.db_path) != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
            return conn
        except sqlite3.Error as e:
            raise MemoryStorageError(f"Failed to connect to memory database: {type(e).__name__}") from e

    def _init_database(self) -> None:
        """Apply initial schema and version migrations."""
        try:
            with self._get_connection() as conn:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at REAL NOT NULL
                    );
                    """
                )
                cur = conn.execute("SELECT MAX(version) FROM schema_migrations;")
                row = cur.fetchone()
                current_ver = row[0] if (row and row[0] is not None) else 0

                if current_ver < 1:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS user_memories (
                            memory_id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            category TEXT NOT NULL,
                            key TEXT NOT NULL,
                            content TEXT NOT NULL,
                            status TEXT NOT NULL CHECK(status IN ('active', 'superseded', 'forgotten')),
                            superseded_by TEXT REFERENCES user_memories(memory_id) DEFERRABLE INITIALLY DEFERRED,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL
                        );
                        """
                    )
                    # Database-level partial unique index: exactly one 'active' record per (user_id, key)
                    conn.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS uq_user_memories_active_key
                        ON user_memories(user_id, key)
                        WHERE status = 'active';
                        """
                    )
                    conn.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_user_memories_user_status
                        ON user_memories(user_id, status);
                        """
                    )
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?);",
                        (1, time.time()),
                    )
                conn.execute("COMMIT;")
        except sqlite3.Error as e:
            raise MemoryStorageError(f"Database initialization failed: {type(e).__name__}") from e

    # ── Input Validation Helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise MemoryValidationError("user_id cannot be empty or whitespace.")
        return user_id.strip()

    @staticmethod
    def _validate_field(val: str, field_name: str) -> str:
        if not val or not isinstance(val, str) or not val.strip():
            raise MemoryValidationError(f"{field_name} cannot be empty or whitespace.")
        return val.strip()

    # ── Core Operations ──────────────────────────────────────────────────────

    def remember(
        self,
        user_id: str,
        category: str,
        key: str,
        content: str,
    ) -> MemoryItem:
        """Store or update a user memory with automatic idempotence and superseding.

        - If no active memory exists for (user_id, key), insert a new active record.
        - If active memory exists with identical content and category, return it (idempotent).
        - If active memory exists with different content or category, atomically mark
          the old memory as 'superseded' pointing to the new memory, and insert the new active record.
        """
        uid = self._validate_user_id(user_id)
        cat = self._validate_field(category, "category")
        k = self._validate_field(key, "key")
        cnt = self._validate_field(content, "content")

        now = time.time()
        new_id = str(uuid.uuid4())

        try:
            with self._get_connection() as conn:
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    cur = conn.execute(
                        """
                        SELECT memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                        FROM user_memories
                        WHERE user_id = ? AND key = ? AND status = 'active';
                        """,
                        (uid, k),
                    )
                    existing = cur.fetchone()

                    if existing:
                        # Check for identical content
                        if existing["content"] == cnt and existing["category"] == cat:
                            conn.execute("COMMIT;")
                            return MemoryItem(**dict(existing))

                        old_id = existing["memory_id"]
                        # 1. Update previous active record to superseded (clearing active status so unique index allows new active)
                        conn.execute(
                            """
                            UPDATE user_memories
                            SET status = 'superseded', superseded_by = ?, updated_at = ?
                            WHERE memory_id = ?;
                            """,
                            (new_id, now, old_id),
                        )
                        # 2. Insert new active record
                        conn.execute(
                            """
                            INSERT INTO user_memories (
                                memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?);
                            """,
                            (new_id, uid, cat, k, cnt, now, now),
                        )
                    else:
                        conn.execute(
                            """
                            INSERT INTO user_memories (
                                memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?);
                            """,
                            (new_id, uid, cat, k, cnt, now, now),
                        )
                    conn.execute("COMMIT;")
                except Exception:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    raise

                return MemoryItem(
                    memory_id=new_id,
                    user_id=uid,
                    category=cat,
                    key=k,
                    content=cnt,
                    status="active",
                    superseded_by=None,
                    created_at=now,
                    updated_at=now,
                )
        except sqlite3.Error as e:
            raise MemoryStorageError(f"Failed to record memory: {type(e).__name__}") from e

    def recall(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[MemoryItem]:
        """Recall active memories for a given user, optionally filtered by category.

        Always excludes superseded and forgotten records.
        """
        uid = self._validate_user_id(user_id)
        try:
            with self._get_connection() as conn:
                if category:
                    cat = self._validate_field(category, "category")
                    cur = conn.execute(
                        """
                        SELECT memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                        FROM user_memories
                        WHERE user_id = ? AND status = 'active' AND category = ?
                        ORDER BY updated_at DESC;
                        """,
                        (uid, cat),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                        FROM user_memories
                        WHERE user_id = ? AND status = 'active'
                        ORDER BY updated_at DESC;
                        """,
                        (uid,),
                    )
                rows = cur.fetchall()
                return [MemoryItem(**dict(r)) for r in rows]
        except sqlite3.Error as e:
            raise MemoryStorageError(f"Failed to recall memories: {type(e).__name__}") from e

    def list_memories(
        self,
        user_id: str,
        include_inactive: bool = False,
    ) -> List[MemoryItem]:
        """List memories for inspection for a specific user.

        If include_inactive is True, returns active and superseded records (excluding forgotten).
        """
        uid = self._validate_user_id(user_id)
        try:
            with self._get_connection() as conn:
                if include_inactive:
                    cur = conn.execute(
                        """
                        SELECT memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                        FROM user_memories
                        WHERE user_id = ? AND status != 'forgotten'
                        ORDER BY key ASC, created_at DESC;
                        """,
                        (uid,),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                        FROM user_memories
                        WHERE user_id = ? AND status = 'active'
                        ORDER BY key ASC;
                        """,
                        (uid,),
                    )
                rows = cur.fetchall()
                return [MemoryItem(**dict(r)) for r in rows]
        except sqlite3.Error as e:
            raise MemoryStorageError(f"Failed to list memories: {type(e).__name__}") from e

    def correct(
        self,
        user_id: str,
        key: str,
        new_content: str,
    ) -> MemoryItem:
        """Semantically correct an existing active memory.

        Raises MemoryNotFoundError if no active memory exists for the given key.
        """
        uid = self._validate_user_id(user_id)
        k = self._validate_field(key, "key")
        cnt = self._validate_field(new_content, "new_content")

        now = time.time()
        new_id = str(uuid.uuid4())

        try:
            with self._get_connection() as conn:
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    cur = conn.execute(
                        """
                        SELECT memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                        FROM user_memories
                        WHERE user_id = ? AND key = ? AND status = 'active';
                        """,
                        (uid, k),
                    )
                    existing = cur.fetchone()
                    if not existing:
                        conn.execute("ROLLBACK;")
                        raise MemoryNotFoundError(f"Cannot correct: no active memory found for key '{k}'.")

                    category = existing["category"]
                    old_id = existing["memory_id"]

                    # 1. Update prior to superseded (removes active status)
                    conn.execute(
                        """
                        UPDATE user_memories
                        SET status = 'superseded', superseded_by = ?, updated_at = ?
                        WHERE memory_id = ?;
                        """,
                        (new_id, now, old_id),
                    )

                    # 2. Insert corrected active memory
                    conn.execute(
                        """
                        INSERT INTO user_memories (
                            memory_id, user_id, category, key, content, status, superseded_by, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?);
                        """,
                        (new_id, uid, category, k, cnt, now, now),
                    )
                    conn.execute("COMMIT;")
                except MemoryNotFoundError:
                    raise
                except Exception:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    raise

                return MemoryItem(
                    memory_id=new_id,
                    user_id=uid,
                    category=category,
                    key=k,
                    content=cnt,
                    status="active",
                    superseded_by=None,
                    created_at=now,
                    updated_at=now,
                )
        except MemoryNotFoundError:
            raise
        except sqlite3.Error as e:
            raise MemoryStorageError(f"Failed to correct memory: {type(e).__name__}") from e

    def forget(
        self,
        user_id: str,
        *,
        key: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> int:
        """Forget memory records for a given user by key or by specific memory_id.

        Comprehensive Forget Invariant:
        When forgetting by key, ALL records (active and superseded) for that user and key
        have their status set to 'forgotten' and their content overwritten/wiped.
        When forgetting by memory_id, if the record was active, its key is freed;
        if other historical versions share the key, they are also wiped to prevent residual leakage.

        Returns the number of records forgotten.
        """
        uid = self._validate_user_id(user_id)
        if not key and not memory_id:
            raise MemoryValidationError("Must specify either key or memory_id to forget.")

        now = time.time()
        try:
            with self._get_connection() as conn:
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    if key:
                        k = self._validate_field(key, "key")
                        cur = conn.execute(
                            """
                            UPDATE user_memories
                            SET status = 'forgotten', content = '[FORGOTTEN]', updated_at = ?
                            WHERE user_id = ? AND key = ? AND status != 'forgotten';
                            """,
                            (now, uid, k),
                        )
                        count = cur.rowcount
                    else:
                        mid = self._validate_field(memory_id, "memory_id")
                        # First discover the key associated with this memory_id for this user
                        cur = conn.execute(
                            "SELECT key FROM user_memories WHERE user_id = ? AND memory_id = ?;",
                            (uid, mid),
                        )
                        row = cur.fetchone()
                        if not row:
                            conn.execute("COMMIT;")
                            return 0
                        target_key = row["key"]
                        # Mark all records for that key forgotten to avoid orphaned stale history
                        cur = conn.execute(
                            """
                            UPDATE user_memories
                            SET status = 'forgotten', content = '[FORGOTTEN]', updated_at = ?
                            WHERE user_id = ? AND key = ? AND status != 'forgotten';
                            """,
                            (now, uid, target_key),
                        )
                        count = cur.rowcount

                    conn.execute("COMMIT;")
                    return count
                except Exception:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    raise
        except sqlite3.Error as e:
            raise MemoryStorageError(f"Failed to forget memory: {type(e).__name__}") from e

    def export_json(self, user_id: str) -> str:
        """Export active memories for a given user as a sanitized JSON string.

        Guaranteed to exclude superseded and forgotten memories.
        """
        memories = self.recall(user_id=user_id)
        data = [m.model_dump() for m in memories]
        return json.dumps(data, indent=2)

    def close(self) -> None:
        """No persistent background connection pool needed; each query opens and closes safely."""
        pass
