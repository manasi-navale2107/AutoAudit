"""
Application database infrastructure for AutoAudit.

Features:
- Async SQLite connections.
- Explicit transactions.
- Foreign-key enforcement.
- WAL journaling.
- Bounded lock waiting.
- Versioned schema initialization.
- Read connections that reject writes.

This module owns only the application database.
LangGraph manages its separate checkpoint database.

Do not hold a write transaction during LLM calls, model training,
file uploads or other long-running operations.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger


logger = get_logger(__name__)

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5000


class DatabaseInitializationError(RuntimeError):
    """Database schema is unavailable or incompatible."""


# ============================================================
# INITIAL SCHEMA
#
# JSON payloads are serialized from validated Pydantic models.
# Repository code must use parameterized SQL for all values.
#
# owner_id/reviewer_id must come from a trusted identity/session
# layer. Their presence does not implement authentication.
# ============================================================

SCHEMA_V1 = (
    """
    CREATE TABLE datasets (
        dataset_id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        profile_json TEXT,
        storage_key TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (dataset_id, owner_id)
    )
    """,

    """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        request_sha256 TEXT NOT NULL
            CHECK (length(request_sha256) = 64),
        request_json TEXT NOT NULL,
        snapshot_json TEXT NOT NULL,
        status TEXT NOT NULL,
        state_version INTEGER NOT NULL
            CHECK (state_version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,

        UNIQUE (owner_id, request_id),
        UNIQUE (run_id, dataset_id),

        FOREIGN KEY (dataset_id, owner_id)
            REFERENCES datasets (dataset_id, owner_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE cleaning_plans (
        plan_id TEXT NOT NULL,
        revision INTEGER NOT NULL
            CHECK (revision >= 1),
        run_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        execution_plan_revision INTEGER NOT NULL
            CHECK (execution_plan_revision >= 1),
        plan_json TEXT NOT NULL,
        created_at TEXT NOT NULL,

        PRIMARY KEY (plan_id, revision),
        UNIQUE (run_id, revision),
        UNIQUE (plan_id, revision, run_id),

        FOREIGN KEY (run_id, dataset_id)
            REFERENCES runs (run_id, dataset_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE reviews (
        review_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        request_sha256 TEXT NOT NULL
            CHECK (length(request_sha256) = 64),
        plan_id TEXT NOT NULL,
        plan_revision INTEGER NOT NULL
            CHECK (plan_revision >= 1),
        action TEXT NOT NULL
            CHECK (
                action IN (
                    'approve',
                    'request_changes',
                    'reject'
                )
            ),
        review_json TEXT NOT NULL,
        reviewed_at TEXT NOT NULL,

        UNIQUE (run_id, reviewer_id, request_id),

        -- A reviewed revision cannot receive a second decision.
        -- A change request must produce a new plan revision.
        UNIQUE (plan_id, plan_revision),

        FOREIGN KEY (run_id, dataset_id)
            REFERENCES runs (run_id, dataset_id)
            ON DELETE RESTRICT,

        FOREIGN KEY (plan_id, plan_revision, run_id)
            REFERENCES cleaning_plans (
                plan_id,
                revision,
                run_id
            )
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE artifacts (
        artifact_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        storage_key TEXT NOT NULL UNIQUE,
        sha256 TEXT NOT NULL
            CHECK (length(sha256) = 64),
        size_bytes INTEGER NOT NULL
            CHECK (size_bytes > 0),
        created_at TEXT NOT NULL,

        FOREIGN KEY (run_id)
            REFERENCES runs (run_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE run_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,

        FOREIGN KEY (run_id)
            REFERENCES runs (run_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE workflow_outbox (
        message_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,

        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (
                status IN (
                    'pending',
                    'processing',
                    'completed',
                    'failed'
                )
            ),

        attempts INTEGER NOT NULL DEFAULT 0
            CHECK (attempts >= 0),

        available_at TEXT NOT NULL,
        locked_until TEXT,
        lock_token TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,

        FOREIGN KEY (run_id)
            REFERENCES runs (run_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE INDEX idx_datasets_owner_created
    ON datasets (owner_id, created_at)
    """,

    """
    CREATE INDEX idx_runs_owner_created
    ON runs (owner_id, created_at)
    """,

    """
    CREATE INDEX idx_runs_dataset_owner
    ON runs (dataset_id, owner_id)
    """,

    """
    CREATE INDEX idx_runs_status
    ON runs (status)
    """,

    """
    CREATE INDEX idx_reviews_run
    ON reviews (run_id, reviewed_at)
    """,

    """
    CREATE INDEX idx_artifacts_run
    ON artifacts (run_id, created_at)
    """,

    """
    CREATE INDEX idx_run_events_run
    ON run_events (run_id, event_id)
    """,

    """
    CREATE INDEX idx_workflow_outbox_pending
    ON workflow_outbox (status, available_at)
    """,

    """
    CREATE INDEX idx_workflow_outbox_run
    ON workflow_outbox (run_id)
    """,
)


# ============================================================
# CONNECTION MANAGEMENT
# ============================================================

class Database:
    """Application-database access with short-lived connections."""

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        if settings is None:
            settings = get_settings()

        self._path = settings.app_database_path

    @asynccontextmanager
    async def _connect(
        self,
        *,
        allow_create: bool = False,
    ) -> AsyncIterator[aiosqlite.Connection]:
        """
        Open one connection.

        Normal operations use mode=rw so a missing database
        is not silently recreated without its schema.
        """

        mode = "rwc" if allow_create else "rw"
        database_uri = f"{self._path.as_uri()}?mode={mode}"

        async with aiosqlite.connect(
            database_uri,
            uri=True,
            timeout=BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
        ) as connection:
            connection.row_factory = aiosqlite.Row

            async with connection.execute(
                "PRAGMA foreign_keys = ON"
            ):
                pass

            async with connection.execute(
                f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}"
            ):
                pass

            async with connection.execute(
                "PRAGMA synchronous = FULL"
            ):
                pass

            async with connection.execute(
                "PRAGMA foreign_keys"
            ) as cursor:
                row = await cursor.fetchone()

            if row is None or row[0] != 1:
                raise DatabaseInitializationError(
                    "SQLite foreign-key enforcement is unavailable."
                )

            yield connection

    # ========================================================
    # SCHEMA INITIALIZATION
    # ========================================================

    async def initialize(self) -> None:
        """
        Initialize schema version 1 exactly once.

        Safe to call again against the same initialized database.
        Unknown versions fail instead of attempting a destructive
        or guessed migration.

        Future schema changes require a new explicit migration.
        """

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        async with self._connect(
            allow_create=True
        ) as connection:
            async with connection.execute(
                "PRAGMA journal_mode = WAL"
            ) as cursor:
                row = await cursor.fetchone()

            if row is None or str(row[0]).lower() != "wal":
                raise DatabaseInitializationError(
                    "Could not enable SQLite WAL mode."
                )

            async with connection.execute("BEGIN IMMEDIATE"):
                pass

            try:
                async with connection.execute(
                    "PRAGMA user_version"
                ) as cursor:
                    row = await cursor.fetchone()

                current_version = int(row[0])

                if current_version not in {0, SCHEMA_VERSION}:
                    raise DatabaseInitializationError(
                        "Unsupported application database version. "
                        "An explicit migration is required."
                    )

                if current_version == 0:
                    # Refuse to adopt an unrelated existing database.
                    async with connection.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                          AND name NOT LIKE 'sqlite_%'
                        """
                    ) as cursor:
                        existing_tables = await cursor.fetchall()

                    if existing_tables:
                        raise DatabaseInitializationError(
                            "An unversioned database already contains "
                            "tables. Refusing automatic initialization."
                        )

                    # Avoid executescript(): migration statements
                    # must remain inside this explicit transaction.
                    for statement in SCHEMA_V1:
                        async with connection.execute(statement):
                            pass

                    async with connection.execute(
                        f"PRAGMA user_version = {SCHEMA_VERSION}"
                    ):
                        pass

                await connection.commit()

            except BaseException:
                # Include cancellation so unfinished writes roll back.
                await connection.rollback()
                raise

        logger.info(
            "Application database initialized; schema_version=%d.",
            SCHEMA_VERSION,
        )

    # ========================================================
    # READ ACCESS
    # ========================================================

    @asynccontextmanager
    async def read_connection(
        self,
    ) -> AsyncIterator[aiosqlite.Connection]:
        """
        Provide a read-only connection with a consistent snapshot.

        Repository methods must close cursors promptly.
        """

        async with self._connect() as connection:
            async with connection.execute(
                "PRAGMA query_only = ON"
            ):
                pass

            async with connection.execute("BEGIN"):
                pass

            try:
                yield connection

            finally:
                await connection.rollback()

    # ========================================================
    # WRITE TRANSACTIONS
    # ========================================================

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[aiosqlite.Connection]:
        """
        Commit all operations together or roll them back.

        BEGIN IMMEDIATE acquires the write transaction before
        checking/updating revisions.

        Do not nest this method. Pass the same connection to
        repository operations that belong to one transaction.

        Cancellation around commit can leave the caller unsure
        whether the commit completed. Idempotency records let
        the service safely determine the result on retry.
        """

        async with self._connect() as connection:
            async with connection.execute("BEGIN IMMEDIATE"):
                pass

            try:
                yield connection
                await connection.commit()

            except BaseException:
                await connection.rollback()
                raise

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    async def check_health(self) -> bool:
        """
        Check connectivity and schema version.

        Database errors propagate to the API/service layer.
        This is not a full corruption/integrity scan.
        """

        async with self.read_connection() as connection:
            async with connection.execute(
                "PRAGMA user_version"
            ) as cursor:
                row = await cursor.fetchone()

            return (
                row is not None
                and row[0] == SCHEMA_VERSION
            )