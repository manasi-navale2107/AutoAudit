"""
Repository operations for AutoAudit.

Usage:
    async with database.transaction() as connection:
        repository = Repository(connection)
        # Perform related writes using this repository.

    async with database.read_connection() as connection:
        repository = Repository(connection)
        # Perform reads.

Rules:
- This class does not commit or open its own connection.
- Values are passed through SQL parameters.
- owner_id/reviewer_id must come from trusted server context.
- Returned storage keys are internal, not public API responses.
- Services enforce workflow transitions and dataset-dependent rules.
"""

import hashlib
import json

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

import aiosqlite

from app.schemas.cleaning import CleaningPlan
from app.schemas.dataset import (
    DatasetMetadata,
    DatasetProfile,
    DatasetStatus,
)
from app.schemas.handoff import ArtifactReference
from app.schemas.review import (
    ReviewRecord,
    ReviewRequest,
    validate_review_against_current_plan,
)
from app.schemas.run import (
    RunCreateRequest,
    RunResponse,
    RunStatus,
)


# ============================================================
# ERRORS
# ============================================================

class RecordNotFoundError(LookupError):
    """Record does not exist or is not accessible to this owner."""


class RepositoryConflictError(RuntimeError):
    """Duplicate request mismatch, stale version or stale plan."""


# ============================================================
# INTERNAL RECORDS
# ============================================================

@dataclass(frozen=True)
class StoredDataset:
    metadata: DatasetMetadata
    profile: DatasetProfile | None
    storage_key: str


@dataclass(frozen=True)
class StoredArtifact:
    reference: ArtifactReference
    storage_key: str


# ============================================================
# HELPERS
# ============================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _request_hash(request: Any) -> str:
    payload = _json(request.model_dump(mode="json"))

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _storage_key(value: str) -> str:
    """
    Require a normalized relative POSIX storage key.

    The artifact/file service must additionally resolve the key
    under its configured storage root and check containment.
    """

    if not value or "\\" in value or ":" in value:
        raise ValueError("Invalid storage key.")

    if any(ord(character) < 32 for character in value):
        raise ValueError("Invalid storage key.")

    path = PurePosixPath(value)

    if (
        path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or str(path) == "."
    ):
        raise ValueError(
            "Storage keys must be normalized relative paths."
        )

    return value


# ============================================================
# REPOSITORY
# ============================================================

class Repository:
    def __init__(
        self,
        connection: aiosqlite.Connection,
    ) -> None:
        self._connection = connection

    async def _one(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> aiosqlite.Row | None:
        async with self._connection.execute(
            sql,
            parameters,
        ) as cursor:
            return await cursor.fetchone()

    async def _write(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> int:
        if not self._connection.in_transaction:
            raise RuntimeError(
                "Writes require Database.transaction()."
            )

        # Database.read_connection() also opens a transaction.
        # Its query_only pragma must not be bypassed.
        row = await self._one("PRAGMA query_only")

        if row is None or row[0] != 0:
            raise RuntimeError(
                "A read-only connection cannot perform writes."
            )

        async with self._connection.execute(
            sql,
            parameters,
        ) as cursor:
            return cursor.rowcount

    # ========================================================
    # DATASETS
    # ========================================================

    async def create_dataset(
        self,
        *,
        owner_id: str,
        metadata: DatasetMetadata,
        storage_key: str,
    ) -> None:
        if not owner_id.strip():
            raise ValueError("owner_id must not be blank.")

        if metadata.status != DatasetStatus.UPLOADED:
            raise ValueError(
                "New dataset records must have uploaded status."
            )

        timestamp = metadata.uploaded_at.isoformat()

        await self._write(
            """
            INSERT INTO datasets (
                dataset_id, owner_id, metadata_json,
                profile_json, storage_key,
                created_at, updated_at
            )
            VALUES (?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                str(metadata.dataset_id),
                owner_id,
                metadata.model_dump_json(),
                _storage_key(storage_key),
                timestamp,
                timestamp,
            ),
        )

    async def get_dataset(
        self,
        *,
        dataset_id: UUID,
        owner_id: str,
    ) -> StoredDataset:
        row = await self._one(
            """
            SELECT metadata_json, profile_json, storage_key
            FROM datasets
            WHERE dataset_id = ? AND owner_id = ?
            """,
            (str(dataset_id), owner_id),
        )

        if row is None:
            raise RecordNotFoundError("Dataset not found.")

        return StoredDataset(
            metadata=DatasetMetadata.model_validate_json(
                row["metadata_json"]
            ),
            profile=(
                DatasetProfile.model_validate_json(
                    row["profile_json"]
                )
                if row["profile_json"] is not None
                else None
            ),
            storage_key=row["storage_key"],
        )

    async def save_dataset_profile(
        self,
        *,
        owner_id: str,
        profile: DatasetProfile,
    ) -> None:
        stored = await self.get_dataset(
            dataset_id=profile.dataset_id,
            owner_id=owner_id,
        )

        metadata_data = stored.metadata.model_dump()
        metadata_data["status"] = DatasetStatus.PROFILED
        metadata = DatasetMetadata.model_validate(metadata_data)

        affected = await self._write(
            """
            UPDATE datasets
            SET metadata_json = ?, profile_json = ?, updated_at = ?
            WHERE dataset_id = ? AND owner_id = ?
              AND profile_json IS NULL
            """,
            (
                metadata.model_dump_json(),
                profile.model_dump_json(),
                _utc_now(),
                str(profile.dataset_id),
                owner_id,
            ),
        )

        if affected != 1:
            raise RepositoryConflictError(
                "Dataset profile already exists or is unavailable."
            )

    # ========================================================
    # RUNS
    # ========================================================

    async def get_run(
        self,
        *,
        run_id: UUID,
        owner_id: str,
    ) -> RunResponse:
        row = await self._one(
            """
            SELECT snapshot_json
            FROM runs
            WHERE run_id = ? AND owner_id = ?
            """,
            (str(run_id), owner_id),
        )

        if row is None:
            raise RecordNotFoundError("Run not found.")

        return RunResponse.model_validate_json(
            row["snapshot_json"]
        )

    async def create_run(
        self,
        *,
        owner_id: str,
        request: RunCreateRequest,
        snapshot: RunResponse,
    ) -> tuple[RunResponse, bool]:
        """
        Return (stored_run, created).

        created=False means an identical request was already saved.
        The caller must not enqueue duplicate work in that case.
        """

        request_sha256 = _request_hash(request)

        existing = await self._one(
            """
            SELECT request_sha256, snapshot_json
            FROM runs
            WHERE owner_id = ? AND request_id = ?
            """,
            (owner_id, str(request.request_id)),
        )

        if existing is not None:
            if existing["request_sha256"] != request_sha256:
                raise RepositoryConflictError(
                    "This request ID was used with different content."
                )

            return (
                RunResponse.model_validate_json(
                    existing["snapshot_json"]
                ),
                False,
            )

        await self.get_dataset(
            dataset_id=request.dataset_id,
            owner_id=owner_id,
        )

        if (
            snapshot.dataset_id != request.dataset_id
            or snapshot.objective != request.objective
            or snapshot.status != RunStatus.QUEUED
            or snapshot.state_version != 1
        ):
            raise ValueError(
                "New run snapshot must match the request "
                "and start queued at version 1."
            )

        if (
            request.provider is not None
            and request.provider != snapshot.llm.provider
        ):
            raise ValueError("Requested provider does not match.")

        if request.model is not None and request.model != snapshot.llm.model:
            raise ValueError("Requested model does not match.")

        if request.language is not None and request.language != snapshot.language:
            raise ValueError("Requested language does not match.")

        await self._write(
            """
            INSERT INTO runs (
                run_id, owner_id, dataset_id, request_id,
                request_sha256, request_json, snapshot_json,
                status, state_version, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(snapshot.run_id),
                owner_id,
                str(snapshot.dataset_id),
                str(request.request_id),
                request_sha256,
                request.model_dump_json(),
                snapshot.model_dump_json(),
                snapshot.status.value,
                snapshot.state_version,
                snapshot.created_at.isoformat(),
                snapshot.updated_at.isoformat(),
            ),
        )

        return snapshot, True

    async def update_run(
        self,
        *,
        owner_id: str,
        snapshot: RunResponse,
        expected_version: int,
    ) -> None:
        current = await self.get_run(
            run_id=snapshot.run_id,
            owner_id=owner_id,
        )

        if current.state_version != expected_version:
            raise RepositoryConflictError("Run state has changed.")

        if snapshot.state_version != expected_version + 1:
            raise ValueError(
                "Updated state_version must increase by exactly one."
            )

        if (
            snapshot.dataset_id != current.dataset_id
            or snapshot.objective != current.objective
            or snapshot.llm != current.llm
            or snapshot.created_at != current.created_at
        ):
            raise ValueError(
                "Run identity, objective and LLM configuration "
                "cannot be replaced during an update."
            )

        if snapshot.updated_at < current.updated_at:
            raise ValueError("Run timestamps must not move backwards.")

        affected = await self._write(
            """
            UPDATE runs
            SET snapshot_json = ?, status = ?,
                state_version = ?, updated_at = ?
            WHERE run_id = ? AND owner_id = ?
              AND state_version = ?
            """,
            (
                snapshot.model_dump_json(),
                snapshot.status.value,
                snapshot.state_version,
                snapshot.updated_at.isoformat(),
                str(snapshot.run_id),
                owner_id,
                expected_version,
            ),
        )

        if affected != 1:
            raise RepositoryConflictError("Run state has changed.")

    # ========================================================
    # CLEANING PLANS
    # ========================================================

    async def get_latest_cleaning_plan(
        self,
        *,
        run_id: UUID,
        owner_id: str,
    ) -> CleaningPlan | None:
        await self.get_run(run_id=run_id, owner_id=owner_id)

        row = await self._one(
            """
            SELECT plan_json
            FROM cleaning_plans
            WHERE run_id = ?
            ORDER BY revision DESC
            LIMIT 1
            """,
            (str(run_id),),
        )

        if row is None:
            return None

        return CleaningPlan.model_validate_json(row["plan_json"])

    async def save_cleaning_plan(
        self,
        *,
        owner_id: str,
        plan: CleaningPlan,
    ) -> None:
        run = await self.get_run(
            run_id=plan.run_id,
            owner_id=owner_id,
        )

        if (
            plan.dataset_id != run.dataset_id
            or plan.execution_plan_revision != run.execution_plan_revision
            or run.execution_plan is None
        ):
            raise RepositoryConflictError(
                "Cleaning plan does not match the current execution plan."
            )

        previous = await self.get_latest_cleaning_plan(
            run_id=plan.run_id,
            owner_id=owner_id,
        )

        expected_revision = (
            previous.revision + 1
            if previous is not None
            else 1
        )

        if plan.revision != expected_revision:
            raise RepositoryConflictError(
                "Cleaning revision must increase by exactly one."
            )

        stored_dataset = await self.get_dataset(
            dataset_id=run.dataset_id,
            owner_id=owner_id,
        )

        if stored_dataset.profile is None:
            raise RepositoryConflictError("Dataset profile is unavailable.")

        plan.proposal.validate_against_context(
            execution_plan=run.execution_plan,
            profile=stored_dataset.profile,
        )

        await self._write(
            """
            INSERT INTO cleaning_plans (
                plan_id, revision, run_id, dataset_id,
                execution_plan_revision, plan_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(plan.plan_id),
                plan.revision,
                str(plan.run_id),
                str(plan.dataset_id),
                plan.execution_plan_revision,
                plan.model_dump_json(),
                plan.created_at.isoformat(),
            ),
        )

    # ========================================================
    # HUMAN REVIEW
    # ========================================================

    async def record_review(
        self,
        *,
        owner_id: str,
        record: ReviewRecord,
    ) -> tuple[ReviewRecord, bool]:
        """
        Return (stored_review, created).

        Caller must update the run and enqueue the resume event
        in this SAME transaction, only when created=True.
        """

        run = await self.get_run(
            run_id=record.run_id,
            owner_id=owner_id,
        )

        request: ReviewRequest = record.decision
        request_sha256 = _request_hash(request)

        existing = await self._one(
            """
            SELECT request_sha256, review_json
            FROM reviews
            WHERE run_id = ? AND reviewer_id = ? AND request_id = ?
            """,
            (
                str(record.run_id),
                record.reviewer_id,
                str(request.request_id),
            ),
        )

        if existing is not None:
            if existing["request_sha256"] != request_sha256:
                raise RepositoryConflictError(
                    "This review request ID was used with different content."
                )

            return (
                ReviewRecord.model_validate_json(existing["review_json"]),
                False,
            )

        if (
            run.status != RunStatus.AWAITING_REVIEW
            or run.pending_review is None
            or run.execution_plan_revision is None
        ):
            raise RepositoryConflictError(
                "The run is not awaiting cleaning review."
            )

        plan = await self.get_latest_cleaning_plan(
            run_id=record.run_id,
            owner_id=owner_id,
        )

        if plan is None:
            raise RepositoryConflictError("Cleaning plan is unavailable.")

        if (
            record.dataset_id != run.dataset_id
            or record.execution_plan_revision != run.execution_plan_revision
            or run.pending_review.plan_id != plan.plan_id
            or run.pending_review.plan_revision != plan.revision
        ):
            raise RepositoryConflictError("Review context has changed.")

        validate_review_against_current_plan(
            request=request,
            current_plan=plan,
            expected_run_id=run.run_id,
            expected_dataset_id=run.dataset_id,
            current_execution_plan_revision=run.execution_plan_revision,
        )

        if record.reviewed_at < plan.created_at:
            raise ValueError("Review cannot precede the cleaning plan.")

        previous_decision = await self._one(
            """
            SELECT review_id FROM reviews
            WHERE plan_id = ? AND plan_revision = ?
            """,
            (str(plan.plan_id), plan.revision),
        )

        if previous_decision is not None:
            raise RepositoryConflictError(
                "This cleaning-plan revision has already been reviewed."
            )

        await self._write(
            """
            INSERT INTO reviews (
                review_id, run_id, dataset_id, reviewer_id,
                request_id, request_sha256, plan_id, plan_revision,
                action, review_json, reviewed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.review_id),
                str(record.run_id),
                str(record.dataset_id),
                record.reviewer_id,
                str(request.request_id),
                request_sha256,
                str(request.plan_id),
                request.plan_revision,
                request.action.value,
                record.model_dump_json(),
                record.reviewed_at.isoformat(),
            ),
        )

        return record, True

    async def list_reviews(
        self,
        *,
        run_id: UUID,
        owner_id: str,
    ) -> list[ReviewRecord]:
        await self.get_run(run_id=run_id, owner_id=owner_id)

        async with self._connection.execute(
            """
            SELECT review_json FROM reviews
            WHERE run_id = ?
            ORDER BY plan_revision
            """,
            (str(run_id),),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            ReviewRecord.model_validate_json(row["review_json"])
            for row in rows
        ]

    # ========================================================
    # ARTIFACTS
    # ========================================================

    async def save_artifact(
        self,
        *,
        owner_id: str,
        run_id: UUID,
        reference: ArtifactReference,
        storage_key: str,
    ) -> None:
        await self.get_run(run_id=run_id, owner_id=owner_id)

        await self._write(
            """
            INSERT INTO artifacts (
                artifact_id, run_id, kind, storage_key,
                sha256, size_bytes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(reference.artifact_id),
                str(run_id),
                reference.kind.value,
                _storage_key(storage_key),
                reference.sha256,
                reference.size_bytes,
                _utc_now(),
            ),
        )

    async def get_artifact(
        self,
        *,
        artifact_id: UUID,
        owner_id: str,
    ) -> StoredArtifact:
        row = await self._one(
            """
            SELECT a.*
            FROM artifacts AS a
            JOIN runs AS r ON r.run_id = a.run_id
            WHERE a.artifact_id = ? AND r.owner_id = ?
            """,
            (str(artifact_id), owner_id),
        )

        if row is None:
            raise RecordNotFoundError("Artifact not found.")

        return StoredArtifact(
            reference=ArtifactReference(
                artifact_id=row["artifact_id"],
                kind=row["kind"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
            ),
            storage_key=row["storage_key"],
        )

    # ========================================================
    # AUDIT EVENTS AND WORKFLOW OUTBOX
    # ========================================================

    async def append_run_event(
        self,
        *,
        owner_id: str,
        run_id: UUID,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        await self.get_run(run_id=run_id, owner_id=owner_id)

        if not event_type.strip():
            raise ValueError("event_type must not be blank.")

        await self._write(
            """
            INSERT INTO run_events (
                run_id, event_type, payload_json, created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                str(run_id),
                event_type,
                _json(payload),
                _utc_now(),
            ),
        )

    async def enqueue_workflow_event(
        self,
        *,
        owner_id: str,
        message_id: UUID,
        run_id: UUID,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Persist work to be processed after the transaction commits.

        Worker claiming, lease renewal and completion handling
        will be implemented with the workflow runner.

        Payloads should contain IDs/revisions, not secrets,
        raw datasets or full model prompts.
        """

        await self.get_run(run_id=run_id, owner_id=owner_id)

        if not event_type.strip():
            raise ValueError("event_type must not be blank.")

        timestamp = _utc_now()

        await self._write(
            """
            INSERT INTO workflow_outbox (
                message_id, run_id, event_type, payload_json,
                status, attempts, available_at,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?)
            """,
            (
                str(message_id),
                str(run_id),
                event_type,
                _json(payload),
                timestamp,
                timestamp,
                timestamp,
            ),
        )