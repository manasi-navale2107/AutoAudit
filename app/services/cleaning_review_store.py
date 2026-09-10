from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID

from app.schemas.cleaning import CleaningProposal


DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "runtime"
    / "database"
    / "cleaning_reviews.sqlite3"
)


class ReviewConflictError(ValueError):
    """The requested action conflicts with the current review state."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: dict) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_hash(value: str) -> str:
    normalized = value.lower()

    if (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError("A valid SHA-256 hexadecimal digest is required.")

    return normalized


class CleaningReviewStore:
    """
    Persist proposals and human review decisions.

    This class does not execute cleaning or train models.

    The dataset hash and execution-plan hash must be computed by
    trusted backend code, not accepted blindly from browser input.
    """

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cleaning_reviews (
                    run_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    plan_sha256 TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    proposal_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'pending',
                            'approved',
                            'changes_requested',
                            'rejected'
                        )
                    ),
                    feedback TEXT,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    PRIMARY KEY (run_id, revision)
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=10,
        )

        connection.row_factory = sqlite3.Row

        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _latest(
        connection: sqlite3.Connection,
        run_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT *
            FROM cleaning_reviews
            WHERE run_id = ?
            ORDER BY revision DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["proposal"] = json.loads(result.pop("proposal_json"))
        return result

    def get_latest(self, *, run_id: UUID) -> dict:
        with self._connect() as connection:
            row = self._latest(connection, str(run_id))

            if row is None:
                raise KeyError("No cleaning proposal exists for this run.")

            return self._decode(row)

    def submit(
        self,
        *,
        run_id: UUID,
        proposal: CleaningProposal,
        dataset_sha256: str,
        plan_sha256: str,
        expected_revision: int,
    ) -> dict:
        """
        Submit the first proposal with expected_revision=0.

        Submit a revised proposal only after changes_requested,
        using the previous revision as expected_revision.

        Every submitted proposal starts as pending.
        """
        if expected_revision < 0:
            raise ValueError("expected_revision must not be negative.")

        dataset_hash = _validate_hash(dataset_sha256)
        plan_hash = _validate_hash(plan_sha256)

        validated = CleaningProposal.model_validate_json(
            proposal.model_dump_json()
        )

        proposal_json = _canonical_json(
            validated.model_dump(mode="json")
        )

        proposal_hash = hashlib.sha256(
            proposal_json.encode("utf-8")
        ).hexdigest()

        key = str(run_id)

        with self._connect() as connection:
            # Serialize competing submissions and review decisions.
            connection.execute("BEGIN IMMEDIATE")

            previous = self._latest(connection, key)

            actual_revision = (
                previous["revision"]
                if previous is not None
                else 0
            )

            if expected_revision != actual_revision:
                raise ReviewConflictError(
                    "The proposal revision changed. Reload the latest review."
                )

            if previous is not None:
                if previous["status"] != "changes_requested":
                    raise ReviewConflictError(
                        "Request changes before submitting a revised proposal. "
                        "Use a new run if the previous review is already final."
                    )

                if (
                    previous["dataset_sha256"] != dataset_hash
                    or previous["plan_sha256"] != plan_hash
                ):
                    raise ReviewConflictError(
                        "The dataset or execution plan changed. "
                        "Create a new run for the changed context."
                    )

            revision = actual_revision + 1

            connection.execute(
                """
                INSERT INTO cleaning_reviews (
                    run_id,
                    revision,
                    dataset_sha256,
                    plan_sha256,
                    proposal_json,
                    proposal_sha256,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    key,
                    revision,
                    dataset_hash,
                    plan_hash,
                    proposal_json,
                    proposal_hash,
                    _now(),
                ),
            )

            return self._decode(self._latest(connection, key))

    def decide(
        self,
        *,
        run_id: UUID,
        revision: int,
        action: str,
        feedback: str | None = None,
    ) -> dict:
        """
        Supported actions: approve, changes, reject.

        Only the latest pending revision can receive a decision.
        """
        statuses = {
            "approve": "approved",
            "changes": "changes_requested",
            "reject": "rejected",
        }

        if action not in statuses:
            raise ValueError("Action must be approve, changes, or reject.")

        if revision < 1:
            raise ValueError("Revision must be positive.")

        comment = feedback.strip() if feedback is not None else None

        if feedback is not None and not comment:
            raise ValueError("Feedback must not be blank.")

        if comment is not None and len(comment) > 4_000:
            raise ValueError("Feedback must not exceed 4000 characters.")

        if action == "changes" and not comment:
            raise ValueError("Request changes requires feedback.")

        if action == "approve" and comment is not None:
            raise ValueError(
                "Use Request changes for feedback that modifies a proposal."
            )

        key = str(run_id)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            row = self._latest(connection, key)

            if row is None:
                raise KeyError("No cleaning proposal exists for this run.")

            if row["revision"] != revision:
                raise ReviewConflictError(
                    "This revision is stale. Review the latest proposal."
                )

            desired_status = statuses[action]

            # An exact duplicate decision is harmless.
            if (
                row["status"] == desired_status
                and row["feedback"] == comment
            ):
                return self._decode(row)

            if row["status"] != "pending":
                raise ReviewConflictError(
                    "This revision already has a different review decision."
                )

            proposal = CleaningProposal.model_validate_json(
                row["proposal_json"]
            )

            if action == "approve" and proposal.unresolved_questions:
                raise ReviewConflictError(
                    "Resolve the proposal's questions before approval."
                )

            connection.execute(
                """
                UPDATE cleaning_reviews
                SET status = ?, feedback = ?, decided_at = ?
                WHERE run_id = ? AND revision = ?
                """,
                (
                    desired_status,
                    comment,
                    _now(),
                    key,
                    revision,
                ),
            )

            return self._decode(self._latest(connection, key))

    def require_approved(
        self,
        *,
        run_id: UUID,
        revision: int,
        dataset_sha256: str,
        plan_sha256: str,
    ) -> CleaningProposal:
        """
        Return the exact approved proposal for the supplied context.

        The training service must use this returned proposal and verify
        the dataset bytes before execution.
        """
        dataset_hash = _validate_hash(dataset_sha256)
        plan_hash = _validate_hash(plan_sha256)

        row = self.get_latest(run_id=run_id)

        if row["revision"] != revision or row["status"] != "approved":
            raise ReviewConflictError(
                "The current cleaning revision is not approved."
            )

        if (
            row["dataset_sha256"] != dataset_hash
            or row["plan_sha256"] != plan_hash
        ):
            raise ReviewConflictError(
                "Approval does not match the dataset and execution plan."
            )

        serialized = _canonical_json(row["proposal"])

        actual_hash = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

        if actual_hash != row["proposal_sha256"]:
            raise ReviewConflictError(
                "Stored proposal failed its integrity check."
            )

        proposal = CleaningProposal.model_validate_json(serialized)

        if proposal.unresolved_questions:
            raise ReviewConflictError(
                "The approved proposal contains unresolved questions."
            )

        return proposal