from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import joblib

from sklearn.utils.validation import check_is_fitted


if TYPE_CHECKING:
    from app.ml.training import TrainingResult
    from app.schemas.cleaning import ExecutionPlan
    from app.services.cleaning_review_store import CleaningReviewStore


DEFAULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runtime"
    / "runs"
)

ARTIFACT_FILES = (
    "model.joblib",
    "results.json",
    "splits.json",
    "execution_plan.json",
    "cleaning_proposal.json",
    "approval.json",
    "environment.json",
    "README.txt",
)


class ArtifactIntegrityError(ValueError):
    """Saved artifacts failed an integrity or context check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    """
    Write inside an unpublished temporary directory.
    """
    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )

        stream.flush()
        os.fsync(stream.fileno())


class ArtifactService:
    """
    Publish immutable artifact versions and create handoff archives.

    root_dir is backend configuration, never a browser-supplied path.

    No stored joblib file is loaded by this service.
    """

    def __init__(
        self,
        *,
        review_store: CleaningReviewStore,
        root_dir: str | Path = DEFAULT_ROOT,
    ) -> None:
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        self._reviews = review_store

    def _directory(
        self,
        run_id: UUID,
        artifact_id: UUID,
    ) -> Path:
        path = (
            self.root
            / str(UUID(str(run_id)))
            / "artifacts"
            / str(UUID(str(artifact_id)))
        )

        if not path.resolve().is_relative_to(self.root):
            raise ArtifactIntegrityError(
                "Artifact path escapes the configured root."
            )

        return path

    def save(
        self,
        *,
        result: TrainingResult,
        execution_plan: ExecutionPlan,
    ) -> dict:
        from app.agents.model_training_agent import (
            execution_plan_sha256,
        )
        from app.schemas.cleaning import ExecutionPlan

        # Create a JSON-compatible snapshot of the training report.
        report = json.loads(
            json.dumps(
                result.report,
                allow_nan=False,
            )
        )

        plan = ExecutionPlan.model_validate_json(
            execution_plan.model_dump_json()
        )

        run_id = UUID(report["run_id"])
        artifact_id = uuid4()

        if (
            report.get("status") != "trained"
            or report.get("test_evaluated") is not False
            or report.get("audit_status") != "not_performed"
        ):
            raise ArtifactIntegrityError(
                "Expected an unaudited training-stage result."
            )

        if (
            execution_plan_sha256(plan)
            != report["execution_plan_sha256"]
        ):
            raise ArtifactIntegrityError(
                "Execution plan differs from the training result."
            )

        if (
            result.split.to_dict()["counts"]
            != report["split_counts"]
        ):
            raise ArtifactIntegrityError(
                "Split counts differ from the training report."
            )

        check_is_fitted(result.pipeline)

        approved = self._reviews.require_approved(
            run_id=run_id,
            revision=report["approved_revision"],
            dataset_sha256=report["dataset_sha256"],
            plan_sha256=report["execution_plan_sha256"],
        )

        review = self._reviews.get_latest(
            run_id=run_id,
        )

        if (
            review["revision"] != report["approved_revision"]
            or review["status"] != "approved"
        ):
            raise ArtifactIntegrityError(
                "Approval changed before saving."
            )

        destination = self._directory(
            run_id,
            artifact_id,
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with tempfile.TemporaryDirectory(
            dir=destination.parent,
            prefix="building-",
        ) as temporary:
            stage = Path(temporary) / "payload"
            stage.mkdir()

            joblib.dump(
                result.pipeline,
                stage / "model.joblib",
                compress=3,
            )

            _write_json(
                stage / "results.json",
                report,
            )

            _write_json(
                stage / "splits.json",
                {
                    **result.split.to_dict(),
                    "dataset_sha256": report["dataset_sha256"],
                },
            )

            _write_json(
                stage / "execution_plan.json",
                plan.model_dump(mode="json"),
            )

            _write_json(
                stage / "cleaning_proposal.json",
                approved.model_dump(mode="json"),
            )

            _write_json(
                stage / "environment.json",
                report["environment"],
            )

            _write_json(
                stage / "approval.json",
                {
                    key: review[key]
                    for key in (
                        "run_id",
                        "revision",
                        "dataset_sha256",
                        "plan_sha256",
                        "proposal_sha256",
                        "status",
                        "created_at",
                        "decided_at",
                    )
                },
            )

            (stage / "README.txt").write_text(
                "AutoAudit: construction-stage handoff\n\n"
                "Audit status: not performed. "
                "Final test has not been scored.\n"
                "The selected pipeline remains fitted "
                "on training rows only.\n"
                "Original CSV is not included. "
                "Use the CSV matching dataset_sha256.\n"
                "Split IDs are zero-based data-row positions, "
                "excluding the header.\n"
                "Keep the original AutoAudit project available "
                "when loading the model:\n"
                "FeatureCaster requires app/ml/preprocessing.py "
                "and its imports.\n"
                "Use the Python/package versions recorded "
                "in environment.json.\n"
                "Only load joblib files from a trusted producer.\n"
                "Hashes detect changes; "
                "they are not digital signatures.\n",
                encoding="utf-8",
            )

            manifest = {
                "schema_version": 1,
                "run_id": str(run_id),
                "artifact_id": str(artifact_id),
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "dataset_sha256": report["dataset_sha256"],
                "execution_plan_sha256": (
                    report["execution_plan_sha256"]
                ),
                "approved_revision": report["approved_revision"],
                "status": "ready_for_audit",
                "audit_status": "not_performed",
                "test_evaluated": False,
                "files": {
                    name: {
                        "sha256": _sha256(stage / name),
                        "bytes": (stage / name).stat().st_size,
                    }
                    for name in ARTIFACT_FILES
                },
            }

            _write_json(
                stage / "manifest.json",
                manifest,
            )

            # Publish only after every artifact has been written.
            os.replace(stage, destination)

        return manifest

    def get_manifest(
        self,
        *,
        run_id: UUID,
        artifact_id: UUID,
    ) -> dict:
        directory = self._directory(
            run_id,
            artifact_id,
        )

        path = directory / "manifest.json"

        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                "Artifact manifest not found."
            )

        manifest = json.loads(
            path.read_text(encoding="utf-8")
        )

        if (
            manifest.get("schema_version") != 1
            or manifest.get("run_id") != str(UUID(str(run_id)))
            or manifest.get("artifact_id") != str(UUID(str(artifact_id)))
            or manifest.get("status") != "ready_for_audit"
            or manifest.get("audit_status") != "not_performed"
            or manifest.get("test_evaluated") is not False
            or set(manifest.get("files", {})) != set(ARTIFACT_FILES)
        ):
            raise ArtifactIntegrityError(
                "Invalid artifact manifest."
            )

        for name in ARTIFACT_FILES:
            source = directory / name
            expected = manifest["files"][name]

            if (
                not source.is_file()
                or source.is_symlink()
                or source.stat().st_size != expected["bytes"]
                or _sha256(source) != expected["sha256"]
            ):
                raise ArtifactIntegrityError(
                    f"Artifact failed verification: {name}"
                )

        return manifest

    def create_handoff_zip(
        self,
        *,
        run_id: UUID,
        artifact_id: UUID,
    ) -> Path:
        manifest = self.get_manifest(
            run_id=run_id,
            artifact_id=artifact_id,
        )

        directory = self._directory(
            run_id,
            artifact_id,
        )

        archive_path = directory / "handoff.zip"

        descriptor, temporary = tempfile.mkstemp(
            dir=directory,
            suffix=".zip",
        )

        os.close(descriptor)

        try:
            with zipfile.ZipFile(
                temporary,
                "w",
                zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        indent=2,
                    ),
                )

                for name in ARTIFACT_FILES:
                    # Hash the same bytes placed in the archive.
                    digest = hashlib.sha256()

                    with (directory / name).open("rb") as source:
                        with archive.open(
                            name,
                            "w",
                            force_zip64=True,
                        ) as target:
                            for block in iter(
                                lambda: source.read(1024 * 1024),
                                b"",
                            ):
                                digest.update(block)
                                target.write(block)

                    if (
                        digest.hexdigest()
                        != manifest["files"][name]["sha256"]
                    ):
                        raise ArtifactIntegrityError(
                            "Artifact changed during ZIP creation."
                        )

            os.replace(temporary, archive_path)

        finally:
            Path(temporary).unlink(missing_ok=True)

        return archive_path