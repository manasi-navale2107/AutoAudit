
"""Secure, incremental file workspace for AutoAudit-generated ML projects.

A validated ProjectBlueprint is registered once per run. Authorized stages
then save the actual bytes of their planned files. This service verifies
paths, ownership, hashes and simple file syntax; it does NOT ask an LLM to
write code, run user code, approve training, test the project or create ZIPs.

The workspace is local, backend-owned storage. Do not expose its paths as
client-controlled API parameters or mount it as public static content.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import UUID

from filelock import FileLock

from app.schemas.project import (
    FileKind,
    ProjectBlueprint,
    ProjectStage,
    validate_relative_path,
)


MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 5 * 1024 * 1024
MAX_ARTIFACT_BYTES = 200 * 1024 * 1024


class WorkspaceError(RuntimeError):
    """A workspace operation was rejected or could not be completed."""


class WorkspaceConflict(WorkspaceError):
    """A different blueprint or competing file revision already exists."""


class WorkspaceIntegrityError(WorkspaceError):
    """The on-disk workspace does not match its manifest."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    """Best-effort metadata durability on filesystems supporting fsync."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write a complete file and atomically publish it in the same folder."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".autoaudit_", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise WorkspaceIntegrityError("A workspace file is a symbolic link.")
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ProjectWorkspaceService:
    """Manage one isolated, planned project tree per UUID run.

    Layout:
        runtime/runs/<run_id>/workspace/
            project_manifest.json        # backend-generated record
            app.py                       # produced by Model Training stage
            src/...                      # produced over multiple stages

    All public methods take a trusted run ID. The Planner cannot choose a
    filesystem location. File paths must already exist in the blueprint.
    """

    def __init__(self, root_dir: str | Path = "runtime/runs") -> None:
        self.root = Path(root_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: UUID | str) -> Path:
        try:
            run = str(UUID(str(run_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise WorkspaceError("Invalid run_id.") from exc
        path = self.root / run
        if path.is_symlink():
            raise WorkspaceIntegrityError("Run directory must not be a symlink.")
        return path

    def _workspace(self, run_id: UUID | str) -> Path:
        path = self._run_dir(run_id) / "workspace"
        if path.is_symlink():
            raise WorkspaceIntegrityError("Workspace must not be a symlink.")
        return path

    def _lock(self, run_id: UUID | str) -> FileLock:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return FileLock(str(run_dir / ".workspace.lock"), timeout=30)

    @staticmethod
    def _file_path(workspace: Path, relative_path: str) -> Path:
        relative_path = validate_relative_path(relative_path)
        path = workspace.joinpath(*relative_path.split("/"))
        # The local workspace root is trusted; reject symlinks at every level.
        current = workspace
        for segment in relative_path.split("/"):
            current = current / segment
            if current.is_symlink():
                raise WorkspaceIntegrityError(
                    f"Symbolic links are prohibited: {relative_path}"
                )
        if not path.resolve().is_relative_to(workspace.resolve()):
            raise WorkspaceIntegrityError("File escapes the project workspace.")
        return path

    @staticmethod
    def _check_source(path: str, kind: FileKind, content: bytes) -> None:
        limit = (
            MAX_ARTIFACT_BYTES
            if kind in {FileKind.MODEL, FileKind.DATA}
            else MAX_SOURCE_BYTES
        )
        if not content or len(content) > limit:
            raise WorkspaceError("File content is empty or exceeds its size limit.")

        if kind not in {FileKind.MODEL, FileKind.DATA}:
            try:
                source = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WorkspaceError("Source and report files must be UTF-8.") from exc
            if path.endswith(".py"):
                try:
                    compile(source, path, "exec")
                except SyntaxError as exc:
                    raise WorkspaceError(
                        f"Generated Python syntax is invalid in {path}: {exc.msg}"
                    ) from exc
            if path.endswith(".json"):
                try:
                    json.loads(source)
                except json.JSONDecodeError as exc:
                    raise WorkspaceError(f"Invalid JSON in {path}.") from exc

    @staticmethod
    def _read_manifest(workspace: Path) -> dict:
        path = workspace / "project_manifest.json"
        if path.is_symlink():
            raise WorkspaceIntegrityError("Project manifest is a symbolic link.")
        try:
            if path.stat().st_size > MAX_MANIFEST_BYTES:
                raise WorkspaceIntegrityError("Project manifest exceeds size limit.")
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise WorkspaceIntegrityError("Cannot read a valid project manifest.") from exc
        if not isinstance(manifest, dict):
            raise WorkspaceIntegrityError("Project manifest must be a JSON object.")
        try:
            blueprint = ProjectBlueprint.model_validate(manifest["blueprint"])
            digest = _hash(_canonical_json(blueprint.model_dump(mode="json")))
            if digest != manifest["blueprint_sha256"]:
                raise WorkspaceIntegrityError("Project blueprint fingerprint changed.")
            if manifest["blueprint_revision"] != blueprint.revision:
                raise WorkspaceIntegrityError("Project blueprint revision changed.")
            files = manifest["files"]
            if not isinstance(files, dict) or set(files) != {
                f.path for f in blueprint.files
            }:
                raise WorkspaceIntegrityError("Manifest file list differs from blueprint.")
            for planned in blueprint.files:
                info = files[planned.path]
                if not isinstance(info, dict) or info.get("producer") != planned.producer.value:
                    raise WorkspaceIntegrityError("Manifest file owner changed.")
                if info.get("status") not in {"planned", "created", "revised"}:
                    raise WorkspaceIntegrityError("Manifest contains unknown file status.")
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkspaceIntegrityError("Invalid or corrupted project manifest.") from exc
        return manifest

    def initialize(
        self, *, run_id: UUID | str, blueprint: ProjectBlueprint,
    ) -> dict:
        """Record the whole blueprint without creating fake project files.

        Repeating the same request is idempotent. Changing an existing
        blueprint requires a future explicit, revision-aware migration.
        """
        blueprint = ProjectBlueprint.model_validate(blueprint)
        run_dir = self._run_dir(run_id)
        with self._lock(run_id):
            workspace = self._workspace(run_id)
            workspace.mkdir(parents=True, exist_ok=True)
            path = workspace / "project_manifest.json"
            wanted = blueprint.model_dump(mode="json")
            fingerprint = _hash(_canonical_json(wanted))
            if path.exists() or path.is_symlink():
                existing = self._read_manifest(workspace)
                if (existing["run_id"] != str(UUID(str(run_id)))
                        or existing["blueprint_sha256"] != fingerprint):
                    raise WorkspaceConflict(
                        "This run already uses a different project blueprint."
                    )
                return existing

            stamp = _timestamp()
            manifest = {
                "manifest_version": "1.0",
                "run_id": str(UUID(str(run_id))),
                "project_slug": blueprint.project_slug,
                "blueprint_revision": blueprint.revision,
                "blueprint_sha256": fingerprint,
                "blueprint": wanted,
                "created_at": stamp,
                "updated_at": stamp,
                "files": {
                    entry.path: {
                        "kind": entry.kind.value,
                        "producer": entry.producer.value,
                        "purpose": entry.purpose,
                        "depends_on": entry.depends_on,
                        "required": entry.required,
                        "status": "planned",
                        "revision": 0,
                        "sha256": None,
                        "size_bytes": None,
                        "written_by": None,
                        "updated_at": None,
                    }
                    for entry in blueprint.files
                },
                "stage_status": {
                    stage.value: "pending"
                    for stage in ProjectStage
                    if stage != ProjectStage.PLANNER
                },
                "final_zip_ready": False,
            }
            _atomic_write(path, _canonical_json(manifest))
            return manifest

    def get_manifest(self, *, run_id: UUID | str) -> dict:
        """Return a snapshot; not a claim that all files exist."""
        with self._lock(run_id):
            workspace = self._workspace(run_id)
            manifest = self._read_manifest(workspace)
            if manifest["run_id"] != str(UUID(str(run_id))):
                raise WorkspaceIntegrityError("Manifest belongs to another run.")
            return manifest

    def save_file(
        self,
        *,
        run_id: UUID | str,
        stage: ProjectStage,
        relative_path: str,
        content: str | bytes,
        replace: bool = False,
        expected_sha256: str | None = None,
    ) -> dict[str, str]:
        """Persist actual planned bytes and return a verifiable receipt.

        A stage owns its planned files. The leakage/refactoring stage may
        explicitly replace an EXISTING earlier-stage file, using its old
        SHA-256 as a concurrency check. Silent overwrites are forbidden.

        Human approvals and stage eligibility must be checked by RunService
        before it invokes this storage method.
        """
        stage = ProjectStage(stage)
        relative_path = validate_relative_path(relative_path)
        if not isinstance(content, (str, bytes)):
            raise TypeError("content must be UTF-8 text or bytes.")
        payload = content.encode("utf-8") if isinstance(content, str) else content
        with self._lock(run_id):
            workspace = self._workspace(run_id)
            manifest = self._read_manifest(workspace)
            if manifest["run_id"] != str(UUID(str(run_id))):
                raise WorkspaceIntegrityError("Manifest belongs to another run.")
            info = manifest["files"].get(relative_path)
            if info is None:
                raise WorkspaceError("Agent attempted to create an unplanned file.")
            blueprint = ProjectBlueprint.model_validate(manifest["blueprint"])
            planned = next(file for file in blueprint.files if file.path == relative_path)
            original_owner = ProjectStage(info["producer"])
            is_refactor = (
                stage == ProjectStage.LEAKAGE_REFACTORING
                and original_owner in {
                    ProjectStage.DATA_ENGINEERING,
                    ProjectStage.MODEL_TRAINING,
                }
                and replace
            )
            if stage != original_owner and not is_refactor:
                raise WorkspaceError("Stage cannot write a file owned by another stage.")
            self._check_source(relative_path, planned.kind, payload)
            target = self._file_path(workspace, relative_path)
            known = info["sha256"]
            desired = _hash(payload)

            if known is not None:
                if not target.is_file() or _hash(target.read_bytes()) != known:
                    raise WorkspaceIntegrityError(
                        "Existing file differs from recorded checksum."
                    )
                if desired == known:
                    return {"path": relative_path, "sha256": desired}
                if not replace or expected_sha256 != known:
                    raise WorkspaceConflict(
                        "Replacement requires the current expected_sha256."
                    )
            elif target.exists():
                # File may have been atomically published before a crash
                # prevented the manifest update. Exact replay is safe.
                if not target.is_file() or _hash(target.read_bytes()) != desired:
                    raise WorkspaceConflict("Untracked conflicting file exists.")
            elif is_refactor:
                raise WorkspaceError("Refactoring cannot invent an earlier-stage file.")

            target.parent.mkdir(parents=True, exist_ok=True)
            target = self._file_path(workspace, relative_path)
            if known is None and target.exists() and _hash(target.read_bytes()) == desired:
                pass  # Recover from a crash between file and manifest writes.
            else:
                _atomic_write(target, payload)

            info.update({
                "status": "revised" if known is not None else "created",
                "revision": int(info["revision"]) + 1,
                "sha256": desired,
                "size_bytes": len(payload),
                "written_by": stage.value,
                "updated_at": _timestamp(),
            })
            manifest["updated_at"] = _timestamp()
            if is_refactor:
                # Changes after testing invalidate later quality signoffs.
                for dependent_stage in (
                    ProjectStage.TESTING,
                    ProjectStage.VERIFICATION,
                    ProjectStage.DOCUMENTATION,
                    ProjectStage.PACKAGING,
                ):
                    manifest["stage_status"][dependent_stage.value] = "pending"
            _atomic_write(workspace / "project_manifest.json", _canonical_json(manifest))
            return {"path": relative_path, "sha256": desired}

    def save_stage_files(
        self,
        *,
        run_id: UUID | str,
        stage: ProjectStage,
        files: Mapping[str, str | bytes],
    ) -> list[dict[str, str]]:
        """Save code supplied by a trusted producer, without generating it.

        Repeating exact writes is safe. A partial failure leaves completed
        files recorded; the stage is NOT marked complete automatically.
        """
        receipts: list[dict[str, str]] = []
        for path, content in files.items():
            receipts.append(self.save_file(
                run_id=run_id, stage=stage,
                relative_path=path, content=content,
            ))
        return receipts

    def verify_stage(
        self, *, run_id: UUID | str, stage: ProjectStage,
    ) -> list[dict[str, str]]:
        """Verify that every required file for a stage exists and matches."""
        stage = ProjectStage(stage)
        with self._lock(run_id):
            workspace = self._workspace(run_id)
            manifest = self._read_manifest(workspace)
            if manifest["run_id"] != str(UUID(str(run_id))):
                raise WorkspaceIntegrityError("Manifest belongs to another run.")
            receipts: list[dict[str, str]] = []
            for path, info in manifest["files"].items():
                if info["producer"] != stage.value:
                    continue
                checksum = info["sha256"]
                if checksum is None:
                    if info["required"]:
                        raise WorkspaceIntegrityError(
                            f"Stage {stage.value} has not produced {path}."
                        )
                    continue
                target = self._file_path(workspace, path)
                if not target.is_file() or _hash(target.read_bytes()) != checksum:
                    raise WorkspaceIntegrityError(f"File integrity failure: {path}")
                receipts.append({"path": path, "sha256": checksum})
            return receipts

    def mark_stage_complete(
        self, *, run_id: UUID | str, stage: ProjectStage,
    ) -> dict:
        """Mark file production complete, NOT testing or human approval.

        For first-three-agent handoff only. Quality-control agent signoff is
        owned by later team stages and must not be inferred from this flag.
        """
        stage = ProjectStage(stage)
        if stage not in {ProjectStage.DATA_ENGINEERING, ProjectStage.MODEL_TRAINING}:
            raise WorkspaceError("Only first-three-agent file stages are managed here.")
        self.verify_stage(run_id=run_id, stage=stage)
        with self._lock(run_id):
            workspace = self._workspace(run_id)
            manifest = self._read_manifest(workspace)
            manifest["stage_status"][stage.value] = "files_written"
            manifest["updated_at"] = _timestamp()
            _atomic_write(workspace / "project_manifest.json", _canonical_json(manifest))
            return manifest
