
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from app.schemas.handoff import (
    ArtifactReference,
    AuditorHandoff,
)


class HandoffValidationError(ValueError):
    """Raised when the auditor handoff cannot be validated."""


class HandoffService:
    """Validate handoff data and verify referenced artifact files."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()

    def verify_artifact(
        self,
        artifact: ArtifactReference,
        file_path: str | Path,
    ) -> Path:
        """Verify an artifact's existence, size, and SHA-256 hash."""
        path = Path(file_path).resolve()

        if not path.is_relative_to(self.root_dir):
            raise HandoffValidationError(
                "Artifact path is outside the configured root directory."
            )

        if not path.is_file():
            raise HandoffValidationError(
                f"Artifact file is missing: {artifact.kind.value}"
            )

        digest = hashlib.sha256()
        size_bytes = 0

        try:
            with path.open("rb") as file:
                for chunk in iter(
                    lambda: file.read(1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)
                    size_bytes += len(chunk)
        except OSError:
            raise HandoffValidationError(
                "Artifact file could not be read."
            ) from None

        if size_bytes != artifact.size_bytes:
            raise HandoffValidationError(
                f"Artifact size mismatch: {artifact.kind.value}"
            )

        if digest.hexdigest() != artifact.sha256:
            raise HandoffValidationError(
                f"Artifact SHA-256 mismatch: {artifact.kind.value}"
            )

        return path

    def validate_handoff(
        self,
        data: dict[str, Any],
    ) -> AuditorHandoff:
        """Validate the handoff schema without claiming audit completion."""
        try:
            return AuditorHandoff.model_validate(data)
        except ValidationError as exc:
            raise HandoffValidationError(
                "Handoff data does not satisfy the required schema."
            ) from exc
