
import hashlib

import pytest

from app.schemas.handoff import ArtifactKind, ArtifactReference
from app.services.handoff_service import (
    HandoffService,
    HandoffValidationError,
)


@pytest.fixture
def handoff_root(tmp_path):
    return tmp_path


@pytest.fixture
def service(handoff_root):
    return HandoffService(root_dir=handoff_root)


def create_artifact_reference(content, kind=ArtifactKind.SPLIT_MANIFEST):
    return ArtifactReference(
        artifact_id="12345678-1234-5678-1234-567812345678",
        kind=kind,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def test_verify_artifact_accepts_valid_file(service, handoff_root):
    content = b'{"train": [0, 1], "test": [2]}'
    path = handoff_root / "splits.json"
    path.write_bytes(content)

    reference = create_artifact_reference(content)

    verified_path = service.verify_artifact(reference, path)

    assert verified_path == path.resolve()


def test_verify_artifact_rejects_missing_file(service, handoff_root):
    content = b"expected content"
    reference = create_artifact_reference(content)
    path = handoff_root / "missing.json"

    with pytest.raises(
        HandoffValidationError,
        match="Artifact file is missing",
    ):
        service.verify_artifact(reference, path)


def test_verify_artifact_rejects_size_mismatch(service, handoff_root):
    path = handoff_root / "splits.json"
    path.write_bytes(b"actual content")

    reference = create_artifact_reference(b"different content")

    with pytest.raises(
        HandoffValidationError,
        match="Artifact size mismatch",
    ):
        service.verify_artifact(reference, path)


def test_verify_artifact_rejects_hash_mismatch(service, handoff_root):
    path = handoff_root / "splits.json"
    path.write_bytes(b"actual content")

    reference = ArtifactReference(
        artifact_id="12345678-1234-5678-1234-567812345678",
        kind=ArtifactKind.SPLIT_MANIFEST,
        sha256="a" * 64,
        size_bytes=len(b"actual content"),
    )

    with pytest.raises(
        HandoffValidationError,
        match="SHA-256 mismatch",
    ):
        service.verify_artifact(reference, path)


def test_verify_artifact_rejects_path_outside_root(service, tmp_path):
    outside = tmp_path.parent / "outside-artifact.json"
    outside.write_bytes(b"content")

    reference = create_artifact_reference(b"content")

    try:
        with pytest.raises(
            HandoffValidationError,
            match="outside the configured root",
        ):
            service.verify_artifact(reference, outside)
    finally:
        outside.unlink(missing_ok=True)


def test_verify_artifact_rejects_directory(service, handoff_root):
    directory = handoff_root / "artifact-directory"
    directory.mkdir()

    reference = create_artifact_reference(b"content")

    with pytest.raises(
        HandoffValidationError,
        match="Artifact file is missing",
    ):
        service.verify_artifact(reference, directory)
