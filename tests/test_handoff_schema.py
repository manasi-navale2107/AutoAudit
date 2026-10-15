import pytest
from pydantic import ValidationError

from app.schemas.handoff import (
    ArtifactKind,
    ArtifactReference,
    SplitEvidence,
)


def make_artifact_reference(kind=ArtifactKind.SPLIT_MANIFEST):
    return ArtifactReference(
        artifact_id="12345678-1234-5678-1234-567812345678",
        kind=kind,
        sha256="a" * 64,
        size_bytes=100,
    )


def test_valid_split_evidence_is_accepted():
    evidence = SplitEvidence(
        manifest=make_artifact_reference(),
        total_rows=100,
        train_rows=70,
        validation_rows=15,
        test_rows=15,
    )

    assert evidence.total_rows == 100
    assert (
        evidence.train_rows
        + evidence.validation_rows
        + evidence.test_rows
        == evidence.total_rows
    )


def test_split_evidence_rejects_wrong_artifact_kind():
    with pytest.raises(ValidationError):
        SplitEvidence(
            manifest=make_artifact_reference(
                ArtifactKind.TRAINED_PIPELINE
            ),
            total_rows=100,
            train_rows=70,
            validation_rows=15,
            test_rows=15,
        )


def test_split_evidence_rejects_incorrect_row_total():
    with pytest.raises(ValidationError):
        SplitEvidence(
            manifest=make_artifact_reference(),
            total_rows=100,
            train_rows=70,
            validation_rows=15,
            test_rows=10,
        )


def test_split_evidence_rejects_empty_partition():
    with pytest.raises(ValidationError):
        SplitEvidence(
            manifest=make_artifact_reference(),
            total_rows=100,
            train_rows=85,
            validation_rows=15,
            test_rows=0,
        )


def test_artifact_reference_rejects_invalid_sha256():
    with pytest.raises(ValidationError):
        ArtifactReference(
            artifact_id="12345678-1234-5678-1234-567812345678",
            kind=ArtifactKind.SPLIT_MANIFEST,
            sha256="not-a-valid-hash",
            size_bytes=100,
        )


def test_artifact_reference_rejects_zero_size():
    with pytest.raises(ValidationError):
        ArtifactReference(
            artifact_id="12345678-1234-5678-1234-567812345678",
            kind=ArtifactKind.SPLIT_MANIFEST,
            sha256="a" * 64,
            size_bytes=0,
        )

def test_environment_lock_requires_correct_artifact_kind():
    from app.schemas.handoff import ReproducibilityInfo

    with pytest.raises(ValidationError):
        ReproducibilityInfo(
            random_seed=42,
            python_version="3.11.9",
            platform="Windows",
            packages=[
                {"name": "scikit-learn", "version": "1.5.0"}
            ],
            environment_lock=make_artifact_reference(
                ArtifactKind.SPLIT_MANIFEST
            ),
        )


def test_environment_lock_accepts_correct_artifact_kind():
    from app.schemas.handoff import ReproducibilityInfo

    info = ReproducibilityInfo(
        random_seed=42,
        python_version="3.11.9",
        platform="Windows",
        packages=[
            {"name": "scikit-learn", "version": "1.5.0"}
        ],
        environment_lock=make_artifact_reference(
            ArtifactKind.ENVIRONMENT_LOCK
        ),
    )

    assert info.random_seed == 42
    assert info.environment_lock.kind == ArtifactKind.ENVIRONMENT_LOCK
