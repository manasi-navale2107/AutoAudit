"""
Auditor handoff contracts for AutoAudit.

The handoff is a versioned evidence package produced after
the first three agents finish their work.

It does not claim that the pipeline is leakage-free or verified.

Artifact IDs are resolved by the backend artifact service.
Do not expose internal filesystem paths or deserialize
untrusted model artifacts in an API request handler.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.cleaning import CleaningPlan
from app.schemas.dataset import DatasetMetadata, DatasetProfile
from app.schemas.planning import (
    CandidateModel,
    EvaluationMetric,
    ExecutionPlan,
)
from app.schemas.review import ReviewAction, ReviewRecord


# ============================================================
# BASE SCHEMA
# ============================================================

class HandoffSchema(BaseModel):
    """Strict, versionable handoff structure."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        hide_input_in_errors=True,
    )


# ============================================================
# ARTIFACT REFERENCES
# ============================================================

class ArtifactKind(str, Enum):
    PIPELINE_SOURCE = "pipeline_source"
    TRAINED_PIPELINE = "trained_pipeline"
    SPLIT_MANIFEST = "split_manifest"
    ENVIRONMENT_LOCK = "environment_lock"
    REVIEW_HISTORY = "review_history"


class ArtifactReference(HandoffSchema):
    """
    Reference to an immutable, server-managed artifact.

    The service must verify that:
    - The artifact belongs to the current run.
    - Its stored bytes match the declared hash and size.
    - The requesting user may access it.

    UUIDs and hashes alone do not enforce access control.
    """

    artifact_id: UUID

    kind: ArtifactKind

    sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
    )

    size_bytes: int = Field(
        strict=True,
        ge=1,
    )


# ============================================================
# SPLIT EVIDENCE
# ============================================================

class SplitEvidence(HandoffSchema):
    """
    Actual split sizes and the manifest that records membership.

    The manifest must contain stable source-row identifiers,
    the split strategy, seed and relevant group/time metadata.

    Counts alone do not prove that partitions are disjoint.
    The auditor/verifier must inspect the manifest and data.

    The initial implementation assigns every uploaded row to
    exactly one partition. Missing/invalid targets must be resolved
    explicitly before execution; rows must not silently disappear.
    """

    manifest: ArtifactReference

    total_rows: int = Field(
        strict=True,
        ge=3,
    )

    train_rows: int = Field(
        strict=True,
        ge=1,
    )

    validation_rows: int = Field(
        strict=True,
        ge=1,
    )

    test_rows: int = Field(
        strict=True,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_split_evidence(self) -> Self:
        if self.manifest.kind != ArtifactKind.SPLIT_MANIFEST:
            raise ValueError(
                "Split evidence requires a split-manifest artifact."
            )

        if (
            self.train_rows
            + self.validation_rows
            + self.test_rows
            != self.total_rows
        ):
            raise ValueError(
                "Partition row counts must sum to total_rows."
            )

        return self


# ============================================================
# MODEL VALIDATION RESULTS
# ============================================================

class ModelValidationResult(HandoffSchema):
    """
    Primary validation score for one successfully trained model.

    Initial scope uses one explicit validation partition.
    Failed candidate attempts belong in run history.

    These are construction-stage results, not independently
    verified metrics.
    """

    model: CandidateModel

    metric: EvaluationMetric

    score: float = Field(
        ge=0.0,
        le=1.0,
    )

    validation_rows: int = Field(
        strict=True,
        ge=1,
    )

    training_seconds: float = Field(
        ge=0.0,
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

class PackageVersion(HandoffSchema):
    name: str = Field(
        min_length=1,
        max_length=200,
    )

    version: str = Field(
        min_length=1,
        max_length=100,
    )

    @field_validator("name", "version")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Package names and versions must not be blank."
            )

        return value


class ReproducibilityInfo(HandoffSchema):
    """
    Observed environment details from the execution process.

    The service reads these values from the actual environment.
    The LLM must not invent package versions or runtime details.

    Matching seeds and package versions do not guarantee
    bitwise-identical results across all hardware.
    """

    random_seed: int = Field(
        strict=True,
        ge=0,
        le=4_294_967_295,
    )

    python_version: str = Field(
        min_length=1,
        max_length=100,
    )

    platform: str = Field(
        min_length=1,
        max_length=300,
    )

    packages: list[PackageVersion] = Field(
        min_length=1,
    )

    environment_lock: ArtifactReference

    @field_validator("python_version", "platform")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Environment details must not be blank."
            )

        return value

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if (
            self.environment_lock.kind
            != ArtifactKind.ENVIRONMENT_LOCK
        ):
            raise ValueError(
                "An environment-lock artifact is required."
            )

        # Python package names normalize hyphens, underscores
        # and periods to equivalent separators.
        normalized_names = [
            package.name.lower().replace("_", "-").replace(".", "-")
            for package in self.packages
        ]

        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError(
                "Package versions must not contain duplicate names."
            )

        return self


# ============================================================
# COMPLETE AUDITOR HANDOFF
# ============================================================

class AuditorHandoff(HandoffSchema):
    """
    Evidence package for the next team's Leakage Auditor.

    The backend builds this from persisted run records and
    actual artifacts after training.

    Server-controlled revisions and artifact ownership must
    also be checked in the handoff service.
    """

    schema_version: Literal["1.0"] = "1.0"

    status: Literal["ready_for_audit"] = "ready_for_audit"

    run_id: UUID

    dataset: DatasetMetadata

    execution_plan: ExecutionPlan

    execution_plan_revision: int = Field(
        strict=True,
        ge=1,
    )

    cleaning_plan: CleaningPlan

    approval: ReviewRecord

    review_history: ArtifactReference

    split_evidence: SplitEvidence

    pipeline_source: ArtifactReference

    trained_pipeline: ArtifactReference

    validation_results: list[ModelValidationResult] = Field(
        min_length=1,
    )

    selected_model: CandidateModel

    reproducibility: ReproducibilityInfo

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "created_at must include timezone information."
            )

        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_handoff_integrity(self) -> Self:
        # ----------------------------------------------------
        # Run and dataset identity
        # ----------------------------------------------------

        if self.cleaning_plan.run_id != self.run_id:
            raise ValueError(
                "Cleaning plan belongs to a different run."
            )

        if self.approval.run_id != self.run_id:
            raise ValueError(
                "Approval belongs to a different run."
            )

        if (
            self.cleaning_plan.dataset_id
            != self.dataset.dataset_id
            or self.approval.dataset_id
            != self.dataset.dataset_id
        ):
            raise ValueError(
                "Dataset references must match."
            )

        # ----------------------------------------------------
        # Plan revisions and approval
        # ----------------------------------------------------

        if (
            self.cleaning_plan.execution_plan_revision
            != self.execution_plan_revision
            or self.approval.execution_plan_revision
            != self.execution_plan_revision
        ):
            raise ValueError(
                "Execution-plan revisions must match."
            )

        decision = self.approval.decision

        if decision.action != ReviewAction.APPROVE:
            raise ValueError(
                "Auditor handoff requires an approved cleaning plan."
            )

        if decision.plan_id != self.cleaning_plan.plan_id:
            raise ValueError(
                "Approval references a different cleaning plan."
            )

        if decision.plan_revision != self.cleaning_plan.revision:
            raise ValueError(
                "Approval references an outdated cleaning revision."
            )

        self.cleaning_plan.require_resolved_questions()

        if self.approval.reviewed_at < self.cleaning_plan.created_at:
            raise ValueError(
                "Approval must not precede the cleaning plan."
            )

        if self.created_at < self.approval.reviewed_at:
            raise ValueError(
                "Handoff must not precede approval."
            )

        # ----------------------------------------------------
        # Artifact types
        # ----------------------------------------------------

        if self.pipeline_source.kind != ArtifactKind.PIPELINE_SOURCE:
            raise ValueError(
                "pipeline_source requires a source-code artifact."
            )

        if (
            self.trained_pipeline.kind
            != ArtifactKind.TRAINED_PIPELINE
        ):
            raise ValueError(
                "trained_pipeline requires a trained-model artifact."
            )

        if self.review_history.kind != ArtifactKind.REVIEW_HISTORY:
            raise ValueError(
                "review_history requires a review-history artifact."
            )

        artifact_ids = [
            self.pipeline_source.artifact_id,
            self.trained_pipeline.artifact_id,
            self.review_history.artifact_id,
            self.split_evidence.manifest.artifact_id,
            self.reproducibility.environment_lock.artifact_id,
        ]

        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError(
                "Different artifact roles must use distinct IDs."
            )

        # ----------------------------------------------------
        # Validation results
        # ----------------------------------------------------

        models = [
            result.model
            for result in self.validation_results
        ]

        if len(models) != len(set(models)):
            raise ValueError(
                "Each candidate model must have one validation result."
            )

        if self.selected_model not in models:
            raise ValueError(
                "The selected model requires a validation result."
            )

        planned_models = set(
            self.execution_plan.candidate_models
        )

        for result in self.validation_results:
            if result.model not in planned_models:
                raise ValueError(
                    "Validation results contain an unplanned model."
                )

            if result.metric != self.execution_plan.primary_metric:
                raise ValueError(
                    "Results must use the planned primary metric."
                )

            if (
                result.validation_rows
                != self.split_evidence.validation_rows
            ):
                raise ValueError(
                    "Results must use the declared validation split."
                )

        # All currently supported primary metrics are maximized.
        best_score = max(
            result.score
            for result in self.validation_results
        )

        selected_result = next(
            result
            for result in self.validation_results
            if result.model == self.selected_model
        )

        if selected_result.score < best_score:
            raise ValueError(
                "The selected model must maximize the primary "
                "validation metric. Exact ties are allowed."
            )

        return self

    def validate_against_profile(
        self,
        profile: DatasetProfile,
    ) -> None:
        """
        Explicitly verify the external dataset-profile relationship.

        Call from the service before saving the handoff.
        File hashes, artifact bytes and split membership require
        additional checks outside this schema.
        """

        if profile.dataset_id != self.dataset.dataset_id:
            raise ValueError(
                "The profile belongs to a different dataset."
            )

        if self.split_evidence.total_rows != profile.row_count:
            raise ValueError(
                "Split evidence must account for every dataset row."
            )

        self.cleaning_plan.proposal.validate_against_context(
            execution_plan=self.execution_plan,
            profile=profile,
        )