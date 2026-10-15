
"""Validated request and response contracts for AutoAudit runs.

The existing dataset-backed RunCreateRequest and RunResponse remain
compatible with the first-three-agent workflow.

Separate goal-only project planning contracts allow the Planner to
design a complete project without pretending that a dataset,
trained model, generated source files, or ZIP already exists.

These are data contracts, not route implementations or authorization
checks.
"""

from __future__ import annotations

import re

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

from app.schemas.planning import ExecutionPlan

from app.schemas.project import (
    DatasetStatus,
    ProjectBlueprint,
    ProjectStage,
    validate_relative_path,
)


# ============================================================
# SHARED BASE SCHEMA
# ============================================================

class RunSchema(BaseModel):
    """Reject unknown fields rather than silently ignoring inputs."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        hide_input_in_errors=True,
    )


# ============================================================
# PROVIDER
# ============================================================

class ProviderName(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"


# ============================================================
# EXISTING RUN STATUSES
# ============================================================

class RunStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PROPOSING_CLEANING = "proposing_cleaning"
    AWAITING_REVIEW = "awaiting_review"
    REVISING_CLEANING = "revising_cleaning"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    READY_FOR_AUDIT = "ready_for_audit"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# RUN ERROR CODES
# ============================================================

class RunErrorCode(str, Enum):
    DATASET_UNAVAILABLE = "dataset_unavailable"
    INVALID_DATASET = "invalid_dataset"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    INVALID_PLAN = "invalid_plan"
    SPLIT_FAILED = "split_failed"
    PREPROCESSING_FAILED = "preprocessing_failed"
    TRAINING_FAILED = "training_failed"
    EXECUTION_TIMEOUT = "execution_timeout"
    STORAGE_FAILED = "storage_failed"
    INTERNAL_ERROR = "internal_error"


LanguageCode = Literal["en", "mr", "hi"]


# ============================================================
# EXISTING DATASET-BACKED RUN REQUEST
# ============================================================

class RunCreateRequest(RunSchema):
    """
    Original dataset-backed run request.

    The current RunService.start() loads dataset_id before planning.

    This request still requires a dataset so existing training runs
    cannot accidentally start without real training data.
    """

    request_id: UUID

    dataset_id: UUID

    objective: str = Field(
        min_length=5,
        max_length=5000,
    )

    target_column: str | None = Field(
        default=None,
        min_length=1,
    )

    language: LanguageCode | None = None

    provider: ProviderName | None = None

    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    @field_validator("objective", "model")
    @classmethod
    def normalize_text(
        cls,
        value: str | None,
    ) -> str | None:

        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "Text values must not be blank."
            )

        return value

    @field_validator("target_column")
    @classmethod
    def validate_target_column(
        cls,
        value: str | None,
    ) -> str | None:

        if value is not None and not value.strip():
            raise ValueError(
                "Use null when the target is not specified."
            )

        # Preserve the exact dataset column spelling.
        return value

    @model_validator(mode="after")
    def validate_provider_selection(self) -> Self:

        if self.model is not None and self.provider is None:
            raise ValueError(
                "Select a provider when specifying a model."
            )

        return self


# ============================================================
# NEW: GOAL-ONLY PROJECT PLANNING REQUEST
# ============================================================

class ProjectPlanningRequest(RunSchema):
    """
    Goal-only or dataset-assisted project-blueprint request.

    A dataset is OPTIONAL for planning.

    A dataset is NOT optional for actual fitted-model training.

    RunService and API routes must separately implement support
    for this request.
    """

    request_id: UUID

    objective: str = Field(
        min_length=5,
        max_length=5000,
    )

    dataset_id: UUID | None = None

    target_column: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    language: LanguageCode | None = None

    provider: ProviderName | None = None

    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    @field_validator("objective", "model")
    @classmethod
    def trim_text(
        cls,
        value: str | None,
    ) -> str | None:

        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "Text must not be blank."
            )

        return value

    @field_validator("target_column")
    @classmethod
    def keep_exact_target(
        cls,
        value: str | None,
    ) -> str | None:

        if value is not None and not value.strip():
            raise ValueError(
                "Use null for an unknown target column."
            )

        return value

    @model_validator(mode="after")
    def validate_selection(self) -> Self:

        if self.model is not None and self.provider is None:
            raise ValueError(
                "Select a provider when specifying a model."
            )

        return self


# ============================================================
# SERVER-RESOLVED LLM CONFIGURATION
# ============================================================

class RunLLMConfiguration(RunSchema):
    """
    Immutable, server-resolved, nonsecret LLM configuration.

    Never expose API keys through this schema.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        hide_input_in_errors=True,
        frozen=True,
    )

    provider: ProviderName

    model: str = Field(
        min_length=1,
        max_length=200,
    )

    temperature: float = Field(
        ge=0.0,
        le=2.0,
    )

    max_output_tokens: int = Field(
        strict=True,
        ge=256,
        le=32768,
    )

    timeout_seconds: float = Field(
        gt=0.0,
        le=1800.0,
    )

    max_retries: int = Field(
        strict=True,
        ge=0,
        le=5,
    )

    ollama_num_ctx: int | None = Field(
        strict=True,
        ge=1024,
        le=131072,
    )

    @field_validator("model")
    @classmethod
    def validate_model_name(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Resolved model ID must not be blank."
            )

        return value

    @model_validator(mode="after")
    def validate_provider_options(self) -> Self:

        if self.provider == ProviderName.OLLAMA:

            if self.ollama_num_ctx is None:
                raise ValueError(
                    "Ollama requires a context limit."
                )

        elif self.ollama_num_ctx is not None:
            raise ValueError(
                "Groq must use null for ollama_num_ctx."
            )

        return self


# ============================================================
# PUBLIC RUN ERROR
# ============================================================

class RunError(RunSchema):
    """
    Public-safe error details.

    Never copy credentials, raw tracebacks or internal paths here.
    """

    code: RunErrorCode

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    retryable: bool = Field(
        strict=True,
    )

    @field_validator("message")
    @classmethod
    def reject_blank_message(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "An error message must not be blank."
            )

        return value


# ============================================================
# EXISTING HUMAN REVIEW REFERENCE
# ============================================================

class PendingReviewReference(RunSchema):
    """
    Reference to the exact cleaning plan awaiting user review.
    """

    plan_id: UUID

    plan_revision: int = Field(
        strict=True,
        ge=1,
    )

    execution_plan_revision: int = Field(
        strict=True,
        ge=1,
    )


# ============================================================
# EXISTING RUN RESPONSE
# ============================================================

class RunResponse(RunSchema):
    """
    Public snapshot of a dataset-backed workflow run.

    Existing Human-in-the-Loop validation is preserved.
    """

    run_id: UUID

    dataset_id: UUID

    status: RunStatus

    state_version: int = Field(
        strict=True,
        ge=1,
    )

    objective: str = Field(
        min_length=5,
        max_length=5000,
    )

    language: LanguageCode

    llm: RunLLMConfiguration

    progress_message: str = Field(
        min_length=1,
        max_length=2000,
    )

    execution_plan: ExecutionPlan | None = None

    execution_plan_revision: int | None = Field(
        default=None,
        strict=True,
        ge=1,
    )

    clarification_questions: list[str] = Field(
        default_factory=list,
    )

    pending_review: PendingReviewReference | None = None

    error: RunError | None = None

    handoff_artifact_id: UUID | None = None

    created_at: datetime

    updated_at: datetime

    @field_validator("objective", "progress_message")
    @classmethod
    def reject_blank_text(
        cls,
        value: str,
    ) -> str:

        if not value.strip():
            raise ValueError(
                "Text fields must not be blank."
            )

        return value

    @field_validator("clarification_questions")
    @classmethod
    def validate_questions(
        cls,
        values: list[str],
    ) -> list[str]:

        if any(
            not question.strip()
            for question in values
        ):
            raise ValueError(
                "Clarification questions must not be blank."
            )

        return values

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime,
    ) -> datetime:

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "Run timestamps must include a timezone."
            )

        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:

        if self.updated_at < self.created_at:
            raise ValueError(
                "updated_at must not precede created_at."
            )

        if (
            (self.execution_plan is None)
            != (self.execution_plan_revision is None)
        ):
            raise ValueError(
                "Execution plan and revision must appear together."
            )

        if self.status == RunStatus.AWAITING_CLARIFICATION:

            if not self.clarification_questions:
                raise ValueError(
                    "Awaiting clarification requires questions."
                )

        elif self.clarification_questions:
            raise ValueError(
                "Questions require awaiting_clarification status."
            )

        if self.status == RunStatus.AWAITING_REVIEW:

            if (
                self.pending_review is None
                or self.execution_plan is None
            ):
                raise ValueError(
                    "Cleaning review requires plan and review reference."
                )

            if (
                self.pending_review.execution_plan_revision
                != self.execution_plan_revision
            ):
                raise ValueError(
                    "Review must reference current execution plan."
                )

        elif self.pending_review is not None:
            raise ValueError(
                "Pending review requires awaiting_review status."
            )

        if self.status == RunStatus.FAILED:

            if self.error is None:
                raise ValueError(
                    "Failed runs require safe error details."
                )

        elif self.error is not None:
            raise ValueError(
                "Error details require failed status."
            )

        stages_requiring_plan = {
            RunStatus.PROPOSING_CLEANING,
            RunStatus.AWAITING_REVIEW,
            RunStatus.REVISING_CLEANING,
            RunStatus.PREPROCESSING,
            RunStatus.TRAINING,
            RunStatus.READY_FOR_AUDIT,
        }

        if (
            self.status in stages_requiring_plan
            and self.execution_plan is None
        ):
            raise ValueError(
                "This stage requires an execution plan."
            )

        if self.status == RunStatus.READY_FOR_AUDIT:

            if self.handoff_artifact_id is None:
                raise ValueError(
                    "Ready-for-audit requires an artifact reference."
                )

        elif self.handoff_artifact_id is not None:
            raise ValueError(
                "Handoff reference is only for ready-for-audit runs."
            )

        return self


# ============================================================
# NEW: PROJECT PLANNING STATUSES
# ============================================================

class ProjectPlanningStatus(str, Enum):

    PLANNING = "planning"

    PLANNED = "planned"

    AWAITING_DATASET = "awaiting_dataset"

    NEEDS_CLARIFICATION = "needs_clarification"

    UNSUPPORTED = "unsupported"

    FAILED = "failed"


# ============================================================
# NEW: PROJECT PLANNING RESPONSE
# ============================================================

class ProjectPlanningResponse(RunSchema):
    """
    Response for goal-only or dataset-assisted project planning.

    A blueprint is NOT a trained model or completed ZIP.
    """

    run_id: UUID

    objective: str = Field(
        min_length=5,
        max_length=5000,
    )

    dataset_id: UUID | None = None

    status: ProjectPlanningStatus

    language: LanguageCode

    blueprint: ProjectBlueprint | None = None

    clarification_questions: list[str] = Field(
        default_factory=list,
        max_length=3,
    )

    error: RunError | None = None

    final_zip_ready: Literal[False] = False

    @field_validator("objective")
    @classmethod
    def ensure_goal(
        cls,
        value: str,
    ) -> str:

        if not value.strip():
            raise ValueError(
                "Project objective must not be blank."
            )

        return value

    @field_validator("clarification_questions")
    @classmethod
    def ensure_questions(
        cls,
        questions: list[str],
    ) -> list[str]:

        if any(
            not question.strip()
            for question in questions
        ):
            raise ValueError(
                "Clarification questions must be nonblank."
            )

        return questions

    @model_validator(mode="after")
    def check_project_state(self) -> Self:

        has_plan = self.status in {
            ProjectPlanningStatus.PLANNED,
            ProjectPlanningStatus.AWAITING_DATASET,
        }

        if has_plan != (self.blueprint is not None):
            raise ValueError(
                "A planned state must include a blueprint."
            )

        if self.blueprint is not None:

            if self.blueprint.goal != self.objective:
                raise ValueError(
                    "Blueprint goal differs from user objective."
                )

            expected_dataset_status = (
                DatasetStatus.AVAILABLE
                if self.dataset_id is not None
                else DatasetStatus.REQUIRED
            )

            if (
                self.blueprint.dataset.status
                != expected_dataset_status
            ):
                raise ValueError(
                    "Blueprint dataset status differs from request."
                )

            if (
                self.status == ProjectPlanningStatus.AWAITING_DATASET
                and self.dataset_id is not None
            ):
                raise ValueError(
                    "Awaiting dataset cannot report an attached dataset."
                )

            if (
                self.status == ProjectPlanningStatus.PLANNED
                and self.dataset_id is None
            ):
                raise ValueError(
                    "Without dataset, status must be awaiting_dataset."
                )

        if self.status == ProjectPlanningStatus.NEEDS_CLARIFICATION:

            if not self.clarification_questions:
                raise ValueError(
                    "Clarification status requires questions."
                )

        elif self.clarification_questions:
            raise ValueError(
                "Questions require needs_clarification status."
            )

        if self.status == ProjectPlanningStatus.FAILED:

            if self.error is None:
                raise ValueError(
                    "Failed planning requires safe error details."
                )

        elif self.error is not None:
            raise ValueError(
                "Error details require failed status."
            )

        return self


# ============================================================
# NEW: INDIVIDUAL PROJECT FILE PROGRESS
# ============================================================

class ProjectFileProgress(RunSchema):
    """
    Tracks one planned or physically generated project file.

    A file cannot be marked generated without a recorded checksum.
    """

    path: str

    producer: ProjectStage

    status: Literal[
        "planned",
        "created",
        "revised",
    ]

    revision: int = Field(
        strict=True,
        ge=0,
    )

    sha256: str | None = None

    @field_validator("path")
    @classmethod
    def check_path(
        cls,
        value: str,
    ) -> str:

        return validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def check_hash(
        cls,
        value: str | None,
    ) -> str | None:

        if (
            value is not None
            and not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise ValueError(
                "SHA-256 must be 64 lowercase hex characters."
            )

        return value

    @model_validator(mode="after")
    def check_file_state(self) -> Self:

        if self.status == "planned":

            if (
                self.revision != 0
                or self.sha256 is not None
            ):
                raise ValueError(
                    "Planned files must have revision 0 and no hash."
                )

        else:

            if (
                self.revision < 1
                or self.sha256 is None
            ):
                raise ValueError(
                    "Generated files require revision and checksum."
                )

        return self


# ============================================================
# NEW: PROJECT WORKSPACE PROGRESS RESPONSE
# ============================================================

class ProjectProgressResponse(RunSchema):
    """
    Read-only project generation progress.

    This contract covers the initial project-building workflow,
    not the final Packaging Agent.
    """

    run_id: UUID

    planned: bool

    project_slug: str | None = None

    blueprint_revision: int | None = Field(
        default=None,
        ge=1,
    )

    files: list[ProjectFileProgress] = Field(
        default_factory=list,
    )

    stage_status: dict[str, str] = Field(
        default_factory=dict,
    )

    final_zip_ready: bool = False

    @model_validator(mode="after")
    def check_progress(self) -> Self:

        if self.planned != (self.project_slug is not None):
            raise ValueError(
                "A planned workspace requires its project slug."
            )

        if self.planned != (self.blueprint_revision is not None):
            raise ValueError(
                "Blueprint revision must match planned state."
            )

        if not self.planned:

            if (
                self.files
                or self.stage_status
                or self.final_zip_ready
            ):
                raise ValueError(
                    "Unplanned projects cannot have generated files or ZIPs."
                )

        if self.final_zip_ready:
            raise ValueError(
                "Only the later Packaging Agent can enable ZIP delivery."
            )

        file_paths = {
            item.path.casefold()
            for item in self.files
        }

        if len(file_paths) != len(self.files):
            raise ValueError(
                "Duplicate project file progress paths."
            )

        return self
