"""
Run contracts for AutoAudit's first-three-agent workflow.

Responsibilities:
- Validate new-run requests.
- Represent server-resolved LLM configuration.
- Describe workflow progress and review checkpoints.
- Return safe, structured run-status responses.

These schemas do not start background work, authorize users,
enforce state transitions, or make provider API calls.

The run service must enforce those responsibilities.
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

from app.schemas.planning import ExecutionPlan


# ============================================================
# SHARED BASE
# ============================================================

class RunSchema(BaseModel):
    """Reject unexpected fields and invalid numeric values."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        hide_input_in_errors=True,
    )


# ============================================================
# SUPPORTED VALUES
# ============================================================

class ProviderName(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"


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
# NEW RUN REQUEST
# ============================================================

class RunCreateRequest(RunSchema):
    """
    User request to start a workflow.

    request_id:
        Idempotency identifier. Reuse the same UUID when retrying
        the same submission.

    target_column:
        Optional user-specified target. Preserve its exact name.
        When omitted, the Planner may ask for clarification.

    provider:
        Optional provider override. Otherwise use server defaults.

    model:
        Optional model selection for the explicitly chosen provider.
        The backend must check its allowlist and availability.

    This request deliberately excludes API keys, server URLs,
    filesystem paths, raw Python code and ownership fields.
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

        # Do not strip: match the original dataset column exactly.
        return value

    @model_validator(mode="after")
    def validate_provider_selection(self) -> Self:
        if self.model is not None and self.provider is None:
            raise ValueError(
                "Select a provider when specifying a model."
            )

        if len(self.objective) < 5:
            raise ValueError(
                "Provide an objective with at least "
                "five non-padding characters."
            )

        return self


# ============================================================
# SERVER-RESOLVED LLM CONFIGURATION
# ============================================================

class RunLLMConfiguration(RunSchema):
    """
    Non-secret LLM settings fixed for this run.

    The server resolves defaults before creating this object.
    Provider-specific capabilities require additional checks.

    A provider/model change applies to a new run.
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

    # Required key; null for Groq.
    ollama_num_ctx: int | None = Field(
        strict=True,
        ge=1024,
        le=131072,
    )

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "The resolved model ID must not be blank."
            )

        return value

    @model_validator(mode="after")
    def validate_provider_options(self) -> Self:
        if self.provider == ProviderName.OLLAMA:
            if self.ollama_num_ctx is None:
                raise ValueError(
                    "Ollama requires an explicit context limit."
                )

        elif self.ollama_num_ctx is not None:
            raise ValueError(
                "Groq configuration must use null "
                "for ollama_num_ctx."
            )

        return self


# ============================================================
# PUBLIC ERROR DETAILS
# ============================================================

class RunError(RunSchema):
    """
    Safe error information for the interface.

    message must be composed by the application.
    Do not copy raw provider responses, exception messages,
    stack traces, credentials or internal paths into it.

    retryable is guidance, not authorization to replay work.
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
    def reject_blank_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "An error message must not be blank."
            )

        return value


# ============================================================
# PENDING CLEANING REVIEW
# ============================================================

class PendingReviewReference(RunSchema):
    """
    Reference to the exact cleaning plan awaiting user review.

    The review endpoint uses these values to detect stale approval.
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
# RUN STATUS RESPONSE
# ============================================================

class RunResponse(RunSchema):
    """
    Public snapshot of a workflow run.

    The backend assigns IDs, timestamps, state_version and status.

    state_version supports optimistic concurrency in storage.
    This schema alone does not enforce atomic state updates.

    progress_message describes the current stage.
    It is not a fabricated percentage estimate.
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

    # Server-generated artifact reference, not a filesystem path.
    handoff_artifact_id: UUID | None = None

    created_at: datetime

    updated_at: datetime

    @field_validator("objective", "progress_message")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
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
        if any(not value.strip() for value in values):
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
                "Run timestamps must include timezone information."
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
                "Execution plan and revision must be "
                "present together."
            )

        if self.status == RunStatus.AWAITING_CLARIFICATION:
            if not self.clarification_questions:
                raise ValueError(
                    "Awaiting clarification requires questions."
                )

        elif self.clarification_questions:
            raise ValueError(
                "Active clarification questions require "
                "awaiting_clarification status."
            )

        if self.status == RunStatus.AWAITING_REVIEW:
            if self.pending_review is None:
                raise ValueError(
                    "Awaiting review requires a plan reference."
                )

            if self.execution_plan is None:
                raise ValueError(
                    "Cleaning review requires an execution plan."
                )

            if (
                self.pending_review.execution_plan_revision
                != self.execution_plan_revision
            ):
                raise ValueError(
                    "Pending review must match the current "
                    "execution-plan revision."
                )

        elif self.pending_review is not None:
            raise ValueError(
                "A pending review reference requires "
                "awaiting_review status."
            )

        if self.status == RunStatus.FAILED:
            if self.error is None:
                raise ValueError(
                    "Failed runs require safe error details."
                )

        elif self.error is not None:
            raise ValueError(
                "Active error details require failed status."
            )

        statuses_requiring_plan = {
            RunStatus.PROPOSING_CLEANING,
            RunStatus.AWAITING_REVIEW,
            RunStatus.REVISING_CLEANING,
            RunStatus.PREPROCESSING,
            RunStatus.TRAINING,
            RunStatus.READY_FOR_AUDIT,
        }

        if (
            self.status in statuses_requiring_plan
            and self.execution_plan is None
        ):
            raise ValueError(
                "This workflow stage requires an execution plan."
            )

        if self.status == RunStatus.READY_FOR_AUDIT:
            if self.handoff_artifact_id is None:
                raise ValueError(
                    "Ready-for-audit runs require "
                    "a handoff artifact reference."
                )

        elif self.handoff_artifact_id is not None:
            raise ValueError(
                "The handoff reference is exposed only "
                "when the run is ready for audit."
            )

        return self