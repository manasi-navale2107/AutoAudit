"""
Human-review contracts for AutoAudit.

Responsibilities:
- Validate approval, change requests and rejection.
- Bind a decision to a specific cleaning-plan revision.
- Represent server-owned review history.
- Provide a consistency guard before processing a review.

These schemas do not authenticate users, persist reviews,
resume LangGraph, or authorize training by themselves.

The review service must atomically check the current pending
plan, record the decision, and update the run state.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.cleaning import CleaningPlan


# ============================================================
# BASE SCHEMA
# ============================================================

class ReviewSchema(BaseModel):
    """Reject unexpected fields and hide validation inputs."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        hide_input_in_errors=True,
    )


# ============================================================
# SUPPORTED REVIEW ACTIONS
# ============================================================

class ReviewAction(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


class FeedbackSource(str, Enum):
    TEXT = "text"
    VOICE_TRANSCRIPT = "voice_transcript"


# ============================================================
# ERRORS
# ============================================================

class ReviewConflictError(ValueError):
    """
    Review refers to an outdated or mismatched plan.

    The API layer should map this to HTTP 409 Conflict.
    """


class ReviewNotAllowedError(ValueError):
    """
    Current plan cannot accept the requested review action.

    The API layer should map this to HTTP 409 Conflict.
    """


# ============================================================
# USER REVIEW REQUEST
# ============================================================

class ReviewRequest(ReviewSchema):
    """
    Browser-submitted decision about a displayed cleaning plan.

    request_id:
        A UUID generated once per submission.
        Reuse it when retrying the same HTTP request.

    plan_id / plan_revision:
        Values from the plan displayed to the user.

    confirmed:
        True only after an explicit UI submission.
        This is not authentication or proof of user identity.

    Voice provides editable feedback text only.
    Voice transcription must not automatically submit approval.
    """

    request_id: UUID

    plan_id: UUID

    plan_revision: int = Field(
        strict=True,
        ge=1,
    )

    action: ReviewAction

    feedback: str = Field(
        default="",
        max_length=5000,
    )

    feedback_source: FeedbackSource = FeedbackSource.TEXT

    confirmed: bool = Field(
        strict=True,
    )

    @field_validator("feedback")
    @classmethod
    def normalize_feedback(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_review_action(self) -> Self:
        if not self.confirmed:
            raise ValueError(
                "An explicit user submission is required."
            )

        if self.action == ReviewAction.APPROVE:
            if self.feedback:
                raise ValueError(
                    "Approval must not include change instructions. "
                    "Use request_changes to modify the plan."
                )

            if self.feedback_source != FeedbackSource.TEXT:
                raise ValueError(
                    "Approval must use the explicit approval "
                    "control, not voice transcription."
                )

        elif self.action == ReviewAction.REQUEST_CHANGES:
            if not self.feedback:
                raise ValueError(
                    "Describe the requested changes."
                )

        if (
            self.feedback_source == FeedbackSource.VOICE_TRANSCRIPT
            and not self.feedback
        ):
            raise ValueError(
                "Voice feedback requires a reviewed transcript."
            )

        return self


# ============================================================
# SERVER-OWNED REVIEW HISTORY
# ============================================================

class ReviewRecord(ReviewSchema):
    """
    Persisted review event created by the backend.

    The server assigns review_id and reviewed_at.

    reviewer_id must come from a trusted identity/session layer,
    never directly from a browser-supplied field.

    Do not include raw feedback in ordinary application logs.
    """

    review_id: UUID

    run_id: UUID

    dataset_id: UUID

    reviewer_id: str = Field(
        min_length=1,
        max_length=200,
    )

    execution_plan_revision: int = Field(
        strict=True,
        ge=1,
    )

    decision: ReviewRequest

    reviewed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "reviewer_id must not be blank."
            )

        return value

    @field_validator("reviewed_at")
    @classmethod
    def validate_review_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "reviewed_at must include timezone information."
            )

        return value.astimezone(timezone.utc)


# ============================================================
# API RESPONSE
# ============================================================

class ReviewResponse(ReviewSchema):
    """
    Confirms that the review decision was recorded.

    Recording approval does not mean cleaning/training finished.
    Run progress is returned by the run-status API separately.
    """

    review: ReviewRecord

    message: str = Field(
        min_length=1,
        max_length=1000,
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Response message must not be blank."
            )

        return value


# ============================================================
# CURRENT-PLAN CONSISTENCY GUARD
# ============================================================

def validate_review_against_current_plan(
    *,
    request: ReviewRequest,
    current_plan: CleaningPlan,
    expected_run_id: UUID,
    expected_dataset_id: UUID,
    current_execution_plan_revision: int,
) -> None:
    """
    Check a review against trusted, server-loaded state.

    All values except request must come from the backend.

    The service must run this inside the same transaction or
    concurrency-control boundary used to record the review.

    This function does not check:
    - User authorization.
    - Whether the run is awaiting review.
    - Whether the request was already processed.
    - Whether the proposed operations remain safe.

    Those checks belong to the review/execution services.
    """

    if current_plan.run_id != expected_run_id:
        raise ReviewConflictError(
            "The cleaning plan does not belong to this run."
        )

    if current_plan.dataset_id != expected_dataset_id:
        raise ReviewConflictError(
            "The cleaning plan does not belong to this dataset."
        )

    if request.plan_id != current_plan.plan_id:
        raise ReviewConflictError(
            "The displayed cleaning plan is no longer current. "
            "Reload the plan before submitting a decision."
        )

    if request.plan_revision != current_plan.revision:
        raise ReviewConflictError(
            "The cleaning plan has changed. "
            "Review its latest revision."
        )

    if (
        current_plan.execution_plan_revision
        != current_execution_plan_revision
    ):
        raise ReviewConflictError(
            "The execution plan has changed. "
            "Regenerate the cleaning plan before review."
        )

    if request.action == ReviewAction.APPROVE:
        try:
            current_plan.require_resolved_questions()

        except ValueError as exc:
            raise ReviewNotAllowedError(
                "Resolve the cleaning plan's open questions "
                "before approval."
            ) from exc