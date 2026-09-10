"""
Cleaning-plan contracts for AutoAudit.

Responsibilities:
- Represent proposed preprocessing decisions.
- Reject incompatible preprocessing combinations.
- Check column references against the execution plan and profile.
- Attach server-owned dataset/run identity and plan revision.

These schemas do not modify data or authorize execution.

Initial scope:
- Numeric, categorical, boolean and text features.
- Median/mode/constant imputation.
- Standard/robust scaling.
- One-hot encoding.
- Explicit feature exclusion.

Automatic row deletion, outlier removal and datetime feature
engineering are not supported in the initial implementation.
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

from app.schemas.dataset import (
    ColumnKind,
    DatasetProfile,
)

from app.schemas.planning import ExecutionPlan


# ============================================================
# BASE SCHEMA
# ============================================================

class CleaningSchema(BaseModel):
    """
    Preserve exact column names and reject unexpected fields.

    LLM-facing proposal fields are required.
    Inapplicable values must explicitly contain null.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        hide_input_in_errors=True,
    )


# ============================================================
# SUPPORTED OPERATIONS
# ============================================================

class FeatureAction(str, Enum):
    KEEP = "keep"
    EXCLUDE = "exclude"


class MissingValueStrategy(str, Enum):
    NONE = "none"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class ScalingStrategy(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"


class EncodingStrategy(str, Enum):
    NONE = "none"
    ONE_HOT = "one_hot"


# ============================================================
# ONE FEATURE'S PROPOSED POLICY
# ============================================================

class ColumnCleaningPolicy(CleaningSchema):
    """
    Proposed operations for one input feature.

    feature_kind is the proposed treatment, which may differ
    from inferred CSV dtype. For example, an integer-coded
    category may need categorical treatment.

    Constant filling currently supports a non-empty string
    for categorical/text/boolean features.

    Exclusion removes a column only from model inputs.
    It does not delete the original dataset column.
    """

    column: str = Field(min_length=1)

    action: FeatureAction

    feature_kind: ColumnKind

    missing_strategy: MissingValueStrategy

    fill_value: str | None = Field(
        min_length=1,
        max_length=200,
    )

    scaling: ScalingStrategy

    encoding: EncodingStrategy

    reason: str = Field(
        min_length=1,
        max_length=3000,
    )

    @field_validator("column", "reason", "fill_value")
    @classmethod
    def reject_blank_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "Text fields must not contain only whitespace."
            )

        return value

    @model_validator(mode="after")
    def validate_operations(self) -> Self:
        if self.action == FeatureAction.EXCLUDE:
            if (
                self.missing_strategy != MissingValueStrategy.NONE
                or self.scaling != ScalingStrategy.NONE
                or self.encoding != EncodingStrategy.NONE
                or self.fill_value is not None
            ):
                raise ValueError(
                    "Excluded features must not have "
                    "preprocessing operations."
                )

            return self

        if self.missing_strategy == MissingValueStrategy.CONSTANT:
            if self.fill_value is None:
                raise ValueError(
                    "Constant imputation requires fill_value."
                )

        elif self.fill_value is not None:
            raise ValueError(
                "fill_value must be null unless "
                "constant imputation is selected."
            )

        if self.feature_kind == ColumnKind.NUMERIC:
            if self.encoding != EncodingStrategy.NONE:
                raise ValueError(
                    "Numeric features must not use one-hot "
                    "encoding unless treated as categorical."
                )

            if self.missing_strategy == MissingValueStrategy.CONSTANT:
                raise ValueError(
                    "Numeric constant imputation is not supported "
                    "in the initial implementation."
                )

        elif self.feature_kind in {
            ColumnKind.CATEGORICAL,
            ColumnKind.BOOLEAN,
            ColumnKind.TEXT,
        }:
            if self.scaling != ScalingStrategy.NONE:
                raise ValueError(
                    "Non-numeric features must not be scaled."
                )

            if self.missing_strategy == MissingValueStrategy.MEDIAN:
                raise ValueError(
                    "Median imputation requires numeric treatment."
                )

            if self.encoding != EncodingStrategy.ONE_HOT:
                raise ValueError(
                    "Retained non-numeric features require "
                    "one-hot encoding in this implementation."
                )

        else:
            raise ValueError(
                "Retained datetime or unknown features need "
                "a supported feature-engineering policy first."
            )

        return self


# ============================================================
# LLM-FACING CLEANING PROPOSAL
# ============================================================

class CleaningProposal(CleaningSchema):
    """
    The Data Engineering Agent's proposed cleaning actions.

    Include exactly one policy for every non-target column.
    The target must never be imputed by feature preprocessing.

    unresolved_questions must be empty before approval.
    """

    summary: str = Field(
        min_length=1,
        max_length=4000,
    )

    policies: list[ColumnCleaningPolicy] = Field(
        min_length=1,
    )

    unresolved_questions: list[str]

    @field_validator("summary")
    @classmethod
    def reject_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "The cleaning summary must not be blank."
            )

        return value

    @field_validator("unresolved_questions")
    @classmethod
    def validate_questions(
        cls,
        values: list[str],
    ) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError(
                "Unresolved questions must not be blank."
            )

        return values

    @model_validator(mode="after")
    def validate_policy_list(self) -> Self:
        names = [
            policy.column
            for policy in self.policies
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Each column must have exactly one policy."
            )

        if not any(
            policy.action == FeatureAction.KEEP
            for policy in self.policies
        ):
            raise ValueError(
                "At least one input feature must be retained."
            )

        return self

    def validate_against_context(
        self,
        execution_plan: ExecutionPlan,
        profile: DatasetProfile,
    ) -> None:
        """
        Explicitly validate against trusted dataset/plan context.

        The service must call this after generating or revising
        a proposal and again before execution.

        Data-level validation remains necessary. For example:
        - Numeric conversion must succeed.
        - Training partitions must contain usable values.
        - One-hot cardinality must remain within resource limits.
        - Feature availability must match prediction time.
        """

        execution_plan.validate_against_profile(profile)

        expected_columns = {
            column.name
            for column in profile.columns
            if column.name != execution_plan.target_column
        }

        proposed_columns = {
            policy.column
            for policy in self.policies
        }

        if proposed_columns != expected_columns:
            raise ValueError(
                "Cleaning policies must cover every non-target "
                "column exactly once and must exclude the target."
            )

        profile_by_name = {
            column.name: column
            for column in profile.columns
        }

        required_exclusions = {
            exclusion.column
            for exclusion in execution_plan.feature_exclusions
        }

        for policy in self.policies:
            if policy.action == FeatureAction.EXCLUDE:
                continue

            if policy.column in required_exclusions:
                raise ValueError(
                    "A feature excluded by the execution plan "
                    "cannot be retained without revising that plan."
                )

            column_profile = profile_by_name[policy.column]

            if column_profile.missing_count == profile.row_count:
                raise ValueError(
                    "An entirely missing feature must be excluded "
                    "in the initial implementation."
                )

            if (
                column_profile.missing_count > 0
                and policy.missing_strategy
                == MissingValueStrategy.NONE
            ):
                raise ValueError(
                    "Retained features with missing values "
                    "require an explicit imputation strategy."
                )

            if (
                column_profile.numeric_summary is not None
                and column_profile.numeric_summary.infinite_count > 0
            ):
                raise ValueError(
                    "Retained features containing infinity "
                    "need an explicit supported repair first."
                )


# ============================================================
# SERVER-OWNED VERSIONED PLAN
# ============================================================

class CleaningPlan(CleaningSchema):
    """
    Persisted wrapper around a validated cleaning proposal.

    The backend, not the LLM or browser, assigns:
    - plan_id
    - run_id
    - dataset_id
    - revision
    - execution_plan_revision
    - created_at

    Every proposal change creates a new revision and invalidates
    previous approval. Database/service logic must enforce this.
    """

    plan_id: UUID

    run_id: UUID

    dataset_id: UUID

    revision: int = Field(
        strict=True,
        ge=1,
    )

    execution_plan_revision: int = Field(
        strict=True,
        ge=1,
    )

    proposal: CleaningProposal

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "created_at must include timezone information."
            )

        return value.astimezone(timezone.utc)

    def require_resolved_questions(self) -> None:
        """
        Necessary precondition for approval, not approval itself.

        The review service must additionally verify the current
        plan revision, run identity and explicit human decision.
        """

        if self.proposal.unresolved_questions:
            raise ValueError(
                "Resolve cleaning questions before approval."
            )