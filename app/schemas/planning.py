"""
Planning contracts for AutoAudit.

The Planner proposes:
- The classification task and target.
- Evaluation metric.
- Validation strategy.
- Candidate models.
- Feature exclusions and their reasons.

A valid plan is a proposal, not approval to clean or train.

All LLM-facing fields are required. Fields that do not apply
must explicitly contain null or an empty list.
"""

from enum import Enum
from math import isclose

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from typing import Self

from app.schemas.dataset import DatasetProfile


# ============================================================
# SHARED BASE
# ============================================================

class PlanningSchema(BaseModel):
    """Strict structure without silently changing column names."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        hide_input_in_errors=True,
    )


# ============================================================
# SUPPORTED VALUES
# ============================================================

class PlanningStatus(str, Enum):
    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class ClassificationTask(str, Enum):
    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"


class EvaluationMetric(str, Enum):
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    ROC_AUC = "roc_auc"


class SplitStrategy(str, Enum):
    STRATIFIED = "stratified"
    GROUP = "group"
    TEMPORAL = "temporal"


class CandidateModel(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"


# ============================================================
# FEATURE EXCLUSION PROPOSAL
# ============================================================

class FeatureExclusion(PlanningSchema):
    """
    Proposed exclusion from model inputs.

    The original column remains in the raw dataset.
    The target is handled separately and must not appear here.
    """

    column: str = Field(min_length=1)

    reason: str = Field(
        min_length=1,
        max_length=2000,
    )

    @field_validator("column", "reason")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Column and reason must not be blank."
            )

        return value


# ============================================================
# SPLIT PLAN
# ============================================================

class SplitPlan(PlanningSchema):
    """
    Fractions refer to the complete eligible dataset.

    For group splitting, whole groups stay together.
    For temporal splitting, chronology takes precedence.

    Fractions are desired proportions; group/time boundaries
    may prevent exact proportions.

    Actual split integrity is checked by the splitting service.
    """

    strategy: SplitStrategy

    train_fraction: float = Field(
        gt=0.0,
        lt=1.0,
    )

    validation_fraction: float = Field(
        gt=0.0,
        lt=1.0,
    )

    test_fraction: float = Field(
        gt=0.0,
        lt=1.0,
    )

    # Required keys, explicitly null when not applicable.
    group_column: str | None = Field(
        min_length=1,
    )

    time_column: str | None = Field(
        min_length=1,
    )

    rationale: str = Field(
        min_length=1,
        max_length=3000,
    )

    @field_validator(
        "group_column",
        "time_column",
        "rationale",
    )
    @classmethod
    def reject_blank_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "Use null for an inapplicable column; "
                "text values must not be blank."
            )

        return value

    @model_validator(mode="after")
    def validate_split_policy(self) -> Self:
        total = (
            self.train_fraction
            + self.validation_fraction
            + self.test_fraction
        )

        if not isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Train, validation and test fractions "
                "must sum to 1."
            )

        if self.strategy == SplitStrategy.STRATIFIED:
            if (
                self.group_column is not None
                or self.time_column is not None
            ):
                raise ValueError(
                    "Stratified splitting must not declare "
                    "group or time columns."
                )

        elif self.strategy == SplitStrategy.GROUP:
            if self.group_column is None:
                raise ValueError(
                    "Group splitting requires group_column."
                )

            if self.time_column is not None:
                raise ValueError(
                    "Combined group-and-time splitting is not "
                    "supported by this initial split contract."
                )

        elif self.strategy == SplitStrategy.TEMPORAL:
            if self.time_column is None:
                raise ValueError(
                    "Temporal splitting requires time_column."
                )

            if self.group_column is not None:
                raise ValueError(
                    "Combined group-and-time splitting is not "
                    "supported by this initial split contract."
                )

        return self


# ============================================================
# COMPLETE EXECUTION PLAN
# ============================================================

class ExecutionPlan(PlanningSchema):
    """
    A proposed classification plan.

    Schema validation checks consistency, not domain correctness.
    User requirements and actual data must also be checked.
    """

    objective: str = Field(
        min_length=5,
        max_length=5000,
    )

    task_type: ClassificationTask

    target_column: str = Field(
        min_length=1,
    )

    # Describe when predictions will be made.
    # This helps identify features unavailable at prediction time.
    prediction_context: str = Field(
        min_length=1,
        max_length=3000,
    )

    primary_metric: EvaluationMetric

    metric_rationale: str = Field(
        min_length=1,
        max_length=2000,
    )

    # Required for binary ROC-AUC in this initial implementation.
    # The training service must resolve this to an actual label.
    positive_class_label: str | None = Field(
        min_length=1,
    )

    split: SplitPlan

    candidate_models: list[CandidateModel] = Field(
        min_length=1,
        max_length=2,
    )

    feature_exclusions: list[FeatureExclusion]

    assumptions: list[str]

    @field_validator(
        "objective",
        "target_column",
        "prediction_context",
        "metric_rationale",
        "positive_class_label",
    )
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

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(
        cls,
        values: list[str],
    ) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError(
                "Assumptions must not contain blank entries."
            )

        return values

    @model_validator(mode="after")
    def validate_plan_consistency(self) -> Self:
        if len(self.candidate_models) != len(
            set(self.candidate_models)
        ):
            raise ValueError(
                "Candidate models must not repeat."
            )

        excluded_columns = [
            item.column
            for item in self.feature_exclusions
        ]

        if len(excluded_columns) != len(set(excluded_columns)):
            raise ValueError(
                "Feature exclusions must not repeat."
            )

        if self.target_column in excluded_columns:
            raise ValueError(
                "The target is handled separately; "
                "do not list it as a feature exclusion."
            )

        if self.target_column in {
            self.split.group_column,
            self.split.time_column,
        }:
            raise ValueError(
                "The target must not also be the group "
                "or time column."
            )

        if self.primary_metric == EvaluationMetric.ROC_AUC:
            if self.task_type != ClassificationTask.BINARY:
                raise ValueError(
                    "The initial ROC-AUC implementation "
                    "supports binary classification only."
                )

            if self.positive_class_label is None:
                raise ValueError(
                    "Binary ROC-AUC requires an explicit "
                    "positive_class_label."
                )

        if (
            self.task_type == ClassificationTask.MULTICLASS
            and self.positive_class_label is not None
        ):
            raise ValueError(
                "Multiclass plans must use null "
                "for positive_class_label."
            )

        return self

    def validate_against_profile(
        self,
        profile: DatasetProfile,
    ) -> None:
        """
        Check references against the actual dataset profile.

        This method must be called explicitly by the service.
        Pydantic cannot automatically validate an external dataset.

        Target-class counts, missing labels, chronological validity
        and minimum split sizes require separate data-level checks.
        """

        known_columns = {
            column.name
            for column in profile.columns
        }

        referenced_columns = {
            self.target_column,
            *(
                exclusion.column
                for exclusion in self.feature_exclusions
            ),
        }

        if self.split.group_column is not None:
            referenced_columns.add(
                self.split.group_column
            )

        if self.split.time_column is not None:
            referenced_columns.add(
                self.split.time_column
            )

        if not referenced_columns.issubset(known_columns):
            raise ValueError(
                "The plan references columns absent "
                "from the dataset profile."
            )

        excluded_columns = {
            item.column
            for item in self.feature_exclusions
        }

        usable_features = (
            known_columns
            - {self.target_column}
            - excluded_columns
        )

        if not usable_features:
            raise ValueError(
                "The plan leaves no candidate input features."
            )


# ============================================================
# PLANNER AGENT OUTPUT
# ============================================================

class PlannerDecision(PlanningSchema):
    """
    LLM-facing output contract.

    READY:
        plan is present; clarification_questions is empty.

    NEEDS_CLARIFICATION:
        plan is null; clarification_questions is non-empty.

    UNSUPPORTED:
        plan is null; clarification_questions is empty;
        explanation describes the unsupported requirement.
    """

    status: PlanningStatus

    explanation: str = Field(
        min_length=1,
        max_length=3000,
    )

    plan: ExecutionPlan | None

    clarification_questions: list[str]

    @field_validator("explanation")
    @classmethod
    def reject_blank_explanation(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Explanation must not be blank."
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

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.status == PlanningStatus.READY:
            if self.plan is None:
                raise ValueError(
                    "A ready decision requires a plan."
                )

            if self.clarification_questions:
                raise ValueError(
                    "A ready decision cannot contain "
                    "unresolved clarification questions."
                )

        elif self.status == PlanningStatus.NEEDS_CLARIFICATION:
            if self.plan is not None:
                raise ValueError(
                    "Do not emit an executable plan "
                    "while clarification is pending."
                )

            if not self.clarification_questions:
                raise ValueError(
                    "Provide at least one clarification question."
                )

        elif self.status == PlanningStatus.UNSUPPORTED:
            if self.plan is not None:
                raise ValueError(
                    "Unsupported requests must not contain a plan."
                )

            if self.clarification_questions:
                raise ValueError(
                    "Use needs_clarification when user answers "
                    "could resolve the request."
                )

        return self