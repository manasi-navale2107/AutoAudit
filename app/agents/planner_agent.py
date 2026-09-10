"""
Planner Agent for AutoAudit.

Responsibilities:
- Prepare bounded planning context.
- Invoke the configured LLM using PlannerDecision.
- Validate the returned plan against actual dataset information.
- Return clarification when the target is missing or ambiguous.

The full DataFrame is used locally for consistency checks.
Raw dataset rows are not included in the LLM request.

The agent does not perform cleaning, splitting, training,
human approval, or pipeline auditing.
"""

import asyncio
import json
import math

from collections.abc import Sequence

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger

from app.llm.provider_factory import LLMProvider

from app.schemas.dataset import DatasetProfile

from app.schemas.planning import (
    CandidateModel,
    ClassificationTask,
    PlannerDecision,
    PlanningStatus,
    SplitStrategy,
)

from app.schemas.run import RunCreateRequest


logger = get_logger(__name__)


# ============================================================
# CONTEXT LIMITS
# ============================================================

MAX_PROMPT_FILE_BYTES = 64_000
MAX_CONTEXT_CHARACTERS = 24_000
MAX_USER_ANSWERS = 10
MAX_ANSWER_CHARACTERS = 2000
MAX_DISCLOSED_CLASS_LABELS = 20
MAX_DISCLOSED_LABEL_CHARACTERS = 200


# ============================================================
# ERRORS
# ============================================================

class PlanningContextError(ValueError):
    """The supplied local planning context is inconsistent."""


class PlanningValidationError(ValueError):
    """A generated plan violates application-level rules."""


# ============================================================
# LOCALIZED CLARIFICATION TEXT
# ============================================================

TARGET_NOT_FOUND = {
    "en": (
        "The selected target does not match a dataset column.",
        "Which existing column should the model predict?",
    ),
    "mr": (
        "निवडलेला target datasetमधील columnशी जुळत नाही.",
        "Modelने datasetमधला नेमका कोणता column predict करायचा आहे?",
    ),
    "hi": (
        "चुना गया target dataset के किसी column से मेल नहीं खाता।",
        "Model को dataset का कौन-सा column predict करना है?",
    ),
}


# ============================================================
# PLANNER AGENT
# ============================================================

class PlannerAgent:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings or get_settings()

    # ========================================================
    # PROMPT LOADING
    # ========================================================

    def _read_system_prompt(self) -> str:
        path = self._settings.prompts_dir / "planner.txt"

        try:
            with path.open("rb") as stream:
                content = stream.read(MAX_PROMPT_FILE_BYTES + 1)

        except OSError:
            raise PlanningContextError(
                "The Planner prompt file could not be read."
            ) from None

        if len(content) > MAX_PROMPT_FILE_BYTES:
            raise PlanningContextError(
                "The Planner prompt file exceeds its size limit."
            )

        try:
            prompt = content.decode("utf-8-sig").strip()

        except UnicodeDecodeError:
            raise PlanningContextError(
                "The Planner prompt must use UTF-8 encoding."
            ) from None

        if not prompt:
            raise PlanningContextError(
                "The Planner prompt file is empty."
            )

        # Keep the stored objective identical to the request.
        # Explanations may use the selected response language.
        return (
            prompt
            + "\n\nApplication contract: copy user_objective exactly "
            "into plan.objective when returning a ready plan."
        )

    # ========================================================
    # LOCAL CONTEXT CHECKS
    # ========================================================

    def _validate_local_context(
        self,
        *,
        request: RunCreateRequest,
        profile: DatasetProfile,
        frame: pd.DataFrame,
        user_answers: Sequence[str],
    ) -> None:
        if request.dataset_id != profile.dataset_id:
            raise PlanningContextError(
                "Request and profile refer to different datasets."
            )

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "frame must be a pandas DataFrame."
            )

        if frame.shape != (
            profile.row_count,
            profile.column_count,
        ):
            raise PlanningContextError(
                "Dataset dimensions do not match the profile."
            )

        profile_columns = [
            column.name
            for column in profile.columns
        ]

        if list(frame.columns) != profile_columns:
            raise PlanningContextError(
                "Dataset columns do not match the profile."
            )

        if isinstance(user_answers, (str, bytes)):
            raise PlanningContextError(
                "user_answers must be a sequence of answer strings."
            )

        if len(user_answers) > MAX_USER_ANSWERS:
            raise PlanningContextError(
                "Too many clarification answers were supplied."
            )

        for answer in user_answers:
            if (
                not isinstance(answer, str)
                or not answer.strip()
                or len(answer) > MAX_ANSWER_CHARACTERS
            ):
                raise PlanningContextError(
                    "Clarification answers must be non-blank "
                    "strings within the configured size limit."
                )

        configuration = self._provider.configuration

        if (
            request.provider is not None
            and request.provider != configuration.provider
        ):
            raise PlanningContextError(
                "The request provider does not match "
                "the run's fixed provider."
            )

        if (
            request.model is not None
            and request.model != configuration.model
        ):
            raise PlanningContextError(
                "The request model does not match "
                "the run's fixed model."
            )

    # ========================================================
    # BOUNDED TARGET INFORMATION
    # ========================================================

    def _target_metadata(
        self,
        *,
        frame: pd.DataFrame,
        target: str | None,
    ) -> dict | None:
        """
        Include labels only for an explicitly requested target.

        No individual rows or other feature values are disclosed.
        Large label sets are summarized using counts only.
        """

        if target is None or target not in frame.columns:
            return None

        series = frame[target]
        observed = series.dropna()

        counts = observed.astype("string").value_counts(
            dropna=True
        )

        class_count = len(counts)

        labels_fit = (
            class_count <= MAX_DISCLOSED_CLASS_LABELS
            and all(
                len(str(label)) <= MAX_DISCLOSED_LABEL_CHARACTERS
                for label in counts.index
            )
        )

        label_counts = (
            [
                {
                    "label": str(label),
                    "count": int(count),
                }
                for label, count in counts.items()
            ]
            if labels_fit
            else []
        )

        return {
            "column": target,
            "missing_count": int(series.isna().sum()),
            "observed_unique_count": class_count,
            "label_counts": label_counts,
            "labels_omitted": not labels_fit,
            "interpretation_note": (
                "Observed values do not by themselves establish "
                "that the task is classification."
            ),
        }

    # ========================================================
    # LLM CONTEXT
    # ========================================================

    def _build_context(
        self,
        *,
        request: RunCreateRequest,
        profile: DatasetProfile,
        frame: pd.DataFrame,
        user_answers: Sequence[str],
    ) -> str:
        language = (
            request.language
            or self._settings.default_language
        )

        # Send structural facts, not global numeric statistics
        # that might be reused as fitted preprocessing values.
        compact_columns = [
            {
                "name": column.name,
                "inferred_kind": column.kind.value,
                "missing_count": column.missing_count,
                "unique_count": column.unique_count,
            }
            for column in profile.columns
        ]

        compact_warnings = [
            {
                "code": warning.code.value,
                "columns": warning.columns,
            }
            for warning in profile.warnings
        ]

        context = {
            "user_objective": request.objective,
            "requested_target": request.target_column,
            "response_language": language,
            "dataset_profile": {
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "duplicate_row_count": profile.duplicate_row_count,
                "columns": compact_columns,
                "warnings": compact_warnings,
            },
            "target_metadata": self._target_metadata(
                frame=frame,
                target=request.target_column,
            ),
            "planning_defaults": {
                "candidate_models": [
                    CandidateModel.LOGISTIC_REGRESSION.value,
                    CandidateModel.RANDOM_FOREST.value,
                ],
                "train_fraction": self._settings.training_size,
                "validation_fraction": self._settings.validation_size,
                "test_fraction": self._settings.test_size,
            },
            "user_answers": list(user_answers),
        }

        serialized = json.dumps(
            context,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

        if len(serialized) > MAX_CONTEXT_CHARACTERS:
            raise PlanningContextError(
                "The planning context exceeds the supported size. "
                "Reduce the context or add a bounded summarization step."
            )

        # This character bound limits request growth.
        # It is not an exact model-token count or fit guarantee.
        return serialized

    # ========================================================
    # GENERATED PLAN VALIDATION
    # ========================================================

    def _validate_decision(
        self,
        *,
        decision: PlannerDecision,
        request: RunCreateRequest,
        profile: DatasetProfile,
        frame: pd.DataFrame,
    ) -> None:
        if decision.status != PlanningStatus.READY:
            if len(decision.clarification_questions) > 3:
                raise PlanningValidationError(
                    "The Planner returned too many clarification questions."
                )

            return

        plan = decision.plan

        if plan is None:
            raise PlanningValidationError(
                "A ready decision must contain a plan."
            )

        try:
            plan.validate_against_profile(profile)

        except ValueError:
            raise PlanningValidationError(
                "The generated plan is inconsistent with the dataset."
            ) from None

        if plan.objective != request.objective:
            raise PlanningValidationError(
                "The Planner changed the original prediction objective."
            )

        if (
            request.target_column is not None
            and plan.target_column != request.target_column
        ):
            raise PlanningValidationError(
                "The Planner changed the explicitly selected target."
            )

        expected_fractions = (
            self._settings.training_size,
            self._settings.validation_size,
            self._settings.test_size,
        )

        actual_fractions = (
            plan.split.train_fraction,
            plan.split.validation_fraction,
            plan.split.test_fraction,
        )

        if not all(
            math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for actual, expected in zip(
                actual_fractions,
                expected_fractions,
            )
        ):
            raise PlanningValidationError(
                "The Planner changed the configured split fractions."
            )

        target = frame[plan.target_column]

        if target.isna().any():
            raise PlanningValidationError(
                "Missing target labels require clarification "
                "before a ready plan."
            )

        labels = target.astype("string")
        class_count = int(labels.nunique())

        if class_count < 2:
            raise PlanningValidationError(
                "Classification requires at least two observed classes."
            )

        expected_task = (
            ClassificationTask.BINARY
            if class_count == 2
            else ClassificationTask.MULTICLASS
        )

        if plan.task_type != expected_task:
            raise PlanningValidationError(
                "The task type does not match the observed class count."
            )

        if (
            plan.positive_class_label is not None
            and not bool(
                labels.eq(plan.positive_class_label).any()
            )
        ):
            raise PlanningValidationError(
                "The positive class is absent from the target."
            )

        if plan.split.strategy == SplitStrategy.STRATIFIED:
            if int(labels.value_counts().min()) < 3:
                raise PlanningValidationError(
                    "Every class needs at least three observations "
                    "to appear in train, validation and test."
                )

        elif plan.split.strategy == SplitStrategy.GROUP:
            groups = frame[plan.split.group_column]

            if groups.isna().any():
                raise PlanningValidationError(
                    "The selected group column contains missing values."
                )

            if groups.nunique() < 3:
                raise PlanningValidationError(
                    "Three non-empty group partitions require "
                    "at least three distinct groups."
                )

        elif plan.split.strategy == SplitStrategy.TEMPORAL:
            timestamps = frame[plan.split.time_column]

            if timestamps.isna().any():
                raise PlanningValidationError(
                    "The selected time column contains missing values."
                )

        # These are necessary checks, not complete split feasibility.
        # Actual partition sizes, chronology, class coverage and
        # group isolation are verified by the splitting module.

    # ========================================================
    # PUBLIC AGENT METHOD
    # ========================================================

    async def plan(
        self,
        *,
        request: RunCreateRequest,
        profile: DatasetProfile,
        frame: pd.DataFrame,
        user_answers: Sequence[str] = (),
    ) -> PlannerDecision:
        """
        Generate a plan using the run's configured provider.

        The caller supplies an owner-authorized, integrity-checked
        frame from DatasetService.

        Domain correctness and feature availability may require
        human clarification beyond what schemas can prove.
        """

        self._validate_local_context(
            request=request,
            profile=profile,
            frame=frame,
            user_answers=user_answers,
        )

        language = (
            request.language
            or self._settings.default_language
        )

        if (
            request.target_column is not None
            and request.target_column not in frame.columns
        ):
            explanation, question = TARGET_NOT_FOUND[language]

            return PlannerDecision(
                status=PlanningStatus.NEEDS_CLARIFICATION,
                explanation=explanation,
                plan=None,
                clarification_questions=[question],
            )

        system_prompt = await asyncio.to_thread(
            self._read_system_prompt
        )

        user_prompt = await asyncio.to_thread(
            self._build_context,
            request=request,
            profile=profile,
            frame=frame,
            user_answers=user_answers,
        )

        logger.info("Planner request started.")

        decision = await self._provider.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=PlannerDecision,
        )

        await asyncio.to_thread(
            self._validate_decision,
            decision=decision,
            request=request,
            profile=profile,
            frame=frame,
        )

        logger.info(
            "Planner decision validated; status=%s.",
            decision.status.value,
        )

        return decision