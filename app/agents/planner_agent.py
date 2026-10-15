
"""AutoAudit Planner Agent.

Legacy plan() returns the existing validated classification PlannerDecision.
New plan_project() returns an end-to-end project file blueprint.
No file creation, cleaning, training, or packaging happens in this agent.
"""

import asyncio
import json
import math
import re

from collections.abc import Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.llm.provider_factory import LLMProvider
from app.schemas.dataset import DatasetProfile

from app.schemas.planning import (
    CandidateModel,
    ClassificationTask,
    PlannerDecision,
    ExecutionPlan,
    PlanningStatus,
    SplitStrategy,
)

from app.schemas.run import RunCreateRequest

from app.schemas.project import (
    DatasetStatus,
    MLTask,
    ProjectBlueprint,
    ProjectStage,
    FileKind,
    PlannedFile,
    InputMode,
    ProjectDatasetSpec,
    StreamlitSpec,
)

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



# AUTOAUDIT_BLUEPRINT_V2
# Small LLM contract. The full file/dependency graph is built and
# validated in application code, not generated as one huge LLM JSON.
class _BlueprintIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: MLTask
    project_title: str = Field(min_length=3, max_length=100)
    summary: str = Field(min_length=10, max_length=700)
    dataset_description: str = Field(min_length=5, max_length=500)
    prediction_explanation: str = Field(min_length=5, max_length=500)
    target_column: str | None = Field(default=None, max_length=200)
    input_mode: InputMode = InputMode.BOTH

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

        # Send structural facts rather than fitted statistics.

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

        return serialized

    # ========================================================
    # GENERATED EXECUTION PLAN VALIDATION
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

        # Complete split integrity is verified downstream.

    # ========================================================
    # EXISTING PLAN METHOD
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
        Generate the existing supported ML execution plan.

        The caller must supply an authorized, integrity-checked
        DataFrame and its matching profile.
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

        # AUTOAUDIT_PRESERVE_ORIGINAL_OBJECTIVE
        # Preserve the exact validated user objective.
        if (
            decision.status == PlanningStatus.READY
            and decision.plan is not None
            and decision.plan.objective != request.objective
        ):
            payload = decision.model_dump(mode="json")
            payload["plan"]["objective"] = request.objective
            decision = PlannerDecision.model_validate(payload)
            logger.info("Restored original Planner objective.")

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

    # ========================================================
    # NEW: FULL PROJECT BLUEPRINT
    # ========================================================

    def _validate_project_context(
        self,
        *,
        goal: str,
        profile: DatasetProfile | None,
        requested_target: str | None,
        user_answers: Sequence[str],
        language: str,
    ) -> None:

        if (
            not isinstance(goal, str)
            or not (5 <= len(goal.strip()) <= 5000)
        ):
            raise PlanningContextError(
                "Project goal must contain 5-5000 characters."
            )

        if len(goal) != len(goal.strip()):
            raise PlanningContextError(
                "Provide the already-trimmed project goal."
            )

        if language not in {"en", "mr", "hi"}:
            raise PlanningContextError(
                "Unsupported response language."
            )

        if (
            profile is not None
            and not isinstance(profile, DatasetProfile)
        ):
            raise PlanningContextError(
                "profile must be a DatasetProfile or None."
            )

        if requested_target is not None:
            if (
                not isinstance(requested_target, str)
                or not requested_target.strip()
            ):
                raise PlanningContextError(
                    "requested_target must be a nonblank column."
                )

            if profile is not None and requested_target not in {
                column.name
                for column in profile.columns
            }:
                raise PlanningContextError(
                    "Requested target is not in the dataset."
                )

        if (
            isinstance(user_answers, (str, bytes))
            or not isinstance(user_answers, Sequence)
        ):
            raise PlanningContextError(
                "user_answers must be a sequence."
            )

        if len(user_answers) > MAX_USER_ANSWERS:
            raise PlanningContextError(
                "Too many clarification answers."
            )

        if any(
            not isinstance(answer, str)
            or not answer.strip()
            or len(answer) > MAX_ANSWER_CHARACTERS
            for answer in user_answers
        ):
            raise PlanningContextError(
                "Invalid clarification answers."
            )

    # ========================================================
    # NEW: BUILD PROJECT PLANNING CONTEXT
    # ========================================================

    def _build_project_context(
        self,
        *,
        goal: str,
        profile: DatasetProfile | None,
        requested_target: str | None,
        user_answers: Sequence[str],
        language: str,
    ) -> str:
        """
        Build a bounded project-planning context.

        Only trusted dataset metadata is disclosed.
        No raw dataset rows or individual feature values.
        """

        compact_profile = None

        if profile is not None:
            compact_profile = {
                "row_count": profile.row_count,
                "column_count": profile.column_count,

                "columns": [
                    {
                        "name": column.name,
                        "kind": column.kind.value,
                        "missing_count": column.missing_count,
                        "unique_count": column.unique_count,
                    }
                    for column in profile.columns
                ],

                "warnings": [
                    {
                        "code": warning.code.value,
                        "columns": warning.columns,
                    }
                    for warning in profile.warnings
                ],
            }

        context = {
            "mode": "project_blueprint",
            "user_objective": goal,
            "requested_target": requested_target,
            "response_language": language,
            "dataset_available": profile is not None,
            "dataset_profile": compact_profile,
            "user_answers": list(user_answers),

            "project_capabilities": {
                "blueprint_task_types": [
                    task.value for task in MLTask
                ],

                "currently_implemented_training_tasks": [
                    MLTask.BINARY_CLASSIFICATION.value,
                    MLTask.MULTICLASS_CLASSIFICATION.value,
                ],

                "target_app_framework": "streamlit",
                "human_approval_required": True,

                "first_three_agents": [
                    ProjectStage.PLANNER.value,
                    ProjectStage.DATA_ENGINEERING.value,
                    ProjectStage.MODEL_TRAINING.value,
                ],

                "final_packaging_after_all_agents": True,
            },
        }

        payload = json.dumps(
            context,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

        if len(payload) > MAX_CONTEXT_CHARACTERS:
            raise PlanningContextError(
                "Project planning context is too large."
            )

        return payload

    # ========================================================
    # NEW: VALIDATE PROJECT BLUEPRINT
    # ========================================================

    def _validate_project_blueprint(
        self,
        *,
        blueprint: ProjectBlueprint,
        goal: str,
        profile: DatasetProfile | None,
        requested_target: str | None,
    ) -> None:
        """
        Check the LLM blueprint against trusted dataset facts
        and the ownership rules of the full team workflow.
        """

        if blueprint.goal != goal:
            raise PlanningValidationError(
                "The Planner changed the original goal."
            )

        if blueprint.revision != 1:
            raise PlanningValidationError(
                "A first project blueprint must be revision 1."
            )

        expected_availability = (
            DatasetStatus.AVAILABLE
            if profile is not None
            else DatasetStatus.REQUIRED
        )

        if blueprint.dataset.status != expected_availability:
            raise PlanningValidationError(
                "Dataset availability was misrepresented."
            )

        if (
            requested_target is not None
            and blueprint.dataset.target_column != requested_target
        ):
            raise PlanningValidationError(
                "The user-selected target was changed."
            )

        if profile is None:
            if (
                blueprint.dataset.target_column is not None
                and requested_target is None
            ):
                raise PlanningValidationError(
                    "Without a dataset, the Planner "
                    "cannot invent a target column."
                )

            if blueprint.dataset.expected_feature_columns:
                raise PlanningValidationError(
                    "Without a dataset profile, "
                    "feature column names are unverified."
                )

        else:
            known = {
                column.name
                for column in profile.columns
            }

            if (
                blueprint.dataset.target_column is not None
                and blueprint.dataset.target_column not in known
            ):
                raise PlanningValidationError(
                    "Blueprint target not in dataset."
                )

            if not set(
                blueprint.dataset.expected_feature_columns
            ).issubset(known):
                raise PlanningValidationError(
                    "Blueprint references unknown columns."
                )

        files = {
            entry.path.casefold(): entry
            for entry in blueprint.files
        }

        if (
            files["app.py"].producer
            != ProjectStage.MODEL_TRAINING
        ):
            raise PlanningValidationError(
                "The Model Training stage must "
                "generate the Streamlit app.py."
            )

        for filename in (
            "requirements.txt",
            "README.md",
        ):
            if (
                files[filename.casefold()].producer
                != ProjectStage.DOCUMENTATION
            ):
                raise PlanningValidationError(
                    f"Documentation stage must produce {filename}."
                )

        if not any(
            entry.path.endswith(".py")
            and entry.producer == ProjectStage.DATA_ENGINEERING
            for entry in blueprint.files
        ):
            raise PlanningValidationError(
                "At least one Data Engineering "
                "Python source file is required."
            )

        if not any(
            entry.path.endswith(".py")
            and entry.producer == ProjectStage.MODEL_TRAINING
            for entry in blueprint.files
        ):
            raise PlanningValidationError(
                "At least one Model Training "
                "Python source file is required."
            )

        # Regression can be described in the blueprint,
        # but is not yet implemented in the current executor.

        if (
            blueprint.task_type == MLTask.REGRESSION
            and not any(
                "regression" in assumption.casefold()
                and (
                    "not" in assumption.casefold()
                    or "requires" in assumption.casefold()
                )
                for assumption in blueprint.assumptions
            )
        ):
            raise PlanningValidationError(
                "A regression blueprint must disclose "
                "current training limitations."
            )

    # ========================================================
    # NEW: PUBLIC PROJECT PLANNING METHOD
    # ========================================================

    # AUTOAUDIT_BLUEPRINT_V2
    @staticmethod
    def _project_files() -> list[PlannedFile]:
        """Supported, versioned interfaces for a runnable Streamlit project.

        The Planner decides which template is appropriate; later agents
        create the files, run tests and package only verified outputs.
        """
        specs = [
            ("preprocessing.py", "python", "data_engineering",
             "Implement approved feature casting, imputation, scaling and encoding; fit on train partition only.", []),
            ("reports/execution_plan.json", "report", "data_engineering",
             "Preserve the validated ML execution plan exactly.", []),
            ("reports/cleaning_proposal.json", "report", "data_engineering",
             "Preserve the human-approved feature cleaning policies.", []),
            ("data/sample_input.csv", "data", "data_engineering",
             "A header-only prediction-input template using approved features; never fabricated training data.", []),
            ("models/model.joblib", "model", "model_training",
             "Copy the real trained model artifact, including its fitted preprocessing pipeline.", []),
            ("reports/results.json", "report", "model_training",
             "Real training and validation metrics; not final test results.", []),
            ("reports/splits.json", "report", "model_training",
             "Preserve row partition identifiers without re-fitting on test data.", []),
            ("reports/environment.json", "report", "model_training",
             "Record exact training runtime and package versions.", []),
            ("reports/approval.json", "report", "model_training",
             "Record revision-matched human cleaning approval.", []),
            ("reports/input_schema.json", "report", "model_training",
             "Define the prediction input columns and target without leakage.", []),
            ("predict.py", "python", "model_training",
             "Load the verified fitted model and safely predict from approved input schema without training at request time.",
             ["preprocessing.py", "models/model.joblib", "reports/input_schema.json"]),
            ("app.py", "python", "model_training",
             "Streamlit inference interface using predict.py for validated manual and CSV inputs.",
             ["predict.py", "reports/input_schema.json"]),
            ("reports/leakage_checks.json", "report", "leakage_refactoring",
             "Record leakage findings and approved repairs before tests.",
             ["reports/execution_plan.json", "reports/cleaning_proposal.json", "models/model.joblib"]),
            ("tests/test_prediction.py", "test", "testing",
             "Exercise loading and inference inputs, missing columns, and representative prediction behavior.",
             ["app.py", "predict.py", "reports/input_schema.json"]),
            ("reports/test_results.json", "report", "testing",
             "Store actual test execution outcomes; do not invent passing checks.",
             ["tests/test_prediction.py"]),
            ("reports/verification.json", "report", "verification",
             "Record runnable-project verification evidence and blocker resolution.",
             ["reports/test_results.json"]),
            ("requirements.txt", "documentation", "documentation",
             "List pinned, compatible runtime dependencies for the standalone generated app.",
             ["app.py"]),
            ("README.md", "documentation", "documentation",
             "Describe extraction, installation, launching Streamlit, inference examples and known limitations.",
             ["requirements.txt", "reports/verification.json"]),
            ("reports/package_validation.json", "report", "packaging",
             "Record packaging integrity and runnable ZIP verification before delivery.",
             ["README.md", "reports/verification.json"]),
        ]
        return [
            PlannedFile(path=path, kind=FileKind(kind), producer=ProjectStage(stage),
                        purpose=purpose, depends_on=dependencies, required=True)
            for path, kind, stage, purpose, dependencies in specs
        ]

    @staticmethod
    def _safe_project_slug(value: str) -> str:
        name = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:55]
        if not name or not name[0].isalpha():
            name = "autoaudit_" + name
        return (name[:64] if len(name) >= 3 else "autoaudit_project")

    async def plan_project(
        self, *, goal: str, profile: DatasetProfile | None = None,
        requested_target: str | None = None,
        user_answers: Sequence[str] = (), language: str | None = None,
        execution_plan: ExecutionPlan | None = None,
    ) -> ProjectBlueprint:
        """Return a schema-validated project-wide file plan, no generated code.

        Dataset-backed LangGraph planning reuses the already approved/validated
        ExecutionPlan and never makes a second large Ollama blueprint call.
        Goal-only planning asks Ollama for a small intent schema instead.
        """
        chosen_language = language or self._settings.default_language
        self._validate_project_context(
            goal=goal, profile=profile, requested_target=requested_target,
            user_answers=user_answers, language=chosen_language,
        )

        if execution_plan is not None:
            if profile is None:
                raise PlanningContextError("An execution plan needs its dataset profile.")
            plan = ExecutionPlan.model_validate(execution_plan)
            plan.validate_against_profile(profile)
            if requested_target is not None and plan.target_column != requested_target:
                raise PlanningValidationError("Execution plan target differs from selected target.")
            target = plan.target_column
            task = MLTask(plan.task_type.value)
            excluded = {e.column for e in plan.feature_exclusions}
            feature_names = [c.name for c in profile.columns
                             if c.name != target and c.name not in excluded]
            title = f"{target} Predictor"[:120]
            summary = (
                f"Train and evaluate a {task.value} project predicting {target}; "
                f"use {plan.primary_metric.value if hasattr(plan.primary_metric, 'value') else plan.primary_metric} "
                "for validation, review preprocessing before training, and hold out test data for audit."
            )
            description = f"Uploaded CSV: {profile.row_count} rows and {profile.column_count} columns."
            explanation = "Predictions use approved input features and the trained pipeline; validation is not a final test or audit."
            mode = InputMode.BOTH
        else:
            # Keep the LLM request small: the prompt and schema together are
            # a fraction of the original complete ProjectBlueprint output.
            context = self._build_project_context(
                goal=goal, profile=profile, requested_target=requested_target,
                user_answers=user_answers, language=chosen_language,
            )
            prompt = (
                "You are the AutoAudit project intent Planner. "
                "Return only the compact structured intent, not files/code. "
                "Determine whether the user wants binary_classification, "
                "multiclass_classification or regression. "
                "If a dataset is absent, target_column MUST be null unless "
                "requested_target is explicitly provided. "
                "If a dataset is present, use exact existing column spelling. "
                "A dataset is required to train, but is optional to plan. "
                "Never invent columns, metrics, model scores or project files."
            )
            intent = await self._provider.generate_structured(
                system_prompt=prompt, user_prompt=context,
                schema=_BlueprintIntent,
            )
            if not isinstance(intent, _BlueprintIntent):
                intent = _BlueprintIntent.model_validate(intent)
            task = intent.task_type
            title = intent.project_title
            summary = intent.summary
            description = intent.dataset_description
            explanation = intent.prediction_explanation
            mode = intent.input_mode
            if requested_target is not None:
                target = requested_target
            elif profile is not None and intent.target_column in {
                c.name for c in profile.columns
            }:
                target = intent.target_column
            else:
                target = None
            # These are hypotheses until dataset-backed ML Planner performs
            # explicit feature exclusion and leakage validation.
            feature_names = []

        assumptions = [
            "All generated source code and trained artifacts require testing, verification and packaging before delivery.",
            "Final test split is reserved for independent audit, not selection.",
        ]
        if task == MLTask.REGRESSION:
            assumptions.append(
                "Regression training requires implementation; the current executor does not support regression."
            )
        if profile is None:
            assumptions.append("Dataset must be uploaded and profiled before preprocessing or model training.")

        blueprint = ProjectBlueprint(
            project_slug=self._safe_project_slug(title),
            goal=goal,
            summary=summary,
            task_type=task,
            dataset=ProjectDatasetSpec(
                status=(DatasetStatus.AVAILABLE if profile is not None
                        else DatasetStatus.REQUIRED),
                description=description,
                target_column=target,
                expected_feature_columns=feature_names,
                source_notes=(
                    "Use verified uploaded dataset metadata; final features come from the validated ExecutionPlan and approved CleaningProposal."
                    if profile is not None else
                    "No dataset yet. Confirm columns and target after upload; do not fabricate training data."
                ),
            ),
            streamlit=StreamlitSpec(
                title=title,
                description=goal[:1500],
                input_mode=mode,
                show_metrics=True,
                show_probabilities=False,
                prediction_explanation=explanation,
            ),
            files=self._project_files(),
            assumptions=assumptions,
            cleaning_approval_required=True,
        )
        self._validate_project_blueprint(
            blueprint=blueprint, goal=goal, profile=profile,
            requested_target=requested_target,
        )
        logger.info("Planner blueprint ready: task=%s, files=%d.",
                    task.value, len(blueprint.files))
        return blueprint
