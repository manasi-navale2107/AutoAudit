from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal
from uuid import UUID

import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.agents.data_engineering_agent import StructuredProvider

from app.ml.model_factory import (
    ModelSettings,
    build_candidate_models,
)

from app.schemas.cleaning import ExecutionPlan

from app.services.cleaning_review_store import CleaningReviewStore


logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "model_training.txt"
)


class TrainingDecision(BaseModel):
    """Proposed model settings, not execution results."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    settings: ModelSettings

    rationale: str = Field(
        min_length=1,
        max_length=3000,
    )

    @field_validator("rationale")
    @classmethod
    def reject_blank_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Training rationale must not be blank."
            )

        return value


def execution_plan_sha256(
    execution_plan: ExecutionPlan,
) -> str:
    """
    Compute a stable fingerprint of an execution plan.

    Use this same function when submitting a proposal for review.
    """
    serialized = json.dumps(
        execution_plan.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


class ModelTrainingAgent:
    """
    Propose bounded settings after checking stored human approval.

    The caller must supply the verified dataset's training partition.

    This agent does not split data, fit models, or score validation/test.
    """

    def __init__(
        self,
        *,
        provider: StructuredProvider,
        review_store: CleaningReviewStore,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    ) -> None:
        path = Path(prompt_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Training prompt not found: {path}"
            )

        prompt = path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not prompt:
            raise ValueError(
                "Model Training prompt is empty."
            )

        self._provider = provider
        self._reviews = review_store
        self._system_prompt = prompt

    async def propose(
        self,
        *,
        run_id: UUID,
        revision: int,
        dataset_sha256: str,
        execution_plan: ExecutionPlan,
        training_frame: pd.DataFrame,
        language: Literal["en", "mr", "hi"] = "mr",
    ) -> TrainingDecision:
        if language not in {"en", "mr", "hi"}:
            raise ValueError(
                "Language must be en, mr, or hi."
            )

        # Take a validated snapshot rather than retaining
        # a reference to a mutable execution plan.
        plan = ExecutionPlan.model_validate_json(
            execution_plan.model_dump_json()
        )

        plan_hash = execution_plan_sha256(plan)

        # Approval must match this run, revision, dataset and plan.
        approved = self._reviews.require_approved(
            run_id=run_id,
            revision=revision,
            dataset_sha256=dataset_sha256,
            plan_sha256=plan_hash,
        )

        if not isinstance(training_frame, pd.DataFrame):
            raise TypeError(
                "training_frame must be a pandas DataFrame."
            )

        if (
            training_frame.empty
            or not training_frame.columns.is_unique
        ):
            raise ValueError(
                "Training needs rows and unique column names."
            )

        target = plan.target_column

        if target not in training_frame.columns:
            raise ValueError(
                "Training target column is missing."
            )

        labels = training_frame[target]

        if labels.isna().any() or labels.nunique() < 2:
            raise ValueError(
                "Training requires nonmissing labels "
                "and at least two classes."
            )

        policies = approved.model_dump(
            mode="json"
        )["policies"]

        retained = [
            policy["column"]
            for policy in policies
            if policy["action"] == "keep"
        ]

        if target in retained or not retained:
            raise ValueError(
                "Approved input features are invalid."
            )

        if any(
            name not in training_frame.columns
            for name in retained
        ):
            raise ValueError(
                "An approved feature is missing from training."
            )

        # Reject unsupported candidates before calling the LLM.
        # Constructing these objects does not fit them.
        build_candidate_models(
            candidate_models=plan.candidate_models,
        )

        context = {
            "execution_plan": plan.model_dump(
                mode="json"
            ),
            "approved_cleaning_proposal": approved.model_dump(
                mode="json"
            ),
            "approval_context": {
                "status": "approved",
                "revision": revision,
                "execution_plan_sha256": plan_hash,
                "dataset_sha256": dataset_sha256,
            },
            "training_profile": {
                "row_count": len(training_frame),
                "retained_feature_count": len(retained),
                "class_count": int(labels.nunique()),
            },
            "training_constraints": {
                "settings_schema": (
                    ModelSettings.model_json_schema()
                ),
                "default_settings": (
                    ModelSettings().model_dump(mode="json")
                ),
                "hyperparameter_search_supported": False,
                "random_seed_and_parallelism": (
                    "controlled_by_backend"
                ),
                "preprocessing_changes_allowed": False,
                "final_test_evaluation_allowed": False,
            },
            "previous_training_results": None,
            "response_language": language,
        }

        payload = json.dumps(
            context,
            ensure_ascii=False,
            allow_nan=False,
        )

        if len(payload) > 45_000:
            raise ValueError(
                "Training context exceeds the supported size."
            )

        logger.info(
            "Training-settings proposal requested."
        )

        # Existing provider handles transport retries and
        # structured-output generation.
        response = await self._provider.generate_structured(
            system_prompt=self._system_prompt,
            user_prompt=payload,
            schema=TrainingDecision,
        )

        if not isinstance(response, TrainingDecision):
            raise TypeError(
                "Provider must return a TrainingDecision."
            )

        decision = TrainingDecision.model_validate_json(
            response.model_dump_json()
        )

        # Validate the returned settings against our actual
        # supported model constructors.
        build_candidate_models(
            candidate_models=plan.candidate_models,
            settings=decision.settings,
        )

        # Recheck approval after the potentially slow provider call.
        current = self._reviews.require_approved(
            run_id=run_id,
            revision=revision,
            dataset_sha256=dataset_sha256,
            plan_sha256=plan_hash,
        )

        if (
            current.model_dump(mode="json")
            != approved.model_dump(mode="json")
        ):
            raise ValueError(
                "Approved proposal changed during this request."
            )

        logger.info(
            "Training-settings proposal validated."
        )

        return decision