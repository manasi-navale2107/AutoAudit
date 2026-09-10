from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Protocol, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.ml.profiling import profile_dataset
from app.schemas.cleaning import (
    CleaningProposal,
    DatasetProfile,
    ExecutionPlan,
)


OutputModel = TypeVar("OutputModel", bound=BaseModel)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "data_engineering.txt"
)


class StructuredProvider(Protocol):
    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[OutputModel],
    ) -> OutputModel:
        ...


class CleaningProposalError(ValueError):
    """The proposal cannot safely proceed to human review."""


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def validate_proposal_against_frame(
    *,
    proposal: CleaningProposal,
    execution_plan: ExecutionPlan,
    profile: DatasetProfile,
    frame: pd.DataFrame,
    max_categories: int = 100,
) -> None:
    """
    Validate the proposal without modifying data or fitting transformers.

    Call this again before eventual execution. Training-partition checks
    are still required after splitting.
    """
    if max_categories < 2:
        raise ValueError("max_categories must be at least 2.")

    proposal.validate_against_context(
        execution_plan=execution_plan,
        profile=profile,
    )

    expected = {
        column
        for column in frame.columns
        if column != execution_plan.target_column
    }

    proposed = {policy.column for policy in proposal.policies}

    if proposed != expected:
        raise CleaningProposalError(
            "Policies must cover every non-target column exactly once."
        )

    for policy in proposal.policies:
        if _enum_value(policy.action) == "exclude":
            continue

        series = frame[policy.column]
        kind = _enum_value(policy.feature_kind)
        missing_strategy = _enum_value(policy.missing_strategy)

        if series.isna().all():
            raise CleaningProposalError(
                f"Column {policy.column!r} is entirely missing "
                "and must be excluded."
            )

        if series.isna().any() and missing_strategy == "none":
            raise CleaningProposalError(
                f"Column {policy.column!r} contains missing values "
                "but has no imputation policy."
            )

        if kind == "numeric":
            converted = pd.to_numeric(series, errors="coerce")

            invalid = series.notna() & converted.isna()

            if invalid.any():
                raise CleaningProposalError(
                    f"Column {policy.column!r} contains values that "
                    "cannot be converted to numbers."
                )

            values = converted.to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )

            if np.isinf(values).any():
                raise CleaningProposalError(
                    f"Column {policy.column!r} contains infinity. "
                    "An explicit supported repair is required."
                )

        elif kind in {"categorical", "boolean", "text"}:
            # Match the string representation used for category-like data.
            categories = series.dropna().astype("string")

            if categories.nunique() > max_categories:
                raise CleaningProposalError(
                    f"Column {policy.column!r} exceeds the "
                    f"{max_categories}-category limit."
                )

            if (
                missing_strategy == "constant"
                and categories.eq(policy.fill_value).any()
            ):
                raise CleaningProposalError(
                    f"The missing-value label for {policy.column!r} "
                    "already exists as an observed category."
                )

        else:
            raise CleaningProposalError(
                f"Retaining feature kind {kind!r} is not supported."
            )


class DataEngineeringAgent:
    """
    Produce a validated cleaning proposal for human review.

    No dataset mutation, fitting, approval, or training takes place here.
    """

    def __init__(
        self,
        *,
        provider: StructuredProvider,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        max_categories: int = 100,
        max_context_characters: int = 45_000,
    ) -> None:
        if max_categories < 2:
            raise ValueError("max_categories must be at least 2.")

        if max_context_characters < 1_000:
            raise ValueError(
                "max_context_characters must be at least 1000."
            )

        path = Path(prompt_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Data Engineering prompt file not found: {path}"
            )

        prompt = path.read_text(encoding="utf-8-sig").strip()

        if not prompt:
            raise ValueError("Data Engineering prompt is empty.")

        self._provider = provider
        self._system_prompt = prompt
        self._max_categories = max_categories
        self._max_context_characters = max_context_characters

    async def propose(
        self,
        *,
        execution_plan: ExecutionPlan,
        profile: DatasetProfile,
        frame: pd.DataFrame,
        language: Literal["en", "mr", "hi"] = "mr",
        previous_proposal: CleaningProposal | None = None,
        user_feedback: str | None = None,
    ) -> CleaningProposal:
        if language not in {"en", "mr", "hi"}:
            raise ValueError("Language must be en, mr, or hi.")

        if frame.empty:
            raise ValueError("The dataset is empty.")

        if not frame.columns.is_unique:
            raise ValueError("Dataset column names must be unique.")

        if any(
            not isinstance(column, str) or not column.strip()
            for column in frame.columns
        ):
            raise ValueError(
                "Dataset columns must have nonblank string names."
            )

        if execution_plan.target_column not in frame.columns:
            raise ValueError("The target column is missing.")

        if frame[execution_plan.target_column].isna().any():
            raise ValueError(
                "Missing target labels require review before cleaning."
            )

        feedback = (
            user_feedback.strip()
            if user_feedback is not None
            else None
        )

        if user_feedback is not None and not feedback:
            raise ValueError("Feedback must not be blank.")

        if feedback is not None and len(feedback) > 4_000:
            raise ValueError("Feedback must not exceed 4000 characters.")

        if (previous_proposal is None) != (feedback is None):
            raise ValueError(
                "A revision requires both previous_proposal "
                "and user_feedback."
            )

        # Recompute observations from the supplied frame so that stale
        # missing-value counts do not control the new proposal.
        fresh_profile = profile_dataset(
            frame=frame,
            dataset_id=profile.dataset_id,
        )

        execution_plan.validate_against_profile(fresh_profile)

        if previous_proposal is not None:
            # Data-level checks are applied to the revised output.
            # The previous proposal is supplied only as revision context.
            previous_proposal.validate_against_context(
                execution_plan=execution_plan,
                profile=fresh_profile,
            )

        context = {
            "execution_plan": execution_plan.model_dump(mode="json"),
            "dataset_profile": self._bounded_profile(
                frame=frame,
                profile=fresh_profile,
            ),
            "response_language": language,
            "previous_proposal": (
                previous_proposal.model_dump(mode="json")
                if previous_proposal is not None
                else None
            ),
            "user_feedback": feedback,
            "preprocessing_constraints": {
                "max_categories_per_feature": self._max_categories,
                "retained_feature_kinds": [
                    "numeric",
                    "categorical",
                    "boolean",
                    "text",
                ],
                "text_treatment": "category-like values only",
                "numeric_constant_imputation_supported": False,
                "row_deletion_supported": False,
                "outlier_removal_supported": False,
                "infinity_repair_supported": False,
                "datetime_feature_extraction_supported": False,
                "fit_preprocessing_on": "training_partition_only",
                "human_approval_required": True,
                "revised_proposal_requires_new_approval": True,
            },
        }

        user_prompt = json.dumps(
            context,
            ensure_ascii=False,
            allow_nan=False,
        )

        if len(user_prompt) > self._max_context_characters:
            raise CleaningProposalError(
                "Cleaning context exceeds the configured size limit. "
                "Reduce the number of columns or shorten feedback."
            )

        logger.info(
            "Cleaning proposal requested; revision=%s.",
            previous_proposal is not None,
        )

        # Provider already owns transport retries and schema validation.
        # Do not add another retry loop here.
        response = await self._provider.generate_structured(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            schema=CleaningProposal,
        )

        if not isinstance(response, CleaningProposal):
            raise TypeError(
                "Provider must return a CleaningProposal instance."
            )

        # Revalidate even if a provider returns an existing model instance.
        proposal = CleaningProposal.model_validate_json(
            response.model_dump_json()
        )

        validate_proposal_against_frame(
            proposal=proposal,
            execution_plan=execution_plan,
            profile=fresh_profile,
            frame=frame,
            max_categories=self._max_categories,
        )

        logger.info(
            "Cleaning proposal validated; policies=%d; questions=%d.",
            len(proposal.policies),
            len(proposal.unresolved_questions),
        )

        # This return value is a proposal, never an approval.
        return proposal

    @staticmethod
    def _bounded_profile(
        *,
        frame: pd.DataFrame,
        profile: DatasetProfile,
    ) -> dict:
        """
        Send descriptive counts and inferred types, not raw rows,
        sample values, category lists, or fitted statistics.
        """
        observations = {
            column.name: column.model_dump(mode="json")
            for column in profile.columns
        }

        columns = []

        for name in frame.columns:
            series = frame[name]

            item = {
                "name": name,
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "all_missing": bool(series.isna().all()),
                "storage_dtype": str(series.dtype),
            }

            # Preserve whichever type field the existing profile defines.
            observed = observations[name]

            for key in (
                "kind",
                "inferred_kind",
                "inferred_type",
                "column_kind",
            ):
                if key in observed:
                    item[key] = observed[key]

            converted = pd.to_numeric(series, errors="coerce")

            item["numeric_conversion_failure_count"] = int(
                (series.notna() & converted.isna()).sum()
            )

            values = converted.to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )

            item["infinity_count"] = int(np.isinf(values).sum())

            columns.append(item)

        return {
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "duplicate_row_count": int(frame.duplicated().sum()),
            "columns": columns,
        }