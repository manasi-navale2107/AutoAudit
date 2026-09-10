from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelSettings(BaseModel):
    """
    Supported, bounded settings for the initial training implementation.

    The Model Training Agent may propose these settings.
    Validation does not mean the settings are optimal.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    logistic_c: Annotated[
        float,
        Field(ge=0.01, le=10.0),
    ] = 1.0

    logistic_max_iter: Annotated[
        int,
        Field(ge=100, le=3000),
    ] = 1500

    rf_n_estimators: Annotated[
        int,
        Field(ge=50, le=300),
    ] = 100

    rf_max_depth: Annotated[
        int,
        Field(ge=2, le=20),
    ] = 10

    rf_min_samples_split: Annotated[
        int,
        Field(ge=2, le=20),
    ] = 2

    rf_min_samples_leaf: Annotated[
        int,
        Field(ge=1, le=10),
    ] = 1


class TrainingRuntimeSettings(BaseModel):
    """
    Backend-controlled settings.

    These values must not be selected by the LLM.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    random_seed: Annotated[
        int,
        Field(ge=0, le=2**32 - 1),
    ] = 42

    n_jobs: Annotated[
        int,
        Field(ge=1, le=4),
    ] = 2


SUPPORTED_MODELS = frozenset(
    {
        "logistic_regression",
        "random_forest",
    }
)


def build_candidate_models(
    *,
    candidate_models: Sequence[str | Enum],
    settings: ModelSettings | None = None,
    runtime: TrainingRuntimeSettings | None = None,
) -> dict[str, BaseEstimator]:
    """
    Construct fresh, unfitted models in the Planner's requested order.

    This function does not:
    - Fit preprocessing or models.
    - Change the candidate-model list.
    - Select a winning model.
    - Perform hyperparameter search.
    """
    if isinstance(candidate_models, (str, bytes)):
        raise TypeError(
            "candidate_models must be a sequence, not a single string."
        )

    names: list[str] = []

    for candidate in candidate_models:
        name = candidate.value if isinstance(candidate, Enum) else candidate

        if not isinstance(name, str):
            raise TypeError("Every model identifier must be a string.")

        if name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model identifier: {name!r}. "
                f"Supported models: {sorted(SUPPORTED_MODELS)}"
            )

        names.append(name)

    if not names:
        raise ValueError("At least one candidate model is required.")

    if len(names) != len(set(names)):
        raise ValueError("Candidate models must not contain duplicates.")

    chosen_settings = settings if settings is not None else ModelSettings()
    chosen_runtime = (
        runtime if runtime is not None else TrainingRuntimeSettings()
    )

    # Revalidate to guard against instances created using model_construct()
    # or model_copy(update=...), which can bypass normal validation.
    chosen_settings = ModelSettings.model_validate(
        chosen_settings.model_dump()
    )

    chosen_runtime = TrainingRuntimeSettings.model_validate(
        chosen_runtime.model_dump()
    )

    models: dict[str, BaseEstimator] = {}

    for name in names:
        if name == "logistic_regression":
            models[name] = LogisticRegression(
                C=chosen_settings.logistic_c,
                max_iter=chosen_settings.logistic_max_iter,
                solver="lbfgs",
                random_state=chosen_runtime.random_seed,
            )

        elif name == "random_forest":
            models[name] = RandomForestClassifier(
                n_estimators=chosen_settings.rf_n_estimators,
                max_depth=chosen_settings.rf_max_depth,
                min_samples_split=(
                    chosen_settings.rf_min_samples_split
                ),
                min_samples_leaf=(
                    chosen_settings.rf_min_samples_leaf
                ),
                random_state=chosen_runtime.random_seed,
                n_jobs=chosen_runtime.n_jobs,
            )

    return models