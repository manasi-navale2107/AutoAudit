from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from app.schemas.cleaning import CleaningProposal


class FeatureCaster(TransformerMixin, BaseEstimator):
    """
    Convert retained features without modifying the original DataFrame.

    policies must contain JSON-compatible policy dictionaries.
    """

    def __init__(
        self,
        policies: list[dict],
        max_categories: int = 100,
    ):
        # Keep constructor parameters unchanged for sklearn cloning.
        self.policies = policies
        self.max_categories = max_categories

    def _convert(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Preprocessing requires a pandas DataFrame.")

        if not frame.columns.is_unique:
            raise ValueError("Input column names must be unique.")

        result = pd.DataFrame(index=frame.index)

        for policy in self.policies:
            if policy["action"] != "keep":
                continue

            name = policy["column"]

            if name not in frame.columns:
                raise ValueError(f"Required feature is missing: {name!r}")

            series = frame[name]

            if policy["feature_kind"] == "numeric":
                converted = pd.to_numeric(series, errors="coerce")

                if (series.notna() & converted.isna()).any():
                    raise ValueError(
                        f"Feature {name!r} contains nonnumeric values."
                    )

                values = converted.to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )

                if np.isinf(values).any():
                    raise ValueError(
                        f"Feature {name!r} contains infinity."
                    )

                result[name] = values

            elif policy["feature_kind"] in {
                "categorical",
                "boolean",
                "text",
            }:
                # Use a consistent category representation.
                # Literal strings such as "NA" remain ordinary strings.
                values = series.astype("string").astype(object)

                result[name] = values.where(
                    series.notna(),
                    np.nan,
                )

            else:
                raise ValueError(
                    f"Unsupported retained feature kind for {name!r}."
                )

            if (
                policy["missing_strategy"] == "none"
                and result[name].isna().any()
            ):
                raise ValueError(
                    f"Feature {name!r} contains missing values "
                    "without an approved imputation policy."
                )

            if policy["missing_strategy"] == "constant":
                observed = result[name].dropna()

                if observed.eq(policy["fill_value"]).any():
                    raise ValueError(
                        f"Missing-category label collides with an "
                        f"observed value in {name!r}."
                    )

        if result.shape[1] == 0:
            raise ValueError("At least one feature must be retained.")

        return result

    def fit(self, X: pd.DataFrame, y=None):
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2.")

        converted = self._convert(X)

        if converted.empty:
            raise ValueError("Training data must not be empty.")

        for policy in self.policies:
            if policy["action"] != "keep":
                continue

            name = policy["column"]
            series = converted[name]

            if series.isna().all():
                raise ValueError(
                    f"Feature {name!r} is entirely missing in training. "
                    "A revised cleaning proposal is required."
                )

            if (
                policy["feature_kind"] != "numeric"
                and series.nunique(dropna=True) > self.max_categories
            ):
                raise ValueError(
                    f"Feature {name!r} exceeds the "
                    f"{self.max_categories}-category training limit."
                )

        self.retained_columns_ = tuple(converted.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "retained_columns_")

        converted = self._convert(X)

        if tuple(converted.columns) != self.retained_columns_:
            raise ValueError("Retained feature configuration changed.")

        return converted


def build_preprocessor(
    *,
    proposal: CleaningProposal,
    max_categories: int = 100,
) -> Pipeline:
    """
    Build an unfitted preprocessor from a resolved proposal.

    This function does not verify user approval or split the dataset.
    The training service must verify approval and supply training rows
    to fit(), then validation/inference rows to transform().
    """
    validated = CleaningProposal.model_validate_json(
        proposal.model_dump_json()
    )

    if validated.unresolved_questions:
        raise ValueError(
            "Resolve cleaning questions before building preprocessing."
        )

    if max_categories < 2:
        raise ValueError("max_categories must be at least 2.")

    policies = validated.model_dump(mode="json")["policies"]

    transformers = []

    for index, policy in enumerate(policies):
        if policy["action"] == "exclude":
            continue

        steps = []

        missing_strategy = policy["missing_strategy"]

        if missing_strategy != "none":
            imputer_options = {
                "strategy": missing_strategy,
                "keep_empty_features": True,
            }

            if missing_strategy == "constant":
                imputer_options["fill_value"] = policy["fill_value"]

            steps.append(
                ("imputer", SimpleImputer(**imputer_options))
            )

        if policy["feature_kind"] == "numeric":
            scaling = policy["scaling"]

            if scaling == "standard":
                steps.append(("scaler", StandardScaler()))

            elif scaling == "robust":
                steps.append(("scaler", RobustScaler()))

        else:
            steps.append(
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                        dtype=np.float64,
                    ),
                )
            )

        transformer = Pipeline(steps) if steps else "passthrough"

        transformers.append(
            (
                f"feature_{index}",
                transformer,
                [policy["column"]],
            )
        )

    if not transformers:
        raise ValueError("No retained features are available.")

    return Pipeline(
        [
            (
                "cast",
                FeatureCaster(
                    policies=policies,
                    max_categories=max_categories,
                ),
            ),
            (
                "columns",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    sparse_threshold=1.0,
                ),
            ),
        ]
    )