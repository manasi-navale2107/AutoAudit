from __future__ import annotations

import csv
import hashlib
import io
import logging
import math
import platform
import time
import warnings

from dataclasses import dataclass
from importlib.metadata import version
from uuid import UUID

import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from threadpoolctl import threadpool_limits

from app.agents.data_engineering_agent import (
    validate_proposal_against_frame,
)
from app.agents.model_training_agent import (
    TrainingDecision,
    execution_plan_sha256,
)
from app.ml.model_factory import (
    TrainingRuntimeSettings,
    build_candidate_models,
)
from app.ml.preprocessing import build_preprocessor
from app.ml.profiling import profile_dataset
from app.ml.splitting import DatasetSplit, split_dataset
from app.schemas.cleaning import ExecutionPlan
from app.services.cleaning_review_store import CleaningReviewStore


logger = logging.getLogger(__name__)


def load_training_frame(
    dataset_bytes: bytes,
) -> pd.DataFrame:
    """
    Parse the exact uploaded CSV bytes.

    Literal strings such as NA and NULL remain ordinary values.
    Empty fields are treated as missing.
    """
    if not isinstance(dataset_bytes, bytes):
        raise TypeError("dataset_bytes must be bytes.")

    if not 0 < len(dataset_bytes) <= 25 * 1024 * 1024:
        raise ValueError(
            "CSV must be nonempty and at most 25 MB."
        )

    try:
        text = dataset_bytes.decode("utf-8-sig")

        if "\x00" in text:
            raise ValueError(
                "CSV contains a null character."
            )

        reader = csv.reader(
            io.StringIO(text, newline=""),
            strict=True,
        )

        header = next(reader)

        if (
            not 2 <= len(header) <= 100
            or len(header) != len(set(header))
            or any(
                not name.strip() or len(name) > 200
                for name in header
            )
        ):
            raise ValueError(
                "CSV needs 2–100 unique, nonblank column names."
            )

        count = 0

        for record in reader:
            count += 1

            if len(record) != len(header):
                raise ValueError(
                    "CSV records must match the header width."
                )

            if count > 100_000:
                raise ValueError(
                    "CSV exceeds the 100000-row limit."
                )

        if count < 10:
            raise ValueError(
                "Provide at least 10 data rows."
            )

    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise ValueError(
            "Provide a valid UTF-8 CSV."
        ) from exc

    frame = pd.read_csv(
        io.StringIO(text),
        dtype="string",
        keep_default_na=False,
        na_values=[""],
        skip_blank_lines=False,
    )

    if (
        frame.shape != (count, len(header))
        or list(frame.columns) != header
    ):
        raise ValueError(
            "CSV parsing did not preserve its record structure."
        )

    return frame


@dataclass(frozen=True)
class TrainingResult:
    """
    Keep the fitted pipeline separate from JSON-compatible results.
    """

    pipeline: Pipeline
    split: DatasetSplit
    report: dict


def _metric_name(plan: ExecutionPlan) -> str:
    value = str(
        getattr(
            plan.primary_metric,
            "value",
            plan.primary_metric,
        )
    )

    aliases = {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    value = aliases.get(value, value)

    supported = {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
    }

    if value not in supported:
        raise ValueError(
            f"Unsupported primary metric: {value!r}"
        )

    return value


def _validation_score(
    pipeline,
    features,
    labels,
    metric,
) -> float:
    if metric == "roc_auc":
        classes = pipeline.classes_
        probabilities = pipeline.predict_proba(features)

        if len(classes) == 2:
            score = roc_auc_score(
                (labels == classes[1]).astype(int),
                probabilities[:, 1],
            )

        else:
            score = roc_auc_score(
                labels,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )

    else:
        predictions = pipeline.predict(features)

        if metric == "accuracy":
            score = accuracy_score(
                labels,
                predictions,
            )

        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(
                labels,
                predictions,
            )

        else:
            score = f1_score(
                labels,
                predictions,
                labels=pipeline.classes_,
                average=metric.removeprefix("f1_"),
                zero_division=0,
            )

    if not math.isfinite(float(score)):
        raise ValueError(
            "Validation produced a nonfinite score."
        )

    return float(score)


class TrainingExecutor:
    """
    Execute approved training synchronously.

    A later run service must invoke this in a worker process
    to enforce a wall-clock timeout and keep the web server responsive.
    """

    def __init__(
        self,
        *,
        review_store: CleaningReviewStore,
    ) -> None:
        self._reviews = review_store

    def fit(
        self,
        *,
        run_id: UUID,
        revision: int,
        dataset_id: UUID,
        dataset_bytes: bytes,
        execution_plan: ExecutionPlan,
        decision: TrainingDecision,
        runtime: TrainingRuntimeSettings | None = None,
    ) -> TrainingResult:
        # Snapshot and revalidate caller-supplied configurations.
        plan = ExecutionPlan.model_validate_json(
            execution_plan.model_dump_json()
        )

        choice = TrainingDecision.model_validate_json(
            decision.model_dump_json()
        )

        settings = (
            runtime
            if runtime is not None
            else TrainingRuntimeSettings()
        )

        settings = TrainingRuntimeSettings.model_validate(
            settings.model_dump()
        )

        if not isinstance(dataset_bytes, bytes):
            raise TypeError(
                "dataset_bytes must be bytes."
            )

        dataset_hash = hashlib.sha256(
            dataset_bytes
        ).hexdigest()

        plan_hash = execution_plan_sha256(plan)

        # Approval must match the actual CSV bytes and current plan.
        approved = self._reviews.require_approved(
            run_id=run_id,
            revision=revision,
            dataset_sha256=dataset_hash,
            plan_sha256=plan_hash,
        )

        frame = load_training_frame(dataset_bytes)

        profile = profile_dataset(
            frame=frame,
            dataset_id=dataset_id,
        )

        validate_proposal_against_frame(
            proposal=approved,
            execution_plan=plan,
            profile=profile,
            frame=frame,
        )

        split = split_dataset(
            frame=frame,
            execution_plan=plan,
            random_seed=settings.random_seed,
        )

        target = plan.target_column

        train_rows = list(split.train)
        validation_rows = list(split.validation)

        X_train = frame.iloc[train_rows].drop(
            columns=[target]
        )

        y_train = frame.iloc[train_rows][target].astype(str)

        X_validation = frame.iloc[validation_rows].drop(
            columns=[target]
        )

        y_validation = (
            frame.iloc[validation_rows][target].astype(str)
        )

        metric = _metric_name(plan)

        if (
            metric == "roc_auc"
            and set(y_validation) != set(y_train)
        ):
            raise ValueError(
                "ROC AUC needs every training class in validation."
            )

        models = build_candidate_models(
            candidate_models=plan.candidate_models,
            settings=choice.settings,
            runtime=settings,
        )

        results = []

        best_pipeline = None
        best_model = None
        best_score = -math.inf

        with threadpool_limits(limits=settings.n_jobs):
            for name, model in models.items():
                # Each model gets a fresh, unfitted preprocessor.
                pipeline = Pipeline(
                    [
                        (
                            "preprocessing",
                            build_preprocessor(
                                proposal=approved
                            ),
                        ),
                        ("model", model),
                    ]
                )

                logger.info(
                    "Candidate training started; model=%s.",
                    name,
                )

                started = time.monotonic()

                with warnings.catch_warnings(
                    record=True
                ) as captured:
                    warnings.simplefilter("always")

                    # All learned preprocessing is fitted here,
                    # using training rows only.
                    pipeline.fit(X_train, y_train)

                    score = _validation_score(
                        pipeline,
                        X_validation,
                        y_validation,
                        metric,
                    )

                converged = not any(
                    issubclass(
                        item.category,
                        ConvergenceWarning,
                    )
                    for item in captured
                )

                results.append(
                    {
                        "model": name,
                        "validation_score": score,
                        "metric": metric,
                        "eligible_for_selection": converged,
                        "warning_types": sorted(
                            {
                                item.category.__name__
                                for item in captured
                            }
                        ),
                        "elapsed_seconds": round(
                            time.monotonic() - started,
                            3,
                        ),
                        "parameters": model.get_params(
                            deep=False
                        ),
                    }
                )

                # Exact ties preserve the Planner's candidate order.
                if converged and score > best_score:
                    best_pipeline = pipeline
                    best_model = name
                    best_score = score

                logger.info(
                    "Candidate training completed; model=%s.",
                    name,
                )

        if best_pipeline is None:
            raise ValueError(
                "No candidate converged. "
                "Review the training settings."
            )

        # Verify that approval still matches before returning results.
        current = self._reviews.require_approved(
            run_id=run_id,
            revision=revision,
            dataset_sha256=dataset_hash,
            plan_sha256=plan_hash,
        )

        if (
            current.model_dump(mode="json")
            != approved.model_dump(mode="json")
        ):
            raise ValueError(
                "The approved proposal changed during training."
            )

        report = {
            "schema_version": 1,
            "run_id": str(run_id),
            "dataset_id": str(dataset_id),
            "dataset_sha256": dataset_hash,
            "execution_plan_sha256": plan_hash,
            "approved_revision": revision,
            "selected_model": best_model,
            "primary_metric": metric,
            "selected_validation_score": best_score,
            "candidates": results,
            "training_decision": choice.model_dump(
                mode="json"
            ),
            "runtime": settings.model_dump(mode="json"),
            "split_counts": split.to_dict()["counts"],
            "test_evaluated": False,
            "audit_status": "not_performed",
            "status": "trained",
            "selection_rule": (
                "Highest validation score among converged "
                "candidates; first candidate wins ties."
            ),
            "environment": {
                "python": platform.python_version(),
                "packages": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "pandas",
                        "scikit-learn",
                        "pydantic",
                        "joblib",
                    )
                },
            },
        }

        return TrainingResult(
            pipeline=best_pipeline,
            split=split,
            report=report,
        )