
import numpy as np
import pandas as pd
import pytest

from app.schemas.cleaning import ExecutionPlan
from app.ml.splitting import split_dataset


def make_classification_frame():
    """Create a small binary classification dataset."""
    return pd.DataFrame(
        {
            "feature_a": list(range(30)),
            "feature_b": [value % 5 for value in range(30)],
            "target": [0, 1] * 15,
        }
    )


def make_plan(
    strategy="stratified",
    group_column=None,
    time_column=None,
    train_fraction=0.6,
    validation_fraction=0.2,
    test_fraction=0.2,
):
    """Create a valid ExecutionPlan for split tests."""
    return ExecutionPlan.model_validate(
        {
            "objective": "Predict the target class",
            "task_type": "binary_classification",
            "target_column": "target",
            "prediction_context": "Predict the class for new observations",
            "primary_metric": "accuracy",
            "metric_rationale": "Measure overall classification correctness",
            "positive_class_label": None,
            "split": {
                "strategy": strategy,
                "train_fraction": train_fraction,
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "group_column": group_column,
                "time_column": time_column,
                "rationale": "Test split integrity",
            },
            "candidate_models": ["logistic_regression"],
            "feature_exclusions": [],
            "assumptions": [],
        }
    )


def test_stratified_split_has_integrity_and_is_reproducible():
    frame = make_classification_frame()
    plan = make_plan()

    first = split_dataset(
        frame=frame,
        execution_plan=plan,
        random_seed=42,
    )
    second = split_dataset(
        frame=frame,
        execution_plan=plan,
        random_seed=42,
    )

    assert first == second

    train = set(first.train)
    validation = set(first.validation)
    test = set(first.test)

    assert train
    assert validation
    assert test

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

    assert train | validation | test == set(range(len(frame)))


def test_group_split_keeps_each_group_in_one_partition():
    frame = pd.DataFrame(
        {
            "feature": range(36),
            "group": [f"group_{value // 3}" for value in range(36)],
            "target": [0, 1, 0] * 12,
        }
    )
    plan = make_plan(
        strategy="group",
        group_column="group",
    )

    result = split_dataset(
        frame=frame,
        execution_plan=plan,
        random_seed=42,
    )

    train_groups = set(
        frame.iloc[list(result.train)]["group"]
    )
    validation_groups = set(
        frame.iloc[list(result.validation)]["group"]
    )
    test_groups = set(
        frame.iloc[list(result.test)]["group"]
    )

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_temporal_split_preserves_chronological_order():
    frame = pd.DataFrame(
        {
            "feature": range(30),
            "event_time": pd.date_range(
                "2025-01-01",
                periods=30,
                freq="D",
            ).astype(str),
            "target": [0, 1] * 15,
        }
    )
    plan = make_plan(
        strategy="temporal",
        time_column="event_time",
    )

    result = split_dataset(
        frame=frame,
        execution_plan=plan,
        random_seed=42,
    )

    train_times = pd.to_datetime(
        frame.iloc[list(result.train)]["event_time"],
        utc=True,
    )
    validation_times = pd.to_datetime(
        frame.iloc[list(result.validation)]["event_time"],
        utc=True,
    )
    test_times = pd.to_datetime(
        frame.iloc[list(result.test)]["event_time"],
        utc=True,
    )

    assert train_times.max() < validation_times.min()
    assert validation_times.max() < test_times.min()


def test_missing_target_labels_are_rejected():
    frame = make_classification_frame()
    frame.loc[0, "target"] = np.nan
    plan = make_plan()

    with pytest.raises(ValueError):
        split_dataset(frame=frame, execution_plan=plan)


def test_single_class_target_is_rejected():
    frame = make_classification_frame()
    frame["target"] = 1
    plan = make_plan()

    with pytest.raises(ValueError):
        split_dataset(frame=frame, execution_plan=plan)


def test_duplicate_column_names_are_rejected():
    frame = make_classification_frame()
    frame.columns = ["feature", "feature", "target"]
    plan = make_plan()

    with pytest.raises(ValueError):
        split_dataset(frame=frame, execution_plan=plan)


def test_missing_group_identifiers_are_rejected():
    frame = pd.DataFrame(
        {
            "feature": range(30),
            "group": [
                f"group_{value // 3}" for value in range(30)
            ],
            "target": [0, 1] * 15,
        }
    )
    frame.loc[0, "group"] = None

    plan = make_plan(
        strategy="group",
        group_column="group",
    )

    with pytest.raises(ValueError):
        split_dataset(frame=frame, execution_plan=plan)


def test_invalid_random_seed_is_rejected():
    frame = make_classification_frame()
    plan = make_plan()

    with pytest.raises(ValueError):
        split_dataset(
            frame=frame,
            execution_plan=plan,
            random_seed=-1,
        )


def test_invalid_split_fractions_are_rejected():
    frame = make_classification_frame()

    plan = make_plan(
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    # Make the plan invalid after schema validation.
    plan.split.test_fraction = 0.3

    with pytest.raises(ValueError):
        split_dataset(
            frame=frame,
            execution_plan=plan,
        )
