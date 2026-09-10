from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)

from app.schemas.cleaning import ExecutionPlan


@dataclass(frozen=True)
class DatasetSplit:
    """
    Zero-based row positions, independent of DataFrame index labels.
    """

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    strategy: str
    random_seed: int

    def to_dict(self) -> dict:
        return {
            "row_identifier": "zero-based row position",
            "strategy": self.strategy,
            "random_seed": self.random_seed,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
        }


def _validate_partitions(
    *,
    partitions: tuple[np.ndarray, np.ndarray, np.ndarray],
    row_count: int,
) -> None:
    sets = []

    for partition in partitions:
        positions = partition.tolist()
        unique = set(positions)

        if not positions:
            raise ValueError("Every split must contain at least one row.")

        if len(positions) != len(unique):
            raise ValueError("A split contains repeated row positions.")

        sets.append(unique)

    train, validation, test = sets

    if train & validation or train & test or validation & test:
        raise ValueError("Split partitions overlap.")

    if train | validation | test != set(range(row_count)):
        raise ValueError("Split partitions do not cover the dataset exactly.")


def _group_split(
    *,
    positions: np.ndarray,
    groups: pd.Series,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if groups.isna().any():
        raise ValueError("Group identifiers must not contain missing values.")

    if groups.nunique() < 3:
        raise ValueError("Group splitting requires at least three groups.")

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=random_seed,
    )

    train_validation, test = next(
        outer.split(positions, groups=groups)
    )

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction / (1.0 - test_fraction),
        random_state=random_seed,
    )

    # The inner splitter returns positions relative to train_validation.
    relative_train, relative_validation = next(
        inner.split(
            train_validation,
            groups=groups.iloc[train_validation],
        )
    )

    train = train_validation[relative_train]
    validation = train_validation[relative_validation]

    group_sets = [
        set(groups.iloc[part])
        for part in (train, validation, test)
    ]

    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise ValueError("Group isolation failed.")

    return train, validation, test


def _temporal_split(
    *,
    timestamps: pd.Series,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(
        timestamps,
        errors="coerce",
        utc=True,
        format="ISO8601",
    )

    if dates.isna().any():
        raise ValueError(
            "Temporal splitting requires nonmissing ISO-8601 timestamps."
        )

    unique_dates = pd.Index(dates.unique()).sort_values()
    count = len(unique_dates)

    if count < 3:
        raise ValueError(
            "Temporal splitting requires three distinct timestamps."
        )

    # Allocate whole timestamp blocks. Identical timestamps never
    # appear in different partitions.
    train_boundary = max(
        1,
        min(count - 2, int(count * train_fraction)),
    )

    validation_boundary = max(
        train_boundary + 1,
        min(
            count - 1,
            int(count * (train_fraction + validation_fraction)),
        ),
    )

    first_validation_time = unique_dates[train_boundary]
    first_test_time = unique_dates[validation_boundary]

    train = np.flatnonzero(
        (dates < first_validation_time).to_numpy()
    )

    validation = np.flatnonzero(
        (
            (dates >= first_validation_time)
            & (dates < first_test_time)
        ).to_numpy()
    )

    test = np.flatnonzero(
        (dates >= first_test_time).to_numpy()
    )

    # Return each partition in chronological order.
    def chronological(positions: np.ndarray) -> np.ndarray:
        order = np.argsort(
            dates.iloc[positions].astype("int64").to_numpy(),
            kind="stable",
        )
        return positions[order]

    return (
        chronological(train),
        chronological(validation),
        chronological(test),
    )


def split_dataset(
    *,
    frame: pd.DataFrame,
    execution_plan: ExecutionPlan,
    random_seed: int = 42,
) -> DatasetSplit:
    """
    Construct splits without changing data or fitting preprocessing.

    Group fractions apply to groups; temporal fractions apply to
    distinct timestamp blocks. Their row proportions may differ.

    No fallback to another strategy is performed on failure.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("The dataset is empty.")

    if not frame.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")

    if not 0 <= random_seed <= 2**32 - 1:
        raise ValueError("random_seed is outside the supported range.")

    target_column = execution_plan.target_column

    if target_column not in frame.columns:
        raise ValueError("The target column is missing.")

    target = frame[target_column]

    if target.isna().any():
        raise ValueError("Target labels must not contain missing values.")

    if target.nunique() < 2:
        raise ValueError("Classification requires at least two classes.")

    specification = execution_plan.split

    train_fraction = float(specification.train_fraction)
    validation_fraction = float(specification.validation_fraction)
    test_fraction = float(specification.test_fraction)

    fractions = (
        train_fraction,
        validation_fraction,
        test_fraction,
    )

    if not all(np.isfinite(value) and 0 < value < 1 for value in fractions):
        raise ValueError("Split fractions must be finite and between 0 and 1.")

    if not isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to one.")

    strategy = str(
        getattr(specification.strategy, "value", specification.strategy)
    )

    positions = np.arange(len(frame))

    if strategy == "stratified":
        train_validation, test = train_test_split(
            positions,
            test_size=test_fraction,
            stratify=target,
            random_state=random_seed,
        )

        train, validation = train_test_split(
            train_validation,
            test_size=validation_fraction / (1.0 - test_fraction),
            stratify=target.iloc[train_validation],
            random_state=random_seed,
        )

    elif strategy == "group":
        column = specification.group_column

        if column is None or column not in frame.columns:
            raise ValueError("A valid group column is required.")

        if column == target_column:
            raise ValueError("The target cannot be the group column.")

        train, validation, test = _group_split(
            positions=positions,
            groups=frame[column],
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            random_seed=random_seed,
        )

    elif strategy == "temporal":
        column = specification.time_column

        if column is None or column not in frame.columns:
            raise ValueError("A valid time column is required.")

        if column == target_column:
            raise ValueError("The target cannot be the time column.")

        train, validation, test = _temporal_split(
            timestamps=frame[column],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )

    else:
        raise ValueError(f"Unsupported split strategy: {strategy!r}")

    _validate_partitions(
        partitions=(train, validation, test),
        row_count=len(frame),
    )

    training_classes = set(target.iloc[train])
    validation_classes = set(target.iloc[validation])

    if len(training_classes) < 2:
        raise ValueError(
            "Training contains fewer than two classes. "
            "Review the dataset or split plan."
        )

    if not validation_classes.issubset(training_classes):
        raise ValueError(
            "Validation contains a class absent from training. "
            "Review the split plan."
        )

    return DatasetSplit(
        train=tuple(int(value) for value in train),
        validation=tuple(int(value) for value in validation),
        test=tuple(int(value) for value in test),
        strategy=strategy,
        random_seed=random_seed,
    )