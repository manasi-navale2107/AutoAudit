"""
Read-only dataset profiling for AutoAudit.

Responsibilities:
- Count missing and unique values.
- Infer conservative descriptive column types.
- Identify potential identifiers and ISO datetime columns.
- Summarize finite numeric values.
- Report observations without modifying the dataset.

Important:
- Inferred types are proposals, not approved preprocessing.
- Duplicate rows are not automatically removed.
- High cardinality is not proof that a column is an identifier.
- Descriptive statistics must not be reused as fitted
  preprocessing parameters.
"""

import re

from uuid import UUID

import numpy as np
import pandas as pd

from app.schemas.dataset import (
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    DatasetWarning,
    DatasetWarningCode,
    NumericSummary,
    WarningSeverity,
)


# ============================================================
# PROFILING POLICY
# ============================================================

HIGH_MISSING_RATIO = 0.30
HIGH_CARDINALITY_COUNT = 50
HIGH_CARDINALITY_RATIO = 0.50
IDENTIFIER_MIN_ROWS = 20
IDENTIFIER_UNIQUE_RATIO = 0.98

BOOLEAN_VALUES = {
    "true",
    "false",
    "yes",
    "no",
}

# Detect integer-like values where numeric conversion would
# remove meaningful leading zeros, such as customer ID "001".
LEADING_ZERO_INTEGER = re.compile(
    r"^[+-]?0\d+$"
)

# Deliberately conservative. Other date formats remain strings
# until their interpretation is explicitly established.
ISO_DATE_CANDIDATE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ].+)?$"
)

IDENTIFIER_NAMES = {
    "id",
    "uuid",
    "identifier",
    "zip",
    "zipcode",
    "postal_code",
    "postcode",
    "phone",
    "phone_number",
    "mobile",
    "mobile_number",
    "account_number",
}


# ============================================================
# HELPERS
# ============================================================

def _name_suggests_identifier(name: str) -> bool:
    normalized = re.sub(
        r"[\s-]+",
        "_",
        name.strip().lower(),
    )

    return (
        normalized in IDENTIFIER_NAMES
        or normalized.endswith("_id")
        or normalized.endswith("_uuid")
        or normalized.endswith("_code")
    )


def _warning(
    *,
    code: DatasetWarningCode,
    message: str,
    column: str | None = None,
    severity: WarningSeverity = WarningSeverity.WARNING,
) -> DatasetWarning:
    return DatasetWarning(
        code=code,
        severity=severity,
        message=message,
        columns=[column] if column is not None else [],
    )


def _numeric_summary(
    values: np.ndarray,
) -> NumericSummary:
    """
    Summarize a fully numeric, non-missing column.

    Use scaled arithmetic to reduce overflow risk for very
    large finite values.

    Counts refer to non-missing observations only.
    """

    finite_values = values[np.isfinite(values)]
    infinite_count = int(np.isinf(values).sum())

    if len(finite_values) == 0:
        return NumericSummary(
            finite_count=0,
            infinite_count=infinite_count,
        )

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))

    scale = float(np.max(np.abs(finite_values)))

    if scale == 0.0:
        mean = 0.0
        median = 0.0
        std = 0.0

    else:
        scaled = finite_values / scale

        mean = float(np.mean(scaled)) * scale
        median = float(np.median(scaled)) * scale
        std = float(np.std(scaled, ddof=0)) * scale

        # Floating-point rounding must not place location
        # statistics outside the observed interval.
        mean = min(max(mean, minimum), maximum)
        median = min(max(median, minimum), maximum)

    return NumericSummary(
        finite_count=int(len(finite_values)),
        infinite_count=infinite_count,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        median=median,
        std=std,
    )


def _is_iso_datetime(
    text_values: pd.Series,
) -> bool:
    """
    Require every non-missing value to look like a valid ISO date.

    UTC normalization is used only for this diagnostic check.
    It does not establish the dataset's actual time semantics.
    """

    matches = text_values.str.fullmatch(
        ISO_DATE_CANDIDATE,
        na=False,
    )

    if not bool(matches.all()):
        return False

    parsed = pd.to_datetime(
        text_values,
        format="ISO8601",
        errors="coerce",
        utc=True,
    )

    return bool(parsed.notna().all())


# ============================================================
# COLUMN PROFILING
# ============================================================

def _profile_column(
    *,
    name: str,
    series: pd.Series,
) -> tuple[ColumnProfile, list[DatasetWarning]]:
    warnings: list[DatasetWarning] = []

    row_count = len(series)
    missing_count = int(series.isna().sum())

    non_missing = series.dropna()
    non_missing_count = len(non_missing)
    unique_count = int(non_missing.nunique(dropna=True))

    base_profile = {
        "name": name,
        "pandas_dtype": str(series.dtype),
        "missing_count": missing_count,
        "unique_count": unique_count,
    }

    if missing_count:
        missing_ratio = missing_count / row_count

        warnings.append(
            _warning(
                code=DatasetWarningCode.MISSING_VALUES,
                column=name,
                severity=(
                    WarningSeverity.WARNING
                    if missing_ratio >= HIGH_MISSING_RATIO
                    else WarningSeverity.INFO
                ),
                message=(
                    f"{missing_count} of {row_count} values are missing. "
                    "Choose an explicit preprocessing policy."
                ),
            )
        )

    if non_missing_count == 0:
        warnings.append(
            _warning(
                code=DatasetWarningCode.ALL_MISSING_COLUMN,
                column=name,
                message=(
                    "This column contains no observed values. "
                    "It cannot provide a learned feature as-is."
                ),
            )
        )

        return (
            ColumnProfile(
                **base_profile,
                kind=ColumnKind.UNKNOWN,
            ),
            warnings,
        )

    if unique_count == 1:
        warnings.append(
            _warning(
                code=DatasetWarningCode.CONSTANT_COLUMN,
                column=name,
                severity=WarningSeverity.INFO,
                message=(
                    "All observed values are identical. "
                    "Review its usefulness before retaining it."
                ),
            )
        )

    # Work on a separate string representation for inference.
    # Original values, spaces and column names remain unchanged.
    text_values = non_missing.astype("string").str.strip()

    name_hint = _name_suggests_identifier(name)

    has_leading_zeros = bool(
        text_values.str.fullmatch(
            LEADING_ZERO_INTEGER,
            na=False,
        ).any()
    )

    nearly_unique = (
        non_missing_count >= IDENTIFIER_MIN_ROWS
        and unique_count / non_missing_count
        >= IDENTIFIER_UNIQUE_RATIO
    )

    if name_hint or has_leading_zeros or nearly_unique:
        warnings.append(
            _warning(
                code=DatasetWarningCode.POSSIBLE_IDENTIFIER,
                column=name,
                severity=WarningSeverity.INFO,
                message=(
                    "The name or value pattern may indicate an identifier. "
                    "Confirm its meaning; uniqueness alone is not proof."
                ),
            )
        )

    high_cardinality = (
        unique_count > HIGH_CARDINALITY_COUNT
        and unique_count / non_missing_count
        >= HIGH_CARDINALITY_RATIO
    )

    # Names and leading zeros receive conservative categorical
    # treatment. The agent may propose another interpretation
    # after obtaining task context.
    if name_hint or has_leading_zeros:
        kind = ColumnKind.CATEGORICAL
        numeric_summary = None

    elif set(text_values.str.lower().unique()).issubset(
        BOOLEAN_VALUES
    ):
        kind = ColumnKind.BOOLEAN
        numeric_summary = None

    elif _is_iso_datetime(text_values):
        kind = ColumnKind.DATETIME
        numeric_summary = None

        warnings.append(
            _warning(
                code=DatasetWarningCode.POSSIBLE_DATETIME,
                column=name,
                severity=WarningSeverity.INFO,
                message=(
                    "Observed values can be parsed as ISO dates/times. "
                    "Confirm timezone and prediction-time meaning."
                ),
            )
        )

    else:
        numeric_values = pd.to_numeric(
            text_values,
            errors="coerce",
        )

        numeric_count = int(numeric_values.notna().sum())

        if numeric_count == non_missing_count:
            values = numeric_values.to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )

            kind = ColumnKind.NUMERIC
            numeric_summary = _numeric_summary(values)

            if numeric_summary.infinite_count:
                warnings.append(
                    _warning(
                        code=DatasetWarningCode.INFINITE_VALUES,
                        column=name,
                        message=(
                            "The column contains positive or negative "
                            "infinity. An explicit repair is required."
                        ),
                    )
                )

        else:
            kind = (
                ColumnKind.TEXT
                if high_cardinality
                else ColumnKind.CATEGORICAL
            )

            numeric_summary = None

            if numeric_count > 0:
                warnings.append(
                    _warning(
                        code=DatasetWarningCode.MIXED_TYPES,
                        column=name,
                        message=(
                            "Some values parse as numbers and others do not. "
                            "Confirm whether this is categorical data "
                            "or inconsistent numeric formatting."
                        ),
                    )
                )

    if (
        high_cardinality
        and kind in {
            ColumnKind.CATEGORICAL,
            ColumnKind.TEXT,
        }
    ):
        warnings.append(
            _warning(
                code=DatasetWarningCode.HIGH_CARDINALITY,
                column=name,
                message=(
                    f"This column has {unique_count} distinct observed "
                    "values. Check feature meaning and encoding cost."
                ),
            )
        )

    return (
        ColumnProfile(
            **base_profile,
            kind=kind,
            numeric_summary=numeric_summary,
        ),
        warnings,
    )


# ============================================================
# PUBLIC PROFILING FUNCTION
# ============================================================

def profile_dataset(
    *,
    frame: pd.DataFrame,
    dataset_id: UUID,
) -> DatasetProfile:
    """
    Build a profile without mutating the DataFrame.

    Intended input: the bounded CSV frame returned by
    DatasetService.load_dataframe().

    This is a synchronous CPU operation. Async services should
    call it through asyncio.to_thread().

    Full-data summaries are diagnostic only. They must not be
    passed into fit operations or used to tune model performance.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError(
            "A non-empty dataset is required for profiling."
        )

    if len(frame.columns) < 2:
        raise ValueError(
            "At least two columns are required."
        )

    if not frame.columns.is_unique:
        raise ValueError(
            "Dataset column names must be unique."
        )

    if any(
        not isinstance(name, str) or not name.strip()
        for name in frame.columns
    ):
        raise ValueError(
            "Dataset columns must have non-blank string names."
        )

    profiles: list[ColumnProfile] = []
    warnings: list[DatasetWarning] = []

    for name in frame.columns:
        column_profile, column_warnings = _profile_column(
            name=name,
            series=frame[name],
        )

        profiles.append(column_profile)
        warnings.extend(column_warnings)

    duplicate_count = int(
        frame.duplicated(keep="first").sum()
    )

    if duplicate_count:
        warnings.append(
            _warning(
                code=DatasetWarningCode.DUPLICATE_ROWS,
                message=(
                    f"{duplicate_count} rows repeat an earlier row. "
                    "Confirm whether these are duplicate records "
                    "or valid repeated observations before any removal."
                ),
            )
        )

    return DatasetProfile(
        dataset_id=dataset_id,
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=profiles,
        duplicate_row_count=duplicate_count,
        warnings=warnings,
    )