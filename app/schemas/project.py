
"""Contracts for AutoAudit's end-to-end generated ML projects.

The Planner produces a blueprint *before* any code is generated.
Only downstream agents/services create or change project files.
The final ZIP is created by the packaging stage, not by the first
three agents.

These schemas validate proposed filenames and responsibilities.
They do not execute generated code or grant human approval.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_PROJECT_FILES = 100
MAX_RELATIVE_PATH = 240

# The runtime owns these files/directories; an LLM cannot place code in them.
BLOCKED_PARTS = frozenset({
    ".git", ".env", ".venv", "venv", "__pycache__", "runtime",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules",
})
RESERVED_MANIFEST = "project_manifest.json"
WINDOWS_DEVICE_NAMES = frozenset({
    "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})


class ProjectSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class MLTask(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class DatasetStatus(str, Enum):
    AVAILABLE = "available"
    REQUIRED = "required"


class ProjectStage(str, Enum):
    """Stages in the full team workflow; only the first 3 are ours."""

    PLANNER = "planner"
    DATA_ENGINEERING = "data_engineering"
    MODEL_TRAINING = "model_training"
    LEAKAGE_REFACTORING = "leakage_refactoring"
    TESTING = "testing"
    VERIFICATION = "verification"
    DOCUMENTATION = "documentation"
    PACKAGING = "packaging"


class FileKind(str, Enum):
    PYTHON = "python"
    MODEL = "model"
    DATA = "data"
    CONFIG = "config"
    REPORT = "report"
    TEST = "test"
    DOCUMENTATION = "documentation"


class InputMode(str, Enum):
    FORM = "form"
    CSV = "csv_upload"
    BOTH = "both"


class ProjectDatasetSpec(ProjectSchema):
    """Dataset needs; REQUIRED is a blocker for actual model training."""

    status: DatasetStatus
    description: str = Field(min_length=5, max_length=2000)
    target_column: str | None = Field(default=None, max_length=200)
    expected_feature_columns: list[str] = Field(default_factory=list, max_length=200)
    source_notes: str = Field(min_length=1, max_length=2000)

    @field_validator("description", "source_notes")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank.")
        return value.strip()

    @field_validator("target_column")
    @classmethod
    def nonblank_target(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Use null when the target is not yet known.")
        # Preserve exact dataset column names, including meaningful spaces.
        return value

    @field_validator("expected_feature_columns")
    @classmethod
    def unique_feature_columns(cls, value: list[str]) -> list[str]:
        if any(not name.strip() or len(name) > 200 for name in value):
            raise ValueError("Feature names must be nonblank and <= 200 characters.")
        if len(value) != len(set(value)):
            raise ValueError("Feature names must be unique.")
        return value

    @model_validator(mode="after")
    def target_is_not_feature(self) -> Self:
        if self.target_column in self.expected_feature_columns:
            raise ValueError("The target must not appear among input features.")
        return self


class StreamlitSpec(ProjectSchema):
    """UI specification; generated app code comes later."""

    framework: Literal["streamlit"] = "streamlit"
    title: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=5, max_length=1500)
    input_mode: InputMode = InputMode.FORM
    show_metrics: bool = True
    show_probabilities: bool = False
    prediction_explanation: str = Field(min_length=5, max_length=1000)

    @field_validator("title", "description", "prediction_explanation")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Streamlit UI text must not be blank.")
        return value.strip()


def validate_relative_path(raw_path: str) -> str:
    """Reject unsafe, ambiguous, and OS-dependent generated filenames."""

    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("A project file needs a relative path.")
    if len(raw_path) > MAX_RELATIVE_PATH:
        raise ValueError("Project file path is too long.")
    if "\\" in raw_path or "\x00" in raw_path or raw_path.startswith(("/", "~")):
        raise ValueError("Only relative POSIX file paths are allowed.")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_path):
        raise ValueError("Control characters are not allowed in paths.")

    parts = raw_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Empty and traversal path segments are forbidden.")
    for part in parts:
        if len(part) > 100 or not re.fullmatch(r"[A-Za-z0-9_.-]+", part):
            raise ValueError("Use only letters, numbers, underscore, dash and dot in paths.")
        if part.endswith(".") or part.casefold() in BLOCKED_PARTS:
            raise ValueError("This path includes a reserved or unsafe segment.")
        if part.split(".", 1)[0].casefold() in WINDOWS_DEVICE_NAMES:
            raise ValueError("Windows device filenames are not allowed.")

    if PurePosixPath(raw_path).as_posix() != raw_path:
        raise ValueError("Project file path is not canonical.")
    if raw_path.casefold() == RESERVED_MANIFEST:
        raise ValueError("The backend, not the LLM, writes project_manifest.json.")
    return raw_path


class PlannedFile(ProjectSchema):
    """One file the Planner assigns to a future implementation stage.

    A planned file is NOT evidence that the file has been created.
    """

    path: str = Field(min_length=1, max_length=MAX_RELATIVE_PATH)
    kind: FileKind
    producer: ProjectStage
    purpose: str = Field(min_length=5, max_length=1000)
    depends_on: list[str] = Field(default_factory=list, max_length=20)
    required: bool = True

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @field_validator("depends_on")
    @classmethod
    def safe_dependencies(cls, values: list[str]) -> list[str]:
        normalized = [validate_relative_path(value) for value in values]
        if len(normalized) != len({value.casefold() for value in normalized}):
            raise ValueError("Duplicate file dependencies are not allowed.")
        return normalized

    @field_validator("purpose")
    @classmethod
    def nonblank_purpose(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("File purpose must not be blank.")
        return value.strip()

    @model_validator(mode="after")
    def check_file_owner(self) -> Self:
        if self.producer == ProjectStage.PLANNER:
            raise ValueError("Planner specifies files; it does not write project code.")
        if self.path.casefold() in {name.casefold() for name in self.depends_on}:
            raise ValueError("A file cannot depend on itself.")
        return self


class ProjectBlueprint(ProjectSchema):
    """The Planner's versioned project-wide contract.

    Downstream agents append actual files and evidence in the shared
    workspace. They must not pretend to complete future agents' tasks.
    """

    schema_version: Literal["1.0"] = "1.0"
    revision: int = Field(default=1, ge=1)
    project_slug: str = Field(min_length=3, max_length=64)
    goal: str = Field(min_length=5, max_length=5000)
    summary: str = Field(min_length=10, max_length=2000)
    task_type: MLTask
    dataset: ProjectDatasetSpec
    streamlit: StreamlitSpec
    files: list[PlannedFile] = Field(min_length=5, max_length=MAX_PROJECT_FILES)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    cleaning_approval_required: Literal[True] = True

    @field_validator("project_slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", value):
            raise ValueError("Slug must use lowercase letters, numbers or underscore.")
        return value

    @field_validator("goal", "summary")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Goal and summary must not be blank.")
        return value.strip()

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 500 for value in values):
            raise ValueError("Assumptions must be nonblank and <= 500 characters.")
        return values

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        index = {item.path.casefold(): item for item in self.files}
        if len(index) != len(self.files):
            raise ValueError("Project filenames must be unique, ignoring case.")

        for required in ("app.py", "requirements.txt", "README.md"):
            if required.casefold() not in index:
                raise ValueError(f"Every generated project must plan {required}.")
        if index["app.py"].kind != FileKind.PYTHON:
            raise ValueError("app.py must be a Python file for Streamlit.")

        if not any(item.producer == ProjectStage.DATA_ENGINEERING for item in self.files):
            raise ValueError("A Data Engineering file must be planned.")
        if not any(item.producer == ProjectStage.MODEL_TRAINING for item in self.files):
            raise ValueError("A Model Training file must be planned.")

        # The planner cannot introduce a file where another file is a directory.
        keys = set(index)
        for key in keys:
            segments = key.split("/")
            for length in range(1, len(segments)):
                if "/".join(segments[:length]) in keys:
                    raise ValueError("A project path cannot be both a file and directory.")

        graph: dict[str, list[str]] = {}
        stage_order = list(ProjectStage)
        for item in self.files:
            dependencies = []
            for raw_dependency in item.depends_on:
                dep_key = raw_dependency.casefold()
                dependency = index.get(dep_key)
                if dependency is None:
                    raise ValueError(f"Unknown file dependency: {raw_dependency}.")
                if stage_order.index(dependency.producer) > stage_order.index(item.producer):
                    raise ValueError("A file cannot depend on a later stage's output.")
                dependencies.append(dep_key)
            graph[item.path.casefold()] = dependencies

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError("The project contains cyclic file dependencies.")
            if key in visited:
                return
            visiting.add(key)
            for dependency in graph[key]:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in graph:
            visit(key)

        if self.task_type == MLTask.REGRESSION and self.streamlit.show_probabilities:
            raise ValueError("Regression apps do not display class probabilities.")

        return self

    def files_for(self, stage: ProjectStage) -> list[PlannedFile]:
        """Return planned files without claiming they have been produced."""
        return [item for item in self.files if item.producer == stage]
