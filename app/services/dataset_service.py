from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from app.ml.profiling import profile_dataset
from app.ml.training import load_training_frame


DEFAULT_UPLOAD_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runtime"
    / "uploads"
)

MAX_CSV_BYTES = 25 * 1024 * 1024
MAX_METADATA_BYTES = 128 * 1024


class DatasetIntegrityError(ValueError):
    """A saved dataset or its metadata failed an integrity check."""


def _display_filename(filename: str) -> str:
    if not isinstance(filename, str):
        raise TypeError(
            "filename must be a string."
        )

    # Browser filenames are labels, never storage paths.
    name = (
        filename
        .replace("\\", "/")
        .rsplit("/", 1)[-1]
        .strip()
    )

    if (
        not name
        or len(name) > 255
        or not name.casefold().endswith(".csv")
        or name.casefold() == ".csv"
        or any(
            unicodedata.category(character).startswith("C")
            for character in name
        )
    ):
        raise ValueError(
            "Provide a valid CSV filename without control characters."
        )

    return name


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    dataset_id: UUID

    filename: str = Field(
        min_length=1,
        max_length=255,
    )

    dataset_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    size_bytes: int = Field(
        gt=0,
        le=MAX_CSV_BYTES,
        strict=True,
    )

    row_count: int = Field(
        ge=10,
        le=100_000,
        strict=True,
    )

    column_count: int = Field(
        ge=2,
        le=100,
        strict=True,
    )

    columns: list[str] = Field(
        min_length=2,
        max_length=100,
    )

    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_metadata(self):
        if self.filename != _display_filename(self.filename):
            raise ValueError(
                "Stored filename must contain only its display name."
            )

        if (
            len(self.columns) != self.column_count
            or len(self.columns) != len(set(self.columns))
            or any(
                not name.strip() or len(name) > 200
                for name in self.columns
            )
        ):
            raise ValueError(
                "Dataset column metadata is inconsistent."
            )

        return self


class DatasetService:
    """
    Store immutable original CSV uploads with validated metadata.

    Storage:
        <root>/<dataset_id>/dataset.csv
        <root>/<dataset_id>/metadata.json

    Methods are synchronous.
    API routes should call them using asyncio.to_thread.

    Pass this instance's load_bytes method to:
        RunService(dataset_loader=dataset_service.load_bytes)
    """

    def __init__(
        self,
        *,
        root_dir: str | Path = DEFAULT_UPLOAD_ROOT,
    ) -> None:
        self.root = (
            Path(root_dir)
            .expanduser()
            .resolve()
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _directory(
        self,
        dataset_id: UUID | str,
    ) -> Path:
        directory = (
            self.root
            / str(UUID(str(dataset_id)))
        )

        if directory.is_symlink():
            raise DatasetIntegrityError(
                "Dataset directory must not be a symbolic link."
            )

        if not directory.exists():
            raise FileNotFoundError(
                "Dataset not found."
            )

        if not directory.is_dir():
            raise DatasetIntegrityError(
                "Expected a dataset directory."
            )

        return directory

    @staticmethod
    def _read_file(
        path: Path,
        *,
        maximum_bytes: int,
    ) -> bytes:
        try:
            info = path.lstat()

        except FileNotFoundError as exc:
            raise DatasetIntegrityError(
                "A stored dataset file is missing."
            ) from exc

        if not stat.S_ISREG(info.st_mode):
            raise DatasetIntegrityError(
                "Dataset storage requires regular files."
            )

        if info.st_size > maximum_bytes:
            raise DatasetIntegrityError(
                "A stored file exceeds its size limit."
            )

        with path.open("rb") as stream:
            content = stream.read(
                maximum_bytes + 1
            )

        if len(content) > maximum_bytes:
            raise DatasetIntegrityError(
                "A stored file exceeds its size limit."
            )

        return content

    def _metadata(
        self,
        dataset_id: UUID | str,
    ) -> DatasetMetadata:
        directory = self._directory(
            dataset_id
        )

        content = self._read_file(
            directory / "metadata.json",
            maximum_bytes=MAX_METADATA_BYTES,
        )

        try:
            metadata = DatasetMetadata.model_validate_json(
                content
            )

        except ValidationError as exc:
            raise DatasetIntegrityError(
                "Stored dataset metadata is invalid."
            ) from exc

        if str(metadata.dataset_id) != directory.name:
            raise DatasetIntegrityError(
                "Dataset identity does not match its directory."
            )

        return metadata

    def save_bytes(
        self,
        *,
        filename: str,
        content: bytes,
    ) -> dict:
        filename = _display_filename(
            filename
        )

        if not isinstance(content, bytes):
            raise TypeError(
                "content must be bytes."
            )

        if not 0 < len(content) <= MAX_CSV_BYTES:
            raise ValueError(
                "CSV must be nonempty and at most 25 MB."
            )

        # Use the same parser and missing-value rules as training.
        # Parsing validates data; stored bytes are never rewritten.
        frame = load_training_frame(
            content
        )

        metadata = DatasetMetadata(
            dataset_id=uuid4(),
            filename=filename,
            dataset_sha256=hashlib.sha256(
                content
            ).hexdigest(),
            size_bytes=len(content),
            row_count=len(frame),
            column_count=len(frame.columns),
            columns=list(frame.columns),
            created_at=datetime.now(
                timezone.utc
            ),
        )

        metadata_bytes = metadata.model_dump_json(
            indent=2
        ).encode("utf-8")

        if len(metadata_bytes) > MAX_METADATA_BYTES:
            raise ValueError(
                "Dataset metadata exceeds the supported size."
            )

        destination = (
            self.root
            / str(metadata.dataset_id)
        )

        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                "Dataset ID already exists. Retry this upload."
            )

        # Publish the CSV and metadata together on the same filesystem.
        with tempfile.TemporaryDirectory(
            dir=self.root,
            prefix="upload-",
        ) as temporary:
            stage = Path(temporary) / "payload"

            stage.mkdir()

            for name, payload in (
                ("dataset.csv", content),
                ("metadata.json", metadata_bytes),
            ):
                with (stage / name).open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())

            os.rename(
                stage,
                destination,
            )

        return metadata.model_dump(
            mode="json"
        )

    def get_metadata(
        self,
        dataset_id: UUID | str,
    ) -> dict:
        """
        Read the manifest.

        Use load_bytes to verify the CSV content hash.
        """
        return self._metadata(
            dataset_id
        ).model_dump(mode="json")

    def load_bytes(
        self,
        dataset_id: UUID | str,
    ) -> bytes:
        """Return original bytes after size and hash validation."""
        metadata = self._metadata(
            dataset_id
        )

        directory = self._directory(
            dataset_id
        )

        content = self._read_file(
            directory / "dataset.csv",
            maximum_bytes=MAX_CSV_BYTES,
        )

        if (
            len(content) != metadata.size_bytes
            or hashlib.sha256(content).hexdigest()
            != metadata.dataset_sha256
        ):
            raise DatasetIntegrityError(
                "Stored CSV differs from its upload metadata."
            )

        return content

    def get_profile(
        self,
        dataset_id: UUID | str,
    ) -> dict:
        """Profile verified bytes without modifying the CSV."""
        identifier = UUID(
            str(dataset_id)
        )

        content = self.load_bytes(
            identifier
        )

        frame = load_training_frame(
            content
        )

        profile = profile_dataset(
            frame=frame,
            dataset_id=identifier,
        )

        # Reject non-JSON or nonfinite values from the profiler.
        return json.loads(
            json.dumps(
                profile.model_dump(mode="json"),
                allow_nan=False,
            )
        )

    def list_datasets(
        self,
        *,
        limit: int = 50,
    ) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError(
                "limit must be an integer between 1 and 100."
            )

        datasets = []

        for entry in self.root.iterdir():
            try:
                identifier = UUID(
                    entry.name
                )

            except ValueError:
                # Ignore unfinished upload staging directories.
                continue

            if str(identifier) != entry.name:
                continue

            datasets.append(
                self._metadata(identifier)
            )

        datasets.sort(
            key=lambda item: (
                item.created_at,
                str(item.dataset_id),
            ),
            reverse=True,
        )

        return [
            item.model_dump(mode="json")
            for item in datasets[:limit]
        ]