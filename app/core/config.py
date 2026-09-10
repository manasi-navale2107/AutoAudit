"""
Central configuration for AutoAudit.

Responsibilities:
- Load environment settings from the project-root .env file.
- Validate configuration values.
- Resolve application paths independently of the terminal location.
- Protect API keys from ordinary representation and serialization.
- Validate provider configuration before an LLM run starts.

Importing this module does not create folders, connect to a database,
download models, or make network requests.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# config.py -> core -> app -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

LLMProvider = Literal["groq", "ollama"]
LanguageCode = Literal["en", "mr", "hi"]


class ConfigurationError(RuntimeError):
    """Raised when a requested feature lacks valid configuration."""


class Settings(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    # ========================================================
    # APPLICATION
    # ========================================================

    app_name: str = Field(
        default="AutoAudit",
        min_length=1,
        max_length=100,
    )

    app_env: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    debug: bool = False

    host: str = Field(
        default="127.0.0.1",
        min_length=1,
    )

    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    # ========================================================
    # LLM SETTINGS
    # ========================================================

    llm_provider: LLMProvider = "groq"

    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    llm_max_output_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
    )

    llm_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        le=600.0,
    )

    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
    )

    # SecretStr masks the value in ordinary representations.
    # exclude=True also omits it from model_dump().
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        repr=False,
        exclude=True,
    )

    groq_model: str = Field(
        default="",
        max_length=200,
    )

    ollama_base_url: str = "http://localhost:11434"

    ollama_model: str = Field(
        default="",
        max_length=200,
    )

    ollama_timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        le=1800.0,
    )

    ollama_num_ctx: int = Field(
        default=8192,
        ge=1024,
        le=131072,
    )

    # ========================================================
    # LANGUAGE
    # ========================================================

    default_language: LanguageCode = "mr"

    # ========================================================
    # STORAGE
    # ========================================================

    runtime_dir: Path = Path("runtime")

    app_database_path: Path = Path(
        "runtime/database/autoaudit.sqlite3"
    )

    checkpoint_database_path: Path = Path(
        "runtime/database/checkpoints.sqlite3"
    )

    # ========================================================
    # DATASET LIMITS
    # ========================================================

    max_upload_mb: int = Field(
        default=25,
        ge=1,
        le=500,
    )

    max_dataset_rows: int = Field(
        default=100_000,
        ge=10,
        le=10_000_000,
    )

    max_dataset_columns: int = Field(
        default=200,
        ge=2,
        le=10_000,
    )

    # ========================================================
    # WORKFLOW
    # ========================================================

    max_concurrent_runs: int = Field(
        default=1,
        ge=1,
        le=8,
    )

    max_plan_revisions: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    graph_recursion_limit: int = Field(
        default=50,
        ge=10,
        le=500,
    )

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    random_seed: int = Field(
        default=42,
        ge=0,
        le=4_294_967_295,
    )

    validation_size: float = Field(
        default=0.20,
        gt=0.0,
        lt=1.0,
    )

    test_size: float = Field(
        default=0.20,
        gt=0.0,
        lt=1.0,
    )

    training_timeout_seconds: int = Field(
        default=600,
        ge=10,
        le=86_400,
    )

    # Positive, bounded parallelism to avoid using every CPU.
    model_n_jobs: int = Field(
        default=2,
        ge=1,
        le=32,
    )

    # ========================================================
    # VOICE
    # ========================================================

    voice_enabled: bool = False

    whisper_model: str = Field(
        default="small",
        min_length=1,
        max_length=200,
    )

    whisper_device: Literal[
        "cpu",
        "cuda",
        "auto",
    ] = "cpu"

    whisper_compute_type: Literal[
        "int8",
        "float16",
        "float32",
        "int8_float16",
        "int8_float32",
        "default",
    ] = "int8"

    max_audio_upload_mb: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    max_audio_duration_seconds: int = Field(
        default=60,
        ge=1,
        le=600,
    )

    # ========================================================
    # FIELD VALIDATORS
    # ========================================================

    @field_validator(
        "app_env",
        "llm_provider",
        "default_language",
        "whisper_device",
        "whisper_compute_type",
        mode="before",
    )
    @classmethod
    def normalize_choices(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()

        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()

        return value

    @field_validator(
        "runtime_dir",
        "app_database_path",
        "checkpoint_database_path",
        mode="before",
    )
    @classmethod
    def reject_empty_paths(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()

            if not value:
                raise ValueError("Storage paths must not be empty.")

        return value

    @field_validator(
        "runtime_dir",
        "app_database_path",
        "checkpoint_database_path",
    )
    @classmethod
    def resolve_project_paths(cls, value: Path) -> Path:
        path = value.expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")

        try:
            parsed = urlsplit(value)

            # Accessing .port also validates the port syntax/range.
            parsed.port

        except ValueError:
            raise ValueError(
                "OLLAMA_BASE_URL must contain a valid HTTP(S) URL."
            ) from None

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                "OLLAMA_BASE_URL must use http or https."
            )

        if not parsed.hostname:
            raise ValueError(
                "OLLAMA_BASE_URL must contain a hostname."
            )

        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "Do not include credentials in OLLAMA_BASE_URL."
            )

        if parsed.query or parsed.fragment:
            raise ValueError(
                "OLLAMA_BASE_URL must not contain a query or fragment."
            )

        if parsed.path not in {"", "/"}:
            raise ValueError(
                "Use the Ollama server origin, without /api or /v1."
            )

        return value

    # ========================================================
    # CROSS-FIELD VALIDATION
    # ========================================================

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.app_env == "production" and self.debug:
            raise ValueError(
                "DEBUG must be false in production."
            )

        if self.validation_size + self.test_size >= 1.0:
            raise ValueError(
                "VALIDATION_SIZE + TEST_SIZE must be less than 1."
            )

        if self.runtime_dir == PROJECT_ROOT:
            raise ValueError(
                "RUNTIME_DIR must be a dedicated folder, "
                "not the project root."
            )

        if self.runtime_dir.exists() and not self.runtime_dir.is_dir():
            raise ValueError(
                "RUNTIME_DIR points to a file instead of a folder."
            )

        database_paths = (
            self.app_database_path,
            self.checkpoint_database_path,
        )

        for path in database_paths:
            if (
                path == self.runtime_dir
                or not path.is_relative_to(self.runtime_dir)
            ):
                raise ValueError(
                    "Database files must be inside RUNTIME_DIR."
                )

            if path.exists() and not path.is_file():
                raise ValueError(
                    "A database path points to a directory."
                )

        if self.app_database_path == self.checkpoint_database_path:
            raise ValueError(
                "Application and checkpoint databases "
                "must use different files."
            )

        if (
            self.voice_enabled
            and self.whisper_device == "cpu"
            and self.whisper_compute_type
            in {"float16", "int8_float16"}
        ):
            raise ValueError(
                "For CPU voice processing, use int8, float32, "
                "int8_float32, or default."
            )

        return self

    # ========================================================
    # DERIVED VALUES
    # ========================================================

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def frontend_dir(self) -> Path:
        return PROJECT_ROOT / "frontend"

    @property
    def prompts_dir(self) -> Path:
        return PROJECT_ROOT / "app" / "prompts"

    @property
    def uploads_dir(self) -> Path:
        return self.runtime_dir / "uploads"

    @property
    def runs_dir(self) -> Path:
        return self.runtime_dir / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.runtime_dir / "logs"

    @property
    def audio_dir(self) -> Path:
        return self.runtime_dir / "audio"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_audio_upload_bytes(self) -> int:
        return self.max_audio_upload_mb * 1024 * 1024

    @property
    def training_size(self) -> float:
        return 1.0 - self.validation_size - self.test_size

    # ========================================================
    # PROVIDER VALIDATION
    # ========================================================

    def require_provider(
        self,
        provider: LLMProvider | None = None,
    ) -> str:
        """
        Validate local configuration and return the model ID.

        This does not check network connectivity, account access,
        model availability, or model-specific parameter support.
        The provider implementation will handle those checks.
        """

        selected = (
            self.llm_provider
            if provider is None
            else provider
        )

        if selected == "groq":
            if not self.groq_api_key.get_secret_value().strip():
                raise ConfigurationError(
                    "Set GROQ_API_KEY in .env before using Groq."
                )

            if not self.groq_model:
                raise ConfigurationError(
                    "Set GROQ_MODEL in .env before using Groq."
                )

            return self.groq_model

        if selected == "ollama":
            if not self.ollama_model:
                raise ConfigurationError(
                    "Set OLLAMA_MODEL in .env before using Ollama."
                )

            return self.ollama_model

        raise ConfigurationError(
            "Unsupported LLM provider. Choose groq or ollama."
        )

    # ========================================================
    # EXPLICIT STARTUP INITIALIZATION
    # ========================================================

    def create_runtime_directories(self) -> None:
        """
        Call from application startup, not during module import.

        File-system permission errors propagate to the caller,
        allowing startup to fail clearly instead of silently.
        """

        directories = (
            self.runtime_dir,
            self.uploads_dir,
            self.runs_dir,
            self.logs_dir,
            self.audio_dir,
            self.app_database_path.parent,
            self.checkpoint_database_path.parent,
        )

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load settings once per application process.

    Restart the application after editing .env.
    Tests may call get_settings.cache_clear() between scenarios.
    """

    return Settings()