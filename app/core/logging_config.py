"""
Logging configuration for AutoAudit.

Features:
- Readable console logs.
- JSON logs with UTC timestamps.
- Rotating log files.
- Request/run context using ContextVar.
- Credential masking.
- Repeatable configuration without duplicate handlers.

Never log raw datasets, prompts, audio transcripts, API keys,
or complete settings objects.
"""

import json
import logging
import re
import threading

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Iterator

from app.core.config import Settings, get_settings


# ============================================================
# CONSTANTS AND CONTEXT
# ============================================================

LOGGER_NAME = "autoaudit"

LOG_FILE_NAME = "autoaudit.jsonl"
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5

_CONFIG_LOCK = threading.RLock()

_RUN_ID: ContextVar[str | None] = ContextVar(
    "autoaudit_run_id",
    default=None,
)

# Context identifiers must be application-generated IDs.
_VALID_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Mask common Groq key and Authorization patterns.
_GROQ_KEY_PATTERN = re.compile(
    r"\bgsk_[A-Za-z0-9_-]+\b"
)

_BEARER_PATTERN = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/-]+=*",
    flags=re.IGNORECASE,
)

_CREDENTIAL_PATTERN = re.compile(
    r"""(
        ["']?
        (?:api[_-]?key|access[_-]?token|password|secret)
        ["']?
        \s*[:=]\s*
    )
    (?:
        "[^"]*"
        |
        '[^']*'
        |
        [^\s,;}\]]+
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# SECRET REDACTION
# ============================================================

class SecretRedactor:
    """
    Redact configured secrets and common credential patterns.

    This is defense in depth, not a replacement for avoiding
    sensitive content in log messages.
    """

    def __init__(self, known_secrets: tuple[str, ...] = ()) -> None:
        self._known_secrets = tuple(
            sorted(
                {
                    value
                    for value in known_secrets
                    if value
                },
                key=len,
                reverse=True,
            )
        )

    def redact(self, text: str) -> str:
        for secret in self._known_secrets:
            text = text.replace(secret, "[REDACTED]")

        text = _GROQ_KEY_PATTERN.sub(
            "[REDACTED]",
            text,
        )

        text = _BEARER_PATTERN.sub(
            "Bearer [REDACTED]",
            text,
        )

        text = _CREDENTIAL_PATTERN.sub(
            lambda match: (
                f"{match.group(1)}[REDACTED]"
            ),
            text,
        )

        return text


# ============================================================
# LOG CONTEXT
# ============================================================

class RunContextFilter(logging.Filter):
    """Attach the current run ID before formatting a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID.get()
        return True


@contextmanager
def log_context(run_id: str) -> Iterator[None]:
    """
    Attach a run ID to logs within this context.

    Always restores the previous value, including on exceptions.

    Pass an application-generated run ID, not user input.
    """

    if not _VALID_RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must contain 1-128 letters, digits, "
            "underscores, or hyphens."
        )

    token = _RUN_ID.set(run_id)

    try:
        yield

    finally:
        _RUN_ID.reset(token)


# ============================================================
# FORMATTERS
# ============================================================

def utc_timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(
        record.created,
        tz=timezone.utc,
    ).isoformat(timespec="milliseconds")


class ConsoleFormatter(logging.Formatter):
    """Human-readable, single-line console output."""

    def __init__(self, redactor: SecretRedactor) -> None:
        super().__init__()
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        run_id = getattr(record, "run_id", None) or "-"

        output = (
            f"{utc_timestamp(record)} | "
            f"{record.levelname:<8} | "
            f"{record.name} | "
            f"run_id={run_id} | "
            f"{record.getMessage()}"
        )

        if record.exc_info:
            output += "\n" + self.formatException(
                record.exc_info
            )

        if record.stack_info:
            output += "\n" + self.formatStack(
                record.stack_info
            )

        output = self._redactor.redact(output)

        # Prevent embedded line breaks/control characters from
        # creating misleading extra console log entries.
        return "".join(
            character
            if character.isprintable()
            else repr(character)[1:-1]
            for character in output
        )


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def __init__(self, redactor: SecretRedactor) -> None:
        super().__init__()
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": utc_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "message": self._redactor.redact(
                record.getMessage()
            ),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
        }

        if record.exc_info:
            payload["exception"] = self._redactor.redact(
                self.formatException(record.exc_info)
            )

        if record.stack_info:
            payload["stack"] = self._redactor.redact(
                self.formatStack(record.stack_info)
            )

        # Explicit fields only: arbitrary record extras are not
        # serialized because they may contain sensitive objects.
        return json.dumps(
            payload,
            ensure_ascii=False,
        )


# ============================================================
# LOGGER ACCESS
# ============================================================

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger under the AutoAudit namespace.

    Example:
        logger = get_logger(__name__)
    """

    if name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)

    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)

    return logging.getLogger(
        f"{LOGGER_NAME}.{name}"
    )


# ============================================================
# APPLICATION LOGGING SETUP
# ============================================================

def configure_logging(
    settings: Settings | None = None,
) -> None:
    """
    Configure application logs.

    Call once during application startup, before accepting work.

    Repeated calls replace AutoAudit-owned handlers instead of
    adding duplicates. Third-party/root loggers are not modified.

    The rotating file handler is intended for a single application
    process. Multiple server workers need a centralized logging
    design instead of sharing this rotating file.
    """

    if settings is None:
        settings = get_settings()

    with _CONFIG_LOCK:
        settings.logs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        redactor = SecretRedactor(
            known_secrets=(
                settings.groq_api_key.get_secret_value(),
            )
        )

        context_filter = RunContextFilter()
        level = settings.log_level

        # Construct the file handler before changing the active
        # logger, so a file permission error fails clearly.
        file_handler = RotatingFileHandler(
            filename=settings.logs_dir / LOG_FILE_NAME,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
            errors="backslashreplace",
        )

        file_handler.setLevel(level)
        file_handler.setFormatter(
            JsonFormatter(redactor)
        )
        file_handler.addFilter(context_filter)
        file_handler.set_name("autoaudit_file")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(
            ConsoleFormatter(redactor)
        )
        console_handler.addFilter(context_filter)
        console_handler.set_name("autoaudit_console")

        application_logger = logging.getLogger(
            LOGGER_NAME
        )

        # Replace only the handlers owned by this module.
        owned_handler_names = {
            "autoaudit_file",
            "autoaudit_console",
        }

        for handler in tuple(application_logger.handlers):
            if handler.get_name() in owned_handler_names:
                application_logger.removeHandler(handler)
                handler.close()

        application_logger.setLevel(level)
        application_logger.disabled = False

        # Prevent duplicate output through root handlers.
        application_logger.propagate = False

        application_logger.addHandler(console_handler)
        application_logger.addHandler(file_handler)