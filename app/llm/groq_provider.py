"""
Groq client construction for AutoAudit.

Responsibilities:
- Resolve default Groq configuration.
- Build a ChatGroq client from trusted per-run settings.
- Use the backend-owned API key and official endpoint.
- Apply request timeouts and output limits.

Client construction does not send an inference request.

Structured-output validation, overall deadlines and retries
belong to provider_factory.py.
"""

import httpx

from langchain_groq import ChatGroq

from app.core.config import (
    ConfigurationError,
    Settings,
    get_settings,
)

from app.core.logging_config import get_logger

from app.schemas.run import (
    ProviderName,
    RunLLMConfiguration,
)


logger = get_logger(__name__)


# ============================================================
# TRUSTED PROVIDER ENDPOINT
# ============================================================

# Keep this backend-controlled.
# The browser must not supply an arbitrary provider URL.
GROQ_BASE_URL = "https://api.groq.com"


# ============================================================
# DEFAULT GROQ CONFIGURATION
# ============================================================

def resolve_groq_configuration(
    settings: Settings | None = None,
) -> RunLLMConfiguration:
    """
    Build a Groq configuration snapshot from server defaults.

    Validates that a model name and API key are configured.
    It does not verify the key with the remote service.
    """

    if settings is None:
        settings = get_settings()

    model_name = settings.require_provider("groq")

    return RunLLMConfiguration(
        provider=ProviderName.GROQ,
        model=model_name,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,

        # Local Ollama context settings do not apply to Groq.
        ollama_num_ctx=None,
    )


# ============================================================
# CLIENT FACTORY
# ============================================================

def build_groq_chat_model(
    configuration: RunLLMConfiguration,
    settings: Settings | None = None,
) -> ChatGroq:
    """
    Construct a Groq client for a server-approved run.

    The provider factory must validate any UI-selected model
    against the backend's allowed/available model list.

    This function accepts the model from the run configuration,
    so a valid run-specific model does not depend on GROQ_MODEL
    still having the same value.

    API keys are never accepted from the browser request.
    """

    if settings is None:
        settings = get_settings()

    if configuration.provider != ProviderName.GROQ:
        raise ConfigurationError(
            "Groq client construction requires "
            "a Groq run configuration."
        )

    if configuration.ollama_num_ctx is not None:
        raise ConfigurationError(
            "Groq configuration must not contain "
            "an Ollama context setting."
        )

    api_key = settings.groq_api_key.get_secret_value()

    if not api_key.strip():
        raise ConfigurationError(
            "Set GROQ_API_KEY in .env before using Groq."
        )

    if api_key != api_key.strip() or any(
        character.isspace()
        for character in api_key
    ):
        raise ConfigurationError(
            "GROQ_API_KEY must not contain whitespace."
        )

    request_timeout = httpx.Timeout(
        timeout=configuration.timeout_seconds,
        connect=min(
            10.0,
            configuration.timeout_seconds,
        ),
        pool=min(
            10.0,
            configuration.timeout_seconds,
        ),
    )

    client = ChatGroq(
        model=configuration.model,

        # Secret stays in backend memory.
        api_key=settings.groq_api_key,

        # Explicit endpoint avoids accidental base-URL overrides.
        base_url=GROQ_BASE_URL,

        temperature=configuration.temperature,
        max_tokens=configuration.max_output_tokens,

        timeout=request_timeout,

        # Avoid SDK retries multiplying retries in our common
        # invocation layer. That layer uses configuration.max_retries.
        max_retries=0,

        # One completion per invocation.
        n=1,

        # Collect a complete response before schema validation.
        streaming=False,
        disable_streaming=True,

        # Disable LangChain's global response cache for this client.
        cache=False,

        # Do not enable verbose model interaction output.
        verbose=False,
    )

    # Do not log API keys, settings objects, prompts or responses.
    logger.debug(
        "Groq client initialized; inference has not started."
    )

    return client