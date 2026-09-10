"""
Ollama client construction for AutoAudit.

Responsibilities:
- Resolve the default Ollama run configuration.
- Build a ChatOllama client from trusted run settings.
- Apply request timeouts and generation limits.

This module does not:
- Download models.
- Execute generated Python.
- Change a run's selected provider.
- Make inference calls during client construction.

Shared structured-output validation and bounded retries belong
to provider_factory.py so Groq and Ollama follow one contract.
"""

import httpx

from langchain_ollama import ChatOllama

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
# DEFAULT OLLAMA CONFIGURATION
# ============================================================

def resolve_ollama_configuration(
    settings: Settings | None = None,
) -> RunLLMConfiguration:
    """
    Build a configuration snapshot from server defaults.

    Calling this function does not change LLM_PROVIDER.
    Groq may remain the application's default provider.
    """

    if settings is None:
        settings = get_settings()

    model_name = settings.require_provider("ollama")

    return RunLLMConfiguration(
        provider=ProviderName.OLLAMA,
        model=model_name,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=settings.ollama_timeout_seconds,
        max_retries=settings.llm_max_retries,
        ollama_num_ctx=settings.ollama_num_ctx,
    )


# ============================================================
# CLIENT FACTORY
# ============================================================

def build_ollama_chat_model(
    configuration: RunLLMConfiguration,
    settings: Settings | None = None,
) -> ChatOllama:
    """
    Construct an Ollama client for a server-approved run.

    The provider factory must validate any UI-selected model
    against the backend's allowed/available models before calling
    this function.

    configuration contains the immutable per-run model settings.
    settings supplies the trusted server URL.

    HTTPX timeouts limit individual network operations.
    The shared invocation layer must separately enforce the
    overall deadline across generation and retries.
    """

    if settings is None:
        settings = get_settings()

    if configuration.provider != ProviderName.OLLAMA:
        raise ConfigurationError(
            "Ollama client construction requires "
            "an Ollama run configuration."
        )

    if configuration.ollama_num_ctx is None:
        raise ConfigurationError(
            "Ollama requires an explicit context size."
        )

    if (
        configuration.max_output_tokens
        >= configuration.ollama_num_ctx
    ):
        raise ConfigurationError(
            "Ollama output-token limit must be smaller than "
            "its context size, leaving space for the prompt."
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

    client = ChatOllama(
        model=configuration.model,
        base_url=settings.ollama_base_url,

        # Generation configuration.
        temperature=configuration.temperature,
        num_predict=configuration.max_output_tokens,
        num_ctx=configuration.ollama_num_ctx,

        # Retain a loaded model briefly between agent calls.
        keep_alive="5m",

        # Model availability is checked explicitly by the
        # provider service rather than during construction.
        validate_model_on_init=False,

        # Prevent LangChain's response cache from reusing
        # outputs across requests through a global cache.
        cache=False,

        # The shared invocation layer collects a complete
        # response before validating the structured output.
        disable_streaming=True,

        client_kwargs={
            "timeout": request_timeout,

            # Avoid routing local requests through proxy
            # settings inherited from the environment.
            "trust_env": False,

            # Do not follow redirects to another server.
            "follow_redirects": False,
        },
    )

    # Do not log prompts, model output, URLs or configuration
    # objects. Model names come from trusted run configuration.
    logger.debug(
        "Ollama client initialized; inference has not started."
    )

    return client