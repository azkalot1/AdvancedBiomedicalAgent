from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


# Optional: enable this if you want an outer retry wrapper on any Runnable
# from langchain_core.runnables import Runnable

_SUPPORTED = ("openai", "local", "llamacpp", "openrouter")


def get_chat_model(
    model_name: str,
    provider: str,
    model_parameters: dict[str, Any] | None = None,
    *,
    max_retries: int = 4,
    request_timeout: float = 60.0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """
    Return a LangChain chat model with sensible retries/timeouts.

    provider:
      - "openai"           → ChatOpenAI (OpenAI API)
      - "openrouter"       → ChatOpenAI pointed at an OpenRouter server
      - "openai_compat"    → ChatOpenAI pointed at a custom base_url (e.g., llama.cpp server)
      - "local"/"llamacpp" → alias of openai_compat

    Notes:
      - For openai_compat, pass base_url like "http://localhost:8080/v1"
        and any api_key string (some servers ignore it but the client requires one).
      - The returned model itself will retry network/transient errors up to max_retries.
    """
    model_parameters = dict(model_parameters or {})
    provider = provider.lower()


    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            max_retries=max_retries,
            timeout=request_timeout,  # langchain-openai supports `timeout`
            **model_parameters,
        )  # type: ignore[no-any-return]

    if provider in ("openai_compat", "local", "llamacpp"):
        # Point ChatOpenAI at an OpenAI-compatible server (llama.cpp, vLLM, LM Studio, etc.)
        # Prefer explicit args; fall back to env vars.
        _base_url = base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:8080/"
        _api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-local"

        return ChatOpenAI(
            model=model_name,
            base_url=_base_url,  # works on recent langchain-openai
            api_key=_api_key,  # type: ignore
            max_retries=max_retries,
            timeout=request_timeout,
            **model_parameters,
        )  # type: ignore[no-any-return]

    if provider == "openrouter":
        _api_key = os.getenv("OPENROUTER_API_KEY")
        _base_url = os.getenv("OPENROUTER_BASE_URL")
        return ChatOpenAI(
            model=model_name,
            api_key=_api_key,
            base_url=_base_url,
            **model_parameters,
        )

    raise ValueError(f"Unsupported provider '{provider}'. " f"Supported providers are: {', '.join(_SUPPORTED)}")

    # If you want a universal outer retry, uncomment below (Tenacity via LCEL):
    # return cast(Runnable, llm).with_retry(
    #     stop_after_attempt=max_retries,
    #     wait_exponential_jitter=True,
    #     retry_if_exception_type=(HTTPError, TimeoutException),
    # )
