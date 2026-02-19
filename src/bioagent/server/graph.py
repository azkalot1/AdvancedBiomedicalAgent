from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from bioagent.agent import get_chat_model
from bioagent.agent.tools import get_summarized_tools, think, with_tool_status_streaming

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel

SYSTEM_PROMPT = """You are a biomedical co-scientist.

Your task is to answer users' questions by planning steps, using available tools,
and synthesizing evidence into clear, actionable outputs.

When tool outputs are long, they may be summarized and stored as reports.
Use:
- `list_research_outputs()` to browse what is already available.
- `retrieve_full_output(reference_id=...)` when full details are needed.
"""

_MODEL_NAME_KEYS = ("model", "model_name", "llm_model")
DEFAULT_SUMMARIZER_MODEL = "google/gemini-3-flash-preview"
ALLOWED_CHAT_MODELS = (
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "moonshotai/kimi-k2.5",
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview",
    "openai/gpt-5.2",
)


def _env_true(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_primary_model_name() -> str:
    configured = os.getenv("BIOAGENT_MODEL", "google/gemini-3-flash-preview").strip()
    if configured in ALLOWED_CHAT_MODELS:
        return configured
    return "google/gemini-3-flash-preview"


class RuntimeModelSelectionMiddleware(AgentMiddleware):
    """Switch the agent chat model per request from config.configurable.model(_name)."""

    def __init__(
        self,
        *,
        provider: str,
        default_model_name: str,
        model_parameters: dict[str, Any],
        allowed_models: set[str],
    ) -> None:
        super().__init__()
        self.provider = provider
        self.default_model_name = default_model_name
        self.model_parameters = dict(model_parameters)
        self.allowed_models = set(allowed_models)
        self._cache: dict[str, BaseChatModel] = {}
        self._cache[default_model_name] = get_chat_model(
            default_model_name,
            provider,
            model_parameters=self.model_parameters,
        )

    def _extract_requested_model(self, request: ModelRequest) -> str | None:
        requested: str | None = None
        try:
            from langgraph.config import get_config

            cfg = get_config() or {}
            configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
            if isinstance(configurable, dict):
                for key in _MODEL_NAME_KEYS:
                    value = configurable.get(key)
                    if isinstance(value, str) and value.strip():
                        requested = value.strip()
                        break
        except Exception:
            requested = None

        if requested:
            return requested

        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        if isinstance(context, dict):
            for key in _MODEL_NAME_KEYS:
                value = context.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _resolve_model(self, model_name: str) -> BaseChatModel | None:
        if model_name not in self.allowed_models:
            return None
        cached = self._cache.get(model_name)
        if cached is not None:
            return cached
        try:
            model = get_chat_model(
                model_name,
                self.provider,
                model_parameters=self.model_parameters,
            )
        except Exception:
            return None
        self._cache[model_name] = model
        return model

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        requested = self._extract_requested_model(request)
        if not requested or requested == self.default_model_name:
            return handler(request)
        model = self._resolve_model(requested)
        if model is None:
            return handler(request)
        return handler(request.override(model=model))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        requested = self._extract_requested_model(request)
        if not requested or requested == self.default_model_name:
            return await handler(request)
        model = self._resolve_model(requested)
        if model is None:
            return await handler(request)
        return await handler(request.override(model=model))


def _build_checkpointer() -> Any | None:
    # Aegra provides platform-managed checkpointing/store. Keep it as the
    # default there to avoid double-initializing a second checkpointer.
    aegra_detected = bool((os.getenv("AEGRA_POSTGRES_URI") or os.getenv("AEGRA_REDIS_URI") or "").strip())
    if aegra_detected and not _env_true("BIOAGENT_FORCE_INTERNAL_CHECKPOINTER", default=False):
        print("[bioagent] Checkpointer: disabled (managed by Aegra)")
        return None

    checkpoint_path = (
        os.getenv("BIOAGENT_CHECKPOINT_DB", ".bioagent/checkpoints.db")
        or ".bioagent/checkpoints.db"
    ).strip()
    checkpoint_db = Path(checkpoint_path)
    if checkpoint_db != Path(":memory:"):
        checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        print(f"[bioagent] Checkpointer: SQLite ({checkpoint_db})")
        return SqliteSaver.from_conn_string(str(checkpoint_db))
    except Exception as exc:
        print(f"[bioagent] Warning: could not initialize SQLite checkpointer, falling back to in-memory: {exc}")

    try:
        from langgraph.checkpoint.memory import InMemorySaver

        print("[bioagent] Checkpointer: InMemory")
        return InMemorySaver()
    except Exception:
        print("[bioagent] Checkpointer: disabled (no SQLite/InMemory backend available)")
        return None


def _build_middleware(*, provider: str, model_name: str, temperature: float) -> list[Any]:
    try:
        from langchain.agents.middleware import ModelRetryMiddleware

        return [
            RuntimeModelSelectionMiddleware(
                provider=provider,
                default_model_name=model_name,
                model_parameters={"temperature": temperature},
                allowed_models=set(ALLOWED_CHAT_MODELS),
            ),
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            )
        ]
    except Exception:
        return []


def _build_agent() -> Any:
    model_name = _resolve_primary_model_name()
    summarizer_model_name = DEFAULT_SUMMARIZER_MODEL
    provider = os.getenv("BIOAGENT_PROVIDER", "openrouter")
    temperature = float(os.getenv("BIOAGENT_TEMPERATURE", "0.5"))

    model = get_chat_model(
        model_name,
        provider,
        model_parameters={"temperature": temperature},
    )
    summarizer = get_chat_model(
        summarizer_model_name,
        provider,
        model_parameters={"temperature": float(os.getenv("BIOAGENT_SUMMARIZER_TEMPERATURE", "0.2"))},
    )

    tools = [with_tool_status_streaming(think)] + get_summarized_tools(summarizer)
    kwargs: dict[str, Any] = {
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "tools": tools,
        "debug": os.getenv("BIOAGENT_AGENT_DEBUG", "false").lower() == "true",
        "middleware": _build_middleware(provider=provider, model_name=model_name, temperature=temperature),
    }

    checkpointer = _build_checkpointer()
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    return create_agent(**kwargs)


graph = _build_agent()
