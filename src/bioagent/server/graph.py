from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, LLMToolSelectorMiddleware, TodoListMiddleware
from langchain.agents.middleware.tool_selection import _create_tool_selection_response
from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import AIMessage
from langgraph.channels.ephemeral_value import EphemeralValue

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
DEFAULT_SUMMARIZER_MODEL = "google/gemini-3.1-flash-lite-preview"
DEFAULT_LLM_TOOL_SELECTOR_MODEL = "google/gemini-3.1-flash-lite-preview"
ALLOWED_CHAT_MODELS = (
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "z-ai/glm-5",
    "moonshotai/kimi-k2.5",
    "minimax/minimax-m2.5",
    "qwen/qwen3.5-35b-a3b",
)
DEFAULT_CONTEXT_WINDOW_TOKENS = 200_000
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic/claude-opus-4.6": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "anthropic/claude-sonnet-4.6": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "anthropic/claude-haiku-4.5": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "google/gemini-3.1-pro-preview": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "google/gemini-3-flash-preview": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "google/gemini-3.1-flash-lite-preview": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "z-ai/glm-5": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "moonshotai/kimi-k2.5": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "minimax/minimax-m2.5": DEFAULT_CONTEXT_WINDOW_TOKENS,
    "qwen/qwen3.5-35b-a3b": DEFAULT_CONTEXT_WINDOW_TOKENS,
}
DEFAULT_CONTEXT_SUMMARY_TRIGGER_FRACTION = 0.8
DEFAULT_CONTEXT_SUMMARY_KEEP_FRACTION = 0.5
MAX_TOOLS_COUNT = 4


class BioAgentState(AgentState[Any]):
    """Agent state with explicit tool-discovery requests."""

    requested_tools: NotRequired[Annotated[list[str], EphemeralValue, PrivateStateAttr]]


def _env_true(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_primary_model_name() -> str:
    configured = os.getenv("BIOAGENT_MODEL", "anthropic/claude-sonnet-4.6").strip()
    if configured in ALLOWED_CHAT_MODELS:
        return configured
    return "anthropic/claude-sonnet-4.6"


def _resolve_model_context_window(model_name: str) -> int:
    return MODEL_CONTEXT_WINDOWS.get(model_name, DEFAULT_CONTEXT_WINDOW_TOKENS)


def _clamp_fraction(value: float, *, default: float) -> float:
    if not 0 < value <= 1:
        return default
    return value


def _context_summary_trigger_fraction() -> float:
    raw = os.getenv("BIOAGENT_CONTEXT_SUMMARY_TRIGGER_FRACTION", str(DEFAULT_CONTEXT_SUMMARY_TRIGGER_FRACTION)).strip()
    try:
        return _clamp_fraction(float(raw), default=DEFAULT_CONTEXT_SUMMARY_TRIGGER_FRACTION)
    except ValueError:
        return DEFAULT_CONTEXT_SUMMARY_TRIGGER_FRACTION


def _context_summary_keep_fraction() -> float:
    raw = os.getenv("BIOAGENT_CONTEXT_SUMMARY_KEEP_FRACTION", str(DEFAULT_CONTEXT_SUMMARY_KEEP_FRACTION)).strip()
    try:
        return _clamp_fraction(float(raw), default=DEFAULT_CONTEXT_SUMMARY_KEEP_FRACTION)
    except ValueError:
        return DEFAULT_CONTEXT_SUMMARY_KEEP_FRACTION


def _context_edit_keep_tool_results() -> int:
    raw = os.getenv("BIOAGENT_CONTEXT_EDIT_KEEP_TOOL_RESULTS", "3").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def _context_edit_clear_tool_inputs() -> bool:
    return _env_true("BIOAGENT_CONTEXT_EDIT_CLEAR_TOOL_INPUTS", default=False)


def _context_edit_token_count_method() -> str:
    raw = os.getenv("BIOAGENT_CONTEXT_EDIT_TOKEN_COUNT_METHOD", "approximate").strip().lower()
    return raw if raw in {"approximate", "model"} else "approximate"


def _runtime_provider() -> str:
    return os.getenv("BIOAGENT_PROVIDER", "openrouter")


def _runtime_temperature() -> float:
    return float(os.getenv("BIOAGENT_TEMPERATURE", "0.5"))


def _runtime_summarizer_temperature() -> float:
    return float(os.getenv("BIOAGENT_SUMMARIZER_TEMPERATURE", "0.2"))


def _runtime_summarizer_model_name() -> str:
    return os.getenv("BIOAGENT_SUMMARIZER_MODEL", DEFAULT_SUMMARIZER_MODEL)


def _build_summarizer_model() -> Any:
    return get_chat_model(
        _runtime_summarizer_model_name(),
        _runtime_provider(),
        model_parameters={"temperature": _runtime_summarizer_temperature()},
    )


def _emit_context_summary_event(
    *,
    message: str,
    model_name: str,
    context_window_tokens: int,
    trigger_tokens: int,
    keep_tokens: int,
) -> None:
    try:
        from datetime import datetime, timezone
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return

    try:
        writer(
            {
                "type": "context_updated",
                "reason": "conversation_summarized",
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "context_window_tokens": context_window_tokens,
                "trigger_tokens": trigger_tokens,
                "keep_tokens": keep_tokens,
                "summarized": True,
            }
        )
    except Exception:
        return


def _emit_context_edit_event(
    *,
    message: str,
    model_name: str,
    context_window_tokens: int,
    trigger_tokens: int,
    cleared_tool_results: int,
    cleared_tool_inputs: int,
    keep_recent_tool_results: int,
) -> None:
    try:
        from datetime import datetime, timezone
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return

    try:
        writer(
            {
                "type": "context_updated",
                "reason": "tool_results_cleared",
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "context_window_tokens": context_window_tokens,
                "trigger_tokens": trigger_tokens,
                "cleared_tool_results": cleared_tool_results,
                "cleared_tool_inputs": cleared_tool_inputs,
                "keep_recent_tool_results": keep_recent_tool_results,
            }
        )
    except Exception:
        return


def _message_context_editing_metadata(message: Any) -> dict[str, Any]:
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        metadata = response_metadata.get("context_editing")
        if isinstance(metadata, dict):
            return metadata
    return {}


def _count_cleared_tool_results(messages: list[Any]) -> int:
    return sum(1 for message in messages if _message_context_editing_metadata(message).get("cleared") is True)


def _count_cleared_tool_inputs(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        cleared_ids = _message_context_editing_metadata(message).get("cleared_tool_inputs")
        if isinstance(cleared_ids, list):
            total += len(cleared_ids)
    return total


class ReportedTokenAwareSummarizationMiddleware(AgentMiddleware):
    """Wrap LangChain summarization with provider-agnostic token checks."""

    def __init__(
        self,
        *,
        summarizer_model: Any,
        trigger_tokens: int,
        keep_tokens: int,
    ) -> None:
        super().__init__()
        try:
            from langchain.agents.middleware import SummarizationMiddleware
        except Exception as exc:  # pragma: no cover - dependency version mismatch
            raise RuntimeError("SummarizationMiddleware is unavailable in this LangChain version.") from exc

        class _DelegateSummarizationMiddleware(SummarizationMiddleware):
            def _should_summarize_based_on_reported_tokens(self, messages: list[Any], threshold: float) -> bool:
                last_ai_message = next(
                    (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
                    None,
                )
                if not isinstance(last_ai_message, AIMessage) or last_ai_message.usage_metadata is None:
                    return False
                usage_metadata = last_ai_message.usage_metadata
                reported_tokens = (
                    usage_metadata.get("input_tokens")
                    or usage_metadata.get("prompt_tokens")
                    or usage_metadata.get("total_tokens")
                    or -1
                )
                return isinstance(reported_tokens, (int, float)) and reported_tokens >= threshold

        self._delegate = _DelegateSummarizationMiddleware(
            model=summarizer_model,
            trigger=("tokens", max(1, int(trigger_tokens))),
            keep=("tokens", max(1, int(keep_tokens))),
        )

    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        return self._delegate.before_model(state, runtime)

    async def abefore_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        return await self._delegate.abefore_model(state, runtime)


class ContextWindowSummarizationMiddleware(AgentMiddleware):
    """Summarize conversation history near the selected model's context limit."""

    summary_notice = "Context was automatically summarized to stay within the model context window."

    def __init__(
        self,
        *,
        summarizer_model: Any,
        default_model_name: str,
        trigger_fraction: float,
        keep_fraction: float,
    ) -> None:
        super().__init__()
        self.summarizer_model = summarizer_model
        self.default_model_name = default_model_name
        self.trigger_fraction = trigger_fraction
        self.keep_fraction = keep_fraction

    def _requested_model_name(self) -> str:
        try:
            from langgraph.config import get_config

            cfg = get_config() or {}
            configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
            if isinstance(configurable, dict):
                for key in _MODEL_NAME_KEYS:
                    value = configurable.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            pass
        return self.default_model_name

    def _build_delegate(self) -> tuple[ReportedTokenAwareSummarizationMiddleware, str, int, int, int]:
        model_name = self._requested_model_name()
        context_window_tokens = _resolve_model_context_window(model_name)
        trigger_tokens = max(1, int(context_window_tokens * self.trigger_fraction))
        keep_tokens = max(1, int(context_window_tokens * self.keep_fraction))
        delegate = ReportedTokenAwareSummarizationMiddleware(
            summarizer_model=self.summarizer_model,
            trigger_tokens=trigger_tokens,
            keep_tokens=keep_tokens,
        )
        return delegate, model_name, context_window_tokens, trigger_tokens, keep_tokens

    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        delegate, model_name, context_window_tokens, trigger_tokens, keep_tokens = self._build_delegate()
        result = delegate.before_model(state, runtime)
        if result is not None:
            _emit_context_summary_event(
                message=self.summary_notice,
                model_name=model_name,
                context_window_tokens=context_window_tokens,
                trigger_tokens=trigger_tokens,
                keep_tokens=keep_tokens,
            )
        return result

    async def abefore_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        delegate, model_name, context_window_tokens, trigger_tokens, keep_tokens = self._build_delegate()
        result = await delegate.abefore_model(state, runtime)
        if result is not None:
            _emit_context_summary_event(
                message=self.summary_notice,
                model_name=model_name,
                context_window_tokens=context_window_tokens,
                trigger_tokens=trigger_tokens,
                keep_tokens=keep_tokens,
            )
        return result


class ContextWindowEditingMiddleware(AgentMiddleware):
    """Clear older tool results near the selected model's context limit."""

    notice = "Older tool results were automatically cleared to stay within the model context window."

    def __init__(
        self,
        *,
        default_model_name: str,
        trigger_fraction: float,
        keep_recent_tool_results: int,
        clear_tool_inputs: bool,
        token_count_method: str,
    ) -> None:
        super().__init__()
        self.default_model_name = default_model_name
        self.trigger_fraction = trigger_fraction
        self.keep_recent_tool_results = keep_recent_tool_results
        self.clear_tool_inputs = clear_tool_inputs
        self.token_count_method = token_count_method

        try:
            from langchain.agents.middleware import ContextEditingMiddleware
            from langchain.agents.middleware.context_editing import ClearToolUsesEdit
        except Exception as exc:  # pragma: no cover - dependency version mismatch
            raise RuntimeError("ContextEditingMiddleware is unavailable in this LangChain version.") from exc

        self._context_editing_cls = ContextEditingMiddleware
        self._clear_tool_uses_cls = ClearToolUsesEdit

    def _requested_model_name(self) -> str:
        try:
            from langgraph.config import get_config

            cfg = get_config() or {}
            configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
            if isinstance(configurable, dict):
                for key in _MODEL_NAME_KEYS:
                    value = configurable.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            pass
        return self.default_model_name

    def _build_delegate(self) -> tuple[Any, str, int, int]:
        model_name = self._requested_model_name()
        context_window_tokens = _resolve_model_context_window(model_name)
        trigger_tokens = max(1, int(context_window_tokens * self.trigger_fraction))
        edit = self._clear_tool_uses_cls(
            trigger=trigger_tokens,
            keep=self.keep_recent_tool_results,
            clear_tool_inputs=self.clear_tool_inputs,
        )
        delegate = self._context_editing_cls(
            edits=[edit],
            token_count_method=self.token_count_method,
        )
        return delegate, model_name, context_window_tokens, trigger_tokens

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> Any:
        delegate, model_name, context_window_tokens, trigger_tokens = self._build_delegate()
        original_messages = list(request.messages or [])
        captured_request: ModelRequest | None = None

        def instrumented_handler(next_request: ModelRequest) -> ModelResponse:
            nonlocal captured_request
            captured_request = next_request
            return handler(next_request)

        result = delegate.wrap_model_call(request, instrumented_handler)
        if captured_request is not None:
            edited_messages = list(captured_request.messages or [])
            cleared_tool_results = max(0, _count_cleared_tool_results(edited_messages) - _count_cleared_tool_results(original_messages))
            cleared_tool_inputs = max(0, _count_cleared_tool_inputs(edited_messages) - _count_cleared_tool_inputs(original_messages))
            if cleared_tool_results > 0 or cleared_tool_inputs > 0:
                _emit_context_edit_event(
                    message=self.notice,
                    model_name=model_name,
                    context_window_tokens=context_window_tokens,
                    trigger_tokens=trigger_tokens,
                    cleared_tool_results=cleared_tool_results,
                    cleared_tool_inputs=cleared_tool_inputs,
                    keep_recent_tool_results=self.keep_recent_tool_results,
                )
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        delegate, model_name, context_window_tokens, trigger_tokens = self._build_delegate()
        original_messages = list(request.messages or [])
        captured_request: ModelRequest | None = None

        async def instrumented_handler(next_request: ModelRequest) -> ModelResponse:
            nonlocal captured_request
            captured_request = next_request
            return await handler(next_request)

        result = await delegate.awrap_model_call(request, instrumented_handler)
        if captured_request is not None:
            edited_messages = list(captured_request.messages or [])
            cleared_tool_results = max(0, _count_cleared_tool_results(edited_messages) - _count_cleared_tool_results(original_messages))
            cleared_tool_inputs = max(0, _count_cleared_tool_inputs(edited_messages) - _count_cleared_tool_inputs(original_messages))
            if cleared_tool_results > 0 or cleared_tool_inputs > 0:
                _emit_context_edit_event(
                    message=self.notice,
                    model_name=model_name,
                    context_window_tokens=context_window_tokens,
                    trigger_tokens=trigger_tokens,
                    cleared_tool_results=cleared_tool_results,
                    cleared_tool_inputs=cleared_tool_inputs,
                    keep_recent_tool_results=self.keep_recent_tool_results,
                )
        return result


def _serialize_state_update(value: Any) -> Any:
    if isinstance(value, list):
        return [_serialize_state_update(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_state_update(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _serialize_state_update(value.model_dump())
    return value


def _materialize_state_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages

    try:
        from langchain_core.messages import convert_to_messages
    except Exception:
        return messages

    try:
        return list(convert_to_messages(messages))
    except Exception:
        return messages


def _materialize_state_values_for_middleware(state_values: dict[str, Any]) -> dict[str, Any]:
    values = dict(state_values)
    if "messages" in values:
        values["messages"] = _materialize_state_messages(values["messages"])
    return values


async def build_manual_context_summary_update(
    *,
    state_values: dict[str, Any],
    model_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    context_window_tokens = _resolve_model_context_window(model_name)
    keep_tokens = max(1, int(context_window_tokens * _context_summary_keep_fraction()))
    middleware = ReportedTokenAwareSummarizationMiddleware(
        summarizer_model=_build_summarizer_model(),
        trigger_tokens=1,
        keep_tokens=keep_tokens,
    )
    update = await middleware.abefore_model(_materialize_state_values_for_middleware(state_values), None)

    result: dict[str, Any] = {
        "ok": True,
        "model_name": model_name,
        "context_window_tokens": context_window_tokens,
        "trigger_tokens": 1,
        "keep_tokens": keep_tokens,
    }
    if not update:
        return None, {
            **result,
            "summarized": False,
            "message": "Context summary was not needed for the current thread state.",
        }
    return _serialize_state_update(update), {
        **result,
        "summarized": True,
        "message": ContextWindowSummarizationMiddleware.summary_notice,
    }


async def manually_summarize_thread_context(
    *,
    thread_id: str,
    user_id: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Force a conversation summary for the given thread when possible."""

    requested_model = (model_name or _resolve_primary_model_name()).strip() or _resolve_primary_model_name()
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "conversation_uuid": thread_id,
            "model_name": requested_model,
            **({"user_id": user_id} if user_id else {}),
        }
    }

    snapshot = await graph.aget_state(config)
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    update, result = await build_manual_context_summary_update(state_values=values, model_name=requested_model)
    if not update:
        return {**result, "thread_id": thread_id}

    await graph.aupdate_state(config, update, as_node="model")
    return {**result, "thread_id": thread_id}


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


class DiscoveryAwareLLMToolSelectorMiddleware(LLMToolSelectorMiddleware):
    """Extend LLM selection with user/agent-requested tools from state."""

    def __init__(self, **kwargs: Any) -> None:
        always_include = kwargs.get("always_include") or []
        self._base_always_include: tuple[str, ...] = tuple(str(name) for name in always_include)
        super().__init__(**kwargs)

    @staticmethod
    def _extract_requested_tools(state: Any) -> list[str]:
        if not isinstance(state, dict):
            return []
        raw_requested = state.get("requested_tools")
        if not isinstance(raw_requested, list):
            return []

        requested: list[str] = []
        for raw_name in raw_requested:
            name = str(raw_name).strip()
            if name and name not in requested:
                requested.append(name)
        return requested

    def _prepare_selection_request(self, request: ModelRequest) -> Any:
        requested = self._extract_requested_tools(getattr(request, "state", None))
        merged = list(dict.fromkeys([*self._base_always_include, *requested]))
        previous = list(self.always_include)
        self.always_include = merged
        try:
            return super()._prepare_selection_request(request)
        finally:
            self.always_include = previous

    def _process_selection_response(
        self,
        response: dict[str, Any],
        available_tools: list[Any],
        valid_tool_names: list[str],
        request: ModelRequest,
    ) -> ModelRequest:
        """Ignore invalid selector outputs and fall back gracefully."""

        raw_tool_names = response.get("tools", [])
        if not isinstance(raw_tool_names, list):
            return request

        selected_tool_names: list[str] = []
        invalid_tool_selections: list[str] = []

        for raw_name in raw_tool_names:
            tool_name = str(raw_name).strip()
            if not tool_name:
                continue
            if tool_name not in valid_tool_names:
                invalid_tool_selections.append(tool_name)
                continue
            if tool_name not in selected_tool_names and (
                self.max_tools is None or len(selected_tool_names) < self.max_tools
            ):
                selected_tool_names.append(tool_name)

        if invalid_tool_selections and not selected_tool_names:
            return request

        selected_tools = [
            tool for tool in available_tools if getattr(tool, "name", None) in selected_tool_names
        ]
        always_included_tools = [
            tool
            for tool in request.tools
            if not isinstance(tool, dict) and getattr(tool, "name", None) in self.always_include
        ]
        provider_tools = [tool for tool in request.tools if isinstance(tool, dict)]
        return request.override(tools=[*selected_tools, *always_included_tools, *provider_tools])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> Any:
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return handler(request)

        try:
            type_adapter = _create_tool_selection_response(selection_request.available_tools)
            schema = type_adapter.json_schema()
            structured_model = selection_request.model.with_structured_output(schema).with_config(
                {"callbacks": [], "run_name": "tool_selector_internal", "tags": ["tool_selector_internal"]}
            )
            response = structured_model.invoke(
                [
                    {"role": "system", "content": selection_request.system_message},
                    selection_request.last_user_message,
                ],
                config={"callbacks": [], "run_name": "tool_selector_internal", "tags": ["tool_selector_internal"]},
            )
            if not isinstance(response, dict):
                return handler(request)
            modified_request = self._process_selection_response(
                response,
                selection_request.available_tools,
                selection_request.valid_tool_names,
                request,
            )
            return handler(modified_request)
        except Exception:
            return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return await handler(request)

        try:
            type_adapter = _create_tool_selection_response(selection_request.available_tools)
            schema = type_adapter.json_schema()
            structured_model = selection_request.model.with_structured_output(schema).with_config(
                {"callbacks": [], "run_name": "tool_selector_internal", "tags": ["tool_selector_internal"]}
            )
            response = await structured_model.ainvoke(
                [
                    {"role": "system", "content": selection_request.system_message},
                    selection_request.last_user_message,
                ],
                config={"callbacks": [], "run_name": "tool_selector_internal", "tags": ["tool_selector_internal"]},
            )
            if not isinstance(response, dict):
                return await handler(request)
            modified_request = self._process_selection_response(
                response,
                selection_request.available_tools,
                selection_request.valid_tool_names,
                request,
            )
            return await handler(modified_request)
        except Exception:
            return await handler(request)


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


def _build_middleware(*, provider: str, model_name: str, temperature: float, summarizer: Any) -> list[Any]:
    middleware: list[Any] = [
        RuntimeModelSelectionMiddleware(
            provider=provider,
            default_model_name=model_name,
            model_parameters={"temperature": temperature},
            allowed_models=set(ALLOWED_CHAT_MODELS),
        ),
        ContextWindowSummarizationMiddleware(
            summarizer_model=summarizer,
            default_model_name=model_name,
            trigger_fraction=_context_summary_trigger_fraction(),
            keep_fraction=_context_summary_keep_fraction(),
        ),
        ContextWindowEditingMiddleware(
            default_model_name=model_name,
            trigger_fraction=_context_summary_trigger_fraction(),
            keep_recent_tool_results=_context_edit_keep_tool_results(),
            clear_tool_inputs=_context_edit_clear_tool_inputs(),
            token_count_method=_context_edit_token_count_method(),
        ),
    ]
    try:
        from langchain.agents.middleware import ModelRetryMiddleware

        middleware.append(
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
        )
    except Exception:
        pass  # retry middleware optional; continue without it

    llm_tool_selector_model = os.getenv("BIOAGENT_LLM_TOOL_SELECTOR_MODEL", DEFAULT_LLM_TOOL_SELECTOR_MODEL)
    llm_tool_selector = get_chat_model(llm_tool_selector_model, provider)
    middleware.append(
        DiscoveryAwareLLMToolSelectorMiddleware(
            model=llm_tool_selector,
            max_tools=MAX_TOOLS_COUNT,
            always_include=[
                "think",
                "list_available_tools",
                "request_tools",
                "tavily_search",
                "scrape_url_content",
                "web_search",
            ],
        ),
    )
    middleware.append(TodoListMiddleware())
    return middleware


def _build_agent() -> Any:
    model_name = _resolve_primary_model_name()
    summarizer_model_name = os.getenv("BIOAGENT_SUMMARIZER_MODEL", DEFAULT_SUMMARIZER_MODEL)
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
        "state_schema": BioAgentState,
        "debug": os.getenv("BIOAGENT_AGENT_DEBUG", "false").lower() == "true",
        "middleware": _build_middleware(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            summarizer=summarizer,
        ),
    }

    checkpointer = _build_checkpointer()
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    return create_agent(**kwargs)


graph = _build_agent()
