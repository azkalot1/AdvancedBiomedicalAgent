# thinking.py
from __future__ import annotations

from contextvars import ContextVar
from typing import Annotated
from functools import wraps
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])



# Guardrail: track if agent has thought before acting
_thinking_used: ContextVar[bool] = ContextVar("thinking_used", default=False)


def mark_thinking_used() -> None:
    _thinking_used.set(True)


def has_thinking_been_used() -> bool:
    return _thinking_used.get()


def reset_thinking_tracker() -> None:
    _thinking_used.set(False)


def require_thinking(fn: F) -> F:
    """Decorator: raises if agent hasn't called 'think' first."""
    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not has_thinking_been_used():
            raise RuntimeError(
                "Please use the 'think' tool first to plan your approach."
            )
        return await fn(*args, **kwargs)
    return cast(F, wrapper)