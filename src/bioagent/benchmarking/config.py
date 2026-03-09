from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    if isinstance(value, dict):
        return {str(key): _substitute_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    return value


def _string_list(raw_values: Any, *, field_name: str) -> list[str]:
    if raw_values is None:
        return []
    if not isinstance(raw_values, list):
        raise ValueError(f"Profile field '{field_name}' must be a list when provided.")
    return [str(item).strip() for item in raw_values if str(item).strip()]


def _tool_mode(raw_value: Any) -> str:
    value = str(raw_value or "all").strip().lower().replace("-", "_")
    if value not in {"all", "selective", "none"}:
        raise ValueError(f"Profile field 'run_policy.tool_mode' must be one of: all, selective, none. Got '{raw_value}'.")
    return value


@dataclass(frozen=True)
class ModelProfile:
    provider: str = "openrouter"
    model_name: str = "google/gemini-3-flash-preview"
    temperature: float = 0.0
    base_url: str | None = None
    api_key: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    def server_env(self) -> dict[str, str]:
        env = {
            "BIOAGENT_PROVIDER": self.provider,
            "BIOAGENT_MODEL": self.model_name,
            "BIOAGENT_TEMPERATURE": str(self.temperature),
        }
        if self.base_url:
            env["OPENAI_BASE_URL"] = self.base_url
        if self.api_key:
            if self.provider == "openrouter":
                env["OPENROUTER_API_KEY"] = self.api_key
            else:
                env["OPENAI_API_KEY"] = self.api_key
        env.update(self.extra_env)
        return env


@dataclass(frozen=True)
class BenchmarkServerConfig:
    base_url: str = "http://localhost:8000"
    assistant_id: str = "co_scientist"
    api_token: str | None = None
    user_id: str | None = None
    timeout_seconds: float = 180.0
    launch_command: list[str] = field(default_factory=list)
    launch_cwd: str | None = None
    launch_env: dict[str, str] = field(default_factory=dict)
    readiness_path: str = "/health"
    readiness_timeout_seconds: float = 90.0

    @property
    def launch_enabled(self) -> bool:
        return bool(self.launch_command)


@dataclass(frozen=True)
class BenchmarkRunPolicy:
    tool_mode: str = "all"
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    max_total_tool_calls: int | None = None
    max_tools_per_step: int | None = None
    retry_attempts: int = 0
    stream_tool_args: bool = False
    per_case_timeout_seconds: float = 180.0
    concurrency: int = 1


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    model: ModelProfile
    server: BenchmarkServerConfig
    run_policy: BenchmarkRunPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": {
                "provider": self.model.provider,
                "model_name": self.model.model_name,
                "temperature": self.model.temperature,
                "base_url": self.model.base_url,
                "extra_env": dict(self.model.extra_env),
            },
            "server": {
                "base_url": self.server.base_url,
                "assistant_id": self.server.assistant_id,
                "user_id": self.server.user_id,
                "timeout_seconds": self.server.timeout_seconds,
                "launch_command": list(self.server.launch_command),
                "launch_cwd": self.server.launch_cwd,
                "readiness_path": self.server.readiness_path,
                "readiness_timeout_seconds": self.server.readiness_timeout_seconds,
            },
            "run_policy": {
                "tool_mode": self.run_policy.tool_mode,
                "allowed_tools": list(self.run_policy.allowed_tools),
                "disallowed_tools": list(self.run_policy.disallowed_tools),
                "max_total_tool_calls": self.run_policy.max_total_tool_calls,
                "max_tools_per_step": self.run_policy.max_tools_per_step,
                "retry_attempts": self.run_policy.retry_attempts,
                "stream_tool_args": self.run_policy.stream_tool_args,
                "per_case_timeout_seconds": self.run_policy.per_case_timeout_seconds,
                "concurrency": self.run_policy.concurrency,
            },
        }


def _build_profile(name: str, payload: dict[str, Any]) -> BenchmarkProfile:
    payload = _substitute_env(payload)
    model_payload = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    server_payload = payload.get("server") if isinstance(payload.get("server"), dict) else {}
    run_payload = payload.get("run_policy") if isinstance(payload.get("run_policy"), dict) else {}

    model = ModelProfile(
        provider=str(model_payload.get("provider", "openrouter")).strip() or "openrouter",
        model_name=str(model_payload.get("model_name", "google/gemini-3-flash-preview")).strip()
        or "google/gemini-3-flash-preview",
        temperature=float(model_payload.get("temperature", 0.0)),
        base_url=str(model_payload["base_url"]).strip() if model_payload.get("base_url") else None,
        api_key=str(model_payload["api_key"]).strip() if model_payload.get("api_key") else None,
        extra_env={
            str(key): str(value)
            for key, value in (model_payload.get("extra_env", {}) or {}).items()
            if str(key).strip()
        },
    )
    server = BenchmarkServerConfig(
        base_url=str(server_payload.get("base_url", "http://localhost:8000")).rstrip("/"),
        assistant_id=str(server_payload.get("assistant_id", "co_scientist")).strip() or "co_scientist",
        api_token=str(server_payload["api_token"]).strip() if server_payload.get("api_token") else None,
        user_id=str(server_payload["user_id"]).strip() if server_payload.get("user_id") else None,
        timeout_seconds=float(server_payload.get("timeout_seconds", 180.0)),
        launch_command=_string_list(server_payload.get("launch_command"), field_name="server.launch_command"),
        launch_cwd=str(server_payload["launch_cwd"]).strip() if server_payload.get("launch_cwd") else None,
        launch_env={
            str(key): str(value)
            for key, value in (server_payload.get("launch_env", {}) or {}).items()
            if str(key).strip()
        },
        readiness_path=str(server_payload.get("readiness_path", "/health")).strip() or "/health",
        readiness_timeout_seconds=float(server_payload.get("readiness_timeout_seconds", 90.0)),
    )
    run_policy = BenchmarkRunPolicy(
        tool_mode=_tool_mode(run_payload.get("tool_mode", "all")),
        allowed_tools=_string_list(run_payload.get("allowed_tools"), field_name="run_policy.allowed_tools"),
        disallowed_tools=_string_list(run_payload.get("disallowed_tools"), field_name="run_policy.disallowed_tools"),
        max_total_tool_calls=int(run_payload["max_total_tool_calls"]) if run_payload.get("max_total_tool_calls") is not None else None,
        max_tools_per_step=int(run_payload["max_tools_per_step"]) if run_payload.get("max_tools_per_step") is not None else None,
        retry_attempts=max(0, int(run_payload.get("retry_attempts", 0))),
        stream_tool_args=bool(run_payload.get("stream_tool_args", False)),
        per_case_timeout_seconds=float(run_payload.get("per_case_timeout_seconds", server.timeout_seconds)),
        concurrency=max(1, int(run_payload.get("concurrency", 1))),
    )
    return BenchmarkProfile(name=name, model=model, server=server, run_policy=run_policy)


def load_profiles(path: str | Path) -> dict[str, BenchmarkProfile]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
        raise ValueError(f"Profile file '{config_path}' must contain a top-level 'profiles' mapping.")

    profiles: dict[str, BenchmarkProfile] = {}
    for name, item in payload["profiles"].items():
        if not isinstance(item, dict):
            raise ValueError(f"Profile '{name}' in '{config_path}' must be a mapping.")
        profiles[str(name)] = _build_profile(str(name), item)
    return profiles


def load_profile(path: str | Path, profile_name: str) -> BenchmarkProfile:
    profiles = load_profiles(path)
    try:
        return profiles[profile_name]
    except KeyError as exc:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown benchmark profile '{profile_name}'. Available profiles: {available}") from exc
