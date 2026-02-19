import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from bioagent.agent import get_chat_model
from bioagent.persistence import (
    delete_report,
    get_report,
    get_report_content,
    list_prompt_click_popularity,
    list_reports,
    normalize_report_id,
    persist_prompt_click,
)


class ApiErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ApiErrorBody


class HealthResponse(BaseModel):
    ok: bool = True
    version: str = "v1"


class MeResponse(BaseModel):
    user_id: str
    auth_required: bool


class ReportMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    ref_id: str | None = None
    filename: str | None = None
    tool_name: str | None = None
    status: str | None = None
    size_chars: int | None = None
    one_line: str | None = None
    display_name: str | None = None
    thread_id: str | None = None
    user_id: str | None = None
    created_at: str | None = None
    path: str | None = None


class ThreadDisplayNameMessage(BaseModel):
    role: str
    content: str


class GenerateThreadDisplayNameRequest(BaseModel):
    messages: list[ThreadDisplayNameMessage] = Field(default_factory=list)
    min_messages: int = Field(default=2, ge=1, le=20)
    max_messages: int = Field(default=2, ge=1, le=20)
    force: bool = False


class GenerateThreadDisplayNameResponse(BaseModel):
    thread_id: str
    display_name: str
    generated: bool


class ReportsListResponse(BaseModel):
    items: list[ReportMetadata]
    total: int
    limit: int
    offset: int
    has_more: bool


class ReportContentResponse(BaseModel):
    id: str
    filename: str
    content: str
    total_chars: int
    returned_chars: int
    offset: int
    max_chars: int
    truncated: bool


class DeleteReportResponse(BaseModel):
    ok: bool
    deleted: str


class LegacyContentResponse(BaseModel):
    id: str
    filename: str
    content: str


class PromptClickRequest(BaseModel):
    prompt_text: str = Field(min_length=1, max_length=4000)
    category_id: str | None = Field(default=None, max_length=120)
    category_title: str | None = Field(default=None, max_length=200)
    thread_id: str | None = Field(default=None, max_length=200)
    source: str | None = Field(default="web_starter_prompt", max_length=120)


class PromptClickResponse(BaseModel):
    ok: bool
    click_id: int
    clicked_at: str


class PromptPopularityItem(BaseModel):
    category_id: str | None = None
    category_title: str | None = None
    prompt_text: str
    click_count: int
    last_clicked_at: str


class PromptPopularityResponse(BaseModel):
    items: list[PromptPopularityItem]
    limit: int
    days: int
    scope: str


class AuthSettings:
    def __init__(self, token_map: dict[str, str], required: bool, default_user_id: str) -> None:
        self.token_map = token_map
        self.required = required
        self.default_user_id = default_user_id

    @classmethod
    def from_env(cls) -> "AuthSettings":
        default_user_id = os.getenv("BIOAGENT_DEFAULT_USER_ID", "anonymous").strip() or "anonymous"
        token_map: dict[str, str] = {}

        raw_tokens = os.getenv("BIOAGENT_API_TOKENS", "").strip()
        if raw_tokens:
            try:
                payload = json.loads(raw_tokens)
                if isinstance(payload, dict):
                    token_map = {
                        str(token).strip(): str(user).strip() or default_user_id
                        for token, user in payload.items()
                        if str(token).strip()
                    }
            except json.JSONDecodeError:
                for item in raw_tokens.split(","):
                    token, sep, user = item.partition(":")
                    token = token.strip()
                    user = user.strip()
                    if token and sep:
                        token_map[token] = user or default_user_id

        single_token = os.getenv("BIOAGENT_API_TOKEN", "").strip()
        if single_token:
            token_map[single_token] = os.getenv("BIOAGENT_API_USER_ID", default_user_id).strip() or default_user_id

        auth_required_env = os.getenv("BIOAGENT_AUTH_REQUIRED")
        if auth_required_env is None:
            required = bool(token_map)
        else:
            required = auth_required_env.strip().lower() in {"1", "true", "yes", "on"}

        return cls(token_map=token_map, required=required, default_user_id=default_user_id)


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def _request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    return str(request_id) if request_id else "unknown_request"


def _error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    payload = ErrorEnvelope(
        error=ApiErrorBody(
            code=code,
            message=message,
            details=details,
            request_id=_request_id(request),
        )
    )
    response = JSONResponse(status_code=status_code, content=payload.model_dump())
    response.headers["X-Request-ID"] = _request_id(request)
    return response


def _auth_user_id(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if isinstance(user_id, str) and user_id.strip():
        return user_id.strip()
    raise ApiError(
        status_code=status.HTTP_401_UNAUTHORIZED,
        code="unauthorized",
        message="Missing authenticated user context.",
    )


def _as_report_metadata(item: dict[str, Any]) -> ReportMetadata:
    return ReportMetadata.model_validate(item)


def _validated_report_id(report_id: str) -> str:
    safe_report_id = normalize_report_id(report_id)
    if not safe_report_id:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="invalid_report_id",
            message=(
                "Invalid report_id format. Allowed: letters, digits, underscores, dashes, dots; "
                "must start with a letter or digit."
            ),
        )
    return safe_report_id


def _normalize_limit(
    limit: int | None,
    *,
    default_value: int,
    max_value: int,
) -> int:
    normalized = default_value if limit is None else int(limit)
    if normalized < 1:
        normalized = 1
    if normalized > max_value:
        normalized = max_value
    return normalized


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


THREAD_DISPLAY_NAME_MODEL = "google/gemini-3-flash-preview"
THREAD_DISPLAY_NAME_PROMPT = PromptTemplate.from_template(
    """
You write concise biomedical conversation titles.

Conversation snippets:
{conversation}

Return only a short title (max 8 words), no quotes, no punctuation at the end.
"""
)


def _thread_name_fallback(messages: list[dict[str, str]]) -> str:
    for item in messages:
        if item.get("role") == "user":
            text = item.get("content", "").strip()
            if text:
                words = text.split()
                return " ".join(words[:8]).strip()
    return "Biomedical research session"


def _sanitize_title(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = text.strip("`\"' ")
    text = text.rstrip(".,;:!?")
    return text[:80]


async def _generate_thread_display_name(messages: list[dict[str, str]]) -> str:
    date_prefix = datetime.now().strftime("%d-%m_%Y %H-%M")
    provider = os.getenv("BIOAGENT_PROVIDER", "openrouter")

    formatted_lines = [
        f"{item['role']}: {item['content']}"
        for item in messages
        if item.get("content")
    ]
    conversation = "\n".join(formatted_lines)[:6000]

    try:
        llm = get_chat_model(
            THREAD_DISPLAY_NAME_MODEL,
            provider,
            model_parameters={"temperature": 0.2},
        )
        chain = THREAD_DISPLAY_NAME_PROMPT | llm | StrOutputParser()
        generated = await chain.ainvoke({"conversation": conversation})
        title = _sanitize_title(str(generated))
    except Exception:
        title = _sanitize_title(_thread_name_fallback(messages))

    if not title:
        title = "Biomedical research session"
    return f"{date_prefix} {title}".strip()[:120]


def create_webapp() -> FastAPI:
    app = FastAPI(
        title="AdvancedBiomedicalAgent Custom API",
        version="v1",
    )
    app.state.auth_settings = AuthSettings.from_env()
    app.state.reports_default_limit = int(os.getenv("BIOAGENT_REPORTS_DEFAULT_LIMIT", "20"))
    app.state.reports_max_limit = int(os.getenv("BIOAGENT_REPORTS_MAX_LIMIT", "100"))
    app.state.reports_max_fetch = int(os.getenv("BIOAGENT_REPORTS_MAX_FETCH", "2000"))
    app.state.report_content_default_chars = int(os.getenv("BIOAGENT_REPORT_CONTENT_DEFAULT_CHARS", "12000"))
    app.state.report_content_max_chars = int(os.getenv("BIOAGENT_REPORT_CONTENT_MAX_CHARS", "100000"))
    app.state.trust_x_user_id = _env_flag("BIOAGENT_TRUST_X_USER_ID", True)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id

        public_paths = {"/ok", "/v1/ok", "/health", "/ready", "/live", "/docs", "/openapi.json", "/redoc"}
        settings: AuthSettings = request.app.state.auth_settings

        if request.url.path in public_paths:
            request.state.user_id = settings.default_user_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer ") :].strip()

        resolved_user: str | None = None
        if token and settings.token_map:
            resolved_user = settings.token_map.get(token)
            if not resolved_user:
                return _error_response(
                    request=request,
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    code="unauthorized",
                    message="Invalid API token.",
                )
        elif settings.required:
            return _error_response(
                request=request,
                status_code=status.HTTP_401_UNAUTHORIZED,
                code="unauthorized",
                message="Missing Authorization: Bearer <token> header.",
            )
        else:
            resolved_user = settings.default_user_id

        if request.app.state.trust_x_user_id:
            header_user = request.headers.get("X-Bioagent-User-Id", "").strip()
            if header_user:
                resolved_user = header_user

        request.state.user_id = resolved_user or settings.default_user_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=exc.status_code,
            code="http_error",
            message=str(exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="validation_error",
            message="Request validation failed.",
            details=exc.errors(),
        )

    @app.get("/ok", response_model=HealthResponse, tags=["system"])
    async def ok() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/ok", response_model=HealthResponse, tags=["system"])
    async def ok_v1() -> HealthResponse:
        return HealthResponse()

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=HealthResponse, tags=["system"])
    async def ready() -> HealthResponse:
        return HealthResponse()

    @app.get("/live", response_model=HealthResponse, tags=["system"])
    async def live() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/me", response_model=MeResponse, tags=["auth"])
    async def me(request: Request) -> MeResponse:
        settings: AuthSettings = request.app.state.auth_settings
        return MeResponse(user_id=_auth_user_id(request), auth_required=settings.required)

    @app.post("/v1/analytics/prompt-clicks", response_model=PromptClickResponse, tags=["analytics"])
    async def log_prompt_click(payload: PromptClickRequest, request: Request) -> PromptClickResponse:
        user_id = _auth_user_id(request)
        try:
            record = await persist_prompt_click(
                user_id=user_id,
                prompt_text=payload.prompt_text,
                category_id=payload.category_id,
                category_title=payload.category_title,
                thread_id=payload.thread_id,
                source=payload.source,
            )
        except ValueError as exc:
            raise ApiError(
                status_code=status.HTTP_400_BAD_REQUEST,
                code="invalid_prompt_click",
                message=str(exc),
            ) from exc

        return PromptClickResponse(
            ok=True,
            click_id=int(record["id"]),
            clicked_at=str(record["clicked_at"]),
        )

    @app.get("/v1/analytics/prompt-clicks/top", response_model=PromptPopularityResponse, tags=["analytics"])
    async def top_prompt_clicks(
        request: Request,
        limit: int = Query(default=20, ge=1, le=200),
        days: int = Query(default=90, ge=1, le=3650),
        scope: str = Query(default="global"),
    ) -> PromptPopularityResponse:
        user_id = _auth_user_id(request)
        normalized_scope = scope.strip().lower()
        if normalized_scope not in {"global", "me"}:
            raise ApiError(
                status_code=status.HTTP_400_BAD_REQUEST,
                code="invalid_scope",
                message="scope must be 'global' or 'me'.",
            )

        rows = await list_prompt_click_popularity(
            limit=limit,
            days=days,
            user_id=user_id if normalized_scope == "me" else None,
        )
        return PromptPopularityResponse(
            items=[PromptPopularityItem.model_validate(item) for item in rows],
            limit=limit,
            days=days,
            scope=normalized_scope,
        )

    @app.post("/v1/threads/{thread_id}/display-name/generate", response_model=GenerateThreadDisplayNameResponse, tags=["threads"])
    async def generate_thread_display_name(
        thread_id: str,
        payload: GenerateThreadDisplayNameRequest,
        request: Request,
    ) -> GenerateThreadDisplayNameResponse:
        _auth_user_id(request)

        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ApiError(
                status_code=status.HTTP_400_BAD_REQUEST,
                code="invalid_thread_id",
                message="thread_id cannot be empty.",
            )

        normalized_messages = [
            {"role": item.role.strip().lower()[:20], "content": item.content.strip()}
            for item in payload.messages
            if item.content.strip()
        ]
        message_subset = normalized_messages[:2]

        if len(message_subset) < payload.min_messages and not payload.force:
            fallback = await asyncio.to_thread(_thread_name_fallback, message_subset)
            date_prefix = datetime.now().strftime("%d-%m_%Y %H-%M")
            return GenerateThreadDisplayNameResponse(
                thread_id=normalized_thread_id,
                display_name=f"{date_prefix} {fallback}".strip()[:120],
                generated=False,
            )

        display_name = await _generate_thread_display_name(message_subset)
        return GenerateThreadDisplayNameResponse(
            thread_id=normalized_thread_id,
            display_name=display_name,
            generated=True,
        )

    @app.get("/v1/reports", response_model=ReportsListResponse, tags=["reports"])
    async def list_reports_v1(
        request: Request,
        thread_id: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1),
        offset: int = Query(default=0, ge=0),
    ) -> ReportsListResponse:
        user_id = _auth_user_id(request)
        reports_default_limit = int(request.app.state.reports_default_limit)
        reports_max_limit = int(request.app.state.reports_max_limit)
        reports_max_fetch = int(request.app.state.reports_max_fetch)

        normalized_limit = _normalize_limit(limit, default_value=reports_default_limit, max_value=reports_max_limit)
        fetch_cap = min(reports_max_fetch, max(normalized_limit + offset, reports_default_limit))
        raw_items = await list_reports(
            user_id=user_id,
            thread_id=thread_id,
            limit=fetch_cap,
        )

        total = len(raw_items)
        sliced = raw_items[offset : offset + normalized_limit]
        items = [_as_report_metadata(item) for item in sliced]
        return ReportsListResponse(
            items=items,
            total=total,
            limit=normalized_limit,
            offset=offset,
            has_more=(offset + len(items)) < total,
        )

    @app.get("/v1/reports/{report_id}", response_model=ReportMetadata, tags=["reports"])
    async def get_report_v1(report_id: str, request: Request) -> ReportMetadata:
        safe_report_id = _validated_report_id(report_id)
        user_id = _auth_user_id(request)
        report = await get_report(
            user_id=user_id,
            report_id=safe_report_id,
        )
        if not report:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="report_not_found",
                message=f"Report '{safe_report_id}' not found.",
            )
        return _as_report_metadata(report)

    @app.get("/v1/reports/{report_id}/content", response_model=ReportContentResponse, tags=["reports"])
    async def get_report_content_v1(
        report_id: str,
        request: Request,
        offset: int = Query(default=0, ge=0),
        max_chars: int | None = Query(default=None, ge=1),
    ) -> ReportContentResponse:
        safe_report_id = _validated_report_id(report_id)
        user_id = _auth_user_id(request)
        report = await get_report(
            user_id=user_id,
            report_id=safe_report_id,
        )
        if not report:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="report_not_found",
                message=f"Report '{safe_report_id}' not found.",
            )

        content = await get_report_content(
            user_id=user_id,
            report_id=safe_report_id,
        )
        if content is None:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="report_content_not_found",
                message=f"Report '{safe_report_id}' content not found.",
            )

        content_default_limit = int(request.app.state.report_content_default_chars)
        content_max_limit = int(request.app.state.report_content_max_chars)
        normalized_chars = _normalize_limit(
            max_chars,
            default_value=content_default_limit,
            max_value=content_max_limit,
        )

        total_chars = len(content)
        snippet = content[offset : offset + normalized_chars] if offset < total_chars else ""
        return ReportContentResponse(
            id=safe_report_id,
            filename=str(report.get("filename", f"{safe_report_id}.md")),
            content=snippet,
            total_chars=total_chars,
            returned_chars=len(snippet),
            offset=offset,
            max_chars=normalized_chars,
            truncated=(offset + len(snippet)) < total_chars,
        )

    @app.delete("/v1/reports/{report_id}", response_model=DeleteReportResponse, tags=["reports"])
    async def delete_report_v1(report_id: str, request: Request) -> DeleteReportResponse:
        safe_report_id = _validated_report_id(report_id)
        user_id = _auth_user_id(request)
        deleted = await delete_report(
            user_id=user_id,
            report_id=safe_report_id,
        )
        if not deleted:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="report_not_found",
                message=f"Report '{safe_report_id}' not found.",
            )
        return DeleteReportResponse(ok=True, deleted=safe_report_id)

    # Backward-compatible routes (deprecated): enforce authenticated user ownership.
    @app.get("/users/{user_id}/reports", tags=["reports", "legacy"])
    async def get_user_reports_legacy(
        user_id: str,
        request: Request,
        thread_id: str | None = None,
    ) -> list[dict[str, Any]]:
        auth_user = _auth_user_id(request)
        if user_id != auth_user:
            raise ApiError(
                status_code=status.HTTP_403_FORBIDDEN,
                code="forbidden_user_scope",
                message="Path user_id does not match authenticated user.",
            )
        reports = await list_reports(
            user_id=auth_user,
            thread_id=thread_id,
        )
        return reports

    @app.get("/users/{user_id}/reports/{report_id}", tags=["reports", "legacy"])
    async def get_user_report_legacy(user_id: str, report_id: str, request: Request) -> dict[str, Any]:
        safe_report_id = _validated_report_id(report_id)
        auth_user = _auth_user_id(request)
        if user_id != auth_user:
            raise ApiError(
                status_code=status.HTTP_403_FORBIDDEN,
                code="forbidden_user_scope",
                message="Path user_id does not match authenticated user.",
            )
        report = await get_report(
            user_id=auth_user,
            report_id=safe_report_id,
        )
        if not report:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="report_not_found",
                message=f"Report '{safe_report_id}' not found.",
            )
        return report

    @app.get("/users/{user_id}/reports/{report_id}/content", response_model=LegacyContentResponse, tags=["reports", "legacy"])
    async def get_user_report_content_legacy(user_id: str, report_id: str, request: Request) -> LegacyContentResponse:
        safe_report_id = _validated_report_id(report_id)
        auth_user = _auth_user_id(request)
        if user_id != auth_user:
            raise ApiError(
                status_code=status.HTTP_403_FORBIDDEN,
                code="forbidden_user_scope",
                message="Path user_id does not match authenticated user.",
            )

        report = await get_report(
            user_id=auth_user,
            report_id=safe_report_id,
        )
        if not report:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="report_not_found",
                message=f"Report '{safe_report_id}' not found.",
            )
        content = await get_report_content(
            user_id=auth_user,
            report_id=safe_report_id,
        )
        if content is None:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="report_content_not_found",
                message=f"Report '{safe_report_id}' content not found.",
            )
        return LegacyContentResponse(
            id=safe_report_id,
            filename=str(report.get("filename", f"{safe_report_id}.md")),
            content=content,
        )

    @app.delete("/users/{user_id}/reports/{report_id}", response_model=DeleteReportResponse, tags=["reports", "legacy"])
    async def delete_user_report_legacy(user_id: str, report_id: str, request: Request) -> DeleteReportResponse:
        safe_report_id = _validated_report_id(report_id)
        auth_user = _auth_user_id(request)
        if user_id != auth_user:
            raise ApiError(
                status_code=status.HTTP_403_FORBIDDEN,
                code="forbidden_user_scope",
                message="Path user_id does not match authenticated user.",
            )
        deleted = await delete_report(
            user_id=auth_user,
            report_id=safe_report_id,
        )
        if not deleted:
            raise ApiError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="report_not_found",
                message=f"Report '{safe_report_id}' not found.",
            )
        return DeleteReportResponse(ok=True, deleted=safe_report_id)

    return app


app = create_webapp()
