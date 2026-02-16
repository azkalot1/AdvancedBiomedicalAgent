import json
import os
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from bioagent.persistence import delete_report, get_report, get_report_content, list_reports, normalize_report_id


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
    thread_id: str | None = None
    user_id: str | None = None
    created_at: str | None = None
    path: str | None = None


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


def _get_store(request: Request) -> Any | None:
    request_state = getattr(request, "state", None)
    app_state = getattr(request.app, "state", None)
    return (
        getattr(request_state, "store", None)
        or getattr(app_state, "store", None)
        or getattr(app_state, "langgraph_store", None)
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

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id

        public_paths = {"/ok", "/v1/ok", "/docs", "/openapi.json", "/redoc"}
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

    @app.get("/v1/me", response_model=MeResponse, tags=["auth"])
    async def me(request: Request) -> MeResponse:
        settings: AuthSettings = request.app.state.auth_settings
        return MeResponse(user_id=_auth_user_id(request), auth_required=settings.required)

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
            store=_get_store(request),
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
            store=_get_store(request),
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
            store=_get_store(request),
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
            store=_get_store(request),
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
            store=_get_store(request),
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
            store=_get_store(request),
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
            store=_get_store(request),
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
            store=_get_store(request),
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
            store=_get_store(request),
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
            store=_get_store(request),
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
