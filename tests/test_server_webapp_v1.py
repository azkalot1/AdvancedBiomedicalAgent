from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import bioagent.server.webapp as webapp


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_v1_me_requires_auth_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("BIOAGENT_API_TOKEN", "test-token")
    monkeypatch.setenv("BIOAGENT_API_USER_ID", "user_a")
    monkeypatch.delenv("BIOAGENT_AUTH_REQUIRED", raising=False)

    app = webapp.create_webapp()
    client = TestClient(app)

    unauthorized = client.get("/v1/me")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "unauthorized"
    assert "X-Request-ID" in unauthorized.headers

    authorized = client.get("/v1/me", headers=_auth_header("test-token"))
    assert authorized.status_code == 200
    payload = authorized.json()
    assert payload["user_id"] == "user_a"
    assert payload["auth_required"] is True


def test_legacy_scope_is_forbidden_for_different_user(monkeypatch) -> None:
    monkeypatch.setenv("BIOAGENT_API_TOKEN", "scope-token")
    monkeypatch.setenv("BIOAGENT_API_USER_ID", "user_scope")

    async def fake_list_reports(**_: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(webapp, "list_reports", fake_list_reports)

    app = webapp.create_webapp()
    client = TestClient(app)

    response = client.get("/users/not_the_same/reports", headers=_auth_header("scope-token"))
    assert response.status_code == 403
    payload = response.json()
    assert payload["error"]["code"] == "forbidden_user_scope"
    assert "X-Request-ID" in response.headers


def test_v1_reports_pagination(monkeypatch) -> None:
    monkeypatch.setenv("BIOAGENT_API_TOKEN", "pagination-token")
    monkeypatch.setenv("BIOAGENT_API_USER_ID", "user_paginated")

    sample_reports = [
        {"id": "r5", "filename": "r5.md"},
        {"id": "r4", "filename": "r4.md"},
        {"id": "r3", "filename": "r3.md"},
        {"id": "r2", "filename": "r2.md"},
        {"id": "r1", "filename": "r1.md"},
    ]

    async def fake_list_reports(**_: Any) -> list[dict[str, Any]]:
        return sample_reports

    monkeypatch.setattr(webapp, "list_reports", fake_list_reports)

    app = webapp.create_webapp()
    client = TestClient(app)

    response = client.get(
        "/v1/reports",
        params={"limit": 2, "offset": 1},
        headers=_auth_header("pagination-token"),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 5
    assert payload["limit"] == 2
    assert payload["offset"] == 1
    assert payload["has_more"] is True
    assert [item["id"] for item in payload["items"]] == ["r4", "r3"]


def test_v1_report_content_limits(monkeypatch) -> None:
    monkeypatch.setenv("BIOAGENT_API_TOKEN", "content-token")
    monkeypatch.setenv("BIOAGENT_API_USER_ID", "user_content")

    async def fake_get_report(**_: Any) -> dict[str, Any] | None:
        return {"id": "rep_1", "filename": "rep_1.md"}

    async def fake_get_report_content(**_: Any) -> str | None:
        return "abcdefghij"

    monkeypatch.setattr(webapp, "get_report", fake_get_report)
    monkeypatch.setattr(webapp, "get_report_content", fake_get_report_content)

    app = webapp.create_webapp()
    client = TestClient(app)

    response = client.get(
        "/v1/reports/rep_1/content",
        params={"offset": 2, "max_chars": 4},
        headers=_auth_header("content-token"),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "rep_1"
    assert payload["content"] == "cdef"
    assert payload["total_chars"] == 10
    assert payload["returned_chars"] == 4
    assert payload["offset"] == 2
    assert payload["max_chars"] == 4
    assert payload["truncated"] is True


def test_v1_report_not_found_envelope(monkeypatch) -> None:
    monkeypatch.setenv("BIOAGENT_API_TOKEN", "missing-token")
    monkeypatch.setenv("BIOAGENT_API_USER_ID", "user_missing")

    async def fake_get_report(**_: Any) -> dict[str, Any] | None:
        return None

    monkeypatch.setattr(webapp, "get_report", fake_get_report)

    app = webapp.create_webapp()
    client = TestClient(app)

    response = client.get("/v1/reports/does-not-exist", headers=_auth_header("missing-token"))
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "report_not_found"
    assert "request_id" in payload["error"]
