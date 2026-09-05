from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from app.utils.helpers import SupabaseHelper


def helper(monkeypatch: pytest.MonkeyPatch) -> SupabaseHelper:
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "dummy-key")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "secret")
    return SupabaseHelper()


def test_valid_token_returns_user_id_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = helper(monkeypatch)
    calls: list[str] = []

    def get_claims(*, jwt: str) -> dict[str, Any]:
        calls.append(jwt)
        return {"claims": {"sub": "user-1"}}

    monkeypatch.setattr(h.supabase.auth, "get_claims", get_claims)

    assert h.get_user_id("token") == "user-1"
    assert h.validate_token("token") is True
    assert calls == ["token", "token"]


def test_invalid_token_is_401_and_not_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    h = helper(monkeypatch)

    def get_claims(*, jwt: str) -> None:
        raise RuntimeError("bad token")

    monkeypatch.setattr(h.supabase.auth, "get_claims", get_claims)

    with pytest.raises(HTTPException) as exc:
        h.get_user_id("bad")
    assert exc.value.status_code == 401
    assert h.validate_token("bad") is False


def test_missing_header_is_401_without_calling_supabase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = helper(monkeypatch)

    def get_claims(*, jwt: str) -> Any:
        raise AssertionError("get_claims should not be called")

    monkeypatch.setattr(h.supabase.auth, "get_claims", get_claims)

    with pytest.raises(HTTPException) as exc:
        h.get_user_id(None)
    assert exc.value.status_code == 401
    assert h.validate_token(None) is False


def test_bearer_prefix_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    h = helper(monkeypatch)
    calls: list[str] = []

    def get_claims(*, jwt: str) -> dict[str, Any]:
        calls.append(jwt)
        return {"claims": {"sub": "user-1"}}

    monkeypatch.setattr(h.supabase.auth, "get_claims", get_claims)

    assert h.get_user_id("Bearer raw.jwt.token") == "user-1"
    assert calls == ["raw.jwt.token"]


def test_claims_without_sub_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    h = helper(monkeypatch)
    monkeypatch.setattr(h.supabase.auth, "get_claims", lambda *, jwt: {"claims": {}})

    with pytest.raises(HTTPException) as exc:
        h.get_user_id("token")
    assert exc.value.status_code == 404


def test_get_user_id_calls_get_claims_once(monkeypatch: pytest.MonkeyPatch) -> None:
    h = helper(monkeypatch)
    calls = 0

    def get_claims(*, jwt: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"claims": {"sub": "user-1"}}

    monkeypatch.setattr(h.supabase.auth, "get_claims", get_claims)

    assert h.get_user_id("token") == "user-1"
    assert calls == 1
