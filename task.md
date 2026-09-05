# AGENT-40: fix broken auth token validation (real bug, blocks all authenticated requests)

## Objective

`app/utils/helpers.py::SupabaseHelper.validate_token()` calls a private
method that no longer exists in the pinned `supabase` package version.
Every real (non-mocked) call to `/ask` or `/ask/stream` with a genuine
Supabase JWT currently fails with a misleading `404`. This has never
been caught because (a) the entire test suite mocks `SupabaseHelper`
away completely — zero tests exercise this file's real logic — and (b)
`app/utils/helpers.py` is not in the Makefile's lint/mypy coverage at
all, so nothing has ever statically checked it either. Fix the bug, and
close both of those gaps so a break like this can't hide again.

This was found live, today, by Prakash testing the AGENT-37/38 dev
console with a real account for the first time — not a hypothetical.

## Grounding (already done, don't re-derive)

- **Root cause, fully diagnosed and confirmed empirically** (not
  guessed): `requirements.txt` pins `supabase==2.31.0`. In that version,
  the auth client's private `_decode_jwt` method
  (`SyncSupabaseAuthClient`, package `supabase_auth`) **does not
  exist** — confirmed via
  `AttributeError: 'SyncSupabaseAuthClient' object has no attribute '_decode_jwt'`
  against the project's actual `.venv`. `validate_token()`
  (`helpers.py:25-36`) calls exactly this method, the `AttributeError`
  gets swallowed by its own `except Exception as e: return False`, so
  `validate_token()` returns `False` for every single real token,
  always. `get_user_id()` (`helpers.py:38-63`) then raises
  `HTTPException(401, "Invalid token")` — which its own *outer*
  `except Exception as e: raise HTTPException(404, ...)` catches and
  re-wraps into the confusing `404` with `"Error fetching user
  details: 401: Invalid token"` you've been seeing.
- **The environment that actually runs the server has this exact
  version** — confirmed via `.venv/bin/pip list`: `supabase 2.31.0`,
  `supabase-auth 2.31.0`. (Aside, not part of this task: a bare
  `python3` in some shells on this machine resolves to a different,
  older interpreter without this package at all — that's a
  Makefile/environment hygiene issue, not something to fix here. Always
  verify checks for this task against `.venv/bin/python`, not bare
  `python3`.)
- **The replacement is a public, stable method that already exists in
  this exact installed version**: `self.supabase.auth.get_claims(jwt=<raw_token>)`.
  Confirmed its source directly in `.venv`:
  - Signature: `get_claims(self, jwt: Optional[str] = None, jwks: Optional[JWKSet] = None) -> Optional[ClaimsResponse]`.
  - It calls `decode_jwt(token)` internally (`supabase_auth.helpers`),
    which **requires a bare token** — `token.split(".")` must yield
    exactly 3 parts where `parts[0]` is a valid base64url-encoded JSON
    header. **Passing `"Bearer <jwt>"` directly will break this** (the
    header segment becomes `"Bearer eyJhbGci..."`, not valid base64) —
    unlike the old broken method, which only ever looked at
    `parts[1]` and happened to tolerate the prefix by accident. **The
    "Bearer " prefix must be stripped before calling `get_claims`.**
    This project's `/ask` handlers never did this stripping before
    calling `validate_token` (only `get_user_id` stripped it, and only
    *after* the validate step) — that ordering must change.
  - It calls `validate_exp(payload["exp"])` internally
    (`supabase_auth.helpers`), which raises `AuthInvalidJwtError` (not
    `jwt.ExpiredSignatureError` — that's a PyJWT exception type that
    was never actually raised by the old code either; the existing
    `except jwt.ExpiredSignatureError:` clause at `helpers.py:33` is
    dead code left over from an earlier implementation and should be
    removed, not preserved).
  - For this project's tokens specifically (confirmed via the actual
    JWT header from a real login: `{"alg":"HS256","kid":"..."}`),
    `get_claims` takes the **symmetric-algorithm branch**: `if "kid"
    not in header or header["alg"] == "HS256": self.get_user(token);
    return ClaimsResponse(claims=payload, ...)` — i.e. **it already
    calls `get_user()` internally** for HS256 tokens. This means
    `get_claims()` alone gives you both "is this valid" *and* the
    claims (including `sub`, the user id) in one call — there's no
    need for a second, separate `get_user()` call afterward. The old
    code's `validate_token()` → `get_user_id()`'s own `get_user()` call
    is a **redundant second network round-trip to Supabase per
    request** once `get_claims` is used correctly for validation — the
    fix should not just patch the broken call in place, it should
    consolidate to one `get_claims()` call shared between
    `validate_token()`/`get_user_id()`, not add a second network call.
  - On any invalid input (malformed structure, expired, bad signature),
    `get_claims`/its internals raise `AuthInvalidJwtError`. It only
    returns `None` when `jwt` is omitted and there's no active session
    — a path this project's synchronous-per-request usage never hits,
    since `jwt=raw_token` is always passed explicitly. So: **catch
    exceptions from `get_claims`, don't rely on a `None` return** to
    detect an invalid token.
- **`app/utils/helpers.py` is entirely absent from the Makefile's lint
  targets** (`make lint`'s ruff/ruff-format/mypy commands list specific
  files, and this one isn't in any of them) and has **zero unit tests**
  anywhere in `tests/` — every test that touches `/ask` uses a
  hand-written `FakeSupabaseHelper` (see `tests/test_ask_pipeline.py`,
  `tests/test_degraded_modes.py`) that never calls the real class. That
  combination is exactly why this shipped invisibly. Checked what
  adding this file to strict mypy/ruff surfaces right now (before any
  fix): 8 mypy errors, 2 ruff errors — small and bounded, not a rabbit
  hole. Listed below; fix them as part of this task.

## What to build

### 1. `app/utils/helpers.py` — fix the actual bug

Replace the broken decode call and consolidate to one `get_claims()`
call. Shape (adapt to the file's existing style, this is the logic not
literal final code):

```py
def _extract_bearer(token: str | None) -> str | None:
    if not token:
        return None
    return token.split(" ", 1)[1] if token.lower().startswith("bearer ") else token

class SupabaseHelper:
    ...
    def _get_claims(self, token: str | None) -> Any | None:
        raw = _extract_bearer(token)
        if not raw:
            return None
        try:
            return self.supabase.auth.get_claims(jwt=raw)
        except Exception:
            return None

    def validate_token(self, token: str | None) -> bool:
        return self._get_claims(token) is not None

    def get_user_id(self, token: str | None) -> str:
        claims = self._get_claims(token)
        if claims is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = claims.claims.get("sub")
        if not user_id:
            raise HTTPException(status_code=404, detail="User not found")
        return str(user_id)
```

Notes on this shape:
- `get_user_id`'s current `-> Dict` return type annotation is already
  wrong today (it returns `response.user.id`, a string) — fix it to
  `-> str` as part of this change, now that mypy will actually check
  this file.
- Remove the dead `except jwt.ExpiredSignatureError:` clause (see
  grounding above — it never fired, even before this bug).
- `get_user_id` no longer needs its own outer
  `except Exception as e: raise HTTPException(404, f"Error fetching...")`
  wrapper — that wrapper is *what caused* the 401-masked-as-404 bug you
  triggered live. With `_get_claims` catching everything internally and
  returning `None` on any failure, `get_user_id` can raise the correct
  status code directly (401 for invalid/unparseable token, 404 only for
  "valid token, but no `sub` claim" — which shouldn't normally happen,
  but keep the check).
- Trade-off worth knowing: collapsing all `_get_claims` failures
  (malformed token, expired token, and genuine Supabase network/infra
  errors) into a single `None` → 401 means a real infra failure now
  looks like "bad token" to the client instead of surfacing the actual
  error. That's an intentional, correct choice for what the client
  sees (don't leak infra details into a 401 body), but log the real
  exception server-side inside `_get_claims`'s `except` block using the
  existing `from app.utils.loggers import logger` pattern already used
  in `app/main.py`, so it's not silently lost for debugging.
- `check_daily_quota`, `get_user_chat_history`, `transform_qa_data` are
  untouched — not part of this bug, out of scope.

### 2. Makefile — add this file to lint coverage

Add `app/utils/helpers.py` to the file lists in the `ruff check`, `ruff
format --check`, and `mypy --strict ...` lines (same three lines every
other file in `app/` already appears in). Fix whatever it surfaces —
expected to be small (the 8 mypy / 2 ruff issues already identified in
grounding, most of which the fix above already resolves as a side
effect: the unused `except ... as e`, the wrong `-> Dict` return type,
the unused `fastapi.Header` import). If something unexpected and large
turns up outside the scope of this fix, stop and flag it rather than
scope-creeping into an unrelated cleanup.

### 3. New tests — `tests/test_helpers.py`

This file has zero tests today; that's part of why this bug shipped.
Add real coverage of `SupabaseHelper.get_user_id`/`validate_token`
against a **mocked `self.supabase.auth.get_claims`** (can't hit real
Supabase in unit tests) reflecting the actual shape confirmed above —
a `ClaimsResponse`-like object with a `.claims` dict containing `sub`.
Cover at minimum:
- Valid token (mocked `get_claims` returns claims with a `sub`) →
  `get_user_id` returns that `sub` as a string, `validate_token` is
  `True`.
- `get_claims` raising `AuthInvalidJwtError` (or any exception — don't
  couple the test to that exact class name from a third-party package)
  → `get_user_id` raises `HTTPException` with `status_code == 401`,
  `validate_token` is `False`.
- Missing/`None` header → 401, without ever calling `get_claims` (fails
  fast in `_extract_bearer`).
- A `"Bearer <token>"`-prefixed header actually gets the prefix
  stripped before reaching `get_claims` — assert on what
  `get_claims`/the mock was actually called with, not just the return
  value, so a future regression back to passing the raw header
  wouldn't silently pass this test.
- Claims present but no `sub` key → 404 (the existing, narrower "user
  not found" case).
- `get_claims` is called exactly once per `get_user_id` call — this is
  the regression test for the "don't add a second redundant network
  call" requirement in grounding above.

You'll need to construct `SupabaseHelper` without hitting real network
calls — `SupabaseHelper.__init__` calls `create_client(url, key)`
which itself doesn't make a network call (it's a local client
construction), so instantiating with dummy `SUPABASE_URL`/
`SUPABASE_KEY`/`SUPABASE_JWT_SECRET` env vars (via `monkeypatch.setenv`)
and then monkeypatching `instance.supabase.auth.get_claims` should
work without needing a running Supabase project.

## Explicitly forbidden

- Any change to `check_daily_quota`, `get_user_chat_history`,
  `transform_qa_data`, or the `chat`/`config` table logic — unrelated
  to this bug.
- Any change to `app/main.py`'s `/ask`/`/ask/stream` route logic itself
  — this task fixes the helper they call, not the routes.
- Adding a second network call to Supabase per request where one
  suffices (see the redundant-`get_user()`-call point in grounding).
- Silently swallowing the real exception with no server-side log —
  the old bug's core failure mode was exactly this pattern (broad
  except, no log, no signal) and directly caused this to go unnoticed
  for the entire project's history so far.
- Boiling the ocean on `app/utils/helpers.py`'s other pre-existing
  issues (the dead `SUPABASE_JWT_SECRET`/commented-out PyJWT line,
  `check_daily_quota`'s broad excepts, etc.) beyond what's needed to
  pass lint/mypy cleanly for the lines this task actually touches.

## Required checks

Run everything against `.venv/bin/python`, not bare `python3` (see
grounding — bare `python3` may resolve to a different interpreter
missing the pinned `supabase` version on this machine, which would
silently hide exactly this class of bug again):

```
.venv/bin/python -m pytest tests/
.venv/bin/python -m ruff check <updated file list incl. app/utils/helpers.py>
.venv/bin/python -m ruff format --check <same>
.venv/bin/python -m mypy --strict --follow-imports=skip --disable-error-code=misc --disable-error-code=import-untyped <same>
```

Manual verification (describe in the completion report): if you have
network access to reach the real Supabase project, the same login +
`/ask` flow Prakash used should now succeed end-to-end. If you don't
have that access in your environment, say so explicitly rather than
claiming it was verified — the unit tests against a mocked
`get_claims` are the real correctness bar for this task regardless.

## Commit authorship

Every commit authored `Prakash Basnet <basnetprakash090@gmail.com>` — no
AI attribution, no `Co-Authored-By: Claude` trailer.

## Branch

`agent/fix-auth-token-validation`, off `dev`.
