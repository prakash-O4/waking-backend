# Task PC-A — Phase C: BS↔AD Canonical Calendar + Romanized Nepali Eval Slice

**Engineer:** Pi  
**Branch:** `phase-c/language-hardening`  
**Base branch:** `dev`  
**Status:** ASSIGNED

---

## Objective

Two deliverables:

1. **BS↔AD canonical calendar** — replace the `bs_to_ad_approx()` approximation in `app/authority/parser.py` with a lookup against embedded authoritative data. Create the DB provenance table, the in-memory lookup module, a seed script, and tests. Implement PS-5 fully.

2. **Romanized Nepali eval slice** — build the eval harness infrastructure and a 10-query golden set of romanized Nepali queries. Wire into `make eval` with Recall@5 reporting. Implement PS-8 infrastructure.

---

## Acceptance criteria

1. `app/authority/bs_ad_calendar.py` exports `lookup(bs_year, bs_month, bs_day=1) -> tuple[date, bool]`.
2. `lookup()` returns `(ad_date, False)` for known dates; `(ad_date, True)` for dates in `_BOUNDARY_WINDOWS`; raises `BeyondCalendarRange` for dates outside BS 2000–2090.
3. `app/authority/parser.py` no longer calls `bs_to_ad_approx()`. The `PHASE-C-TODO` comment is gone.
4. `migrations/003_bs_ad_calendar.sql` creates `bs_ad_calendar` and `bs_ad_boundary_window` tables.
5. `scripts/seed_bs_ad_calendar.py` inserts all (bs_year, bs_month, bs_day, ad_date) rows from the embedded constant into the DB. Idempotent (ON CONFLICT DO NOTHING).
6. `app/eval/romanized_slice.py` loads `app/eval/golden/romanized.json`, runs each query through `dumb_retriever.retrieve()`, and reports `Recall@5` for the slice.
7. `make eval` prints the romanized slice Recall@5 result (requires env vars; skips with a clear message if DB/OpenSearch unavailable).
8. `app/eval/golden/romanized.json` has ≥10 romanized Nepali legal queries with real `expected_uris` derived from `laws.jsonl`.
9. `make test` green (15 passed + new tests, 1 skipped).
10. `make lint` green (include all new files).
11. `make eval-gates` green (zero-tolerance gates untouched).

---

## Section 1: BS↔AD canonical calendar

### `migrations/003_bs_ad_calendar.sql`

```sql
CREATE TABLE IF NOT EXISTS bs_ad_calendar (
    bs_year   int NOT NULL,
    bs_month  int NOT NULL CHECK (bs_month BETWEEN 1 AND 12),
    bs_day    int NOT NULL CHECK (bs_day BETWEEN 1 AND 32),
    ad_date   date NOT NULL,
    source_kind text NOT NULL DEFAULT 'official_panchanga',
    version   text NOT NULL DEFAULT '1.0',
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (bs_year, bs_month, bs_day, version)
);

-- Dates where multiple published sources disagree by ±1 day.
-- Initially empty; operators populate as discrepancies are discovered.
CREATE TABLE IF NOT EXISTS bs_ad_boundary_window (
    bs_year  int NOT NULL,
    bs_month int NOT NULL,
    bs_day   int NOT NULL,
    description text,
    PRIMARY KEY (bs_year, bs_month, bs_day)
);
```

Update `scripts/migrate.py` to apply migration 003 as well.

### `app/authority/bs_ad_calendar.py`

Embed the canonical month-length data for **BS 2000–2090**.

**Data sourcing:** Copy the month-length constants from the `nepali-datetime` library source code (MIT-licensed, available at https://github.com/arneec/nepali-datetime). The library has a `CALENDAR` or `_BS_CALENDAR` constant — a dict mapping `bs_year -> [days_in_month_1, ..., days_in_month_12]`. Do not add the library to `requirements.txt` — embed the data directly.

**Epoch reference:** BS 2000/01/01 = AD 1943-04-14.

**Module structure:**

```python
from __future__ import annotations
from datetime import date, timedelta

# Keys: bs_year (int). Values: list of 12 ints — days in each month.
# Source: nepali-datetime library constants (MIT), cross-verified with GoN panchanga.
# Coverage: BS 2000–2090 (AD 1943-04-14 to approx AD 2033-04-13).
_MONTH_LENGTHS: dict[int, list[int]] = {
    2000: [30, 32, 31, 32, 31, 30, 30, 30, 29, 30, 29, 31],
    2001: [31, 31, 32, 31, 31, 31, 30, 29, 30, 29, 30, 30],
    # ... through 2090 ...
}

_EPOCH_BS_YEAR = 2000
_EPOCH_BS_MONTH = 1
_EPOCH_BS_DAY = 1
_EPOCH_AD = date(1943, 4, 14)

# Dates in a known ±1-day disagreement window.
# Populated by operators; empty for Phase C initial release.
_BOUNDARY_WINDOWS: frozenset[tuple[int, int, int]] = frozenset()


class BeyondCalendarRange(Exception):
    """Raised when (bs_year, bs_month) is outside the embedded data range."""


def lookup(bs_year: int, bs_month: int, bs_day: int = 1) -> tuple[date, bool]:
    """
    Return (ad_date, boundary_warning).
    boundary_warning=True if this exact (year, month, day) is a known ambiguous window.
    Raises BeyondCalendarRange for dates outside BS 2000-2090.
    """
    if bs_year not in _MONTH_LENGTHS:
        raise BeyondCalendarRange(f"BS {bs_year} is outside the embedded calendar range")
    month_lengths = _MONTH_LENGTHS[bs_year]
    if not (1 <= bs_month <= 12):
        raise BeyondCalendarRange(f"BS month {bs_month} out of range")
    if not (1 <= bs_day <= month_lengths[bs_month - 1]):
        raise BeyondCalendarRange(f"BS {bs_year}/{bs_month}/{bs_day} invalid — month has {month_lengths[bs_month - 1]} days")

    # Compute days elapsed from epoch
    days = 0
    for y in range(_EPOCH_BS_YEAR, bs_year):
        days += sum(_MONTH_LENGTHS[y])
    for m in range(1, bs_month):
        days += _MONTH_LENGTHS[bs_year][m - 1]
    days += (bs_day - 1)

    ad = _EPOCH_AD + timedelta(days=days)
    boundary = (bs_year, bs_month, bs_day) in _BOUNDARY_WINDOWS
    return ad, boundary
```

### `scripts/seed_bs_ad_calendar.py`

```
python3 scripts/seed_bs_ad_calendar.py
```

Iterates over all (bs_year, bs_month, bs_day) in `_MONTH_LENGTHS`, calls `lookup()`, and inserts into `bs_ad_calendar`. ON CONFLICT DO NOTHING. Prints progress every 10 years.

### Update `app/authority/parser.py`

Replace (at the call site on line 92):
```python
return bs_to_ad_approx(year, month)
```
with:
```python
from app.authority.bs_ad_calendar import lookup, BeyondCalendarRange
try:
    ad_date, _ = lookup(year, month, day)
    return ad_date
except BeyondCalendarRange:
    return None
```

Delete the `bs_to_ad_approx` function entirely. Update the extraction logic to also capture `day` from the regex match (the regex `_DATE_RE` already captures it — see group 3).

### `tests/test_bs_ad_calendar.py`

Cover:
1. `lookup(2000, 1, 1)` returns `(date(1943, 4, 14), False)` — epoch check.
2. `lookup(2080, 1, 1)` returns a date in April 2023 range (cross-check ±2 days is fine).
3. `lookup(9999, 1, 1)` raises `BeyondCalendarRange`.
4. `lookup(2000, 1, 0)` raises `BeyondCalendarRange` (invalid day).
5. Day offset monotonically increases: `lookup(2050, 3, 2)` == `lookup(2050, 3, 1)[0] + timedelta(days=1)`.

---

## Section 2: Romanized Nepali eval slice

### `app/eval/golden/romanized.json`

10 romanized Nepali legal queries with real expected component URIs from the ingested corpus.

**What "romanized Nepali" means:** Nepali words written in Latin script phonetically, e.g.:
- "bhrastachar niwaran ain" = भ्रष्टाचार निवारण ऐन
- "muluki ain ko dafa" = मुलुकी ऐन को दफा
- "karya sathar ain" = कार्यस्थल ऐन
- "shram ain" = श्रम ऐन

**How to find real URIs:** Inspect `laws.jsonl` at repo root. For each law, the URI follows the pattern `f"/np/{doc_type}/{bs_year_ascii}/{record['_id']}"` (from `parser.make_uri()`). Look up the `_id` field for well-known laws, then find specific component URIs by appending `/{component_type}/{number}`.

The component URI pattern: `{work_uri}/{component_type}/{number}` where `component_type` is `dafa`, `parichheda`, `dhara`, or `full`, and `number` is the section number or `0` for preambles.

Seed the golden set with romanized queries for at least these laws (find their `_id` in laws.jsonl):
- Prevention of Corruption Act (english_name contains "Corruption")
- Labor Act (english_name contains "Labor" or "Labour")
- Civil Service Act (english_name contains "Civil Service")
- Some other Act of your choice

Format:
```json
[
  {
    "query": "bhrastachar niwaran ain",
    "as_of": "2024-01-01",
    "expected_uris": ["/np/act/2059/<id>"],
    "note": "expect work-level or any component of this act"
  },
  ...
]
```

Note: `expected_uris` at the work URI prefix level is fine — the eval harness checks if any result `component_uri` starts with any `expected_uri` prefix (prefix match, not exact).

### `app/eval/romanized_slice.py`

```python
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
from typing import Any

from app.retrieval.dumb_retriever import retrieve

_GOLDEN = Path(__file__).parent / "golden" / "romanized.json"


def run_slice(k: int = 5) -> dict[str, Any]:
    """
    Load golden set, run each query through dumb_retriever.retrieve().
    A query is a "hit" if any result component_uri starts with any expected_uri prefix.
    Returns {"recall_at_k": float, "k": int, "n_queries": int, "hits": int}.
    Raises RuntimeError if DB/OpenSearch unavailable (caller prints skip message).
    """
    golden = json.loads(_GOLDEN.read_text())
    hits = 0
    for entry in golden:
        as_of = date.fromisoformat(entry["as_of"])
        results = retrieve(entry["query"], as_of, k=k)
        result_uris = [r["component_uri"] for r in results]
        for expected_prefix in entry["expected_uris"]:
            if any(u.startswith(expected_prefix) for u in result_uris):
                hits += 1
                break
    n = len(golden)
    return {"recall_at_k": hits / n if n else 0.0, "k": k, "n_queries": n, "hits": hits}


if __name__ == "__main__":
    try:
        result = run_slice()
        print(f"romanized-recall@{result['k']}: {result['hits']}/{result['n_queries']} = {result['recall_at_k']:.2f}")
    except Exception as e:
        print(f"romanized slice skipped: {e}")
```

### `Makefile` — update `eval` target

```makefile
eval: ## romanized slice Recall@5 (requires DB + OpenSearch env vars)
	python3 -m app.eval.romanized_slice
```

### `tests/test_romanized_eval.py`

Mock-based. Do not require live DB or OpenSearch.

Cover:
1. `run_slice()` with mocked `retrieve` that returns the expected URI → `recall_at_k = 1.0`.
2. `run_slice()` with mocked `retrieve` that returns empty → `recall_at_k = 0.0`.
3. Prefix-match logic: expected `/np/act/2059/foo`, result `/np/act/2059/foo/dafa/1` → counts as hit.
4. `run_slice()` when `retrieve` raises `RuntimeError` propagates to caller (harness handles it with skip message in `__main__`).

---

## Makefile — lint paths to add

Add to all three lint lines:
- `app/authority/bs_ad_calendar.py`
- `app/eval/romanized_slice.py`
- `tests/test_bs_ad_calendar.py`
- `tests/test_romanized_eval.py`
- `scripts/seed_bs_ad_calendar.py`

---

## Do NOT touch

- `app/retrieval/` — any existing file.
- `app/main.py` — no changes.
- `app/eval/gates.py` — must stay green.
- `migrations/001_bitemporal_schema.sql`, `002_gate_suspend_fix.sql`.

---

## Forbidden

- Do not add `nepali-datetime` or any calendar library to `requirements.txt`. Embed the data directly.
- Do not call any external API or library at runtime for calendar conversion.
- No new pip dependencies.
- The `bs_to_ad_approx` function must be fully deleted (not left as a fallback).

---

## PS requirements in scope

| PS | Enforcement |
|---|---|
| PS-5 | `lookup()` raises `BeyondCalendarRange` for dates outside BS 2000-2090; `_BOUNDARY_WINDOWS` enables future ambiguity warnings |
| PS-8 | Romanized Nepali eval slice with its own Recall@5 target wired into `make eval` |

## Zero-tolerance gates (must stay 0)
- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`

---

## Required checks

```
make lint
make test
make eval-gates
```

---

## Commit authorship

Every commit MUST use:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By: Claude`, no "Generated with Claude", no AI attribution of any kind.

---

## Return to Claude (via Prakash)

- Commit hash
- Changed/created files list
- `make lint` output
- `make test` output
- `make eval-gates` output
- `python3 -m app.authority.bs_ad_calendar` output (add a `__main__` block that prints `lookup(2000, 1, 1)` and `lookup(2080, 1, 1)`)
- Assumptions and remaining risks
