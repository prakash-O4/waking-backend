# AGENT-14 — Fix दफा/परिच्छेद/धारा header over-matching + canonicalize source_sha256

## Objective

`app/authority/parser.py`'s `_HEADER_RE` has three "loose" (non-bold)
alternatives meant to catch headers not wrapped in `**bold**` markdown:
`दफा\s+(\d+)[.।\s]`, `परिच्छेद[-–]\s*(\d+)`, `धारा\s+(\d+)[.।\s]`. None of
them require the match to sit at the start of a line. Corpus-wide match
count against all 677 `laws.jsonl` records (script-verified, not
hypothesized):

```
दफा loose alternative:      10,766 total matches,  8 at true line-start
परिच्छेद loose alternative:  3,270 total matches,  7 at true line-start
धारा loose alternative:        651 total matches,  0 at true line-start
```

The overwhelming majority are inline cross-references inside a
*different* section's body text (e.g. "यस ऐनको दफा ३ बमोजिम..."), not
real headers. Each spurious match becomes a fake component boundary in
`parse_law()`: `_component_kind()` computes the same `kind`/`number` as
the real दफा (so it collides on the same `uri`), but the sliced text
differs — `upsert_component`'s `ON CONFLICT (uri) DO NOTHING` keeps only
one component row, while `upsert_expression`'s `(component_uri, as_of,
text_hash)` key does **not** collide (different text → different hash),
so every spurious fragment survives as a permanent extra `expression`
row for that component_uri.

**Confirmed against the live corpus:** `आयकर_ऐन_२०५८_...`
(`/np/act/unknown/477390c2-.../dafa/3`) has 2 genuine bold दफा-३ header
matches vs. 36 loose cross-reference matches to "दफा ३" — this lines up
with AGENT-13's finding of 33 duplicate expression rows for that exact
component_uri.

**Already in the live local DB** (from AGENT-13's backfill run,
2026-08-30): 3,712 of 16,459 `component` rows have this problem, adding
up to several thousand garbage `expression` rows out of 23,187 total.

Separately, while grounding this task I found a second, independent
hash bug: `parser.py::parse_law()`'s `source_sha256` hashes raw
(non-NFC-normalized) content, while `pipeline.py::_content_hash()` (used
for `documents.content_hash`) NFC-normalizes first. **2 of 677** corpus
records are not already in NFC form, so for those 2 documents
`source_publication.source_sha256` will never match `documents
.content_hash` for identical content — breaking the hash-based
provenance chain PS-3 requires.

## Assigned branch

`agent/dafa-header-dedup` (base: `dev`)

## Scope — three parts, all required

### A. Fix `_HEADER_RE` false-positive matching (`app/authority/parser.py`)

- Anchor the three loose (non-bold) alternatives (दफा, परिच्छेद, धारा) so
  they only match genuine headers, not inline cross-references.
  Line-start anchoring is the evidence-backed direction (see counts
  above) but verify against the corpus yourself before locking it in —
  look at what the handful of true line-start matches actually are, and
  confirm no corpus document depends on a loose header that ISN'T at
  line-start (leading whitespace/indentation, etc.) before ruling that
  out.
- Do not touch the bold-header alternative (group 1/2) — no evidence
  it's broken.
- Re-run the corpus-wide match-count check (regex against all 677
  `laws.jsonl` records) before/after your fix and report the numbers in
  your return message — that count delta is the acceptance bar, not
  just unit tests passing.
- Extend `tests/test_parser.py` with a case reproducing the
  cross-reference false positive (a body containing "...यस ऐनको दफा ३
  बमोजिम..." must NOT create a new component boundary) alongside the
  existing legitimate-header cases.

### B. Clean up already-corrupted `expression` rows in the live local DB

- After A is fixed, `parse_law()` on the same 345 documents produces
  the correct (deduplicated) component/expression set. Write a script
  (follow `scripts/backfill_authority_layer.py`'s pattern: per-document
  commit/rollback, `--dry-run` flag, sourced from `laws.jsonl` by
  `source_id`, `_content_hash`/`uri` assertions before writing anything)
  that, for each already-ingested document: re-parses with the fixed
  `parse_law()`, and for every `component_uri` where live `expression`
  rows exist whose `text_hash` does not match any hash the fixed parser
  would produce for that `component_uri` at today's `as_of`, deletes
  exactly those stale rows.
- Do not touch `component` or `lifecycle_effect` rows — component URIs
  are unaffected by this fix (only fragment *count* changes per URI),
  and lifecycle proposals are keyed by URI, not by which expression
  fragment produced them.
- This is a deletion from the bitemporal authority store (Core
  Invariant #1). The rows being deleted are not legitimate historical
  text versions — they are artifacts of today's parsing bug, written by
  AGENT-13's backfill run earlier today. Still: dry-run first, hand
  -verify a couple of known offenders (e.g. the आयकर ऐन दफा-३ case
  should end at exactly 1 expression row) before running for real.
- Run it for real against the live local DB and report the final
  `expression` row count (currently 23,187) and the reduction.

### C. Canonicalize `source_sha256` (`app/authority/parser.py::parse_law`)

- Change `source_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest()`
  to NFC-normalize first, matching `pipeline.py::_content_hash()`
  exactly (same normalize-then-hash order), so the two hashes agree for
  identical content going forward.
- Existing `source_publication.source_sha256` values for the 2
  already-non-NFC corpus documents will now be stale against the
  corrected function. Decide whether to also correct those 2 rows in
  the same cleanup pass from part B (recommended: yes, same script,
  same document loop) or leave them — your call, but state which you
  did and why in your return message.
- Add a test asserting `parse_law(record).source_sha256` matches an
  NFC-normalized-then-hashed expectation for a non-NFC input fixture.

## Out of scope — do not touch

- VALIDATE-stage structural hardening (duplicate/broken section
  numbering beyond what part A's regression test covers, malformed
  markup, doc-type allowlist, chunk-size violations) — split out to
  AGENT-17.
- The bare `5000` tariff-threshold literal in
  `app/ingestion/tariff_chunker.py:38` — split out to AGENT-17.
- `eligibility_gate.py` rewiring to the bitemporal layer — separate
  future task, unaffected by this one.
- `app/ingestion/pipeline.py`'s `PROPOSE_LIFECYCLE`/commencement
  extraction path — untouched by this bug (keyed by `law.uri`/
  `component_uri`, not by expression fragment count).

## Relevant System Design sections

- `system-design.md` §2 Core Invariant #1 (bitemporal store is single
  authority — the cleanup script must not weaken this, hence the
  deletion discipline in part B) and #10 (provenance propagates —
  motivates part C).
- PS-3 (citations resolve to the authoritative instrument chain; CI
  reconstructs every expression from base + effects — hash consistency
  between `documents.content_hash` and `source_publication.source_sha256`
  is part of that chain).

## Required checks

- `make test`
- `make lint`
- `make eval-gates` (zero-tolerance gates must stay at 0 — this task
  touches authority-layer writes, not retrieval, so no gate should move;
  confirm and report).

## Explicitly forbidden changes

- No changes to `component` or `lifecycle_effect` tables/writers.
- No changes to the bold-header regex alternative.
- No changes to `eligibility_gate.py`, `pipeline.py`'s retrieval-facing
  code, or anything outside `app/authority/parser.py`, the new cleanup
  script, and their tests.
- Do not touch VALIDATE-stage code in `pipeline.py` (`_DAFA_ANCHOR_RE`
  etc.) — separate task (AGENT-17).

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — never AI-attributed, no
`Co-Authored-By: Claude` trailer, no "Generated with Claude" line. Use:

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

---

## Rework — round 1 (2026-08-30)

Review of `9c843ff`: part A (regex fix) and part C (`source_sha256`
canonicalization) are correct — independently verified against the
corpus. Part B (the cleanup script) has a real gap, confirmed against
the live local DB, not hypothesized.

### The gap

`_expected_hashes()` builds its expected `{component_uri: text_hash}`
map from `law.components` — i.e. only from URIs the **corrected**
parser still produces. `_stale_expression_ids()` only ever looks up
expression rows for URIs in that map. This misses the case where a
URI's *only* match, under the old buggy parser, was a false-positive
cross-reference with **zero** genuine header anywhere in the document —
the task brief's assumption that "component URIs are unaffected by this
fix (only fragment *count* changes per URI)" was wrong for this case.
When every match for a given दफा/परिच्छेद number in a document was a
false positive, that URI disappears entirely from `law.components`
post-fix, `_expected_hashes()` never contains it, and the script silently
never looks at its rows.

**Verified against the live local DB** (branch `agent/dafa-header-dedup`,
fixed parser): corpus-wide, 255 documents have at least one दफा number
whose only prior match was the false-positive loose pattern (856 such
numbers total). Querying the live DB directly for `component` rows with
no matching URI in the corrected parse output, across all 345 ingested
documents:

```
orphaned component rows:            1,901
orphaned expression rows under them: 2,350
lifecycle_effect rows referencing
  those orphaned component_uris:    1,238 (all approval_status='pending',
                                     none 'approved' — confirmed by
                                     GROUP BY query)
```

None of the 1,901 `component` rows, their 2,350 `expression` children,
or the 1,238 pending `lifecycle_effect` proposals keyed to them get
touched by the current script. They are sitting in the bitemporal
authority store as fully fictitious records — Core Invariant #1
territory (`system-design.md` §2.1: "the bitemporal store is the single
authority... never a source of derivative garbage" in spirit, even
though not literally quoted that way).

### What to do

Extend the cleanup script with a second pass: for each document, after
computing `expected_uris = {c.uri for c in law.components}`, fetch the
DB's actual `component` URIs for that `work_id` and compute the set
difference (DB URIs not in `expected_uris`). For each orphaned URI:

1. Delete its `expression` rows.
2. Delete its `lifecycle_effect` rows — but **only** if every row for
   that URI has `approval_status='pending'`. If you find even one
   `approved` row (the live DB currently has none, per the breakdown
   above, but don't assume that stays true — check it live before
   deleting), stop for that URI, do not delete anything under it, and
   report it instead — an approved lifecycle fact needs a human decision,
   not a script deleting it.
3. Delete the `component` row itself.

Same discipline as the rest of this task: per-document commit/rollback,
`--dry-run` first, hand-verify a couple of the orphaned URIs you found
before running for real, run for real against the live DB afterward,
and report the before/after counts for `component`, `expression`, and
`lifecycle_effect` (not just `expression` as in the first pass).

Re-run `make test`, `make lint`, `make eval-gates` after the change and
report results, same as before.
