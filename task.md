# task.md — AGENT-28: citation authority-chain rendering

## Objective

`validation_gate.py::_citation()` currently renders a citation by reading
denormalized fields straight off the `chunks` row it was called with
(`c.act_name`, `c.case_id`, `c.source_type`) plus one join to
`source_publication` for `kind`/`ocr_confidence` (AGENT-22's fix). It never
touches `component`, `work`, or `lifecycle_effect` at all. This means the
citation shows *a* source, but never *the authoritative chain*: which
instrument originally enacted this provision, and which approved amendments
already apply to it as of the claim's `as_of`. Confirmed live gap (finding
#4 in the retrieval-quality review, see `.agent/PROGRESS.md`'s "Retrieval-
quality review + 6-task program" entry) — `_citation()`
(`validation_gate.py:12`) reads chunk metadata directly, no
`component`→`expression`→amending-`lifecycle_effect` resolution, no
`derived` labeling.

Fix: resolve the citation through the actual authority chain —
`component`/`work` for the canonical title/type, plus every **approved**
`amend` effect on that component already in force by `as_of` (each pointing
to its own amending `source_publication` row) — and label the result
`derived` when the base source itself is a consolidation, not an original or
amending instrument. Per `system-design.md` §7.4/PS-3.

**Naming trap, read before writing code**: `_citation()`'s current
parameter is named `component_uri` but the value it actually receives at
its call site in `validate_and_render()` is the **chunk's UUID id**
(`claim["evidence_id"]`), not the TEXT `component.uri`. The *real*
component URI only exists as `chunks.component_uri`, fetched via the
already-existing `_authority_component_uri(conn, evidence_id)` helper. Do
not conflate the two inside your rewrite — `component`, `work`, and
`lifecycle_effect` are all keyed by the TEXT component URI, never by the
chunk's UUID id. Rename `_citation()`'s own parameter to `evidence_id` for
clarity (its call site in `validate_and_render()` is unaffected — that's a
positional call, not a keyword one — do not touch `validate_and_render()`
itself). Leave the same misleading `component_uri` variable name everywhere
else in the codebase alone; it's pre-existing and out of scope.

## Acceptance criteria

1. **Resolve the real component URI first**: call
   `_authority_component_uri(conn, evidence_id)` (already exists, reuse —
   do not re-derive it with a new query) to get the TEXT `component.uri`,
   or `None` for a chunk with no link (e.g. `subsection`-level chunks,
   deliberately unlinked by design per the AGENT-24/25 history in
   `.agent/PROGRESS.md`).
2. **Linked case** (`component_uri` present): resolve, in this order:
   - `component` + `work` (join on `component.work_id = work.id`, filter
     `component.uri = component_uri`) → canonical `title_ne`/`title_en`,
     `component_type`, `number`. This replaces `c.act_name`/`c.case_id` as
     the title source for linked chunks — canonical, not a per-chunk
     denormalized copy that can drift.
   - **Base source**: keep the existing chunk→`documents`→
     `source_publication` join (by `evidence_id`, i.e. the chunk's own
     `document_id`) — this is still the correct way to find the specific
     source backing *this chunk's exact span*. Add `source_url` to what's
     already selected (`kind`, `ocr_confidence`) — cheap, already on the
     row, useful for "resolve to the authoritative instrument chain."
   - **Amending chain**: every `lifecycle_effect` row where
     `component_uri = <resolved component uri>`, `effect_type = 'amend'`,
     `approval_status = 'approved'`, and `lower(legal_valid_time) <=
     as_of` (i.e. already in force by the claim's declared as-of — same
     comparison direction as `_terminated_before`'s existing per-claim
     as-of logic, Core Invariant #6). Join each to its `source_publication`
     via `source_pub_id` for `kind`/`ocr_confidence`/`source_url`. Order by
     `effective_date ASC`. Only `effect_type='amend'` — repeal/expiry/
     suspend/declared_invalid are irrelevant here because a claim whose
     component is currently repealed/terminated never reaches `_citation()`
     at all (`validate_and_render`'s `ok` chain short-circuits before the
     `_citation()` call) — do not add effect types beyond `amend`.
3. **Unlinked case** (`component_uri` is `None`): fall back to exactly
   today's behavior — the existing chunk-level
   `act_name`/`case_id`/`source_type` + single `source_publication` join,
   with `amendments: []` and `derived` computed from that single source's
   `kind` (see #4). No behavior change for this branch beyond the new
   `derived`/`amendments` keys existing with empty/single-source values —
   this is the documented fallback invariant from AGENT-26, not a gap to
   close here.
4. **`derived` flag**: `True` when the base source's `kind` is
   `verified_internal_consolidation` or `derived_verified` (i.e. it's
   someone's reading/consolidation, not a primary instrument); `False` for
   `official_original`, `amending_instrument`, `official_copy_unverified`
   (a real instrument, provenance-uncertain is what `ocr_confidence`
   already signals — don't conflate "uncertain OCR" with "not authority").
5. **Return shape** — extend, don't replace, the existing citation dict
   (nothing downstream pattern-matches specific keys today — confirmed by
   grep, the whole `all_results` list is JSON-serialized wholesale into the
   composer prompt — but preserve the existing key names anyway, no reason
   to churn them):
   ```
   {
     "component_uri": <resolved uri or evidence_id fallback>,
     "work_title_ne": ..., "work_title_en": ...,
     "as_of": ...,
     "source_kind": <base source kind>, "ocr_confidence": <base>,
     "source_url": <base, new>,
     "derived": <bool, new>,
     "amendments": [  # new, [] when none apply or unlinked
       {"effective_date": ..., "source_kind": ..., "ocr_confidence": ...,
        "source_url": ...},
       ...
     ],
   }
   ```
6. The claim/quote text itself never appears in the citation — the citation
   is built entirely from `component`/`work`/`lifecycle_effect`/
   `source_publication`, canonical tables, never from `claim.get("claim")`
   or `claim.get("quote")` (Core Invariant #8: citations rendered from
   canonical metadata, never copied from model output).
7. No change to `_expression()`, `_terminated_before()`,
   `_authority_component_uri()`, `eligible_chunk_ids()`, or the `ok = ok
   and ...` chain / call sequence in `validate_and_render()` — this task
   only rewrites what happens inside `_citation()` and how it's called
   (still `_citation(conn, evidence_id, as_of) if ok else None`, unchanged
   call site).

## Branch

`agent/citation-authority-chain` (already created off `dev`, in sync).

## Governing design references

- `AGENTS.md` — non-negotiables: model never writes citations; every claim
  validates against its declared as-of; retrieved text is untrusted (the
  citation must not be built from claim/quote text).
- `system-design.md` §4 Data model — `component`, `work`,
  `source_publication` (kind enum), `lifecycle_effect` (effect_type enum,
  `source_pub_id`).
- `system-design.md` §7.4 Consolidation is not authority — citations
  resolve to the authoritative chain: base-Act source span + each amending
  instrument's source span, rendered from `source_publication` rows of kind
  `official_original`/`amending_instrument`, never from `expression`.
- `system-design.md` Core Invariant #1 (derivatives revalidated against
  authority), #8 (citations from canonical metadata, never model output),
  #10 (provenance propagates to the user).
- `system-design.md` §14 **PS-3** (citations resolve to the authoritative
  instrument chain; consolidations labeled `derived`) — this task's primary
  target. **PS-10** (OCR/source-kind provenance renders as a reliability
  badge) — extended here to cover the amending chain too, not just the base
  source.

## Allowed scope

- `app/retrieval/validation_gate.py` — **only** `_citation()`. Do not touch
  `_expression()`, `_terminated_before()`, `_authority_component_uri()`,
  `eligible_chunk_ids` (imported, not defined here), `_normalize()`,
  `_claim_supported()`, or `validate_and_render()`.
- `tests/test_validation_gate.py`
- `.agent/PROGRESS.md` — do not edit; Claude updates this after review.

## Explicitly forbidden

- Do not touch `_expression()` or read from the `expression` table at all
  for citation purposes — it is non-authoritative by design (§7.4); this
  task explicitly does not use it.
- Do not add a re-chunk/re-embed-on-amendment mechanism, or any change to
  the ingestion pipeline. If, while grounding this task, you notice that
  `chunks.chunk_text` is never refreshed when a `lifecycle_effect` amend is
  approved after initial ingestion (i.e. the retrieved text could be stale
  relative to an amendment this task's new `amendments` list will now
  surface) — **do not attempt to fix it**. Report it back to Claude as a
  new, separate finding. This task renders accurate provenance metadata
  about what's approved; it does not guarantee the chunk text already
  reflects it, and conflating the two is a bigger, separate problem.
- Do not add a new dependency, service, table, or abstraction.
- Do not touch `gated_orchestrator.py`, `query_graph.py`, or anything
  precedent-related — all out of scope, each a separate queued task or
  explicitly deferred.
- Do not rename `component_uri` anywhere outside `_citation()`'s own
  parameter and local variables.

## Required checks

- `make test`
- `make lint`
- `make eval-gates` — all three zero-tolerance gates
  (`repealed-as-current`, `not-yet-effective-as-current`,
  `overruled-as-good-law`) must stay at `0` — untouched by this task, must
  show no regression, not zero-by-virtue-of-untouched.

## Zero-tolerance gates guarded

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

## Self-review before returning

- Confirm you did not conflate chunk UUID id with the TEXT component URI
  anywhere in the new query logic (see the naming-trap note above) — trace
  each new query's parameter back to where it came from.
- Confirm the linked/unlinked branches both return the same dict shape
  (same keys present, `amendments: []` rather than a missing key when
  there's nothing to report) — downstream code should never need an
  `if "amendments" in citation` check.
- Confirm `tests/test_validation_gate.py`'s `GateCursor`/`GateConn` mock
  (or a new equivalent, matching the existing squashed-SQL-prefix dispatch
  pattern already used there) gets new branches for the `component`+`work`
  join and the `lifecycle_effect` amend-chain query — do not leave these
  new queries untested or test them only with canned booleans.
- New tests must prove, with seeded/mocked data, not canned booleans: a
  linked component with zero approved amendments renders `amendments: []`,
  `derived: false` for an `official_original` base; a linked component with
  two approved amendments (one before `as_of`, one strictly after) renders
  only the one before `as_of` in `amendments`, in effective-date order
  (per-claim as-of direction, same class of test as
  `test_terminated_before_true_when_repeal_on_or_before_as_of` /
  `..._false_when_repeal_strictly_after_as_of`); a base source of kind
  `verified_internal_consolidation` renders `derived: true`; the unlinked
  fallback path still renders the pre-existing `act_name`/`case_id` title
  behavior unchanged.
- Report which existing tests you had to update (not just which you added)
  and why, same as prior task reviews in `.agent/PROGRESS.md` expect.

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line, in any commit on this
branch. Use `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
