# task.md — AGENT-31: real stress/red-team suite (final task in the program)

## Objective

`make stress` is a no-op today (`Makefile:26-27`: `@python3 -c 'print("no
stress cases yet")'`). `system-design.md` §14 **PS-12** requires "no
regression on the append-only suite AND fresh red-team round per release,"
and §10 says coverage should be reported by taxonomy cells, not payload
count. Confirmed gap (finding #8 in the retrieval-quality review, see
`.agent/PROGRESS.md`'s "Retrieval-quality review + 6-task program" entry).

This task builds that suite for the **six taxonomy cells already named in
the program table**: repealed/current, not-yet-effective, romanized,
cross-ref, proviso, enabling-power. It runs last in the program (after
AGENT-26–30) so it tests the *fixed* invariants, not the gaps that used to
exist.

**Scope boundary, read this before starting**: `system-design.md` §10
describes three stress-test surfaces — #1 ingestion (corpus poisoning, fake
amending instruments, OCR attacks), #2 query (jailbreaks/injection, PII
extraction, cost amplification), #3 retrieval (injection in statutory text,
canary tokens). **This task is scoped to surface #3's correctness/temporal
angle only** — the six named cells, all of which are about whether
already-built retrieval/co-retrieval mechanisms correctly respect the
eligibility/temporal invariants. Corpus poisoning, jailbreak/injection
red-teaming, PII extraction, and cost-amplification are explicitly **out of
scope** — a separate, larger security-focused initiative if/when Prakash
wants it, not something to fold in here.

## What's already covered vs. the real gap — read before writing tests

All three co-retrieval mechanisms below already apply an eligibility
filter in their SQL. The gap is not that they're unfiltered — it's that
their **existing tests mock `eligible_chunk_ids` with a hardcoded set**
(`lambda conn, as_of: {"chunk-100", "chunk-456"}`-style), so nothing today
actually proves a *repealed* or *not-yet-effective* target gets excluded —
only that a chunk id not present in an arbitrarily-chosen set gets
excluded. This is the exact "canned boolean vs. actually-evaluated
predicate" gap this whole program has been closing everywhere else
(`FilteringCursor` in `tests/test_eligibility_gate.py`,
`TerminationCursor`/`CitationCursor` in `tests/test_validation_gate.py`,
`CoRetrieveCursor`-style harnesses in `tests/test_co_retrieve_parent.py`).
Match that same standard here — every stress test must evaluate a real
predicate against seeded lifecycle-effect-shaped data, not a canned
boolean or hardcoded set.

## Taxonomy cells

### 1. repealed/current
The single most important cell — this is literally what
`repealed-as-current = 0` means, and until now nothing has proven it holds
through the **full live path** (AGENT-26's finding: `app/eval/gates.py`
only tests the isolated `is_eligible()` SQL function against synthetic
insert/rollback data, never the wired-together `eligibility_gate.py` +
`validation_gate.py` path a real query actually takes).
- Seed a component with an approved `commence` effect in the past and an
  approved `repeal` effect at a later date `T`.
- Prove, through `eligibility_gate.eligible_chunk_ids()`: eligible at
  `as_of < T`, excluded at `as_of >= T`.
- Prove, through `validation_gate.validate_and_render()` (not just
  `_terminated_before` in isolation): a claim citing this component's
  chunk id renders normally (not abstained) at `as_of < T`, and abstains
  (`citation is None`) at `as_of >= T` — even if the claim somehow carries
  a valid span hash and a genuine supporting quote (i.e., prove termination
  is checked independently of the other checks, not just that the whole
  chain happens to fail together).

### 2. not-yet-effective
- Seed a component with an approved `commence` effect whose `legal_valid_time`
  starts in the future relative to a test `as_of`. Prove exclusion via
  `eligible_chunk_ids()` before the commence date, inclusion on/after it.
- Separately, seed a `commencement_dependency` (pending Gazette
  notification, PS-2) — prove it stays excluded even past what would
  otherwise be its commence date, since the dependency is still pending.

### 3. romanized
Distinct from `app/eval/romanized_slice.py` (that's a live-DB recall@k
quality eval — do not duplicate it or touch that file). This cell is about
whether the *romanized query path specifically* can bypass eligibility —
a real concern since it's a different code path (`translate_query` →
Devanagari translation → embed) than the direct-Devanagari path, and
nothing today proves eligibility filtering applies equally to both.
- Using `tests/test_retrieval.py`'s existing `Cursor`/`Conn` mock pattern
  (extend it, don't replace it — check `patch_common()` first), issue a
  romanized-script query (`r._is_devanagari(...)` is `False` for it) where
  the eligible set excludes the chunk the vector/lexical search would
  otherwise return. Prove `retrieve_postgres()` returns nothing for it,
  the same way it would for an ineligible chunk queried in Devanagari —
  eligibility isn't a silent bypass surface for either script path.

### 4. cross-ref
`gated_orchestrator.py::_resolve_cross_refs` (`:481`). Existing tests
(`tests/test_orchestrator.py::test_resolve_cross_refs_finds_section_reference`)
hardcode `eligible_chunk_ids` to a fixed set that always includes the
target — never proves exclusion.
- New test: seed a hit whose text contains a valid दफा cross-reference to
  a section that is **not** in a genuinely-evaluated eligible set (e.g.
  because it's repealed) — prove `_resolve_cross_refs` does not add it.
- Companion test: the same reference, but the target **is** eligible —
  prove it **is** added (a real positive case using the same
  predicate-evaluation harness, not just relying on the existing hardcoded
  -set test for the positive case).

### 5. proviso
`gated_orchestrator.py::_resolve_co_retrieve_parents` (`:416`, PS-16).
`tests/test_co_retrieve_parent.py::test_co_retrieve_parent_ineligible_parent_does_not_drop_hit`
already exists but uses `eligible_chunk_ids: lambda _conn, _as_of: set()`
(the whole set is empty) — sufficient to prove "nothing is eligible" but
not "a specific repealed parent is excluded while an in-force one is kept."
- New test: seed **two** candidate hits whose co-retrieve parents are
  different components — one in-force, one repealed (real predicate,
  not an empty set) — prove only the in-force parent is added, and the
  hit whose parent was excluded is still retained (unchanged from before,
  per `_resolve_co_retrieve_parents`'s existing "never drop the original
  hit" contract — assert this stays true, don't just assert the new part).

### 6. enabling-power
`query_graph.py::_fetch_enabling_chunk` (`:166`) already checks
`if enabling_chunk_id not in eligible: return None` (`:227`) — this is
exactly `system-design.md` §7.5's "orphaned Rules are flagged when the
enabling section is repealed" requirement, already implemented, but
**currently untested at all** for the repeal-orphaning case specifically
(check `tests/test_orchestrator.py`/`tests/test_ask_pipeline.py` for
existing `_fetch_enabling_chunk` coverage before assuming a gap — if it's
already covered with a real predicate, say so and skip re-testing it, note
why in your report).
- New test: seed a subordinate regulation chunk whose `work_relations` row
  points to a real enabling Act section; seed that enabling section as
  **repealed** (real predicate) as of the test `as_of` — prove
  `_fetch_enabling_chunk` returns `None` (the orphaning case). Companion
  test: same setup but the enabling section is still in force — prove it
  **is** returned.

## Suite structure

- New `tests/stress/` directory, one file per cell (e.g.
  `test_repealed_current.py`, `test_not_yet_effective.py`,
  `test_romanized_eligibility.py`, `test_cross_ref_eligibility.py`,
  `test_proviso_eligibility.py`, `test_enabling_power_orphan.py`) — check
  whether `tests/` needs any pytest config change for a subdirectory
  before assuming none is needed (there's no `pytest.ini`/`pyproject.toml`
  pytest config today, so default recursive discovery should just work —
  verify this rather than assume it).
- `Makefile`: replace the `stress` target's placeholder with
  `python3 -m pytest tests/stress/ -v` (verbose — PS-12 wants coverage
  visible by taxonomy cell, not just a pass count). Do not exclude
  `tests/stress/` from the existing `test` target's `pytest tests/` — it
  should run there too, for continuous regression protection on every
  commit, not just at release time. `make stress` existing as a separate,
  explicit target is what satisfies PS-12's "gate reported by taxonomy
  cell" framing, not an exclusion.
- Every new test must be a pure, deterministic, mock-based test (same
  style as the rest of `tests/`) — no live DB required to run this suite,
  consistent with `AGENTS.md`'s "fast local evals, run constantly."

## Governing design references

- `AGENTS.md` — non-negotiables: eligibility gate on every path; every
  claim validates against its declared as-of; abstention is server-owned.
- `system-design.md` §7.5 (enabling-power link; orphaned Rules flagged
  when the enabling section is repealed), §10 (stress-test surfaces and
  taxonomy-cell reporting — surface #3's correctness angle only, per the
  scope boundary above), §14 **PS-2** (commencement-pending), **PS-12**
  (stress gate = no regression + taxonomy-cell coverage), **PS-16**
  (provisos/स्पष्टीकरण co-retrieve with their operative clause,
  eval-asserted).
- `.agent/PROGRESS.md`'s AGENT-26 entry — the specific finding that
  `app/eval/gates.py` tests the isolated SQL predicate, never the live
  wired-together path. Cell #1 (repealed/current) directly closes this.

## Allowed scope

- New `tests/stress/` directory and its files.
- `Makefile` — only the `stress` target.
- `.agent/PROGRESS.md` — do not edit; Claude updates this after review.

## Explicitly forbidden

- Do not touch `app/eval/gates.py`, `app/eval/romanized_slice.py`, or any
  other existing eval slice — this task adds a new, separate suite, it
  does not modify the existing ones.
- Do not touch any production code (`app/retrieval/*`, `app/eligibility_gate.py`,
  `validation_gate.py`, etc.) — if a stress test reveals an actual bug in
  one of these mechanisms (not expected, given the eligibility filters
  already exist, but possible), **stop and report it, do not fix it
  silently** — that would be a new task, not this one.
- Do not build the ingestion-poisoning, jailbreak/injection, PII-extraction,
  or cost-amplification stress surfaces from `system-design.md` §10 — out
  of scope per the boundary above.
- Do not add a new dependency, service, or CI framework — pytest only,
  same as the rest of the repo.

## Required checks

- `make test` (must still pass, now including `tests/stress/`)
- `make stress` (new target, must pass and print per-file/per-case output)
- `make lint`
- `make eval-gates` — all three zero-tolerance gates must stay at `0`,
  untouched by this task.

## Zero-tolerance gates guarded

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

(This task's cell #1/#2 are the first tests in the whole repo proving
these hold through the *live, wired-together* path rather than an isolated
predicate — report explicitly whether that's an accurate characterization
after you've read the existing `app/eval/gates.py` and confirmed it, don't
just assume my framing is right.)

## Self-review before returning

- For each of the six cells, confirm the new test evaluates a real
  predicate against seeded data (dates, approval status, effect types) —
  not a hardcoded set or canned boolean. If you reused/extended an
  existing mock class (e.g. from `tests/test_eligibility_gate.py` or
  `tests/test_co_retrieve_parent.py`) rather than writing a new one, say
  which one and why — Ponytail favors reuse.
- Confirm cell #1's `validate_and_render()` test proves termination is
  checked independently (not just that a fully-broken claim happens to
  abstain for some other reason too).
- Confirm cell #6 either finds real new coverage or explicitly reports
  that existing coverage already proves the orphaning case with a real
  predicate (and if so, does not duplicate it).
- Confirm `make stress` and the relevant slice of `make test` both pass
  and report the same results (no drift between running the suite
  standalone vs. as part of the full test run).

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line, in any commit on this
branch. Use `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
