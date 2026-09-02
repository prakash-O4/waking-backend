# task.md — AGENT-24: wire PS-16 co-retrieve chain into query-side context assembly

## Objective
`chunks.co_retrieve_parent_id` is written correctly at ingest time (verified
live: 51/51 `proviso` chunks and all 9,072 `tariff_row`/`tariff_note` chunks
are already linked to their parent — **no reingestion needed, do not touch
any chunker or the pgvector indexer**) but is never read anywhere on the
query side. Grep confirms it: `grep -rn "co_retrieve_parent_id" --include="*.py" .`
matches only `app/ingestion/pgvector_indexer.py`. This is a live, evidenced
gap against `system-design.md` §14 **PS-16**: *"Provisos and स्पष्टीकरण
co-retrieve with their operative clause (eval-asserted)"* and
`docs/ingestion_design.md:56`: *"Context assembly on the query side must
always fetch `co_retrieve_parent_id` chains — this is the deterministic
enforcement of PS-16."* Today a proviso (or tariff row/note) retrieved on
its own never pulls in the operative clause / tariff heading it depends on.

This is pure retrieval-path wiring — the schema, the data, and the linking
logic all already exist and are already correct. Do not re-derive or
second-guess them.

## Assigned branch
`agent/co-retrieve-parent-context` (already created off `dev`, checked out).

## What to build
1. In `app/retrieval/gated_orchestrator.py`, add a new resolver function —
   mirror the existing `_resolve_cross_refs` (lines ~384-459) and
   `_fetch_enabling_chunk`/`enabling_power_resolver_node` pattern in
   `app/retrieval/query_graph.py` exactly, including their fields on
   returned hit dicts (`component_uri`, `text_ne`, `text_hash`, `score: 0.0`,
   `work_title_ne`, `chunk_type`, `section_number`, `document_source_id`,
   `"co_retrieved": True`, `"_issue_idx"` inherited from the originating
   hit) so downstream code (`_structured_claims`, `validate_and_render`,
   answer composition) treats it identically to the existing two co-retrieve
   mechanisms.
2. For the current hit set, resolve each hit's `co_retrieve_parent_id`
   (not present in `_hit()`'s row shape today — needs its own query, e.g. a
   join from `chunks c` on `c.co_retrieve_parent_id = p.id` filtered to
   `c.id::text = ANY(hit_ids)`) and, for any non-null parent not already
   present among current hits, fetch its display row and gate it through
   `eligible_chunk_ids(conn, as_of)` before adding it — same eligibility
   discipline `_resolve_cross_refs` already uses. If the parent isn't
   eligible, drop it silently (matches existing gate philosophy — write a
   test proving this, don't guess the behavior).
3. **This is a single hop by construction, not a chain to walk
   recursively**: `subsection`-level chunks never have their own
   `co_retrieve_parent_id` (verified — see 5.), and `tariff_heading` chunks
   never have one either. Only `proviso` and `tariff_row`/`tariff_note`
   chunks do, and their parent is always a non-`co_retrieve`-linked chunk.
   Don't build chain-walking logic the data doesn't need — skip hits
   already flagged `co_retrieved` the same way `enabling_power_resolver_node`
   does (`if h.get("co_retrieved"): continue`), to avoid resolving a hop off
   a chunk this same pass just added.
4. Wire it into the graph in `app/retrieval/query_graph.py` as a new node
   (e.g. `co_retrieve_parent_resolver_node`), inserted between
   `authority_ranker` and `cross_ref_resolver`:
   `authority_ranker → co_retrieve_parent_resolver → cross_ref_resolver →
   enabling_power_resolver → reasoner`. Follow `cross_ref_resolver_node`'s
   exact shape (Langfuse span, empty-dict return when nothing added).
5. New test(s) proving real behavior against a stub cursor (follow the
   `TerminationCursor`/`FilteringCursor` pattern already used in this repo
   — actually evaluate a seeded predicate, not a string-only SQL assertion):
   - a `proviso` hit whose `co_retrieve_parent_id` resolves to an eligible
     chunk → parent gets added, correct fields, `co_retrieved: True`.
   - a hit with `co_retrieve_parent_id IS NULL` → nothing added.
   - a resolvable parent that is **not** eligible → nothing added (and
     confirm the original hit is untouched, not dropped).
   - a hit already flagged `co_retrieved` → skipped, not re-resolved.
   - graph-level: `build_graph()` includes the new node in the edge chain.
   Put these in `tests/test_retrieval.py` or a new
   `tests/test_co_retrieve_parent.py` — either is fine, match existing
   file organization.

## Explicitly forbidden / out of scope
- No changes to any chunker (`laws_chunker.py`, `tariff_chunker.py`,
  `nkp_chunker.py`), `pgvector_indexer.py`, or any migration. The write-side
  is correct and already ingested into the live DB — confirmed by live
  query, don't touch it.
- No reingestion, no backfill script.
- No changes to `eligibility_gate.py` or `validation_gate.py` logic itself
  — call `eligible_chunk_ids`, don't modify it.
- No changes to `app/retrieval/retrieval_orchestrator.py` — it's dead code
  (not imported by `app/main.py`, references modules that may not even
  exist). Leave it alone; it's not part of this task.
- No changes to `app/eval/gates.py` or the `make eval-gates` zero-tolerance
  triad — PS-16 is not one of the three zero-tolerance gates
  (`repealed-as-current`, `not-yet-effective-as-current`,
  `overruled-as-good-law`); don't add it there. A regular test is the
  correct enforcement mechanism here, per `AGENTS.md`'s "touches a
  fixed-invariant path → the relevant PS-* test is written/updated."

## Governing references
- `system-design.md` §14 **PS-16**.
- `docs/ingestion_design.md:55-56` (chunking design — subsection vs. proviso
  vs. tariff parent-child semantics).
- `AGENTS.md` non-negotiables: eligibility gate on every path; retrieved
  text is always untrusted (the new resolver's output goes through the same
  claim/validation path as every other hit — don't special-case it past
  that).

## Required checks
- `make test`
- `make lint`
- `make eval-gates` (must stay at 0/0/0 — this task shouldn't touch any of
  the three gates, confirm no regression)

## Self-review checklist before returning
- Does every new DB-fetched chunk pass through `eligible_chunk_ids` before
  being added to `all_hits`? (No path that skips the eligibility gate.)
- Does the new resolver only ever *add* hits, never mutate or drop existing
  ones (except the "already co_retrieved" skip in item 3)?
- Confirm live-DB assumption still holds if you re-run the audit query
  below — it should not change (you're not touching ingestion):
  ```sql
  SELECT level,
         COUNT(*) FILTER (WHERE parent_section IS NOT NULL) AS total_children,
         COUNT(*) FILTER (WHERE parent_section IS NOT NULL AND co_retrieve_parent_id IS NOT NULL) AS linked_children
  FROM chunks GROUP BY level ORDER BY level;
  ```

## Commit authorship
Every commit must be authored as `Prakash Basnet <basnetprakash090@gmail.com>`
(`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`). Never
author, co-author, or attribute any commit to Claude, Anthropic, or any AI
tool. No `Co-Authored-By: Claude` trailer or similar.

## On completion
Commit and push to `agent/co-retrieve-parent-context`. Report: commit
hash(es), changed files, checks run/results, assumptions made, remaining
risks. No extra markdown handoff files — this `task.md` is the only one.
