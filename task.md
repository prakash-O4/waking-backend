# task.md — AGENT-43: fix needless abstention from the quote-support check

## How this was found

Following on from AGENT-42 (which fixed a JSON-parse bug), Prakash posed
a real question through the console: "List me the basic rights of
labor". The Langfuse trace shows `_structured_claims` (the reasoner)
now works correctly and found three genuinely on-topic claims — दफा ३,
दफा ९, and दफा १५३ of श्रम/रोजगारी legislation. But `validation` shows
`claims_passed: 0, claims_abstained: 3` — **every single claim was
rejected**, and the final `/ask/stream` response correctly (per the
gate's own logic) came back as a full abstain (`abstained: true,
relevant_sections: []`). The gate did its job — but the underlying law
existed and should have been citable. Two independent, unrelated causes
were confirmed live, and this task fixes both.

## Root cause 1 — the reasoner splices non-adjacent text with "..."

For the दफा ३ claim, `_structured_claims` returned:

```
"quote": "३. रोजगारीको अधिकार: (१) प्रत्येक नागरिकलाई रोजगारी पाउने अधिकार हुनेछ । (२) ... उचित श्रम अभ्यासको अधिकार हुनेछ ।"
```

The real chunk text has full words between subsection (१) and (२), not
literal `...` — so this is not a genuine contiguous substring of the
source and `_claim_supported()` correctly rejects it.

**Confirmed empirically** (not guessed): re-ran the exact real
production chunk (दफा ३, `component_uri
236df29b-012c-4c65-a655-70e6c495aebf` — pulled verbatim from the live
`chunks` table) through `_structured_claims` three times with the
current prompt. Two runs truncated to only subsection (१), one run
spliced (१) and (२) together with `...` exactly as seen in the original
trace — confirming this is a real, reproducible weakness in the current
prompt's quote instructions, not a one-off fluke.

## Root cause 2 — the quote-support check chokes on the corpus's own markdown

**This is the more foundational bug — confirmed directly against the
live database, and it likely affects most citations of a दफा heading
across the whole corpus, not just this multi-subsection case.**

Every दफा/धारा heading in `chunks.chunk_text` is wrapped in literal
markdown bold syntax by the ingestion pipeline convention — e.g. the
real, live row for the दफा ३ chunk above reads (pulled directly from
Postgres):

```
**३. रोजगारीको अधिकार:** (१) प्रत्येक नागरिकलाई रोजगारी पाउने अधिकार हुनेछ ।

(२) उपदफा (१) को प्रयोजनको लागि प्रत्येक नागरिकलाई यो ऐन वा प्रचलित कानूनको अधीनमा रही उचित श्रम अभ्यासको अधिकार हुनेछ ।
```

The `**` characters are literal bytes in the stored `chunk_text` — not
rendering markup applied later. `app/ingestion/laws_chunker.py`'s
`_DAFA_HEADING_RE = re.compile(r"\*\*([०-९]+)\.\s*([^\n*]+?)\*\*")`
generates/matches this convention, and `app/ingestion/pipeline.py`'s
`_DAFA_ANCHOR_RE` **rejects the entire source document at ingestion
time** if no `**N.` anchor is found — so `**` is a reserved structural
marker, guaranteed never to appear as real legal-text content anywhere
in the corpus. The same is true of `## परिच्छेद-N` / `## भाग-N` chapter
markers (`_CHAPTER_RE` in both `laws_chunker.py` and
`tariff_chunker.py`) — these can appear *inside* a दफा's `chunk_text`
when a chapter boundary falls mid-chunk (confirmed: this is exactly
what an earlier trace showed for a different section — `"...\n\n##
परिच्छेद-३\n\nरोजगार सेवा केन्द्र"` trailing inside a दफा ९ chunk).

The LLM naturally does **not** reproduce `**`/`##` decoration when
quoting the substantive legal text (it reads as formatting, not
content) — but `_normalize()` in `app/retrieval/validation_gate.py`
does no markdown-stripping at all, so any otherwise-perfect, fully
verbatim, non-spliced quote of a दफा heading **still fails** the
exact-substring check purely because of these formatting bytes.
**Confirmed empirically**: took a clean, complete, non-spliced quote of
the दफा ३ text and confirmed it fails `_claim_supported()` against the
real DB row until `**` is stripped from the chunk side, at which point
it passes.

**What must NOT be touched, confirmed via the chunker/test fixtures**:
- `<amend>...</amend>` tags and `✂` elision marks (`laws_chunker.py`)
  are provenance markup, not decoration — PS-10 requires they survive
  in `chunk_text`. Never strip these.
- Leading `-`/`--` inside tariff goods-description cells (see
  `tests/test_tariff_chunker.py`'s fixtures, e.g. `-दुरम गहुँ:`,
  `--बिउ`) are real legal content (HS-code indentation), not markdown
  list syntax. Never strip leading dashes.
- `|`-table syntax in tariff chunks is the actual stored row content
  for tariff-dominant acts — out of scope, not touched by this fix
  (this task only strips `**` and `##`, nothing else).

## Objective

1. Make `_normalize()` in `app/retrieval/validation_gate.py` strip
   literal `**` and `##` sequences (in addition to its existing NFC
   normalization, Devanagari-digit folding, and whitespace collapsing)
   before the exact-substring comparison in `_claim_supported()`. A
   plain `str.replace("**", "").replace("##", "")` is sufficient and
   safest — no regex, no risk of over-matching, since both sequences
   are confirmed never to appear in real legal-text content anywhere in
   the corpus (see grounding above). Apply this inside `_normalize()`
   itself so it symmetrically affects both the `quote` and the
   `chunk_text` side of every comparison in `_claim_supported()` —
   whether or not the model happens to include the decoration in its
   quote, the match now works either way.

2. Tighten the `"quote"` field instructions in `_structured_claims`'s
   system prompt (`app/retrieval/gated_orchestrator.py`) to stop the
   model from splicing non-adjacent text with `...`. Empirically
   verified replacement wording (confirmed against the real live model,
   3/3 clean runs on the exact chunk that previously failed):

   Replace:
   ```
   "quote": "<short exact contiguous substring copied verbatim from the CONTEXT chunk you cite in evidence_id — not a paraphrase, not a translation, not assembled from multiple places — that supports the claim>",
   ```
   with:
   ```
   "quote": "<exact contiguous substring copied verbatim, character-for-character, from the CONTEXT chunk you cite in evidence_id — not a paraphrase, not a translation. Never use '...' or any other joiner to skip words inside a quote: quote one single unbroken run of text exactly as it appears, even if that means including a few extra words. If the supporting text comes from two or more non-adjacent parts of the same chunk (e.g. two different subsections), output a SEPARATE claim object for each part instead of combining them into one quote>",
   ```
   (Dropping the word "short" is deliberate — it was nudging the model
   toward truncating/splicing to stay brief; the new wording explicitly
   permits a longer quote over a spliced one.)

## Acceptance criteria

- `_normalize()` strips `**` and `##` (literal substring removal, not
  regex) in addition to its existing behavior. `_claim_supported()`'s
  matching semantics stay a strict exact-substring check — this is a
  normalization correction, not a loosening of the anti-hallucination
  guarantee: the actual words of the law must still match exactly,
  character-for-character, after stripping only bytes that are
  confirmed structural decoration, never real content.
- `_structured_claims`'s system prompt updated as specified above.
- New tests in `tests/test_validation_gate.py`:
  - A quote that omits `**...**` around a दफा heading, compared against
    a `chunk_text` fixture that includes it, is now `_claim_supported()
    == True` (previously `False`).
  - A quote compared against a `chunk_text` fixture containing a
    trailing `## परिच्छेद-N` line still matches correctly when the
    quote doesn't include that line.
  - A negative-control test: a tariff-style `chunk_text` fixture with a
    leading `-`/`--` in the actual legal content (e.g. `-दुरम गहुँ:`)
    is unaffected — `_normalize()` must not alter or strip those dashes
    (guards against a future regex-based reimplementation accidentally
    touching real content).
  - `<amend>`/`</amend>` tags and `✂` marks stay untouched by
    `_normalize()` (existing tests may already cover this indirectly —
    confirm, and add a direct case if not).
- Existing `tests/test_validation_gate.py` tests (plain-prose fixtures,
  no markdown) must keep passing unmodified — this change is additive.
- Manual empirical re-verification against the real Azure endpoint (the
  established standard this session — see AGENT-42's entry in
  `.agent/PROGRESS.md`): re-run the real दफा ३ chunk from the live
  `chunks` table through the actual wired-in `_structured_claims()`
  with the updated prompt, confirm the returned quote now passes
  `_claim_supported()` against the real chunk text with the fixed
  `_normalize()`. (Claude already did exactly this standalone and
  confirmed it works — this step re-confirms the fully wired-in code,
  not the standalone mechanism.)
- `make test`, `make lint` — **run via `.venv/bin/python`, not bare
  `python3`** (the Makefile's bare `python3` resolves to system Python
  3.9.6 on this machine and will produce spurious unrelated failures —
  see AGENT-40/AGENT-42's entries in `.agent/PROGRESS.md`).

## Explicitly forbidden

- Touching `_MIN_QUOTE_CHARS` or `_claim_supported()`'s core
  exact-substring semantics — this task corrects what gets compared,
  not how strict the comparison is.
- Touching `<amend>`/`</amend>` handling, `✂` elision handling, or any
  tariff-chunk dash/table content — all confirmed real
  content/provenance, not decoration.
- Touching `app/ingestion/laws_chunker.py`, `app/ingestion/
  tariff_chunker.py`, or any ingestion/chunking code — the markdown
  convention in storage is intentional and correct; this task only
  changes how the *comparison* handles it, never the stored format.
- Touching `_extractive_claim()`'s fallback logic (separate, already
  deferred question from AGENT-42).
- Touching `_compose_answer`'s or `_fact_extract`'s prompts — neither
  is affected by this bug class.
- Any new dependency, regex-based markdown parser, or fuzzy/similarity
  matching — a plain literal substring strip is sufficient and is the
  minimal, safest fix per the grounding above.

## Governing references

- `AGENTS.md` prime directive: abstaining when an answer actually
  exists in the corpus is still a quality failure, even though it's the
  *safe* failure mode — this task closes a real, avoidable gap, it does
  not touch the safety property itself.
- `system-design.md` §2, Core Invariant 4 ("server-side validation gate
  ... verifies claim support ... only then renders citations") and
  Core Invariant 8 ("all retrieved text is untrusted") — both stay
  fully intact; this task does not weaken what counts as "supported,"
  it fixes what the check is actually comparing.
- PS-6 (per-claim as-of validation) and PS-16 (provisos/स्पष्टीकरण
  co-retrieve with their operative clause) are adjacent but untouched.

## Required checks

- `.venv/bin/python -m pytest tests/` (full suite)
- `.venv/bin/python -m ruff check` / `ruff format --check` / `mypy
  --strict` on the exact Makefile file list (see `Makefile`'s `lint`
  target for the full list — copy it verbatim, just substitute
  `.venv/bin/python` for `python3`)
- Manual live-Azure re-verification as described above

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — never Claude, Anthropic, Pi, or any AI
attribution. Enforce via:
`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`
No `Co-Authored-By` trailers, no "Generated with" lines.
