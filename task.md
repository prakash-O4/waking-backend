# AGENT-9 — TariffChunker + detection gate

## Objective

`भन्सार_महसुल_ऐन_२०८१` is a 1.3M-char tariff schedule act. The current
`LawsChunker` treats it as prose and produces 892 chunks with wrong metadata
(tariff rows inherit `section_number = ३`, rows cross HS headings). This task
adds a structure-aware tariff chunker and a detection gate that routes
tariff-dominant acts away from `LawsChunker`.

Distribution analysis on all 677 laws confirmed: only one document has
HS-code density > 5,000 (`भन्सार_महसुल_ऐन_२०८१` with 13,022 hits). The
detection signal is reliable with zero false positives at that threshold.

## Assigned branch

`agent/tariff-chunker` (base: `dev` at `7356246`)

## Acceptance criteria

1. `is_tariff_dominant(content)` returns `True` only for content with
   >5,000 HS-code-like patterns AND a tariff keyword — returns `False` for
   all prose acts, including high-pipe-ratio registration/building regulations
   with zero HS codes.
2. `TariffChunker.chunk_text(content, act_name)` produces row-level chunks:
   - Each heading row (`| १०.०१ | | गहुँ र मेसलिन |`) → `level='tariff_heading'`
   - Each subheading row (`| | १००१.११.०० | --बिउ | रु.१ |`) → `level='tariff_row'`,
     linked to its heading via `co_retrieve_parent_index` (PS-16)
   - `section_number` = HS code of that row (not a दफा number)
   - `chunk_text` = verbatim pipe-table row(s), NFC-normalized
   - `embed_text` = context-rich string built from structured fields (see below)
   - `keywords` and `relevant_questions` populated deterministically — no LLM
3. Pipeline routes tariff-dominant content to `TariffChunker`, skips
   `enrich_law_chunks`, and emits EXTRACT_METADATA span with `llm_calls=0`.
4. `make test` green, `make lint` green, `make eval-gates` green.

## Files to create / modify

| Action | File |
|---|---|
| **CREATE** | `app/ingestion/tariff_chunker.py` |
| **MODIFY** | `app/ingestion/pipeline.py` — routing condition only |

Do not touch: `laws_chunker.py`, `nkp_chunker.py`, `pgvector_indexer.py`,
`metadata_enricher.py`, `postgres_retriever.py`, or any retrieval/gate code.

## Relevant design refs

- `docs/legal_rag_ingestion_best_practices.md` — §Tariff schedule ingestion,
  §Deterministic first LLM second, §Parent-child chunks
- `system-design.md` §2 Invariant 1, §14 PS-10 and PS-16
- `app/ingestion/laws_chunker.py` — follow the same dataclass + `_emit()` pattern

## TariffChunk dataclass

Use identical field names to `LawChunk` so `PgvectorIndexer._chunk_row()` needs
no changes. Map fields as:

| Field | Tariff meaning |
|---|---|
| `chunk_index` | document-order 0-based index |
| `chunk_text` | verbatim pipe-table row(s), NFC-normalized |
| `embed_text` | context-rich string (see format below) |
| `level` | `'tariff_heading'` \| `'tariff_row'` \| `'tariff_note'` |
| `section_number` | HS code (e.g. `"१०.०१"`, `"१००१.११.००"`) |
| `section_title` | goods description text |
| `chapter_number` | भाग/part number if detectable |
| `parent_section` | heading HS code when `level='tariff_row'` |
| `co_retrieve_parent_index` | chunk_index of parent heading chunk |
| `effective_date_ad` | `None` — filled by pipeline |
| `keywords` | deterministic list |
| `relevant_questions` | deterministic list |

## embed_text format

Build from structured fields — never copy raw pipe syntax:

```
ऐन: {act_name}।
भाग {chapter}: {chapter_title}।
शीर्षक {heading_code}: {heading_description}।
उपशीर्षक {subheading_code}।
वस्तु: {goods_description}।
महसुल दर: {duty_rate}।
```

Omit lines where the field is absent.

## Deterministic keywords and relevant_questions

For `level='tariff_row'`:
```python
keywords = [x for x in [goods_description, heading_code, subheading_code, chapter_title, act_name] if x]
relevant_questions = [
    f"{goods_description} को महसुल दर कति हो?",
    f"{subheading_code} अन्तर्गत कुन वस्तु पर्छ?",
    f"{goods_description} पैठारी गर्दा महसुल कति लाग्छ?",
]
```

For `level='tariff_heading'`:
```python
keywords = [x for x in [heading_description, heading_code, chapter_title, act_name] if x]
relevant_questions = [
    f"{heading_description} को भन्सार शीर्षक कोड के हो?",
    f"शीर्षक {heading_code} अन्तर्गत कुन वस्तुहरू पर्छन्?",
]
```

## Detection function (place in tariff_chunker.py, module level)

```python
_HS_CODE_RE = re.compile(r'[०-९\d]{2,4}\.[०-९\d]{2}')
_TARIFF_KW_RE = re.compile(r'(सार्क|पैठारी|उपशीर्षक|महसुल दर)')

def is_tariff_dominant(content: str) -> bool:
    return (
        len(_HS_CODE_RE.findall(content)) > 5000
        and bool(_TARIFF_KW_RE.search(content))
    )
```

## Pipeline routing change

In `ingest_law()`, wrap the existing chunker call (~line 248):

```python
from app.ingestion.tariff_chunker import TariffChunker, is_tariff_dominant

if is_tariff_dominant(content):
    chunks = TariffChunker().chunk_text(content, act_name=record.get("name", ""))
    summary = f"{record.get('name', '')} — भन्सार महसुल दर तालिका"
    # skip enrich_law_chunks entirely; EXTRACT_METADATA span → llm_calls=0
else:
    chunks = self._laws_chunker.chunk_text(content)
    # existing enrich_law_chunks block runs unchanged
```

The EXTRACT_METADATA Langfuse span must still be emitted for tariff docs —
set `llm_calls=0`, `keywords_extracted=len(chunks)`, `summary_extracted=True`.

## PS requirements in scope

- **PS-10** — `chunk_type` for tariff chunks must be `"tariff_heading"` /
  `"tariff_row"`, not a दफा reference. Verify that `PgvectorIndexer._chunk_row()`
  picks this up from `chunk.level` without modification (it does for law chunks
  at line ~269 — confirm same path works for TariffChunk or add a branch).
- **PS-16** — heading → row linkage via `co_retrieve_parent_index` (same
  mechanism as स्पष्टीकरण → operative clause in LawsChunker).

## Zero-tolerance gates — do not touch

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

TariffChunker does not touch eligibility gate, validation gate, or any
bitemporal path.

## Required checks

```
make test
make lint
make eval-gates
```

## Tests to write

In `tests/test_tariff_chunker.py` (new file):

1. `test_is_tariff_dominant_true` — >5000 HS codes + tariff keyword → True
2. `test_is_tariff_dominant_false_prose` — normal दफा prose → False
3. `test_is_tariff_dominant_false_pipe_only` — high pipes, zero HS codes → False
4. `test_tariff_chunker_heading_row_linkage` — heading chunk at index N; row
   chunks have `co_retrieve_parent_index == N`
5. `test_tariff_chunker_embed_text_context_rich` — `embed_text` contains
   act name, HS code, goods description; no raw `|` characters
6. `test_tariff_chunker_deterministic_questions` — `relevant_questions` and
   `keywords` populated; no mock of any LLM needed
7. `test_pipeline_routes_tariff_to_tariff_chunker` — mock `is_tariff_dominant`
   → True; assert `TariffChunker.chunk_text` called, `enrich_law_chunks` NOT called

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No `Co-Authored-By`, no AI attribution of any kind.

## Return

Commit hash, changed files, checks run + results, assumptions, remaining risks
(especially: any table patterns in the actual file that the parser does not
handle — nested rows, note rows, multi-line cells).
