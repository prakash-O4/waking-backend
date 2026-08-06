# Task: PG-A — RAGAS evaluation slices per pipeline phase

## Branch
`PG-A/ragas-eval` (branched from `dev`)

## Objective
Add RAGAS v0.2.x evaluation to the existing eval harness so that
`make eval` produces a unified per-phase quality report alongside the
existing Recall@5 and zero-tolerance gate checks. Evaluation must cover
every live pipeline phase and must not disturb the deterministic safety
gates (`make eval-gates`).

## Acceptance criteria
- [ ] `ragas==0.2.*` added to `requirements.txt`
- [ ] `app/eval/ragas_eval.py` — shared RAGAS setup (LLM/embedding wrapper, `run_ragas()` helper)
- [ ] `app/eval/phase_a_slice.py` — Phase A (QA pipeline): Faithfulness + ResponseRelevancy
- [ ] `app/eval/phase_c_slice.py` — Phase C (romanized retrieval): ContextRecall + NonLLMContextPrecisionWithReference
- [ ] `app/eval/phase_d_slice.py` — Phase D (precedent): Faithfulness + ContextRecall (stubbed: prints "skipped — no precedent corpus" if precedent table is empty)
- [ ] `app/eval/phase_ef_slice.py` — Phase EF (summary): Faithfulness (summary vs. source chunks)
- [ ] `app/eval/metrics/temporal_faithfulness.py` — custom `TemporalFaithfulness` metric (see spec below)
- [ ] Golden sets: `app/eval/golden/phase_a_qa.json`, `app/eval/golden/phase_c_romanized.json`, `app/eval/golden/phase_d_precedent.json`
- [ ] `Makefile`: `make eval` runs all slices and prints a unified score table; existing targets unchanged
- [ ] `make test` green; `make lint` green; `make eval-gates` unchanged and green

## Governing design refs
- `AGENTS.md` §Prime directive, §Definition of done
- `system-design.md` §2 Core Invariants (esp. invariants 4 and 7)
- `system-design.md` §14 PS-13 (LLM-judge κ ≥ 0.6, per-slice and rolling)
- `system-design.md` §14 PS-6 (every claim validates against its declared as-of)
- `docs/ingestion_design.md` (PE-A pipeline context)

## PS requirements in scope
- **PS-6**: TemporalFaithfulness metric asserts as-of compliance per claim
- **PS-13**: RAGAS faithfulness IS the LLM-judge; eval must print per-slice scores

## Allowed scope
`requirements.txt`, `app/eval/`, `Makefile` only.
Do NOT touch any file outside these paths.

## Forbidden changes
- Do not modify `app/eval/gates.py` or `app/eval/romanized_slice.py` — they stay exactly as-is
- Do not add any new DB tables, migrations, or Postgres queries
- Do not modify any ingestion, retrieval, or gate code
- Do not add a `Co-Authored-By: Claude` trailer or AI attribution to any commit

## Implementation details

### 1. Dependency
Add to `requirements.txt` after `langchain-openai==1.4.1`:
```
ragas==0.2.*
```

### 2. `app/eval/ragas_eval.py` — shared setup

```python
from langchain.chat_models import init_chat_model
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from app.ingestion.config import get_settings  # verify correct import path

def get_evaluator_llm():
    settings = get_settings()
    return LangchainLLMWrapper(init_chat_model(settings.LLM_MODEL))

def get_evaluator_embeddings():
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings())

def run_ragas(dataset: EvaluationDataset, metrics: list, label: str) -> dict:
    result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False, show_progress=False)
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    print(f"\n=== {label} ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")
    return scores
```

Check where `get_settings()` is actually defined (`app/config.py` or `app/ingestion/config.py`) and use the correct import.

### 3. Phase A slice — `app/eval/phase_a_slice.py`

**Metrics**: `Faithfulness`, `ResponseRelevancy` (reference-free)

**Golden set**: `app/eval/golden/phase_a_qa.json` — 10 items:
```json
[
  {"query": "muluki aparadh samhita dafa 177 ko byawastha", "as_of": "2024-01-01", "note": "Section 177 of Criminal Code"},
  {"query": "shram ain ma bida ko byawastha", "as_of": "2024-01-01", "note": "Leave provisions in Labour Act"},
  {"query": "upabhokta sanrakshan ain anusaar phal ko adhikar", "as_of": "2024-01-01", "note": "Consumer rights under Consumer Protection Act"},
  {"query": "nijamati sewak ko sewaanibritta umera", "as_of": "2024-01-01", "note": "Retirement age for civil servants"},
  {"query": "bhrastachar niwaran ain ko danda", "as_of": "2024-01-01", "note": "Penalties under Anti-Corruption Act"},
  {"query": "bal ain anusaar balbalika ko paribhasha", "as_of": "2024-01-01", "note": "Definition of child under Children's Act"},
  {"query": "batawaran sanrakshan ain ko uddeshya", "as_of": "2024-01-01", "note": "Objectives of Environment Protection Act"},
  {"query": "samaajik suraksha kosh ko sthapana", "as_of": "2024-01-01", "note": "Social Security Fund establishment"},
  {"query": "praman ain ma sakshi ko byawastha", "as_of": "2024-01-01", "note": "Witness provisions in Evidence Act"},
  {"query": "muluki dewani samhita anusaar sambida ko prakar", "as_of": "2024-01-01", "note": "Types of contracts under Civil Code"}
]
```

**Slice logic**:
1. Load golden set
2. Connect to DB via `app.authority.writer.connect()`; if fails, print `"phase_a skipped — no DB"` and exit 0
3. For each entry: call `retrieve_postgres(conn, query, date.fromisoformat(as_of))` from `app.retrieval.postgres_retriever`
4. Build `retrieved_contexts = [h["text_ne"] for h in hits]`; if empty, skip entry
5. Generate response: use `init_chat_model(settings.LLM_MODEL)` with the same system prompt as `_model_claims` in `gated_orchestrator.py`. Parse the JSON claims and join claim texts as the response string.
6. Build `SingleTurnSample(user_input=query, response=response, retrieved_contexts=retrieved_contexts)`
7. Collect all samples into `EvaluationDataset`, call `run_ragas(..., label="Phase A — QA Pipeline")`

### 4. Phase C slice — `app/eval/phase_c_slice.py`

**Metrics**: `ContextRecall`, `NonLLMContextPrecisionWithReference`

**Golden set**: Create `app/eval/golden/phase_c_romanized.json` — copy of `romanized.json` with a `reference` field added to each entry. Do NOT modify `romanized.json`.

References to add:
```
bhrastachar niwaran ain     → "भ्रष्टाचार निवारण ऐन, २०५९"
shram ain                   → "श्रम ऐन, २०७४"
nijamati sewa ain           → "निजामती सेवा ऐन, २०४९"
praman ain                  → "प्रमाण ऐन, २०३१"
bal shram nishedh           → "बाल श्रम (निषेध तथा नियमित गर्ने) ऐन, २०५६"
upabhokta sanrakshan        → "उपभोक्ता संरक्षण ऐन, २०७५"
batawaran sanrakshan        → "वातावरण संरक्षण ऐन, २०७६"
samaajik suraksha           → "सामाजिक सुरक्षा ऐन, २०७५"
muluki aparadh samhita      → "मुलुकी अपराध संहिता, २०७४"
muluki dewani samhita       → "मुलुकी देवानी संहिता, २०७४"
```

**Slice logic**:
1. Connect to DB; if fails, print `"phase_c skipped — no DB"` and exit 0
2. For each entry: retrieve chunks, build `SingleTurnSample(user_input=query, retrieved_contexts=chunks, reference=reference)`
3. Run `run_ragas(..., label="Phase C — Romanized Retrieval")`

### 5. Phase D slice — `app/eval/phase_d_slice.py`

**Metrics**: `Faithfulness`, `ContextRecall`

**Guard**: `SELECT COUNT(*) FROM precedent` — if 0, print `"phase_d skipped — precedent table empty"` and exit 0.

**Golden set**: `app/eval/golden/phase_d_precedent.json` — 5 placeholder items:
```json
[
  {"query": "placeholder precedent query 1", "as_of": "2024-01-01", "reference": "placeholder holding", "note": "placeholder — populate when precedent corpus is ingested"},
  {"query": "placeholder precedent query 2", "as_of": "2024-01-01", "reference": "placeholder holding", "note": "placeholder — populate when precedent corpus is ingested"},
  {"query": "placeholder precedent query 3", "as_of": "2024-01-01", "reference": "placeholder holding", "note": "placeholder — populate when precedent corpus is ingested"},
  {"query": "placeholder precedent query 4", "as_of": "2024-01-01", "reference": "placeholder holding", "note": "placeholder — populate when precedent corpus is ingested"},
  {"query": "placeholder precedent query 5", "as_of": "2024-01-01", "reference": "placeholder holding", "note": "placeholder — populate when precedent corpus is ingested"}
]
```

### 6. Phase EF slice — `app/eval/phase_ef_slice.py`

**Metrics**: `Faithfulness`

**Logic**:
1. Connect to DB; if fails, print `"phase_ef skipped — no DB"` and exit 0
2. Query: `SELECT d.id, d.summary, array_agg(c.text_ne) FROM documents d JOIN chunks c ON c.document_id = d.id WHERE d.summary IS NOT NULL GROUP BY d.id LIMIT 10`
3. If result empty, print `"phase_ef skipped — no documents with summary"` and exit 0
4. For each row: `SingleTurnSample(user_input="Summarize this document.", response=summary, retrieved_contexts=list_of_chunks)`
5. Run `run_ragas(..., label="Phase EF — Summary Faithfulness")`

### 7. Custom metric — `app/eval/metrics/temporal_faithfulness.py`

Create `app/eval/metrics/__init__.py` (empty).

```python
from __future__ import annotations
import json
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
from ragas.dataset_schema import SingleTurnSample

class TemporalFaithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "temporal_faithfulness"
    _required_columns: dict = {"user_input", "response", "retrieved_contexts"}

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks=None) -> float:
        as_of = (sample.additional_metadata or {}).get("as_of", "unknown")
        context_str = "\n".join(sample.retrieved_contexts or [])
        prompt = (
            f"The legal question is answered as of {as_of}.\n"
            f"Response: {sample.response}\n"
            f"Context: {context_str}\n\n"
            "Does this response contain any claim implying a provision was in force "
            "before its commencement or after its repeal?\n"
            'Reply with JSON only: {"temporal_violation": true|false, "reason": "..."}'
        )
        result = await self.llm.agenerate([[{"role": "user", "content": prompt}]])
        try:
            parsed = json.loads(result.generations[0][0].text)
            return 0.0 if parsed.get("temporal_violation") else 1.0
        except Exception:
            return 1.0  # assume no violation on parse failure
```

### 8. Makefile update

Replace the `eval:` target:
```makefile
eval: ## per-phase RAGAS quality report + romanized Recall@5
	python3 -m app.eval.romanized_slice
	python3 -m app.eval.phase_a_slice
	python3 -m app.eval.phase_c_slice
	python3 -m app.eval.phase_d_slice
	python3 -m app.eval.phase_ef_slice
```

Extend the `lint:` target to include the new eval files. Look at the existing `lint:` target and append:
`app/eval/ragas_eval.py app/eval/phase_a_slice.py app/eval/phase_c_slice.py app/eval/phase_d_slice.py app/eval/phase_ef_slice.py app/eval/metrics/temporal_faithfulness.py`

to the `ruff check`, `ruff format --check`, and `mypy` commands.

## Zero-tolerance gates guarded
- `repealed-as-current = 0` (unchanged — `gates.py` untouched)
- `not-yet-effective-as-current = 0` (unchanged — `gates.py` untouched)
- `overruled-as-good-law = 0` (unchanged — `gates.py` untouched)

## Required checks before committing
```
make test
make lint
make eval-gates
```

## Commit authorship (mandatory)
Every commit on this branch must be authored as:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
Never add `Co-Authored-By: Claude`, `Co-Authored-By: Kimi`, or any AI attribution. Non-negotiable.

## Return to Claude
When done, return:
1. Commit hash(es)
2. Files changed
3. Checks run and results
4. Any assumptions made or risks remaining
