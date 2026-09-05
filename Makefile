.PHONY: setup test lint eval eval-gates stress

setup: ## Install deps and run migration
	python3 -m pip install -r requirements.txt
	python3 scripts/migrate.py

test: ## pytest tests/
	python3 -m pytest tests/

lint: ## ruff check . && ruff format --check .
	python3 -m ruff check app/main.py app/utils/helpers.py app/authority app/retrieval/__init__.py app/retrieval/eligibility_gate.py app/retrieval/precedent_retriever.py app/retrieval/postgres_retriever.py app/retrieval/gated_orchestrator.py app/retrieval/validation_gate.py app/eval/__init__.py app/eval/gates.py app/eval/romanized_slice.py app/eval/retrieval_slice.py app/eval/ragas_eval.py app/eval/phase_a_slice.py app/eval/phase_c_slice.py app/eval/phase_d_slice.py app/eval/phase_ef_slice.py app/eval/metrics/temporal_faithfulness.py tests scripts/migrate.py scripts/ingest_laws.py scripts/query.py scripts/seed_bs_ad_calendar.py
	python3 -m ruff format --check app/main.py app/utils/helpers.py app/authority app/retrieval/__init__.py app/retrieval/eligibility_gate.py app/retrieval/precedent_retriever.py app/retrieval/postgres_retriever.py app/retrieval/gated_orchestrator.py app/retrieval/validation_gate.py app/eval/__init__.py app/eval/gates.py app/eval/romanized_slice.py app/eval/retrieval_slice.py app/eval/ragas_eval.py app/eval/phase_a_slice.py app/eval/phase_c_slice.py app/eval/phase_d_slice.py app/eval/phase_ef_slice.py app/eval/metrics/temporal_faithfulness.py tests scripts/migrate.py scripts/ingest_laws.py scripts/query.py scripts/seed_bs_ad_calendar.py
	python3 -m mypy --strict --follow-imports=skip --disable-error-code=misc --disable-error-code=import-untyped app/main.py app/utils/helpers.py app/authority app/retrieval/__init__.py app/retrieval/eligibility_gate.py app/retrieval/precedent_retriever.py app/retrieval/postgres_retriever.py app/retrieval/gated_orchestrator.py app/retrieval/validation_gate.py app/eval/__init__.py app/eval/gates.py app/eval/romanized_slice.py app/eval/retrieval_slice.py app/eval/ragas_eval.py app/eval/phase_a_slice.py app/eval/phase_c_slice.py app/eval/phase_d_slice.py app/eval/phase_ef_slice.py app/eval/metrics/temporal_faithfulness.py tests scripts/migrate.py scripts/ingest_laws.py scripts/query.py scripts/seed_bs_ad_calendar.py

eval: ## per-phase RAGAS quality report + romanized Recall@5
	python3 -m app.eval.romanized_slice
	python3 -m app.eval.retrieval_slice
	python3 -m app.eval.phase_a_slice
	python3 -m app.eval.phase_c_slice
	python3 -m app.eval.phase_d_slice
	python3 -m app.eval.phase_ef_slice

eval-gates:
	python3 -m app.eval.gates

stress: ## pytest stress suite by taxonomy cell
	python3 -m pytest tests/stress/ -v
