.PHONY: setup test lint eval eval-gates stress

setup: ## Install deps, run migration, start OpenSearch, create index
	python3 -m pip install -r requirements.txt
	python3 scripts/migrate.py
	@if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then \
		docker compose up -d opensearch; \
		for i in $$(seq 1 60); do \
			curl -fsS "$${OPENSEARCH_URL:-http://localhost:9200}" >/dev/null 2>&1 && break; \
			sleep 2; \
		done; \
		curl -fsS "$${OPENSEARCH_URL:-http://localhost:9200}" >/dev/null; \
		python3 -m app.search.client; \
	else \
		echo "Docker unavailable; skipping local OpenSearch startup"; \
	fi

test: ## pytest tests/
	python3 -m pytest tests/

lint: ## ruff check . && ruff format --check .
	python3 -m ruff check app/authority app/search app/retrieval/__init__.py app/retrieval/eligibility_gate.py app/retrieval/dumb_retriever.py app/retrieval/validation_gate.py app/eval tests scripts/migrate.py scripts/ingest_laws.py scripts/query.py
	python3 -m ruff format --check app/authority app/search app/retrieval/__init__.py app/retrieval/eligibility_gate.py app/retrieval/dumb_retriever.py app/retrieval/validation_gate.py app/eval tests scripts/migrate.py scripts/ingest_laws.py scripts/query.py
	python3 -m mypy --strict --follow-imports=skip --disable-error-code=misc app/authority app/search app/retrieval/__init__.py app/retrieval/eligibility_gate.py app/retrieval/dumb_retriever.py app/retrieval/validation_gate.py app/eval tests scripts/migrate.py scripts/ingest_laws.py scripts/query.py

eval: ## no cases yet
	@python3 -c 'print("no eval cases yet")'

eval-gates:
	python3 -m app.eval.gates

stress: ## no cases yet
	@python3 -c 'print("no stress cases yet")'
