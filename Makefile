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
	python3 -m ruff check app/authority app/search tests scripts/migrate.py
	python3 -m ruff format --check app/authority app/search tests scripts/migrate.py
	python3 -m mypy --strict app/authority app/search tests scripts/migrate.py

eval: ## no cases yet
	@python3 -c 'print("no eval cases yet")'

eval-gates: ## no gate cases yet
	@python3 -c 'print("no eval gate cases yet")'

stress: ## no cases yet
	@python3 -c 'print("no stress cases yet")'
