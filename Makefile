.PHONY: test lint format-check security audit check

test:
	PYTHONDONTWRITEBYTECODE=1 python -m pytest --cov --cov-report=term-missing

lint:
	python -m ruff check .

format-check:
	python -m ruff format --check .

security:
	python -m bandit -q -r config.py detector.py main.py pipeline.py security.py server.py utils -s B104

audit:
	python -m pip_audit -r requirements.lock

check: format-check lint test security audit
