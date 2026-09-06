.PHONY: test lint format-check security audit doctor lock check

test:
	PYTHONDONTWRITEBYTECODE=1 python -m pytest --cov --cov-report=term-missing

lint:
	python -m ruff check .

format-check:
	python -m ruff format --check .

security:
	python -m bandit -q -r alarms.py config.py detector.py face_enhancement.py main.py pipeline.py security.py server.py scripts utils -s B104

audit:
	python -m pip_audit -r requirements.lock
	python -m pip_audit -r requirements-anpr.lock
	python -m pip_audit -r requirements-face.lock
	python -m pip_audit -r requirements-dev.lock

doctor:
	python scripts/doctor.py

lock:
	uv pip compile requirements.txt --universal --python-version 3.11 --upgrade --custom-compile-command 'make lock' --output-file requirements.lock
	uv pip compile requirements-anpr.txt --universal --python-version 3.11 --upgrade --custom-compile-command 'make lock' --output-file requirements-anpr.lock
	uv pip compile requirements-face.txt --universal --python-version 3.11 --upgrade --custom-compile-command 'make lock' --output-file requirements-face.lock
	uv pip compile requirements-dev.txt --universal --python-version 3.11 --upgrade --custom-compile-command 'make lock' --output-file requirements-dev.lock

check: format-check lint test security audit
