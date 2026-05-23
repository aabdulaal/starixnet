PROJECT=starixnet
SOURCE_OBJECTS=starixnet benchmark tests

# Auto-detect uv; fall back to assuming the user has the project's venv
# activated (or is running on a system Python where the deps are installed).
HAS_UV := $(shell command -v uv 2>/dev/null)
ifdef HAS_UV
	RUN := uv run
	SETUP := uv sync --extra dev
else
	RUN :=
	SETUP := python -m pip install -r requirements.txt && python -m pip install -e . --no-deps
endif


# -------- setup --------

setup:
	$(SETUP)

setup.uv:
	uv sync --extra dev

setup.pip:
	python -m pip install -r requirements.txt
	python -m pip install -e . --no-deps

setup.uninstall:
	rm -rf .venv

# Regenerate requirements.txt from uv.lock so non-uv users get a pinned install.
requirements:
	uv export --format requirements-txt --no-hashes --no-emit-project --extra dev -o requirements.txt


# -------- lint --------

lints: lints.ruff
lints.ci: lints.ruff.ci
lints.ruff:
	$(RUN) ruff check $(SOURCE_OBJECTS)
	$(RUN) ruff format --check $(SOURCE_OBJECTS)
lints.ruff.ci:
	$(RUN) ruff check --output-format=github $(SOURCE_OBJECTS)
lints.fix:
	$(RUN) ruff check --fix $(SOURCE_OBJECTS)
	$(RUN) ruff format $(SOURCE_OBJECTS)


# -------- test --------

test: setup test.unit test.integration
test.integration:
	$(RUN) pytest -s \
		--cov \
		--cov-fail-under=0 \
		--cov-report xml:./int-coverage.xml \
		--junitxml=./int-results.xml \
		tests/integration
test.unit:
	$(RUN) pytest -s \
		--cov \
		--cov-fail-under=80 \
		--cov-report xml:./coverage.xml \
		--cov-report term \
		--junitxml=./results.xml \
		--ignore=tests/integration

.PHONY: setup setup.uv setup.pip setup.uninstall requirements \
        lints lints.ci lints.ruff lints.ruff.ci lints.fix \
        test test.integration test.unit
