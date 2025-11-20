PROJECT=starixnet
SOURCE_OBJECTS=starixnet
PYTHON_VERSION=3.11.12


lints.ci: lints.flake8.ci
lints: lints.flake8
lints.flake8:
	poetry run flake8 ${SOURCE_OBJECTS}
lints.flake8.ci:
	poetry run flake8 --output-file=output.txt --exit-zero ${SOURCE_OBJECTS}


setup: setup.sysdeps setup.python setup.project
setup.uninstall:
	poetry env remove ${PYTHON_VERSION} || true
setup.project:
	@poetry env use $$(python -c "import sys; print(sys.executable)")
	@echo "Active interpreter path: $$(poetry env info --path)/bin/python"
	poetry install
setup.python.activation:
	@asdf local python ${PYTHON_VERSION} >/dev/null 2>&1 /dev/null || true
setup.python: setup.python.activation
	@echo "Active Python version: $$(python --version)"
	@echo "Base Interpreter path: $$(python -c 'import sys; print(sys.executable)')"
	@test "$$(python --version | cut -d' ' -f2)" = "${PYTHON_VERSION}" \
        || (echo "Please activate python ${PYTHON_VERSION}" && exit 1)
setup.sysdeps:
	asdf plugin update --all
	@for p in $$(cut -d" " -f1 .tool-versions | sort | tr '\n' ' '); do \
        asdf plugin add $$p || true; \
        done;
	asdf install


test: setup test.unit test.integration
test.integration:
	poetry run pytest -s \
		--cov \
		--cov-fail-under=0 \
		--cov-report \
		xml:./int-coverage.xml \
		--junitxml=./int-results.xml \
		tests/integration
test.unit:
	poetry run pytest -s \
		--cov \
		--cov-fail-under=80 \
		--cov-report xml:./coverage.xml \
		--cov-report term \
		--junitxml=./results.xml \
		--ignore=tests/integration