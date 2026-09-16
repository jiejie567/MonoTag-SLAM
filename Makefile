PYTHON ?= python3
VENV ?= .venv
DEPS_PREFIX ?=
JOBS ?= 4

.PHONY: help setup smoke native native-smoke

help:
	@printf '%s\n' \
	  'make setup                         Create the Python environment' \
	  'make smoke                         Run the portable release gate' \
	  'make native DEPS_PREFIX=PATH       Build the native backend' \
	  'make native-smoke DEPS_PREFIX=PATH Run native regression binaries'

setup:
	PYTHON="$(PYTHON)" bash setup.sh

smoke:
	@test -x "$(VENV)/bin/python" || { echo 'Run make setup first.' >&2; exit 2; }
	"$(VENV)/bin/python" scripts/run_release_smoke.py

native:
	@test -n "$(DEPS_PREFIX)" || { echo 'Set DEPS_PREFIX to the dependency install prefix.' >&2; exit 2; }
	bash scripts/build_linux.sh "$(DEPS_PREFIX)" "$(JOBS)"

native-smoke:
	@test -n "$(DEPS_PREFIX)" || { echo 'Set DEPS_PREFIX to the dependency install prefix.' >&2; exit 2; }
	bash scripts/run_native_smoke.sh "$(DEPS_PREFIX)"
