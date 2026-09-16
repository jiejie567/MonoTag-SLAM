#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
uv venv --python 3.12 --allow-existing "$project_dir/.venv-vla"
uv pip install --python "$project_dir/.venv-vla/bin/python" \
  -r "$project_dir/requirements-vla.txt"
printf 'VLA export environment ready: %s\n' "$project_dir/.venv-vla/bin/python"
