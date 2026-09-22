#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
"${PYTHON:-python3}" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

if [[ "$(uname -s)" == "Darwin" && -d third_party/camtint ]]; then
  make -C third_party/camtint
fi

echo "Ready. Run: ./.venv/bin/python tools/make_band.py"
