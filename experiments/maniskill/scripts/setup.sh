#!/usr/bin/env bash
# Isolated Python environment. Never changes system drivers or the legacy APC venv.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Environment: PYTHON=python3.12 VENV=<workspace>/.venv TORCH_SPEC=torch TORCH_INDEX_URL=<optional>'
  echo 'Creates/updates that venv, installs this workspace, and saves host diagnostics. No training.'
  exit 0
fi
if [[ $# -ne 0 ]]; then echo 'Unknown argument; use --help' >&2; exit 2; fi
python_bin="${PYTHON:-python3.12}"
venv="${VENV:-$workspace/.venv}"
command -v "$python_bin" >/dev/null || { echo "Missing $python_bin; set PYTHON=python3.11 or install Python 3.12." >&2; exit 1; }
"$python_bin" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "Use Python 3.11 or 3.12"'
if [[ ! -d "$venv" ]]; then "$python_bin" -m venv "$venv"; fi
py="$venv/bin/python"
"$py" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "Existing venv has wrong Python"'
"$py" -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  "$py" -m pip install "${TORCH_SPEC:-torch}" --index-url "$TORCH_INDEX_URL"
else
  "$py" -m pip install "${TORCH_SPEC:-torch}"
fi
"$py" -m pip install -e "$workspace[dev]"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$workspace/.assets}"
mkdir -p "$MS_ASSET_DIR"
setup_run="$workspace/runs/setup-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$setup_run"
"$py" -m pip freeze > "$setup_run/requirements.freeze.txt"
"$py" -m apc_maniskill doctor > "$setup_run/doctor.json"
printf 'Environment: %s\nDiagnostics: %s\n' "$venv" "$setup_run"
printf 'Next: VENV="%s" bash "%s/scripts/run_iteration.sh"\n' "$venv" "$workspace"
