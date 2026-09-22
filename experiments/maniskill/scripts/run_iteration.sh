#!/usr/bin/env bash
# One bounded environment trial plus a summary, not a scientific pass/fail gate.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
to_host_path() {
  if command -v cygpath >/dev/null; then
    cygpath -w "$1"
  elif command -v wslpath >/dev/null; then
    wslpath -w "$1"
  else
    printf '%s\n' "$1"
  fi
}
config="${1:-$workspace/configs/pickcube_cpu.json}"
venv="${VENV:-$workspace/.venv}"
if [[ -x "$venv/bin/python" ]]; then
  py="$venv/bin/python"
  config_py="$config"
elif [[ -x "$venv/Scripts/python.exe" ]]; then
  # venv uses Scripts/python.exe when the host interpreter is Windows.
  py="$venv/Scripts/python.exe"
  config_py="$(to_host_path "$config")"
else
  echo "Could not find the venv Python under $venv; run setup.sh first." >&2
  exit 1
fi
if [[ -z "${MS_ASSET_DIR:-}" ]]; then
  MS_ASSET_DIR="$workspace/.assets"
fi
if [[ -n "${config_py:-}" && "$config_py" != "$config" ]]; then
  export MS_ASSET_DIR="$(to_host_path "$MS_ASSET_DIR")"
else
  export MS_ASSET_DIR
fi
# Missing assets prompt normally; download only the selected task's assets explicitly.
out="$workspace/runs/trial-$(date -u +%Y%m%dT%H%M%SZ)-$$"
out_py="$out"
if [[ -n "${config_py:-}" && "$config_py" != "$config" ]]; then
  out_py="$(to_host_path "$out")"
fi
if "$py" -m apc_maniskill rollout --config "$config_py" --out "$out_py"; then
  "$py" -m apc_maniskill summarize "$out_py" > "$out/summary.json"
  printf 'Collected: %s\nNext: inspect episodes.jsonl and add a short iteration note.\n' "$out"
else
  status=$?
  if [[ -f "$out/manifest.json" ]]; then
    "$py" -m apc_maniskill summarize "$out_py" > "$out/summary.json" || true
  fi
  printf 'Trial could not finish; inspect %s. Existing runs were not overwritten.\n' "$out" >&2
  exit "$status"
fi
