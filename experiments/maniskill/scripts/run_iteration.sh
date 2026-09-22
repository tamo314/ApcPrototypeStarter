#!/usr/bin/env bash
# One bounded environment trial plus a summary, not a scientific pass/fail gate.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:-$workspace/configs/pickcube_cpu.json}"
venv="${VENV:-$workspace/.venv}"
py="$venv/bin/python"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$workspace/.assets}"
# Missing assets prompt normally; download only the selected task's assets explicitly.
out="$workspace/runs/trial-$(date -u +%Y%m%dT%H%M%SZ)-$$"
if "$py" -m apc_maniskill rollout --config "$config" --out "$out"; then
  "$py" -m apc_maniskill summarize "$out" > "$out/summary.json"
  printf 'Collected: %s\nNext: inspect episodes.jsonl and add a short iteration note.\n' "$out"
else
  status=$?
  if [[ -f "$out/manifest.json" ]]; then
    "$py" -m apc_maniskill summarize "$out" > "$out/summary.json" || true
  fi
  printf 'Trial could not finish; inspect %s. Existing runs were not overwritten.\n' "$out" >&2
  exit "$status"
fi
