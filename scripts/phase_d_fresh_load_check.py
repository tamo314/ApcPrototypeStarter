"""Independent-process Phase-D candidate reload and full metric parity check.

This entry point is intentionally read-only.  It reconstructs only from the
candidate manifest and bank-structure artifact, asks ``load_bundle`` to check
every declared artifact first, then re-runs the registered evaluation cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from apc.evaluation.phase_d_executor import (  # noqa: E402
    REQUIRED_LOAD_CHECKS,
    _evaluate_causal_controls,
    _evaluate_panels,
    manifest_from_json_dict,
)
from apc.evaluation.shared_encoder_architecture_gate import (  # noqa: E402
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.primitives.bank import PrimitiveBank  # noqa: E402
from apc.utils import model_bundle as mb  # noqa: E402


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--bank-structure", type=Path, required=True)
    parser.add_argument("--expected-metrics", type=Path, required=True)
    parser.add_argument("--mode", choices=("nominal", "diagnostic"), required=True)
    args = parser.parse_args()

    candidate_record = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    expected = json.loads(args.expected_metrics.read_text(encoding="utf-8"))
    manifest = manifest_from_json_dict(candidate_record["manifest"])
    op_to_id = {str(key): int(value) for key, value in candidate_record["op_to_id"].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = mb.load_bundle(manifest, mode=args.mode, expected_primitive_count=16)

    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=manifest.model_seed, vocab_size=10, device=str(device))
    )
    core = arch.core
    core.model.load_state_dict(loaded.core_state_dict, strict=True)
    core.model.to(device)
    core.model.eval()
    for parameter in core.model.parameters():
        parameter.requires_grad_(False)

    bank = PrimitiveBank.load_manifest(args.bank_structure)
    bank.load_state_dict(
        {
            f"_primitives.{primitive_id}.{key}": value
            for primitive_id, state in loaded.primitive_state_dicts.items()
            for key, value in state.items()
        },
        strict=True,
    )
    bank.to(device)
    bank.freeze_all()
    bank.eval()

    actual = {
        "manifest": {
            "bundle_id": manifest.bundle_id,
            "content_manifest_digest": manifest.content_manifest_digest,
        },
        "panels": _evaluate_panels(core, bank, op_to_id, manifest.model_seed),
        "causal_controls": _evaluate_causal_controls(core, bank, op_to_id),
    }
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in actual
        if _canonical(expected.get(key)) != _canonical(actual.get(key))
    }
    result = {
        "separate_process": True,
        "candidate_bundle_id": manifest.bundle_id,
        "candidate_content_manifest_digest": manifest.content_manifest_digest,
        "mode": args.mode,
        "load_bundle_checks": list(loaded.checks_performed),
        "required_load_checks_present": REQUIRED_LOAD_CHECKS.issubset(loaded.checks_performed),
        "metric_mismatches": mismatches,
        "parity_pass": not mismatches
        and REQUIRED_LOAD_CHECKS.issubset(loaded.checks_performed),
        "device": str(device),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["parity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
