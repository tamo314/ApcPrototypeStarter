"""Fresh-process checkpoint parity for CNP repair candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from apc.cnp.contracts import SelectArguments, SetState
from apc.cnp.repair import CorrectedConditionalSelectPrimitive


def write_probe(
    model: torch.nn.Module,
    *,
    batches: dict[str, tuple[SetState, SelectArguments]],
    path: Path,
    device: torch.device,
) -> None:
    """Persist fixed inputs and the in-process outputs required for parity."""

    probe: dict[str, dict[str, torch.Tensor]] = {}
    model.eval()
    with torch.inference_mode():
        for name, (state, arguments) in batches.items():
            result = model(state, arguments)
            probe[name] = {
                "values": state.values.detach().cpu(),
                "valid": state.valid.detach().cpu(),
                "item_ids": state.item_ids.detach().cpu(),
                "query": arguments.query.detach().cpu(),
                "threshold": arguments.threshold.detach().cpu(),
                "expected_logits": result.logits.detach().cpu(),
                "expected_selected": result.selected.detach().cpu(),
            }
    torch.save(probe, path)


def verify_checkpoint(
    checkpoint_path: Path, probe_path: Path, *, device: torch.device
) -> dict[str, object]:
    """Load a candidate independently and compare logits and masks against a probe."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise ValueError("candidate checkpoint has no model state dictionary")
    model = CorrectedConditionalSelectPrimitive(0)
    if any(name.startswith("adapter.") for name in state_dict):
        model.attach_adapter()
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    if not isinstance(probe, dict):
        raise ValueError("parity probe payload must be a mapping")
    max_difference = 0.0
    masks_match = True
    with torch.inference_mode():
        for name, payload in probe.items():
            if not isinstance(name, str) or not isinstance(payload, dict):
                raise ValueError("invalid parity probe entry")
            state = SetState(
                values=payload["values"].to(device),
                valid=payload["valid"].to(device),
                item_ids=payload["item_ids"].to(device),
            )
            arguments = SelectArguments(
                query=payload["query"].to(device), threshold=payload["threshold"].to(device)
            )
            result = model(state, arguments)
            max_difference = max(
                max_difference,
                float((result.logits.cpu() - payload["expected_logits"]).abs().max()),
            )
            masks_match = masks_match and bool(
                torch.equal(result.selected.cpu(), payload["expected_selected"])
            )
    return {
        "status": "PASS" if masks_match and max_difference <= 1e-6 else "FAIL",
        "max_logit_abs_difference": max_difference,
        "selection_masks_match": masks_match,
        "atol": 1e-6,
        "rtol": 1e-5,
        "device": str(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an independently loaded CNP repair candidate"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = verify_checkpoint(
        arguments.checkpoint, arguments.probe, device=torch.device("cuda:0")
    )
    arguments.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
