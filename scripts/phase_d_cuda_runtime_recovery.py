"""Record the Phase D CUDA runtime recovery precondition without APC data access.

This probe deliberately uses only a small, fixed PyTorch module.  It proves that
the isolated interpreter can execute CUDA tensors on the visible device, while
keeping cohort construction, data generation, and sealed partitions untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

SEED = 20260913
ATOL = 1e-5
RTOL = 1e-5


class RuntimeProbe(nn.Module):
    """Tiny fixed-shape module used only for the CUDA environment check."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(16, 32)
        self.out_proj = nn.Linear(32, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.nn.functional.gelu(self.in_proj(inputs)))


def _configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _digest(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _model_and_input() -> tuple[RuntimeProbe, torch.Tensor]:
    torch.manual_seed(SEED)
    model = RuntimeProbe()
    inputs = torch.randn(16, 16)
    return model, inputs


def _finite_gradients(model: nn.Module) -> bool:
    return all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def _fresh_load(checkpoint: Path) -> dict[str, Any]:
    _configure_determinism()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RuntimeProbe()
    model.load_state_dict(payload["state_dict"])
    inputs = payload["inputs"]
    with torch.no_grad():
        cpu_output = model(inputs)
        gpu_output = model.to("cuda")(inputs.to("cuda")).cpu()
    return {
        "cpu_output_digest": _digest(cpu_output),
        "gpu_output_digest": _digest(gpu_output),
        "cpu_gpu_max_abs_error": float((cpu_output - gpu_output).abs().max().item()),
    }


def run_probe(output_dir: Path) -> dict[str, Any]:
    """Run the non-APC CUDA availability, numeric, determinism, and reload checks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_determinism()
    cuda_ready = torch.backends.cuda.is_built() and torch.cuda.is_available()
    result: dict[str, Any] = {
        "seed": SEED,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_built": torch.backends.cuda.is_built(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_ready else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if cuda_ready else None,
        "cohort_construction": "NOT_PERFORMED",
        "model_data_or_sealed_access": 0,
    }
    if not cuda_ready:
        result.update({"runtime_gate": "FAIL", "failure": "cuda_unavailable"})
        return result

    cpu_model, inputs = _model_and_input()
    gpu_model = RuntimeProbe()
    gpu_model.load_state_dict(cpu_model.state_dict())
    gpu_model.to("cuda")
    with torch.no_grad():
        cpu_output = cpu_model(inputs)
        gpu_output_first = gpu_model(inputs.to("cuda")).cpu()
        gpu_output_second = gpu_model(inputs.to("cuda")).cpu()

    cpu_gpu_error = float((cpu_output - gpu_output_first).abs().max().item())
    deterministic_forward = torch.equal(gpu_output_first, gpu_output_second)

    cpu_model.zero_grad(set_to_none=True)
    gpu_model.zero_grad(set_to_none=True)
    cpu_loss = cpu_model(inputs).square().mean()
    gpu_loss = gpu_model(inputs.to("cuda")).square().mean()
    cpu_loss.backward()
    gpu_loss.backward()
    gpu_gradients = [parameter.grad.detach().cpu() for parameter in gpu_model.parameters()]
    cpu_gradients = [parameter.grad.detach() for parameter in cpu_model.parameters()]
    gradient_error = max(
        float((cpu_gradient - gpu_gradient).abs().max().item())
        for cpu_gradient, gpu_gradient in zip(cpu_gradients, gpu_gradients, strict=True)
    )
    checkpoint = output_dir / "runtime_probe_checkpoint.pt"
    torch.save({"state_dict": cpu_model.state_dict(), "inputs": inputs}, checkpoint)
    fresh_json = output_dir / "fresh_load.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--fresh-load",
            str(checkpoint),
            "--json",
            str(fresh_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    fresh: dict[str, Any] = {}
    if completed.returncode == 0 and fresh_json.exists():
        fresh = json.loads(fresh_json.read_text(encoding="utf-8"))
    fresh_load_passed = (
        completed.returncode == 0
        and fresh.get("cpu_output_digest") == _digest(cpu_output)
        and fresh.get("gpu_output_digest") == _digest(gpu_output_first)
        and fresh.get("cpu_gpu_max_abs_error", float("inf")) <= ATOL
    )
    result.update(
        {
            "cpu_gpu_output_max_abs_error": cpu_gpu_error,
            "cpu_gpu_gradient_max_abs_error": gradient_error,
            "forward_deterministic_bitwise": deterministic_forward,
            "forward_backward_finite": (
                _finite_gradients(cpu_model) and _finite_gradients(gpu_model)
            ),
            "fresh_load": fresh,
            "fresh_load_subprocess_returncode": completed.returncode,
            "fresh_load_subprocess_stderr": completed.stderr,
            "fresh_load_passed": fresh_load_passed,
        }
    )
    passed = all(
        [
            cpu_gpu_error <= ATOL,
            gradient_error <= ATOL,
            deterministic_forward,
            result["forward_backward_finite"],
            fresh_load_passed,
        ]
    )
    result["runtime_gate"] = "PASS" if passed else "FAIL"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fresh-load", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir is None) == (args.fresh_load is None):
        parser.error("provide exactly one of --output-dir or --fresh-load")
    report = _fresh_load(args.fresh_load) if args.fresh_load else run_probe(args.output_dir)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
