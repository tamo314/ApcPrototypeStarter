from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from apc.cnp.artifacts import CNPArtifactError, build_manifest, load_bundle, save_bundle
from apc.cnp.contracts import SelectArguments, SetState
from apc.cnp.primitive import ConditionalSelectPrimitive
from apc.cnp.seed_registry import audit_cnp_seed_registry
from apc.cnp.training import adapter_parameters, clone_local_candidate, one_training_step


def _inputs() -> tuple[SetState, SelectArguments, torch.Tensor]:
    state = SetState(
        values=torch.rand((1, 3, 8), dtype=torch.float32),
        valid=torch.tensor([[True, True, True]]),
        item_ids=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )
    arguments = SelectArguments(torch.zeros((1, 8), dtype=torch.float32), torch.tensor([0.5]))
    return state, arguments, torch.tensor([[True, False, True]])


def test_local_candidate_freezes_base_and_updates_only_adapter() -> None:
    torch.manual_seed(7)
    parent = ConditionalSelectPrimitive(3)
    candidate = clone_local_candidate(parent)
    assert candidate.adapter_parameter_count == 1024
    assert all(
        not parameter.requires_grad
        for name, parameter in candidate.named_parameters()
        if not name.startswith("adapter.")
    )
    optimizer = torch.optim.AdamW(adapter_parameters(candidate), lr=0.001, weight_decay=0.0)
    state, arguments, target = _inputs()
    before = {
        name: value.detach().clone()
        for name, value in candidate.named_parameters()
        if not name.startswith("adapter.")
    }
    one_training_step(candidate, state, arguments, target, optimizer)
    assert all(
        torch.equal(before[name], value)
        for name, value in candidate.named_parameters()
        if name in before
    )


def test_next_local_candidate_copies_one_existing_adapter_without_stacking() -> None:
    parent = clone_local_candidate(ConditionalSelectPrimitive(4))
    candidate = clone_local_candidate(parent)
    assert candidate.adapter_parameter_count == 1024
    assert candidate.adapter is not parent.adapter
    assert all(parameter.requires_grad for parameter in adapter_parameters(candidate))


def test_cnp_bundle_is_hash_checked_and_never_overwritten(tmp_path) -> None:
    config = {"program": "cnp_v1", "value": 1}
    primitive = ConditionalSelectPrimitive(5)
    manifest = build_manifest(
        primitive, model_seed=610100, config=config, usage_conditions={"lengths": [1, 2]}
    )
    directory = tmp_path / "bundle"
    save_bundle(directory, primitive, manifest)
    restored = ConditionalSelectPrimitive(5)
    loaded = load_bundle(directory, restored, expected_config=config)
    assert loaded.base_weights_hash == manifest.base_weights_hash
    with pytest.raises(CNPArtifactError, match="overwrite"):
        save_bundle(directory, primitive, manifest)
    raw = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    raw["config_hash"] = "bad"
    (directory / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CNPArtifactError, match="configuration"):
        load_bundle(directory, restored, expected_config=config)


def test_cnp_bundle_loads_in_a_fresh_process(tmp_path) -> None:
    config = {"program": "cnp_v1", "value": 2}
    primitive = ConditionalSelectPrimitive(6)
    manifest = build_manifest(
        primitive, model_seed=610101, config=config, usage_conditions={"lengths": [4]}
    )
    directory = tmp_path / "fresh_bundle"
    save_bundle(directory, primitive, manifest)
    program = "\n".join(
        (
            "import json",
            "from pathlib import Path",
            "from apc.cnp.artifacts import load_bundle",
            "from apc.cnp.primitive import ConditionalSelectPrimitive",
            f"bundle = Path({str(directory)!r})",
            "model = ConditionalSelectPrimitive(6)",
            "config = {'program': 'cnp_v1', 'value': 2}",
            "manifest = load_bundle(bundle, model, expected_config=config)",
            "result = {'schema': manifest.schema_version}",
            "result['parameters'] = model.base_parameter_count",
            "print(json.dumps(result))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {"schema": "cnp_bundle_v1", "parameters": 8449}


def test_static_seed_registry_passes_and_rejects_legacy_collision(tmp_path) -> None:
    assert audit_cnp_seed_registry()["status"] == "PASS"
    invalid = {
        "schema_version": 1,
        "seeds": {
            "world": 610000,
            "data": 610010,
            "development_model": [40],
            "confirmation_model": [610200],
        },
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="collide"):
        audit_cnp_seed_registry(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CNP CUDA parity requires CUDA")
def test_cpu_cuda_forward_masks_match() -> None:
    torch.manual_seed(8)
    cpu_primitive = ConditionalSelectPrimitive(7).eval()
    state, arguments, _ = _inputs()
    cpu_result = cpu_primitive(state, arguments)
    cuda_primitive = ConditionalSelectPrimitive(7).cuda().eval()
    cuda_primitive.load_state_dict(cpu_primitive.state_dict())
    cuda_state = SetState(
        values=state.values.cuda(), valid=state.valid.cuda(), item_ids=state.item_ids.cuda()
    )
    cuda_arguments = SelectArguments(arguments.query.cuda(), arguments.threshold.cuda())
    cuda_result = cuda_primitive(cuda_state, cuda_arguments)
    assert torch.allclose(cpu_result.logits, cuda_result.logits.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.equal(cpu_result.selected, cuda_result.selected.cpu())
