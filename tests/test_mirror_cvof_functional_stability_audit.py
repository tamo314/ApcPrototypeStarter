"""Unit and regression tests for Task B-C005REC-004Y:
I03 CVOF Functional-Stability Gate Identifiability Audit.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from apc.environments.generator import Example
from apc.evaluation import mirror_cvof_functional_stability_audit as rec004y
from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.primitives.primitive import CrossPositionLengthBiasPrimitive


@pytest.fixture(scope="module")
def shared_dataset() -> tuple[list[Example], dict[str, Any]]:
    seed = rec004y.REC004Y_SEED
    return rec004y.prepare_rec004y_dataset(seed)


def test_rec004y_dataset_disjointness_and_locking(
    shared_dataset: tuple[list[Example], dict[str, Any]],
) -> None:
    """Verifies that the cvof_functional_gate_probe_v1 dataset:
    1. Contains 1024 total examples (512 normal + 512 length10).
    2. Has zero duplicate digests internally.
    3. Has disjoint_verified=True in manifest.
    """
    examples, manifest = shared_dataset

    assert len(examples) == rec004y.REC004Y_TOTAL_EXAMPLES
    assert manifest["n_normal"] == rec004y.REC004Y_NORMAL_EXAMPLES
    assert manifest["n_length10"] == rec004y.REC004Y_LENGTH10_EXAMPLES
    assert manifest["disjoint_verified"] is True

    digests = rec004y._digest_examples(examples)
    assert len(digests) == len(examples)

    # Verify normal half vs length10 half disjointness
    normal_digests = rec004y._digest_examples(examples[: rec004y.REC004Y_NORMAL_EXAMPLES])
    l10_digests = rec004y._digest_examples(examples[rec004y.REC004Y_NORMAL_EXAMPLES :])
    assert len(normal_digests.intersection(l10_digests)) == 0


def test_rec004y_attention_reference_caching(
    shared_dataset: tuple[list[Example], dict[str, Any]],
) -> None:
    """Verifies that cache_reference_attention_7500 properly caches non-oracle
    attention distributions with correct shapes and determinism."""
    core, _, _ = rec004t._load_runtime_core(seed=rec004y.REC004Y_SEED)
    model = mpbr._new_arm_primitive(core, rec004y.REC004Y_BASE_PRIMITIVE)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)

    examples, _ = shared_dataset
    small_exs = examples[:8]

    cached_attns, manifest = rec004y.cache_reference_attention_7500(
        core, model, small_exs, batch_size=4, device=core.device
    )
    assert len(cached_attns) == 8
    assert manifest["non_oracle_boundary_enforced"] is True
    assert manifest["target_tokens_used"] is False

    for idx, ex in enumerate(small_exs):
        c_len = len(ex.input_tokens)
        o_len = len(ex.target_tokens)
        a = cached_attns[idx]
        assert a.shape == (model.n_head, o_len, c_len)
        # Probability sum across content positions should be ~1.0
        prob_sums = a.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)


def test_rec004y_parity_at_reference(
    shared_dataset: tuple[list[Example], dict[str, Any]],
) -> None:
    """Verifies that 7500 vs 7500 yields exactly zero displacement across
    all metrics M1-M4 and identical cross-entropy."""
    core, _, _ = rec004t._load_runtime_core(seed=rec004y.REC004Y_SEED)
    model = mpbr._new_arm_primitive(core, rec004y.REC004Y_BASE_PRIMITIVE)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)

    examples, _ = shared_dataset
    small_exs = examples[:16]

    cached_attns, _ = rec004y.cache_reference_attention_7500(
        core, model, small_exs, batch_size=8, device=core.device
    )

    metrics = rec004y.compute_functional_metrics(
        core,
        model,
        model,
        small_exs,
        cached_attns,
        batch_size=8,
        use_fixed_non_cvof=True,
        device=core.device,
    )

    assert metrics["M1_logit_rms"] < 1e-6
    assert metrics["M2_output_kl"] < 1e-6
    assert metrics["M3_post_ffn_displacement"] < 1e-6
    assert metrics["M4_attn_out_displacement"] < 1e-6


def test_rec004y_fixed_non_cvof_diagnostic_forward(
    shared_dataset: tuple[list[Example], dict[str, Any]],
) -> None:
    """Verifies that mutating non-CVOF parameters (e.g. readout) in the checkpoint
    has ZERO effect on the primary analysis (use_fixed_non_cvof=True), but DOES
    affect the auxiliary analysis (use_fixed_non_cvof=False)."""
    core, _, _ = rec004t._load_runtime_core(seed=rec004y.REC004Y_SEED)
    ref_model = mpbr._new_arm_primitive(core, rec004y.REC004Y_BASE_PRIMITIVE)
    assert isinstance(ref_model, CrossPositionLengthBiasPrimitive)
    import copy
    ckpt_model = copy.deepcopy(ref_model)

    # Mutate ONLY readout (a non-CVOF component) in ckpt_model
    with torch.no_grad():
        ckpt_model.readout.bias.add_(10.0)

    examples, _ = shared_dataset
    small_exs = examples[:8]
    cached_attns, _ = rec004y.cache_reference_attention_7500(
        core, ref_model, small_exs, batch_size=4, device=core.device
    )

    # Primary analysis: non-CVOF components are fixed to ref_model, so displacement must be 0
    primary_metrics = rec004y.compute_functional_metrics(
        core,
        ckpt_model,
        ref_model,
        small_exs,
        cached_attns,
        use_fixed_non_cvof=True,
        device=core.device,
    )
    assert primary_metrics["M1_logit_rms"] < 1e-6

    # Auxiliary analysis: non-CVOF components come from ckpt_model, so displacement must be > 0
    aux_metrics = rec004y.compute_functional_metrics(
        core,
        ckpt_model,
        ref_model,
        small_exs,
        cached_attns,
        use_fixed_non_cvof=False,
        device=core.device,
    )
    assert aux_metrics["M1_logit_rms"] > 1.0


def test_rec004y_virtual_rollback_positive_control(
    shared_dataset: tuple[list[Example], dict[str, Any]],
) -> None:
    """Verifies that rolling back all CVOF groups (RB_CVOF) yields exactly zero
    functional displacement against the reference model."""
    core, _, _ = rec004t._load_runtime_core(seed=rec004y.REC004Y_SEED)
    ref_model = mpbr._new_arm_primitive(core, rec004y.REC004Y_BASE_PRIMITIVE)
    assert isinstance(ref_model, CrossPositionLengthBiasPrimitive)

    s7500 = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}
    # Perturbed state 7525
    s7525 = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}
    with torch.no_grad():
        s7525["content_in_proj.weight"].add_(0.5)

    examples, _ = shared_dataset
    small_exs = examples[:8]
    cached_attns, _ = rec004y.cache_reference_attention_7500(
        core, ref_model, small_exs, batch_size=4, device=core.device
    )

    vr_results = rec004y.run_virtual_rollback_attribution(
        core,
        s7500,
        s7525,
        ref_model,
        small_exs,
        cached_attns,
        device=core.device,
    )

    # R0 has displacement > 0
    assert vr_results["R0"]["M1_logit_rms"] > 0.01

    # RB_CVOF must have displacement = 0 (within float tolerance)
    assert vr_results["RB_CVOF"]["M1_logit_rms"] < 1e-6
    assert vr_results["RB_CVOF"]["M2_output_kl"] < 1e-6
