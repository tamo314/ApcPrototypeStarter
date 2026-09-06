"""Focused protocol and CPU smoke coverage for B-C005G."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from apc.evaluation.hard_negative_repair_gate import (
    HardNegativeRegateConfig,
    run_hard_negative_regate,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def _smoke_config(output_dir: Path) -> HardNegativeRegateConfig:
    return HardNegativeRegateConfig(
        sealed_seeds=(20,),
        development_seeds=(10,),
        bank_sizes=(16,),
        levels=(HardNegativeLevel.L0_ORTHOGONAL, HardNegativeLevel.L4_CONFUSABLE_FAMILY),
        target_operations=("SHIFT",),
        support_examples=32,
        query_examples=4,
        router_train_examples=4,
        router_steps=4,
        top_k=2,
        max_support=32,
        initial_support=32,
        support_increment=32,
        device="cpu",
        bank_checkpoint_dir=output_dir / "bank_ckpt",
        output_dir=output_dir,
    )


def test_regate_rejects_historical_or_development_overlap() -> None:
    with pytest.raises(ValueError, match="original B-C005"):
        HardNegativeRegateConfig(sealed_seeds=(0,), development_seeds=(10,))
    with pytest.raises(ValueError, match="disjoint"):
        HardNegativeRegateConfig(sealed_seeds=(20,), development_seeds=(20,))


def test_regate_writes_seal_and_rejects_post_seal_mutation(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path / "sealed")
    report = run_hard_negative_regate(config)
    assert report["task_id"] == "B-C005G"
    assert report["sealed_evaluation_partition"] == [20]
    assert (config.output_dir / "protocol.json").is_file()  # type: ignore[operator]
    assert (config.output_dir / "metrics.jsonl").is_file()  # type: ignore[operator]
    assert (config.output_dir / "report.md").is_file()  # type: ignore[operator]
    changed = replace(config, top_k=1)
    with pytest.raises(ValueError, match="post-seal"):
        run_hard_negative_regate(changed)
