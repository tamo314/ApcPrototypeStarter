from __future__ import annotations

import torch

from apc.evaluation import (
    mirror_cross_position_cross_length_score_gradient_interference_audit as audit,
)


def test_fixed_census_is_unique_and_covers_the_preregistered_interval() -> None:
    assert len(audit.REC004R_STEPS) == 128
    assert len(set(audit.REC004R_STEPS)) == 128
    assert audit.REC004R_STEPS[0] == 6001
    assert audit.REC004R_STEPS[-1] < 12001


def test_aggregate_preserves_target_same_and_cross_length_partitions() -> None:
    strata = {
        f"length{length}_position{position}": {
            "ALL_SCORE": torch.tensor([float(length * 10 + position)])
        }
        for length in range(6, 11)
        for position in range(length)
    }
    target, same, cross, all_other = audit._aggregate(strata, "ALL_SCORE", 4)
    assert target.item() == 104.0
    assert same.item() == sum(100 + pos for pos in range(10) if pos != 4)
    assert cross.item() == sum(
        length * 10 + pos for length in range(6, 10) for pos in range(length)
    )
    assert torch.equal(all_other, same + cross)


def test_operational_decision_requires_both_terminal_arms_and_successful_control() -> None:
    supported = {
        "I03_SCORE_ONLY_12000": {
            "same_length_negative_fraction": 0.6,
            "cross_length_negative_fraction": 0.1,
            "median_retention_ratio": 0.7,
        },
        "I03_CP_SCORE_12000": {
            "same_length_negative_fraction": 0.8,
            "cross_length_negative_fraction": 0.1,
            "median_retention_ratio": 0.7,
        },
        "I04_P_7000": {
            "same_length_negative_fraction": 0.59,
            "cross_length_negative_fraction": 0.1,
            "median_retention_ratio": 0.7,
        },
    }
    assert audit._decision(supported)["label"] == "CROSS_POSITION_GRADIENT_INTERFERENCE_SUPPORTED"
    supported["I04_P_7000"]["same_length_negative_fraction"] = 0.60
    assert (
        audit._decision(supported)["label"]
        == "SHARED_PARAMETER_GRADIENT_INTERFERENCE_NOT_SUPPORTED"
    )
