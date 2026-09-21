from __future__ import annotations

import json
from pathlib import Path

from apc.cnp.confirmation import (
    _checkpoint_path,
    _confirmation_boundary_v2,
    _g2,
    _load_v2_confirmation_spec,
    _quantile_linear,
)


def _passing_row() -> dict[str, object]:
    return {
        "length": 1,
        "threshold": 0.5,
        "metrics": {"balanced_accuracy": 0.96, "mean_set_f1": 0.91},
        "exemplar_f1_p10": 0.81,
        "causal": {
            "controls": {
                name: {
                    "effectful_items": 32,
                    "effectful_sets": 32,
                    "correct": 0.96,
                    "correct_minus_intervention": 0.50,
                }
                for name in ("wrong_family", "wrong_argument", "none")
            }
        },
    }


def test_linear_exemplar_quantile_interpolates_between_sorted_values() -> None:
    assert _quantile_linear([0.0, 0.4, 1.0], 0.25) == 0.2


def test_g2_requires_the_wrong_argument_gap_without_mean_rescue() -> None:
    fresh = {"status": "PASS", "logits_exact": True, "mask_exact": True}
    assert _g2([_passing_row()], fresh)["status"] == "PASS"
    failing = _passing_row()
    causal = failing["causal"]
    assert isinstance(causal, dict)
    controls = causal["controls"]
    assert isinstance(controls, dict)
    wrong_argument = controls["wrong_argument"]
    assert isinstance(wrong_argument, dict)
    wrong_argument["correct_minus_intervention"] = 0.49
    result = _g2([failing], fresh)
    assert result["status"] == "FAIL"
    assert result["failed_cells"][0]["causal"]["minimum_gap"] == 0.49


def test_raw_distance_correction_uses_the_fixed_1000_step_checkpoint() -> None:
    root = Path("source")
    assert _checkpoint_path(root, "RAW_DISTANCE_FIT", 610200).name == "step_1000.pt"
    assert _checkpoint_path(root, "CONDITIONAL_MLP", 610200).name == "step_4000.pt"


def test_v2_confirmation_spec_locks_the_v1_source_and_unused_panel() -> None:
    spec_path = Path("configs/cnp/v2_confirmation.json")
    spec, _, source_config, source_config_path = _load_v2_confirmation_spec(spec_path)
    assert source_config_path == Path("configs/cnp/v1.json")
    assert spec["new_training_steps"] == 0
    assert source_config["program"] == "cnp_v1"
    boundary = _confirmation_boundary_v2(
        source_config,
        {
            "source_query_count": 64,
            "opened_v1_confirmation_query_count": 32,
            "v2_confirmation_query_count": 32,
        },
    )
    assert boundary["status"] == "PASS"
    assert json.loads(spec_path.read_text(encoding="utf-8"))["panel"]["role"] == "confirm_v2_eval"
