from __future__ import annotations

from pathlib import Path

from apc.cnp.confirmation import _checkpoint_path, _g2, _quantile_linear


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
