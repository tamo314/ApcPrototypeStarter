from __future__ import annotations

from apc.cnp.development import _g1, _source_condition, _source_length


def test_source_schedule_is_deterministic_and_balances_conditions_and_lengths() -> None:
    draws = 64 * 3 * 5
    conditions = [_source_condition(draw) for draw in range(draws)]
    lengths = [_source_length(draw) for draw in range(draws)]
    assert conditions == [_source_condition(draw) for draw in range(draws)]
    assert len(set(conditions)) == 64 * 3
    assert {length: lengths.count(length) for length in set(lengths)} == {
        1: 192,
        2: 192,
        4: 192,
        8: 192,
        16: 192,
    }


def test_g1_requires_each_length_threshold_cell_to_meet_both_floors() -> None:
    passing = {
        "length": 1,
        "threshold": 0.5,
        "metrics": {"balanced_accuracy": 0.95, "mean_set_f1": 0.90},
    }
    failing = {
        "length": 16,
        "threshold": 1.1,
        "metrics": {"balanced_accuracy": 0.99, "mean_set_f1": 0.89},
    }
    assert _g1([passing])["status"] == "PASS"
    result = _g1([passing, failing])
    assert result["status"] == "FAIL"
    assert result["failed_cells"] == [
        {"length": 16, "threshold": 1.1, "balanced_accuracy": 0.99, "mean_set_f1": 0.89}
    ]
