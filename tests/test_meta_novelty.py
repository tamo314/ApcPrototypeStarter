from __future__ import annotations

import math

import pytest

from apc.meta.novelty import NoveltyConfig, NoveltyEstimator, NoveltySignals


def test_default_config_weights_error_and_entropy_equally() -> None:
    estimator = NoveltyEstimator()
    signals = NoveltySignals(error=0.4, router_entropy=0.2)
    assert estimator.score(signals) == pytest.approx(0.6)


def test_custom_weights_scale_each_term() -> None:
    estimator = NoveltyEstimator(NoveltyConfig(alpha=2.0, gamma=0.5))
    signals = NoveltySignals(error=0.3, router_entropy=0.4)
    assert estimator.score(signals) == pytest.approx(2.0 * 0.3 + 0.5 * 0.4)


def test_entropy_normalized_by_max_router_entropy() -> None:
    estimator = NoveltyEstimator(NoveltyConfig(alpha=0.0, gamma=1.0))
    signals = NoveltySignals(
        error=0.0, router_entropy=math.log(4), max_router_entropy=math.log(4)
    )
    assert estimator.score(signals) == pytest.approx(1.0)


def test_entropy_left_unnormalized_when_max_not_given() -> None:
    estimator = NoveltyEstimator(NoveltyConfig(alpha=0.0, gamma=1.0))
    signals = NoveltySignals(error=0.0, router_entropy=1.3)
    assert estimator.score(signals) == pytest.approx(1.3)


def test_zero_max_router_entropy_falls_back_to_raw_value() -> None:
    estimator = NoveltyEstimator(NoveltyConfig(alpha=0.0, gamma=1.0))
    signals = NoveltySignals(error=0.0, router_entropy=0.7, max_router_entropy=0.0)
    assert estimator.score(signals) == pytest.approx(0.7)


def test_config_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="alpha"):
        NoveltyConfig(alpha=-1.0)
    with pytest.raises(ValueError, match="gamma"):
        NoveltyConfig(gamma=-1.0)


def test_log_entry_includes_step_and_all_signals() -> None:
    estimator = NoveltyEstimator(NoveltyConfig(alpha=1.0, gamma=1.0))
    signals = NoveltySignals(error=0.5, router_entropy=0.5, max_router_entropy=1.0)
    entry = estimator.log_entry(7, signals)
    assert entry["step"] == 7
    assert entry["error"] == 0.5
    assert entry["router_entropy"] == 0.5
    assert entry["max_router_entropy"] == 1.0
    assert entry["score"] == pytest.approx(1.0)


def test_signals_to_dict_round_trips_fields() -> None:
    signals = NoveltySignals(error=0.2, router_entropy=0.1, max_router_entropy=None)
    assert signals.to_dict() == {
        "error": 0.2,
        "router_entropy": 0.1,
        "max_router_entropy": None,
    }
