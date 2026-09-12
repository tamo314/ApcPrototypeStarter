"""Focused invariants for REC-004AF's non-oracle control matching."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module() -> object:
    path = Path("scripts/diagnose_rec004af_matched_endpoints.py")
    spec = importlib.util.spec_from_file_location("rec004af", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matching_uses_only_endpoint_identity_and_input_conditions() -> None:
    module = _module()
    inputs = [(index * 4, 0, 2, 3, 4) for index in range(20)]
    controls, rows = module.build_preregistered_controls("fixed_split", inputs)
    assert len(rows) == len(inputs) * 5
    for (example_index, _), selected in controls.items():
        assert len(selected) == 4
        assert example_index not in selected
        assert len(set(selected)) == 4
    assert {row["relation"] for row in rows} == {"MIRROR_HALVES"}
    assert all("target" not in key and "oracle" not in key for row in rows for key in row)


def test_matching_stops_when_fixed_control_pool_is_too_small() -> None:
    module = _module()
    inputs = [(0, 1, 2, 3, 4), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4)]
    try:
        module.build_preregistered_controls("tiny_split", inputs)
    except RuntimeError as exc:
        assert "MATCH_POOL_TOO_SMALL" in str(exc)
    else:
        raise AssertionError("undersized preregistered pool must stop")
