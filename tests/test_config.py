"""Config loader/saver round-trip and error-handling tests."""

from pathlib import Path

import pytest

from apc.utils.config import load_config, save_config


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    config = {"seed": 42, "model": {"hidden_size": 192, "layers": 4}}
    config_path = tmp_path / "nested" / "config.yaml"

    save_config(config, config_path)
    loaded = load_config(config_path)

    assert loaded == config


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_non_mapping_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_path)


def test_load_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_config(config_path) == {}
