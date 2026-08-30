"""Explicit, serializable YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file does not parse to a mapping.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {config_path}")
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a config dict to a YAML file, creating parent directories as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
