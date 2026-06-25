"""Configuration save/load utilities."""

import json
import os
from pathlib import Path


def save_config(experiment_dir: Path, config: dict) -> None:
    """Save the configuration to a JSON file.

    Args:
        experiment_dir: Directory to save the config file in
        config: Dictionary containing configuration parameters
    """
    config_copy = config.copy()

    # Convert any Path objects to strings for JSON serialization
    for key, value in config_copy.items():
        if isinstance(value, Path):
            config_copy[key] = str(value)

    with Path(experiment_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_copy, f, indent=4)


def resolve_config_path(value: str | Path) -> Path:
    """Resolve paths stored in experiment configs.

    Archived public configs may use environment-variable placeholders such as
    `${DATASETS_DIR}` instead of machine-specific absolute paths.
    """
    raw = str(value)
    expanded = os.path.expandvars(raw)
    if "$" in expanded:
        raise RuntimeError(
            f"Could not resolve config path {raw!r}. "
            "Set the referenced environment variable, usually DATASETS_DIR."
        )
    return Path(expanded).expanduser()
