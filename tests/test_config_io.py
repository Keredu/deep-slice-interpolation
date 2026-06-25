"""Tests for config_io module."""

import json
from pathlib import Path

from phd.config_io import save_config


class TestSaveConfig:
    """Tests for save_config function."""

    def test_saves_config_file(self, tmp_path: Path) -> None:
        """Test that save_config creates config.json."""
        config = {"learning_rate": 0.001, "batch_size": 64}
        save_config(tmp_path, config)

        config_path = tmp_path / "config.json"
        assert config_path.exists()

    def test_converts_path_to_string(self, tmp_path: Path) -> None:
        """Test that Path objects are converted to strings."""
        config = {"data_path": Path("/some/path"), "batch_size": 64}
        save_config(tmp_path, config)

        with (tmp_path / "config.json").open() as f:
            loaded = json.load(f)
        assert loaded["data_path"] == "/some/path"
        assert isinstance(loaded["data_path"], str)

    def test_preserves_values(self, tmp_path: Path) -> None:
        """Test that all values are preserved after save/load."""
        config = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }
        save_config(tmp_path, config)

        with (tmp_path / "config.json").open() as f:
            loaded = json.load(f)

        assert loaded["learning_rate"] == 0.001
        assert loaded["batch_size"] == 64
        assert loaded["nested"] == {"key": "value"}
        assert loaded["list"] == [1, 2, 3]
