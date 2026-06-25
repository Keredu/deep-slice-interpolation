"""Tests for experiment naming utilities."""

import re

from phd.training.experiments import generate_experiment_name


class TestGenerateExperimentName:
    """Tests for generate_experiment_name function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        config = {"loss": {"name": "ssim"}}
        result = generate_experiment_name(config)
        assert isinstance(result, str)

    def test_contains_loss_name(self) -> None:
        """Test that result contains the loss name."""
        config = {"loss": {"name": "ssim+l1"}}
        result = generate_experiment_name(config)
        assert result.startswith("ssim+l1_")

    def test_has_deterministic_hash_suffix(self) -> None:
        """Test that result has a 6-character hex suffix."""
        config = {"loss": {"name": "mse"}}
        result = generate_experiment_name(config)

        # Format: loss_name_lr..._hex6
        parts = result.split("_")
        assert len(parts) == 3
        assert parts[0] == "mse"
        assert parts[1].startswith("lr")
        assert len(parts[2]) == 6
        assert re.match(r"^[0-9a-f]+$", parts[2])

    def test_same_config_gives_same_name(self) -> None:
        """Test that the same config produces the same deterministic name."""
        config = {"loss": {"name": "ssim"}}
        results = [generate_experiment_name(config) for _ in range(10)]

        assert len(set(results)) == 1

    def test_different_configs_give_different_names(self) -> None:
        """Test that behavior-changing config fields affect the hash."""
        config_a = {"loss": {"name": "ssim"}, "optimizer": {"params": {"lr": 8e-4}}}
        config_b = {"loss": {"name": "ssim"}, "optimizer": {"params": {"lr": 3e-4}}}

        assert generate_experiment_name(config_a) != generate_experiment_name(config_b)

    def test_various_loss_names(self) -> None:
        """Test with various loss names."""
        loss_names = ["mse", "l1", "ssim", "msssim+l1"]
        for loss_name in loss_names:
            config = {"loss": {"name": loss_name}}
            result = generate_experiment_name(config)
            assert result.startswith(f"{loss_name}_")
