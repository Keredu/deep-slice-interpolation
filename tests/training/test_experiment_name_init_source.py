"""Tests for experiment naming with transfer-initialization config."""

from phd.training.experiments import generate_experiment_name


class TestExperimentNameInitSource:
    """Tests for generate_experiment_name with init_from_experiment fields."""

    @staticmethod
    def _base_config() -> dict:
        return {
            "loss": {"name": "msssim+l1", "params": {"msssim_weight": 0.5, "l1_weight": 0.5}},
            "optimizer": {"name": "adamw", "params": {"lr": 8e-4}},
            "scheduler": {"name": "cosine_warmup", "params": {"warmup_epochs": 5, "eta_min": 1e-6}},
            "batch_size": 96,
            "num_epochs": 500,
            "early_stopping_patience": 15,
        }

    def test_init_source_changes_name(self) -> None:
        """Two otherwise-equal configs should differ when init source differs."""
        base = self._base_config()
        with_init = {**base, "init_from_experiment": "msssim+l1_lr8e-4_e6d845"}

        assert generate_experiment_name(base) != generate_experiment_name(with_init)

    def test_none_init_source_preserves_existing_hash(self) -> None:
        """Explicit None should hash like an absent init source."""
        base = self._base_config()
        with_none = {**base, "init_from_experiment": None}

        assert generate_experiment_name(base) == generate_experiment_name(with_none)

