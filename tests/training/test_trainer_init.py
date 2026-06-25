"""Tests for Trainer.__init__ initialization."""


from phd.datasets.interpolation.two_to_one_slice_cropped import NUM_CROPS
from phd.training.trainer import Trainer


class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_init_sets_config(self, minimal_config: dict) -> None:
        """Test that config is stored as attribute."""
        trainer = Trainer(minimal_config)
        assert trainer.config == minimal_config
        assert trainer.config["exp_name"] == "test_experiment"

    def test_init_core_components_none(self, trainer: Trainer) -> None:
        """Test that core training components start as None."""
        assert trainer.device is None
        assert trainer.model is None
        assert trainer.criterion is None
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer.early_stopping is None
        assert trainer.scaler is None

    def test_init_dataloaders_none(self, trainer: Trainer) -> None:
        """Test that dataloaders start as None."""
        assert trainer.train_loader is None
        assert trainer.valid_loader is None
        assert trainer.test_dataset_target_is_real is None
        assert trainer.test_dataset_target_is_interpolated is None

    def test_init_training_state_defaults(self, trainer: Trainer) -> None:
        """Test training state has correct defaults."""
        assert trainer.start_epoch == 0
        assert trainer.best_valid_loss == float("inf")
        assert trainer.train_losses == []
        assert trainer.valid_losses == []
        assert trainer.best_epochs == []

    def test_init_metric_histories_structure(self, trainer: Trainer) -> None:
        """Test metric histories has correct 6 metric keys with empty lists."""
        expected_keys = {"ssim", "ms_ssim", "mae", "gradient_mae", "psnr", "ncc"}
        assert set(trainer.metric_histories.keys()) == expected_keys

        for key in expected_keys:
            assert trainer.metric_histories[key] == []
            assert isinstance(trainer.metric_histories[key], list)

    def test_init_per_crop_histories_structure(self, trainer: Trainer) -> None:
        """Test per-crop metric histories has 9 crop indices."""
        assert len(trainer.per_crop_metric_histories) == NUM_CROPS

        for crop_idx in range(NUM_CROPS):
            assert crop_idx in trainer.per_crop_metric_histories
            crop_metrics = trainer.per_crop_metric_histories[crop_idx]

            expected_keys = {"ssim", "ms_ssim", "mae", "gradient_mae", "psnr", "ncc"}
            assert set(crop_metrics.keys()) == expected_keys

            for key in expected_keys:
                assert crop_metrics[key] == []

    def test_init_timing_histories_empty(self, trainer: Trainer) -> None:
        """Test that timing histories start empty."""
        assert trainer.learning_rates == []
        assert trainer.epoch_times == []
        assert trainer.train_times == []
        assert trainer.valid_times == []

    def test_init_directories_none(self, trainer: Trainer) -> None:
        """Test that directories start as None."""
        assert trainer.experiment_dir is None
        assert trainer.epochs_dir is None

    def test_init_internal_state(self, trainer: Trainer) -> None:
        """Test internal state attributes."""
        assert trainer._fault_file is None
        assert trainer._run_epoch_times == []

    def test_init_config_is_same_reference(self, minimal_config: dict) -> None:
        """Test that config is stored by reference, not copied."""
        trainer = Trainer(minimal_config)
        # Modifying the original config affects the trainer's config
        minimal_config["test_key"] = "test_value"
        assert trainer.config.get("test_key") == "test_value"
