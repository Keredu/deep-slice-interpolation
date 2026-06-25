"""Tests for training configuration utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from phd.training.config import (
    SUPPORTED_LOSSES,
    SUPPORTED_MODELS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SCHEDULERS,
    create_config,
    get_dataset_dir,
    validate_config,
)

# Minimal valid config for tests (all experiment params required)
MINIMAL_VALID_CONFIG = {
    "model": {"type": "unet", "encoder_name": "tu-tf_efficientnetv2_s", "pretrained": True},
    "loss": {"name": "ssim"},
    "scheduler": {"name": "cosine", "params": {"eta_min": 1e-6}},
    "optimizer": {"name": "adamw", "params": {"lr": 0.001}},
    "batch_size": 64,
    "num_epochs": 100,
}


class TestGetDatasetDir:
    """Tests for get_dataset_dir function."""

    def test_missing_env_var_raises(self) -> None:
        """Test that missing DATASETS_DIR raises RuntimeError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="DATASETS_DIR environment variable is not set"):
                get_dataset_dir()

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        """Test that non-existent directory raises RuntimeError."""
        fake_dir = str(tmp_path / "nonexistent")
        with patch.dict("os.environ", {"DATASETS_DIR": fake_dir}):
            with pytest.raises(RuntimeError, match="does not exist"):
                get_dataset_dir()

    def test_valid_directory_returns_path(self, tmp_path: Path) -> None:
        """Test that valid directory returns correct path."""
        # Create the expected directory structure
        expected_dir = tmp_path / "pre/rsna-intracranial-hemorrhage-detection/1x512x512_-20_107"
        expected_dir.mkdir(parents=True)

        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            result = get_dataset_dir()
            assert result == expected_dir


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config_passes(self) -> None:
        """Test that valid config doesn't raise."""
        validate_config(MINIMAL_VALID_CONFIG)

    def test_invalid_batch_size_raises(self) -> None:
        """Test that non-positive batch size raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "batch_size": 0}
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            validate_config(config)

    def test_invalid_num_epochs_raises(self) -> None:
        """Test that non-positive num_epochs raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "num_epochs": 0}
        with pytest.raises(ValueError, match="num_epochs must be > 0"):
            validate_config(config)

    def test_invalid_valid_batch_size_raises(self) -> None:
        """Test that non-positive validation batch size raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "valid_batch_size": 0}
        with pytest.raises(ValueError, match="valid_batch_size must be > 0"):
            validate_config(config)

    def test_negative_valid_num_workers_raises(self) -> None:
        """Test that negative validation workers raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "valid_num_workers": -1}
        with pytest.raises(ValueError, match="valid_num_workers must be >= 0"):
            validate_config(config)

    def test_invalid_valid_prefetch_factor_raises(self) -> None:
        """Test that non-positive validation prefetch factor raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "valid_prefetch_factor": 0}
        with pytest.raises(ValueError, match="valid_prefetch_factor must be > 0"):
            validate_config(config)

    def test_init_from_experiment_must_be_non_empty_string(self) -> None:
        """Test init_from_experiment validation."""
        config = {**MINIMAL_VALID_CONFIG, "init_from_experiment": ""}
        with pytest.raises(ValueError, match="init_from_experiment must be a non-empty string"):
            validate_config(config)

    def test_init_from_checkpoint_requires_source_experiment(self) -> None:
        """Test init_from_checkpoint requires init_from_experiment."""
        config = {**MINIMAL_VALID_CONFIG, "init_from_checkpoint": "latest_epoch.pth"}
        with pytest.raises(ValueError, match="init_from_checkpoint requires init_from_experiment"):
            validate_config(config)

    def test_missing_loss_raises(self) -> None:
        """Test that missing loss config raises ValueError."""
        config = {
            "batch_size": 64,
            "num_epochs": 100,
            "scheduler": {"name": "cosine", "params": {"eta_min": 1e-6}},
            "optimizer": {"name": "adamw", "params": {"lr": 0.001}},
        }
        with pytest.raises(ValueError, match="'loss' configuration is required"):
            validate_config(config)

    def test_missing_loss_name_raises(self) -> None:
        """Test that missing loss.name raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "loss": {"params": {}}}
        with pytest.raises(ValueError, match=r"'loss\.name' is required"):
            validate_config(config)

    def test_unsupported_loss_raises(self) -> None:
        """Test that unsupported loss name raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "loss": {"name": "unsupported_loss"}}
        with pytest.raises(ValueError, match="Unsupported loss name"):
            validate_config(config)

    @pytest.mark.parametrize("loss_name", list(SUPPORTED_LOSSES))
    def test_all_supported_losses_pass(self, loss_name: str) -> None:
        """Test that all supported loss names pass validation."""
        # Combined losses require weight params that sum to 1.0
        combined_loss_params = {
            "ssim+l1": {"ssim_weight": 0.8, "l1_weight": 0.2},
            "msssim+l1": {"msssim_weight": 0.8, "l1_weight": 0.2},
        }
        params = combined_loss_params.get(loss_name, {})
        config = {
            **MINIMAL_VALID_CONFIG,
            "loss": {"name": loss_name, "params": params},
        }
        validate_config(config)

    def test_combined_loss_missing_weights_raises(self) -> None:
        """Test that combined losses without weights raise ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "loss": {"name": "ssim+l1", "params": {}}}
        with pytest.raises(ValueError, match="requires weights"):
            validate_config(config)

    def test_combined_loss_weights_not_sum_to_one_raises(self) -> None:
        """Test that combined loss weights not summing to 1.0 raise ValueError."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "loss": {"name": "ssim+l1", "params": {"ssim_weight": 0.5, "l1_weight": 0.3}},
        }
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            validate_config(config)

    def test_missing_scheduler_raises(self) -> None:
        """Test that missing scheduler config raises ValueError."""
        config = {
            "batch_size": 64,
            "num_epochs": 100,
            "loss": {"name": "ssim"},
            "optimizer": {"name": "adamw", "params": {"lr": 0.001}},
        }
        with pytest.raises(ValueError, match="'scheduler' configuration is required"):
            validate_config(config)

    def test_missing_scheduler_name_raises(self) -> None:
        """Test that missing scheduler.name raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "scheduler": {"params": {}}}
        with pytest.raises(ValueError, match=r"'scheduler\.name' is required"):
            validate_config(config)

    def test_unsupported_scheduler_raises(self) -> None:
        """Test that unsupported scheduler name raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "scheduler": {"name": "unsupported_scheduler"}}
        with pytest.raises(ValueError, match="Unsupported scheduler name"):
            validate_config(config)

    @pytest.mark.parametrize("scheduler_name", list(SUPPORTED_SCHEDULERS))
    def test_all_supported_schedulers_pass(self, scheduler_name: str) -> None:
        """Test that all supported scheduler names pass validation."""
        # All schedulers require explicit params (no defaults)
        scheduler_params = {
            "cosine": {"eta_min": 1e-6},
            "cosine_restarts": {"T_0": 20, "T_mult": 2, "eta_min": 1e-6},
            "cosine_warmup": {"warmup_epochs": 5, "eta_min": 1e-6},
            "onecycle": {"max_lr": 0.01, "pct_start": 0.3, "anneal_strategy": "cos"},
            "plateau": {"mode": "min", "factor": 0.5, "patience": 10, "min_lr": 1e-7},
            "step": {"step_size": 10, "gamma": 0.1},
            "none": {},
        }
        params = scheduler_params.get(scheduler_name, {})
        config = {
            **MINIMAL_VALID_CONFIG,
            "scheduler": {"name": scheduler_name, "params": params},
        }
        validate_config(config)

    def test_cosine_warmup_missing_warmup_epochs_raises(self) -> None:
        """Test that cosine_warmup without warmup_epochs raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "scheduler": {"name": "cosine_warmup", "params": {"eta_min": 1e-6}}}
        with pytest.raises(ValueError, match=r"requires parameters.*warmup_epochs"):
            validate_config(config)

    def test_negative_warmup_epochs_raises(self) -> None:
        """Test that negative warmup_epochs raises ValueError."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "scheduler": {"name": "cosine_warmup", "params": {"warmup_epochs": -1, "eta_min": 1e-6}},
        }
        with pytest.raises(ValueError, match=r"warmup_epochs must be >= 0"):
            validate_config(config)

    def test_onecycle_missing_max_lr_raises(self) -> None:
        """Test that onecycle without max_lr raises ValueError."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "scheduler": {"name": "onecycle", "params": {"pct_start": 0.3, "anneal_strategy": "cos"}},
        }
        with pytest.raises(ValueError, match=r"requires parameters.*max_lr"):
            validate_config(config)

    def test_step_missing_params_raises(self) -> None:
        """Test that step scheduler without required params raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "scheduler": {"name": "step", "params": {}}}
        with pytest.raises(ValueError, match="requires parameters"):
            validate_config(config)

    def test_cosine_restarts_missing_params_raises(self) -> None:
        """Test that cosine_restarts without required params raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "scheduler": {"name": "cosine_restarts", "params": {}}}
        with pytest.raises(ValueError, match="requires parameters"):
            validate_config(config)

    def test_cosine_restarts_invalid_t0_raises(self) -> None:
        """Test that T_0 <= 0 raises ValueError."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "scheduler": {"name": "cosine_restarts", "params": {"T_0": 0, "T_mult": 2, "eta_min": 1e-6}},
        }
        with pytest.raises(ValueError, match=r"T_0 must be > 0"):
            validate_config(config)

    def test_cosine_restarts_invalid_t_mult_raises(self) -> None:
        """Test that T_mult < 1 raises ValueError."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "scheduler": {"name": "cosine_restarts", "params": {"T_0": 20, "T_mult": 0, "eta_min": 1e-6}},
        }
        with pytest.raises(ValueError, match=r"T_mult must be >= 1"):
            validate_config(config)


class TestCreateConfig:
    """Tests for create_config function."""

    @pytest.fixture
    def setup_dataset_dir(self, tmp_path: Path) -> Path:
        """Create and return the expected dataset directory."""
        expected_dir = tmp_path / "pre/rsna-intracranial-hemorrhage-detection/1x512x512_-20_107"
        expected_dir.mkdir(parents=True)
        return expected_dir

    def test_creates_complete_config(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that create_config returns complete config dict."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            config = create_config(
                exp_name="test_exp",
                model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                loss={"name": "ssim"},
                scheduler={"name": "cosine", "params": {"eta_min": 1e-6}},
                optimizer={"name": "adamw", "params": {"lr": 0.001}},
                batch_size=64,
                num_epochs=100,
            )

        assert isinstance(config, dict)
        assert config["exp_name"] == "test_exp"
        assert config["batch_size"] == 64
        assert config["num_epochs"] == 100
        assert config["data_path"] == setup_dataset_dir

    def test_sets_infrastructure_defaults(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that infrastructure defaults are set."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            config = create_config(
                exp_name="test_exp",
                model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                loss={"name": "ssim"},
                scheduler={"name": "cosine", "params": {"eta_min": 1e-6}},
                optimizer={"name": "adamw", "params": {"lr": 0.001}},
                batch_size=64,
                num_epochs=100,
            )

        # Infrastructure defaults
        assert config["crop_size"] == 256
        assert config["flip_prob"] == 0.5
        assert config["num_workers"] == 2
        assert config["valid_batch_size"] is None
        assert config["valid_num_workers"] == 0
        assert config["valid_pin_memory"] is False
        assert config["valid_prefetch_factor"] == 1
        assert config["early_stopping_patience"] == 7
        assert config["init_from_experiment"] is None
        assert config["init_from_checkpoint"] is None

    def test_loss_config_set(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that loss config is set correctly."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            config = create_config(
                exp_name="test_exp",
                model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                loss={"name": "ssim", "params": {"data_range": 1.0}},
                scheduler={"name": "cosine", "params": {"eta_min": 1e-6}},
                optimizer={"name": "adamw", "params": {"lr": 0.001}},
                batch_size=64,
                num_epochs=100,
            )

        assert config["loss"]["name"] == "ssim"
        assert config["loss"]["params"]["data_range"] == 1.0

    def test_scheduler_config_set(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that scheduler config is set correctly."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            config = create_config(
                exp_name="test_exp",
                model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                loss={"name": "ssim"},
                scheduler={"name": "cosine_warmup", "params": {"warmup_epochs": 10, "eta_min": 1e-6}},
                optimizer={"name": "adamw", "params": {"lr": 0.001}},
                batch_size=64,
                num_epochs=100,
            )

        assert config["scheduler"]["name"] == "cosine_warmup"
        assert config["scheduler"]["params"]["warmup_epochs"] == 10

    def test_optimizer_config_set(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that optimizer config is set correctly."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            config = create_config(
                exp_name="test_exp",
                model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                loss={"name": "ssim"},
                scheduler={"name": "cosine", "params": {"eta_min": 1e-6}},
                optimizer={"name": "sgd", "params": {"lr": 0.01, "momentum": 0.9}},
                batch_size=64,
                num_epochs=100,
            )

        assert config["optimizer"]["name"] == "sgd"
        assert config["optimizer"]["params"]["lr"] == 0.01
        assert config["optimizer"]["params"]["momentum"] == 0.9

    def test_invalid_batch_size_raises(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that invalid batch_size raises ValueError."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            with pytest.raises(ValueError, match="batch_size must be > 0"):
                create_config(
                    exp_name="test_exp",
                    model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                    loss={"name": "ssim"},
                    scheduler={"name": "cosine", "params": {"eta_min": 1e-6}},
                    optimizer={"name": "adamw", "params": {"lr": 0.001}},
                    batch_size=0,
                    num_epochs=100,
                )

    def test_train_size_and_valid_size_passed_through(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that train_size and valid_size are included in config."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            config = create_config(
                exp_name="test_exp",
                model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                loss={"name": "ssim"},
                scheduler={"name": "cosine", "params": {"eta_min": 1e-6}},
                optimizer={"name": "adamw", "params": {"lr": 0.001}},
                batch_size=64,
                num_epochs=100,
                train_size=100,
                valid_size=50,
            )

        assert config["train_size"] == 100
        assert config["valid_size"] == 50

    def test_train_size_and_valid_size_default_to_none(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that train_size and valid_size default to None when not specified."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            config = create_config(
                exp_name="test_exp",
                model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                loss={"name": "ssim"},
                scheduler={"name": "cosine", "params": {"eta_min": 1e-6}},
                optimizer={"name": "adamw", "params": {"lr": 0.001}},
                batch_size=64,
                num_epochs=100,
            )

        assert config["train_size"] is None
        assert config["valid_size"] is None

    def test_init_from_experiment_fields_passed_through(self, tmp_path: Path, setup_dataset_dir: Path) -> None:
        """Test that init_from_* fields are included in config."""
        with patch.dict("os.environ", {"DATASETS_DIR": str(tmp_path)}):
            config = create_config(
                exp_name="test_exp",
                model={"type": "unet", "encoder_name": "resnet18", "pretrained": True},
                loss={"name": "ssim"},
                scheduler={"name": "cosine", "params": {"eta_min": 1e-6}},
                optimizer={"name": "adamw", "params": {"lr": 0.001}},
                batch_size=64,
                num_epochs=100,
                init_from_experiment="source_exp",
                init_from_checkpoint="epochs/12/weights.pth",
            )

        assert config["init_from_experiment"] == "source_exp"
        assert config["init_from_checkpoint"] == "epochs/12/weights.pth"


class TestSupportedLosses:
    """Tests for SUPPORTED_LOSSES constant."""

    def test_contains_basic_losses(self) -> None:
        """Test that basic losses are supported."""
        basic_losses = ["mse", "l1", "ssim"]
        for loss in basic_losses:
            assert loss in SUPPORTED_LOSSES

    def test_contains_combined_losses(self) -> None:
        """Test that combined losses are supported."""
        combined_losses = ["ssim+l1", "msssim+l1"]
        for loss in combined_losses:
            assert loss in SUPPORTED_LOSSES


class TestSupportedSchedulers:
    """Tests for SUPPORTED_SCHEDULERS constant."""

    def test_contains_basic_schedulers(self) -> None:
        """Test that basic schedulers are supported."""
        basic_schedulers = ["cosine", "cosine_restarts", "step", "plateau", "none"]
        for scheduler in basic_schedulers:
            assert scheduler in SUPPORTED_SCHEDULERS

    def test_contains_warmup_schedulers(self) -> None:
        """Test that warmup schedulers are supported."""
        warmup_schedulers = ["cosine_warmup", "onecycle"]
        for scheduler in warmup_schedulers:
            assert scheduler in SUPPORTED_SCHEDULERS


class TestSupportedModels:
    """Tests for SUPPORTED_MODELS constant."""

    def test_contains_unet(self) -> None:
        """Test that unet model is supported."""
        assert "unet" in SUPPORTED_MODELS


class TestModelValidation:
    """Tests for model configuration validation."""

    def test_valid_model_config_passes(self) -> None:
        """Test that valid model config doesn't raise."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "model": {"type": "unet", "encoder_name": "tu-tf_efficientnetv2_s"},
        }
        validate_config(config)

    def test_missing_model_type_raises(self) -> None:
        """Test that missing model.type raises ValueError."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "model": {"encoder_name": "tu-tf_efficientnetv2_s"},
        }
        with pytest.raises(ValueError, match=r"'model\.type' is required"):
            validate_config(config)

    def test_unsupported_model_type_raises(self) -> None:
        """Test that unsupported model type raises ValueError."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "model": {"type": "unsupported_model"},
        }
        with pytest.raises(ValueError, match="Unsupported model type"):
            validate_config(config)

    def test_unet_missing_encoder_raises(self) -> None:
        """Test that unet without encoder_name raises ValueError."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "model": {"type": "unet"},
        }
        with pytest.raises(ValueError, match=r"'model\.encoder_name' is required for unet"):
            validate_config(config)

    @pytest.mark.parametrize("model_type", list(SUPPORTED_MODELS))
    def test_all_supported_models_pass(self, model_type: str) -> None:
        """Test that all supported model types pass validation."""
        # UNet requires encoder_name
        model_params = {
            "unet": {"encoder_name": "tu-tf_efficientnetv2_s"},
        }
        params = model_params.get(model_type, {})
        config = {
            **MINIMAL_VALID_CONFIG,
            "model": {"type": model_type, **params},
        }
        validate_config(config)


class TestSupportedOptimizers:
    """Tests for SUPPORTED_OPTIMIZERS constant."""

    def test_contains_basic_optimizers(self) -> None:
        """Test that basic optimizers are supported."""
        basic_optimizers = ["adamw", "adam", "sgd"]
        for optimizer in basic_optimizers:
            assert optimizer in SUPPORTED_OPTIMIZERS


class TestOptimizerValidation:
    """Tests for optimizer configuration validation."""

    def test_valid_optimizer_config_passes(self) -> None:
        """Test that valid optimizer config doesn't raise."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "optimizer": {"name": "adamw", "params": {"lr": 0.001, "weight_decay": 0.01}},
        }
        validate_config(config)

    def test_missing_optimizer_raises(self) -> None:
        """Test that missing optimizer config raises ValueError."""
        config = {
            "batch_size": 64,
            "num_epochs": 100,
            "loss": {"name": "ssim"},
            "scheduler": {"name": "cosine", "params": {"eta_min": 1e-6}},
        }
        with pytest.raises(ValueError, match="'optimizer' configuration is required"):
            validate_config(config)

    def test_missing_optimizer_name_raises(self) -> None:
        """Test that missing optimizer.name raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "optimizer": {"params": {"lr": 0.001}}}
        with pytest.raises(ValueError, match=r"'optimizer\.name' is required"):
            validate_config(config)

    def test_unsupported_optimizer_raises(self) -> None:
        """Test that unsupported optimizer name raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "optimizer": {"name": "unsupported_optimizer", "params": {"lr": 0.001}}}
        with pytest.raises(ValueError, match="Unsupported optimizer name"):
            validate_config(config)

    def test_missing_lr_raises(self) -> None:
        """Test that missing optimizer.params.lr raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "optimizer": {"name": "adamw", "params": {}}}
        with pytest.raises(ValueError, match=r"'optimizer\.params\.lr' is required"):
            validate_config(config)

    def test_invalid_lr_raises(self) -> None:
        """Test that non-positive lr raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "optimizer": {"name": "adamw", "params": {"lr": 0}}}
        with pytest.raises(ValueError, match=r"optimizer\.params\.lr must be > 0"):
            validate_config(config)

    def test_negative_lr_raises(self) -> None:
        """Test that negative lr raises ValueError."""
        config = {**MINIMAL_VALID_CONFIG, "optimizer": {"name": "adamw", "params": {"lr": -0.001}}}
        with pytest.raises(ValueError, match=r"optimizer\.params\.lr must be > 0"):
            validate_config(config)

    @pytest.mark.parametrize("optimizer_name", list(SUPPORTED_OPTIMIZERS))
    def test_all_supported_optimizers_pass(self, optimizer_name: str) -> None:
        """Test that all supported optimizer names pass validation."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "optimizer": {"name": optimizer_name, "params": {"lr": 0.001}},
        }
        validate_config(config)

    def test_adamw_with_weight_decay_passes(self) -> None:
        """Test that adamw with weight_decay passes validation."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "optimizer": {"name": "adamw", "params": {"lr": 0.001, "weight_decay": 0.01}},
        }
        validate_config(config)

    def test_sgd_with_momentum_passes(self) -> None:
        """Test that sgd with momentum passes validation."""
        config = {
            **MINIMAL_VALID_CONFIG,
            "optimizer": {"name": "sgd", "params": {"lr": 0.01, "momentum": 0.9}},
        }
        validate_config(config)
