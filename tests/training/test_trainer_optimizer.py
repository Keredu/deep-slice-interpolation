"""Tests for Trainer._create_optimizer method."""

import pytest
from torch.optim import SGD, Adam, AdamW

from phd.training.trainer import Trainer

from .conftest import SimpleModel


class TestCreateOptimizer:
    """Tests for _create_optimizer method."""

    def test_creates_adamw_with_params(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test AdamW optimizer creation with parameters."""
        trainer_with_model.config["optimizer"] = {
            "name": "adamw",
            "params": {
                "lr": 0.001,
                "weight_decay": 0.01,
            },
        }

        optimizer = trainer_with_model._create_optimizer()

        assert isinstance(optimizer, AdamW)
        assert optimizer.defaults["lr"] == 0.001
        assert optimizer.defaults["weight_decay"] == 0.01

    def test_creates_adam_with_params(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test Adam optimizer creation with parameters."""
        trainer_with_model.config["optimizer"] = {
            "name": "adam",
            "params": {
                "lr": 0.0005,
                "weight_decay": 0.0,
            },
        }

        optimizer = trainer_with_model._create_optimizer()

        assert isinstance(optimizer, Adam)
        assert optimizer.defaults["lr"] == 0.0005
        assert optimizer.defaults["weight_decay"] == 0.0

    def test_creates_sgd_with_momentum(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test SGD optimizer creation with momentum."""
        trainer_with_model.config["optimizer"] = {
            "name": "sgd",
            "params": {
                "lr": 0.01,
                "weight_decay": 0.001,
                "momentum": 0.9,
                "nesterov": True,
            },
        }

        optimizer = trainer_with_model._create_optimizer()

        assert isinstance(optimizer, SGD)
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["nesterov"] is True
        assert optimizer.defaults["weight_decay"] == 0.001

    def test_unsupported_optimizer_raises(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test that unsupported optimizer raises ValueError."""
        trainer_with_model.config["optimizer"] = {
            "name": "rmsprop",  # Not supported
            "params": {"lr": 0.001},
        }

        with pytest.raises(ValueError, match="Unsupported optimizer: rmsprop"):
            trainer_with_model._create_optimizer()

    def test_adamw_betas_from_config(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test that custom betas are applied to AdamW."""
        custom_betas = (0.85, 0.95)
        trainer_with_model.config["optimizer"] = {
            "name": "adamw",
            "params": {
                "lr": 0.001,
                "betas": custom_betas,
            },
        }

        optimizer = trainer_with_model._create_optimizer()

        assert optimizer.defaults["betas"] == custom_betas

    def test_adam_eps_from_config(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test that custom eps is applied to Adam."""
        trainer_with_model.config["optimizer"] = {
            "name": "adam",
            "params": {
                "lr": 0.001,
                "eps": 1e-6,
            },
        }

        optimizer = trainer_with_model._create_optimizer()

        assert optimizer.defaults["eps"] == 1e-6

    def test_default_lr_when_not_specified(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test default learning rate when not specified in params."""
        trainer_with_model.config["optimizer"] = {
            "name": "adamw",
            "params": {},  # No lr specified
        }

        optimizer = trainer_with_model._create_optimizer()

        # Default lr is 3e-4
        assert optimizer.defaults["lr"] == 3e-4

    def test_default_weight_decay_when_not_specified(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test default weight decay when not specified in params."""
        trainer_with_model.config["optimizer"] = {
            "name": "adamw",
            "params": {"lr": 0.001},  # No weight_decay specified
        }

        optimizer = trainer_with_model._create_optimizer()

        # Default weight_decay is 0.0
        assert optimizer.defaults["weight_decay"] == 0.0

    def test_optimizer_name_case_insensitive(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test that optimizer name is case insensitive."""
        trainer_with_model.config["optimizer"] = {
            "name": "AdAmW",  # Mixed case
            "params": {"lr": 0.001},
        }

        optimizer = trainer_with_model._create_optimizer()

        assert isinstance(optimizer, AdamW)

    def test_sgd_default_momentum(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test SGD default momentum when not specified."""
        trainer_with_model.config["optimizer"] = {
            "name": "sgd",
            "params": {"lr": 0.01},  # No momentum specified
        }

        optimizer = trainer_with_model._create_optimizer()

        # Default momentum is 0.0
        assert optimizer.defaults["momentum"] == 0.0
        assert optimizer.defaults["nesterov"] is False

    def test_params_key_missing_uses_defaults(
        self, trainer_with_model: Trainer, simple_model: SimpleModel
    ) -> None:
        """Test that missing params key uses all defaults."""
        trainer_with_model.config["optimizer"] = {
            "name": "adamw",
            # No params key at all
        }

        optimizer = trainer_with_model._create_optimizer()

        assert optimizer.defaults["lr"] == 3e-4
        assert optimizer.defaults["weight_decay"] == 0.0
