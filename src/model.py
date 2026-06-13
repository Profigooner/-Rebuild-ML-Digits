"""Shared CNN architecture and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CHECKPOINT_VERSION = 1
DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "channels": [32, 64],
    "hidden_features": 128,
    "dropout": 0.35,
    "num_classes": 10,
}


class DigitCNN(nn.Module):
    """Compact CNN for 28x28 grayscale digit images."""

    def __init__(
        self,
        channels: list[int] | tuple[int, int] = (32, 64),
        hidden_features: int = 128,
        dropout: float = 0.35,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError("DigitCNN requires exactly two channel groups.")

        first, second = (int(value) for value in channels)
        self.config = {
            "channels": [first, second],
            "hidden_features": int(hidden_features),
            "dropout": float(dropout),
            "num_classes": int(num_classes),
        }

        self.features = nn.Sequential(
            nn.Conv2d(1, first, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True),
            nn.Conv2d(first, first, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(first, second, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),
            nn.Conv2d(second, second, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second * 7 * 7, int(hidden_features), bias=False),
            nn.BatchNorm1d(int(hidden_features)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_features), int(num_classes)),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def create_model(config: dict[str, Any] | None = None) -> DigitCNN:
    """Build a model from a serialized configuration."""

    merged = dict(DEFAULT_MODEL_CONFIG)
    if config:
        merged.update(config)
    return DigitCNN(**merged)


def save_checkpoint(
    path: str | Path,
    model: DigitCNN,
    *,
    epoch: int,
    val_accuracy: float,
    test_accuracy: float | None = None,
) -> None:
    """Save the portable inference checkpoint consumed by the GUI."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_name": "DigitCNN",
        "model_config": dict(model.config),
        "model_state_dict": model.state_dict(),
        "classes": list(range(model.config["num_classes"])),
        "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
        "epoch": int(epoch),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": (
            None if test_accuracy is None else float(test_accuracy)
        ),
    }
    torch.save(checkpoint, destination)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[DigitCNN, dict[str, Any]]:
    """Load and validate a DigitCNN checkpoint."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {source}")

    checkpoint = torch.load(
        source,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint: expected a dictionary.")

    required = {
        "checkpoint_version",
        "model_name",
        "model_config",
        "model_state_dict",
        "classes",
        "normalization",
        "epoch",
        "val_accuracy",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            f"Invalid checkpoint: missing fields {', '.join(missing)}."
        )
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported checkpoint version "
            f"{checkpoint['checkpoint_version']}."
        )
    if checkpoint["model_name"] != "DigitCNN":
        raise ValueError(
            f"Unsupported model type: {checkpoint['model_name']}."
        )
    if checkpoint["classes"] != list(range(10)):
        raise ValueError("Checkpoint classes must be the digits 0 through 9.")

    normalization = checkpoint["normalization"]
    if not isinstance(normalization, dict):
        raise ValueError("Invalid checkpoint normalization metadata.")
    if "mean" not in normalization or "std" not in normalization:
        raise ValueError("Checkpoint normalization requires mean and std.")
    if float(normalization["std"]) <= 0:
        raise ValueError("Checkpoint normalization std must be positive.")

    try:
        model = create_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Incompatible DigitCNN checkpoint: {exc}") from exc

    model.eval()
    return model, checkpoint

