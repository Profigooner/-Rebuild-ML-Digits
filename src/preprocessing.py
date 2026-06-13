"""Convert freehand grayscale drawings into MNIST-compatible tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .model import MNIST_MEAN, MNIST_STD


class EmptyDrawingError(ValueError):
    """Raised when a canvas does not contain a visible stroke."""


@dataclass(frozen=True)
class PreprocessedDigit:
    """The display-ready 28x28 image and normalized model input."""

    image: np.ndarray
    tensor: torch.Tensor


def _shift_without_wrap(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros_like(image)
    height, width = image.shape

    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)

    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = image[
            src_y0:src_y1, src_x0:src_x1
        ]
    return shifted


def preprocess_digit(
    image: np.ndarray,
    *,
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
    threshold: int = 8,
) -> PreprocessedDigit:
    """Crop, scale, center and normalize a white-on-black drawing."""

    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float32).mean(axis=2)
    if array.ndim != 2:
        raise ValueError("Drawing must be a 2D grayscale or RGB image.")

    grayscale = np.clip(array, 0, 255).astype(np.uint8)
    foreground = grayscale > int(threshold)
    if not np.any(foreground):
        raise EmptyDrawingError("The drawing canvas is empty.")

    rows, columns = np.nonzero(foreground)
    cropped = grayscale[
        rows.min() : rows.max() + 1,
        columns.min() : columns.max() + 1,
    ]

    height, width = cropped.shape
    scale = 20.0 / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(cropped, mode="L").resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )

    centered = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - resized_height) // 2
    left = (28 - resized_width) // 2
    centered[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized

    mass = centered.astype(np.float64)
    total = float(mass.sum())
    if total <= 0:
        raise EmptyDrawingError("The drawing has no usable foreground pixels.")
    y_coordinates, x_coordinates = np.indices(centered.shape)
    center_y = float((y_coordinates * mass).sum() / total)
    center_x = float((x_coordinates * mass).sum() / total)
    dy = int(round(13.5 - center_y))
    dx = int(round(13.5 - center_x))
    centered = _shift_without_wrap(centered, dy, dx)

    normalized = centered.astype(np.float32) / 255.0
    normalized = (normalized - float(mean)) / float(std)
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    return PreprocessedDigit(image=centered, tensor=tensor)

