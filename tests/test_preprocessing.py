import numpy as np
import pytest

from src.preprocessing import EmptyDrawingError, preprocess_digit


def test_empty_drawing_is_rejected():
    with pytest.raises(EmptyDrawingError):
        preprocess_digit(np.zeros((280, 280), dtype=np.uint8))


def test_preprocessing_shapes_range_and_centering():
    drawing = np.zeros((280, 280), dtype=np.uint8)
    drawing[25:210, 190:230] = 255

    result = preprocess_digit(drawing)

    assert result.image.shape == (28, 28)
    assert result.image.dtype == np.uint8
    assert 0 <= int(result.image.min()) <= int(result.image.max()) <= 255
    assert result.tensor.shape == (1, 28, 28)
    assert np.isfinite(result.tensor.numpy()).all()

    mass = result.image.astype(np.float64)
    y, x = np.indices(result.image.shape)
    center_y = float((y * mass).sum() / mass.sum())
    center_x = float((x * mass).sum() / mass.sum())
    assert abs(center_y - 13.5) <= 0.6
    assert abs(center_x - 13.5) <= 0.6

