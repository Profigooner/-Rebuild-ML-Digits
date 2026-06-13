import numpy as np
import torch
from PyQt6 import QtWidgets

from src.app import DigitPredictor, DigitRecognizerWindow
from src.model import DigitCNN


def get_application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_draw_recognize_and_clear_flow():
    app = get_application()
    torch.manual_seed(3)
    predictor = DigitPredictor(
        DigitCNN(),
        device=torch.device("cpu"),
    )
    window = DigitRecognizerWindow(predictor)

    window.canvas.draw_stroke([(80, 40), (140, 220), (205, 70)])
    assert np.max(window.canvas.to_numpy()) == 255

    window.recognize()
    assert window.prediction_label.text() in set("0123456789")
    assert window.confidence_label.text().startswith("Confidence:")

    window.clear()
    assert np.max(window.canvas.to_numpy()) == 0
    assert window.prediction_label.text() == "-"
    window.close()
    app.processEvents()

