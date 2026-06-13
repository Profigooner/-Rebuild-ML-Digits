"""PyQt6 freehand digit recognition application."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PyQt6 import QtCore, QtGui, QtWidgets

from .model import MNIST_MEAN, MNIST_STD, DigitCNN, load_checkpoint
from .preprocessing import EmptyDrawingError, PreprocessedDigit, preprocess_digit
from .train_cnn import choose_device


@dataclass(frozen=True)
class Prediction:
    digit: int
    confidence: float
    probabilities: np.ndarray
    preprocessed: PreprocessedDigit


class DigitPredictor:
    """Load model metadata and perform normalized digit inference."""

    def __init__(
        self,
        model: DigitCNN,
        *,
        device: torch.device,
        mean: float = MNIST_MEAN,
        std: float = MNIST_STD,
        metadata: dict | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.mean = float(mean)
        self.std = float(std)
        self.metadata = metadata or {}

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device,
    ) -> "DigitPredictor":
        model, checkpoint = load_checkpoint(path, map_location=device)
        normalization = checkpoint["normalization"]
        return cls(
            model,
            device=device,
            mean=float(normalization["mean"]),
            std=float(normalization["std"]),
            metadata=checkpoint,
        )

    @torch.inference_mode()
    def predict(self, drawing: np.ndarray) -> Prediction:
        processed = preprocess_digit(
            drawing,
            mean=self.mean,
            std=self.std,
        )
        batch = processed.tensor.unsqueeze(0).to(self.device)
        logits = self.model(batch)
        if logits.shape != (1, 10):
            raise RuntimeError(
                f"Model returned {tuple(logits.shape)}; expected (1, 10)."
            )
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        digit = int(np.argmax(probabilities))
        return Prediction(
            digit=digit,
            confidence=float(probabilities[digit]),
            probabilities=probabilities,
            preprocessed=processed,
        )


class DrawingCanvas(QtWidgets.QWidget):
    """Fixed-size white-on-black freehand drawing canvas."""

    drawing_changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(280, 280)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StaticContents)
        self._image = QtGui.QImage(
            self.size(),
            QtGui.QImage.Format.Format_Grayscale8,
        )
        self._image.fill(0)
        self._drawing = False
        self._last_point = QtCore.QPoint()
        self.pen_width = 22

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.drawImage(event.rect(), self._image, event.rect())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drawing = True
            self._last_point = event.position().toPoint()
            self._draw_segment(self._last_point, self._last_point)
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._drawing
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            current = event.position().toPoint()
            self._draw_segment(self._last_point, current)
            self._last_point = current
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._drawing
        ):
            self._draw_segment(
                self._last_point,
                event.position().toPoint(),
            )
            self._drawing = False
            event.accept()

    def _draw_segment(
        self,
        start: QtCore.QPoint,
        end: QtCore.QPoint,
    ) -> None:
        painter = QtGui.QPainter(self._image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(
            QtGui.QColor(255, 255, 255),
            self.pen_width,
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)
        painter.drawLine(start, end)
        painter.end()
        self.update()
        self.drawing_changed.emit()

    def draw_stroke(
        self,
        points: list[tuple[int, int]],
    ) -> None:
        """Draw a stroke programmatically, also useful for UI tests."""

        if not points:
            return
        qpoints = [QtCore.QPoint(x, y) for x, y in points]
        if len(qpoints) == 1:
            self._draw_segment(qpoints[0], qpoints[0])
            return
        for start, end in zip(qpoints, qpoints[1:]):
            self._draw_segment(start, end)

    def clear(self) -> None:
        self._image.fill(0)
        self.update()
        self.drawing_changed.emit()

    def to_numpy(self) -> np.ndarray:
        image = self._image.convertToFormat(
            QtGui.QImage.Format.Format_Grayscale8
        )
        pointer = image.bits()
        pointer.setsize(image.sizeInBytes())
        rows = np.frombuffer(pointer, dtype=np.uint8).reshape(
            image.height(),
            image.bytesPerLine(),
        )
        return rows[:, : image.width()].copy()


class DigitRecognizerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        predictor: DigitPredictor | None = None,
        *,
        model_error: str | None = None,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.model_error = model_error
        self.setWindowTitle("MNIST CNN Digit Recognizer")
        self.setMinimumSize(780, 480)

        self.canvas = DrawingCanvas()
        self.canvas.drawing_changed.connect(self._drawing_changed)
        self.prediction_label = QtWidgets.QLabel("-")
        self.prediction_label.setObjectName("predictionLabel")
        self.prediction_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.confidence_label = QtWidgets.QLabel("Confidence: -")
        self.confidence_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.preview_label = QtWidgets.QLabel("28 × 28")
        self.preview_label.setFixedSize(168, 168)
        self.preview_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.preview_label.setStyleSheet(
            "background: #050505; border: 1px solid #3d4652;"
        )

        recognize_button = QtWidgets.QPushButton("Recognize")
        recognize_button.setObjectName("recognizeButton")
        recognize_button.clicked.connect(self.recognize)
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.setObjectName("clearButton")
        clear_button.clicked.connect(self.clear)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(recognize_button)
        controls.addWidget(clear_button)

        canvas_panel = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Draw one digit")
        title.setObjectName("sectionTitle")
        canvas_panel.addWidget(title)
        canvas_panel.addWidget(self.canvas)
        canvas_panel.addLayout(controls)
        canvas_panel.addStretch()

        result_panel = QtWidgets.QVBoxLayout()
        result_title = QtWidgets.QLabel("Prediction")
        result_title.setObjectName("sectionTitle")
        result_panel.addWidget(result_title)
        result_panel.addWidget(self.prediction_label)
        result_panel.addWidget(self.confidence_label)

        probability_group = QtWidgets.QGroupBox("Class probabilities")
        probability_layout = QtWidgets.QGridLayout(probability_group)
        self.probability_bars: list[QtWidgets.QProgressBar] = []
        for digit in range(10):
            label = QtWidgets.QLabel(str(digit))
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFormat("0.0%")
            probability_layout.addWidget(label, digit, 0)
            probability_layout.addWidget(bar, digit, 1)
            self.probability_bars.append(bar)
        result_panel.addWidget(probability_group, 1)

        preview_panel = QtWidgets.QVBoxLayout()
        preview_title = QtWidgets.QLabel("Model input")
        preview_title.setObjectName("sectionTitle")
        preview_panel.addWidget(preview_title)
        preview_panel.addWidget(self.preview_label)
        preview_panel.addStretch()

        content = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(28)
        layout.addLayout(canvas_panel)
        layout.addLayout(result_panel, 1)
        layout.addLayout(preview_panel)
        self.setCentralWidget(content)

        self.status_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status_label)
        self._set_model_status()
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #151a21;
                color: #edf2f7;
                font-size: 14px;
            }
            QLabel#sectionTitle {
                font-size: 18px;
                font-weight: 600;
            }
            QLabel#predictionLabel {
                color: #65d1ff;
                font-size: 72px;
                font-weight: 700;
                min-height: 90px;
            }
            QPushButton {
                background: #2673dd;
                border: 0;
                border-radius: 6px;
                padding: 10px 18px;
                font-weight: 600;
            }
            QPushButton:hover { background: #3484ed; }
            QPushButton#clearButton { background: #3b4655; }
            QGroupBox {
                border: 1px solid #3d4652;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QProgressBar {
                background: #252c36;
                border: 0;
                border-radius: 4px;
                min-height: 16px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #2673dd;
                border-radius: 4px;
            }
            QStatusBar { color: #aeb8c4; }
            """
        )

    def _set_model_status(self) -> None:
        if self.predictor is None:
            self.status_label.setText(
                f"Model unavailable: {self.model_error or 'not loaded'}"
            )
            return
        metadata = self.predictor.metadata
        accuracy = metadata.get("test_accuracy")
        accuracy_text = (
            "not recorded"
            if accuracy is None
            else f"{float(accuracy):.2%}"
        )
        self.status_label.setText(
            f"Device: {self.predictor.device} | "
            f"Test accuracy: {accuracy_text}"
        )

    def _drawing_changed(self) -> None:
        self.prediction_label.setText("-")
        self.confidence_label.setText("Confidence: -")

    def recognize(self) -> None:
        if self.predictor is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Model unavailable",
                self.model_error
                or "Train or select a compatible DigitCNN checkpoint.",
            )
            return
        try:
            prediction = self.predictor.predict(self.canvas.to_numpy())
        except EmptyDrawingError as exc:
            QtWidgets.QMessageBox.warning(self, "Empty canvas", str(exc))
            return
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Recognition failed",
                str(exc),
            )
            return

        self.prediction_label.setText(str(prediction.digit))
        self.confidence_label.setText(
            f"Confidence: {prediction.confidence:.1%}"
        )
        for probability, bar in zip(
            prediction.probabilities,
            self.probability_bars,
        ):
            bar.setValue(int(round(float(probability) * 1000)))
            bar.setFormat(f"{float(probability):.1%}")
        self._show_preview(prediction.preprocessed.image)

    def _show_preview(self, image: np.ndarray) -> None:
        height, width = image.shape
        qimage = QtGui.QImage(
            image.tobytes(),
            width,
            height,
            width,
            QtGui.QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
            self.preview_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.FastTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def clear(self) -> None:
        self.canvas.clear()
        self.prediction_label.setText("-")
        self.confidence_label.setText("Confidence: -")
        self.preview_label.clear()
        self.preview_label.setText("28 × 28")
        for bar in self.probability_bars:
            bar.setValue(0)
            bar.setFormat("0.0%")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/mnist_cnn.pt"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    qt_app = QtWidgets.QApplication(sys.argv[:1])
    try:
        device = choose_device(args.device)
        predictor = DigitPredictor.from_checkpoint(
            args.model,
            device=device,
        )
        model_error = None
    except Exception as exc:
        predictor = None
        model_error = str(exc)

    window = DigitRecognizerWindow(
        predictor,
        model_error=model_error,
    )
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

