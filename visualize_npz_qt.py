import os
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.train_network import network


class NetworkDialog(QtWidgets.QDialog):
    def __init__(self, parent, get_vector, get_net, set_net):
        super().__init__(parent)
        self.setWindowTitle("Network Viewer")
        self._get_vector = get_vector
        self._get_net = get_net
        self._set_net = set_net
        self._activations = []
        self._layer_index = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._step_animation)

        self.hidden_edit = QtWidgets.QLineEdit("64,32")
        self.output_spin = QtWidgets.QSpinBox()
        self.output_spin.setRange(1, 1024)
        self.output_spin.setValue(10)
        self.show_input = QtWidgets.QCheckBox("Show input layer")
        self.show_input.setChecked(False)
        self.max_neurons = QtWidgets.QSpinBox()
        self.max_neurons.setRange(8, 512)
        self.max_neurons.setValue(64)

        create_btn = QtWidgets.QPushButton("Create random network")
        create_btn.clicked.connect(self.create_network)
        save_btn = QtWidgets.QPushButton("Save JSON")
        save_btn.clicked.connect(self.save_network)
        load_btn = QtWidgets.QPushButton("Load JSON")
        load_btn.clicked.connect(self.load_network)

        top = QtWidgets.QGridLayout()
        top.addWidget(QtWidgets.QLabel("Hidden layers (comma):"), 0, 0)
        top.addWidget(self.hidden_edit, 0, 1, 1, 3)
        top.addWidget(QtWidgets.QLabel("Output size:"), 1, 0)
        top.addWidget(self.output_spin, 1, 1)
        top.addWidget(self.show_input, 1, 2)
        top.addWidget(QtWidgets.QLabel("Max neurons/layer:"), 1, 3)
        top.addWidget(self.max_neurons, 1, 4)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(create_btn)
        btns.addWidget(save_btn)
        btns.addWidget(load_btn)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing)

        self.delay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.delay_slider.setRange(50, 1000)
        self.delay_slider.setValue(300)
        self.delay_label = QtWidgets.QLabel("Delay: 300 ms")
        self.delay_slider.valueChanged.connect(
            lambda v: self.delay_label.setText(f"Delay: {v} ms")
        )

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        step_btn = QtWidgets.QPushButton("Step")
        step_btn.clicked.connect(self.step_once)
        simulate_btn = QtWidgets.QPushButton("Simulate feed-forward")
        simulate_btn.clicked.connect(self.start_simulation)

        anim = QtWidgets.QHBoxLayout()
        anim.addWidget(simulate_btn)
        anim.addWidget(self.play_btn)
        anim.addWidget(step_btn)
        anim.addWidget(self.delay_label)
        anim.addWidget(self.delay_slider, 1)

        self.stats_combo = QtWidgets.QComboBox()
        self.stats_text = QtWidgets.QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        stats_btn = QtWidgets.QPushButton("Show layer parameters")
        stats_btn.clicked.connect(self.show_layer_params)

        stats = QtWidgets.QHBoxLayout()
        stats.addWidget(QtWidgets.QLabel("Layer:"))
        stats.addWidget(self.stats_combo, 1)
        stats.addWidget(stats_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(btns)
        layout.addWidget(self.view, 1)
        layout.addLayout(anim)
        layout.addLayout(stats)
        layout.addWidget(self.stats_text)

        self.show_input.stateChanged.connect(self.redraw)
        self.max_neurons.valueChanged.connect(self.redraw)

        self.redraw()

    def _parse_hidden(self):
        text = self.hidden_edit.text().strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return [int(p) for p in parts]

    def create_network(self):
        vec = self._get_vector()
        if vec is None:
            QtWidgets.QMessageBox.warning(self, "Network", "No image vector available yet.")
            return
        input_size = int(vec.size)
        hidden = self._parse_hidden()
        output = int(self.output_spin.value())
        sizes = [input_size] + hidden + [output]
        self._set_net(network(sizes))
        self._refresh_layers()
        self.redraw()

    def save_network(self):
        net = self._get_net()
        if net is None:
            QtWidgets.QMessageBox.warning(self, "Network", "No network to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save network", "", "JSON (*.json)"
        )
        if not path:
            return
        net.to_json(path)

    def load_network(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load network", "", "JSON (*.json)"
        )
        if not path:
            return
        net = network.from_json(path)
        self._set_net(net)
        self._refresh_layers()
        self.redraw()

    def _refresh_layers(self):
        net = self._get_net()
        self.stats_combo.clear()
        if net is None:
            return
        for i in range(len(net.sizes) - 1):
            self.stats_combo.addItem(f"{i}: {net.sizes[i]} -> {net.sizes[i+1]}")

    def start_simulation(self):
        net = self._get_net()
        vec = self._get_vector()
        if net is None or vec is None:
            QtWidgets.QMessageBox.warning(self, "Simulate", "Need a network and an image.")
            return
        if int(vec.size) != int(net.sizes[0]):
            QtWidgets.QMessageBox.warning(
                self, "Simulate", "Input size doesn't match network input."
            )
            return
        self._activations = net.feed_forward_with_activations(vec)
        self._layer_index = 0
        self.redraw()

    def toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("Play")
            return
        self._timer.start(self.delay_slider.value())
        self.play_btn.setText("Pause")

    def step_once(self):
        self._step_animation()

    def _step_animation(self):
        if not self._activations:
            return
        self._layer_index += 1
        if self._layer_index >= len(self._activations):
            self._layer_index = len(self._activations) - 1
            self._timer.stop()
            self.play_btn.setText("Play")
        self.redraw()

    def show_layer_params(self):
        net = self._get_net()
        if net is None:
            return
        idx = self.stats_combo.currentIndex()
        if idx < 0:
            return
        w = net.weights[idx]
        b = net.biases[idx]
        text = []
        text.append("Weights stats:")
        text.append(f"shape: {w.shape}")
        text.append(f"min: {w.min():0.6f}  max: {w.max():0.6f}")
        text.append(f"mean: {w.mean():0.6f}  std: {w.std():0.6f}")
        text.append("")
        text.append("Biases stats:")
        text.append(f"shape: {b.shape}")
        text.append(f"min: {b.min():0.6f}  max: {b.max():0.6f}")
        text.append(f"mean: {b.mean():0.6f}  std: {b.std():0.6f}")
        text.append("\nFull weights (first 2000 chars):")
        text.append(np.array2string(w, max_line_width=120)[:2000])
        text.append("\nFull biases:")
        text.append(np.array2string(b, max_line_width=120))
        self.stats_text.setPlainText("\n".join(text))

    def redraw(self):
        self.scene.clear()
        net = self._get_net()
        if net is None:
            return

        sizes = list(net.sizes)
        show_input = self.show_input.isChecked()
        if not show_input:
            sizes = sizes[1:]

        max_neurons = int(self.max_neurons.value())
        layer_count = len(sizes)
        width = 800
        height = 400
        self.scene.setSceneRect(0, 0, width, height)
        x_gap = width / max(1, layer_count - 1)
        radius = 6

        acts = None
        if self._activations:
            acts = self._activations
            if not show_input:
                acts = acts[1:]

        for li, layer_size in enumerate(sizes):
            display_n = min(layer_size, max_neurons)
            y_gap = height / max(1, display_n)
            for ni in range(display_n):
                x = li * x_gap + 20
                y = ni * y_gap + 20
                color = QtGui.QColor(60, 60, 60)
                if acts is not None and li <= self._layer_index:
                    idx = int(ni * layer_size / display_n)
                    val = float(acts[li].reshape(-1)[idx])
                    r = int(val * 255)
                    b = int((1.0 - val) * 255)
                    color = QtGui.QColor(r, 0, b)
                self.scene.addEllipse(
                    x, y, radius * 2, radius * 2,
                    QtGui.QPen(QtCore.Qt.NoPen),
                    QtGui.QBrush(color),
                )


class NpzViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NPZ Image Viewer")
        self._data = {}
        self._labels = {}
        self._net = None
        self._net_input_size = None

        self.path_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse)
        load_btn = QtWidgets.QPushButton("Load")
        load_btn.clicked.connect(self.load)
        net_btn = QtWidgets.QPushButton("Network…")
        net_btn.clicked.connect(self.open_network_dialog)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("NPZ file:"))
        top.addWidget(self.path_edit, 1)
        top.addWidget(browse_btn)
        top.addWidget(load_btn)
        top.addWidget(net_btn)

        self.split_combo = QtWidgets.QComboBox()
        self.split_combo.currentTextChanged.connect(self.on_split_change)
        self.index_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.index_slider.setMinimum(0)
        self.index_slider.valueChanged.connect(self.on_index_change)
        self.index_label = QtWidgets.QLabel("Index: 0")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Split:"))
        controls.addWidget(self.split_combo)
        controls.addSpacing(12)
        controls.addWidget(self.index_label)
        controls.addWidget(self.index_slider, 1)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(280, 280)
        self.image_label.setStyleSheet("background: #111;")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        self.class_label = QtWidgets.QLabel("Label: -")

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(self.class_label)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_panel)

        self.vector_text = QtWidgets.QPlainTextEdit()
        self.vector_text.setReadOnly(True)
        self.vector_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        self.heatmap_label = QtWidgets.QLabel()
        self.heatmap_label.setFixedSize(280, 280)
        self.heatmap_label.setStyleSheet("background: #111;")
        self.heatmap_label.setAlignment(QtCore.Qt.AlignCenter)

        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["Vector", "Heatmap"])
        self.view_combo.currentTextChanged.connect(self.on_view_change)

        self.right_stack = QtWidgets.QStackedWidget()
        self.right_stack.addWidget(self.vector_text)
        self.right_stack.addWidget(self.heatmap_label)

        self.output_label = QtWidgets.QLabel("Feed-forward output: -")
        self._last_vector = None
        self._net_dialog = None

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(self.view_combo)
        right_panel.addWidget(self.right_stack)
        right_panel.addWidget(self.output_label)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_panel)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(left_widget)
        split.addWidget(right_widget)
        split.setSizes([300, 500])

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(controls)
        layout.addWidget(split)

    def browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select NPZ", "", "NPZ files (*.npz);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)

    def load(self):
        path = self.path_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "NPZ", "Please choose an .npz file.")
            return
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, "NPZ", f"File not found: {path}")
            return
        try:
            npz = np.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return

        self._data = {}
        self._labels = {}
        keys = list(npz.keys())

        # Common MNIST-style keys.
        for x_key, y_key in [("x_train", "y_train"), ("x_test", "y_test")]:
            if x_key in keys:
                self._data[x_key] = npz[x_key]
                if y_key in keys:
                    self._labels[x_key] = npz[y_key]

        # Fallback: use any array that looks like images.
        if not self._data:
            for k in keys:
                arr = npz[k]
                if arr.ndim in (3, 4):
                    self._data[k] = arr
        if not self._data:
            QtWidgets.QMessageBox.warning(self, "NPZ", "No image arrays found.")
            return

        self.split_combo.blockSignals(True)
        self.split_combo.clear()
        self.split_combo.addItems(list(self._data.keys()))
        self.split_combo.blockSignals(False)

        self.on_split_change(self.split_combo.currentText())

    def on_split_change(self, key):
        if not key:
            return
        data = self._data.get(key)
        if data is None:
            return
        self.index_slider.setMaximum(max(0, len(data) - 1))
        self.index_slider.setValue(0)
        self.update_view(key, 0)

    def on_index_change(self, idx):
        key = self.split_combo.currentText()
        if not key:
            return
        self.update_view(key, idx)

    def update_view(self, key, idx):
        data = self._data[key]
        img = data[idx]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.mean(axis=2)

        # Convert to grayscale uint8 for display and vectorization.
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        h, w = arr.shape
        qimg = QtGui.QImage(arr.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            280, 280, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)
        self.index_label.setText(f"Index: {idx}")

        label = "-"
        if key in self._labels and idx < len(self._labels[key]):
            label = str(int(self._labels[key][idx]))
        self.class_label.setText(f"Label: {label}")

        # Flatten to [0, 1] vector for feed-forward.
        vector = arr.astype(np.float32) / 255.0
        flat = vector.reshape(-1)
        grid_lines = [
            " ".join(f"{v:0.3f}" for v in row) for row in vector
        ]
        grid_text = "\n".join(grid_lines)
        self.vector_text.setPlainText(grid_text)

        self._run_feed_forward(flat)
        self._update_heatmap(vector)
        self._last_vector = flat

    def on_view_change(self, _text):
        idx = 0 if self.view_combo.currentText() == "Vector" else 1
        self.right_stack.setCurrentIndex(idx)

    def _update_heatmap(self, vector):
        v = np.clip(vector, 0.0, 1.0)
        h, w = v.shape
        r = (v * 255.0).astype(np.uint8)
        b = ((1.0 - v) * 255.0).astype(np.uint8)
        g = np.zeros_like(r, dtype=np.uint8)
        rgb = np.dstack([r, g, b])
        qimg = QtGui.QImage(rgb.tobytes(), w, h, w * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            280, 280, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation
        )
        self.heatmap_label.setPixmap(pix)

    def _run_feed_forward(self, flat):
        input_size = int(flat.size)
        if self._net is None or self._net_input_size != input_size:
            # Create a simple MLP matching the vector size.
            self._net = network([input_size, 64, 32, 10])
            self._net_input_size = input_size
        out = self._net.feed_forward(flat)
        out_flat = out.reshape(-1)
        pred = int(np.argmax(out_flat))
        self.output_label.setText(
            f"Feed-forward output: pred={pred} | shape={out_flat.shape}"
        )

    def _get_vector(self):
        return self._last_vector

    def _get_net(self):
        return self._net

    def _set_net(self, net):
        self._net = net
        self._net_input_size = net.sizes[0]
        self.output_label.setText("Feed-forward output: -")

    def open_network_dialog(self):
        if self._net_dialog is None:
            self._net_dialog = NetworkDialog(
                self, self._get_vector, self._get_net, self._set_net
            )
        self._net_dialog.show()
        self._net_dialog.raise_()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = NpzViewer()
    win.show()
    sys.exit(app.exec_())
