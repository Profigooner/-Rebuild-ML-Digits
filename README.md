# MNIST CNN 手写数字识别

本项目使用 PyTorch 在 MNIST 数据集上训练卷积神经网络，并提供一个
PyQt6 桌面应用。用户可以在画板上手写数字，应用会显示预测结果、
置信度、0–9 的分类概率以及模型实际接收到的 28×28 图像。

## 环境安装

推荐使用 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 训练模型

以下命令会自动下载 MNIST，创建固定的训练集和验证集划分，并保存验证
准确率最高的 checkpoint：

```bash
python -m src.train_cnn --epochs 12 --model-out models/mnist_cnn.pt
```

默认设备选择顺序为 CUDA、Apple MPS、CPU。也可以明确指定：

```bash
python -m src.train_cnn --device mps --epochs 12
```

常用选项：

```text
--batch-size 128
--lr 0.001
--seed 42
--data-dir data
--workers 0
--limit-train N   # 只用于快速检查训练流程
--limit-test N
```

训练完成后，checkpoint 会记录模型结构、权重、类别、MNIST 归一化参数、
最佳 epoch、验证准确率和测试准确率。

仓库内的 `models/mnist_cnn.pt` 使用 Apple MPS 完成训练。最佳 epoch 为
11，验证准确率为 99.38%，CPU 独立测试准确率为 99.61%。最终测试固定在
CPU 上执行，以规避部分 PyTorch/MPS 版本在长时间训练后出现的评估数值差异。

## 启动识别应用

```bash
python -m src.app --model models/mnist_cnn.pt
```

在黑色画板上用鼠标写一个白色数字，然后点击 **Recognize**。应用会裁剪
笔迹、等比例缩放到 20×20、放入 28×28 图像并按灰度重心居中，最后使用
与训练阶段相同的 MNIST 参数进行归一化。

若模型文件不存在或格式不兼容，应用仍会启动，并在状态栏和点击识别时
显示具体错误。

## 测试

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

测试覆盖模型输入输出、checkpoint 往返、空白画布、图像预处理与居中、
单批次训练，以及 PyQt6 的绘制、识别和清除流程。

## 项目结构

```text
src/model.py          CNN 与 checkpoint 接口
src/preprocessing.py  手写图像预处理
src/train_cnn.py      MNIST 训练 CLI
src/app.py            PyQt6 手写识别应用
tests/                自动化测试
```

`src/train_network.py` 和 `visualize_npz_qt.py` 是早期的 NumPy/PyQt5
实验工具，仅为兼容和参考而保留，不属于当前 CNN 主流程。
