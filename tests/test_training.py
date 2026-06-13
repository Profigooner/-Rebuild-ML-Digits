import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.model import DigitCNN
from src.train_cnn import evaluate, train_one_epoch


def test_training_smoke():
    torch.manual_seed(1)
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    train_loss, train_accuracy = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        torch.device("cpu"),
    )
    eval_loss, eval_accuracy = evaluate(
        model,
        loader,
        criterion,
        torch.device("cpu"),
    )

    assert train_loss > 0
    assert eval_loss > 0
    assert 0 <= train_accuracy <= 1
    assert 0 <= eval_accuracy <= 1

