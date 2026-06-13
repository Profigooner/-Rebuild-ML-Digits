import torch

from src.model import DigitCNN, load_checkpoint, save_checkpoint


def test_model_output_shape():
    model = DigitCNN()
    output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(7)
    model = DigitCNN().eval()
    sample = torch.randn(2, 1, 28, 28)
    expected = model(sample)
    path = tmp_path / "model.pt"

    save_checkpoint(
        path,
        model,
        epoch=3,
        val_accuracy=0.992,
        test_accuracy=0.993,
    )
    restored, metadata = load_checkpoint(path)

    assert torch.allclose(restored(sample), expected)
    assert metadata["epoch"] == 3
    assert metadata["val_accuracy"] == 0.992
    assert metadata["test_accuracy"] == 0.993

