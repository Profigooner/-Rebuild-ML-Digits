"""Train DigitCNN on MNIST and save the best validation checkpoint."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .model import MNIST_MEAN, MNIST_STD, DigitCNN, load_checkpoint, save_checkpoint


def choose_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cuda/mps/cpu to an available torch device."""

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of: auto, cpu, cuda, mps.")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(
    data_dir: str | Path,
    *,
    seed: int,
    val_size: int = 5_000,
    download: bool = True,
    limit_train: int | None = None,
    limit_test: int | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Create deterministic train/validation/test MNIST datasets."""

    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.92, 1.08),
                shear=5,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    train_source = datasets.MNIST(
        root=str(data_dir),
        train=True,
        transform=train_transform,
        download=download,
    )
    validation_source = datasets.MNIST(
        root=str(data_dir),
        train=True,
        transform=evaluation_transform,
        download=download,
    )
    test_source = datasets.MNIST(
        root=str(data_dir),
        train=False,
        transform=evaluation_transform,
        download=download,
    )

    if not 0 < val_size < len(train_source):
        raise ValueError("val_size must be between 1 and the training size.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    validation_indices = indices[:val_size]
    training_indices = indices[val_size:]

    if limit_train is not None:
        training_indices = training_indices[: max(1, limit_train)]
    test_indices = list(range(len(test_source)))
    if limit_test is not None:
        test_indices = test_indices[: max(1, limit_test)]

    return (
        Subset(train_source, training_indices),
        Subset(validation_source, validation_indices),
        Subset(test_source, test_indices),
    )


def create_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, validation_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.detach().cpu().item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_examples += batch_size

    return (
        total_loss / max(1, total_examples),
        total_correct / max(1, total_examples),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.detach().cpu().item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_examples += batch_size

    return (
        total_loss / max(1, total_examples),
        total_correct / max(1, total_examples),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/mnist_cnn.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Use only N training examples for a smoke test.",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Use only N test examples for a smoke test.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1.")
    if args.lr <= 0:
        raise ValueError("lr must be positive.")

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    datasets_tuple = build_datasets(
        args.data_dir,
        seed=args.seed,
        val_size=args.val_size,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    train_loader, validation_loader, _ = create_loaders(
        *datasets_tuple,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    print(
        "Dataset sizes: "
        f"train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)}, "
        f"test={len(datasets_tuple[2])}"
    )

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss, val_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        scheduler.step()
        elapsed = time.perf_counter() - started
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.2%} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.2%} "
            f"time={elapsed:.1f}s"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            save_checkpoint(
                args.model_out,
                model,
                epoch=epoch,
                val_accuracy=val_accuracy,
            )
            print(f"Saved new best checkpoint: {args.model_out}")

    # CPU is used for the final portable benchmark. Some PyTorch/MPS builds
    # can produce backend-dependent evaluation values after long runs.
    test_device = torch.device("cpu")
    best_model, checkpoint = load_checkpoint(args.model_out)
    _, _, cpu_test_loader = create_loaders(
        datasets_tuple[2],
        datasets_tuple[2],
        datasets_tuple[2],
        batch_size=args.batch_size,
        workers=args.workers,
        device=test_device,
    )
    print("Running final test benchmark on CPU.")
    test_loss, test_accuracy = evaluate(
        best_model,
        cpu_test_loader,
        criterion,
        test_device,
    )
    save_checkpoint(
        args.model_out,
        best_model,
        epoch=int(checkpoint["epoch"]),
        val_accuracy=float(checkpoint["val_accuracy"]),
        test_accuracy=test_accuracy,
    )
    print(
        f"Best epoch: {best_epoch} | "
        f"validation accuracy: {best_accuracy:.2%} | "
        f"test loss: {test_loss:.4f} | "
        f"test accuracy: {test_accuracy:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
