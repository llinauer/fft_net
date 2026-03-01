from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from .data import BirdImgDataset
from .model import FFTNet


def _infer_class_stats(ds: BirdImgDataset) -> tuple[int, int, int]:
    labels = [ds._get_class(path) for path in ds.imgs]
    min_label = min(labels)
    max_label = max(labels)
    num_classes = max_label - min_label + 1
    return min_label, max_label, num_classes


def _resolve_device(device_cfg: str, gpu_index: int) -> torch.device:
    device_cfg = str(device_cfg).lower()
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_index}")
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("train.device='cuda' was requested but CUDA is not available")
        return torch.device(f"cuda:{gpu_index}")
    if device_cfg == "cpu":
        return torch.device("cpu")
    raise ValueError("train.device must be one of: auto, cpu, cuda")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    min_label: int,
    log_every_n_steps: int,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (x, y) in enumerate(train_loader, start=1):
        x = x.to(device, non_blocking=True)
        y = (y - min_label).to(device, non_blocking=True)  # convert class ids to 0..(num_classes-1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

        running_loss += loss.item()
        if step % log_every_n_steps == 0:
            print(f"epoch={epoch + 1} step={step} train_loss={running_loss / step:.4f}")

    return {
        "loss": running_loss / max(1, len(train_loader)),
        "acc": correct / max(1, total),
    }


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    min_label: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = (y - min_label).to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()

    return {
        "loss": running_loss / max(1, len(val_loader)),
        "acc": correct / max(1, total),
    }


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(cfg: DictConfig) -> None:
    if not cfg.train.dataset_path:
        raise ValueError("Please provide train.dataset_path")

    dataset_path = Path(cfg.train.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    torch.manual_seed(int(cfg.train.seed))

    img_size = tuple(cfg.train.img_size)
    ds = BirdImgDataset(path=str(dataset_path), img_size=img_size)
    min_label, max_label, num_classes = _infer_class_stats(ds)

    val_size = int(len(ds) * float(cfg.train.val_split_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    device = _resolve_device(cfg.train.device, int(cfg.train.gpu_index))
    print(f"Using device: {device}")

    use_pin_memory = bool(cfg.train.pin_memory) and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=use_pin_memory,
    )

    model = FFTNet(
        img_size=img_size,
        num_classes=num_classes,
        hidden_dims=list(cfg.model.hidden_dims),
        dropout=float(cfg.model.dropout),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    logger = SummaryWriter(log_dir="logs/fft_net")

    for epoch in range(int(cfg.train.n_epochs)):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            min_label=min_label,
            log_every_n_steps=int(cfg.train.log_every_n_steps),
            epoch=epoch,
            device=device,
        )
        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            min_label=min_label,
            device=device,
        )

        logger.add_scalar("Loss/train", train_metrics["loss"], epoch)
        logger.add_scalar("Loss/val", val_metrics["loss"], epoch)
        logger.add_scalar("Accuracy/train", train_metrics["acc"], epoch)
        logger.add_scalar("Accuracy/val", val_metrics["acc"], epoch)
        logger.add_scalar("AccuracyPct/train", train_metrics["acc"] * 100.0, epoch)
        logger.add_scalar("AccuracyPct/val", val_metrics["acc"] * 100.0, epoch)
        logger.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"epoch={epoch+1}/{cfg.train.n_epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"labels=[{min_label}..{max_label}]"
        )

    logger.close()

    torch.save(model.state_dict(), "fft_net_model.pth")
    print("Saved model to fft_net_model.pth")


if __name__ == "__main__":
    main()
