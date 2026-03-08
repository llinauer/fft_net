from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from .data import ImageFolderDataset
from .model import FFTCNN, FFTNet


def _infer_num_classes(ds: ImageFolderDataset) -> int:
    return int(ds.num_classes)


def _validate_gpu_index(gpu_index: int) -> None:
    if gpu_index < 0:
        raise ValueError(f"train.gpu_index must be >= 0, got {gpu_index}")


def _resolve_device(device_cfg: str, gpu_index: int) -> torch.device:
    device_cfg = str(device_cfg).lower()
    _validate_gpu_index(gpu_index)

    if device_cfg == "auto":
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if gpu_index >= n:
                raise ValueError(f"train.gpu_index={gpu_index} out of range for {n} CUDA device(s)")
            return torch.device(f"cuda:{gpu_index}")
        return torch.device("cpu")

    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("train.device='cuda' was requested but CUDA is not available")
        n = torch.cuda.device_count()
        if gpu_index >= n:
            raise ValueError(f"train.gpu_index={gpu_index} out of range for {n} CUDA device(s)")
        return torch.device(f"cuda:{gpu_index}")

    if device_cfg == "cpu":
        return torch.device("cpu")

    raise ValueError("train.device must be one of: auto, cpu, cuda")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
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
        y = y.to(device, non_blocking=True)

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
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
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


def build_model(cfg: DictConfig, img_size: tuple[int, int], num_classes: int) -> nn.Module:
    model_type = str(cfg.model.model_type).lower()

    if model_type == "mlp":
        return FFTNet(
            img_size=img_size,
            num_classes=num_classes,
            hidden_dims=list(cfg.model.hidden_dims),
            dropout=float(cfg.model.dropout),
        )

    if model_type == "cnn":
        return FFTCNN(
            num_classes=num_classes,
            conv_channels=tuple(cfg.model.conv_channels),
            dropout=float(cfg.model.dropout),
        )

    raise ValueError("model.model_type must be one of: mlp, cnn")


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(cfg: DictConfig) -> None:
    if not cfg.train.dataset_path:
        raise ValueError("Please provide train.dataset_path")

    dataset_path = Path(cfg.train.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    torch.manual_seed(int(cfg.train.seed))

    img_size = tuple(cfg.train.img_size)
    train_ds_full = ImageFolderDataset(path=str(dataset_path), img_size=img_size)

    if cfg.train.val_dataset_path:
        val_dataset_path = Path(cfg.train.val_dataset_path)
        if not val_dataset_path.exists():
            raise FileNotFoundError(f"Validation dataset path does not exist: {val_dataset_path}")
        val_ds = ImageFolderDataset(path=str(val_dataset_path), img_size=img_size)
        train_ds = train_ds_full
        num_classes = _infer_num_classes(train_ds_full)
        if val_ds.num_classes != num_classes:
            raise ValueError(
                f"Class-count mismatch: train={num_classes}, val={val_ds.num_classes}. "
                "Ensure both splits share identical class folders."
            )
    else:
        num_classes = _infer_num_classes(train_ds_full)
        val_size = int(len(train_ds_full) * float(cfg.train.val_split_fraction))
        train_size = len(train_ds_full) - val_size
        train_ds, val_ds = random_split(train_ds_full, [train_size, val_size])

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

    model = build_model(cfg=cfg, img_size=img_size, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    scheduler = None
    if bool(cfg.train.use_lr_scheduler):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.n_epochs),
            eta_min=float(cfg.train.min_learning_rate),
        )

    log_dir = Path(str(cfg.train.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard log_dir: {log_dir}")

    if scheduler is None:
        print("LR scheduler: disabled")
    else:
        print(
            "LR scheduler: cosine "
            f"(T_max={int(cfg.train.n_epochs)}, eta_min={float(cfg.train.min_learning_rate)})"
        )

    for epoch in range(int(cfg.train.n_epochs)):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            log_every_n_steps=int(cfg.train.log_every_n_steps),
            epoch=epoch,
            device=device,
        )
        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        logger.add_scalar("Loss/train", train_metrics["loss"], epoch)
        logger.add_scalar("Loss/val", val_metrics["loss"], epoch)
        logger.add_scalar("Accuracy/train", train_metrics["acc"], epoch)
        logger.add_scalar("Accuracy/val", val_metrics["acc"], epoch)
        logger.add_scalar("AccuracyPct/train", train_metrics["acc"] * 100.0, epoch)
        logger.add_scalar("AccuracyPct/val", val_metrics["acc"] * 100.0, epoch)
        logger.add_scalar("LearningRate", current_lr, epoch)

        print(
            f"epoch={epoch+1}/{cfg.train.n_epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"lr={current_lr:.8f}"
        )

    logger.close()

    model_path = log_dir / "fft_net_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
