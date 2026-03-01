from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data import BirdImgDataset
from .model import FFTNet


def _infer_class_stats(ds: BirdImgDataset) -> tuple[int, int, int]:
    labels = [ds._get_class(path) for path in ds.imgs]
    min_label = min(labels)
    max_label = max(labels)
    num_classes = max_label - min_label + 1
    return min_label, max_label, num_classes


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

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )

    model = FFTNet(
        img_size=img_size,
        num_classes=num_classes,
        hidden_dims=list(cfg.model.hidden_dims),
        dropout=float(cfg.model.dropout),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    for epoch in range(int(cfg.train.n_epochs)):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            y = y - min_label  # convert class ids to 0..(num_classes-1)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % int(cfg.train.log_every_n_steps) == 0:
                print(f"epoch={epoch+1} step={step} train_loss={running_loss/step:.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                y = y - min_label
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()

        avg_train = running_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total)
        print(
            f"epoch={epoch+1}/{cfg.train.n_epochs} "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} val_acc={val_acc:.3f} "
            f"labels=[{min_label}..{max_label}]"
        )

    torch.save(model.state_dict(), "fft_net_model.pth")
    print("Saved model to fft_net_model.pth")


if __name__ == "__main__":
    main()
