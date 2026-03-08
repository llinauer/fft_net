#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png"}


def main() -> int:
    p = argparse.ArgumentParser(description="Create deterministic train/val split for CUB class folders")
    p.add_argument("--input-dir", required=True, help="Path to CUB_200_2011/images")
    p.add_argument("--out-dir", required=True, help="Output root (creates train/ and val/)")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Copy files instead of hardlink")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output dir")
    args = p.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0,1)")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {in_dir}")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in in_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class directories found in: {in_dir}")

    rng = random.Random(args.seed)
    total_train = 0
    total_val = 0

    for cls in class_dirs:
        imgs = sorted([p for p in cls.iterdir() if p.is_file() and _is_image(p)])
        if len(imgs) < 2:
            continue

        rng.shuffle(imgs)
        n_val = max(1, int(round(len(imgs) * args.val_fraction)))
        n_val = min(n_val, len(imgs) - 1)

        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        cls_train = train_dir / cls.name
        cls_val = val_dir / cls.name
        cls_train.mkdir(parents=True, exist_ok=True)
        cls_val.mkdir(parents=True, exist_ok=True)

        for src in train_imgs:
            dst = cls_train / src.name
            if dst.exists():
                dst.unlink()
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.hardlink_to(src)
            total_train += 1

        for src in val_imgs:
            dst = cls_val / src.name
            if dst.exists():
                dst.unlink()
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.hardlink_to(src)
            total_val += 1

    print(f"Split complete: train={total_train} val={total_val}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")
    print("Train with:")
    print(
        "python -m fft_net.train "
        f"train.dataset_path={train_dir} train.val_dataset_path={val_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
