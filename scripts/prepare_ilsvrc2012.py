#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_VAL_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/imagenet/val.txt"
)


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, out_path.open("wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f)


def _safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir)


def _extract_train_split(train_tar: Path, train_out: Path) -> None:
    tmp_train = train_out.parent / "_train_raw"
    if not tmp_train.exists():
        print(f"Extracting train archive: {train_tar}")
        _safe_extract_tar(train_tar, tmp_train)

    class_tars = sorted(tmp_train.glob("*.tar"))
    if not class_tars:
        raise RuntimeError(f"No class tar files found in {tmp_train}")

    for i, class_tar in enumerate(class_tars, start=1):
        class_name = class_tar.stem
        class_dir = train_out / class_name
        marker = class_dir / ".done"
        if marker.exists():
            continue
        class_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(class_tar) as tar:
            tar.extractall(class_dir)
        marker.touch()
        if i % 50 == 0:
            print(f"  extracted {i}/{len(class_tars)} train classes")

    print(f"Train split ready at: {train_out}")


def _extract_val_images(val_tar: Path, val_images_out: Path) -> None:
    if not val_images_out.exists() or not any(val_images_out.iterdir()):
        print(f"Extracting val archive: {val_tar}")
        _safe_extract_tar(val_tar, val_images_out)


def _read_val_wnids(path: Path) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # allow formats: "ILSVRC2012_val_00000001.JPEG n01440764" or just "n01440764"
            parts = line.split()
            wnid = parts[-1]
            labels.append(wnid)
    return labels


def _arrange_val_split(val_images_out: Path, val_out: Path, val_wnids_file: Path) -> None:
    images = sorted([p for p in val_images_out.glob("*") if p.is_file()])
    wnids = _read_val_wnids(val_wnids_file)

    if len(images) != len(wnids):
        raise RuntimeError(
            f"Validation image count ({len(images)}) != label count ({len(wnids)}). "
            f"Check labels file: {val_wnids_file}"
        )

    done_marker = val_out / ".done"
    if done_marker.exists():
        print(f"Val split already arranged at: {val_out}")
        return

    for i, (img, wnid) in enumerate(zip(images, wnids), start=1):
        target_dir = val_out / wnid
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / img.name
        if not target.exists():
            shutil.move(str(img), str(target))
        if i % 5000 == 0:
            print(f"  arranged {i}/{len(images)} val images")

    done_marker.touch()
    print(f"Val split ready at: {val_out}")


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare ILSVRC2012 train/val folder structure for training")
    p.add_argument("--train-tar", required=True, help="Path to ILSVRC2012_img_train.tar")
    p.add_argument("--val-tar", required=True, help="Path to ILSVRC2012_img_val.tar")
    p.add_argument(
        "--val-wnids-file",
        default=None,
        help="Text file with one val class wnid per line in val image order",
    )
    p.add_argument("--download-val-wnids-url", default=DEFAULT_VAL_LABELS_URL)
    p.add_argument("--out-dir", default="data/ilsvrc2012")
    p.add_argument("--keep-raw", action="store_true", help="Keep temporary raw extraction dirs")
    args = p.parse_args()

    train_tar = Path(args.train_tar)
    val_tar = Path(args.val_tar)
    out_dir = Path(args.out_dir)

    if not train_tar.exists() or not val_tar.exists():
        raise FileNotFoundError("Both --train-tar and --val-tar must exist")

    train_out = out_dir / "train"
    val_out = out_dir / "val"
    val_images_out = out_dir / "_val_images_raw"

    val_wnids_file = Path(args.val_wnids_file) if args.val_wnids_file else out_dir / "val_wnids.txt"
    if not val_wnids_file.exists():
        print(f"Downloading val wnids file from {args.download_val_wnids_url}")
        _download(args.download_val_wnids_url, val_wnids_file)

    _extract_train_split(train_tar, train_out)
    _extract_val_images(val_tar, val_images_out)
    _arrange_val_split(val_images_out, val_out, val_wnids_file)

    if not args.keep_raw:
        tmp_train = out_dir / "_train_raw"
        if tmp_train.exists():
            shutil.rmtree(tmp_train)
        if val_images_out.exists():
            shutil.rmtree(val_images_out)

    print("\nDone.")
    print(f"Train dir: {train_out}")
    print(f"Val dir:   {val_out}")
    print("You can now train with:")
    print(
        "python -m fft_net.train "
        f"train.dataset_path={train_out} train.val_dataset_path={val_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
