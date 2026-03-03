from __future__ import annotations

import argparse
import csv
import itertools
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


VAL_ACC_RE = re.compile(r"val_acc=([0-9]*\.?[0-9]+)")


@dataclass
class RunResult:
    phase: int
    run_name: str
    model_type: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    dropout: float
    capacity: str
    val_acc: float
    return_code: int


def _slug(parts: list[str]) -> str:
    return "__".join(p.replace("/", "-") for p in parts)


def _run_train(repo_root: Path, dataset_path: str, run_name: str, overrides: list[str]) -> tuple[int, float]:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "fft_net.train",
        f"train.dataset_path={dataset_path}",
        f"train.log_dir=logs/experiments/{run_name}",
        *overrides,
    ]

    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    matches = VAL_ACC_RE.findall(output)
    val_acc = float(matches[-1]) if matches else -1.0

    print(f"[{run_name}] rc={proc.returncode} val_acc={val_acc:.4f}")
    return proc.returncode, val_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase hyperparameter experiments for fft_net")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rng = random.Random(args.seed)

    results: list[RunResult] = []

    # Phase 1 (12 runs): model architecture + optimizer sweep
    phase1 = list(
        itertools.product(
            ["mlp", "cnn"],
            [3e-4, 1e-3, 3e-3],
            [0.0, 1e-4],
        )
    )

    for model_type, lr, wd in phase1:
        if len(results) >= args.max_runs:
            break

        parts = ["p1", f"model={model_type}", f"lr={lr:g}", f"wd={wd:g}"]
        run_name = _slug(parts)

        overrides = [
            f"model.model_type={model_type}",
            f"train.learning_rate={lr}",
            f"train.weight_decay={wd}",
            "train.batch_size=32",
            "model.dropout=0.2",
            "train.img_size=[128,128]",
        ]

        if args.dry_run:
            print(run_name, overrides)
            rc, val_acc = 0, -1.0
        else:
            rc, val_acc = _run_train(repo_root, args.dataset_path, run_name, overrides)

        results.append(
            RunResult(1, run_name, model_type, lr, wd, 32, 0.2, "base", val_acc, rc)
        )

    successful_p1 = [r for r in results if r.phase == 1 and r.return_code == 0 and r.val_acc >= 0.0]
    if not successful_p1:
        if args.dry_run:
            successful_p1 = [r for r in results if r.phase == 1]
        else:
            raise RuntimeError("No successful Phase 1 runs to select best config from")

    best_p1 = max(successful_p1, key=lambda r: r.val_acc)
    best_model = best_p1.model_type
    best_lr = best_p1.learning_rate
    best_wd = best_p1.weight_decay

    print(
        f"Best Phase-1 config: model={best_model} lr={best_lr:g} wd={best_wd:g} "
        f"val_acc={best_p1.val_acc:.4f}"
    )

    # Phase 2 (up to remaining runs, target up to total 30)
    batch_sizes = [16, 32, 64]
    dropouts = [0.0, 0.2, 0.4]

    if best_model == "mlp":
        capacities = ["[256,64]", "[1024,256]"]
    else:
        capacities = ["[16,32,64]", "[32,64,128]"]

    phase2_candidates = list(itertools.product(batch_sizes, dropouts, capacities))
    rng.shuffle(phase2_candidates)

    for bs, drop, cap in phase2_candidates:
        if len(results) >= args.max_runs:
            break

        parts = [
            "p2",
            f"model={best_model}",
            f"lr={best_lr:g}",
            f"wd={best_wd:g}",
            f"bs={bs}",
            f"drop={drop:g}",
            f"cap={cap.replace(',', '-').replace('[', '').replace(']', '')}",
        ]
        run_name = _slug(parts)

        overrides = [
            f"model.model_type={best_model}",
            f"train.learning_rate={best_lr}",
            f"train.weight_decay={best_wd}",
            f"train.batch_size={bs}",
            f"model.dropout={drop}",
            "train.img_size=[128,128]",
        ]
        if best_model == "mlp":
            overrides.append(f"model.hidden_dims={cap}")
            capacity_label = f"hidden_dims={cap}"
        else:
            overrides.append(f"model.conv_channels={cap}")
            capacity_label = f"conv_channels={cap}"

        if args.dry_run:
            print(run_name, overrides)
            rc, val_acc = 0, -1.0
        else:
            rc, val_acc = _run_train(repo_root, args.dataset_path, run_name, overrides)

        results.append(
            RunResult(2, run_name, best_model, best_lr, best_wd, bs, drop, capacity_label, val_acc, rc)
        )

    out_dir = repo_root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "phase",
                "run_name",
                "model_type",
                "learning_rate",
                "weight_decay",
                "batch_size",
                "dropout",
                "capacity",
                "val_acc",
                "return_code",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.phase,
                    r.run_name,
                    r.model_type,
                    r.learning_rate,
                    r.weight_decay,
                    r.batch_size,
                    r.dropout,
                    r.capacity,
                    r.val_acc,
                    r.return_code,
                ]
            )

    ok = [r for r in results if r.return_code == 0 and r.val_acc >= 0.0]
    if ok:
        best = max(ok, key=lambda r: r.val_acc)
        print(f"Best overall: {best.run_name} val_acc={best.val_acc:.4f}")
    print(f"Saved experiment summary to {out_csv}")


if __name__ == "__main__":
    main()
