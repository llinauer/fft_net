# fft_net
Classifying images in frequency space.

## ILSVRC2012 (ImageNet) data prep

You must provide the official archives yourself (license/terms required):
- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_val.tar`

Then prepare train/val class-folder layout:

```bash
python scripts/prepare_ilsvrc2012.py \
  --train-tar /path/to/ILSVRC2012_img_train.tar \
  --val-tar /path/to/ILSVRC2012_img_val.tar \
  --out-dir data/ilsvrc2012
```

Train with explicit train/val splits:

```bash
python -m fft_net.train \
  train.dataset_path=data/ilsvrc2012/train \
  train.val_dataset_path=data/ilsvrc2012/val
```

## Vast.ai experiment runner

### 1) Dispatch run

Use `scripts/vast_run_experiment.py` to spin up/use a Vast.ai instance, upload code, and start experiments in background.

```bash
python scripts/vast_run_experiment.py \
  --dataset-path /workspace/CUB_200_2011/images \
  --max-runs 30 \
  --seed 42
```

Notes:
- resilient retries are built in for search/create/ssh/bootstrap/dispatch steps
- script runs in **dispatch-only** mode: it starts remote experiments and exits
- script prints rsync commands for pulling `experiments/results.csv` and `logs/experiments/`
- optional: `--sync-instructions-file <path>` writes those commands to a file
- optional: `--dry-run` prints planned actions without executing anything

### 2) Check + fetch + optional teardown

Use `scripts/vast_collect_results.py` to check remote status.
- If run is not finished: prints status and exits.
- If finished: fetches results/logs.
- Optionally tears down instance.

```bash
python scripts/vast_collect_results.py \
  --instance-id <ID> \
  --teardown-on-finish
```

Optional teardown on failed run too:

```bash
python scripts/vast_collect_results.py \
  --instance-id <ID> \
  --teardown-on-finish \
  --teardown-on-fail
```
