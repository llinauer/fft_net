# fft_net
Classifying images in frequency space.

## Vast.ai experiment runner

Use `scripts/vast_run_experiment.py` to spin up a Vast.ai instance, run experiments remotely, and pull back results.

Example:

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
