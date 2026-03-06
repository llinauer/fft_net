# fft_net
Classifying images in frequency space.

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
