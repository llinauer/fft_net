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
- resilient retries are built in for search/create/ssh/rsync steps
- SSH failures during polling do **not** terminate the run
- if rsync keeps failing, the script prints manual fallback rsync commands
