#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def retry(fn, retries: int, delay_s: float, label: str):
    last_err = None
    for i in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[{label}] attempt {i}/{retries} failed: {e}")
            if i < retries:
                time.sleep(delay_s)
    raise RuntimeError(f"[{label}] failed after {retries} attempts: {last_err}")


def parse_json_output(cp: subprocess.CompletedProcess) -> list[dict]:
    out = (cp.stdout or "").strip()
    if not out:
        return []
    data = json.loads(out)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def vast_search_offer(query: str) -> int:
    cp = run(["vastai", "search", "offers", query, "--raw"])
    offers = parse_json_output(cp)
    if not offers:
        raise RuntimeError("No offers returned by vastai search")
    offer = offers[0]
    offer_id = offer.get("id") or offer.get("offer_id")
    if offer_id is None:
        raise RuntimeError(f"Could not extract offer id from: {offer}")
    return int(offer_id)


def vast_create_instance(offer_id: int, image: str, disk_gb: int) -> int:
    cp = run(
        [
            "vastai",
            "create",
            "instance",
            str(offer_id),
            "--image",
            image,
            "--disk",
            str(disk_gb),
            "--ssh",
            "--raw",
        ]
    )
    rows = parse_json_output(cp)
    if not rows:
        # fallback: parse plain output
        text = (cp.stdout or "") + "\n" + (cp.stderr or "")
        for tok in text.split():
            if tok.isdigit():
                return int(tok)
        raise RuntimeError(f"Could not parse instance id from create output: {text}")
    row = rows[0]
    inst_id = row.get("new_contract") or row.get("id") or row.get("instance_id")
    if inst_id is None:
        raise RuntimeError(f"Could not extract instance id from: {row}")
    return int(inst_id)


def vast_get_instance(instance_id: int) -> dict:
    cp = run(["vastai", "show", "instances", "--raw"])
    rows = parse_json_output(cp)
    for r in rows:
        rid = r.get("id") or r.get("instance_id")
        if rid is not None and int(rid) == instance_id:
            return r
    raise RuntimeError(f"Instance {instance_id} not found")


def extract_ssh_target(instance: dict) -> tuple[str, int, str]:
    host = instance.get("ssh_host") or instance.get("public_ipaddr") or instance.get("host")
    port = instance.get("ssh_port") or instance.get("port_forwards", {}).get("22") or 22
    user = instance.get("ssh_user") or "root"
    if host is None:
        raise RuntimeError(f"Could not extract ssh host from instance: {instance}")
    return str(host), int(port), str(user)


def ssh_cmd(host: str, port: int, user: str, key_path: str | None, remote_cmd: str) -> list[str]:
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        "-o",
        "ConnectTimeout=15",
        "-p",
        str(port),
    ]
    if key_path:
        cmd += ["-i", key_path]
    cmd += [f"{user}@{host}", remote_cmd]
    return cmd


def rsync_cmd(src: str, dst: str, host: str, port: int, user: str, key_path: str | None, to_remote: bool) -> list[str]:
    ssh_part = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}"
    if key_path:
        ssh_part += f" -i {shlex.quote(key_path)}"

    if to_remote:
        return [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            ".venv",
            "--exclude",
            "__pycache__",
            "-e",
            ssh_part,
            src,
            f"{user}@{host}:{dst}",
        ]

    return [
        "rsync",
        "-az",
        "-e",
        ssh_part,
        f"{user}@{host}:{src}",
        dst,
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Run fft_net experiments on Vast.ai with retry-safe orchestration")
    p.add_argument("--instance-id", type=int, default=None, help="Existing Vast instance id")
    p.add_argument("--search-query", default="reliability > 0.98 num_gpus=1 gpu_ram>=24", help="vastai search offers query")
    p.add_argument("--image", default="nvidia/cuda:12.4.1-devel-ubuntu22.04")
    p.add_argument("--disk-gb", type=int, default=80)
    p.add_argument("--ssh-key", default=None, help="Path to SSH private key")
    p.add_argument("--remote-dir", default="/workspace/fft_net")
    p.add_argument("--dataset-path", required=True, help="Remote dataset path on instance")
    p.add_argument("--max-runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--poll-interval", type=int, default=60)
    p.add_argument("--startup-timeout-min", type=int, default=30)
    p.add_argument("--keep-instance", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print planned actions and exit")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    local_logs_dir = repo_root / "logs" / "experiments"
    local_results_csv = repo_root / "experiments" / "results.csv"
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    local_results_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("DRY RUN: no Vast/SSH/rsync commands will be executed.")
        if args.instance_id is None:
            print(f"- Would search offer with query: {args.search_query}")
            print(f"- Would create instance using image={args.image} disk_gb={args.disk_gb}")
        else:
            print(f"- Would use existing instance_id={args.instance_id}")
        print("- Would wait for SSH readiness")
        print(f"- Would rsync upload repo -> {args.remote_dir}")
        print("- Would bootstrap uv + sync dependencies on remote")
        print(
            "- Would run experiments: "
            f"uv run python scripts/run_experiments.py --dataset-path {args.dataset_path} "
            f"--max-runs {args.max_runs} --seed {args.seed}"
        )
        print(f"- Would poll every {args.poll_interval}s for completion markers")
        print(f"- Would sync back results to: {local_results_csv}")
        print(f"- Would sync back logs to: {local_logs_dir}/")
        if args.keep_instance:
            print("- Would keep instance alive after run")
        else:
            print("- Would destroy instance if created by this script")
        return 0

    created_here = False

    if args.instance_id is None:
        offer_id = retry(lambda: vast_search_offer(args.search_query), retries=5, delay_s=8, label="search-offer")
        print(f"Using offer id: {offer_id}")
        args.instance_id = retry(
            lambda: vast_create_instance(offer_id, args.image, args.disk_gb),
            retries=5,
            delay_s=8,
            label="create-instance",
        )
        created_here = True
        print(f"Created instance: {args.instance_id}")

    deadline = time.time() + args.startup_timeout_min * 60
    host = None
    port = None
    user = None
    while time.time() < deadline:
        try:
            inst = vast_get_instance(args.instance_id)
            host, port, user = extract_ssh_target(inst)
            retry(
                lambda: run(
                    ssh_cmd(host, port, user, args.ssh_key, "echo ready"),
                    check=True,
                    capture=True,
                ),
                retries=3,
                delay_s=5,
                label="ssh-ready",
            )
            break
        except Exception as e:  # noqa: BLE001
            print(f"Waiting for instance SSH... {e}")
            time.sleep(15)

    if not host:
        raise RuntimeError("Instance did not become SSH-ready in time")

    print(f"SSH target: {user}@{host}:{port}")

    retry(
        lambda: run(
            rsync_cmd(str(repo_root) + "/", args.remote_dir, host, port, user, args.ssh_key, to_remote=True),
            check=True,
            capture=True,
        ),
        retries=5,
        delay_s=8,
        label="rsync-upload",
    )

    remote_bootstrap = f"""
set -e
mkdir -p {shlex.quote(args.remote_dir)}
cd {shlex.quote(args.remote_dir)}
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=\"$HOME/.local/bin:$PATH\"
fi
$HOME/.local/bin/uv sync || uv sync
""".strip()

    retry(
        lambda: run(ssh_cmd(host, port, user, args.ssh_key, f"bash -lc {shlex.quote(remote_bootstrap)}"), check=True, capture=True),
        retries=5,
        delay_s=10,
        label="remote-bootstrap",
    )

    remote_run = f"""
set -e
cd {shlex.quote(args.remote_dir)}
mkdir -p .remote_state logs/experiments experiments
RUN_LOG=.remote_state/run.log
DONE=.remote_state/done.ok
FAIL=.remote_state/done.fail
rm -f "$DONE" "$FAIL"
nohup bash -lc '$HOME/.local/bin/uv run python scripts/run_experiments.py --dataset-path {shlex.quote(args.dataset_path)} --max-runs {args.max_runs} --seed {args.seed} > "$RUN_LOG" 2>&1 && touch "$DONE" || touch "$FAIL"' >/dev/null 2>&1 &
echo started
""".strip()

    retry(
        lambda: run(ssh_cmd(host, port, user, args.ssh_key, f"bash -lc {shlex.quote(remote_run)}"), check=True, capture=True),
        retries=5,
        delay_s=8,
        label="remote-start",
    )

    print("Remote experiment started. Polling completion markers...")
    while True:
        try:
            cp = run(
                ssh_cmd(
                    host,
                    port,
                    user,
                    args.ssh_key,
                    f"bash -lc {shlex.quote(f'cd {args.remote_dir} && if [ -f .remote_state/done.ok ]; then echo DONE; elif [ -f .remote_state/done.fail ]; then echo FAIL; else echo RUNNING; fi')}",
                ),
                check=True,
                capture=True,
            )
            state = (cp.stdout or "").strip()
            print(f"Remote state: {state}")
            if state in {"DONE", "FAIL"}:
                break
        except Exception as e:  # noqa: BLE001
            print(f"Poll warning (keeping alive): {e}")
        time.sleep(args.poll_interval)

    # Best-effort download with retries; never terminate hard on sync failure.
    download_failed = False
    try:
        retry(
            lambda: run(
                rsync_cmd(
                    f"{args.remote_dir}/experiments/results.csv",
                    str(local_results_csv),
                    host,
                    port,
                    user,
                    args.ssh_key,
                    to_remote=False,
                ),
                check=True,
                capture=True,
            ),
            retries=8,
            delay_s=10,
            label="rsync-results",
        )
    except Exception as e:  # noqa: BLE001
        download_failed = True
        print(f"WARNING: results.csv sync failed: {e}")

    try:
        retry(
            lambda: run(
                rsync_cmd(
                    f"{args.remote_dir}/logs/experiments/",
                    str(local_logs_dir) + "/",
                    host,
                    port,
                    user,
                    args.ssh_key,
                    to_remote=False,
                ),
                check=True,
                capture=True,
            ),
            retries=8,
            delay_s=10,
            label="rsync-logs",
        )
    except Exception as e:  # noqa: BLE001
        download_failed = True
        print(f"WARNING: log sync failed: {e}")

    if download_failed:
        print("\nManual sync fallback commands:")
        print(
            " ".join(
                rsync_cmd(
                    f"{args.remote_dir}/experiments/results.csv",
                    str(local_results_csv),
                    host,
                    port,
                    user,
                    args.ssh_key,
                    to_remote=False,
                )
            )
        )
        print(
            " ".join(
                rsync_cmd(
                    f"{args.remote_dir}/logs/experiments/",
                    str(local_logs_dir) + "/",
                    host,
                    port,
                    user,
                    args.ssh_key,
                    to_remote=False,
                )
            )
        )

    if created_here and not args.keep_instance:
        try:
            run(["vastai", "destroy", "instance", str(args.instance_id)], check=True, capture=True)
            print(f"Destroyed instance {args.instance_id}")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: failed to destroy instance {args.instance_id}: {e}")

    print("Completed orchestration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
