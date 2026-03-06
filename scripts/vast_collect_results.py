#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


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
        "ConnectTimeout=15",
        "-p",
        str(port),
    ]
    if key_path:
        cmd += ["-i", key_path]
    cmd += [f"{user}@{host}", remote_cmd]
    return cmd


def rsync_cmd(src: str, dst: str, host: str, port: int, user: str, key_path: str | None) -> list[str]:
    ssh_part = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}"
    if key_path:
        ssh_part += f" -i {shlex.quote(key_path)}"
    return [
        "rsync",
        "-az",
        "-e",
        ssh_part,
        f"{user}@{host}:{src}",
        dst,
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Check remote FFT experiment state, fetch results if done, teardown instance")
    p.add_argument("--instance-id", type=int, required=True)
    p.add_argument("--ssh-key", default=None)
    p.add_argument("--remote-dir", default="/workspace/fft_net")
    p.add_argument("--local-results", default="experiments/results.csv")
    p.add_argument("--local-logs-dir", default="logs/experiments")
    p.add_argument("--teardown-on-finish", action="store_true", help="Destroy instance after successful fetch")
    p.add_argument("--teardown-on-fail", action="store_true", help="Destroy instance when remote run failed")
    args = p.parse_args()

    inst = vast_get_instance(args.instance_id)
    host, port, user = extract_ssh_target(inst)

    state_check = (
        f"cd {shlex.quote(args.remote_dir)} && "
        "if [ -f .remote_state/done.ok ]; then echo DONE; "
        "elif [ -f .remote_state/done.fail ]; then echo FAIL; "
        "elif [ -f .remote_state/run.log ]; then echo RUNNING; "
        "else echo NOT_STARTED; fi"
    )

    cp = run(ssh_cmd(host, port, user, args.ssh_key, f"bash -lc {shlex.quote(state_check)}"))
    state = (cp.stdout or "").strip()
    print(f"Remote state: {state}")

    if state in {"RUNNING", "NOT_STARTED"}:
        print("Experiment is not finished yet. Nothing fetched.")
        return 0

    local_results = Path(args.local_results)
    local_logs_dir = Path(args.local_logs_dir)
    local_results.parent.mkdir(parents=True, exist_ok=True)
    local_logs_dir.mkdir(parents=True, exist_ok=True)

    results_cmd = rsync_cmd(
        f"{args.remote_dir}/experiments/results.csv",
        str(local_results),
        host,
        port,
        user,
        args.ssh_key,
    )
    logs_cmd = rsync_cmd(
        f"{args.remote_dir}/logs/experiments/",
        str(local_logs_dir) + "/",
        host,
        port,
        user,
        args.ssh_key,
    )

    fetched_ok = True
    try:
        run(results_cmd)
        print(f"Fetched results to {local_results}")
    except Exception as e:  # noqa: BLE001
        fetched_ok = False
        print(f"WARNING: failed to fetch results.csv: {e}")
        print("Manual command:")
        print(" ".join(results_cmd))

    try:
        run(logs_cmd)
        print(f"Fetched logs to {local_logs_dir}")
    except Exception as e:  # noqa: BLE001
        fetched_ok = False
        print(f"WARNING: failed to fetch logs: {e}")
        print("Manual command:")
        print(" ".join(logs_cmd))

    should_destroy = False
    if state == "DONE" and args.teardown_on_finish and fetched_ok:
        should_destroy = True
    if state == "FAIL" and args.teardown_on_fail:
        should_destroy = True

    if should_destroy:
        try:
            run(["vastai", "destroy", "instance", str(args.instance_id)])
            print(f"Destroyed instance {args.instance_id}")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: failed to destroy instance {args.instance_id}: {e}")
    else:
        print("Instance not destroyed (per flags/state).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
