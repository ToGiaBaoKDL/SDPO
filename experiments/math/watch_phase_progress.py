#!/usr/bin/env python3
"""Watch one SDPO-Math experiment and print compact progress lines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


PROGRESS_KEYS = {
    "training/global_step": "step",
    "critic/score/mean": "score",
    "critic/rewards/mean": "reward",
    "val-core/math_dapo/acc/mean@1": "val_acc",
    "val-aux/math_dapo/incorrect_format/mean@1": "bad_fmt",
    "val-aux/math_dapo/truncated/mean@1": "trunc",
    "self_distillation/reprompt_sample_fraction": "reprompt",
    "self_distillation/feedback_used_fraction": "feedback",
    "self_distillation/reliability_weight_mean": "rel_w",
    "actor/pg_loss": "pg_loss",
    "actor/grad_norm": "grad",
    "perf/throughput": "tok_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--total-steps", required=True, type=int)
    parser.add_argument("--interval", default=10.0, type=float)
    parser.add_argument("--idle-interval", default=120.0, type=float)
    return parser.parse_args()


def metric_path(log_dir: Path, experiment_name: str) -> Path:
    return log_dir / "metrics" / "SDPO-Math" / f"{experiment_name}.jsonl"


def compact_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def progress_line(experiment_name: str, row: dict[str, Any], total_steps: int) -> str:
    data = row.get("data", {})
    step = data.get("training/global_step", row.get("step", "?"))
    if total_steps > 0:
        prefix = f"[progress] {experiment_name} step={step}/{total_steps}"
    else:
        prefix = f"[progress] {experiment_name} step={step}"

    parts = [prefix]
    for key, label in PROGRESS_KEYS.items():
        if key == "training/global_step":
            continue
        if key in data:
            parts.append(f"{label}={compact_number(data[key])}")
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    path = metric_path(args.log_dir, args.experiment_name)
    print(f"[progress] {args.experiment_name} waiting_for_metrics={path}", flush=True)

    offset = 0
    last_printed_step: Any = None
    last_activity = time.monotonic()

    while True:
        if not path.exists():
            now = time.monotonic()
            if now - last_activity >= args.idle_interval:
                print(f"[progress] {args.experiment_name} still_waiting_for_metrics", flush=True)
                last_activity = now
            time.sleep(args.interval)
            continue

        with path.open("rb") as f:
            f.seek(offset)
            lines = f.readlines()
            offset = f.tell()

        printed = False
        for raw in lines:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            data = row.get("data", {})
            step = data.get("training/global_step", row.get("step"))
            if step == last_printed_step and "training/global_step" in data:
                continue
            print(progress_line(args.experiment_name, row, args.total_steps), flush=True)
            last_printed_step = step
            last_activity = time.monotonic()
            printed = True

        if not printed:
            now = time.monotonic()
            if now - last_activity >= args.idle_interval:
                print(f"[progress] {args.experiment_name} no_new_metrics", flush=True)
                last_activity = now

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
