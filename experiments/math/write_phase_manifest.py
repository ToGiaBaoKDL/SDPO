#!/usr/bin/env python3
"""Write a structured manifest for an SDPO-Math phase run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--train-steps", required=True, type=int)
    parser.add_argument("--train-max-samples", required=True)
    parser.add_argument("--val-max-samples", required=True)
    parser.add_argument("--eval-freq", required=True)
    parser.add_argument("--save-freq", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exp-suffix", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    return parser.parse_args()


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    train_bs = os.environ.get("TRAIN_BS")
    rollout_n = os.environ.get("ROLLOUT_N")
    effective_rollouts = None
    if train_bs is not None and rollout_n is not None:
        try:
            effective_rollouts = int(train_bs) * int(rollout_n)
        except ValueError:
            effective_rollouts = None

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_status_short": git_value(["status", "--short"]),
        "project_root": os.environ.get("PROJECT_ROOT"),
        "phase": args.phase,
        "profile": args.profile,
        "effective_rollouts_per_step": effective_rollouts,
        "profile_settings": {
            key.lower(): os.environ.get(key)
            for key in [
                "TRAIN_BS",
                "ROLLOUT_N",
                "AGENT_WORKERS",
                "RESPONSE_LEN",
                "MODEL_LEN",
                "ACTOR_LEN",
                "REPROMPT_LEN",
                "BATCHED_TOKENS",
                "GPU_UTIL",
            ]
        },
        "config_name": args.config_name,
        "model": args.model,
        "variants": args.variants.split(),
        "train_steps": args.train_steps,
        "train_max_samples": args.train_max_samples,
        "val_max_samples": args.val_max_samples,
        "eval_freq": args.eval_freq,
        "save_freq": args.save_freq,
        "seed": args.seed,
        "exp_suffix": args.exp_suffix,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "allow_config_override": os.environ.get("ALLOW_CONFIG_OVERRIDE", "0"),
        "allow_model_override": os.environ.get("ALLOW_MODEL_OVERRIDE", "0"),
        "log_dir": str(args.log_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={args.output}")


if __name__ == "__main__":
    main()
