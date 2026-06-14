#!/usr/bin/env python3
"""Validate that an SDPO-Math phase run has enough structured outputs for reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_VARIANTS = {"base_model", "base_rl", "sdpo_vanilla", "sdpo_reliability"}
TRAINED_VARIANTS = {"base_rl", "sdpo_vanilla", "sdpo_reliability"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument("--expect-phase")
    parser.add_argument("--expect-model")
    parser.add_argument("--expect-profile")
    parser.add_argument("--expect-seed", type=int)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    manifest_path = args.log_dir / "manifest.json"
    summary_path = args.log_dir / "summary.csv"
    require(manifest_path.exists(), f"missing {manifest_path}")
    require(summary_path.exists(), f"missing {summary_path}; run summarize_phase_results.py first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_root = Path(manifest.get("project_root") or ".")
    require(set(manifest["variants"]) == REQUIRED_VARIANTS, f"unexpected variants in manifest: {manifest['variants']}")
    require(manifest["seed"] is not None, "manifest missing seed")
    require(manifest["model"], "manifest missing model")
    require(manifest.get("config_name") == "sdpo_math_a100", f"unexpected config_name: {manifest.get('config_name')}")
    require(manifest.get("profile_settings"), "manifest missing profile_settings")
    require(manifest.get("effective_rollouts_per_step"), "manifest missing effective_rollouts_per_step")
    if args.expect_phase:
        require(manifest.get("phase") == args.expect_phase, f"unexpected phase: {manifest.get('phase')}")
    if args.expect_model:
        require(manifest.get("model") == args.expect_model, f"unexpected model: {manifest.get('model')}")
    if args.expect_profile:
        require(manifest.get("profile") == args.expect_profile, f"unexpected profile: {manifest.get('profile')}")
    if args.expect_seed is not None:
        require(manifest.get("seed") == args.expect_seed, f"unexpected seed: {manifest.get('seed')}")

    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    variants = {row["variant"] for row in rows}
    require(variants == REQUIRED_VARIANTS, f"summary variants mismatch: {variants}")

    for row in rows:
        variant = row["variant"]
        require(row["val_acc_mean"] != "", f"{variant} missing val_acc_mean")
        require(row["incorrect_format_mean"] != "", f"{variant} missing incorrect_format_mean")
        require(row["truncated_mean"] != "", f"{variant} missing truncated_mean")
        if variant.startswith("sdpo_"):
            require(row["sdpo_reprompt_fraction"] != "", f"{variant} missing SDPO reprompt metric")
            require(row["sdpo_feedback_used_fraction"] != "", f"{variant} missing SDPO feedback-used metric")
        if variant == "sdpo_reliability":
            require(row["sdpo_reliability_weight_mean"] != "", "sdpo_reliability missing reliability weight metric")

        validation_dir = args.log_dir / "validation" / f"{variant}_{manifest['exp_suffix']}"
        require(validation_dir.exists(), f"{variant} missing validation dump dir: {validation_dir}")
        require(list(validation_dir.glob("*.jsonl")), f"{variant} missing validation jsonl dumps: {validation_dir}")

    if args.require_checkpoints:
        exp_suffix = manifest["exp_suffix"]
        for variant in TRAINED_VARIANTS:
            ckpt_root = project_root / "checkpoints/sdpo_math" / f"{variant}_{exp_suffix}"
            require(ckpt_root.exists(), f"missing checkpoint root for {variant}: {ckpt_root}")
            require(list(ckpt_root.rglob("global_step_*")), f"missing global_step checkpoint for {variant}: {ckpt_root}")

    print("phase_report_ready_ok")


if __name__ == "__main__":
    main()
