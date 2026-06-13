#!/usr/bin/env python3
"""Preflight checks for the SDPO-Math phase runbook."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq
import yaml


EXPECTED_VARIANTS = ["base_model", "base_rl", "sdpo_vanilla", "sdpo_reliability"]


def require_snippet(path: str, text: str, snippet: str) -> None:
    if snippet not in text:
        raise AssertionError(f"{path} is stale or inconsistent; missing: {snippet}")


def main() -> None:
    config_path = Path("verl/trainer/config/sdpo_math_a100.yaml")
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checks = {
        "train_files": cfg["data"]["train_files"],
        "val_files": cfg["data"]["val_files"],
        "actor_attn": cfg["actor_rollout_ref"]["model"]["override_config"]["attn_implementation"],
        "critic_attn": cfg["critic"]["model"]["override_config"]["attn_implementation"],
        "actor_model": cfg["actor_rollout_ref"]["model"]["path"],
        "critic_model": cfg["critic"]["model"]["path"],
        "use_remove_padding": cfg["actor_rollout_ref"]["model"]["use_remove_padding"],
        "dataloader_workers": cfg["data"]["dataloader_num_workers"],
        "filter_workers": cfg["data"]["filter_overlong_prompts_workers"],
    }
    print("config:", checks)
    assert "data/dapo_math_en/train.parquet" in checks["train_files"][0]
    assert "data/dapo_math_en/val.parquet" in checks["val_files"][0]
    assert checks["actor_attn"] == "sdpa"
    assert checks["critic_attn"] == "sdpa"
    assert checks["actor_model"] == "Qwen/Qwen3.5-9B"
    assert checks["critic_model"] == "Qwen/Qwen3.5-9B"
    assert checks["use_remove_padding"] is False
    assert checks["dataloader_workers"] == 0
    assert checks["filter_workers"] == 1

    train_rows = pq.read_table("data/dapo_math_en/train.parquet").num_rows
    val_rows = pq.read_table("data/dapo_math_en/val.parquet").num_rows
    print("data_rows:", {"train": train_rows, "val": val_rows})
    assert train_rows > 1000
    assert val_rows >= 128

    phase_common_path = "experiments/math/phase_common.sh"
    phase_common = Path(phase_common_path).read_text(encoding="utf-8")
    for snippet in [
        "fast)",
        "balanced)",
        "quality)",
        "high_mem_9b|a100_9b)",
        "AGENT_WORKERS=16",
        "BATCHED_TOKENS=32768",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
    ]:
        require_snippet(phase_common_path, phase_common, snippet)

    runner_path = "experiments/math/run_sdpo_math_benchmark.sh"
    runner = Path(runner_path).read_text(encoding="utf-8")
    variant_match = re.search(r'^VARIANTS="\$\{VARIANTS:-(?P<variants>[^"]+)\}"', runner, re.MULTILINE)
    if not variant_match:
        raise AssertionError(f"{runner_path} is missing the VARIANTS default")
    actual_variants = variant_match.group("variants").split()
    if actual_variants != EXPECTED_VARIANTS:
        raise AssertionError(
            f"{runner_path} has stale benchmark variants: "
            f"actual={actual_variants}, expected={EXPECTED_VARIANTS}. "
            "Run git pull in /root/SDPO or copy the latest benchmark script."
        )

    for snippet in [
        "scale_decision|ablation)",
        "thesis)",
        "scale_9b)",
        "--config-name \"${CONFIG_NAME}\"",
        "trainer.validation_data_dir",
        "DRY_RUN",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
    ]:
        require_snippet(runner_path, runner, snippet)

    print("phase0_ok")


if __name__ == "__main__":
    main()
