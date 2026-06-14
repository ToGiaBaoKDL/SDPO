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


def forbid_snippet(path: str, text: str, snippet: str) -> None:
    if snippet in text:
        raise AssertionError(f"{path} is stale or inconsistent; remove: {snippet}")


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
        "train_batch_size": cfg["data"]["train_batch_size"],
        "agent_workers": cfg["actor_rollout_ref"]["rollout"]["agent"]["num_workers"],
        "max_num_batched_tokens": cfg["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"],
        "enforce_eager": cfg["actor_rollout_ref"]["rollout"]["enforce_eager"],
        "use_remove_padding": cfg["actor_rollout_ref"]["model"]["use_remove_padding"],
        "dataloader_workers": cfg["data"]["dataloader_num_workers"],
        "filter_workers": cfg["data"]["filter_overlong_prompts_workers"],
    }
    print("config:", checks)
    assert "data/dapo_math_en/train.parquet" in checks["train_files"][0]
    assert "data/dapo_math_en/val.parquet" in checks["val_files"][0]
    assert checks["actor_attn"] == "sdpa"
    assert checks["critic_attn"] == "sdpa"
    assert checks["actor_model"] == "Qwen/Qwen3-8B"
    assert checks["critic_model"] == "Qwen/Qwen3-8B"
    assert checks["train_batch_size"] == 24
    assert "val_batch_size" not in cfg["data"]
    assert checks["agent_workers"] == 32
    assert checks["max_num_batched_tokens"] == 49152
    assert checks["enforce_eager"] is True
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
        "a100:fast)",
        "a100:balanced)",
        "a100:quality)",
        "h100:fast)",
        "h100:balanced)",
        "h100:quality)",
        "TRAIN_BS=32",
        "TRAIN_BS=48",
        "TRAIN_BS=64",
        "AGENT_WORKERS=64",
        "AGENT_WORKERS=96",
        "AGENT_WORKERS=128",
        "BATCHED_TOKENS=65536",
        "BATCHED_TOKENS=131072",
        "ENFORCE_EAGER",
        "effective_rollouts",
        'actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}"',
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.01",
    ]:
        require_snippet(phase_common_path, phase_common, snippet)
    live_preflight_path = "experiments/math/run_sdpo_math_live_preflight.sh"
    live_preflight = Path(live_preflight_path).read_text(encoding="utf-8")
    for snippet in [
        'VARIANTS="${VARIANTS:-base_model}"',
        'TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-64}"',
        'BASE_MODEL_TRAIN_MAX_SAMPLES="${BASE_MODEL_TRAIN_MAX_SAMPLES:-64}"',
        'VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"',
        'bash "${SCRIPT_DIR}/run_sdpo_math_benchmark.sh"',
    ]:
        require_snippet(live_preflight_path, live_preflight, snippet)
    forbid_snippet(live_preflight_path, live_preflight, 'RUN_PROFILE="${RUN_PROFILE:-fast}"')

    setup_path = "experiments/math/setup_math_notebook.sh"
    setup = Path(setup_path).read_text(encoding="utf-8")
    for snippet in [
        "SDPO_PYTHON_VERSION",
        'RUN_CPU_CHECK="${RUN_CPU_CHECK:-0}"',
        'VERIFY_HF_MODELS="${VERIFY_HF_MODELS:-0}"',
        "transformers==4.57.1",
        "numpy==2.1.0",
    ]:
        require_snippet(setup_path, setup, snippet)
    for snippet in [
        "RUN_TRANSFORMERS_LOAD_SMOKE",
        "--load-smoke-model",
        "RUN_VLLM_LOAD_SMOKE",
        "RUN_STANDALONE_VLLM_LOAD_SMOKE",
        "--vllm-smoke-model",
    ]:
        forbid_snippet(setup_path, setup, snippet)

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
        "scale_decision)",
        "thesis)",
        "HARDWARE_PROFILE",
        "RUN_PROFILE=fast",
        "RUN_PROFILE=balanced",
        "RUN_PROFILE=quality",
        "Refusing CONFIG_NAME",
        "Refusing MODEL_PATH",
        "--config-name \"${CONFIG_NAME}\"",
        "trainer.validation_data_dir",
        "DRY_RUN",
        "BASE_MODEL_TRAIN_MAX_SAMPLES",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
    ]:
        require_snippet(runner_path, runner, snippet)

    print("phase0_ok")


if __name__ == "__main__":
    main()
