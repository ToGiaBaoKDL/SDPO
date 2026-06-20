#!/usr/bin/env python3
"""Preflight checks for the SDPO-Math phase runbook."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.data_preprocess.dapo_math_processed import DEFAULT_PROMPT_SUFFIX


EXPECTED_VARIANTS = ["base_rl", "sdpo_vanilla", "sdpo_reliability", "sdpo_reliability_gate"]


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
        "max_num_seqs": cfg["actor_rollout_ref"]["rollout"]["max_num_seqs"],
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
    assert checks["max_num_seqs"] == 64
    assert checks["enforce_eager"] is True
    assert checks["use_remove_padding"] is False
    assert checks["dataloader_workers"] == 0
    assert checks["filter_workers"] == 1

    train_table = pq.read_table("data/dapo_math_en/train.parquet", columns=["prompt"])
    val_table = pq.read_table("data/dapo_math_en/val.parquet", columns=["prompt"])
    train_rows = train_table.num_rows
    val_rows = val_table.num_rows
    print("data_rows:", {"train": train_rows, "val": val_rows})
    assert train_rows > 1000
    assert val_rows >= 128
    for table in (train_table, val_table):
        assert all(
            row["prompt"][0]["content"].endswith(DEFAULT_PROMPT_SUFFIX)
            for row in table.to_pylist()
        ), "prepared DAPO-Math prompts are stale; rerun setup_math_notebook.sh"

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
        "ROLLOUT_N=2",
        'AGENT_WORKERS="${AGENT_WORKERS:-32}"',
        "RESPONSE_LEN=1024",
        "RESPONSE_LEN=1536",
        "RESPONSE_LEN=2048",
        "BATCHED_TOKENS=32768",
        "BATCHED_TOKENS=49152",
        "BATCHED_TOKENS=65536",
        'MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"',
        'ROLLOUT_TP="${ROLLOUT_TP:-2}"',
        'GPU_UTIL="${GPU_UTIL:-0.86}"',
        'GPU_UTIL="${GPU_UTIL:-0.80}"',
        'GPU_UTIL="${GPU_UTIL:-0.76}"',
        'ENFORCE_EAGER="${ENFORCE_EAGER:-True}"',
        "effective_rollouts",
        'actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"',
        'actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}"',
        'actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}"',
        'actor_rollout_ref.rollout.quantization="${ROLLOUT_QUANTIZATION:-null}"',
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
        'SKIP_INSTALL_IF_READY="${SKIP_INSTALL_IF_READY:-1}"',
        'FORCE_REINSTALL="${FORCE_REINSTALL:-0}"',
        "skip_dependency_install=1",
        "transformers==4.57.1",
        "numpy==2.1.0",
        "update_prepared_prompts.py",
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
        'TRAIN_STEPS="${TRAIN_STEPS:-12}"',
        'TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"',
        'TRAIN_STEPS="${TRAIN_STEPS:-32}"',
        'TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1024}"',
        "ROLLOUT_TP=2",
        "ROLLOUT_QUANTIZATION=null",
        'EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"',
        'SAVE_FREQ="${SAVE_FREQ:-${TRAIN_STEPS}}"',
        "Refusing CONFIG_NAME",
        "Refusing MODEL_PATH",
        "--config-name \"${CONFIG_NAME}\"",
        "trainer.validation_data_dir",
        "DRY_RUN",
        "BASE_MODEL_TRAIN_MAX_SAMPLES",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        "RELIABILITY_GATE_THRESHOLD",
        "reliability_gate_threshold",
        "sdpo_reliability",
        "sdpo_reliability_gate",
    ]:
        require_snippet(runner_path, runner, snippet)

    manifest_path = "experiments/math/write_phase_manifest.py"
    manifest = Path(manifest_path).read_text(encoding="utf-8")
    for snippet in [
        "variant_hyperparameters",
        "sdpo_reliability",
        "sdpo_reliability_gate",
        "reliability_weighting",
        "RELIABILITY_GATE_THRESHOLD",
        "ROLLOUT_QUANTIZATION",
        "ROLLOUT_TP",
        "MAX_NUM_SEQS",
    ]:
        require_snippet(manifest_path, manifest, snippet)

    report_ready_path = "experiments/math/check_phase_report_ready.py"
    report_ready = Path(report_ready_path).read_text(encoding="utf-8")
    for snippet in [
        'REQUIRED_VARIANTS = {"base_rl", "sdpo_vanilla", "sdpo_reliability", "sdpo_reliability_gate"}',
        "reliability_weighting",
        "sdpo_reliability",
        "sdpo_reliability_gate",
        "reliability_gate_threshold",
    ]:
        require_snippet(report_ready_path, report_ready, snippet)

    main_ppo_path = "verl/trainer/main_ppo.py"
    main_ppo = Path(main_ppo_path).read_text(encoding="utf-8")
    for snippet in [
        "def write_progress_heartbeat",
        'write_progress_heartbeat(config, "ray_init_start")',
        'write_progress_heartbeat(config, "task_start")',
        'write_progress_heartbeat(config, "init_workers_start")',
        'write_progress_heartbeat(config, "fit_start")',
    ]:
        require_snippet(main_ppo_path, main_ppo, snippet)

    trainer_path = "verl/trainer/ppo/ray_trainer.py"
    trainer = Path(trainer_path).read_text(encoding="utf-8")
    for snippet in [
        "def _progress_heartbeat",
        "VERL_FILE_LOGGER_ROOT",
        ".progress.jsonl",
        'self._progress_heartbeat("step_start")',
        'self._progress_heartbeat("gen_start")',
        'self._progress_heartbeat("actor_update_done")',
    ]:
        require_snippet(trainer_path, trainer, snippet)

    watcher_path = "experiments/math/watch_phase_progress.py"
    watcher = Path(watcher_path).read_text(encoding="utf-8")
    for snippet in [
        "def progress_path",
        "waiting_for_progress",
        "stage=",
        '"timing_s/gen": "gen_s"',
        '"timing_s/old_log_prob": "oldlp_s"',
        '"response_length/mean": "resp_tok"',
        "read_jsonl_from",
    ]:
        require_snippet(watcher_path, watcher, snippet)

    summary_path = "experiments/math/summarize_phase_results.py"
    summary = Path(summary_path).read_text(encoding="utf-8")
    for snippet in [
        "sorted(VARIANTS, key=len, reverse=True)",
        "time_per_step_s",
        "old_log_prob_s",
        "response_length_mean",
        "response_length_clip_ratio",
        'data.get("timing_s/update_actor", "")',
    ]:
        require_snippet(summary_path, summary, snippet)

    download_path = "experiments/math/download_phase_artifacts.py"
    download = Path(download_path).read_text(encoding="utf-8")
    for snippet in [
        "latest_thesis_log_dir.txt",
        "include-checkpoints",
        "require-checkpoints",
        "latest_checkpointed_iteration.txt",
    ]:
        require_snippet(download_path, download, snippet)

    print("phase0_ok")


if __name__ == "__main__":
    main()
