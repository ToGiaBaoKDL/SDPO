#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON:-python3}"

echo "[1/5] Checking shell script syntax"
bash -n \
  experiments/math/math_env.sh \
  experiments/math/setup_math_notebook.sh \
  experiments/math/run_sdpo_math_benchmark.sh \
  experiments/math/run_sdpo_math_live_preflight.sh \
  experiments/math/run_sdpo_math_vanilla.sh \
  experiments/math/run_sdpo_math_reliability.sh \
  experiments/math/run_sdpo_math_smoke.sh
"${PYTHON_BIN}" -m py_compile \
  experiments/math/preflight_phase.py \
  experiments/math/print_failure_context.py \
  experiments/math/check_phase_report_ready.py \
  experiments/math/inspect_phase_logs.py \
  experiments/math/summarize_phase_results.py \
  experiments/math/verify_hf_models.py \
  experiments/math/watch_phase_progress.py \
  experiments/math/write_phase_manifest.py \
  experiments/math/validate_benchmark_dryrun.py
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

expected_defaults = {
    "experiments/math/run_sdpo_math_smoke.sh": "Qwen/Qwen3-1.7B",
    "experiments/math/run_sdpo_math_vanilla.sh": "Qwen/Qwen3-8B",
    "experiments/math/run_sdpo_math_reliability.sh": "Qwen/Qwen3-8B",
}
for path, expected in expected_defaults.items():
    text = Path(path).read_text(encoding="utf-8")
    assert expected in text, f"{path} missing default {expected}"

runner = Path("experiments/math/run_sdpo_math_benchmark.sh").read_text(encoding="utf-8")
for expected in ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
    assert expected in runner, f"benchmark runner missing model default {expected}"
for snippet in [
    'VARIANTS="${VARIANTS:-base_rl sdpo_vanilla sdpo_reliability_gate}"',
    'TRAIN_STEPS="${TRAIN_STEPS:-12}"',
    'TRAIN_STEPS="${TRAIN_STEPS:-32}"',
    'TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1024}"',
    'EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"',
    'SAVE_FREQ="${SAVE_FREQ:-${TRAIN_STEPS}}"',
    'RELIABILITY_GATE_THRESHOLD="${RELIABILITY_GATE_THRESHOLD:-0.4}"',
    "ROLLOUT_TP=1",
    "ROLLOUT_QUANTIZATION=null",
    'actor_rollout_ref.actor.self_distillation.reliability_gate_threshold="${reliability_gate_threshold}"',
]:
    assert snippet in runner, f"benchmark runner missing gate/default logic: {snippet}"

manifest = Path("experiments/math/write_phase_manifest.py").read_text(encoding="utf-8")
for snippet in [
    "variant_hyperparameters",
    "sdpo_reliability_gate",
    "RELIABILITY_GATE_THRESHOLD",
    "ROLLOUT_QUANTIZATION",
    "ROLLOUT_TP",
]:
    assert snippet in manifest, f"manifest writer missing gate hyperparameter: {snippet}"

main_ppo = Path("verl/trainer/main_ppo.py").read_text(encoding="utf-8")
for snippet in [
    "def write_progress_heartbeat",
    'write_progress_heartbeat(config, "ray_init_start")',
    'write_progress_heartbeat(config, "task_start")',
    'write_progress_heartbeat(config, "init_workers_start")',
    'write_progress_heartbeat(config, "fit_start")',
]:
    assert snippet in main_ppo, f"main_ppo missing startup progress heartbeat: {snippet}"

trainer = Path("verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
for snippet in [
    "def _progress_heartbeat",
    "VERL_FILE_LOGGER_ROOT",
    ".progress.jsonl",
    'self._progress_heartbeat("step_start")',
    'self._progress_heartbeat("gen_start")',
    'self._progress_heartbeat("actor_update_done")',
]:
    assert snippet in trainer, f"trainer missing progress heartbeat: {snippet}"

watcher = Path("experiments/math/watch_phase_progress.py").read_text(encoding="utf-8")
for snippet in [
    "def progress_path",
    "waiting_for_progress",
    "stage=",
    "read_jsonl_from",
]:
    assert snippet in watcher, f"watcher missing progress heartbeat support: {snippet}"

phase_common = Path("experiments/math/phase_common.sh").read_text(encoding="utf-8")
assert "+ray_kwargs.ray_init.log_to_driver=False" in phase_common
assert "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=" in phase_common
assert "\n      ray_kwargs.ray_init.log_to_driver=False" not in phase_common
for snippet in [
    "a100:fast)",
    "a100:balanced)",
    "a100:quality)",
    "h100:fast)",
    "h100:balanced)",
    "h100:quality)",
    "TRAIN_BS=32",
    "ROLLOUT_N=2",
    'ROLLOUT_TP="${ROLLOUT_TP:-2}"',
    "AGENT_WORKERS=32",
    'MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"',
    'actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"',
    'ENFORCE_EAGER="${ENFORCE_EAGER:-True}"',
    'actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}"',
    'actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}"',
    'actor_rollout_ref.rollout.quantization="${ROLLOUT_QUANTIZATION:-null}"',
]:
    assert snippet in phase_common, f"missing H100 profile setting: {snippet}"

quiet_env = Path("experiments/math/common_quiet_env.sh").read_text(encoding="utf-8")
assert 'VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"' in quiet_env
assert "unset RAY_BACKEND_LOG_LEVEL" in quiet_env
assert "export RAY_BACKEND_LOG_LEVEL" not in quiet_env

setup = Path("experiments/math/setup_math_notebook.sh").read_text(encoding="utf-8")
for snippet in [
    'RUN_CPU_CHECK="${RUN_CPU_CHECK:-0}"',
    'VERIFY_HF_MODELS="${VERIFY_HF_MODELS:-0}"',
    'SKIP_INSTALL_IF_READY="${SKIP_INSTALL_IF_READY:-1}"',
    'FORCE_REINSTALL="${FORCE_REINSTALL:-0}"',
    "ensure_uv",
    "uv venv .venv",
    'uv pip install -q -e ".[vllm]"',
    "skip_dependency_install=1",
]:
    assert snippet in setup, f"setup script missing lightweight default: {snippet}"
for snippet in [
    "python3 -m pip",
    "python -m pip",
    "uv pip install -q -U pip",
    "RUN_TRANSFORMERS_LOAD_SMOKE",
    "--load-smoke-model",
    "RUN_VLLM_LOAD_SMOKE",
    "RUN_STANDALONE_VLLM_LOAD_SMOKE",
    "--vllm-smoke-model",
]:
    assert snippet not in setup, f"setup script should not run standalone model smoke: {snippet}"
PY

echo "[2/5] Checking YAML config"
"${PYTHON_BIN}" - <<'PY'
import yaml

with open("verl/trainer/config/sdpo_math_a100.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

assert cfg["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == "sdpo"
assert cfg["actor_rollout_ref"]["model"]["path"] == "Qwen/Qwen3-8B"
assert cfg["critic"]["model"]["path"] == "Qwen/Qwen3-8B"
assert cfg["data"]["train_batch_size"] == 24
assert "val_batch_size" not in cfg["data"]
assert cfg["actor_rollout_ref"]["rollout"]["agent"]["num_workers"] == 32
assert cfg["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"] == 49152
assert cfg["actor_rollout_ref"]["rollout"]["max_num_seqs"] == 64
assert cfg["actor_rollout_ref"]["rollout"]["enforce_eager"] is True
assert cfg["actor_rollout_ref"]["rollout"]["val_kwargs"]["temperature"] == 0.01
assert cfg["actor_rollout_ref"]["model"]["lora_rank"] > 0
assert cfg["actor_rollout_ref"]["actor"]["self_distillation"]["reliability_weighting"] is False
assert cfg["actor_rollout_ref"]["actor"]["self_distillation"]["reliability_gate_threshold"] == 0.0
assert cfg["trainer"]["n_gpus_per_node"] == 2
assert cfg["reward_manager"]["name"] == "naive"
print("config ok")
PY

echo "[3/5] Checking prepared DAPO-Math parquet"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import pyarrow.parquet as pq

required = [
    Path("data/dapo_math_en/train.parquet"),
    Path("data/dapo_math_en/val.parquet"),
    Path("data/dapo_math_en/removed_duplicates.jsonl"),
    Path("data/dapo_math_en/removed_contaminated.jsonl"),
    Path("reports/decontamination_report.md"),
]
missing = [str(path) for path in required if not path.exists()]
assert not missing, f"missing files: {missing}"

for split, expected_min_rows in [("train", 1), ("val", 1)]:
    rows = pq.read_table(f"data/dapo_math_en/{split}.parquet").to_pylist()
    assert len(rows) >= expected_min_rows
    assert all(row["data_source"] == "math_dapo" for row in rows)
    assert all(row["reward_model"]["ground_truth"] for row in rows)
    assert all(row["extra_info"].get("feedback_mode") == "safe" for row in rows)
    assert all(
        row["prompt"][0]["content"].endswith(
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        for row in rows
    )
    assert not any("Answer:" in row["prompt"][0]["content"] for row in rows)
    print(split, len(rows), "rows ok")
PY

echo "[4/5] Checking math feedback behavior with optional math-verify"
"${PYTHON_BIN}" - <<'PY'
import importlib.util

spec = importlib.util.spec_from_file_location("math_feedback", "verl/utils/reward_score/feedback/math.py")
math_feedback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_feedback)

cases = [
    ("correct", r"Reasoning... \boxed{34}", "34", {"feedback_mode": "safe"}, 1.0, ""),
    (
        "wrong_safe",
        r"Reasoning... \boxed{35}",
        "34",
        {"feedback_mode": "safe"},
        0.0,
        "Your boxed final answer was parsed, but it is incorrect. Recheck the reasoning and final calculation.",
    ),
    ("wrong_none", r"Reasoning... \boxed{35}", "34", {"feedback_mode": "none"}, 0.0, ""),
    (
        "wrong_oracle",
        r"Reasoning... \boxed{35}",
        "34",
        {"feedback_mode": "oracle"},
        0.0,
        "Your answer is incorrect. The correct answer is 34.",
    ),
    (
        "bad_format",
        "Reasoning... final answer is 35",
        "34",
        {"feedback_mode": "safe"},
        0.0,
        "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}.",
    ),
    (
        "truncated",
        "Reasoning...",
        "34",
        {"feedback_mode": "safe", "truncated": True},
        0.0,
        "Your response was truncated because it exceeded the maximum length.",
    ),
]

math_verify_available = None
for name, prediction, ground_truth, extra_info, expected_score, expected_feedback in cases:
    result = math_feedback.compute_score(prediction, ground_truth, extra_info)
    assert result["score"] == expected_score, (name, result)
    assert result["feedback"] == expected_feedback, (name, result)
    assert result["math_verify_available"] in (0, 1), (name, result)
    if math_verify_available is None:
        math_verify_available = result["math_verify_available"]
    assert result["math_verify_available"] == math_verify_available, (name, result)
    print(name, "ok")

print("math_verify_available:", math_verify_available)
if math_verify_available:
    symbolic = math_feedback.compute_score(r"Reasoning... \boxed{1+1}", "2", {"feedback_mode": "safe"})
    print("math_verify symbolic smoke:", symbolic["score"], symbolic["pred"])
PY

echo "[5/5] Checking benchmark variant dry-run"
DRY_RUN=1 \
HARDWARE_PROFILE=a100 \
PHASE=pilot \
TRAIN_STEPS=1 \
VARIANTS="base_rl sdpo_vanilla sdpo_reliability_gate" \
RUN_TAG=cpu_pipeline_dryrun \
EXP_SUFFIX=cpu_pipeline_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_dryrun" \
  --hardware-profile a100 \
  --profile fast \
  --exp-suffix cpu_pipeline_dryrun_seed42

DRY_RUN=1 \
HARDWARE_PROFILE=h100 \
PHASE=pilot \
TRAIN_STEPS=1 \
VARIANTS="base_rl sdpo_vanilla sdpo_reliability_gate" \
RUN_TAG=cpu_pipeline_h100_dryrun \
EXP_SUFFIX=cpu_pipeline_h100_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_h100_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_h100_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_h100_dryrun" \
  --hardware-profile h100 \
  --profile fast \
  --exp-suffix cpu_pipeline_h100_dryrun_seed42

DRY_RUN=1 \
HARDWARE_PROFILE=a100 \
PHASE=scale_decision \
TRAIN_STEPS=1 \
VARIANTS="base_rl sdpo_vanilla sdpo_reliability_gate" \
RUN_TAG=cpu_pipeline_phase2_dryrun \
EXP_SUFFIX=cpu_pipeline_phase2_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_phase2_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_phase2_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_phase2_dryrun" \
  --phase scale_decision \
  --hardware-profile a100 \
  --profile fast \
  --exp-suffix cpu_pipeline_phase2_dryrun_seed42

DRY_RUN=1 \
HARDWARE_PROFILE=a100 \
PHASE=thesis \
TRAIN_STEPS=1 \
VARIANTS="base_rl sdpo_vanilla sdpo_reliability_gate" \
RUN_TAG=cpu_pipeline_thesis_dryrun \
EXP_SUFFIX=cpu_pipeline_thesis_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_thesis_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_thesis_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_thesis_dryrun" \
  --phase thesis \
  --hardware-profile a100 \
  --profile balanced \
  --exp-suffix cpu_pipeline_thesis_dryrun_seed42

echo "CPU pipeline checks passed"
