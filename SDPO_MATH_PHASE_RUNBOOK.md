# SDPO-Math Phase Runbook

Notebook-ready commands for setting up and running the SDPO-Math benchmark phases.
Use each `%%bash` block as one notebook cell.

The phase runner is centralized in `experiments/math/run_sdpo_math_benchmark.sh`.
Profile settings and shared Hydra overrides live in `experiments/math/phase_common.sh`.
Do not copy training logic into new notebook cells unless you also move it back into these scripts.

Validation in these phases uses `data/dapo_math_en/val.parquet`, the held-out DAPO-Math English split. It is not AIME.

## Required Setup

Use a fresh clone at `/root/SDPO` and pull the latest code before running phases. The setup script creates a Python 3.12 uv venv, installs the project with vLLM, pins NumPy to a numba-compatible version, installs `math-verify`, verifies the Hugging Face model ids and Transformers architecture support, prepares the English DAPO-Math split if missing, and runs CPU/data checks. It does not load model weights by default unless `RUN_VLLM_LOAD_SMOKE=1`; Phase 0 performs the real Transformers/vLLM load smoke.

Run once per notebook VM:

%%bash
set -euo pipefail

echo "== Setup SDPO-Math notebook =="
cd /root/SDPO
git pull
chmod +x experiments/math/*.sh experiments/math/*.py
unset PYTHON_VERSION
export SDPO_PYTHON_VERSION=3.12
bash experiments/math/setup_math_notebook.sh

Useful setup flags: `PREPARE_DATA=0` skips data creation, `RUN_CPU_CHECK=0` skips CPU checks, `VERIFY_HF_MODELS=0` skips Hugging Face model checks, `RUN_TRANSFORMERS_LOAD_SMOKE=1` adds a setup-time Transformers load smoke, `RUN_VLLM_LOAD_SMOKE=1` adds a setup-time vLLM load smoke, `NUMPY_SPEC=numpy==2.1.0` controls the NumPy runtime pin, and `INSTALL_MATH_VERIFY=0` skips `math-verify` installation. For thesis runs, keep `INSTALL_MATH_VERIFY=1`.

The runbook uses only the Qwen3 text checkpoints listed below.

If Phase 0 fails with `Numba needs NumPy 2.2 or less. Got NumPy 2.4`, repair the existing venv with:

%%bash
set -euo pipefail

cd /root/SDPO
source .venv/bin/activate
uv pip install -q -U "numpy==2.1.0"
python - <<'PY'
import importlib.metadata as metadata
print("numpy:", metadata.version("numpy"))
print("numba:", metadata.version("numba"))
PY

The setup script intentionally uses `SDPO_PYTHON_VERSION`, not the generic notebook variable `PYTHON_VERSION`. Use Python 3.12 unless you intentionally set `ALLOW_UNTESTED_PYTHON=1`.

## Shared Cell Prefix

After setup, every phase cell uses `cd /root/SDPO` then `source experiments/math/math_env.sh`.

That helper activates `.venv` if it exists, sets `PROJECT_ROOT`, `PYTHONPATH`, `HF_HOME`, `UV_CACHE_DIR`, quiet logging defaults, `WANDB_MODE=offline`, `CUDA_VISIBLE_DEVICES=0,1`, and default model names.

Model ladder:

- Phase 1 pilot: `Qwen/Qwen3-1.7B`.
- Phase 2 scale decision: `Qwen/Qwen3-4B`.
- Phase 4 thesis: `Qwen/Qwen3-8B`.

Setup and Phase 0 check model compatibility before training.

## Benchmark Variants

There are 4 benchmark variants:

- `base_model`: frozen base-model validation only. No training, no LoRA.
- `base_rl`: trained GRPO/RL baseline with vanilla policy loss.
- `sdpo_vanilla`: feedback-enabled SDPO baseline. It uses successful sibling rollouts when available and safe math feedback when no successful demonstration exists.
- `sdpo_reliability`: SDPO+ with safe feedback plus reliability weighting.

Training phases run 3 trained variants: `base_rl`, `sdpo_vanilla`, and `sdpo_reliability`.
The fourth variant, `base_model`, is an evaluation baseline.

Use `sdpo_vanilla` as the main SDPO baseline in thesis tables. `sdpo_reliability` is the SDPO+ thesis method. Because feedback is configured to be used only when no successful solution demonstration exists, monitor `self_distillation/feedback_used_fraction`: if it is `0`, feedback had no practical effect in that batch.

## Profiles

- `fast`: pilot profile for Qwen3-1.7B on 2x A100. `train_batch_size=32`, `rollout.n=4`, `agent_workers=32`, response length `1024`.
- `balanced`: scale-decision profile for Qwen3-4B on 2x A100. `train_batch_size=32`, `rollout.n=4`, `agent_workers=32`, response length `1536`.
- `quality`: default thesis profile for Qwen3-8B on 2x A100. `train_batch_size=24`, `rollout.n=4`, `agent_workers=32`, response length `2048`.
- `high_mem_8b`: optional maximum-throughput Qwen3-8B profile. It raises train batch size to `32`, keeps `agent_workers=32`, and raises vLLM batched tokens to `65536`.

Recommended order:

1. Phase 0: preflight.
2. Phase 1: pilot, confirms all variants run.
3. Phase 2: scale-decision benchmark.
4. Phase 3: inspect logs.
5. Phase 4: thesis 8B run.
6. Phase 5: optional max-throughput 8B run only after Phase 4 is stable.
7. Phase 6: report-readiness check.

Use `ULTRA_QUIET=1` for long runs. It hides Ray worker stdout and writes metrics JSONL under the run log directory. Keep `ULTRA_QUIET=1` for Phase 4 so `summary.csv` and report-readiness checks can be generated from file logs.

## Phase 0. Preflight

Run after every pull/update before longer experiments.
The Transformers CUDA load smoke and vLLM engine smoke run in separate Python processes to avoid CUDA-context interference. The vLLM smoke uses `gpu_memory_utilization=0.70` intentionally; lower values can make vLLM report zero available KV-cache blocks and fail during engine startup.
`math_env.sh` also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, matching the repo Dockerfiles and preventing CUDA re-initialization errors in vLLM engine subprocesses.

%%bash
set -euo pipefail

echo "== Phase 0: preflight =="
cd /root/SDPO
source experiments/math/math_env.sh

python experiments/math/verify_hf_models.py \
  --models "$PILOT_MODEL_PATH" "$SCALE_MODEL_PATH" "$THESIS_MODEL_PATH"

python experiments/math/verify_hf_models.py \
  --models "$PILOT_MODEL_PATH" \
  --load-smoke-model "$PILOT_MODEL_PATH"

ray stop --force >/dev/null 2>&1 || true
python experiments/math/verify_hf_models.py \
  --models "$PILOT_MODEL_PATH" \
  --vllm-smoke-model "$PILOT_MODEL_PATH" \
  --vllm-tensor-parallel-size 1 \
  --vllm-max-model-len 1024 \
  --vllm-gpu-memory-utilization 0.70
ray stop --force >/dev/null 2>&1 || true

python experiments/math/preflight_phase.py

export DRY_RUN=1
export PHASE=pilot
export RUN_PROFILE=fast
export TRAIN_STEPS=1
export VARIANTS="base_model base_rl sdpo_vanilla sdpo_reliability"
export RUN_TAG=preflight_dryrun
export EXP_SUFFIX=preflight_dryrun_seed42
export LOG_DIR="$PROJECT_ROOT/logs/sdpo_math_phase/preflight_dryrun"

bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_preflight_dryrun.log
python experiments/math/validate_benchmark_dryrun.py --log-dir "$LOG_DIR" --exp-suffix "$EXP_SUFFIX"

## Phase 1. Pilot Test

Purpose: confirm the full 4-variant benchmark shape runs cleanly.

Default: `PHASE=pilot`, `RUN_PROFILE=fast`, `TRAIN_STEPS=10`.
For a very quick test, set `VARIANTS="base_model base_rl sdpo_reliability" TRAIN_STEPS=3`.

%%bash
set -euo pipefail

echo "== Phase 1: pilot test =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE="${PHASE:-pilot}"
export RUN_PROFILE="${RUN_PROFILE:-fast}"
export TRAIN_STEPS="${TRAIN_STEPS:-10}"
export ULTRA_QUIET="${ULTRA_QUIET:-0}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 2. Scale-Decision Benchmark

Purpose: useful comparison before spending on the final thesis run.

Default: `PHASE=scale_decision`, `RUN_PROFILE=balanced`, `TRAIN_STEPS=50`.
Scale up if:

- all 3 train variants finish without OOM or chunk errors;
- `base_rl` is stable;
- SDPO variants log nonzero `self_distillation/reprompt_sample_fraction`;
- `sdpo_vanilla` logs nonzero `self_distillation/feedback_used_fraction` at least sometimes;
- `sdpo_reliability` logs reliability metrics and does not collapse reward versus `sdpo_vanilla`.

%%bash
set -euo pipefail

echo "== Phase 2: scale-decision benchmark =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE="${PHASE:-scale_decision}"
export RUN_PROFILE="${RUN_PROFILE:-balanced}"
export TRAIN_STEPS="${TRAIN_STEPS:-50}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 3. Inspect Logs

Run after Phase 1, 2, 4, or 5. Set `LOG_DIR` to a specific directory if needed.

%%bash
set -euo pipefail

echo "== Phase 3: inspect logs =="
cd /root/SDPO
source experiments/math/math_env.sh

LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"
python experiments/math/inspect_phase_logs.py --log-dir "$LOG_DIR"
python experiments/math/summarize_phase_results.py --log-dir "$LOG_DIR" || true

## Phase 4. Main Thesis Runs

Purpose: final Qwen3-8B thesis comparison.

Default: `PHASE=thesis`, `RUN_PROFILE=quality`, `TRAIN_STEPS=300`, validation every `100` steps.
This runs the frozen `base_model` evaluation plus the 3 train variants.

%%bash
set -euo pipefail

echo "== Phase 4: main thesis runs =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=thesis
export RUN_PROFILE="${RUN_PROFILE:-quality}"
export TRAIN_STEPS="${TRAIN_STEPS:-300}"
export EVAL_FREQ="${EVAL_FREQ:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 5. Optional Max-Throughput 8B Run

Use only after Phase 4 is stable. This keeps the same thesis model and variants, but uses the larger `high_mem_8b` profile.

%%bash
set -euo pipefail

echo "== Phase 5: optional max-throughput 8B run =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=scale_8b
export RUN_PROFILE=high_mem_8b
export TRAIN_STEPS="${TRAIN_STEPS:-300}"
export EVAL_FREQ="${EVAL_FREQ:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 6. Report Readiness

After Phase 4 finishes, use this checklist before writing thesis/article claims.

Required from this runbook:

- `logs/sdpo_math_phase/<run>/manifest.json` exists and records git commit, model, phase, profile, seed, variants, train/val sample counts, eval frequency, and save frequency.
- `logs/sdpo_math_phase/<run>/summary.csv` exists after Phase 3.
- `logs/sdpo_math_phase/<run>/validation/<variant>_<exp_suffix>/*.jsonl` exists for every variant.
- All 4 variants are present: `base_model`, `base_rl`, `sdpo_vanilla`, `sdpo_reliability`.
- Main result column is `val-core/math_dapo/acc/mean@1` from the held-out DAPO-Math English validation split.
- Diagnostic columns include format error, truncation, SDPO reprompt fraction, SDPO feedback-used fraction, and reliability weight mean.
- The final thesis comparison uses one shared model, data split, validation size, generation setting, seed, and profile across all variants.
- Phase 6 should be run on the Phase 4 thesis log directory, not the Phase 1/2 pilot directory. Phase 4 writes `logs/sdpo_math_phase/latest_thesis_log_dir.txt`; Phase 6 uses that pointer by default.

For arXiv-quality claims, DAPO-Math validation alone is not enough. Add at least one external held-out math benchmark and, if compute allows, repeat the Phase 4 comparison with multiple seeds such as `SEED=42`, `SEED=43`, and `SEED=44`.

%%bash
set -euo pipefail
cd /root/SDPO
source experiments/math/math_env.sh

if [[ -z "${LOG_DIR:-}" && -f logs/sdpo_math_phase/latest_thesis_log_dir.txt ]]; then
  LOG_DIR="$(< logs/sdpo_math_phase/latest_thesis_log_dir.txt)"
fi
LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"
EXPECT_PROFILE="${EXPECT_PROFILE:-quality}"
EXPECT_SEED="${EXPECT_SEED:-42}"

python experiments/math/summarize_phase_results.py --log-dir "$LOG_DIR"
python experiments/math/check_phase_report_ready.py \
  --log-dir "$LOG_DIR" \
  --require-checkpoints \
  --expect-phase thesis \
  --expect-model "$THESIS_MODEL_PATH" \
  --expect-profile "$EXPECT_PROFILE" \
  --expect-seed "$EXPECT_SEED"
cat "$LOG_DIR/manifest.json"
cat "$LOG_DIR/summary.md"

## Useful Overrides

Run only training variants:

%%bash
set -euo pipefail
cd /root/SDPO
source experiments/math/math_env.sh
export PHASE=scale_decision
export VARIANTS="base_rl sdpo_vanilla sdpo_reliability"
bash experiments/math/run_sdpo_math_benchmark.sh

Run only the strongest SDPO+ variant:

%%bash
set -euo pipefail
cd /root/SDPO
source experiments/math/math_env.sh
export PHASE=thesis
export VARIANTS="sdpo_reliability"
bash experiments/math/run_sdpo_math_benchmark.sh
