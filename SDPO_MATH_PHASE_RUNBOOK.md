# SDPO-Math Runbook

## Setup

%%bash
set -euo pipefail

cd /root/SDPO
git pull
chmod +x experiments/math/*.sh experiments/math/*.py
unset PYTHON_VERSION
export SDPO_PYTHON_VERSION=3.12
export HARDWARE_PROFILE=h200
bash experiments/math/setup_math_notebook.sh

## Thesis H200

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=thesis
export HARDWARE_PROFILE=h200
# Change this when training one variant at a time:
#   base_rl
#   sdpo_vanilla
#   sdpo_reliability_gate
export VARIANTS="${VARIANTS:-sdpo_vanilla}"
export TRAIN_STEPS="${TRAIN_STEPS:-10}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1536}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
export TRAJECTORY_LOG_SAMPLES="${TRAJECTORY_LOG_SAMPLES:-16}"
export TRAJECTORY_LOG_TEXT_CHARS="${TRAJECTORY_LOG_TEXT_CHARS:-6000}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"
export PROGRESS_WATCH="${PROGRESS_WATCH:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Process Logs

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export LOG_DIR="${LOG_DIR:-$(< logs/sdpo_math_phase/latest_thesis_log_dir.txt)}"

python experiments/math/summarize_phase_results.py --log-dir "$LOG_DIR"
python experiments/math/check_phase_report_ready.py \
  --log-dir "$LOG_DIR" \
  --require-checkpoints \
  --expect-phase thesis \
  --expect-model "$THESIS_MODEL_PATH" \
  --expect-profile quality \
  --expect-seed 42

cat "$LOG_DIR/summary.md"

## AIME 2026 Benchmark

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export LOG_DIR="${LOG_DIR:-$(< logs/sdpo_math_phase/latest_thesis_log_dir.txt)}"
export AIME2026_DATASET_NAME="MathArena/aime_2026"
export AIME2026_SPLIT="train"
export AIME2026_PROBLEM_KEY="problem"
export AIME2026_ANSWER_KEY="answer"
export AIME2026_ID_KEY="problem_idx"
export AIME2026_FORCE_PREPARE="${AIME2026_FORCE_PREPARE:-0}"
export AIME2026_N_SAMPLES="${AIME2026_N_SAMPLES:-1}"
export AIME2026_BATCH_SIZE="${AIME2026_BATCH_SIZE:-8}"
export AIME2026_MAX_NEW_TOKENS="${AIME2026_MAX_NEW_TOKENS:-2048}"
export AIME2026_DTYPE="${AIME2026_DTYPE:-bfloat16}"
export AIME2026_DEVICE_MAP="${AIME2026_DEVICE_MAP:-auto}"
# Optional. If unset, benchmarks base_model plus checkpointed trained variants found for LOG_DIR.
# If set, it benchmarks exactly these variants.
# export AIME2026_VARIANTS="sdpo_vanilla"
# export AIME2026_VARIANTS="base_model sdpo_vanilla"

bash experiments/math/run_aime2026_benchmark.sh

## Download Artifacts

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export LOG_DIR="${LOG_DIR:-$(< logs/sdpo_math_phase/latest_thesis_log_dir.txt)}"

python experiments/math/download_phase_artifacts.py \
  --log-dir "$LOG_DIR" \
  --include-checkpoints \
  --require-checkpoints
