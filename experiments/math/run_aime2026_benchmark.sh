#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/math_env.sh"

AIME2026_DATA_PATH="${AIME2026_DATA_PATH:-${PROJECT_ROOT}/data/aime2026/test.parquet}"
AIME2026_DATASET_NAME="${AIME2026_DATASET_NAME:-MathArena/aime_2026}"
AIME2026_SPLIT="${AIME2026_SPLIT:-train}"
AIME2026_PROBLEM_KEY="${AIME2026_PROBLEM_KEY:-problem}"
AIME2026_ANSWER_KEY="${AIME2026_ANSWER_KEY:-answer}"
AIME2026_ID_KEY="${AIME2026_ID_KEY:-problem_idx}"
AIME2026_FORCE_PREPARE="${AIME2026_FORCE_PREPARE:-0}"

if [[ "${AIME2026_FORCE_PREPARE}" == "1" || ! -f "${AIME2026_DATA_PATH}" ]]; then
  PREPARE_ARGS=(
    --dataset-name "${AIME2026_DATASET_NAME}"
    --split "${AIME2026_SPLIT}"
    --output-path "${AIME2026_DATA_PATH}"
    --problem-key "${AIME2026_PROBLEM_KEY}"
    --answer-key "${AIME2026_ANSWER_KEY}"
    --id-key "${AIME2026_ID_KEY}"
  )
  python experiments/math/prepare_aime2026.py "${PREPARE_ARGS[@]}"
fi

BENCHMARK_ARGS=(
  --data-path "${AIME2026_DATA_PATH}" \
  --checkpoint-root "${AIME2026_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/sdpo_math}" \
  --n-samples "${AIME2026_N_SAMPLES:-1}" \
  --batch-size "${AIME2026_BATCH_SIZE:-4}" \
  --max-prompt-tokens "${AIME2026_MAX_PROMPT_TOKENS:-2048}" \
  --max-new-tokens "${AIME2026_MAX_NEW_TOKENS:-2048}" \
  --temperature "${AIME2026_TEMPERATURE:-0.0}" \
  --top-p "${AIME2026_TOP_P:-1.0}" \
  --top-k "${AIME2026_TOP_K:--1}" \
  --device-map "${AIME2026_DEVICE_MAP:-auto}" \
  --dtype "${AIME2026_DTYPE:-bfloat16}" \
  --limit "${AIME2026_LIMIT:--1}"
)
if [[ -n "${AIME2026_VARIANTS:-}" ]]; then
  BENCHMARK_ARGS+=(--variants ${AIME2026_VARIANTS})
fi
if [[ -n "${AIME2026_OUTPUT_DIR:-}" ]]; then
  BENCHMARK_ARGS+=(--output-dir "${AIME2026_OUTPUT_DIR}")
fi
if [[ -n "${AIME2026_MODEL_PATH:-}" ]]; then
  BENCHMARK_ARGS+=(--model-path "${AIME2026_MODEL_PATH}")
fi
if [[ -n "${LOG_DIR:-}" ]]; then
  BENCHMARK_ARGS+=(--log-dir "${LOG_DIR}")
fi
if [[ -n "${AIME2026_EXP_SUFFIX:-}" ]]; then
  BENCHMARK_ARGS+=(--exp-suffix "${AIME2026_EXP_SUFFIX}")
fi

python experiments/math/run_aime2026_benchmark.py "${BENCHMARK_ARGS[@]}"
