#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG_NAME="${CONFIG_NAME:-sdpo_math_l40s}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
EXP_NAME="${EXP_NAME:-sdpo_math_safe_feedback_l40s}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
LOGGER="${LOGGER:-[\"console\"]}"

if [[ ! -f "${PROJECT_ROOT}/data/dapo_math_en/train.parquet" ]]; then
  echo "Missing data/dapo_math_en/train.parquet. Run Stage 1/2 preprocessing first." >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
  --config-name "${CONFIG_NAME}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  critic.model.path="${MODEL_PATH}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.group_name="SDPO-Math-Safe-Feedback" \
  trainer.logger="${LOGGER}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
  "$@"
