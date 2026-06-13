#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/phase_common.sh"

if [[ ! -f "${PROJECT_ROOT}/data/dapo_math_en/train.parquet" ]]; then
  echo "Missing data/dapo_math_en/train.parquet. Run DAPO-Math preprocessing first." >&2
  exit 1
fi

PHASE="${PHASE:-pilot}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOGGER="${LOGGER:-[\"console\"]}"
VARIANTS="${VARIANTS:-base_model base_rl sdpo_vanilla sdpo_reliability}"
DRY_RUN="${DRY_RUN:-0}"

case "${PHASE}" in
  pilot)
    RUN_PROFILE="${RUN_PROFILE:-fast}"
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
    TRAIN_STEPS="${TRAIN_STEPS:-10}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
    EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Pilot}"
    ;;
  scale_decision|ablation)
    RUN_PROFILE="${RUN_PROFILE:-balanced}"
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
    TRAIN_STEPS="${TRAIN_STEPS:-50}"
    case "${RUN_PROFILE}" in
      fast)
        TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1024}"
        VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
        ;;
      balanced)
        TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-2048}"
        VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-256}"
        ;;
      quality)
        TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-4096}"
        VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-256}"
        ;;
      *)
        echo "PHASE=scale_decision supports RUN_PROFILE=fast, balanced, or quality." >&2
        exit 1
        ;;
    esac
    EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Scale-Decision}"
    ;;
  thesis)
    RUN_PROFILE="${RUN_PROFILE:-quality}"
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
    TRAIN_STEPS="${TRAIN_STEPS:-300}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
    EVAL_FREQ="${EVAL_FREQ:-100}"
    SAVE_FREQ="${SAVE_FREQ:-100}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Thesis}"
    ;;
  scale_7b)
    RUN_PROFILE="${RUN_PROFILE:-a100_7b}"
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
    TRAIN_STEPS="${TRAIN_STEPS:-300}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
    EVAL_FREQ="${EVAL_FREQ:-100}"
    SAVE_FREQ="${SAVE_FREQ:-100}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Scale-7B}"
    ;;
  *)
    echo "Unknown PHASE=${PHASE}. Use pilot, scale_decision, thesis, or scale_7b." >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES LOGGER MODEL_PATH
export TRAIN_MAX_SAMPLES VAL_MAX_SAMPLES

RUN_TAG="${RUN_TAG:-${PHASE}_${RUN_PROFILE}_${TRAIN_STEPS}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/sdpo_math_phase/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

sdpo_math_prepare_phase_run "${RUN_PROFILE}" "${LOG_DIR}"

echo "phase=${PHASE} model=${MODEL_PATH} variants=${VARIANTS} dry_run=${DRY_RUN}"
echo "steps=${TRAIN_STEPS} train_max=${TRAIN_MAX_SAMPLES} val_max=${VAL_MAX_SAMPLES} eval_freq=${EVAL_FREQ} save_freq=${SAVE_FREQ}"
echo "logs=${LOG_DIR}"

run_with_log() {
  local exp_name="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf "DRY_RUN command:"
      printf " %q" "$@"
      printf "\n"
    } | tee "${LOG_DIR}/${exp_name}.log"
    return 0
  fi
  ray stop --force >/dev/null 2>&1 || true
  "$@" 2>&1 | tee "${LOG_DIR}/${exp_name}.log"
}

run_base_model_val() {
  local exp_name="$1"
  shift
  run_with_log "${exp_name}" \
    python3 -m verl.trainer.main_ppo \
      --config-name sdpo_math_l40s \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.val_before_train=True \
      trainer.val_only=True \
      trainer.save_freq=-1 \
      data.train_max_samples=8 \
      data.val_max_samples="${VAL_MAX_SAMPLES}" \
      actor_rollout_ref.model.lora_rank=0 \
      actor_rollout_ref.model.lora_alpha=16 \
      actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
      actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
      actor_rollout_ref.actor.self_distillation.reliability_weighting=False \
      "${RAY_LOG_TO_DRIVER_OVERRIDE[@]}" \
      "${COMMON_OVERRIDES[@]}" \
      "$@"
}

run_base_rl() {
  local exp_name="$1"
  shift
  run_with_log "${exp_name}" \
    python3 -m verl.trainer.main_ppo \
      --config-name sdpo_math_l40s \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.total_training_steps="${TRAIN_STEPS}" \
      trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
      trainer.test_freq="${EVAL_FREQ}" \
      trainer.save_freq="${SAVE_FREQ}" \
      data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
      data.val_max_samples="${VAL_MAX_SAMPLES}" \
      actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
      actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
      actor_rollout_ref.actor.self_distillation.reliability_weighting=False \
      "${RAY_LOG_TO_DRIVER_OVERRIDE[@]}" \
      "${COMMON_OVERRIDES[@]}" \
      "$@"
}

run_sdpo_variant() {
  local variant="$1"
  local exp_name="$2"
  shift 2
  local include_feedback=True
  local reliability=False

  case "${variant}" in
    sdpo_vanilla)
      ;;
    sdpo_reliability)
      reliability=True
      ;;
    *)
      echo "Unknown SDPO variant=${variant}" >&2
      exit 1
      ;;
  esac

  run_with_log "${exp_name}" \
    python3 -m verl.trainer.main_ppo \
      --config-name sdpo_math_l40s \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.total_training_steps="${TRAIN_STEPS}" \
      trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
      trainer.test_freq="${EVAL_FREQ}" \
      trainer.save_freq="${SAVE_FREQ}" \
      data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
      data.val_max_samples="${VAL_MAX_SAMPLES}" \
      actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
      actor_rollout_ref.actor.self_distillation.include_environment_feedback="${include_feedback}" \
      actor_rollout_ref.actor.self_distillation.reliability_weighting="${reliability}" \
      "${RAY_LOG_TO_DRIVER_OVERRIDE[@]}" \
      "${COMMON_OVERRIDES[@]}" \
      "$@"
}

for variant in ${VARIANTS}; do
  exp_name="${variant}_${PHASE}_${RUN_PROFILE}_${TRAIN_STEPS}"
  echo
  echo "== variant=${variant} exp=${exp_name} =="
  case "${variant}" in
    base_model)
      run_base_model_val "${exp_name}" "$@"
      ;;
    base_rl)
      run_base_rl "${exp_name}" "$@"
      ;;
    sdpo_vanilla|sdpo_reliability)
      run_sdpo_variant "${variant}" "${exp_name}" "$@"
      ;;
    *)
      echo "Unknown variant=${variant}. Valid: base_model base_rl sdpo_vanilla sdpo_reliability." >&2
      exit 1
      ;;
  esac
done

echo
echo "done logs=${LOG_DIR}"
