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
CONFIG_NAME="${CONFIG_NAME:-sdpo_math_a100}"
VARIANTS="${VARIANTS:-base_model base_rl sdpo_vanilla sdpo_reliability}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-42}"
VERIFY_PHASE_MODEL="${VERIFY_PHASE_MODEL:-1}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"

case "${HARDWARE_PROFILE}" in
  a100|h100)
    ;;
  *)
    echo "Unknown HARDWARE_PROFILE=${HARDWARE_PROFILE}. Use a100 or h100." >&2
    exit 1
    ;;
esac

case "${PHASE}" in
  pilot)
    RUN_PROFILE=fast
    MODEL_PATH="${MODEL_PATH:-${PILOT_MODEL_PATH:-Qwen/Qwen3-1.7B}}"
    TRAIN_STEPS="${TRAIN_STEPS:-10}"
    if [[ "${HARDWARE_PROFILE}" == "h100" ]]; then
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-512}"
    else
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-512}"
    fi
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
    EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Pilot}"
    ;;
  scale_decision)
    RUN_PROFILE=balanced
    MODEL_PATH="${MODEL_PATH:-${SCALE_MODEL_PATH:-Qwen/Qwen3-4B}}"
    TRAIN_STEPS="${TRAIN_STEPS:-50}"
    if [[ "${HARDWARE_PROFILE}" == "h100" ]]; then
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-4096}"
    else
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-4096}"
    fi
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-256}"
    EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Scale-Decision}"
    ;;
  thesis)
    RUN_PROFILE=quality
    MODEL_PATH="${MODEL_PATH:-${THESIS_MODEL_PATH:-Qwen/Qwen3-8B}}"
    TRAIN_STEPS="${TRAIN_STEPS:-300}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
    EVAL_FREQ="${EVAL_FREQ:-100}"
    SAVE_FREQ="${SAVE_FREQ:-100}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Thesis}"
    ;;
  *)
    echo "Unknown PHASE=${PHASE}. Use pilot, scale_decision, or thesis." >&2
    exit 1
    ;;
esac

if [[ "${CONFIG_NAME}" != "sdpo_math_a100" ]]; then
  echo "Refusing CONFIG_NAME=${CONFIG_NAME}. SDPO-Math phases must use sdpo_math_a100." >&2
  exit 1
fi

case "${MODEL_PATH}" in
  Qwen/Qwen3-1.7B|Qwen/Qwen3-4B|Qwen/Qwen3-8B)
    ;;
  *)
    echo "Refusing MODEL_PATH=${MODEL_PATH}. SDPO-Math phases are locked to Qwen3 1.7B/4B/8B." >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES LOGGER MODEL_PATH HARDWARE_PROFILE
export TRAIN_MAX_SAMPLES VAL_MAX_SAMPLES SEED

RUN_TAG="${RUN_TAG:-${PHASE}_${HARDWARE_PROFILE}_${RUN_PROFILE}_${TRAIN_STEPS}_$(date +%Y%m%d_%H%M%S)}"
EXP_SUFFIX="${EXP_SUFFIX:-${RUN_TAG}_seed${SEED}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/sdpo_math_phase/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"
if [[ "${PHASE}" == "thesis" && "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${PROJECT_ROOT}/logs/sdpo_math_phase"
  printf "%s\n" "${LOG_DIR}" > "${PROJECT_ROOT}/logs/sdpo_math_phase/latest_thesis_log_dir.txt"
fi

sdpo_math_prepare_phase_run "${RUN_PROFILE}" "${LOG_DIR}"

echo "phase=${PHASE} model=${MODEL_PATH} variants=${VARIANTS} dry_run=${DRY_RUN}"
echo "hardware=${HARDWARE_PROFILE}"
echo "steps=${TRAIN_STEPS} train_max=${TRAIN_MAX_SAMPLES} val_max=${VAL_MAX_SAMPLES} eval_freq=${EVAL_FREQ} save_freq=${SAVE_FREQ} seed=${SEED}"
echo "exp_suffix=${EXP_SUFFIX}"
echo "logs=${LOG_DIR}"

if [[ "${DRY_RUN}" != "1" && "${VERIFY_PHASE_MODEL}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/verify_hf_models.py" --models "${MODEL_PATH}"
fi

python3 "${SCRIPT_DIR}/write_phase_manifest.py" \
  --output "${LOG_DIR}/manifest.json" \
  --config-name "${CONFIG_NAME}" \
  --phase "${PHASE}" \
  --profile "${RUN_PROFILE}" \
  --model "${MODEL_PATH}" \
  --variants "${VARIANTS}" \
  --train-steps "${TRAIN_STEPS}" \
  --train-max-samples "${TRAIN_MAX_SAMPLES}" \
  --val-max-samples "${VAL_MAX_SAMPLES}" \
  --eval-freq "${EVAL_FREQ}" \
  --save-freq "${SAVE_FREQ}" \
  --seed "${SEED}" \
  --exp-suffix "${EXP_SUFFIX}" \
  --log-dir "${LOG_DIR}"

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
  local base_model_train_max_samples="${BASE_MODEL_TRAIN_MAX_SAMPLES:-$((TRAIN_BS * 2))}"
  run_with_log "${exp_name}" \
    python3 -m verl.trainer.main_ppo \
      --config-name "${CONFIG_NAME}" \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.val_before_train=True \
      trainer.val_only=True \
      trainer.save_freq=-1 \
      trainer.validation_data_dir="${LOG_DIR}/validation/${exp_name}" \
      data.train_max_samples="${base_model_train_max_samples}" \
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
      --config-name "${CONFIG_NAME}" \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.total_training_steps="${TRAIN_STEPS}" \
      trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
      trainer.test_freq="${EVAL_FREQ}" \
      trainer.save_freq="${SAVE_FREQ}" \
      trainer.validation_data_dir="${LOG_DIR}/validation/${exp_name}" \
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
      --config-name "${CONFIG_NAME}" \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.total_training_steps="${TRAIN_STEPS}" \
      trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
      trainer.test_freq="${EVAL_FREQ}" \
      trainer.save_freq="${SAVE_FREQ}" \
      trainer.validation_data_dir="${LOG_DIR}/validation/${exp_name}" \
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
  exp_name="${variant}_${EXP_SUFFIX}"
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
