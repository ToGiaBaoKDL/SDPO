#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_quiet_env.sh"

sdpo_math_configure_profile() {
  local profile="${1:?profile required}"

  case "${profile}" in
    fast)
      TRAIN_BS=32
      ROLLOUT_N=4
      AGENT_WORKERS=32
      RESPONSE_LEN=1024
      MODEL_LEN=4096
      ACTOR_LEN=4096
      REPROMPT_LEN=2048
      BATCHED_TOKENS=32768
      GPU_UTIL=0.82
      ;;
    balanced)
      TRAIN_BS=32
      ROLLOUT_N=4
      AGENT_WORKERS=32
      RESPONSE_LEN=1536
      MODEL_LEN=5120
      ACTOR_LEN=6144
      REPROMPT_LEN=3072
      BATCHED_TOKENS=49152
      GPU_UTIL=0.86
      ;;
    quality)
      TRAIN_BS=24
      ROLLOUT_N=4
      AGENT_WORKERS=32
      RESPONSE_LEN=2048
      MODEL_LEN=6144
      ACTOR_LEN=8192
      REPROMPT_LEN=4096
      BATCHED_TOKENS=49152
      GPU_UTIL=0.88
      ;;
    high_mem_8b|a100_8b)
      TRAIN_BS=32
      ROLLOUT_N=4
      AGENT_WORKERS=32
      RESPONSE_LEN=2048
      MODEL_LEN=6144
      ACTOR_LEN=8192
      REPROMPT_LEN=4096
      BATCHED_TOKENS=65536
      GPU_UTIL=0.90
      ;;
    *)
      echo "Unknown RUN_PROFILE=${profile}. Use fast, balanced, quality, or high_mem_8b." >&2
      return 1
      ;;
  esac

  export TRAIN_BS ROLLOUT_N AGENT_WORKERS RESPONSE_LEN MODEL_LEN ACTOR_LEN REPROMPT_LEN BATCHED_TOKENS GPU_UTIL
}

sdpo_math_validate_profile() {
  local total_rollouts=$((TRAIN_BS * ROLLOUT_N))
  if (( total_rollouts < AGENT_WORKERS )); then
    echo "Invalid profile: train_batch_size * rollout.n must be >= agent workers." >&2
    return 1
  fi
  if (( total_rollouts % AGENT_WORKERS != 0 )); then
    echo "Invalid profile: train_batch_size * rollout.n must be divisible by agent workers." >&2
    return 1
  fi
}

sdpo_math_init_logging() {
  local log_dir="${1:?log_dir required}"

  export LOGGER="${LOGGER:-[\"console\"]}"
  RAY_LOG_TO_DRIVER_OVERRIDE=()

  if [[ "${ULTRA_QUIET:-0}" == "1" ]]; then
    export LOGGER='["file"]'
    export VERL_FILE_LOGGER_ROOT="${log_dir}/metrics"
    RAY_LOG_TO_DRIVER_OVERRIDE=(
      +ray_kwargs.ray_init.log_to_driver=False
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT="${VERL_FILE_LOGGER_ROOT}"
    )
    echo "ultra_quiet=1 metrics=${VERL_FILE_LOGGER_ROOT}"
  else
    echo "ultra_quiet=0 logger=${LOGGER}"
  fi
}

sdpo_math_build_common_overrides() {
  COMMON_OVERRIDES=(
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.override_config.attn_implementation=sdpa
    critic.model.use_remove_padding=False
    critic.model.override_config.attn_implementation=sdpa
    data.dataloader_num_workers=0
    data.filter_overlong_prompts_workers=1
    data.seed="${SEED:-42}"
    data.train_batch_size="${TRAIN_BS}"
    data.max_response_length="${RESPONSE_LEN}"
    rollout_model_len="${MODEL_LEN}"
    actor_max_token_len="${ACTOR_LEN}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BS}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_LEN}"
    actor_rollout_ref.actor.data_loader_seed="${SEED:-42}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}"
    actor_rollout_ref.rollout.max_model_len="${MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${BATCHED_TOKENS}"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_UTIL}"
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MODEL_LEN}"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0.01
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MODEL_LEN}"
    actor_rollout_ref.actor.self_distillation.max_reprompt_len="${REPROMPT_LEN}"
    critic.data_loader_seed="${SEED:-42}"
  )
}

sdpo_math_prepare_phase_run() {
  local profile="${1:?profile required}"
  local log_dir="${2:?log_dir required}"

  sdpo_math_configure_profile "${profile}"
  sdpo_math_validate_profile
  sdpo_math_init_logging "${log_dir}"
  sdpo_math_build_common_overrides

  echo "profile=${profile} train_bs=${TRAIN_BS} rollout_n=${ROLLOUT_N} effective_rollouts=$((TRAIN_BS * ROLLOUT_N)) agent_workers=${AGENT_WORKERS} response_len=${RESPONSE_LEN} model_len=${MODEL_LEN} batched_tokens=${BATCHED_TOKENS}"
}
