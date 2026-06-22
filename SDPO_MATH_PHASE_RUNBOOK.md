# SDPO-Math Phase Runbook

Notebook commands for 2 GPU SDPO-Math runs.

## Defaults

| Item | Value |
|---|---|
| Python | 3.12 |
| Models | Qwen3 1.7B / 8B |
| Hardware | `a100` default, `h100` optional |
| Phase 2/4 variants | `base_rl sdpo_vanilla sdpo_reliability_gate` |
| Optional variant | `sdpo_reliability` |
| Rollout TP | 2 |
| Rollout quantization | `null` for Phase 2/4 |
| Base RL max seqs | 64 |
| SDPO max seqs | 32 on A100, 48 on H100 |
| SDPO activation offload | true |
| Attention | SDPA |
| LoRA | enabled for trained variants |
| Reliability gate execution | Reliability-weighted, DP-aligned sparse student/teacher forwards |

| Phase | Model | Profile | Steps | Train max | Val max |
|---|---|---|---:|---:|---:|
| Pilot | `Qwen/Qwen3-1.7B` | `fast` | 10 | 512 | 128 |
| Scale decision | `Qwen/Qwen3-8B` | `fast` | 12 | 256 | 64 |
| Thesis | `Qwen/Qwen3-8B` | `balanced` | 32 | 1024 | 256 |

| Profile | Train batch | Rollout n | Workers | Response | Model len | Base tokens | A100 SDPO tokens | A100 SDPO actor len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fast` | 32 | 2 | 32 | 1024 | 3072 | 49152 | 32768 | 3072 |
| `balanced` | 32 | 2 | 32 | 1536 | 4096 | 65536 | 49152 | 4096 |
| `quality` | 32 | 2 | 32 | 2048 | 6144 | 98304 | 49152 | 6144 |

SDPO variants use separate memory knobs: `SDPO_BATCHED_TOKENS`, `SDPO_MAX_NUM_SEQS`, `SDPO_GPU_UTIL`, `SDPO_ACTOR_LEN`, and `SDPO_REPROMPT_LEN`.

## Setup

%%bash
set -euo pipefail

cd /root/SDPO
git pull
chmod +x experiments/math/*.sh experiments/math/*.py
unset PYTHON_VERSION
export SDPO_PYTHON_VERSION=3.12
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
bash experiments/math/setup_math_notebook.sh

## Phase 0

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
unset GPU_UTIL MAX_NUM_SEQS BATCHED_TOKENS
unset SDPO_GPU_UTIL SDPO_MAX_NUM_SEQS SDPO_BATCHED_TOKENS SDPO_ACTOR_LEN SDPO_REPROMPT_LEN
unset ROLLOUT_QUANTIZATION
export ROLLOUT_TP=2
export ENFORCE_EAGER=True

ray stop --force >/dev/null 2>&1 || true
bash experiments/math/run_sdpo_math_live_preflight.sh
python experiments/math/preflight_phase.py

export DRY_RUN=1
export PHASE=pilot
export TRAIN_STEPS=1
export VARIANTS="base_rl sdpo_vanilla sdpo_reliability sdpo_reliability_gate"
export RUN_TAG=preflight_dryrun
export EXP_SUFFIX=preflight_dryrun_seed42
export LOG_DIR="$PROJECT_ROOT/logs/sdpo_math_phase/preflight_dryrun"
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_preflight_dryrun.log
python experiments/math/validate_benchmark_dryrun.py \
  --log-dir "$LOG_DIR" \
  --hardware-profile "$HARDWARE_PROFILE" \
  --profile fast \
  --exp-suffix "$EXP_SUFFIX"

## Phase 1

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=pilot
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
unset GPU_UTIL MAX_NUM_SEQS BATCHED_TOKENS
unset SDPO_GPU_UTIL SDPO_MAX_NUM_SEQS SDPO_BATCHED_TOKENS SDPO_ACTOR_LEN SDPO_REPROMPT_LEN
unset ROLLOUT_QUANTIZATION
export ROLLOUT_TP=2
export ENFORCE_EAGER=True
export TRAIN_STEPS="${TRAIN_STEPS:-10}"
export ULTRA_QUIET="${ULTRA_QUIET:-0}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 2

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=scale_decision
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export BATCHED_TOKENS="${BATCHED_TOKENS:-49152}"
export GPU_UTIL="${GPU_UTIL:-0.72}"
export MAX_NUM_SEQS=64
export SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-32768}"
export SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.58}"
export SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-32}"
export SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-3072}"
export SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-1536}"
export SDPO_ACTIVATION_OFFLOAD="${SDPO_ACTIVATION_OFFLOAD:-True}"
export ENFORCE_EAGER=True
export VARIANTS="${VARIANTS:-base_rl sdpo_vanilla sdpo_reliability_gate}"
export TRAIN_STEPS="${TRAIN_STEPS:-12}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
export EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export VERIFY_PHASE_MODEL="${VERIFY_PHASE_MODEL:-0}"
export ULTRA_QUIET="${ULTRA_QUIET:-0}"
export PROGRESS_WATCH="${PROGRESS_WATCH:-1}"
export PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 3

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"
python experiments/math/inspect_phase_logs.py --log-dir "$LOG_DIR"
python experiments/math/summarize_phase_results.py --log-dir "$LOG_DIR" || true

## Phase 4

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=thesis
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export MAX_NUM_SEQS=64
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export BATCHED_TOKENS="${BATCHED_TOKENS:-65536}"
export GPU_UTIL="${GPU_UTIL:-0.72}"
export SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-49152}"
export SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.56}"
export SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-32}"
export SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-4096}"
export SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-2048}"
export SDPO_ACTIVATION_OFFLOAD="${SDPO_ACTIVATION_OFFLOAD:-True}"
export ENFORCE_EAGER=True
export VARIANTS="${VARIANTS:-base_rl sdpo_vanilla sdpo_reliability_gate}"
export TRAIN_STEPS="${TRAIN_STEPS:-32}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1024}"
export EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
export SAVE_FREQ="${SAVE_FREQ:-${TRAIN_STEPS}}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-256}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export ULTRA_QUIET="${ULTRA_QUIET:-0}"
export PROGRESS_WATCH="${PROGRESS_WATCH:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 5

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

if [[ -z "${LOG_DIR:-}" && -f logs/sdpo_math_phase/latest_thesis_log_dir.txt ]]; then
  LOG_DIR="$(< logs/sdpo_math_phase/latest_thesis_log_dir.txt)"
fi
LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"

python experiments/math/summarize_phase_results.py --log-dir "$LOG_DIR"
python experiments/math/check_phase_report_ready.py \
  --log-dir "$LOG_DIR" \
  --require-checkpoints \
  --expect-phase thesis \
  --expect-model "$THESIS_MODEL_PATH" \
  --expect-profile balanced \
  --expect-seed 42
python experiments/math/download_phase_artifacts.py \
  --log-dir "$LOG_DIR" \
  --include-checkpoints \
  --require-checkpoints
cat "$LOG_DIR/manifest.json"
cat "$LOG_DIR/summary.md"

## Speed Probes

| Probe | Setting |
|---|---|
| Fewer Ray agent actors | `export AGENT_WORKERS=16` |
| Safer SDPO memory | `export SDPO_GPU_UTIL=0.50`, `export SDPO_MAX_NUM_SEQS=16`, `export SDPO_ACTOR_LEN=2048` |
| Faster SDPO memory trial | `export SDPO_GPU_UTIL=0.62`, `export SDPO_MAX_NUM_SEQS=32` |
| Smaller validation | `export VAL_MAX_SAMPLES=32` |
| CUDA graph test | `export ENFORCE_EAGER=False` |
| Shorter thesis | `export TRAIN_STEPS=16`, `export TRAIN_MAX_SAMPLES=512` |

Progress lines include `step_s`, `gen_s`, `oldlp_s`, `upd_s`, and `tok_s`.
