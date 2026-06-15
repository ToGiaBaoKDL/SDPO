# SDPO-Math Phase Runbook

Notebook-ready commands for SDPO-Math on 2 GPUs. Default hardware is 2x A100-80GB. For 2x H100, set `HARDWARE_PROFILE=h100`.

Central files:

- `experiments/math/setup_math_notebook.sh`: Python 3.12 uv environment, uv-based dependency install, data, CPU checks.
- `experiments/math/math_env.sh`: repo paths, cache paths, quiet logging, Qwen3 defaults.
- `experiments/math/phase_common.sh`: hardware-optimized `fast`, `balanced`, `quality` profiles.
- `experiments/math/run_sdpo_math_benchmark.sh`: one runner for all benchmark phases.

Do not copy Hydra override blocks into notebooks. If a training setting changes, change the shell scripts first.

## Fixed Shape

Models:

| Phase | Model | Profile |
|---|---|---|
| Pilot | `Qwen/Qwen3-1.7B` | `fast` |
| Scale decision | `Qwen/Qwen3-4B` | `fast`, compute-bounded |
| Thesis | `Qwen/Qwen3-8B` | `balanced`, compute-bounded |

Default phase variants:

| Variant | Meaning |
|---|---|
| `base_rl` | GRPO/RL baseline with vanilla policy loss |
| `sdpo_vanilla` | feedback-enabled SDPO baseline |
| `sdpo_reliability_gate` | SDPO improvement: reliability-weighted targets plus sparse teacher forwards |

Profile settings are selected by `HARDWARE_PROFILE`:

| Hardware | Profile | Train batch | Rollout n | Workers | Response | Model len | vLLM util |
|---|---|---:|---:|---:|---:|---:|---:|
| A100-80GB | `fast` | 32 | 2 | 32 | 1024 | 3072 | 0.86 |
| A100-80GB | `balanced` | 32 | 2 | 32 | 1536 | 4096 | 0.80 |
| A100-80GB | `quality` | 32 | 2 | 32 | 2048 | 6144 | 0.76 |
| H100 | `fast` | 32 | 2 | 32 | 1024 | 3072 | 0.92 |
| H100 | `balanced` | 32 | 2 | 32 | 1536 | 4096 | 0.93 |
| H100 | `quality` | 32 | 2 | 32 | 2048 | 6144 | 0.93 |

Common stability defaults: Qwen3 only, Python 3.12, SDPA attention, `use_remove_padding=False`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, validation temperature `0.01`, and `actor_rollout_ref.rollout.enforce_eager=False`. The public profiles use two rollouts per prompt to preserve SDPO sibling comparison while cutting rollout cost. A100 defaults reserve enough vLLM memory for hybrid training to start reliably. If memory is stable and you want to push throughput, rerun with a higher `GPU_UTIL`. If CUDA graph capture fails, rerun the same phase with `ENFORCE_EAGER=True`.

`sdpo_reliability_gate` uses `RELIABILITY_GATE_THRESHOLD=0.4` by default. This keeps successful demonstrations and safe wrong-answer feedback while skipping lower-reliability teacher-forward targets such as format-only feedback and truncated/no-target samples.

When `ULTRA_QUIET=1`, Ray worker logs are hidden but a compact progress watcher remains enabled. It prints heartbeat stages while a step is running, for example `step=12/50 stage=gen_start`, and metric summaries when a step finishes, for example `step=12/50 reward=... tok_s=...`. Set `PROGRESS_WATCH=0` to disable it or `PROGRESS_INTERVAL=30` to print less often. On trainer failure, the runner prints the variant log tail plus recent Ray/vLLM error blocks; set `FAILURE_CONTEXT=0` only if you want to suppress that diagnostic output.

Startup can still be slow before step 1 because each variant initializes Ray workers, FSDP, LoRA, and a vLLM engine for the selected model. The progress watcher reports these as `ray_init_start`, `task_start`, `checkpoint_local_start`, `dataset_start`, `init_workers_start`, and `fit_start`. If a short Phase 2 run spends too much time in vLLM CUDA graph capture, you may test `ENFORCE_EAGER=True`; this can reduce startup time but usually lowers generation throughput, so keep `ENFORCE_EAGER=False` for final thesis runs unless eager mode is empirically faster end to end.

## Setup

Run once per fresh notebook VM.

%%bash
set -euo pipefail

echo "== Setup SDPO-Math notebook =="
cd /root/SDPO
git pull
chmod +x experiments/math/*.sh experiments/math/*.py
unset PYTHON_VERSION
export SDPO_PYTHON_VERSION=3.12
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
bash experiments/math/setup_math_notebook.sh

Useful setup flags:

- `SKIP_INSTALL_IF_READY=1`: skip dependency installation when `.venv` already matches the pinned runtime; enabled by default.
- `FORCE_REINSTALL=1`: force dependency installation even if `.venv` looks ready.
- `PREPARE_DATA=0`: skip data creation.
- `RUN_CPU_CHECK=1`: run CPU/static checks during setup.
- `VERIFY_HF_MODELS=1`: add a lightweight Hugging Face metadata check during setup.
- `INSTALL_MATH_VERIFY=0`: skip `math-verify`; keep it enabled for thesis.

Setup uses `uv venv` and `uv pip install` for all Python package installs. If `uv` is missing, the setup script bootstraps the `uv` binary with the official installer and then continues with `uv`.

## Phase 0: Preflight

Run after every pull before longer GPU runs.

%%bash
set -euo pipefail

echo "== Phase 0: preflight =="
cd /root/SDPO
source experiments/math/math_env.sh
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-False}"

ray stop --force >/dev/null 2>&1 || true
bash experiments/math/run_sdpo_math_live_preflight.sh

python experiments/math/preflight_phase.py

export DRY_RUN=1
export PHASE=pilot
export TRAIN_STEPS=1
export VARIANTS="base_rl sdpo_vanilla sdpo_reliability_gate"
export RUN_TAG=preflight_dryrun
export EXP_SUFFIX=preflight_dryrun_seed42
export LOG_DIR="$PROJECT_ROOT/logs/sdpo_math_phase/preflight_dryrun"

bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_preflight_dryrun.log
python experiments/math/validate_benchmark_dryrun.py \
  --log-dir "$LOG_DIR" \
  --hardware-profile "$HARDWARE_PROFILE" \
  --profile fast \
  --exp-suffix "$EXP_SUFFIX"

## Phase 1: Pilot

Three-variant pilot on Qwen3-1.7B.

%%bash
set -euo pipefail

echo "== Phase 1: pilot =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=pilot
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
export TRAIN_STEPS="${TRAIN_STEPS:-10}"
export ULTRA_QUIET="${ULTRA_QUIET:-0}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 2: Scale Decision

Qwen3-4B fast stability run before thesis scale.

%%bash
set -euo pipefail

echo "== Phase 2: scale decision =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=scale_decision
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
export VARIANTS="${VARIANTS:-base_rl sdpo_vanilla sdpo_reliability_gate}"
export TRAIN_STEPS="${TRAIN_STEPS:-12}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-512}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
export EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export VERIFY_PHASE_MODEL="${VERIFY_PHASE_MODEL:-0}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"
export PROGRESS_WATCH="${PROGRESS_WATCH:-1}"
export PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"

bash experiments/math/run_sdpo_math_benchmark.sh

This Phase 2 command keeps the compact progress tracker visible under `ULTRA_QUIET=1` and evaluates only at the final step. Move to thesis only if all trained variants finish, SDPO logs reprompt and feedback-used metrics, and `sdpo_reliability_gate` logs reliability weights plus a nonzero gate fraction without reward collapse.

## Phase 3: Inspect

Run after any phase.

%%bash
set -euo pipefail

echo "== Phase 3: inspect =="
cd /root/SDPO
source experiments/math/math_env.sh

LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"
python experiments/math/inspect_phase_logs.py --log-dir "$LOG_DIR"
python experiments/math/summarize_phase_results.py --log-dir "$LOG_DIR" || true

## Phase 4: Thesis

Main Qwen3-8B comparison. The default is compute-bounded: 32 steps over a 1024-example training subset, with final-only validation and checkpointing.

%%bash
set -euo pipefail

echo "== Phase 4: thesis =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=thesis
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
export VARIANTS="${VARIANTS:-base_rl sdpo_vanilla sdpo_reliability_gate}"
export TRAIN_STEPS="${TRAIN_STEPS:-32}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1024}"
export EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
export SAVE_FREQ="${SAVE_FREQ:-${TRAIN_STEPS}}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-256}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"
export PROGRESS_WATCH="${PROGRESS_WATCH:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

For a stronger thesis run, place these overrides inside the Phase 4 cell before `bash experiments/math/run_sdpo_math_benchmark.sh`:

- `export TRAIN_STEPS=64`
- `export TRAIN_MAX_SAMPLES=2048`
- `export VAL_MAX_SAMPLES=512`
- `export EVAL_FREQ=64`
- `export SAVE_FREQ=64`

## Phase 5: Report Check

Run after Phase 4 before writing results.

%%bash
set -euo pipefail

echo "== Phase 5: report check =="
cd /root/SDPO
source experiments/math/math_env.sh

if [[ -z "${LOG_DIR:-}" && -f logs/sdpo_math_phase/latest_thesis_log_dir.txt ]]; then
  LOG_DIR="$(< logs/sdpo_math_phase/latest_thesis_log_dir.txt)"
fi
LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"
EXPECT_PROFILE="${EXPECT_PROFILE:-balanced}"
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

## Report Notes

Primary metric: `val-core/math_dapo/acc/mean@1`. Phase 4 defaults to `VAL_BEFORE_TRAIN=False` for trained variants to avoid repeated initial validations. If you need a frozen reference, run `base_model` explicitly in a separate short baseline phase.

Report reward, incorrect format rate, truncation rate, SDPO reprompt fraction, feedback-used fraction, gated-variant reliability weight mean, gate threshold, gate fraction, throughput, seed, profile, model, hardware profile, git commit, validation dumps, and checkpoint paths.

For a thesis or arXiv-level claim, add at least one external held-out math benchmark such as AIME/MathArena and repeat the thesis run with multiple seeds if compute allows.
