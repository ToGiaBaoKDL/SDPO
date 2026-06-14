# SDPO-Math Phase Runbook

Notebook-ready commands for SDPO-Math on 2 GPUs. Default hardware is 2x A100-80GB. For 2x H100, set `HARDWARE_PROFILE=h100`.

Central files:

- `experiments/math/setup_math_notebook.sh`: Python 3.12 uv environment, dependencies, data, CPU checks.
- `experiments/math/math_env.sh`: repo paths, cache paths, quiet logging, Qwen3 defaults.
- `experiments/math/phase_common.sh`: hardware-optimized `fast`, `balanced`, `quality` profiles.
- `experiments/math/run_sdpo_math_benchmark.sh`: one runner for all benchmark phases.

Do not copy Hydra override blocks into notebooks. If a training setting changes, change the shell scripts first.

## Fixed Shape

Models:

| Phase | Model | Profile |
|---|---|---|
| Pilot | `Qwen/Qwen3-1.7B` | `fast` |
| Scale decision | `Qwen/Qwen3-4B` | `balanced` |
| Thesis | `Qwen/Qwen3-8B` | `quality` |

Variants:

| Variant | Meaning |
|---|---|
| `base_model` | frozen validation baseline, no training, no LoRA |
| `base_rl` | GRPO/RL baseline with vanilla policy loss |
| `sdpo_vanilla` | feedback-enabled SDPO baseline |
| `sdpo_reliability` | SDPO+ with reliability-weighted SDPO targets |

Profile settings are selected by `HARDWARE_PROFILE`:

| Hardware | Profile | Train batch | Rollout n | Workers | Response | Model len | vLLM util |
|---|---|---:|---:|---:|---:|---:|---:|
| A100-80GB | `fast` | 64 | 4 | 128 | 1536 | 5120 | 0.86 |
| A100-80GB | `balanced` | 64 | 4 | 128 | 2048 | 6144 | 0.86 |
| A100-80GB | `quality` | 48 | 4 | 96 | 3072 | 8192 | 0.86 |
| H100 | `fast` | 64 | 4 | 128 | 1536 | 5120 | 0.92 |
| H100 | `balanced` | 64 | 4 | 128 | 2048 | 6144 | 0.93 |
| H100 | `quality` | 48 | 4 | 96 | 3072 | 8192 | 0.93 |

Common stability defaults: Qwen3 only, Python 3.12, SDPA attention, `use_remove_padding=False`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, validation temperature `0.01`, and `actor_rollout_ref.rollout.enforce_eager=False`. A100 defaults target reliable 80GB startup; if memory is stable and you want to push throughput, rerun with `GPU_UTIL=0.90`. If CUDA graph capture fails, rerun the same phase with `ENFORCE_EAGER=True`.

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

- `PREPARE_DATA=0`: skip data creation.
- `RUN_CPU_CHECK=1`: run CPU/static checks during setup.
- `VERIFY_HF_MODELS=1`: add a lightweight Hugging Face metadata check during setup.
- `INSTALL_MATH_VERIFY=0`: skip `math-verify`; keep it enabled for thesis.

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
export VARIANTS="base_model base_rl sdpo_vanilla sdpo_reliability"
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

Full benchmark shape on Qwen3-1.7B.

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

Qwen3-4B stability run before thesis scale.

%%bash
set -euo pipefail

echo "== Phase 2: scale decision =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=scale_decision
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
export TRAIN_STEPS="${TRAIN_STEPS:-50}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

Move to thesis only if all variants finish, SDPO logs reprompt and feedback-used metrics, and `sdpo_reliability` logs reliability weights without reward collapse.

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

Main Qwen3-8B comparison.

%%bash
set -euo pipefail

echo "== Phase 4: thesis =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=thesis
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
export TRAIN_STEPS="${TRAIN_STEPS:-300}"
export EVAL_FREQ="${EVAL_FREQ:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

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

## Report Notes

Primary metric: `val-core/math_dapo/acc/mean@1`.

Report reward, incorrect format rate, truncation rate, SDPO reprompt fraction, feedback-used fraction, reliability weight mean, throughput, seed, profile, model, hardware profile, git commit, validation dumps, and checkpoint paths.

For a thesis or arXiv-level claim, add at least one external held-out math benchmark such as AIME/MathArena and repeat the thesis run with multiple seeds if compute allows.
