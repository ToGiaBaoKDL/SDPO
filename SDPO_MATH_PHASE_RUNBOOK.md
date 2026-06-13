# SDPO-Math Phase Runbook

Notebook-ready commands for setting up and running the SDPO-Math benchmark phases.
Use each `%%bash` block as one notebook cell.

The phase runner is centralized in `experiments/math/run_sdpo_math_benchmark.sh`.
Profile settings and shared Hydra overrides live in `experiments/math/phase_common.sh`.
Do not copy training logic into new notebook cells unless you also move it back into these scripts.

Validation in these phases uses `data/dapo_math_en/val.parquet`, the held-out DAPO-Math English split. It is not AIME.

## Required Setup

Use a fresh clone at `/root/SDPO` and pull the latest code before running phases. The setup script creates a Python 3.10 uv venv, installs the project with vLLM, installs `math-verify`, verifies the Hugging Face model ids, prepares the English DAPO-Math split if missing, and runs CPU/data checks.

Run once per notebook VM:

%%bash
set -euo pipefail

echo "== Setup SDPO-Math notebook =="
cd /root/SDPO
git pull
chmod +x experiments/math/*.sh experiments/math/*.py
bash experiments/math/setup_math_notebook.sh

Useful setup flags: `PREPARE_DATA=0` skips data creation, `RUN_CPU_CHECK=0` skips CPU checks, `VERIFY_HF_MODELS=0` skips Hugging Face model-id checks, and `INSTALL_MATH_VERIFY=0` skips `math-verify` installation. For thesis runs, keep `INSTALL_MATH_VERIFY=1`.

## Shared Cell Prefix

After setup, every phase cell uses `cd /root/SDPO` then `source experiments/math/math_env.sh`.

That helper activates `.venv` if it exists, sets `PROJECT_ROOT`, `PYTHONPATH`, `HF_HOME`, `UV_CACHE_DIR`, quiet logging defaults, `WANDB_MODE=offline`, `CUDA_VISIBLE_DEVICES=0,1`, and default model names.

Model ladder:

- Phase 1 pilot: `Qwen/Qwen2.5-0.5B-Instruct`.
- Phase 2 scale decision: `Qwen/Qwen3.5-4B`.
- Phase 4 thesis: `Qwen/Qwen3.5-9B`.

The public Qwen3.5 model ids verified for this runbook are `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`, not `*-Instruct`.

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

- `fast`: pilot profile for quick correctness checks. `train_batch_size=4`, `rollout.n=2`, `agent_workers=8`, response length `1024`.
- `balanced`: scale-decision profile for Qwen3.5-4B. `train_batch_size=8`, `rollout.n=2`, `agent_workers=8`, response length `1536`.
- `quality`: thesis profile for Qwen3.5-9B. `train_batch_size=8`, `rollout.n=4`, `agent_workers=8`, response length `1536`.
- `high_mem_9b`: optional high-memory Qwen3.5-9B profile. `a100_7b` remains as a legacy alias.

Recommended order:

1. Phase 0: preflight.
2. Phase 1: pilot, confirms all variants run.
3. Phase 2: scale-decision benchmark.
4. Phase 3: inspect logs.
5. Phase 4: thesis 9B run.
6. Phase 5: optional compatibility run only after Phase 2/4 is stable.
7. Phase 6: report-readiness check.

Use `ULTRA_QUIET=1` for long runs. It hides Ray worker stdout and writes metrics JSONL under the run log directory. Keep `ULTRA_QUIET=1` for Phase 4 so `summary.csv` and report-readiness checks can be generated from file logs.

## Phase 0. Preflight

Run after every pull/update before longer experiments.

%%bash
set -euo pipefail

echo "== Phase 0: preflight =="
cd /root/SDPO
source experiments/math/math_env.sh

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

Purpose: final Qwen3.5-9B thesis comparison.

Default: `PHASE=thesis`, `RUN_PROFILE=quality`, `TRAIN_STEPS=300`, validation every `100` steps.
This runs the frozen `base_model` evaluation plus the 3 train variants.

%%bash
set -euo pipefail

echo "== Phase 4: main thesis runs =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE="${PHASE:-thesis}"
export RUN_PROFILE="${RUN_PROFILE:-quality}"
export TRAIN_STEPS="${TRAIN_STEPS:-300}"
export EVAL_FREQ="${EVAL_FREQ:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 5. Optional Compatibility Run

Use only after Phase 2/4 is stable and you want the legacy high-memory profile.

%%bash
set -euo pipefail

echo "== Phase 5: optional compatibility run =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=scale_9b
export RUN_PROFILE=high_mem_9b
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
- All 4 variants are present: `base_model`, `base_rl`, `sdpo_vanilla`, `sdpo_reliability`.
- Main result column is `val-core/math_dapo/acc/mean@1` from the held-out DAPO-Math English validation split.
- Diagnostic columns include format error, truncation, SDPO reprompt fraction, SDPO feedback-used fraction, and reliability weight mean.
- The final thesis comparison uses one shared model, data split, validation size, generation setting, seed, and profile across all variants.

For arXiv-quality claims, DAPO-Math validation alone is not enough. Add at least one external held-out math benchmark and, if compute allows, repeat the Phase 4 comparison with multiple seeds such as `SEED=42`, `SEED=43`, and `SEED=44`.

%%bash
set -euo pipefail
cd /root/SDPO
source experiments/math/math_env.sh

LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"
python experiments/math/summarize_phase_results.py --log-dir "$LOG_DIR"
python experiments/math/check_phase_report_ready.py --log-dir "$LOG_DIR" --require-checkpoints
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
