# SDPO-Math Phase Runbook

Notebook-ready bash for SDPO-Math on 2x A100. Use each `%%bash` block as one notebook cell.

Central files:

- `experiments/math/setup_math_notebook.sh`: Python 3.12 uv environment, dependencies, data, CPU checks.
- `experiments/math/math_env.sh`: repo paths, cache paths, quiet logging, Qwen3 model defaults.
- `experiments/math/phase_common.sh`: shared A100 profiles and Hydra overrides.
- `experiments/math/run_sdpo_math_benchmark.sh`: one runner for all benchmark phases.

Do not copy training logic into notebook cells. If a training setting changes, change it in `phase_common.sh` or `run_sdpo_math_benchmark.sh`.

Validation in this runbook uses `data/dapo_math_en/val.parquet`, the held-out English DAPO-Math split. AIME/MathArena are external benchmarks and are not part of these phase commands yet.

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
bash experiments/math/setup_math_notebook.sh

Useful setup flags:

- `PREPARE_DATA=0`: skip data creation.
- `RUN_CPU_CHECK=0`: skip CPU checks.
- `VERIFY_HF_MODELS=0`: skip Hugging Face model checks.
- `RUN_TRANSFORMERS_LOAD_SMOKE=1`: add a setup-time Transformers load smoke.
- `INSTALL_MATH_VERIFY=0`: skip `math-verify`; keep it enabled for thesis runs.

The setup script does not run standalone `vllm.LLM(...)`. Phase 0 validates the real VERL/vLLM trainer path.

## Fixed Experiment Shape

Models:

- Pilot: `Qwen/Qwen3-1.7B`.
- Scale decision: `Qwen/Qwen3-4B`.
- Thesis: `Qwen/Qwen3-8B`.

Variants:

- `base_model`: frozen model validation, no training, no LoRA.
- `base_rl`: GRPO/RL baseline with vanilla policy loss.
- `sdpo_vanilla`: feedback-enabled SDPO baseline.
- `sdpo_reliability`: SDPO+ with reliability-weighted SDPO targets.

Training phases evaluate all 4 variants and train only the 3 trainable variants. Thesis tables should compare all 4 under the same model, seed, validation split, decoding, and profile.

Profiles:

- `fast`: Qwen3-1.7B pilot, train batch `32`, rollout `n=4`, workers `32`, response `1024`.
- `balanced`: Qwen3-4B scale decision, train batch `32`, rollout `n=4`, workers `32`, response `1536`.
- `quality`: Qwen3-8B thesis default, train batch `24`, rollout `n=4`, workers `32`, response `2048`.
- `high_mem_8b`: optional faster Qwen3-8B profile, train batch `32`, batched tokens `65536`.

The A100 profile uses `attn_implementation=sdpa`, `use_remove_padding=False`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, validation temperature `0.01`, and `actor_rollout_ref.rollout.enforce_eager=True`. Keep this default until Phase 4 is stable.

## Phase 0: Preflight

Run after every pull before longer GPU runs.

%%bash
set -euo pipefail

echo "== Phase 0: preflight =="
cd /root/SDPO
source experiments/math/math_env.sh

python experiments/math/verify_hf_models.py \
  --models "$PILOT_MODEL_PATH" "$SCALE_MODEL_PATH" "$THESIS_MODEL_PATH"

python experiments/math/verify_hf_models.py \
  --models "$PILOT_MODEL_PATH" \
  --load-smoke-model "$PILOT_MODEL_PATH"

ray stop --force >/dev/null 2>&1 || true
bash experiments/math/run_sdpo_math_live_preflight.sh

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
python experiments/math/validate_benchmark_dryrun.py \
  --log-dir "$LOG_DIR" \
  --exp-suffix "$EXP_SUFFIX"

## Phase 1: Pilot

Confirms the full benchmark shape on Qwen3-1.7B.

%%bash
set -euo pipefail

echo "== Phase 1: pilot =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=pilot
export RUN_PROFILE="${RUN_PROFILE:-fast}"
export TRAIN_STEPS="${TRAIN_STEPS:-10}"
export ULTRA_QUIET="${ULTRA_QUIET:-0}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 2: Scale Decision

Uses Qwen3-4B to decide whether the 8B thesis profile is stable enough to run.

%%bash
set -euo pipefail

echo "== Phase 2: scale decision =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=scale_decision
export RUN_PROFILE="${RUN_PROFILE:-balanced}"
export TRAIN_STEPS="${TRAIN_STEPS:-50}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

Scale up only if all variants finish, SDPO logs `self_distillation/reprompt_sample_fraction`, `sdpo_vanilla` sometimes uses feedback, and `sdpo_reliability` logs reliability metrics without reward collapse.

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

Main Qwen3-8B comparison. This is the run to report.

%%bash
set -euo pipefail

echo "== Phase 4: thesis =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=thesis
export RUN_PROFILE="${RUN_PROFILE:-quality}"
export TRAIN_STEPS="${TRAIN_STEPS:-300}"
export EVAL_FREQ="${EVAL_FREQ:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 5: Optional Throughput

Run only after Phase 4 is stable. Same Qwen3-8B benchmark, larger profile.

%%bash
set -euo pipefail

echo "== Phase 5: high-mem 8B =="
cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=scale_8b
export RUN_PROFILE=high_mem_8b
export TRAIN_STEPS="${TRAIN_STEPS:-300}"
export EVAL_FREQ="${EVAL_FREQ:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 6: Report Check

Run after Phase 4 and before writing results.

%%bash
set -euo pipefail

echo "== Phase 6: report check =="
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

Required metrics:

- Main accuracy: `val-core/math_dapo/acc/mean@1`.
- Reward: `val-aux/math_dapo/reward/mean@1`.
- Format diagnostics: incorrect format and truncation.
- SDPO diagnostics: reprompt fraction, feedback-used fraction, reliability weight mean.
- System diagnostics: throughput, seed, profile, model, git commit from `manifest.json`.

For a thesis or arXiv claim, add at least one external held-out benchmark such as AIME/MathArena and repeat the thesis run with multiple seeds if compute allows.
