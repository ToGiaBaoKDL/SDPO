# SDPO-Math Phase Runbook

Notebook-ready commands after `SDPO_MATH_FULL_TEST_BASH.md` Tasks 1-12 pass.
Use each `%%bash` block as one notebook cell.

The phase runner is centralized in `experiments/math/run_sdpo_math_benchmark.sh`.
Profile settings and shared Hydra overrides live in `experiments/math/phase_common.sh`.
Do not copy training logic into new notebook cells unless you also move it back into these scripts.

Validation in these phases uses `data/dapo_math_en/val.parquet`, the held-out DAPO-Math English split. It is not AIME.

## Benchmark Variants

There are 5 benchmark variants:

- `base_model`: frozen base-model validation only. No training, no LoRA.
- `base_rl`: trained GRPO/RL baseline with vanilla policy loss.
- `sdpo_vanilla`: SDPO with successful sibling rollouts as teacher demonstrations.
- `sdpo_safe_feedback`: SDPO plus safe math feedback when no successful demonstration exists.
- `sdpo_reliability`: SDPO plus safe feedback plus reliability weighting.

Training phases run 4 trained variants: `base_rl`, `sdpo_vanilla`, `sdpo_safe_feedback`, and `sdpo_reliability`.
The fifth variant, `base_model`, is an evaluation baseline.

`sdpo_vanilla` and `sdpo_safe_feedback` are not identical. They can behave the same for samples where SDPO already found a successful sibling rollout, because feedback is configured to be used only when no successful solution demonstration exists. Check `self_distillation/feedback_used_fraction`: if it is `0`, safe-feedback had no practical effect in that batch.

## Profiles

- `fast`: pilot profile for quick correctness checks. `train_batch_size=4`, `rollout.n=2`, `agent_workers=8`, response length `1024`.
- `balanced`: scale-decision profile. `train_batch_size=8`, `rollout.n=2`, `agent_workers=8`, response length `1536`.
- `quality`: main 1.5B thesis profile. `train_batch_size=8`, `rollout.n=4`, `agent_workers=8`, response length `1536`.
- `a100_7b`: optional 7B profile for 2x A100. `train_batch_size=8`, `rollout.n=4`, `agent_workers=16`, response length `2048`.

Recommended order:

1. Phase 0: preflight.
2. Phase 1: pilot, confirms all variants run.
3. Phase 2: ablation, decides whether to scale.
4. Phase 3: inspect logs.
5. Phase 4: thesis 1.5B run.
6. Phase 5: optional 7B run only after Phase 2/4 is stable.

Use `ULTRA_QUIET=1` for long runs. It hides Ray worker stdout and writes metrics JSONL under the run log directory.

## Phase 0. Preflight

Run after every pull/update before longer experiments.

%%bash
set -euo pipefail

echo "== Phase 0: preflight =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/phase_common.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
import yaml

with open("verl/trainer/config/sdpo_math_l40s.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

checks = {
    "train_files": cfg["data"]["train_files"],
    "val_files": cfg["data"]["val_files"],
    "actor_attn": cfg["actor_rollout_ref"]["model"]["override_config"]["attn_implementation"],
    "critic_attn": cfg["critic"]["model"]["override_config"]["attn_implementation"],
    "use_remove_padding": cfg["actor_rollout_ref"]["model"]["use_remove_padding"],
    "dataloader_workers": cfg["data"]["dataloader_num_workers"],
    "filter_workers": cfg["data"]["filter_overlong_prompts_workers"],
}
print("config:", checks)
assert "data/dapo_math_en/train.parquet" in checks["train_files"][0]
assert "data/dapo_math_en/val.parquet" in checks["val_files"][0]
assert checks["actor_attn"] == "sdpa"
assert checks["critic_attn"] == "sdpa"
assert checks["use_remove_padding"] is False
assert checks["dataloader_workers"] == 0
assert checks["filter_workers"] == 1

train_rows = pq.read_table("data/dapo_math_en/train.parquet").num_rows
val_rows = pq.read_table("data/dapo_math_en/val.parquet").num_rows
print("data_rows:", {"train": train_rows, "val": val_rows})
assert train_rows > 1000
assert val_rows >= 128

phase_common = Path("experiments/math/phase_common.sh").read_text(encoding="utf-8")
for snippet in [
    "fast)",
    "balanced)",
    "quality)",
    "a100_7b)",
    "AGENT_WORKERS=8",
    "actor_rollout_ref.rollout.val_kwargs.n=1",
]:
    assert snippet in phase_common, snippet

runner = Path("experiments/math/run_sdpo_math_benchmark.sh").read_text(encoding="utf-8")
for snippet in [
    "base_model base_rl sdpo_vanilla sdpo_safe_feedback sdpo_reliability",
    "PHASE=ablation",
    "PHASE=thesis",
    "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
]:
    assert snippet in runner, snippet

print("phase0_ok")
PY

## Phase 1. Pilot Test

Purpose: confirm the full 5-variant benchmark shape runs cleanly.

Default: `PHASE=pilot`, `RUN_PROFILE=fast`, `TRAIN_STEPS=10`.
For a very quick test, set `VARIANTS="base_model base_rl sdpo_reliability" TRAIN_STEPS=3`.

%%bash
set -euo pipefail

echo "== Phase 1: pilot test =="
cd /root/SDPO
source .venv/bin/activate

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PHASE="${PHASE:-pilot}"
export RUN_PROFILE="${RUN_PROFILE:-fast}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export TRAIN_STEPS="${TRAIN_STEPS:-10}"
export ULTRA_QUIET="${ULTRA_QUIET:-0}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 2. Scale-Decision Ablation

Purpose: useful comparison before spending on the final thesis run.

Default: `PHASE=ablation`, `RUN_PROFILE=balanced`, `TRAIN_STEPS=50`.
Scale up if:

- all 4 train variants finish without OOM or chunk errors;
- `base_rl` is stable;
- SDPO variants log nonzero `self_distillation/reprompt_sample_fraction`;
- `sdpo_safe_feedback` logs nonzero `self_distillation/feedback_used_fraction` at least sometimes;
- `sdpo_reliability` logs reliability metrics and does not collapse reward versus SDPO vanilla.

%%bash
set -euo pipefail

echo "== Phase 2: scale-decision ablation =="
cd /root/SDPO
source .venv/bin/activate

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PHASE="${PHASE:-ablation}"
export RUN_PROFILE="${RUN_PROFILE:-balanced}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export TRAIN_STEPS="${TRAIN_STEPS:-50}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 3. Inspect Logs

Run after Phase 1, 2, 4, or 5. Set `LOG_DIR` to a specific directory if needed.

%%bash
set -euo pipefail

echo "== Phase 3: inspect logs =="
cd /root/SDPO

LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"
export LOG_DIR
echo "log_dir=$LOG_DIR"

for log in "$LOG_DIR"/*.log; do
  echo
  echo "## $(basename "$log")"
  grep "step:" "$log" | tail -3 | sed 's/ - /\n  /g' | \
    grep -E "step:|training/global_step|val-core|val-aux|reward|score|acc|format|truncated|self_distillation|actor/grad_norm|actor/pg_loss|perf/throughput" || true
done

python - <<'PY'
import glob
import json
import os
from pathlib import Path

log_dir = Path(os.environ["LOG_DIR"])
metric_files = sorted(glob.glob(str(log_dir / "metrics" / "SDPO-Math" / "*.jsonl")))
if metric_files:
    print("\nfile_logger_metrics:")

for path in metric_files:
    print(f"\n## {Path(path).name}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows[-3:]:
        data = row.get("data", {})
        print(f"step:{row.get('step')}")
        for key, value in sorted(data.items()):
            if any(token in key for token in [
                "training/global_step",
                "val-core",
                "val-aux",
                "reward",
                "score",
                "acc",
                "format",
                "truncated",
                "self_distillation",
                "actor/grad_norm",
                "actor/pg_loss",
                "perf/throughput",
            ]):
                print(f"  {key}:{value}")
PY

echo
echo "checkpoints:"
find checkpoints/sdpo_math -maxdepth 3 -type d -name 'global_step_*' 2>/dev/null | sort | tail -20 || true

## Phase 4. Main Thesis Runs

Purpose: final 1.5B thesis comparison on 2x L4/A10/L40S.

Default: `PHASE=thesis`, `RUN_PROFILE=quality`, `TRAIN_STEPS=300`, validation every `100` steps.
This runs the frozen `base_model` evaluation plus the 4 train variants.

%%bash
set -euo pipefail

echo "== Phase 4: main thesis runs =="
cd /root/SDPO
source .venv/bin/activate

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PHASE="${PHASE:-thesis}"
export RUN_PROFILE="${RUN_PROFILE:-quality}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export TRAIN_STEPS="${TRAIN_STEPS:-300}"
export EVAL_FREQ="${EVAL_FREQ:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Phase 5. Optional 2x A100 7B

Use only after Phase 2/4 is stable and you have 2x A100.

%%bash
set -euo pipefail

echo "== Phase 5: optional 2x A100 7B =="
cd /root/SDPO
source .venv/bin/activate

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PHASE=scale_7b
export RUN_PROFILE=a100_7b
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export TRAIN_STEPS="${TRAIN_STEPS:-300}"
export EVAL_FREQ="${EVAL_FREQ:-100}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-512}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Useful Overrides

Run only training variants:

%%bash
set -euo pipefail
cd /root/SDPO
source .venv/bin/activate
export PHASE=ablation
export VARIANTS="base_rl sdpo_vanilla sdpo_safe_feedback sdpo_reliability"
bash experiments/math/run_sdpo_math_benchmark.sh

Run only the strongest SDPO+ variant:

%%bash
set -euo pipefail
cd /root/SDPO
source .venv/bin/activate
export PHASE=thesis
export VARIANTS="sdpo_reliability"
bash experiments/math/run_sdpo_math_benchmark.sh
