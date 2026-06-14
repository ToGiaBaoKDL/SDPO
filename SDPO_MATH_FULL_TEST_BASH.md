# SDPO-Math Full Notebook Test

Use this in a notebook cloned at `/root/SDPO` with 2x A100 GPUs.

Run model order:

1. Smoke/debug model: `Qwen/Qwen3-1.7B`
2. Scale-decision model: `Qwen/Qwen3-4B`
3. Thesis model: `Qwen/Qwen3-8B`

The math config and thesis phase default to `Qwen/Qwen3-8B`.
Use the 1.7B model for smoke tests, the 4B model for scale-decision runs,
and the 8B model for the main thesis path.

Qwen3 requires an installed Transformers/vLLM stack that supports the text-generation model.
The runtime sanity cell uses
`experiments/math/verify_hf_models.py` to check both Hugging Face access and
Transformers `AutoConfig` compatibility, a real Transformers load smoke, and a
small vLLM load smoke before any Ray/FSDP training starts. If that check fails,
upgrade the model stack and re-run the verifier before starting GPU training.

The default path uses `attn_implementation=sdpa` so setup is fast and avoids
local FlashAttention builds.
Do not build FlashAttention for this test run. FlashAttention is an optional speed
optimization, not required for SDPO correctness. This no-FlashAttention profile also
uses `use_remove_padding=False`, because remove-padding imports `flash_attn.bert_padding`.

Copy each section into one notebook code cell. Use `%%bash` exactly, no space.
The cells source `experiments/math/common_quiet_env.sh` for low-noise defaults.
Quiet mode keeps `RAY_DEDUP_LOGS=1`; `RAY_DEDUP_LOGS=0` disables Ray
deduplication and is useful only when you want raw repeated worker logs for debugging.

## 0. Setup

%%bash
set -euo pipefail

echo "== 0. Setup =="
cd /root/SDPO
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
chmod +x experiments/math/*.sh

echo "repo=$PROJECT_ROOT"
git rev-parse --short HEAD || true

## 1. Create Python 3.12 Environment

%%bash
set -euo pipefail

echo "== 1. Python env =="
cd /root/SDPO

python3 -m pip install -q -U uv
unset PYTHON_VERSION
export SDPO_PYTHON_VERSION=3.12
uv venv .venv --python 3.12
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

python --version
which python

## 2. Install Dependencies

%%bash
set -euo pipefail

echo "== 2. Dependencies =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

uv pip install -q -U pip
uv pip install -q pyyaml pyarrow pandas datasets
uv pip install -q -e ".[vllm]"
uv pip install -q -U "transformers==4.57.1"
uv pip install -q -U "numpy==2.1.0"
uv pip install -q "math-verify[antlr4_9_3]==0.8.0"
python - <<'PY'
import importlib.util
import importlib.metadata as metadata
import transformers
required = ["torch", "ray", "transformers", "vllm", "datasets", "pyarrow", "math_verify"]
for name in required:
    assert importlib.util.find_spec(name), f"missing {name}"
print("deps_ok:", ", ".join(required))
print("transformers_version:", transformers.__version__)
print("numpy_version:", metadata.version("numpy"))
print("numba_version:", metadata.version("numba") if importlib.util.find_spec("numba") else "not_installed")
PY

## 2.1 Attention Preflight

%%bash
set -euo pipefail

echo "== 2.1 Attention preflight =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

python - <<'PY'
import yaml

with open("verl/trainer/config/sdpo_math_a100.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

actor_attn = cfg["actor_rollout_ref"]["model"]["override_config"]["attn_implementation"]
critic_attn = cfg["critic"]["model"]["override_config"]["attn_implementation"]
agent_workers = cfg["actor_rollout_ref"]["rollout"]["agent"]["num_workers"]
train_batch_size = cfg["data"]["train_batch_size"]
val_batch_size = cfg["data"]["val_batch_size"]
batched_tokens = cfg["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"]
use_remove_padding = cfg["actor_rollout_ref"]["model"]["use_remove_padding"]
dataloader_workers = cfg["data"]["dataloader_num_workers"]
filter_workers = cfg["data"]["filter_overlong_prompts_workers"]
print("config_attention:", {"actor": actor_attn, "critic": critic_attn})
print("config_batching:", {"train_batch_size": train_batch_size, "val_batch_size": val_batch_size, "agent_workers": agent_workers, "batched_tokens": batched_tokens})
print("config_use_remove_padding:", use_remove_padding)
print("config_data_workers:", {"dataloader": dataloader_workers, "filter": filter_workers})
assert actor_attn == "sdpa", actor_attn
assert critic_attn == "sdpa", critic_attn
assert train_batch_size == 24, train_batch_size
assert val_batch_size == 128, val_batch_size
assert agent_workers == 32, agent_workers
assert batched_tokens == 49152, batched_tokens
assert use_remove_padding is False, use_remove_padding
assert dataloader_workers == 0, dataloader_workers
assert filter_workers == 1, filter_workers
PY

python - <<'PY'
from pathlib import Path

script_paths = [
    Path("experiments/math/run_sdpo_math_smoke.sh"),
    Path("experiments/math/run_sdpo_math_vanilla.sh"),
    Path("experiments/math/run_sdpo_math_reliability.sh"),
]

for path in script_paths:
    text = path.read_text(encoding="utf-8")
    assert 'source "${SCRIPT_DIR}/common_quiet_env.sh"' in text, path
    assert 'ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"' in text, path
    assert 'USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"' in text, path
    assert 'DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"' in text, path
    assert 'FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-1}"' in text, path
    forbidden = ["flash_attention_2", "flash-attn", "flash_attn"]
    hits = [term for term in forbidden if term in text]
    if hits:
        raise SystemExit(f"Unexpected FlashAttention reference in {path}: {hits}")

smoke_text = Path("experiments/math/run_sdpo_math_smoke.sh").read_text(encoding="utf-8")
assert 'AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-2}"' in smoke_text
for path in [
    Path("experiments/math/run_sdpo_math_vanilla.sh"),
    Path("experiments/math/run_sdpo_math_reliability.sh"),
]:
    text = path.read_text(encoding="utf-8")
    assert 'AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-32}"' in text, path

config_text = Path("verl/trainer/config/sdpo_math_a100.yaml").read_text(encoding="utf-8")
for term in ["flash_attention_2", "flash-attn", "flash_attn"]:
    if term in config_text:
        raise SystemExit(f"Unexpected FlashAttention reference in config: {term}")
assert "enforce_eager: True" in config_text

quiet_text = Path("experiments/math/common_quiet_env.sh").read_text(encoding="utf-8")
assert 'VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"' in quiet_text
assert 'RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"' in quiet_text
assert 'VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"' in quiet_text
assert 'TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"' in quiet_text

model_text = Path("verl/workers/config/model.py").read_text(encoding="utf-8")
assert 'self.override_config.get("attn_implementation", "sdpa")' in model_text

worker_text = Path("verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
checks = [
    'override_model_config.get("attn_implementation", "sdpa")',
    'override_config.get("attn_implementation", "sdpa")',
]
for snippet in checks:
    if snippet not in worker_text:
        raise SystemExit(f"Missing SDPA default in fsdp_workers.py: {snippet}")
if worker_text.count('override_config.get("attn_implementation", "sdpa")') < 2:
    raise SystemExit("Expected both critic/reward SDPA defaults in fsdp_workers.py")

for path in [Path("verl/workers/config/model.py"), Path("verl/workers/fsdp_workers.py")]:
    text = path.read_text(encoding="utf-8")
    needle = 'get("attn_implementation", "flash_attention_2")'
    if needle in text:
        raise SystemExit(f"Unexpected FlashAttention default in {path}: {needle}")
PY

echo "attention_preflight_ok"

## 3. Runtime And Model Sanity

%%bash
set -euo pipefail

echo "== 3. Runtime/model sanity =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export SMOKE_MODEL_PATH="${SMOKE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
export SCALE_MODEL_PATH="${SCALE_MODEL_PATH:-Qwen/Qwen3-4B}"
export THESIS_MODEL_PATH="${THESIS_MODEL_PATH:-Qwen/Qwen3-8B}"
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-$SCALE_MODEL_PATH}"

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

python - <<'PY'
import platform
import importlib.util
import importlib.metadata as metadata
import torch
import ray
import vllm
import transformers

print("python:", platform.python_version())
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
print("torch:", torch.__version__)
print("ray:", ray.__version__)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("numpy:", metadata.version("numpy"))
print("numba:", metadata.version("numba") if importlib.util.find_spec("numba") else "not_installed")

assert torch.cuda.is_available(), "CUDA is not visible"
assert torch.cuda.device_count() >= 2, "Expected at least 2 visible GPUs"
PY

python experiments/math/verify_hf_models.py \
  --models "$SMOKE_MODEL_PATH" "$SCALE_MODEL_PATH" "$THESIS_MODEL_PATH"

python experiments/math/verify_hf_models.py \
  --models "$SMOKE_MODEL_PATH" \
  --load-smoke-model "$SMOKE_MODEL_PATH"

ray stop --force >/dev/null 2>&1 || true
python experiments/math/verify_hf_models.py \
  --models "$SMOKE_MODEL_PATH" \
  --vllm-smoke-model "$SMOKE_MODEL_PATH" \
  --vllm-tensor-parallel-size 1 \
  --vllm-max-model-len 1024 \
  --vllm-gpu-memory-utilization 0.70 \
  --vllm-enforce-eager
ray stop --force >/dev/null 2>&1 || true

## 4. Math-Verify Reward Smoke

%%bash
set -euo pipefail

echo "== 4. math-verify reward smoke =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

python - <<'PY'
import importlib.util

spec = importlib.util.spec_from_file_location("math_feedback", "verl/utils/reward_score/feedback/math.py")
math_feedback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_feedback)

cases = [
    ("integer_correct", r"Reasoning... \boxed{34}", "34", {"feedback_mode": "safe"}, 1.0),
    ("integer_wrong", r"Reasoning... \boxed{35}", "34", {"feedback_mode": "safe"}, 0.0),
    ("bad_format", "Reasoning... final answer is 34", "34", {"feedback_mode": "safe"}, 0.0),
    ("symbolic_math_verify", r"Reasoning... \boxed{1+1}", "2", {"feedback_mode": "safe"}, 1.0),
]

for name, pred, gt, extra, expected_score in cases:
    out = math_feedback.compute_score(pred, gt, extra)
    print(name, "score=", out["score"], "math_verify=", out["math_verify_available"], "feedback=", bool(out["feedback"]))
    assert out["math_verify_available"] == 1, out
    assert out["score"] == expected_score, out

print("math_verify_reward_ok")
PY

## 5. Create DAPO-Math Data

%%bash
set -euo pipefail

echo "== 5. Create DAPO-Math data =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

python examples/data_preprocess/dapo_math_processed.py \
  --dataset_name open-r1/DAPO-Math-17k-Processed \
  --subset en \
  --local_save_dir data/dapo_math_en \
  --report_dir reports \
  --validation_size 512 \
  --seed 42 \
  --feedback_mode safe \
  --deduplicate \
  --decontaminate \
  --ngram_jaccard_threshold 0.70

python - <<'PY'
import pyarrow.parquet as pq
train_n = pq.read_table("data/dapo_math_en/train.parquet").num_rows
val_n = pq.read_table("data/dapo_math_en/val.parquet").num_rows
print("data_rows:", {"train": train_n, "val": val_n})
assert train_n > 1000
assert val_n == 512
PY

## 6. CPU/Data Pipeline Check

%%bash
set -euo pipefail

echo "== 6. CPU/data pipeline check =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

PYTHON=.venv/bin/python bash experiments/math/test_cpu_pipeline.sh

python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

train = pq.read_table("data/dapo_math_en/train.parquet").to_pylist()
val = pq.read_table("data/dapo_math_en/val.parquet").to_pylist()
sample = train[0]
prompt = sample["prompt"][0]["content"]

checks = {
    "train_rows": len(train),
    "val_rows": len(val),
    "data_source": sample["data_source"],
    "feedback_mode": sample["extra_info"].get("feedback_mode"),
    "has_boxed_instruction": "\\boxed{}" in prompt,
    "has_answer_colon": "Answer:" in prompt,
    "has_reports": Path("reports/dapo_math_data_report.md").exists() and Path("reports/decontamination_report.md").exists(),
}
print("data_checks:", checks)

assert checks["train_rows"] > 1000
assert checks["val_rows"] == 512
assert checks["data_source"] == "math_dapo"
assert checks["feedback_mode"] == "safe"
assert checks["has_boxed_instruction"]
assert not checks["has_answer_colon"]
assert checks["has_reports"]
PY

## 7. Common Model Exports

%%bash
set -euo pipefail

echo "== 7. Common model exports =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

export SMOKE_MODEL_PATH="${SMOKE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
export SCALE_MODEL_PATH="${SCALE_MODEL_PATH:-Qwen/Qwen3-4B}"
export THESIS_MODEL_PATH="${THESIS_MODEL_PATH:-Qwen/Qwen3-8B}"
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-$SCALE_MODEL_PATH}"

echo "smoke_model=$SMOKE_MODEL_PATH"
echo "scale_model=$SCALE_MODEL_PATH"
echo "thesis_model=$THESIS_MODEL_PATH"
echo "target_model_for_tests=$TARGET_MODEL_PATH"

## 8. Base Model Validation Smoke

%%bash
set -euo pipefail

echo "== 8. Base model validation smoke =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH="${SMOKE_MODEL_PATH:-Qwen/Qwen3-1.7B}"

echo "model=$MODEL_PATH"
ray stop --force >/dev/null 2>&1 || true

python3 -m verl.trainer.main_ppo \
  --config-name sdpo_math_a100 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=False \
  critic.model.override_config.attn_implementation=sdpa \
  trainer.experiment_name=base_model_val_smoke \
  trainer.group_name=SDPO-Math-Base-Val \
  trainer.logger='["console"]' \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.save_freq=-1 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=1 \
  data.train_max_samples=8 \
  data.val_max_samples=8 \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.max_response_length=1024 \
  rollout_model_len=3072 \
  actor_max_token_len=3072 \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  actor_rollout_ref.actor.self_distillation.reliability_weighting=False

## 9. Base RL Short Train

Keep both attention overrides as `sdpa`. Do not replace them with
`flash_attention_2` unless `flash-attn` is installed and import-tested.

%%bash
set -euo pipefail

echo "== 9. Base RL short train =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-4B}"

echo "model=$MODEL_PATH"
ray stop --force >/dev/null 2>&1 || true

python3 -m verl.trainer.main_ppo \
  --config-name sdpo_math_a100 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=False \
  critic.model.override_config.attn_implementation=sdpa \
  trainer.experiment_name=base_rl_5step \
  trainer.group_name=SDPO-Math-Base-RL \
  trainer.logger='["console"]' \
  trainer.total_training_steps=5 \
  trainer.val_before_train=False \
  trainer.test_freq=5 \
  trainer.save_freq=-1 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=1 \
  data.train_max_samples=128 \
  data.val_max_samples=64 \
  data.train_batch_size=2 \
  data.max_response_length=1024 \
  rollout_model_len=3072 \
  actor_max_token_len=3072 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  actor_rollout_ref.actor.self_distillation.reliability_weighting=False

## 10. SDPO Smoke Tests

%%bash
set -euo pipefail

echo "== 10. SDPO smoke tests =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH="${SMOKE_MODEL_PATH:-Qwen/Qwen3-1.7B}"

echo "model=$MODEL_PATH"

for variant in vanilla reliability; do
  echo "-- smoke_variant=$variant"
  ray stop --force >/dev/null 2>&1 || true
  bash experiments/math/run_sdpo_math_smoke.sh "$variant" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    critic.model.use_remove_padding=False \
    critic.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts_workers=1
done

## 11. Scale-Model SDPO Smoke Tests

%%bash
set -euo pipefail

echo "== 11. Scale-model SDPO smoke tests =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-4B}"

echo "model=$MODEL_PATH"

for variant in vanilla reliability; do
  echo "-- scale_smoke_variant=$variant"
  ray stop --force >/dev/null 2>&1 || true
  bash experiments/math/run_sdpo_math_smoke.sh "$variant" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    critic.model.use_remove_padding=False \
    critic.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts_workers=1
done

## 12. SDPO And SDPO+ Short Train

%%bash
set -euo pipefail

echo "== 12. SDPO/SDPO+ short train =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-4B}"
export TRAIN_MAX_SAMPLES=128
export VAL_MAX_SAMPLES=64
export TOTAL_TRAINING_STEPS=5
export LOGGER='["console"]'

echo "model=$MODEL_PATH"

for script in \
  experiments/math/run_sdpo_math_vanilla.sh \
  experiments/math/run_sdpo_math_reliability.sh
do
  echo "-- train_script=$script"
  ray stop --force >/dev/null 2>&1 || true
  bash "$script" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    critic.model.use_remove_padding=False \
    critic.model.override_config.attn_implementation=sdpa \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts_workers=1 \
    data.train_batch_size=2 \
    data.max_response_length=1024 \
    rollout_model_len=3072 \
    actor_max_token_len=3072 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048
done

## 13. Scale-Model Full Training Debug

Only run this after sections 0-12 pass. For thesis benchmarking, use `SDPO_MATH_PHASE_RUNBOOK.md` Phase 4 instead.

%%bash
set -euo pipefail

echo "== 13. Scale-model full training debug =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-4B}"
export TRAIN_MAX_SAMPLES=-1
export VAL_MAX_SAMPLES=-1
export TOTAL_TRAINING_STEPS=null
export LOGGER='["console"]'

echo "model=$MODEL_PATH"

echo "-- full_train=base_rl"
ray stop --force >/dev/null 2>&1 || true
python3 -m verl.trainer.main_ppo \
  --config-name sdpo_math_a100 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=False \
  critic.model.override_config.attn_implementation=sdpa \
  trainer.experiment_name=base_rl_full \
  trainer.group_name=SDPO-Math-Base-RL \
  trainer.logger="$LOGGER" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=1 \
  data.train_batch_size=2 \
  data.max_response_length=1024 \
  rollout_model_len=3072 \
  actor_max_token_len=3072 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  actor_rollout_ref.actor.self_distillation.reliability_weighting=False

for script in \
  experiments/math/run_sdpo_math_vanilla.sh \
  experiments/math/run_sdpo_math_reliability.sh
do
  echo "-- full_train_script=$script"
  ray stop --force >/dev/null 2>&1 || true
  bash "$script" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    critic.model.use_remove_padding=False \
    critic.model.override_config.attn_implementation=sdpa \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts_workers=1 \
    data.train_batch_size=2 \
    data.max_response_length=1024 \
    rollout_model_len=3072 \
    actor_max_token_len=3072 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048
done

## 14. Pass Criteria

Before full training, confirm:

- `math_verify=1` in the math reward smoke.
- CPU pipeline prints `CPU pipeline checks passed`.
- Base-model validation prints initial validation metrics.
- Base RL 5-step run logs reward/actor metrics.
- SDPO vanilla logs `self_distillation/*` metrics.
- SDPO+ reliability logs `self_distillation/reliability_*` metrics.
- No run ends with CUDA OOM, Ray worker death, or NaN loss.

## 15. OOM Override Cell

%%bash
set -euo pipefail

echo "== 15. OOM override example =="
cd /root/SDPO
source .venv/bin/activate
source experiments/math/common_quiet_env.sh

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-4B}"

ray stop --force >/dev/null 2>&1 || true
bash experiments/math/run_sdpo_math_smoke.sh reliability \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  critic.model.use_remove_padding=False \
  critic.model.override_config.attn_implementation=sdpa \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=1 \
  data.max_response_length=768 \
  rollout_model_len=2560 \
  actor_max_token_len=2560 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=1792 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
