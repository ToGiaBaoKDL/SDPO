#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SDPO_PYTHON_VERSION="${SDPO_PYTHON_VERSION:-3.12}"
ALLOW_UNTESTED_PYTHON="${ALLOW_UNTESTED_PYTHON:-0}"
INSTALL_MATH_VERIFY="${INSTALL_MATH_VERIFY:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
RUN_CPU_CHECK="${RUN_CPU_CHECK:-1}"
VERIFY_HF_MODELS="${VERIFY_HF_MODELS:-1}"
INSTALL_QWEN35_TRANSFORMERS="${INSTALL_QWEN35_TRANSFORMERS:-1}"
QWEN35_TRANSFORMERS_SPEC="${QWEN35_TRANSFORMERS_SPEC:-git+https://github.com/huggingface/transformers.git}"
RUN_QWEN35_TRANSFORMERS_LOAD_SMOKE="${RUN_QWEN35_TRANSFORMERS_LOAD_SMOKE:-1}"
RUN_QWEN35_VLLM_LOAD_SMOKE="${RUN_QWEN35_VLLM_LOAD_SMOKE:-1}"
QWEN35_VLLM_SMOKE_TP="${QWEN35_VLLM_SMOKE_TP:-1}"
QWEN35_VLLM_SMOKE_MAX_MODEL_LEN="${QWEN35_VLLM_SMOKE_MAX_MODEL_LEN:-1024}"
QWEN35_VLLM_SMOKE_GPU_UTIL="${QWEN35_VLLM_SMOKE_GPU_UTIL:-0.25}"

if [[ "${SDPO_PYTHON_VERSION}" != 3.12* && "${ALLOW_UNTESTED_PYTHON}" != "1" ]]; then
  cat >&2 <<EOF
Unsupported SDPO_PYTHON_VERSION=${SDPO_PYTHON_VERSION}.
Use SDPO_PYTHON_VERSION=3.12 for the SDPO-Math notebook environment, or set
ALLOW_UNTESTED_PYTHON=1 if you intentionally want to test another Python.
EOF
  exit 1
fi

export SDPO_SKIP_VENV=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/math_env.sh"

echo "repo=${PROJECT_ROOT}"
echo "sdpo_python_version=${SDPO_PYTHON_VERSION}"
echo "install_math_verify=${INSTALL_MATH_VERIFY}"
echo "prepare_data=${PREPARE_DATA}"
echo "verify_hf_models=${VERIFY_HF_MODELS}"
echo "install_qwen35_transformers=${INSTALL_QWEN35_TRANSFORMERS}"
echo "run_qwen35_transformers_load_smoke=${RUN_QWEN35_TRANSFORMERS_LOAD_SMOKE}"
echo "run_qwen35_vllm_load_smoke=${RUN_QWEN35_VLLM_LOAD_SMOKE}"

if [[ -x .venv/bin/python ]]; then
  EXISTING_PYTHON_VERSION="$(
    .venv/bin/python - <<'PY'
import platform
print(platform.python_version())
PY
  )"
  if [[ "${EXISTING_PYTHON_VERSION}" != 3.12* && "${ALLOW_UNTESTED_PYTHON}" != "1" ]]; then
    cat >&2 <<EOF
Existing .venv uses Python ${EXISTING_PYTHON_VERSION}.
Remove it and re-run setup:
  rm -rf .venv
  export SDPO_PYTHON_VERSION=3.12
  bash experiments/math/setup_math_notebook.sh
EOF
    exit 1
  fi
fi

python3 -m pip install -q -U uv
uv venv .venv --python "${SDPO_PYTHON_VERSION}"

unset SDPO_SKIP_VENV
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/math_env.sh"

python --version
uv pip install -q -U pip
uv pip install -q pyyaml pyarrow pandas datasets
uv pip install -q -e ".[vllm]"

if [[ "${INSTALL_QWEN35_TRANSFORMERS}" == "1" ]]; then
  echo "Installing Qwen3.5-compatible Transformers from ${QWEN35_TRANSFORMERS_SPEC}"
  uv pip install -q -U "${QWEN35_TRANSFORMERS_SPEC}"
fi

if [[ "${INSTALL_MATH_VERIFY}" == "1" ]]; then
  uv pip install -q "math-verify[antlr4_9_3]==0.8.0"
fi

python - <<'PY'
import importlib.util
import transformers

required = ["torch", "ray", "transformers", "vllm", "datasets", "pyarrow"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing dependencies: {missing}")
print("deps_ok:", ", ".join(required))
print("transformers_version:", transformers.__version__)
print("math_verify_available:", int(importlib.util.find_spec("math_verify") is not None))
PY

if [[ "${VERIFY_HF_MODELS}" == "1" ]]; then
  VERIFY_ARGS=(--models "${SMOKE_MODEL_PATH}" "${SCALE_MODEL_PATH}" "${THESIS_MODEL_PATH}")
  if [[ "${RUN_QWEN35_TRANSFORMERS_LOAD_SMOKE}" == "1" ]]; then
    VERIFY_ARGS+=(--load-smoke-model "${PILOT_MODEL_PATH}")
  fi
  if [[ "${RUN_QWEN35_VLLM_LOAD_SMOKE}" == "1" ]]; then
    VERIFY_ARGS+=(
      --vllm-smoke-model "${PILOT_MODEL_PATH}"
      --vllm-tensor-parallel-size "${QWEN35_VLLM_SMOKE_TP}"
      --vllm-max-model-len "${QWEN35_VLLM_SMOKE_MAX_MODEL_LEN}"
      --vllm-gpu-memory-utilization "${QWEN35_VLLM_SMOKE_GPU_UTIL}"
    )
  fi
  python experiments/math/verify_hf_models.py "${VERIFY_ARGS[@]}"
fi

if [[ "${PREPARE_DATA}" == "1" && ! -f data/dapo_math_en/train.parquet ]]; then
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
fi

if [[ "${RUN_CPU_CHECK}" == "1" ]]; then
  PYTHON=.venv/bin/python bash experiments/math/test_cpu_pipeline.sh
fi

echo "setup_ok"
