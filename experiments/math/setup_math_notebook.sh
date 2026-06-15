#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SDPO_PYTHON_VERSION="${SDPO_PYTHON_VERSION:-3.12}"
ALLOW_UNTESTED_PYTHON="${ALLOW_UNTESTED_PYTHON:-0}"
INSTALL_MATH_VERIFY="${INSTALL_MATH_VERIFY:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
RUN_CPU_CHECK="${RUN_CPU_CHECK:-0}"
VERIFY_HF_MODELS="${VERIFY_HF_MODELS:-0}"
STABLE_TRANSFORMERS_SPEC="${STABLE_TRANSFORMERS_SPEC:-transformers==4.57.1}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy==2.1.0}"
SKIP_INSTALL_IF_READY="${SKIP_INSTALL_IF_READY:-1}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"

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
echo "run_cpu_check=${RUN_CPU_CHECK}"
echo "verify_hf_models=${VERIFY_HF_MODELS}"
echo "stable_transformers_spec=${STABLE_TRANSFORMERS_SPEC}"
echo "numpy_spec=${NUMPY_SPEC}"
echo "skip_install_if_ready=${SKIP_INSTALL_IF_READY}"
echo "force_reinstall=${FORCE_REINSTALL}"
echo "vllm_worker_multiproc_method=${VLLM_WORKER_MULTIPROC_METHOD}"

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

VENV_READY=0
if [[ "${FORCE_REINSTALL}" != "1" && "${SKIP_INSTALL_IF_READY}" == "1" && -x .venv/bin/python ]]; then
  if INSTALL_MATH_VERIFY="${INSTALL_MATH_VERIFY}" \
    STABLE_TRANSFORMERS_SPEC="${STABLE_TRANSFORMERS_SPEC}" \
    NUMPY_SPEC="${NUMPY_SPEC}" \
    .venv/bin/python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import os
import sys

required = ["torch", "ray", "transformers", "vllm", "datasets", "pyarrow"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing {missing}")

if os.environ.get("INSTALL_MATH_VERIFY") == "1" and importlib.util.find_spec("math_verify") is None:
    raise SystemExit("missing math_verify")

def require_exact(dist_name: str, spec: str) -> None:
    if "==" not in spec:
        return
    expected = spec.split("==", 1)[1]
    actual = metadata.version(dist_name)
    if actual != expected:
        raise SystemExit(f"{dist_name} {actual} != {expected}")

require_exact("transformers", os.environ["STABLE_TRANSFORMERS_SPEC"])
require_exact("numpy", os.environ["NUMPY_SPEC"])
PY
  then
    VENV_READY=1
  fi
fi

if [[ "${VENV_READY}" == "1" ]]; then
  echo "venv_ready=1 skip_dependency_install=1"
else
  echo "venv_ready=0 installing_dependencies=1"
  python3 -m pip install -q -U uv
  uv venv .venv --python "${SDPO_PYTHON_VERSION}"
fi

unset SDPO_SKIP_VENV
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/math_env.sh"

python --version
if [[ "${VENV_READY}" != "1" ]]; then
  uv pip install -q -U pip
  uv pip install -q pyyaml pyarrow pandas datasets
  uv pip install -q -e ".[vllm]"

  echo "Installing stable Transformers ${STABLE_TRANSFORMERS_SPEC}"
  uv pip install -q -U "${STABLE_TRANSFORMERS_SPEC}"
  echo "Installing NumPy runtime pin ${NUMPY_SPEC}"
  uv pip install -q -U "${NUMPY_SPEC}"

  if [[ "${INSTALL_MATH_VERIFY}" == "1" ]]; then
    uv pip install -q "math-verify[antlr4_9_3]==0.8.0"
  fi
fi

python - <<'PY'
import importlib.util
import importlib.metadata as metadata
import transformers
from packaging.version import Version

required = ["torch", "ray", "transformers", "vllm", "datasets", "pyarrow"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing dependencies: {missing}")
print("deps_ok:", ", ".join(required))
print("transformers_version:", transformers.__version__)
numpy_version = metadata.version("numpy")
numba_version = metadata.version("numba") if importlib.util.find_spec("numba") else "not_installed"
print("numpy_version:", numpy_version)
print("numba_version:", numba_version)
if Version(numpy_version) >= Version("2.3"):
    raise SystemExit(f"numpy {numpy_version} is incompatible with numba/vLLM; expected numpy<2.3")
try:
    print("vllm_version:", metadata.version("vllm"))
except Exception as exc:
    print("vllm_version_unavailable:", type(exc).__name__)
print("math_verify_available:", int(importlib.util.find_spec("math_verify") is not None))
PY

if [[ "${VERIFY_HF_MODELS}" == "1" ]]; then
  python experiments/math/verify_hf_models.py --models "${SMOKE_MODEL_PATH}" "${SCALE_MODEL_PATH}" "${THESIS_MODEL_PATH}"
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
