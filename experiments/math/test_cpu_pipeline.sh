#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON:-python3}"

echo "[1/4] Checking shell script syntax"
bash -n \
  experiments/math/run_sdpo_math_vanilla.sh \
  experiments/math/run_sdpo_math_safe_feedback.sh \
  experiments/math/run_sdpo_math_reliability.sh \
  experiments/math/run_sdpo_math_smoke.sh

echo "[2/4] Checking YAML config"
"${PYTHON_BIN}" - <<'PY'
import yaml

with open("verl/trainer/config/sdpo_math_l40s.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

assert cfg["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == "sdpo"
assert cfg["actor_rollout_ref"]["model"]["lora_rank"] > 0
assert cfg["actor_rollout_ref"]["actor"]["self_distillation"]["reliability_weighting"] is False
assert cfg["trainer"]["n_gpus_per_node"] == 2
assert cfg["reward_manager"]["name"] == "naive"
print("config ok")
PY

echo "[3/4] Checking prepared DAPO-Math parquet"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import pyarrow.parquet as pq

required = [
    Path("data/dapo_math_en/train.parquet"),
    Path("data/dapo_math_en/val.parquet"),
    Path("data/dapo_math_en/removed_duplicates.jsonl"),
    Path("data/dapo_math_en/removed_contaminated.jsonl"),
    Path("reports/decontamination_report.md"),
]
missing = [str(path) for path in required if not path.exists()]
assert not missing, f"missing files: {missing}"

for split, expected_min_rows in [("train", 1), ("val", 1)]:
    rows = pq.read_table(f"data/dapo_math_en/{split}.parquet").to_pylist()
    assert len(rows) >= expected_min_rows
    assert all(row["data_source"] == "math_dapo" for row in rows)
    assert all(row["reward_model"]["ground_truth"] for row in rows)
    assert all(row["extra_info"].get("feedback_mode") == "safe" for row in rows)
    assert all(
        row["prompt"][0]["content"].endswith(
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        for row in rows
    )
    assert not any("Answer:" in row["prompt"][0]["content"] for row in rows)
    print(split, len(rows), "rows ok")
PY

echo "[4/4] Checking math feedback behavior without math-verify"
"${PYTHON_BIN}" - <<'PY'
import importlib.util

spec = importlib.util.spec_from_file_location("math_feedback", "verl/utils/reward_score/feedback/math.py")
math_feedback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_feedback)

cases = [
    ("correct", r"Reasoning... \boxed{34}", "34", {"feedback_mode": "safe"}, 1.0, ""),
    (
        "wrong_safe",
        r"Reasoning... \boxed{35}",
        "34",
        {"feedback_mode": "safe"},
        0.0,
        "Your boxed final answer was parsed, but it is incorrect. Recheck the reasoning and final calculation.",
    ),
    ("wrong_none", r"Reasoning... \boxed{35}", "34", {"feedback_mode": "none"}, 0.0, ""),
    (
        "wrong_oracle",
        r"Reasoning... \boxed{35}",
        "34",
        {"feedback_mode": "oracle"},
        0.0,
        "Your answer is incorrect. The correct answer is 34.",
    ),
    (
        "bad_format",
        "Reasoning... final answer is 35",
        "34",
        {"feedback_mode": "safe"},
        0.0,
        "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}.",
    ),
    (
        "truncated",
        "Reasoning...",
        "34",
        {"feedback_mode": "safe", "truncated": True},
        0.0,
        "Your response was truncated because it exceeded the maximum length.",
    ),
]

for name, prediction, ground_truth, extra_info, expected_score, expected_feedback in cases:
    result = math_feedback.compute_score(prediction, ground_truth, extra_info)
    assert result["score"] == expected_score, (name, result)
    assert result["feedback"] == expected_feedback, (name, result)
    assert result["math_verify_available"] == 0, (name, result)
    print(name, "ok")
PY

echo "CPU pipeline checks passed"
