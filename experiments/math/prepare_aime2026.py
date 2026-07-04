#!/usr/bin/env python3
"""Prepare AIME 2026 problems for SDPO-Math benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_OUTPUT_PATH = Path("data/aime2026/test.parquet")
DEFAULT_PROMPT_SUFFIX = (
    "Solve the problem carefully. Keep the reasoning concise. End with exactly one final answer in \\boxed{}."
)
PROBLEM_FIELDS = ("problem", "prompt", "question", "description", "statement")
ANSWER_FIELDS = ("answer", "ground_truth", "final_answer", "solution")
ID_FIELDS = ("id", "problem_id", "problem_idx", "index", "name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, help="Hugging Face dataset id for AIME 2026.")
    parser.add_argument("--split", default="train", help="Hugging Face split to load.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--problem-key", default="auto")
    parser.add_argument("--answer-key", default="auto")
    parser.add_argument("--id-key", default="auto")
    parser.add_argument("--data-source", default="aime2026")
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def load_rows_from_hf(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Loading from Hugging Face requires `datasets`. Use --local-file or install it.") from exc
    return [dict(row) for row in load_dataset(dataset_name, split=split)]


def nested_get(row: dict[str, Any], key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def infer_key(row: dict[str, Any], explicit_key: str, candidates: tuple[str, ...], label: str) -> str:
    if explicit_key != "auto":
        if nested_get(row, explicit_key) is None:
            raise KeyError(f"{label} key {explicit_key!r} was not found in the first row.")
        return explicit_key
    for key in candidates:
        value = nested_get(row, key)
        if value is not None and str(value).strip():
            return key
    raise KeyError(f"Could not infer {label} key. Available top-level keys: {sorted(row.keys())}")


def format_prompt(problem: str, prompt_suffix: str) -> str:
    return f"{problem.strip()}\n\n{prompt_suffix.strip()}"


def stringify_answer(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value or "").strip()


def convert_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("AIME 2026 source has no rows.")

    problem_key = infer_key(rows[0], args.problem_key, PROBLEM_FIELDS, "problem")
    answer_key = infer_key(rows[0], args.answer_key, ANSWER_FIELDS, "answer")
    id_key = args.id_key
    if id_key == "auto":
        id_key = next((key for key in ID_FIELDS if nested_get(rows[0], key) is not None), "")

    converted: list[dict[str, Any]] = []
    skipped = 0
    for row_idx, row in enumerate(rows):
        problem = str(nested_get(row, problem_key) or "").strip()
        answer = stringify_answer(nested_get(row, answer_key))
        if not problem or not answer:
            skipped += 1
            continue
        index_value = nested_get(row, id_key) if id_key else row_idx
        converted.append(
            {
                "data_source": args.data_source,
                "prompt": [{"role": "user", "content": format_prompt(problem, args.prompt_suffix)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "test",
                    "index": str(index_value),
                    "raw_prompt": problem,
                    "feedback_mode": "none",
                },
            }
        )
        if 0 < args.max_samples <= len(converted):
            break

    if not converted:
        raise ValueError("No rows with both problem and answer were found.")

    metadata = {
        "rows": len(converted),
        "skipped": skipped,
        "problem_key": problem_key,
        "answer_key": answer_key,
        "id_key": id_key or None,
        "data_source": args.data_source,
        "prompt_suffix": args.prompt_suffix,
    }
    return converted, metadata


def main() -> None:
    args = parse_args()
    rows = load_rows_from_hf(args.dataset_name, args.split)
    converted, metadata = convert_rows(rows, args)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(converted)
    pq.write_table(table, args.output_path, compression="snappy")
    metadata_path = args.output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"prepared={args.output_path}")
    print(f"metadata={metadata_path}")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
