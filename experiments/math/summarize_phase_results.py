#!/usr/bin/env python3
"""Create thesis-ready summary tables from SDPO-Math phase metrics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


VARIANTS = ["base_model", "base_rl", "sdpo_vanilla", "sdpo_reliability"]
SUMMARY_COLUMNS = [
    "variant",
    "step",
    "val_acc_mean",
    "val_acc_best",
    "val_reward_mean",
    "incorrect_format_mean",
    "truncated_mean",
    "math_verify_available",
    "sdpo_reprompt_fraction",
    "sdpo_feedback_used_fraction",
    "sdpo_reliability_weight_mean",
    "actor_pg_loss",
    "actor_grad_norm",
    "throughput_tokens_per_s",
    "metric_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def infer_variant(path: Path) -> str:
    stem = path.stem
    for variant in VARIANTS:
        if stem == variant or stem.startswith(f"{variant}_"):
            return variant
    return stem


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pick(data: dict[str, Any], contains: list[str], prefer_prefix: str | None = None) -> Any:
    candidates = [
        (key, value)
        for key, value in data.items()
        if all(token in key for token in contains)
    ]
    if prefer_prefix is not None:
        prefixed = [(key, value) for key, value in candidates if key.startswith(prefer_prefix)]
        if prefixed:
            candidates = prefixed
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def summarize_metric_file(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"empty metrics file: {path}")

    final = rows[-1]
    data = final.get("data", {})
    return {
        "variant": infer_variant(path),
        "step": final.get("step", ""),
        "val_acc_mean": pick(data, ["val-core", "/acc/", "mean"], "val-core/"),
        "val_acc_best": pick(data, ["val-core", "/acc/", "best"], "val-core/"),
        "val_reward_mean": pick(data, ["val-core", "/reward/", "mean"], "val-core/"),
        "incorrect_format_mean": pick(data, ["val-aux", "/incorrect_format/", "mean"], "val-aux/"),
        "truncated_mean": pick(data, ["val-aux", "/truncated/", "mean"], "val-aux/"),
        "math_verify_available": pick(data, ["val-aux", "/math_verify_available/", "mean"], "val-aux/"),
        "sdpo_reprompt_fraction": data.get("self_distillation/reprompt_sample_fraction", ""),
        "sdpo_feedback_used_fraction": data.get("self_distillation/feedback_used_fraction", ""),
        "sdpo_reliability_weight_mean": data.get("self_distillation/reliability_weight_mean", ""),
        "actor_pg_loss": data.get("actor/pg_loss", ""),
        "actor_grad_norm": data.get("actor/grad_norm", ""),
        "throughput_tokens_per_s": data.get("perf/throughput", ""),
        "metric_file": str(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(SUMMARY_COLUMNS[:-1]) + " |\n")
        f.write("| " + " | ".join(["---"] * (len(SUMMARY_COLUMNS) - 1)) + " |\n")
        for row in rows:
            vals = [str(row.get(col, "")) for col in SUMMARY_COLUMNS[:-1]]
            f.write("| " + " | ".join(vals) + " |\n")


def main() -> None:
    args = parse_args()
    metric_files = sorted(glob.glob(str(args.log_dir / "metrics" / "SDPO-Math" / "*.jsonl")))
    if not metric_files:
        raise SystemExit(f"no file logger metrics found under {args.log_dir}/metrics/SDPO-Math")

    rows = [summarize_metric_file(Path(path)) for path in metric_files]
    rows.sort(key=lambda row: VARIANTS.index(row["variant"]) if row["variant"] in VARIANTS else len(VARIANTS))

    output_dir = args.output_dir or args.log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"summary_csv={csv_path}")
    print(f"summary_md={md_path}")


if __name__ == "__main__":
    main()
