#!/usr/bin/env python3
"""Run AIME 2026 benchmark for SDPO-Math checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAINED_VARIANTS = {"base_rl", "sdpo_vanilla", "sdpo_reliability", "sdpo_reliability_gate"}
BASELINE_VARIANTS = {"base_model"}
BENCHMARK_VARIANTS = BASELINE_VARIANTS | TRAINED_VARIANTS


def load_math_feedback_module():
    module_path = PROJECT_ROOT / "verl/utils/reward_score/feedback/math.py"
    spec = importlib.util.spec_from_file_location("sdpo_math_feedback", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load math feedback scorer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


math_feedback = load_math_feedback_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data/aime2026/test.parquet")
    parser.add_argument("--model-path", default=None, help="Base model path. Defaults to manifest model, then THESIS_MODEL_PATH.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Phase log dir; used to infer exp_suffix.")
    parser.add_argument("--exp-suffix", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints/sdpo_math")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["auto"],
        help="Variants to benchmark. Use `auto` for base_model plus checkpointed variants from the run manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def latest_thesis_log_dir() -> Path | None:
    marker = PROJECT_ROOT / "logs/sdpo_math_phase/latest_thesis_log_dir.txt"
    if marker.exists():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return Path(value)
    return None


def resolve_log_dir(log_dir: Path | None) -> Path | None:
    if log_dir is None:
        return latest_thesis_log_dir()
    return log_dir


def load_manifest(log_dir: Path | None) -> dict[str, Any]:
    resolved_log_dir = resolve_log_dir(log_dir)
    if resolved_log_dir is None:
        return {}
    manifest_path = resolved_log_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint_dir(root: Path) -> Path | None:
    tracker = root / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        step = tracker.read_text(encoding="utf-8").strip()
        candidate = root / f"global_step_{step}"
        if candidate.exists():
            return candidate
    candidates = [path for path in root.glob("global_step_*") if path.is_dir()]
    return max(candidates, key=checkpoint_step) if candidates else None


def checkpoint_is_loadable(checkpoint_dir: Path) -> bool:
    adapter = checkpoint_dir / "lora_adapter"
    if adapter.exists() and (adapter / "adapter_config.json").exists():
        return True
    hf_dir = checkpoint_dir / "huggingface"
    return any(hf_dir.glob("*.safetensors")) or any(hf_dir.glob("*.bin")) or any(hf_dir.glob("*.index.json"))


def variant_checkpoint_root(args: argparse.Namespace, variant: str, exp_suffix: str) -> Path:
    return args.checkpoint_root / f"{variant}_{exp_suffix}"


def discover_variants(args: argparse.Namespace, manifest: dict[str, Any], exp_suffix: str | None) -> list[str]:
    if args.variants != ["auto"]:
        invalid = [variant for variant in args.variants if variant not in BENCHMARK_VARIANTS]
        if invalid:
            raise ValueError(
                f"Invalid AIME benchmark variants={invalid}; valid variants={sorted(BENCHMARK_VARIANTS)}"
            )
        return args.variants

    available: list[str] = ["base_model"]
    if not exp_suffix:
        return available

    manifest_variants = [variant for variant in manifest.get("variants", []) if variant in TRAINED_VARIANTS]
    if manifest_variants:
        candidates = manifest_variants
    else:
        candidates = []
        suffix = f"_{exp_suffix}"
        for path in sorted(args.checkpoint_root.glob(f"*{suffix}")):
            variant = path.name[: -len(suffix)]
            if variant in TRAINED_VARIANTS:
                candidates.append(variant)

    missing: list[str] = []
    for variant in candidates:
        latest = latest_checkpoint_dir(variant_checkpoint_root(args, variant, exp_suffix))
        if latest is not None and checkpoint_is_loadable(latest):
            available.append(variant)
        else:
            missing.append(variant)

    if missing:
        print(f"skip_missing_variants={missing}")
    if not available:
        raise FileNotFoundError(
            f"No loadable checkpoints found for exp_suffix={exp_suffix} under {args.checkpoint_root}"
        )
    return available


def resolve_variant_adapter(args: argparse.Namespace, variant: str, exp_suffix: str | None) -> tuple[Path | None, Path | None]:
    if variant == "base_model":
        return None, None
    if variant not in TRAINED_VARIANTS:
        raise ValueError(f"Unknown variant={variant}. Valid trained variants: {sorted(TRAINED_VARIANTS)}")
    if not exp_suffix:
        raise ValueError(f"exp_suffix is required to locate checkpoint for {variant}. Pass --log-dir or --exp-suffix.")

    checkpoint_root = variant_checkpoint_root(args, variant, exp_suffix)
    latest = latest_checkpoint_dir(checkpoint_root)
    if latest is None:
        raise FileNotFoundError(f"Missing checkpoint for {variant}: {checkpoint_root}")

    adapter = latest / "lora_adapter"
    if adapter.exists() and (adapter / "adapter_config.json").exists():
        return latest, adapter

    hf_dir = latest / "huggingface"
    if checkpoint_is_loadable(latest):
        return latest, None
    raise FileNotFoundError(
        f"{variant} checkpoint has no loadable lora_adapter or full Hugging Face weights: {latest}"
    )


def torch_dtype(dtype_name: str):
    import torch

    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Path | None,
    checkpoint_dir: Path | None,
    args: argparse.Namespace,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = str((checkpoint_dir / "huggingface") if checkpoint_dir and (checkpoint_dir / "huggingface").exists() else model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_path = str(checkpoint_dir / "huggingface") if checkpoint_dir and adapter_path is None else model_path
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading SDPO LoRA checkpoints requires `peft`.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    torch.manual_seed(args.seed)
    return model, tokenizer


def load_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def batched(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def generated_token_length(output_ids, prompt_length: int, eos_token_id: int | None) -> tuple[int, bool]:
    new_ids = output_ids[prompt_length:]
    ids = [int(token) for token in new_ids.tolist()]
    if eos_token_id is not None and eos_token_id in ids:
        return ids.index(eos_token_id) + 1, False
    return len(ids), True


def generate_variant(
    *,
    variant: str,
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    do_sample = args.temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
        if args.top_k >= 0:
            generation_kwargs["top_k"] = args.top_k

    records: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    input_device = model_input_device(model)
    for sample_idx in range(args.n_samples):
        for batch_start, batch_rows in batched(rows, args.batch_size):
            messages = [row["prompt"] for row in batch_rows]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                enable_thinking=args.enable_thinking,
            )
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)

            for local_idx, (row, output_ids) in enumerate(zip(batch_rows, outputs)):
                new_token_count, reached_limit = generated_token_length(output_ids, prompt_length, tokenizer.eos_token_id)
                response_ids = output_ids[prompt_length : prompt_length + new_token_count]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                ground_truth = str(row["reward_model"]["ground_truth"])
                extra_info = dict(row.get("extra_info") or {})
                extra_info["truncated"] = bool(reached_limit and new_token_count >= args.max_new_tokens)
                score = math_feedback.compute_score(response, ground_truth, extra_info=extra_info)
                records.append(
                    {
                        "variant": variant,
                        "problem_index": str(extra_info.get("index", batch_start + local_idx)),
                        "sample_index": sample_idx,
                        "ground_truth": ground_truth,
                        "response": response,
                        "response_tokens": new_token_count,
                        **{f"score_{key}": value for key, value in score.items() if key != "feedback"},
                    }
                )
            print(f"[{variant}] sample={sample_idx + 1}/{args.n_samples} batch={batch_start + len(batch_rows)}/{len(rows)}")

    elapsed = time.perf_counter() - start_time
    summary = summarize_records(variant, records, args.n_samples, elapsed)
    return records, summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_records(variant: str, records: list[dict[str, Any]], n_samples: int, elapsed: float) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_problem.setdefault(record["problem_index"], []).append(record)

    first_records = [sorted(problem_records, key=lambda item: item["sample_index"])[0] for problem_records in by_problem.values()]
    passk = [
        1.0 if any(float(record["score_acc"]) >= 1.0 for record in problem_records) else 0.0
        for problem_records in by_problem.values()
    ]
    return {
        "variant": variant,
        "problems": len(by_problem),
        "samples_per_problem": n_samples,
        "acc_at_1": mean([float(record["score_acc"]) for record in first_records]),
        "pass_at_k": mean(passk),
        "incorrect_format_at_1": mean([float(record["score_incorrect_format"]) for record in first_records]),
        "truncated_at_1": mean([float(record["score_truncated"]) for record in first_records]),
        "response_tokens_mean": mean([float(record["response_tokens"]) for record in records]),
        "math_verify_available": mean([float(record["score_math_verify_available"]) for record in records]),
        "elapsed_s": elapsed,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "problems",
        "samples_per_problem",
        "acc_at_1",
        "pass_at_k",
        "incorrect_format_at_1",
        "truncated_at_1",
        "response_tokens_mean",
        "math_verify_available",
        "elapsed_s",
        "checkpoint_step",
        "checkpoint_dir",
        "adapter_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_md(path: Path, summaries: list[dict[str, Any]]) -> None:
    headers = [
        "variant",
        "acc_at_1",
        "pass_at_k",
        "incorrect_format_at_1",
        "truncated_at_1",
        "response_tokens_mean",
        "elapsed_s",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                str(row[key]) if key == "variant" else f"{float(row[key]):.6g}"
                for key in headers
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolved_log_dir = resolve_log_dir(args.log_dir)
    manifest = load_manifest(resolved_log_dir)
    exp_suffix = args.exp_suffix or manifest.get("exp_suffix")
    model_path = args.model_path or manifest.get("model") or os.environ.get("THESIS_MODEL_PATH", "Qwen/Qwen3-8B")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif resolved_log_dir is not None:
        output_dir = resolved_log_dir / "benchmarks" / f"aime2026_{timestamp}"
    else:
        output_dir = PROJECT_ROOT / "benchmarks/aime2026" / str(exp_suffix or timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_examples(args.data_path, args.limit)
    variants = discover_variants(args, manifest, exp_suffix)
    config = {
        "data_path": str(args.data_path),
        "model_path": model_path,
        "log_dir": str(resolved_log_dir) if resolved_log_dir is not None else None,
        "exp_suffix": exp_suffix,
        "checkpoint_root": str(args.checkpoint_root),
        "variants": variants,
        "requested_variants": args.variants,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        checkpoint_dir, adapter_path = resolve_variant_adapter(args, variant, exp_suffix)
        print(f"== variant={variant} checkpoint={checkpoint_dir} adapter={adapter_path} ==")
        model, tokenizer = load_model_and_tokenizer(model_path, adapter_path, checkpoint_dir, args)
        records, summary = generate_variant(variant=variant, rows=rows, model=model, tokenizer=tokenizer, args=args)
        summary["checkpoint_dir"] = str(checkpoint_dir) if checkpoint_dir is not None else ""
        summary["adapter_path"] = str(adapter_path) if adapter_path is not None else ""
        summary["checkpoint_step"] = checkpoint_step(checkpoint_dir) if checkpoint_dir is not None else 0
        all_records.extend(records)
        summaries.append(summary)
        write_jsonl(output_dir / f"{variant}.jsonl", records)
        del model, tokenizer
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    write_jsonl(output_dir / "generations.jsonl", all_records)
    write_summary_csv(output_dir / "summary.csv", summaries)
    write_summary_md(output_dir / "summary.md", summaries)
    if resolved_log_dir is not None:
        marker = resolved_log_dir / "latest_aime2026_benchmark_dir.txt"
        marker.write_text(str(output_dir) + "\n", encoding="utf-8")
    print(f"output_dir={output_dir}")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
