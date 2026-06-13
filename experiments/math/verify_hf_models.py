#!/usr/bin/env python3
"""Verify that configured Hugging Face model ids are accessible before long runs."""

from __future__ import annotations

import argparse

from huggingface_hub import model_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Hugging Face model ids to verify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model_id in args.models:
        info = model_info(model_id)
        if getattr(info, "private", False):
            raise SystemExit(f"model is private: {model_id}")
        if getattr(info, "gated", False):
            raise SystemExit(f"model is gated: {model_id}")
        if getattr(info, "disabled", False):
            raise SystemExit(f"model is disabled: {model_id}")
        print(
            "hf_model_ok:",
            {
                "id": info.modelId,
                "sha": getattr(info, "sha", None),
                "pipeline_tag": getattr(info, "pipeline_tag", None),
            },
        )


if __name__ == "__main__":
    main()
