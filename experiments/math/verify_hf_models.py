#!/usr/bin/env python3
"""Verify that configured Hugging Face model ids are accessible before long runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import model_info
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Hugging Face model ids to verify.")
    parser.add_argument(
        "--allow-automodel-fallback",
        action="store_true",
        help="Allow generic AutoModel fallback. By default this is rejected because RL generation expects a language head.",
    )
    return parser.parse_args()


def select_verl_auto_class(config) -> str:
    """Mirror the FSDP worker's AutoModel selection without loading weights."""
    architectures = getattr(config, "architectures", None) or []
    architecture = architectures[0] if architectures else ""
    auto_map = getattr(config, "auto_map", None)
    if auto_map and architecture:
        for auto_class, remote_class in auto_map.items():
            if architecture in str(remote_class):
                return auto_class

    config_type = type(config)
    if config_type in AutoModelForVision2Seq._model_mapping.keys():
        return "AutoModelForVision2Seq"
    if config_type in AutoModelForCausalLM._model_mapping.keys():
        return "AutoModelForCausalLM"
    if config_type in AutoModelForImageTextToText._model_mapping.keys():
        return "AutoModelForImageTextToText"
    if config_type in AutoModel._model_mapping.keys():
        return "AutoModel"
    return "unsupported"


def main() -> None:
    args = parse_args()
    for model_id in dict.fromkeys(args.models):
        model_path = Path(model_id).expanduser()
        if model_path.exists():
            print("local_model_ok:", {"path": str(model_path)})
            config_source = str(model_path)
        else:
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
            config_source = model_id
        try:
            config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        except Exception as exc:
            raise SystemExit(
                f"transformers_model_config_failed: {model_id}\n"
                f"{type(exc).__name__}: {exc}\n"
                "This model exists on Hugging Face, but the installed Transformers stack cannot load "
                "its architecture. Use a model supported by the installed stack, or install a newer "
                "Transformers/vLLM-compatible stack before running this phase."
            ) from exc
        print("transformers_config_ok:", {"id": model_id, "model_type": getattr(config, "model_type", None)})
        auto_class = select_verl_auto_class(config)
        if auto_class in {"unsupported", "AutoModel"} and not args.allow_automodel_fallback:
            raise SystemExit(
                f"verl_model_class_unsupported: {model_id}\n"
                f"selected_auto_class={auto_class}, architectures={getattr(config, 'architectures', None)}\n"
                "The installed Transformers stack can read the config, but it does not expose a language-generation "
                "AutoModel class that the FSDP worker can use safely. Upgrade Transformers/vLLM or choose another "
                "Qwen3.x checkpoint."
            )
        print("verl_auto_class_ok:", {"id": model_id, "auto_class": auto_class})


if __name__ == "__main__":
    main()
