#!/usr/bin/env python3
"""Validate SDPO-Math benchmark dry-run command semantics."""

from __future__ import annotations

import argparse
from pathlib import Path


COMMON_SNIPPETS = {
    "base_model": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path=Qwen/Qwen3-1.7B",
        "critic.model.path=Qwen/Qwen3-1.7B",
        "trainer.val_before_train=True",
        "trainer.val_only=True",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.model.lora_rank=0",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
    ],
    "base_rl": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path=Qwen/Qwen3-1.7B",
        "critic.model.path=Qwen/Qwen3-1.7B",
        "trainer.total_training_steps=1",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
    ],
    "sdpo_vanilla": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path=Qwen/Qwen3-1.7B",
        "critic.model.path=Qwen/Qwen3-1.7B",
        "trainer.total_training_steps=1",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
    ],
    "sdpo_reliability": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path=Qwen/Qwen3-1.7B",
        "critic.model.path=Qwen/Qwen3-1.7B",
        "trainer.total_training_steps=1",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=True",
    ],
}

FORBIDDEN_SNIPPETS = [
    "data.val_batch_size=",
    "RAY_BACKEND_LOG_LEVEL",
]

PROFILE_EXPECTATIONS = {
    ("a100", "fast"): {
        "train_batch_size": 64,
        "agent_workers": 128,
        "base_model_train_max_samples": 128,
        "gpu_util": "0.86",
        "enforce_eager": "False",
    },
    ("h100", "fast"): {
        "train_batch_size": 64,
        "agent_workers": 128,
        "base_model_train_max_samples": 128,
        "gpu_util": "0.92",
        "enforce_eager": "False",
    },
}


def expected_snippets(hardware_profile: str, profile: str) -> dict[str, list[str]]:
    settings = PROFILE_EXPECTATIONS.get((hardware_profile, profile))
    if settings is None:
        raise AssertionError(
            "validate_benchmark_dryrun.py does not know "
            f"hardware_profile={hardware_profile} profile={profile}. "
            f"Known combinations: {sorted(PROFILE_EXPECTATIONS)}"
        )

    result = {variant: snippets.copy() for variant, snippets in COMMON_SNIPPETS.items()}
    result["base_model"].append(f"data.train_max_samples={settings['base_model_train_max_samples']}")
    for snippets in result.values():
        snippets.append(f"data.train_batch_size={settings['train_batch_size']}")
        snippets.append(f"actor_rollout_ref.rollout.agent.num_workers={settings['agent_workers']}")
        snippets.append(f"actor_rollout_ref.rollout.gpu_memory_utilization={settings['gpu_util']}")
        snippets.append(f"actor_rollout_ref.rollout.enforce_eager={settings['enforce_eager']}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--hardware-profile", default="a100")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--steps", default="1")
    parser.add_argument("--exp-suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing_files: list[str] = []
    missing_snippets: list[str] = []
    exp_suffix = args.exp_suffix or f"{args.phase}_{args.profile}_{args.steps}_seed42"

    for variant, snippets in expected_snippets(args.hardware_profile, args.profile).items():
        path = args.log_dir / f"{variant}_{exp_suffix}.log"
        if not path.exists():
            missing_files.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        if "DRY_RUN command:" not in text:
            missing_snippets.append(f"{path}: missing DRY_RUN command")
        for snippet in snippets:
            if snippet not in text:
                missing_snippets.append(f"{path}: missing {snippet}")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                missing_snippets.append(f"{path}: forbidden {snippet}")

    if missing_files or missing_snippets:
        details = "\n".join(missing_files + missing_snippets)
        raise AssertionError(f"benchmark dry-run validation failed:\n{details}")

    print("benchmark_dryrun_semantics_ok")


if __name__ == "__main__":
    main()
