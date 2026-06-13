#!/usr/bin/env python3
"""Validate SDPO-Math benchmark dry-run command semantics."""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_SNIPPETS = {
    "base_model": [
        "trainer.val_before_train=True",
        "trainer.val_only=True",
        "actor_rollout_ref.model.lora_rank=0",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
    ],
    "base_rl": [
        "trainer.total_training_steps=1",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
    ],
    "sdpo_vanilla": [
        "trainer.total_training_steps=1",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
    ],
    "sdpo_reliability": [
        "trainer.total_training_steps=1",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=True",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--steps", default="1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing_files: list[str] = []
    missing_snippets: list[str] = []

    for variant, snippets in EXPECTED_SNIPPETS.items():
        path = args.log_dir / f"{variant}_{args.phase}_{args.profile}_{args.steps}.log"
        if not path.exists():
            missing_files.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        if "DRY_RUN command:" not in text:
            missing_snippets.append(f"{path}: missing DRY_RUN command")
        for snippet in snippets:
            if snippet not in text:
                missing_snippets.append(f"{path}: missing {snippet}")

    if missing_files or missing_snippets:
        details = "\n".join(missing_files + missing_snippets)
        raise AssertionError(f"benchmark dry-run validation failed:\n{details}")

    print("benchmark_dryrun_semantics_ok")


if __name__ == "__main__":
    main()
