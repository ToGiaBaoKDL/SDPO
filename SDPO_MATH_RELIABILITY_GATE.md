# Proposed Method and Design: Reliability-Gated SDPO for Mathematical Reasoning

This document describes the proposed method used in the thesis experiments:
**Reliability-Gated Self-Distillation Policy Optimization (RG-SDPO)** for
mathematical reasoning. The method extends vanilla SDPO by making the
self-distillation target selective, reliability-weighted, and token-budgeted.

The core motivation is that math feedback is not uniformly reliable. A verified
peer solution is a strong imitation target, while an incorrect or truncated
solution can be fluent but harmful. Vanilla SDPO uses feedback-reprompted
targets whenever they are available, but it does not explicitly distinguish
between high-confidence and low-confidence targets. RG-SDPO keeps the original
reinforcement-learning objective intact and modifies only the auxiliary
self-distillation term.

## 1. Problem Setting

Let `x_i` denote a math problem sampled from the training set. For each prompt,
the rollout policy `pi_theta` generates one or more candidate responses
`y_i = (y_{i,1}, ..., y_{i,T_i})`. A math reward function evaluates the response
against the reference answer and produces:

```text
r_i = r(x_i, y_i)
```

where `r_i = 1` indicates a verified correct answer and `r_i = 0` indicates an
incorrect, malformed, or truncated response. The reward function also provides
structured metadata, including parsed answer, format errors, truncation flags,
and optional feedback text.

The training objective has two components:

```text
L = L_RL + lambda * L_SDPO
```

where `L_RL` is the existing policy-gradient loss used by the RL trainer, and
`L_SDPO` is an auxiliary self-distillation loss. The proposed method changes the
masking and weighting of `L_SDPO`; it does not change reward computation or the
base RL objective.

## 2. Vanilla SDPO

Vanilla SDPO constructs a teacher context by reprompting the model with useful
information from the rollout group. In this math adaptation, the reprompt may
include:

- the original problem statement,
- a verified successful peer solution from another rollout of the same prompt,
- safe feedback explaining that the boxed final answer is incorrect,
- format feedback when the final answer is not in `\boxed{...}`.

Let `c_i` denote this reprompt context. The student distribution is evaluated
under the original prompt context:

```text
p^S_{i,t} = pi_theta(. | x_i, y_{i,<t})
```

The teacher distribution is evaluated under the reprompt context:

```text
p^T_{i,t} = q_phi(. | c_i, y_{i,<t})
```

In the implementation, `q_phi` is an exponential-moving-average teacher of the
student policy. The teacher is used in a teacher-forcing manner: the same
response tokens `y_i` are used for student and teacher likelihood evaluation.
No additional teacher response is generated inside the actor loss.

Let `a_{i,t}` be the response-token mask and let `m_i` be the vanilla SDPO target
mask:

```text
m_i = 1 if sample i has a usable peer solution or usable feedback, else 0.
```

The vanilla SDPO loss is:

```text
L_SDPO^vanilla =
  (1 / Z_v) * sum_i sum_t
    a_{i,t} * m_i * D_alpha(p^S_{i,t}, p^T_{i,t})

Z_v = sum_i sum_t a_{i,t} * m_i
```

The active implementation uses full-logit top-k generalized Jensen-Shannon
distillation. With:

```text
M_alpha = (1 - alpha) * p^S + alpha * p^T
```

the divergence is:

```text
D_alpha(p^S, p^T) =
  (1 - alpha) * KL(p^S || M_alpha)
  + alpha * KL(p^T || M_alpha)
```

The thesis configuration uses:

```text
alpha = 0.5
top_k = 50
tail_mass = enabled
```

Thus, the loss is a symmetric Jensen-Shannon-style divergence over the top-50
distillation support plus a residual tail bucket.

## 3. Limitation of Vanilla SDPO in Math

Vanilla SDPO assumes that every available self-distillation target is useful.
This assumption is weak in mathematical reasoning for three reasons.

First, correctness is sparse and verifiable. A response can contain plausible
reasoning while still ending with an incorrect answer. Distilling from such a
target can reinforce invalid reasoning patterns.

Second, feedback types have different epistemic strength. A verified peer
solution is a much stronger target than generic correctness feedback. Format
feedback can improve answer presentation, but it provides little information
about the correct mathematical derivation.

Third, SDPO is computationally expensive. It adds teacher contexts, teacher
logits, and actor-side distillation computation. Spending this computation on
low-quality targets reduces both training efficiency and signal quality.

## 4. Proposed RG-SDPO

RG-SDPO introduces two additional variables for each sample:

```text
w_i in [0, 1]     reliability weight
g_i in {0, 1}     reliability gate
```

The reliability weight measures how trustworthy the SDPO target is. The gate
decides whether that target should be used by the distillation loss at the
current training step.

The proposed objective is:

```text
L_SDPO^RG =
  (1 / Z_g) * sum_i sum_t
    a_{i,t} * m_i * g_i * w_i * D_alpha(p^S_{i,t}, p^T_{i,t})

Z_g = sum_i sum_t a_{i,t} * m_i * g_i * w_i
```

and the full optimization objective becomes:

```text
L_RG-SDPO = L_RL + lambda * L_SDPO^RG
```

If `g_i = 0` or `w_i = 0`, sample `i` contributes no SDPO loss. The RL reward
and RL advantage for that sample are still processed normally by `L_RL`.

## 5. Reliability Weighting

The reliability function is deterministic and derived from reward metadata.
It does not require an external LLM judge.

| Target source | Weight `w_i` | Rationale |
| --- | ---: | --- |
| Verified successful peer solution | `1.0` | Strongest target; the solution passed math verification. |
| Safe correctness feedback | `0.4` | Useful correction signal, but not a complete verified solution. |
| Format feedback | `0.2` | Helps enforce `\boxed{...}` output format but weak for reasoning. |
| Truncated response | `0.0` | Unsafe target; the reasoning or final answer is incomplete. |
| No peer solution and no usable feedback | `0.0` | No reliable SDPO target is available. |

This weighting scheme reflects the relative confidence of each supervision
source. Verified solutions receive full weight. Feedback-only targets are kept
as weaker supervision because they can still guide correction, but they should
not dominate the distillation objective.

## 6. Adaptive Reliability Gate

The reliability threshold is scheduled over training. Let `t` be the current
training step and `T` be the total number of training steps. Define:

```text
p_t = (t - 1) / max(T - 1, 1)
```

The active reliability threshold is:

```text
tau_t = tau_start + p_t * (tau_end - tau_start)
```

The active compute budget is:

```text
rho_t = rho_start + p_t * (rho_end - rho_start)
```

A target is eligible when:

```text
e_i = m_i * 1[w_i >= tau_t]
```

The thesis configuration uses:

```text
tau_start = 0.2
tau_end   = 0.4
rho_start = 0.6
rho_end   = 0.5
```

This schedule implements a simple curriculum. Early training is more permissive
and can use format-correction targets. Later training is stricter and focuses on
verified peer solutions and safe correctness feedback.

## 7. Token-Budgeted Target Selection

A sample with a long reprompt and long response is more expensive than a short
sample. Therefore, RG-SDPO uses a token-budgeted gate instead of selecting
targets only by sample count.

Let `c_i` be the teacher-token cost of sample `i`, measured from the teacher
attention mask. The method ranks eligible targets by reliability per teacher
token:

```text
u_i = w_i / max(c_i, 1)
```

Then it greedily selects targets with high `u_i` under the active token budget:

```text
sum_i g_i * c_i <= rho_t * sum_i c_i
```

where:

```text
g_i = 1 if target i is selected under the token budget, else 0.
```

This design prefers targets that are both reliable and efficient. It is
especially useful for math reasoning, where responses can be long and SDPO
teacher contexts can significantly increase actor-update cost.

## 8. Sparse Execution Design

The implementation separates target selection from distributed compute
alignment.

The target gate `g_i` determines which samples have nonzero SDPO loss. For
rejected samples, the SDPO reliability weight is set to zero before actor loss
computation.

In distributed data-parallel training, some additional rows may be computed only
to keep per-rank batch shapes aligned. These rows have zero SDPO weight and do
not affect the objective. This preserves correctness while avoiding unnecessary
distillation loss on low-reliability targets.

The implementation logs both quantities:

```text
target_fraction  = fraction selected by g_i
compute_fraction = fraction computed after distributed alignment
```

This distinction is important for reporting both methodological behavior and
actual training cost.

## 9. Comparison With Vanilla SDPO

| Property | Vanilla SDPO | RG-SDPO |
| --- | --- | --- |
| Uses feedback reprompting | Yes | Yes |
| Uses math-verifier metadata | Only for reward | For reward and SDPO target reliability |
| Treats all SDPO targets equally | Yes | No |
| Weights targets by reliability | No | Yes |
| Filters low-confidence targets | No | Yes |
| Uses adaptive gate schedule | No | Yes |
| Uses token-budgeted target selection | No | Yes |
| Reduces noisy imitation risk | Limited | Stronger |
| Provides interpretable gate metrics | Limited | Yes |

The main conceptual improvement is that RG-SDPO changes SDPO from
availability-based imitation to reliability-aware imitation. Vanilla SDPO asks
"is a reprompt target available?" RG-SDPO asks "is the target reliable enough to
imitate, and is it worth the token cost?"

## 10. Algorithm Summary

For each training batch:

```text
1. Generate rollout responses y_i from pi_theta.
2. Score each response with the math reward/verifier.
3. Construct feedback and peer-solution metadata.
4. Build an SDPO reprompt context c_i when a usable target exists.
5. Assign reliability weight w_i from reward metadata.
6. Compute scheduled threshold tau_t and budget rho_t.
7. Mark eligible targets e_i = m_i * 1[w_i >= tau_t].
8. Select gated targets g_i under the token budget.
9. Apply RL loss to all samples.
10. Apply SDPO loss only where m_i * g_i * w_i > 0.
11. Update the EMA teacher.
```

This algorithm preserves the original RL learning signal while making the
auxiliary SDPO signal more selective.

## 11. Implementation Constants Used in the Thesis Runs

| Component | Value |
| --- | --- |
| Dataset | `open-r1/DAPO-Math-17k-Processed`, English subset |
| Training files | `data/dapo_math_en/train.parquet`, `data/dapo_math_en/val.parquet` |
| Main model | `Qwen/Qwen3-8B` |
| Optimization variants | `base_rl`, `sdpo_vanilla`, `sdpo_reliability_gate` |
| LoRA rank / alpha | `32 / 32` |
| Teacher regularization | EMA teacher |
| EMA update rate | `0.01` |
| Distillation divergence | generalized Jensen-Shannon |
| `alpha` | `0.5` |
| Distillation support | top-50 logits plus tail mass |
| Importance-ratio clip | `2.0` |
| Reliability threshold schedule | linear |
| `tau_start`, `tau_end` | `0.2`, `0.4` |
| `rho_start`, `rho_end` | `0.6`, `0.5` |
| Gate budget mode | teacher-token budget |
| Validation decoding | greedy, `n=1`, `temperature=0.01` |
| Required answer format | `\boxed{...}` |

The active thesis H200 profile uses `train_batch_size=48`, `rollout.n=4`, and
therefore `192` rollout responses per training step. The A100/H100 profile uses a
smaller batch size for memory stability.

## 12. Metrics and Qualitative Analysis

The method is evaluated using both outcome metrics and mechanism metrics.

Primary outcome metrics:

| Metric | Meaning |
| --- | --- |
| `val_acc_mean` | Validation accuracy under math verification. |
| `incorrect_format_mean` | Fraction of validation outputs with malformed final answer format. |
| `truncated_mean` | Fraction of outputs clipped by the response-length limit. |
| `response_length_mean` | Average generated response length. |
| `response_length_clip_ratio` | Frequency of responses reaching the length cap. |

Gate and efficiency metrics:

| Metric | Meaning |
| --- | --- |
| `sdpo_reprompt_fraction` | Fraction of samples with an SDPO reprompt target. |
| `sdpo_feedback_used_fraction` | Fraction of samples where feedback is used in the reprompt. |
| `sdpo_reliability_weight_mean` | Average target reliability weight. |
| `sdpo_reliability_gate_threshold` | Active threshold `tau_t`. |
| `sdpo_reliability_gate_schedule_progress` | Schedule progress `p_t`. |
| `sdpo_reliability_gate_max_fraction` | Active budget `rho_t`. |
| `sdpo_reliability_gate_eligible_fraction` | Fraction satisfying `w_i >= tau_t`. |
| `sdpo_reliability_gate_fraction` | Fraction selected as nonzero SDPO targets. |
| `sdpo_reliability_gate_compute_fraction` | Fraction computed after distributed alignment. |
| `sdpo_reliability_gate_target_token_fraction` | Teacher-token fraction selected by the gate. |
| `sdpo_reliability_gate_compute_token_fraction` | Teacher-token fraction spent after alignment. |
| `time_per_step_s` | Wall-clock time per optimization step. |
| `throughput_tokens_per_s` | Overall token throughput. |

Qualitative SDPO trajectories are saved to:

```text
logs/sdpo_math_phase/<run_tag>/trajectories/<variant>.jsonl
```

Each trajectory record stores the original response, reward metadata, feedback,
reprompt, teacher-forced target, reliability weight, gate decision, token budget
usage, and the active SDPO/gate formulas. These records support qualitative
analysis of why a target was selected or rejected.

## 13. Expected Empirical Behavior

RG-SDPO is expected to improve SDPO in two ways.

First, it should reduce noisy imitation. Compared with vanilla SDPO, the gated
method should use fewer low-confidence targets and concentrate the auxiliary
loss on targets supported by verification or stronger feedback.

Second, it should improve the efficiency-quality tradeoff. Since SDPO adds
teacher-side computation, selecting fewer but more reliable targets can reduce
actor-update cost while preserving useful correction signals.

The strongest empirical outcome is:

```text
RG-SDPO >= vanilla SDPO in validation accuracy
and RG-SDPO < vanilla SDPO in SDPO compute fraction or actor-update time.
```

If RG-SDPO obtains similar accuracy with lower compute, it should be reported as
an efficiency improvement. If it obtains higher accuracy with similar or lower
compute, it supports the stronger claim that reliability-aware target selection
improves SDPO for math-domain policy optimization.

## 14. Thesis Claim

The proposed method can be summarized as follows:

```text
Reliability-Gated SDPO improves math-domain self-distillation by replacing
availability-based imitation with reliability-aware, token-budgeted imitation.
It uses verifier-derived metadata to weight and select SDPO targets, reducing
noisy imitation while preserving the correction signal provided by feedback
reprompting.
```

## 15. Code References

| Component | Location |
| --- | --- |
| Phase runner | `experiments/math/run_sdpo_math_benchmark.sh` |
| Hardware/profile settings | `experiments/math/phase_common.sh` |
| Manifest writer | `experiments/math/write_phase_manifest.py` |
| Reliability weights and gate scheduling | `verl/trainer/ppo/ray_trainer.py` |
| SDPO/Jensen-Shannon loss | `verl/trainer/ppo/core_algos.py` |
| Actor sparse SDPO execution | `verl/workers/actor/dp_actor.py` |
| Config defaults | `verl/trainer/config/sdpo_math_a100.yaml` |
