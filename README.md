# Verify-Repair LM

**Inference-time Generate–Verify–Repair for improving small language-model outputs.**

Verify-Repair LM is a lightweight inference-time wrapper around a language model. It does not train or modify the base model. Instead, it improves single-shot generation by creating prompt variants, generating candidate answers, verifying them with task-specific checks, repairing detected failures when possible, and selecting the best final answer.

The goal is simple:

```text
single prompt -> single greedy answer
```

becomes:

```text
user prompt
  -> prompt variants
  -> candidate generation
  -> verifier scoring
  -> repair/synthesis if needed
  -> final selected answer
```

This project is designed for small or self-hosted language models, where greedy decoding can be unreliable and running a larger frontier model may be too expensive or undesirable.

---

## Core idea

Let `x` be the user prompt and let `f_theta` be the base language model.

Instead of generating one greedy answer,

```math
y_{\mathrm{greedy}} = f_{\theta}(x),
```

the system creates several prompt variants,

```math
x_1, x_2, \ldots, x_M,
```

and generates candidate answers,

```math
C(x) = \{ y_{m,k} : y_{m,k} \sim f_{\theta}(x_m) \}.
```

Each candidate is evaluated by a verifier loss,

```math
L(y; x)
=
\lambda_1 L_{\mathrm{coverage}}(y; x)
+ \lambda_2 L_{\mathrm{semantic}}(y; x)
+ \lambda_3 L_{\mathrm{repetition}}(y)
+ \lambda_4 L_{\mathrm{format}}(y)
+ \lambda_5 L_{\mathrm{task}}(y; x).
```

The final answer is selected by

```math
y^\star = \arg\min_{y \in C(x) \cup R(x)} L(y; x),
```

where `R(x)` is the set of repaired or synthesized answers created when the best candidate still violates the verifier.

Equivalently, using a quality score `Q(y; x) = -L(y; x)`,

```math
y^\star = \arg\max_{y \in C(x) \cup R(x)} Q(y; x).
```

---

## Pipeline

### 1. Prompt variants

The original user prompt is rewritten into several variants. Each variant emphasizes a different objective, such as directness, technical correctness, business relevance, risk analysis, or structured final response.

### 2. Candidate generation

The same base model generates answers from each prompt variant.

### 3. Verification

Each answer is scored using generic and task-specific checks.

Generic checks include:

- prompt coverage
- repetition
- role leakage
- code leakage
- prompt copying
- answer length
- structural quality

Task-specific checks depend on the prompt category.

### 4. Repair or synthesis

If the best candidate is still weak, the system creates a repaired or synthesized answer.

For example, in a speculative-decoding explanation task, the verifier checks whether the answer correctly explains:

- the draft model
- the target model
- target-model verification
- accepted text tokens
- why high token acceptance improves speed
- fallback risks

### 5. Final selection

The repaired answer and all generated candidates are re-scored. The lowest-loss answer is returned.

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages:

```text
torch
transformers
numpy
```

---

## Run one example

```bash
python scripts/prompt_ensemble_competition.py
```

Run with a custom prompt:

```bash
PROMPT="Explain speculative decoding clearly to a startup founder. Focus on why high token acceptance matters for speed." \
python scripts/prompt_ensemble_competition.py
```

---

## Run benchmark

```bash
python scripts/prompt_ensemble_benchmark.py
```

Run only the first 5 prompts:

```bash
MAX_PROMPTS=5 python scripts/prompt_ensemble_benchmark.py
```

Use more candidate generations:

```bash
N_VARIANTS=5 SAMPLES_PER_VARIANT=2 python scripts/prompt_ensemble_benchmark.py
```

Use a larger model:

```bash
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct python scripts/prompt_ensemble_benchmark.py
```

---

## Pilot evidence

A pilot benchmark was run with the following setup:

| Setting | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Number of prompts | 20 |
| Prompt variants | 5 |
| Samples per variant | 1 |
| Evaluation metric | Internal verifier score |

Results:

| Metric | Value |
|---|---:|
| Mean greedy score | 33.45 |
| Mean ensemble-best score | 46.15 |
| Mean final score | 49.55 |
| Final win rate vs greedy by verifier score | 90% |
| Mean final gain vs greedy | +16.10 |
| Repair trigger rate | 15% |

These are **internal verifier-score results**, not human-preference results. Human evaluation is still required.

See:

```text
docs/STUDY.md
docs/CLAIMS.md
docs/EVALUATION.md
```

---

## Example failure mode

For the prompt:

```text
Explain speculative decoding clearly to a startup founder.
Focus on why high token acceptance matters for speed.
```

the small model sometimes confuses speculative decoding with unrelated ideas such as generic text generation, market adoption, crypto tokens, or blockchain transactions.

The verifier detects these failures and repairs the answer using the correct technical structure:

```text
draft model proposes text tokens
target model verifies proposed tokens
accepted tokens are appended
rejected tokens cause fallback
high acceptance reduces expensive target-model calls
```

---

## What this project claims

This project currently supports the following limited claim:

> In a pilot benchmark using Qwen2.5-0.5B-Instruct on 20 prompts, Generate–Verify–Repair improved the internal verifier score over greedy decoding on most prompts.

---

## What this project does not claim

This project does **not** claim that it:

- makes any model generally intelligent
- guarantees better answers for arbitrary prompts
- beats frontier systems such as ChatGPT or Claude
- proves human-preference improvement
- eliminates hallucinations
- solves factuality without task-specific verification

The current prototype should be understood as an inference-time reliability wrapper for small language models.

---

## Repository structure

```text
verify-repair-lm/
├── README.md
├── requirements.txt
├── scripts/
│   ├── prompt_ensemble_competition.py
│   └── prompt_ensemble_benchmark.py
├── docs/
│   ├── STUDY.md
│   ├── CLAIMS.md
│   ├── EVALUATION.md
│   ├── RELATED_WORK.md
│   └── ROADMAP.md
├── evidence/
│   └── human_eval_template.csv
├── prompts/
│   └── default_prompts.jsonl
└── examples/
    └── spec_decode_example.md
```

---

## Next milestone

The next milestone is human evaluation.

For each prompt, compare:

```text
greedy answer
vs
final Generate–Verify–Repair answer
```

and label the result as one of:

```text
greedy_better
final_better
tie
both_bad
```

The included template is:

```text
evidence/human_eval_template.csv
```

The project becomes stronger if the internal verifier score correlates with human preference.

---

## Related ideas

This project is related to:

- Best-of-N generation
- self-consistency
- verifier reranking
- Self-Refine
- Reflexion
- Tree of Thoughts
- inference-time scaling

The practical focus here is narrower:

```text
small model
+ prompt ensemble
+ task-specific verifier
+ repair module
+ audit trail
```

---

## License

