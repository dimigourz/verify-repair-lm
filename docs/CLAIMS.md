# Claims and Non-Claims

This document defines what the project currently claims and what it does not claim.

## Supported claim

The project currently supports the following claim:

> In a pilot benchmark using Qwen2.5-0.5B-Instruct on 20 prompts, the Generate–Verify–Repair pipeline achieved a higher internal verifier score than greedy decoding on 90% of prompts.

This claim is based on `evidence/prompt_ensemble_benchmark.json`.

## Stronger but still cautious claim

The project also supports the following cautious interpretation:

> Prompt ensembling and verifier-guided repair can improve the measured quality of small-model outputs when the verifier encodes relevant task constraints.

## Non-claims

The project does not claim that it:

- makes any model generally intelligent,
- makes any model “think” in a human sense,
- guarantees better answers on arbitrary prompts,
- improves human preference across all domains,
- beats frontier models,
- removes hallucinations,
- solves factuality without domain-specific verification,
- works without careful verifier design.

## Recommended public wording

Use:

```text
Generate–Verify–Repair is an inference-time wrapper that improves small-model outputs by generating multiple candidate answers, verifying them against task-specific checks, and repairing detected failures.
```

Avoid:

```text
Make any model think.
```

Avoid:

```text
Beats ChatGPT.
```

Avoid:

```text
Guarantees correctness.
```

## Evidence level

Current evidence level:

```text
pilot benchmark with internal verifier
```

Required next evidence level:

```text
human preference evaluation
```

Stronger future evidence level:

```text
domain-specific objective tests, such as unit tests for code or citation checks for factual QA
```
