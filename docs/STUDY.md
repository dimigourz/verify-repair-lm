# Study: Generate–Verify–Repair for Small Language Models

## 1. Motivation

Small language models are attractive for deployment because they are cheaper and easier to self-host. However, single-shot greedy decoding often produces answers that are incomplete, repetitive, off-topic, or semantically wrong.

The central question of this pilot study is:

> Can an inference-time wrapper improve the answers of a small language model without changing the model weights?

The proposed wrapper is called **Generate–Verify–Repair**.

Instead of asking the model once, the system generates several candidate answers, verifies them using task-specific checks, repairs failures when possible, and selects a final answer.

## 2. Hypothesis

The working hypothesis is:

> A small language model can produce better final answers when it is wrapped in a prompt-ensemble, verifier-guided, repair-capable inference loop.

More precisely:

```text
small model + prompt ensemble + verifier + repair
```

may outperform:

```text
small model + greedy decoding
```

on a verifier-defined quality metric.

This study does not claim that the method improves human preference yet. Human evaluation remains future work.

## 3. Method

The pipeline has five stages.

### 3.1 User prompt

The system receives the original user request.

Example:

```text
Explain speculative decoding clearly to a startup founder.
Focus on why high token acceptance matters for speed.
```

### 3.2 Prompt ensemble

The original prompt is rewritten into several prompt variants. Each variant emphasizes a different aspect of the task.

For the speculative-decoding task, the five prompt variants were:

1. direct clarification,
2. technical correctness,
3. founder/business framing,
4. risks and mitigations,
5. final structured answer.

The goal is to give the small model several chances to produce a useful answer.

### 3.3 Candidate generation

The base model generates candidate answers for each prompt variant. In the pilot benchmark, the same model was used for all candidates:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The benchmark used one sample per prompt variant.

### 3.4 Verification

Each candidate answer is scored by a verifier. The verifier combines generic checks with task-specific checks.

Generic checks include:

- answer length,
- repetition,
- role leakage,
- code leakage,
- prompt copying,
- basic coverage of prompt keywords.

For the speculative-decoding task, the verifier checks whether the answer correctly explains:

- draft model,
- target model,
- target-model verification,
- accepted text tokens,
- speed mechanism,
- fallback or repair behavior,
- relevant risks.

It also penalizes a known failure mode:

```text
confusing LLM speculative decoding with blockchain or crypto tokens
```

### 3.5 Repair

If the best candidate still fails the verifier, a repair step is triggered.

The pilot implementation used a deterministic repair template for the speculative-decoding task. For other tasks, the repair path can be model-based synthesis.

The final answer is selected by re-scoring all candidates and repair outputs.

## 4. Benchmark setup

The pilot benchmark used the following configuration:

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Number of prompts | 20 |
| Prompt variants | 5 |
| Samples per variant | 1 |
| Max new tokens | 320 |
| Repair mode | automatic |
| Evaluation metric | internal verifier score |
| Hardware | CUDA |
| Seed | 123 |

The prompt set included:

- speculative-decoding explanation tasks,
- generic technical explanation tasks,
- email-writing tasks,
- code-writing tasks.

## 5. Results

Aggregate result:

| Metric | Value |
|---|---:|
| Number of prompts | 20 |
| Mean greedy score | 33.45 |
| Mean ensemble-best score | 46.15 |
| Mean final score | 49.55 |
| Ensemble win rate vs greedy by verifier score | 90% |
| Final win rate vs greedy by verifier score | 90% |
| Final win rate vs ensemble by verifier score | 15% |
| Mean ensemble gain vs greedy | +12.70 |
| Mean final gain vs greedy | +16.10 |
| Mean final gain vs ensemble | +3.40 |
| Repair trigger rate | 15% |
| Mean elapsed time per prompt | 18.74 s |
| Total elapsed time | 374.89 s |

### 5.1 Results by task type

| Task mode | n | Greedy score | Ensemble score | Final score | Final gain | Final win rate | Repair trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 2 | 53.04 | 55.07 | 55.07 | +2.03 | 100% | 0% |
| email | 2 | 45.99 | 52.99 | 52.99 | +7.00 | 100% | 0% |
| generic | 13 | 34.74 | 47.91 | 47.91 | +13.17 | 84.6% | 0% |
| spec_decode | 3 | 6.49 | 28.05 | 50.70 | +44.22 | 100% | 100% |

## 6. Interpretation

The benchmark suggests three observations.

### 6.1 Prompt ensembling helped in most cases

The ensemble-best answer improved over greedy decoding on 90% of prompts according to the internal verifier score.

This suggests that prompt variation alone can help a small model escape poor single-prompt behavior.

### 6.2 Repair was important for the hardest domain-specific failure

The repair trigger rate was 15% overall, but 100% for the speculative-decoding tasks. This means the largest improvement came from a task-specific verifier and repair template.

This is important: the repair mechanism works best when the failure mode is known and verifiable.

### 6.3 The result is not yet a human-preference result

The benchmark optimizes an internal verifier score. Since the same verifier selects the final answer and evaluates the final answer, the result is partly circular.

A proper evaluation requires human labels or objective task-specific tests.

## 7. Limitations

The current prototype has several limitations.

### 7.1 Internal verifier bias

The verifier is hand-written. It may reward answers that satisfy its rules without being genuinely better.

### 7.2 Task-specific repair

The strongest result comes from the speculative-decoding repair template. That template is domain-specific and does not automatically generalize.

### 7.3 No human preference labels yet

The current benchmark does not prove that humans prefer the final answers. A human evaluation file is included for the next step.

### 7.4 Extra inference cost

The wrapper is more expensive than greedy decoding because it generates multiple candidates and may run repair.

### 7.5 Small benchmark

The pilot used only 20 prompts. Larger and more diverse evaluation is needed.

## 8. Conclusion

The pilot supports a limited but useful conclusion:

> A prompt-ensemble, verifier-guided, repair-capable wrapper can improve internal verifier scores over greedy decoding for a small language model.

The strongest result appears when the verifier captures a clear, domain-specific failure mode and the repair step can correct it.

The next necessary step is human evaluation and objective task-level evaluation.
