# Evaluation

## Current evaluation

The current evaluation is based on a 20-prompt pilot benchmark.

The benchmark compares:

1. greedy decoding,
2. prompt-ensemble best candidate,
3. prompt-ensemble plus verifier-guided repair.

The metric is an internal verifier score.

## Results

| Metric | Value |
|---|---:|
| Number of prompts | 20 |
| Mean greedy score | 33.45 |
| Mean ensemble-best score | 46.15 |
| Mean final score | 49.55 |
| Ensemble win rate vs greedy by verifier score | 90% |
| Final win rate vs greedy by verifier score | 90% |
| Final win rate vs ensemble by verifier score | 15% |
| Mean final gain vs greedy | +16.10 |
| Repair trigger rate | 15% |
| Mean elapsed time per prompt | 18.74 s |

## Results by task mode

| Task mode | n | Greedy | Ensemble | Final | Final gain | Repair trigger |
|---|---:|---:|---:|---:|---:|---:|
| code | 2 | 53.04 | 55.07 | 55.07 | +2.03 | 0% |
| email | 2 | 45.99 | 52.99 | 52.99 | +7.00 | 0% |
| generic | 13 | 34.74 | 47.91 | 47.91 | +13.17 | 0% |
| spec_decode | 3 | 6.49 | 28.05 | 50.70 | +44.22 | 100% |

## Interpretation

The benchmark shows that the wrapper improves the internal verifier score. It does not yet show human preference improvement.

The most credible result is the speculative-decoding task, where the model repeatedly failed with incorrect explanations and the task-specific verifier/repair mechanism corrected the output.

## Human evaluation protocol

For each prompt, compare:

```text
greedy answer
final answer
```

Label one of:

```text
greedy_better
final_better
tie
both_bad
```

Suggested criteria:

- factual correctness,
- instruction following,
- completeness,
- clarity,
- absence of repetition,
- absence of off-topic content.

## Human evaluation file

Use:

```text
evidence/human_eval_template.csv
```

Columns:

```text
prompt_id
task_mode
prompt
greedy_answer
final_answer
verifier_winner
human_winner
human_notes
```

## Future objective evaluations

For stronger evidence, add task-specific objective tests.

### Code tasks

- unit tests,
- import checks,
- static analysis,
- runtime checks.

### Factual QA

- retrieval-grounded citation checks,
- contradiction checks,
- required-claim coverage.

### Customer support

- policy compliance,
- forbidden-claim detection,
- tone checks,
- escalation rules.

### Scientific writing

- required-concept coverage,
- notation consistency,
- citation support,
- contradiction detection.
