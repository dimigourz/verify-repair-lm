# Related Work

This project is related to inference-time scaling and test-time compute for language models.

## Self-consistency

Self-consistency samples multiple reasoning paths and selects the most consistent answer instead of relying on greedy decoding. It demonstrated that multiple sampled outputs can improve reasoning performance when combined with a selection rule.

Reference:

```text
Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. 2022.
https://arxiv.org/abs/2203.11171
```

## Tree of Thoughts

Tree of Thoughts explores multiple intermediate reasoning paths and evaluates them during inference. It is related to this project because it moves beyond one-shot left-to-right decoding and introduces search and evaluation over larger semantic units.

Reference:

```text
Yao et al. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. 2023.
https://arxiv.org/abs/2305.10601
```

## Self-Refine

Self-Refine generates an initial answer, produces feedback, and refines the answer iteratively. This is closely related to the repair part of Generate–Verify–Repair.

Reference:

```text
Madaan et al. Self-Refine: Iterative Refinement with Self-Feedback. 2023.
https://arxiv.org/abs/2303.17651
```

## Reflexion

Reflexion uses verbal feedback to improve future decisions without updating model weights. It is related to verifier-guided improvement and language-based feedback loops.

Reference:

```text
Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. 2023.
https://arxiv.org/abs/2303.11366
```

## Best-of-N and verifier reranking

The present project is closest to Best-of-N generation with verifier-based reranking:

```text
generate N candidates
score candidates
select best candidate
repair if needed
```

The difference is that this project emphasizes small or self-hosted models, prompt variants, hand-checkable task verifiers, and explicit repair modules.

## Positioning

The project should not be positioned as a new foundation model or a generic reasoning breakthrough.

It should be positioned as:

```text
an inference-time reliability wrapper for small language models
```
