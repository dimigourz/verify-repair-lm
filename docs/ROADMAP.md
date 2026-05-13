# Roadmap

## Current state

The current prototype supports:

- prompt variants,
- candidate generation,
- verifier scoring,
- repair for known failure modes,
- benchmark reporting,
- internal verifier-score comparison.

## Next steps

### 1. Human evaluation

Create a manually labeled comparison between greedy and final answers.

Target:

```text
at least 50 prompts
at least 2 human labels per prompt
report final_better / greedy_better / tie / both_bad
```

### 2. Replace hard-coded repair templates

The current speculative-decoding repair is deterministic and task-specific. Future versions should support:

- modular repair templates,
- model-based repair,
- retrieval-grounded repair,
- domain-specific repair policies.

### 3. Add objective task verifiers

The strongest domains are those with objective checks:

- code,
- data extraction,
- citation-grounded factual QA,
- customer-support policy compliance.

### 4. Report cost and latency

The wrapper improves verifier score but increases compute.

Future evaluation should report:

- total generation calls,
- latency per prompt,
- GPU memory use,
- cost per accepted final answer,
- quality gain per additional generation.

### 5. Add a verifier-correlation study

Measure whether internal verifier scores correlate with human preferences.

This is necessary before making stronger claims.

## Long-term direction

The long-term goal is a modular system:

```text
small LM
  + prompt ensemble
  + candidate generation
  + domain verifier
  + repair module
  + audit trail
```

The strongest product direction is not a generic “thinking” wrapper. It is a reliable, domain-specific verification and repair layer for small/self-hosted models.
