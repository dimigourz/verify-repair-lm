# Verify-Repair LM

Inference-time Generate–Verify–Repair wrapper for improving small language-model outputs.

The pipeline:

```text
user prompt
  -> prompt variants
  -> candidate generation
  -> verifier scoring
  -> repair/synthesis if needed
  -> final selected answer
```

## Install

```bash
pip install -r requirements.txt
```

## Run one example

```bash
python scripts/prompt_ensemble_competition.py
```

## Run benchmark

```bash
python scripts/prompt_ensemble_benchmark.py
```

Run first 5 prompts only:

```bash
MAX_PROMPTS=5 python scripts/prompt_ensemble_benchmark.py
```

## Evidence

Pilot benchmark with Qwen2.5-0.5B-Instruct on 20 prompts:

- mean greedy score: 33.45
- mean final score: 49.55
- final win rate vs greedy by internal verifier score: 90%

These are internal verifier-score results, not human-preference results yet.

See `docs/STUDY.md`, `docs/CLAIMS.md`, and `docs/EVALUATION.md`.
