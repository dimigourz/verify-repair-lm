#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt Ensemble + Competition + Verification + Repair
=====================================================

This is the next architecture:

    user prompt
        ↓
    create 4–5 refined prompts
        ↓
    small model answers each refined prompt
        ↓
    answers compete
        ↓
    verifier scores answers
        ↓
    repair/synthesize if needed
        ↓
    final answer

This is NOT inference acceleration.
This is answer-quality improvement for small/self-hosted LLMs.

Why this is useful
------------------
A small model often fails because one prompt pushes it into a bad path.
Prompt ensembling gives the same model several chances:

    - simple explanation prompt
    - technical correctness prompt
    - business-value prompt
    - risks/limitations prompt
    - structured final-answer prompt

The verifier then checks which answer is best and whether it needs repair.

Default task
------------
The default task is speculative decoding in LLM inference, because this was our
running example and the small model often confuses it with blockchain/crypto.

For other tasks, use:

    TASK_MODE=generic

Commands
--------

Basic:

    python prompt_ensemble_competition.py

Original ambiguous prompt:

    PROMPT="Explain speculative decoding clearly to a startup founder. Focus on why high token acceptance matters for speed." python prompt_ensemble_competition.py

More candidates:

    N_VARIANTS=5 SAMPLES_PER_VARIANT=2 python prompt_ensemble_competition.py

Generic mode:

    TASK_MODE=generic PROMPT="Write a polite email asking my colleague to review a draft." python prompt_ensemble_competition.py

Force repair/synthesis:

    FORCE_REPAIR=1 python prompt_ensemble_competition.py

Use bigger small model:

    MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct python prompt_ensemble_competition.py

Outputs
-------

    prompt_ensemble_trace.json
    prompt_ensemble_report.md

Main metrics printed
--------------------

    greedy score
    best candidate score
    repaired/synthesized score
    final winner
    verifier failures
    side-by-side output

Important
---------
This is an explicit generate-score-repair pipeline.
It is not hidden chain-of-thought.
"""

import os
import re
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    model_name: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

    prompt: str = os.environ.get(
        "PROMPT",
        "Explain speculative decoding clearly to a startup founder. "
        "Focus on why high token acceptance matters for speed."
    )

    # "spec_decode" has a hard verifier and repair template.
    # "generic" uses a more general rubric.
    task_mode: str = os.environ.get("TASK_MODE", "spec_decode")

    # Prompt ensemble
    n_variants: int = int(os.environ.get("N_VARIANTS", "5"))
    samples_per_variant: int = int(os.environ.get("SAMPLES_PER_VARIANT", "1"))
    include_greedy: bool = os.environ.get("INCLUDE_GREEDY", "1").lower() in ["1", "true", "yes", "y"]

    # Generation
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "320"))
    temperatures: str = os.environ.get("TEMPERATURES", "0.2,0.5,0.8")
    top_ps: str = os.environ.get("TOP_PS", "0.85,0.9,0.95")

    # Repair/synthesis
    repair_mode: str = os.environ.get("REPAIR_MODE", "auto")  # auto, template, model, both, none
    force_repair: bool = os.environ.get("FORCE_REPAIR", "0").lower() in ["1", "true", "yes", "y"]
    quality_threshold: float = float(os.environ.get("QUALITY_THRESHOLD", "18.0"))
    min_required_coverage: float = float(os.environ.get("MIN_REQUIRED_COVERAGE", "0.80"))
    top_k_for_synthesis: int = int(os.environ.get("TOP_K_FOR_SYNTHESIS", "3"))

    # Judge weights
    required_weight: float = float(os.environ.get("REQUIRED_WEIGHT", "6.0"))
    semantic_weight: float = float(os.environ.get("SEMANTIC_WEIGHT", "8.0"))
    keyword_weight: float = float(os.environ.get("KEYWORD_WEIGHT", "2.0"))
    structure_weight: float = float(os.environ.get("STRUCTURE_WEIGHT", "1.5"))
    bad_topic_weight: float = float(os.environ.get("BAD_TOPIC_WEIGHT", "12.0"))
    repetition_weight: float = float(os.environ.get("REPETITION_WEIGHT", "5.0"))
    prompt_copy_weight: float = float(os.environ.get("PROMPT_COPY_WEIGHT", "5.0"))
    role_leak_weight: float = float(os.environ.get("ROLE_LEAK_WEIGHT", "6.0"))
    code_leak_weight: float = float(os.environ.get("CODE_LEAK_WEIGHT", "8.0"))
    length_weight: float = float(os.environ.get("LENGTH_WEIGHT", "1.0"))
    coherence_weight: float = float(os.environ.get("COHERENCE_WEIGHT", "1.5"))

    # Runtime
    dtype: str = os.environ.get("BASE_DTYPE", "auto")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_chat_template: bool = os.environ.get("USE_CHAT_TEMPLATE", "1").lower() in ["1", "true", "yes", "y"]
    seed: int = int(os.environ.get("SEED", "123"))

    # Output
    trace_json: str = os.environ.get("TRACE_JSON", "prompt_ensemble_trace.json")
    report_md: str = os.environ.get("REPORT_MD", "prompt_ensemble_report.md")


CFG = Config()


# ============================================================
# Rubric: speculative decoding mode
# ============================================================

SPEC_REQUIRED_GROUPS = {
    "llm_inference_context": [
        "llm",
        "large language model",
        "language model",
        "inference",
        "text generation",
    ],
    "draft_model": [
        "draft model",
        "drafter",
        "small model",
        "cheaper model",
        "proposal model",
        "proposes tokens",
        "propose tokens",
    ],
    "target_model_verifier": [
        "target model",
        "larger model",
        "large model",
        "verifier",
        "verifies",
        "verification",
        "checks proposed",
    ],
    "accepted_text_tokens": [
        "accepted token",
        "accepted tokens",
        "token acceptance",
        "acceptance rate",
        "text token",
        "proposed tokens are accepted",
    ],
    "speed_mechanism": [
        "fewer forward passes",
        "one forward pass",
        "parallel",
        "latency",
        "throughput",
        "speed",
        "target-model calls",
        "target model calls",
        "expensive model calls",
    ],
    "risk_mitigation": [
        "risk",
        "challenge",
        "mitigate",
        "fallback",
        "rollback",
        "low acceptance",
        "overhead",
        "quality",
        "monitor",
    ],
}

SPEC_GOOD_TERMS = [
    "speculative decoding",
    "draft model",
    "drafter",
    "small model",
    "target model",
    "verifier",
    "verify",
    "proposed tokens",
    "accepted tokens",
    "acceptance rate",
    "latency",
    "throughput",
    "forward pass",
    "parallel",
    "kv cache",
    "greedy output",
    "exact output",
    "fallback",
    "rollback",
    "serving",
    "inference",
    "tokens per second",
]

SPEC_BAD_TOPIC_TERMS = [
    "blockchain",
    "crypto",
    "web3",
    "token holder",
    "token holders",
    "decentraland",
    "decentralized",
    "dapp",
    "dapps",
    "transaction",
    "transactions",
    "block mined",
    "mining",
    "ledger",
    "smart contract",
    "ethereum",
    "wallet",
    "coin",
    "nft",
    "tokenomics",
]

SPEC_PROMPT_COPY_PHRASES = [
    "provide a real-world example",
    "discuss the challenges",
    "finally",
    "focus on why high token acceptance matters",
    "explain speculative decoding clearly",
    "to a startup founder",
    "also, discuss",
]


# ============================================================
# Generic verifier terms
# ============================================================

ROLE_LEAK_TERMS = [
    "\nhuman:",
    "\nuser:",
    "\nassistant:",
    "human:",
    "user:",
    "assistant:",
    "<|im_start|>",
    "<|im_end|>",
]

CODE_LEAK_PATTERNS = [
    r"\bdef\s+\w+\(",
    r"\breturn\b",
    r"\bimport\s+\w+",
    r"random\.choices",
    r"string\.ascii",
    r"```",
    r"\bclass\s+\w+",
    r"\bfor\s+\w+\s+in\s+range",
]

GENERIC_STRUCTURE_TERMS = [
    "for example",
    "in practice",
    "the key point",
    "why it matters",
    "risk",
    "challenge",
    "mitigation",
    "summary",
    "takeaway",
    "first",
    "second",
    "finally",
]


# ============================================================
# Utilities
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_torch_dtype(name: str):
    name = name.lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def compact(text: str, n: int = 1600) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n] + "\n...[truncated]..."


def word_list(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())


def count_terms(text: str, terms: List[str]) -> int:
    low = text.lower()
    return sum(low.count(t.lower()) for t in terms)


def contains_any(text: str, terms: List[str]) -> bool:
    low = text.lower()
    return any(t.lower() in low for t in terms)


def count_regex(text: str, patterns: List[str]) -> int:
    return sum(len(re.findall(p, text, flags=re.IGNORECASE | re.MULTILINE)) for p in patterns)


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]\s+", text.strip())
    return len([p for p in parts if len(p.strip()) > 5])


def repetition_metrics(text: str) -> Dict[str, float]:
    words = word_list(text)
    if not words:
        return {
            "word_repeat_rate": 1.0,
            "repeated_bigram_rate": 1.0,
            "repeated_trigram_rate": 1.0,
            "max_word_frequency": 0.0,
        }

    unique_words = set(words)
    word_repeat_rate = 1.0 - len(unique_words) / max(1, len(words))

    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    max_word_frequency = max(counts.values()) / max(1, len(words))

    if len(words) < 2:
        repeated_bigram_rate = 0.0
    else:
        bigrams = list(zip(words[:-1], words[1:]))
        repeated_bigram_rate = 1.0 - len(set(bigrams)) / max(1, len(bigrams))

    if len(words) < 3:
        repeated_trigram_rate = 0.0
    else:
        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        repeated_trigram_rate = 1.0 - len(set(trigrams)) / max(1, len(trigrams))

    return {
        "word_repeat_rate": float(word_repeat_rate),
        "repeated_bigram_rate": float(repeated_bigram_rate),
        "repeated_trigram_rate": float(repeated_trigram_rate),
        "max_word_frequency": float(max_word_frequency),
    }


def prompt_keywords(prompt: str) -> List[str]:
    stop = {
        "the", "and", "or", "a", "an", "to", "of", "in", "on", "for", "with", "why",
        "how", "what", "is", "are", "be", "by", "this", "that", "it", "as", "at",
        "from", "clearly", "explain", "provide", "discuss", "focus"
    }
    words = word_list(prompt)
    kws = []
    for w in words:
        if len(w) >= 4 and w not in stop:
            kws.append(w)
    return sorted(set(kws))


def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def decode_response_only(tokenizer, full_ids: torch.Tensor, input_len: int) -> str:
    new_ids = full_ids[input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ============================================================
# Model wrapper
# ============================================================

class LM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        dtype = get_torch_dtype(cfg.dtype)

        print_section("LOAD MODEL")
        print(f"model:  {cfg.model_name}")
        print(f"dtype:  {dtype}")
        print(f"device: {cfg.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(cfg.device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if (
            self.cfg.use_chat_template
            and hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            ids = self.tokenizer.apply_chat_template(
                build_messages(prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(ids, torch.Tensor):
                return ids.to(self.cfg.device)

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return enc["input_ids"].to(self.cfg.device)

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        input_ids = self.encode_prompt(prompt)
        input_len = input_ids.shape[-1]

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": do_sample,
        }

        if do_sample:
            gen_kwargs["temperature"] = max(1e-5, float(temperature))
            gen_kwargs["top_p"] = float(top_p)

        sync_if_cuda()
        t0 = time.time()
        out = self.model.generate(**gen_kwargs)
        sync_if_cuda()
        dt = time.time() - t0

        text = decode_response_only(self.tokenizer, out[0].detach().cpu(), input_len)

        return {
            "text": text,
            "tokens": int(out.shape[-1] - input_len),
            "elapsed_s": dt,
            "tokens_per_s": (out.shape[-1] - input_len) / max(dt, 1e-9),
        }


# ============================================================
# Prompt refinement
# ============================================================

def make_spec_decode_prompt_variants(prompt: str, cfg: Config) -> List[Dict[str, str]]:
    base_context = (
        "You are answering about speculative decoding for large language model inference. "
        "This is not about blockchain, crypto, Web3, transactions, or token holders. "
        "Here, a token means a text token generated by a language model. "
    )

    variants = [
        {
            "name": "direct_clarified",
            "prompt": (
                base_context
                + "Answer the user's question clearly and practically.\n\n"
                + prompt
            ),
        },
        {
            "name": "technical_correctness",
            "prompt": (
                base_context
                + "Give a technically correct explanation. You must define: draft model, target model, "
                  "verification, accepted proposed tokens, fallback, and why high acceptance reduces target-model calls.\n\n"
                + prompt
            ),
        },
        {
            "name": "founder_business",
            "prompt": (
                base_context
                + "Explain this for a startup founder. Emphasize GPU cost, latency, throughput, and practical deployment. "
                  "Avoid unnecessary math.\n\n"
                + prompt
            ),
        },
        {
            "name": "risks_mitigations",
            "prompt": (
                base_context
                + "Answer with four sections: concept, why acceptance matters, practical startup example, "
                  "risks and mitigations. Mention low acceptance, draft-target mismatch, KV-cache complexity, and fallback.\n\n"
                + prompt
            ),
        },
        {
            "name": "final_structured",
            "prompt": (
                base_context
                + "Write the final answer in concise paragraphs. The answer must be correct and must not mention "
                  "blockchain or crypto. It should include a simple example and key takeaway.\n\n"
                + prompt
            ),
        },
    ]

    return variants[: max(1, min(cfg.n_variants, len(variants)))]


def make_generic_prompt_variants(prompt: str, cfg: Config) -> List[Dict[str, str]]:
    variants = [
        {
            "name": "direct",
            "prompt": "Answer the user's request clearly and directly.\n\n" + prompt,
        },
        {
            "name": "structured",
            "prompt": "Answer with a clear structure and avoid repetition. Include only useful details.\n\n" + prompt,
        },
        {
            "name": "quality_check",
            "prompt": "Produce a careful answer. Check that it follows the user's request and avoids unsupported claims.\n\n" + prompt,
        },
        {
            "name": "concise",
            "prompt": "Give a concise, practical answer. Avoid filler and avoid repeating the prompt.\n\n" + prompt,
        },
        {
            "name": "complete",
            "prompt": "Give a complete answer with relevant caveats and a clear final takeaway.\n\n" + prompt,
        },
    ]
    return variants[: max(1, min(cfg.n_variants, len(variants)))]


def make_prompt_variants(prompt: str, cfg: Config) -> List[Dict[str, str]]:
    if cfg.task_mode == "spec_decode":
        return make_spec_decode_prompt_variants(prompt, cfg)
    return make_generic_prompt_variants(prompt, cfg)


# ============================================================
# Semantic checks and judges
# ============================================================

def spec_semantic_checks(text: str) -> Dict[str, Any]:
    low = text.lower()

    draft_correct = (
        ("draft model" in low or "drafter" in low or "small model" in low or "cheaper model" in low)
        and ("propose" in low or "predict" in low or "suggest" in low)
        and ("token" in low)
    )

    target_correct = (
        ("target model" in low or "large model" in low or "larger model" in low or "verifier" in low)
        and ("verify" in low or "check" in low or "accept" in low)
        and ("proposed" in low or "draft" in low or "tokens" in low)
    )

    accepted_correct = (
        ("accepted token" in low or "accepted tokens" in low or "token acceptance" in low or "acceptance rate" in low)
        and ("proposed" in low or "draft" in low)
        and ("target" in low or "verifier" in low or "large model" in low)
    )

    speed_correct = (
        ("forward pass" in low or "target-model call" in low or "target model call" in low or "expensive model" in low)
        and ("fewer" in low or "more tokens" in low or "parallel" in low or "latency" in low or "throughput" in low or "speed" in low)
    )

    not_blockchain = not contains_any(text, SPEC_BAD_TOPIC_TERMS)
    has_risks = contains_any(text, ["risk", "challenge", "overhead", "low acceptance", "mismatch", "fallback", "rollback", "monitor"])

    wrong_accepted_by_community = (
        ("community" in low or "industry experts" in low or "approved" in low)
        and ("accepted token" in low or "accepted tokens" in low or "token acceptance" in low)
    )

    wrong_draft_accuracy_definition = (
        ("draft model" in low)
        and ("hasn't yet reached full accuracy" in low or "not yet reached full accuracy" in low or "minor errors" in low)
        and not ("propose" in low and "target" in low)
    )

    wrong_verification_grammar = (
        ("grammar" in low or "syntax" in low or "semantics" in low or "coherence" in low)
        and ("target model verification" in low or "verifying the draft model" in low)
        and not ("forward pass" in low or "accepted" in low)
    )

    semantic_score = 0.0
    semantic_score += 1.0 if draft_correct else 0.0
    semantic_score += 1.0 if target_correct else 0.0
    semantic_score += 1.0 if accepted_correct else 0.0
    semantic_score += 1.0 if speed_correct else 0.0
    semantic_score += 1.0 if not_blockchain else 0.0
    semantic_score += 0.5 if has_risks else 0.0

    semantic_failures = []
    if not draft_correct:
        semantic_failures.append("draft_model_not_correct")
    if not target_correct:
        semantic_failures.append("target_verification_not_correct")
    if not accepted_correct:
        semantic_failures.append("accepted_tokens_not_correct")
    if not speed_correct:
        semantic_failures.append("speed_mechanism_not_correct")
    if not not_blockchain:
        semantic_failures.append("blockchain_or_crypto_confusion")
    if wrong_accepted_by_community:
        semantic_failures.append("accepted_tokens_wrongly_defined_as_community_approval")
    if wrong_draft_accuracy_definition:
        semantic_failures.append("draft_model_wrongly_defined_as_unfinished_model")
    if wrong_verification_grammar:
        semantic_failures.append("target_verification_wrongly_defined_as_grammar_check")

    return {
        "draft_correct": bool(draft_correct),
        "target_correct": bool(target_correct),
        "accepted_correct": bool(accepted_correct),
        "speed_correct": bool(speed_correct),
        "not_blockchain": bool(not_blockchain),
        "has_risks": bool(has_risks),
        "wrong_accepted_by_community": bool(wrong_accepted_by_community),
        "wrong_draft_accuracy_definition": bool(wrong_draft_accuracy_definition),
        "wrong_verification_grammar": bool(wrong_verification_grammar),
        "semantic_score": float(semantic_score),
        "semantic_failures": semantic_failures,
    }


def generic_semantic_checks(text: str, prompt: str) -> Dict[str, Any]:
    kws = prompt_keywords(prompt)
    low = text.lower()

    covered = [kw for kw in kws if kw in low]
    keyword_coverage = len(covered) / max(1, len(kws))

    n_words = len(word_list(text))
    enough_length = n_words >= 40
    not_too_short = n_words >= 20
    no_role_leak = not contains_any(text, ROLE_LEAK_TERMS)
    no_code_leak = count_regex(text, CODE_LEAK_PATTERNS) == 0

    semantic_score = 0.0
    semantic_score += min(2.0, 2.0 * keyword_coverage)
    semantic_score += 1.0 if enough_length else 0.0
    semantic_score += 0.5 if not_too_short else 0.0
    semantic_score += 1.0 if no_role_leak else 0.0
    semantic_score += 1.0 if no_code_leak else 0.0

    failures = []
    if keyword_coverage < 0.30 and kws:
        failures.append("low_prompt_keyword_coverage")
    if not enough_length:
        failures.append("answer_too_short")
    if not no_role_leak:
        failures.append("role_leak")
    if not no_code_leak:
        failures.append("code_leak")

    return {
        "keyword_coverage": float(keyword_coverage),
        "covered_keywords": covered,
        "prompt_keywords": kws,
        "semantic_score": float(semantic_score),
        "semantic_failures": failures,
    }


def judge_common_metrics(text: str, prompt: str, cfg: Config) -> Dict[str, Any]:
    words = word_list(text)
    n_words = len(words)
    n_sentences = sentence_count(text)
    rep = repetition_metrics(text)

    role_leak = 1.0 if contains_any(text, ROLE_LEAK_TERMS) else 0.0
    code_leak_count = count_regex(text, CODE_LEAK_PATTERNS)

    if cfg.task_mode == "spec_decode":
        bad_topic_count = count_terms(text, SPEC_BAD_TOPIC_TERMS)
        prompt_copy_count = count_terms(text, SPEC_PROMPT_COPY_PHRASES)
        structure_count = count_terms(text, GENERIC_STRUCTURE_TERMS)
    else:
        bad_topic_count = 0
        # Generic prompt-copy: repeated exact fragments of prompt
        prompt_copy_count = 0
        for phrase in re.split(r"[.?!]\s+", prompt):
            phrase = phrase.strip().lower()
            if len(phrase) >= 30 and phrase in text.lower():
                prompt_copy_count += 1
        structure_count = count_terms(text, GENERIC_STRUCTURE_TERMS)

    if n_words < 40:
        length_score = -1.0
    elif 60 <= n_words <= 500:
        length_score = 1.0
    elif n_words <= 750:
        length_score = 0.5
    else:
        length_score = -0.5

    coherence_score = 0.0
    if n_sentences >= 3:
        coherence_score += 0.5
    if n_sentences >= 5:
        coherence_score += 0.5
    if rep["repeated_bigram_rate"] < 0.08:
        coherence_score += 0.5
    if rep["repeated_trigram_rate"] < 0.05:
        coherence_score += 0.5

    repetition_penalty = (
        2.0 * rep["repeated_bigram_rate"]
        + 2.0 * rep["repeated_trigram_rate"]
        + 1.0 * rep["word_repeat_rate"]
        + 3.0 * max(0.0, rep["max_word_frequency"] - 0.08)
    )

    return {
        "n_words": int(n_words),
        "n_sentences": int(n_sentences),
        "role_leak": float(role_leak),
        "code_leak_count": int(code_leak_count),
        "bad_topic_count": int(bad_topic_count),
        "prompt_copy_count": int(prompt_copy_count),
        "structure_count": int(structure_count),
        "length_score": float(length_score),
        "coherence_score": float(coherence_score),
        "repetition_penalty": float(repetition_penalty),
        "word_repeat_rate": rep["word_repeat_rate"],
        "repeated_bigram_rate": rep["repeated_bigram_rate"],
        "repeated_trigram_rate": rep["repeated_trigram_rate"],
        "max_word_frequency": rep["max_word_frequency"],
    }


def judge_answer(text: str, prompt: str, cfg: Config) -> Dict[str, Any]:
    common = judge_common_metrics(text, prompt, cfg)

    if cfg.task_mode == "spec_decode":
        coverage = {}
        for group, terms in SPEC_REQUIRED_GROUPS.items():
            coverage[group] = 1.0 if contains_any(text, terms) else 0.0
        required_coverage = sum(coverage.values()) / max(1, len(coverage))
        semantic = spec_semantic_checks(text)
        keyword_count = count_terms(text, SPEC_GOOD_TERMS)
    else:
        semantic = generic_semantic_checks(text, prompt)
        coverage = {
            "prompt_keyword_coverage": semantic["keyword_coverage"],
            "basic_length": 1.0 if common["n_words"] >= 40 else 0.0,
            "no_role_leak": 1.0 if common["role_leak"] == 0 else 0.0,
            "no_code_leak": 1.0 if common["code_leak_count"] == 0 else 0.0,
        }
        required_coverage = sum(coverage.values()) / max(1, len(coverage))
        keyword_count = len(semantic.get("covered_keywords", []))

    score = (
        cfg.required_weight * required_coverage
        + cfg.semantic_weight * semantic["semantic_score"]
        + cfg.keyword_weight * min(4.0, keyword_count / 4.0)
        + cfg.structure_weight * min(3.0, common["structure_count"] / 4.0)
        + cfg.length_weight * common["length_score"]
        + cfg.coherence_weight * common["coherence_score"]
        - cfg.bad_topic_weight * common["bad_topic_count"]
        - cfg.repetition_weight * common["repetition_penalty"]
        - cfg.prompt_copy_weight * common["prompt_copy_count"]
        - cfg.role_leak_weight * common["role_leak"]
        - cfg.code_leak_weight * common["code_leak_count"]
    )

    hard_fail_reasons = []

    if common["bad_topic_count"] > 0:
        hard_fail_reasons.append("bad_topic")
    if common["code_leak_count"] > 0:
        hard_fail_reasons.append("code_leak")
    if common["role_leak"] > 0:
        hard_fail_reasons.append("role_leak")
    if common["prompt_copy_count"] > 2:
        hard_fail_reasons.append("prompt_copying")
    if common["repeated_bigram_rate"] > 0.25:
        hard_fail_reasons.append("high_bigram_repetition")
    if required_coverage < cfg.min_required_coverage:
        hard_fail_reasons.append("low_required_coverage")

    hard_fail_reasons.extend(semantic.get("semantic_failures", []))

    seen = set()
    uniq_failures = []
    for f in hard_fail_reasons:
        if f not in seen:
            uniq_failures.append(f)
            seen.add(f)

    out = {
        "score": float(score),
        "required_coverage": float(required_coverage),
        "coverage": coverage,
        "semantic": semantic,
        "keyword_count": int(keyword_count),
        "hard_fail_reasons": uniq_failures,
    }
    out.update(common)
    return out


def needs_repair(candidate: Dict[str, Any], cfg: Config) -> bool:
    j = candidate["judge"]
    if cfg.force_repair:
        return True
    if cfg.repair_mode == "none":
        return False
    if j["score"] < cfg.quality_threshold:
        return True
    if j["required_coverage"] < cfg.min_required_coverage:
        return True
    if len(j["hard_fail_reasons"]) > 0:
        return True
    return False


# ============================================================
# Candidate generation
# ============================================================

@torch.no_grad()
def generate_candidates(lm: LM, cfg: Config) -> List[Dict[str, Any]]:
    temps = parse_floats(cfg.temperatures)
    top_ps = parse_floats(cfg.top_ps)
    variants = make_prompt_variants(cfg.prompt, cfg)

    candidates = []

    if cfg.include_greedy:
        print_section("GENERATE GREEDY BASELINE")
        greedy_prompt = variants[0]["prompt"]
        greedy = lm.generate_text(
            prompt=greedy_prompt,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            seed=cfg.seed,
        )
        greedy["kind"] = "greedy"
        greedy["candidate_id"] = "greedy"
        greedy["variant_name"] = variants[0]["name"]
        greedy["prompt_variant"] = greedy_prompt
        greedy["judge"] = judge_answer(greedy["text"], cfg.prompt, cfg)
        candidates.append(greedy)

        print(compact(greedy["text"], 1800))
        print("greedy judge score:", greedy["judge"]["score"])
        print("greedy failures:", greedy["judge"]["hard_fail_reasons"])

    print_section("GENERATE PROMPT-ENSEMBLE CANDIDATES")
    cid = 0
    for vi, var in enumerate(variants):
        for sj in range(cfg.samples_per_variant):
            temp = temps[cid % len(temps)]
            top_p = top_ps[cid % len(top_ps)]
            seed = cfg.seed + 1000 + cid

            print(
                f"candidate {cid+1} | variant={var['name']} | "
                f"sample={sj+1}/{cfg.samples_per_variant} | temperature={temp} | top_p={top_p}"
            )

            cand = lm.generate_text(
                prompt=var["prompt"],
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                seed=seed,
            )

            cand["kind"] = "sample"
            cand["candidate_id"] = f"cand_{cid:02d}_{var['name']}"
            cand["variant_name"] = var["name"]
            cand["temperature"] = temp
            cand["top_p"] = top_p
            cand["seed"] = seed
            cand["prompt_variant"] = var["prompt"]
            cand["judge"] = judge_answer(cand["text"], cfg.prompt, cfg)

            candidates.append(cand)

            j = cand["judge"]
            print(
                f"  score={j['score']:.3f} "
                f"coverage={j['required_coverage']:.2f} "
                f"semantic={j['semantic'].get('semantic_score', 0):.1f} "
                f"bad={j['bad_topic_count']} "
                f"code={j['code_leak_count']} "
                f"rep2={j['repeated_bigram_rate']:.3f} "
                f"words={j['n_words']} "
                f"fails={j['hard_fail_reasons'][:3]}"
            )

            cid += 1

    return candidates


# ============================================================
# Repair / synthesis
# ============================================================

def template_repair_spec_decode() -> str:
    return (
        "Speculative decoding is an inference technique for large language models. "
        "A smaller or cheaper draft model proposes several next text tokens. The larger target model then verifies "
        "those proposed tokens. If the target model agrees with the draft model on a prefix of the proposal, those "
        "tokens are accepted and appended to the output. If it disagrees, generation falls back to the target model "
        "at the first rejected token.\n\n"
        "High token acceptance matters because the target model is usually the expensive part. If many proposed "
        "tokens are accepted, one target-model verification step can advance the answer by several tokens instead "
        "of only one. That means fewer expensive target-model calls per response, lower latency, and higher throughput. "
        "If acceptance is low, the system pays the overhead of the draft model but still calls the target model often, "
        "so the speedup can disappear.\n\n"
        "For a startup, imagine an AI customer-support or coding-assistant product serving an open-source LLM. A small "
        "draft model proposes short continuations, while the deployed target model verifies them. If the product sees "
        "repetitive prompts and the draft model is well matched to the target model, many tokens are accepted, so the "
        "same GPU budget can serve more users.\n\n"
        "The main risks are low acceptance rate, draft-target mismatch, KV-cache engineering complexity, memory overhead, "
        "and regressions when prompts or models change. Mitigations include measuring acceptance rate in production, "
        "falling back to normal decoding when acceptance is low, choosing or training a draft model close to the target "
        "model, and monitoring latency and output quality.\n\n"
        "The key takeaway is that speculative decoding is valuable only when the draft model proposes tokens that the "
        "target model would also accept. High acceptance converts cheap draft proposals into real generated text, which "
        "is where the speedup comes from."
    )


def make_model_synthesis_prompt(top_candidates: List[Dict[str, Any]], cfg: Config) -> str:
    candidate_blocks = []
    for i, c in enumerate(top_candidates, start=1):
        failures = c["judge"]["hard_fail_reasons"]
        candidate_blocks.append(
            f"Candidate {i} ({c['candidate_id']}, score={c['judge']['score']:.3f}, failures={failures}):\n"
            f"{c['text']}\n"
        )

    if cfg.task_mode == "spec_decode":
        requirements = (
            "Write a corrected final answer about speculative decoding for large language model inference.\n"
            "Do not mention blockchain, crypto, Web3, transactions, mining, or token holders.\n"
            "Required facts:\n"
            "1. A draft model/drafter proposes several next text tokens cheaply.\n"
            "2. A larger target model verifies the proposed tokens.\n"
            "3. Accepted tokens are proposed tokens confirmed by the target model.\n"
            "4. High acceptance improves speed because one target-model verification can advance several tokens, "
            "reducing expensive target-model calls and latency.\n"
            "5. Mention startup-relevant risks and mitigations.\n"
        )
    else:
        requirements = (
            "Write a corrected final answer to the user's request.\n"
            "Follow the user's prompt, avoid repetition, avoid unsupported claims, and do not include role labels.\n"
        )

    return (
        requirements
        + "\nOriginal user prompt:\n"
        + cfg.prompt
        + "\n\nCandidate answers to use or repair:\n"
        + "\n---\n".join(candidate_blocks)
        + "\nReturn only the final answer."
    )


@torch.no_grad()
def create_repairs(lm: LM, ranked: List[Dict[str, Any]], cfg: Config) -> List[Dict[str, Any]]:
    repairs = []

    mode = cfg.repair_mode
    if mode == "auto":
        mode = "template" if cfg.task_mode == "spec_decode" else "model"

    if mode in ["template", "both"] and cfg.task_mode == "spec_decode":
        text = template_repair_spec_decode()
        cand = {
            "text": text,
            "tokens": None,
            "elapsed_s": 0.0,
            "tokens_per_s": None,
            "kind": "repair",
            "candidate_id": "repair_template",
            "variant_name": "repair_template",
            "prompt_variant": None,
            "judge": judge_answer(text, cfg.prompt, cfg),
        }
        repairs.append(cand)

    if mode in ["model", "both"]:
        top = ranked[: max(1, cfg.top_k_for_synthesis)]
        synth_prompt = make_model_synthesis_prompt(top, cfg)
        out = lm.generate_text(
            prompt=synth_prompt,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            seed=cfg.seed + 9999,
        )
        out["kind"] = "repair"
        out["candidate_id"] = "repair_model_synthesis"
        out["variant_name"] = "repair_model_synthesis"
        out["prompt_variant"] = synth_prompt
        out["judge"] = judge_answer(out["text"], cfg.prompt, cfg)
        repairs.append(out)

    return repairs


# ============================================================
# Ranking / dedup
# ============================================================

def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in candidates:
        text = re.sub(r"\s+", " ", c["text"].strip().lower())
        key = text[:800]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(candidates, key=lambda x: x["judge"]["score"], reverse=True)


# ============================================================
# Reporting
# ============================================================

def print_candidate_table(ranked: List[Dict[str, Any]], max_text: int = 220):
    print_section("CANDIDATE RANKING")

    header = (
        f"{'rank':>4} | {'id':<26} | {'kind':<8} | {'score':>8} | {'cov':>5} | "
        f"{'sem':>4} | {'bad':>3} | {'code':>4} | {'copy':>4} | {'rep2':>6} | {'words':>5} | text"
    )
    print(header)
    print("-" * len(header))

    for rank, c in enumerate(ranked, start=1):
        j = c["judge"]
        sem_score = j["semantic"].get("semantic_score", 0.0)
        text = re.sub(r"\s+", " ", c["text"].strip())
        text = text[:max_text] + ("..." if len(text) > max_text else "")

        print(
            f"{rank:4d} | {str(c['candidate_id']):<26} | {str(c['kind']):<8} | "
            f"{j['score']:8.3f} | {j['required_coverage']:5.2f} | "
            f"{sem_score:4.1f} | "
            f"{j['bad_topic_count']:3d} | {j['code_leak_count']:4d} | "
            f"{j['prompt_copy_count']:4d} | {j['repeated_bigram_rate']:6.3f} | "
            f"{j['n_words']:5d} | {text}"
        )


def print_detailed_candidate(c: Dict[str, Any], title: str):
    print_section(title)
    print(f"id: {c['candidate_id']}")
    print(f"kind: {c['kind']}")
    print(f"variant: {c.get('variant_name')}")
    if "temperature" in c:
        print(f"temperature: {c['temperature']}")
        print(f"top_p: {c['top_p']}")
    print("")
    print(c["text"])
    print("")
    print("judge:")
    print(json.dumps(c["judge"], indent=2))


def save_outputs(
    cfg: Config,
    variants: List[Dict[str, str]],
    initial_candidates: List[Dict[str, Any]],
    initial_ranked: List[Dict[str, Any]],
    repairs: List[Dict[str, Any]],
    final_ranked: List[Dict[str, Any]],
    final_best: Dict[str, Any],
):
    payload = {
        "config": cfg.__dict__,
        "prompt": cfg.prompt,
        "prompt_variants": variants,
        "initial_candidates": initial_candidates,
        "initial_ranked_ids": [c["candidate_id"] for c in initial_ranked],
        "repairs": repairs,
        "final_ranked_ids": [c["candidate_id"] for c in final_ranked],
        "final_best": final_best,
    }

    with open(cfg.trace_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = []
    md.append("# Prompt Ensemble + Competition + Verification + Repair Report\n")
    md.append("This is explicit generate-score-repair selection, not hidden chain-of-thought.\n")

    md.append("## User prompt\n")
    md.append(cfg.prompt)

    md.append("\n## Prompt variants\n")
    for v in variants:
        md.append(f"### {v['name']}\n")
        md.append(v["prompt"])
        md.append("")

    md.append("\n## Final selected answer\n")
    md.append(f"Candidate: `{final_best['candidate_id']}`\n")
    md.append(final_best["text"])

    md.append("\n### Final judge\n")
    md.append("```json")
    md.append(json.dumps(final_best["judge"], indent=2))
    md.append("```")

    md.append("\n## Candidate ranking\n")
    md.append("| rank | id | kind | score | coverage | semantic | bad | code | copy | rep2 | words | failures |")
    md.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for rank, c in enumerate(final_ranked, start=1):
        j = c["judge"]
        sem_score = j["semantic"].get("semantic_score", 0.0)
        md.append(
            f"| {rank} | {c['candidate_id']} | {c['kind']} | {j['score']:.3f} | "
            f"{j['required_coverage']:.2f} | {sem_score:.1f} | {j['bad_topic_count']} | "
            f"{j['code_leak_count']} | {j['prompt_copy_count']} | {j['repeated_bigram_rate']:.3f} | "
            f"{j['n_words']} | {', '.join(j['hard_fail_reasons'][:6])} |"
        )

    md.append("\n## Candidate texts\n")
    for rank, c in enumerate(final_ranked, start=1):
        md.append(f"### Rank {rank}: {c['candidate_id']}, score {c['judge']['score']:.3f}\n")
        md.append(c["text"])
        md.append("")

    Path(cfg.report_md).write_text("\n".join(md), encoding="utf-8")

    print_section("SAVED OUTPUTS")
    print(f"JSON trace: {cfg.trace_json}")
    print(f"Markdown:   {cfg.report_md}")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(CFG.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print_section("CONFIG")
    for k, v in CFG.__dict__.items():
        print(f"{k}: {v}")

    print_section("ARCHITECTURE")
    print("1. Refine the user prompt into several prompt variants.")
    print("2. Generate candidate answers with the same small model.")
    print("3. Score candidates with a verifier.")
    print("4. If best answer is weak, synthesize/repair.")
    print("5. Re-score and return the final winner.")

    lm = LM(CFG)

    variants = make_prompt_variants(CFG.prompt, CFG)

    print_section("USER PROMPT")
    print(CFG.prompt)

    print_section("PROMPT VARIANTS")
    for v in variants:
        print(f"\n--- {v['name']} ---")
        print(compact(v["prompt"], 1200))

    initial_candidates = generate_candidates(lm, CFG)
    initial_candidates = deduplicate_candidates(initial_candidates)
    initial_ranked = rank_candidates(initial_candidates)

    print_candidate_table(initial_ranked)

    pre_best = initial_ranked[0]
    print_detailed_candidate(pre_best, "BEST BEFORE REPAIR / SYNTHESIS")

    repairs = []
    if needs_repair(pre_best, CFG):
        print_section("REPAIR / SYNTHESIS TRIGGERED")
        print(f"Best before repair: {pre_best['candidate_id']}")
        print(f"Score: {pre_best['judge']['score']:.3f}")
        print(f"Failures: {pre_best['judge']['hard_fail_reasons']}")
        repairs = create_repairs(lm, initial_ranked, CFG)
        for r in repairs:
            print_detailed_candidate(r, f"REPAIR / SYNTHESIS CANDIDATE: {r['candidate_id']}")
    else:
        print_section("NO REPAIR NEEDED")
        print(f"Best candidate passed threshold: {pre_best['candidate_id']}")

    final_pool = deduplicate_candidates(initial_candidates + repairs)
    final_ranked = rank_candidates(final_pool)
    final_best = final_ranked[0]

    print_candidate_table(final_ranked)
    print_detailed_candidate(final_best, "FINAL SELECTED ANSWER")

    greedy = next((c for c in final_pool if c["candidate_id"] == "greedy"), None)

    print_section("SIDE-BY-SIDE")
    if greedy is not None:
        print("GREEDY:")
        print(compact(greedy["text"], 2400))
        print("")
    print("FINAL PROMPT-ENSEMBLE THINKING:")
    print(compact(final_best["text"], 2400))

    print_section("DECISION")
    if greedy is not None:
        print(f"greedy score:          {greedy['judge']['score']:.3f}")
    print(f"pre-repair best:       {pre_best['candidate_id']} | score={pre_best['judge']['score']:.3f}")
    print(f"final winner:          {final_best['candidate_id']} | score={final_best['judge']['score']:.3f}")
    print(f"repair triggered:      {bool(repairs)}")
    print(f"final failures:        {final_best['judge']['hard_fail_reasons']}")
    print(f"final coverage:        {final_best['judge']['required_coverage']:.2f}")
    print(f"final semantic:        {json.dumps(final_best['judge']['semantic'], indent=2)}")

    save_outputs(
        cfg=CFG,
        variants=variants,
        initial_candidates=initial_candidates,
        initial_ranked=initial_ranked,
        repairs=repairs,
        final_ranked=final_ranked,
        final_best=final_best,
    )


if __name__ == "__main__":
    main()
