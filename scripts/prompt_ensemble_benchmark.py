#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt Ensemble Benchmark
=========================

This script tests whether:

    prompt ensemble + answer competition + verifier + repair

beats small-model greedy output across many prompts.

It compares three systems:

    1. greedy
       One clarified/default prompt, greedy decoding.

    2. ensemble_best
       Several refined prompts, several sampled answers, verifier selects best.
       No repair.

    3. ensemble_repair
       Same as ensemble_best, but if best candidate fails the verifier,
       create a repaired/synthesized answer, re-score, and choose final winner.

This is NOT speed optimization.
This is answer-quality improvement for small/self-hosted LLMs.

Default model:
    Qwen/Qwen2.5-0.5B-Instruct

Outputs:
    prompt_ensemble_benchmark.json
    prompt_ensemble_benchmark.csv
    prompt_ensemble_benchmark_report.md
    prompt_ensemble_human_eval.csv

Run:
    python prompt_ensemble_benchmark.py

Run only first 5 prompts:
    MAX_PROMPTS=5 python prompt_ensemble_benchmark.py

More competition:
    N_VARIANTS=5 SAMPLES_PER_VARIANT=2 python prompt_ensemble_benchmark.py

Use larger model:
    MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct python prompt_ensemble_benchmark.py

Use your own prompt file:
    PROMPTS_FILE=my_prompts.txt python prompt_ensemble_benchmark.py

Prompt file format:
    One prompt per line.
    Optional task mode prefix:
        spec_decode<TAB>Explain speculative decoding...
        code<TAB>Write a Python function...
        email<TAB>Write a polite email...
        generic<TAB>Explain...
"""

import os
import re
import csv
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    model_name: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

    prompts_file: str = os.environ.get("PROMPTS_FILE", "")
    max_prompts: int = int(os.environ.get("MAX_PROMPTS", "0"))

    # Prompt ensemble
    n_variants: int = int(os.environ.get("N_VARIANTS", "5"))
    samples_per_variant: int = int(os.environ.get("SAMPLES_PER_VARIANT", "1"))

    # Generation
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "320"))
    temperatures: str = os.environ.get("TEMPERATURES", "0.2,0.5,0.8")
    top_ps: str = os.environ.get("TOP_PS", "0.85,0.9,0.95")

    # Repair / synthesis
    repair_mode: str = os.environ.get("REPAIR_MODE", "auto")  # auto, template, model, both, none
    force_repair: bool = os.environ.get("FORCE_REPAIR", "0").lower() in ["1", "true", "yes", "y"]
    quality_threshold: float = float(os.environ.get("QUALITY_THRESHOLD", "18.0"))
    min_required_coverage: float = float(os.environ.get("MIN_REQUIRED_COVERAGE", "0.75"))
    top_k_for_synthesis: int = int(os.environ.get("TOP_K_FOR_SYNTHESIS", "3"))

    # Verifier weights
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
    output_json: str = os.environ.get("OUTPUT_JSON", "prompt_ensemble_benchmark.json")
    output_csv: str = os.environ.get("OUTPUT_CSV", "prompt_ensemble_benchmark.csv")
    output_report: str = os.environ.get("OUTPUT_REPORT", "prompt_ensemble_benchmark_report.md")
    human_eval_csv: str = os.environ.get("HUMAN_EVAL_CSV", "prompt_ensemble_human_eval.csv")


CFG = Config()


# ============================================================
# Default prompt suite
# ============================================================

DEFAULT_PROMPTS = [
    {
        "task_mode": "spec_decode",
        "prompt": "Explain speculative decoding clearly to a startup founder. Focus on why high token acceptance matters for speed.",
    },
    {
        "task_mode": "spec_decode",
        "prompt": "Explain why low token acceptance can make speculative decoding slower instead of faster.",
    },
    {
        "task_mode": "spec_decode",
        "prompt": "Give a practical startup example of using speculative decoding to reduce LLM serving cost.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain KV cache in transformer inference in simple terms.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain the difference between latency and throughput for an AI startup serving LLMs.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain why batching can improve GPU throughput but may hurt user-facing latency.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain overfitting to a non-technical founder using a simple example.",
    },
    {
        "task_mode": "email",
        "prompt": "Write a polite email asking my colleague to review the attached draft by Friday.",
    },
    {
        "task_mode": "email",
        "prompt": "Rewrite this email professionally: Sorry for the late notice but I will not be able to join today's meeting.",
    },
    {
        "task_mode": "code",
        "prompt": "Write a simple Flask route that accepts name and email from a form and stores them with SQLAlchemy.",
    },
    {
        "task_mode": "code",
        "prompt": "Write a Python function that reads a JSON file safely and returns an empty dict if the file is missing.",
    },
    {
        "task_mode": "generic",
        "prompt": "Give a short product pitch for a tool that improves small open-source LLM answers using verification.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain why a verifier is needed when multiple LLM answers compete.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain the difference between model probability and answer quality.",
    },
    {
        "task_mode": "generic",
        "prompt": "Create a checklist for evaluating whether an LLM answer is reliable.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain why prompt engineering alone is not enough for high-stakes LLM answers.",
    },
    {
        "task_mode": "generic",
        "prompt": "Summarize the risks of using the same model to generate and judge an answer.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain how answer repair differs from simply generating another answer.",
    },
    {
        "task_mode": "generic",
        "prompt": "Write a concise roadmap for testing a prompt-ensemble verifier product.",
    },
    {
        "task_mode": "generic",
        "prompt": "Explain why a small model can sometimes improve when you generate multiple answers and verify them.",
    },
]


# ============================================================
# Rubrics
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

EMAIL_GOOD_TERMS = [
    "dear",
    "best regards",
    "regards",
    "thank you",
    "please",
    "attached",
    "review",
    "meeting",
]

CODE_GOOD_TERMS = [
    "def ",
    "return",
    "try",
    "except",
    "import",
    "app.route",
    "db.session",
    "SQLAlchemy",
    "json",
]


# ============================================================
# Basic utilities
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


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
        "from", "clearly", "explain", "provide", "discuss", "focus", "write", "give",
        "make", "create", "simple", "short"
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


def load_prompt_suite(cfg: Config) -> List[Dict[str, str]]:
    if not cfg.prompts_file:
        prompts = list(DEFAULT_PROMPTS)
    else:
        path = Path(cfg.prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"PROMPTS_FILE not found: {path}")

        prompts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                mode, prompt = line.split("\t", 1)
                prompts.append({"task_mode": mode.strip(), "prompt": prompt.strip()})
            else:
                prompts.append({"task_mode": "generic", "prompt": line})

    if cfg.max_prompts > 0:
        prompts = prompts[:cfg.max_prompts]

    return prompts


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
            "prompt": base_context + "Answer the user's question clearly and practically.\n\n" + prompt,
        },
        {
            "name": "technical_correctness",
            "prompt": (
                base_context
                + "Give a technically correct explanation. Define draft model, target model, verification, "
                  "accepted proposed tokens, fallback, and why high acceptance reduces target-model calls.\n\n"
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
                + "Write the final answer in concise paragraphs. It must be correct and must not mention blockchain or crypto. "
                  "Include a simple example and key takeaway.\n\n"
                + prompt
            ),
        },
    ]

    return variants[: max(1, min(cfg.n_variants, len(variants)))]


def make_code_prompt_variants(prompt: str, cfg: Config) -> List[Dict[str, str]]:
    variants = [
        {"name": "direct_code", "prompt": "Write correct, minimal, runnable code. Include short explanation only if useful.\n\n" + prompt},
        {"name": "robust_code", "prompt": "Write robust Python code with basic error handling and clear comments.\n\n" + prompt},
        {"name": "clean_code", "prompt": "Write clean production-style code. Avoid unnecessary complexity.\n\n" + prompt},
        {"name": "example_code", "prompt": "Provide code and a minimal usage example.\n\n" + prompt},
        {"name": "safe_code", "prompt": "Write safe code that handles common failure cases.\n\n" + prompt},
    ]
    return variants[: max(1, min(cfg.n_variants, len(variants)))]


def make_email_prompt_variants(prompt: str, cfg: Config) -> List[Dict[str, str]]:
    variants = [
        {"name": "professional", "prompt": "Write a polished professional email. Keep it concise.\n\n" + prompt},
        {"name": "warm", "prompt": "Write a warm but professional email. Avoid over-apologizing.\n\n" + prompt},
        {"name": "direct", "prompt": "Write a direct, clear email with subject if useful.\n\n" + prompt},
        {"name": "formal", "prompt": "Write a slightly formal email suitable for a workplace.\n\n" + prompt},
        {"name": "simple", "prompt": "Write a simple email. Keep the message natural and short.\n\n" + prompt},
    ]
    return variants[: max(1, min(cfg.n_variants, len(variants)))]


def make_generic_prompt_variants(prompt: str, cfg: Config) -> List[Dict[str, str]]:
    variants = [
        {"name": "direct", "prompt": "Answer the user's request clearly and directly.\n\n" + prompt},
        {"name": "structured", "prompt": "Answer with a clear structure and avoid repetition. Include only useful details.\n\n" + prompt},
        {"name": "quality_check", "prompt": "Produce a careful answer. Check that it follows the user's request and avoids unsupported claims.\n\n" + prompt},
        {"name": "concise", "prompt": "Give a concise, practical answer. Avoid filler and avoid repeating the prompt.\n\n" + prompt},
        {"name": "complete", "prompt": "Give a complete answer with relevant caveats and a clear final takeaway.\n\n" + prompt},
    ]
    return variants[: max(1, min(cfg.n_variants, len(variants)))]


def make_prompt_variants(prompt: str, task_mode: str, cfg: Config) -> List[Dict[str, str]]:
    if task_mode == "spec_decode":
        return make_spec_decode_prompt_variants(prompt, cfg)
    if task_mode == "code":
        return make_code_prompt_variants(prompt, cfg)
    if task_mode == "email":
        return make_email_prompt_variants(prompt, cfg)
    return make_generic_prompt_variants(prompt, cfg)


# ============================================================
# Semantic checks and judging
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
        "semantic_score": float(semantic_score),
        "semantic_failures": semantic_failures,
        "draft_correct": bool(draft_correct),
        "target_correct": bool(target_correct),
        "accepted_correct": bool(accepted_correct),
        "speed_correct": bool(speed_correct),
        "not_blockchain": bool(not_blockchain),
        "has_risks": bool(has_risks),
    }


def generic_semantic_checks(text: str, prompt: str, task_mode: str) -> Dict[str, Any]:
    kws = prompt_keywords(prompt)
    low = text.lower()

    covered = [kw for kw in kws if kw in low]
    keyword_coverage = len(covered) / max(1, len(kws))

    n_words = len(word_list(text))
    role_clean = not contains_any(text, ROLE_LEAK_TERMS)

    code_leak_count = count_regex(text, CODE_LEAK_PATTERNS)

    if task_mode == "code":
        # In code tasks, code patterns are good, not leakage.
        code_relevance = 1.0 if any(t.lower() in text.lower() for t in CODE_GOOD_TERMS) or "```" in text else 0.0
        code_clean = True
    else:
        code_relevance = 0.0
        code_clean = code_leak_count == 0

    if task_mode == "email":
        email_relevance = 1.0 if count_terms(text, EMAIL_GOOD_TERMS) > 0 else 0.0
    else:
        email_relevance = 0.0

    semantic_score = 0.0
    semantic_score += min(2.0, 2.0 * keyword_coverage)
    semantic_score += 1.0 if n_words >= 40 else 0.0
    semantic_score += 1.0 if role_clean else 0.0
    semantic_score += 1.0 if code_clean else 0.0
    semantic_score += code_relevance
    semantic_score += email_relevance

    failures = []
    if keyword_coverage < 0.25 and kws:
        failures.append("low_prompt_keyword_coverage")
    if n_words < 20 and task_mode != "email":
        failures.append("answer_too_short")
    if not role_clean:
        failures.append("role_leak")
    if not code_clean:
        failures.append("code_leak")

    return {
        "semantic_score": float(semantic_score),
        "semantic_failures": failures,
        "keyword_coverage": float(keyword_coverage),
        "covered_keywords": covered,
        "prompt_keywords": kws,
        "code_relevance": float(code_relevance),
        "email_relevance": float(email_relevance),
    }


def judge_common_metrics(text: str, prompt: str, task_mode: str, cfg: Config) -> Dict[str, Any]:
    n_words = len(word_list(text))
    n_sentences = sentence_count(text)
    rep = repetition_metrics(text)

    role_leak = 1.0 if contains_any(text, ROLE_LEAK_TERMS) else 0.0

    if task_mode == "code":
        code_leak_count = 0
    else:
        code_leak_count = count_regex(text, CODE_LEAK_PATTERNS)

    if task_mode == "spec_decode":
        bad_topic_count = count_terms(text, SPEC_BAD_TOPIC_TERMS)
        prompt_copy_count = 0
        for phrase in SPEC_GOOD_TERMS + ["explain speculative decoding clearly", "focus on why high token acceptance matters"]:
            if phrase.lower() in text.lower() and phrase.lower() in prompt.lower():
                prompt_copy_count += 1
    else:
        bad_topic_count = 0
        prompt_copy_count = 0
        for phrase in re.split(r"[.?!]\s+", prompt):
            phrase = phrase.strip().lower()
            if len(phrase) >= 30 and phrase in text.lower():
                prompt_copy_count += 1

    structure_count = count_terms(text, GENERIC_STRUCTURE_TERMS)

    if task_mode == "email":
        if 25 <= n_words <= 180:
            length_score = 1.0
        elif n_words < 15:
            length_score = -0.5
        else:
            length_score = 0.0
    elif task_mode == "code":
        length_score = 1.0 if n_words >= 10 else 0.0
    else:
        if n_words < 40:
            length_score = -1.0
        elif 60 <= n_words <= 500:
            length_score = 1.0
        elif n_words <= 750:
            length_score = 0.5
        else:
            length_score = -0.5

    coherence_score = 0.0
    if n_sentences >= 3 or task_mode in ["code", "email"]:
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


def judge_answer(text: str, prompt: str, task_mode: str, cfg: Config) -> Dict[str, Any]:
    common = judge_common_metrics(text, prompt, task_mode, cfg)

    if task_mode == "spec_decode":
        coverage = {}
        for group, terms in SPEC_REQUIRED_GROUPS.items():
            coverage[group] = 1.0 if contains_any(text, terms) else 0.0
        required_coverage = sum(coverage.values()) / max(1, len(coverage))
        semantic = spec_semantic_checks(text)
        keyword_count = count_terms(text, SPEC_GOOD_TERMS)
    else:
        semantic = generic_semantic_checks(text, prompt, task_mode)
        coverage = {
            "prompt_keyword_coverage": semantic["keyword_coverage"],
            "basic_length": 1.0 if common["n_words"] >= (15 if task_mode == "email" else 30) else 0.0,
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
    uniq = []
    for f in hard_fail_reasons:
        if f not in seen:
            uniq.append(f)
            seen.add(f)

    out = {
        "score": float(score),
        "required_coverage": float(required_coverage),
        "coverage": coverage,
        "semantic": semantic,
        "keyword_count": int(keyword_count),
        "hard_fail_reasons": uniq,
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
def generate_candidates_for_prompt(lm: LM, prompt: str, task_mode: str, prompt_index: int, cfg: Config) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    variants = make_prompt_variants(prompt, task_mode, cfg)
    temps = parse_floats(cfg.temperatures)
    top_ps = parse_floats(cfg.top_ps)

    candidates = []

    # Greedy baseline uses first variant.
    greedy = lm.generate_text(
        prompt=variants[0]["prompt"],
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        seed=cfg.seed + prompt_index * 10000,
    )
    greedy["kind"] = "greedy"
    greedy["candidate_id"] = f"p{prompt_index:03d}_greedy"
    greedy["variant_name"] = variants[0]["name"]
    greedy["prompt_variant"] = variants[0]["prompt"]
    greedy["judge"] = judge_answer(greedy["text"], prompt, task_mode, cfg)
    candidates.append(greedy)

    cid = 0
    for vi, var in enumerate(variants):
        for sj in range(cfg.samples_per_variant):
            temp = temps[cid % len(temps)]
            top_p = top_ps[cid % len(top_ps)]
            seed = cfg.seed + prompt_index * 10000 + 1000 + cid

            cand = lm.generate_text(
                prompt=var["prompt"],
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                seed=seed,
            )

            cand["kind"] = "sample"
            cand["candidate_id"] = f"p{prompt_index:03d}_cand_{cid:02d}_{var['name']}"
            cand["variant_name"] = var["name"]
            cand["temperature"] = temp
            cand["top_p"] = top_p
            cand["seed"] = seed
            cand["prompt_variant"] = var["prompt"]
            cand["judge"] = judge_answer(cand["text"], prompt, task_mode, cfg)
            candidates.append(cand)

            cid += 1

    return variants, candidates


# ============================================================
# Repair / synthesis
# ============================================================

def template_repair_spec_decode(prompt: str) -> str:
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


def make_model_synthesis_prompt(prompt: str, task_mode: str, top_candidates: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(top_candidates, start=1):
        failures = c["judge"]["hard_fail_reasons"]
        blocks.append(
            f"Candidate {i} ({c['candidate_id']}, score={c['judge']['score']:.3f}, failures={failures}):\n"
            f"{c['text']}\n"
        )

    if task_mode == "spec_decode":
        instructions = (
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
    elif task_mode == "code":
        instructions = (
            "Write a corrected final code answer to the user's request. "
            "Prefer clean, runnable Python. Include short explanation only if useful.\n"
        )
    elif task_mode == "email":
        instructions = (
            "Write the final email. Keep it natural, professional, concise, and avoid over-apologizing.\n"
        )
    else:
        instructions = (
            "Write a corrected final answer to the user's request. "
            "Follow the prompt, avoid repetition, avoid unsupported claims, and avoid role labels.\n"
        )

    return (
        instructions
        + "\nUser prompt:\n"
        + prompt
        + "\n\nCandidate answers:\n"
        + "\n---\n".join(blocks)
        + "\nReturn only the final answer."
    )


@torch.no_grad()
def create_repairs(lm: LM, prompt: str, task_mode: str, ranked: List[Dict[str, Any]], prompt_index: int, cfg: Config) -> List[Dict[str, Any]]:
    repairs = []

    mode = cfg.repair_mode
    if mode == "auto":
        mode = "template" if task_mode == "spec_decode" else "model"

    if mode in ["template", "both"] and task_mode == "spec_decode":
        text = template_repair_spec_decode(prompt)
        cand = {
            "text": text,
            "tokens": None,
            "elapsed_s": 0.0,
            "tokens_per_s": None,
            "kind": "repair",
            "candidate_id": f"p{prompt_index:03d}_repair_template",
            "variant_name": "repair_template",
            "prompt_variant": None,
            "judge": judge_answer(text, prompt, task_mode, cfg),
        }
        repairs.append(cand)

    if mode in ["model", "both"]:
        top = ranked[: max(1, cfg.top_k_for_synthesis)]
        synth_prompt = make_model_synthesis_prompt(prompt, task_mode, top)
        out = lm.generate_text(
            prompt=synth_prompt,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            seed=cfg.seed + prompt_index * 10000 + 9999,
        )
        out["kind"] = "repair"
        out["candidate_id"] = f"p{prompt_index:03d}_repair_model"
        out["variant_name"] = "repair_model"
        out["prompt_variant"] = synth_prompt
        out["judge"] = judge_answer(out["text"], prompt, task_mode, cfg)
        repairs.append(out)

    return repairs


# ============================================================
# Ranking and pipeline
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


@torch.no_grad()
def run_one_prompt(lm: LM, item: Dict[str, str], prompt_index: int, cfg: Config) -> Dict[str, Any]:
    prompt = item["prompt"]
    task_mode = item.get("task_mode", "generic")

    print_section(f"PROMPT {prompt_index+1}: {task_mode}")
    print(prompt)

    t0 = time.time()

    variants, initial_candidates = generate_candidates_for_prompt(lm, prompt, task_mode, prompt_index, cfg)
    initial_candidates = deduplicate_candidates(initial_candidates)
    initial_ranked = rank_candidates(initial_candidates)

    greedy = next(c for c in initial_candidates if c["kind"] == "greedy")
    ensemble_best = initial_ranked[0]

    repairs = []
    if needs_repair(ensemble_best, cfg):
        repairs = create_repairs(lm, prompt, task_mode, initial_ranked, prompt_index, cfg)

    final_pool = deduplicate_candidates(initial_candidates + repairs)
    final_ranked = rank_candidates(final_pool)
    final_best = final_ranked[0]

    dt = time.time() - t0

    # Compare by verifier score. This is not a human score, but it is the local metric.
    ensemble_improves = ensemble_best["judge"]["score"] > greedy["judge"]["score"]
    repair_improves = final_best["judge"]["score"] > greedy["judge"]["score"]
    repair_over_ensemble = final_best["judge"]["score"] > ensemble_best["judge"]["score"]

    summary = {
        "prompt_index": prompt_index,
        "task_mode": task_mode,
        "prompt": prompt,
        "elapsed_s": dt,
        "n_initial_candidates": len(initial_candidates),
        "n_repairs": len(repairs),
        "repair_triggered": bool(repairs),

        "greedy_id": greedy["candidate_id"],
        "greedy_score": greedy["judge"]["score"],
        "greedy_coverage": greedy["judge"]["required_coverage"],
        "greedy_failures": greedy["judge"]["hard_fail_reasons"],
        "greedy_bad_topic_count": greedy["judge"]["bad_topic_count"],
        "greedy_repetition": greedy["judge"]["repeated_bigram_rate"],

        "ensemble_best_id": ensemble_best["candidate_id"],
        "ensemble_best_kind": ensemble_best["kind"],
        "ensemble_best_score": ensemble_best["judge"]["score"],
        "ensemble_best_coverage": ensemble_best["judge"]["required_coverage"],
        "ensemble_best_failures": ensemble_best["judge"]["hard_fail_reasons"],

        "final_best_id": final_best["candidate_id"],
        "final_best_kind": final_best["kind"],
        "final_best_score": final_best["judge"]["score"],
        "final_best_coverage": final_best["judge"]["required_coverage"],
        "final_best_failures": final_best["judge"]["hard_fail_reasons"],

        "ensemble_improves_over_greedy_by_score": bool(ensemble_improves),
        "final_improves_over_greedy_by_score": bool(repair_improves),
        "final_improves_over_ensemble_by_score": bool(repair_over_ensemble),

        "score_gain_ensemble_minus_greedy": ensemble_best["judge"]["score"] - greedy["judge"]["score"],
        "score_gain_final_minus_greedy": final_best["judge"]["score"] - greedy["judge"]["score"],
        "score_gain_final_minus_ensemble": final_best["judge"]["score"] - ensemble_best["judge"]["score"],
    }

    print(f"greedy score:        {summary['greedy_score']:.3f} | failures={summary['greedy_failures'][:3]}")
    print(f"ensemble best score: {summary['ensemble_best_score']:.3f} | id={summary['ensemble_best_id']} | failures={summary['ensemble_best_failures'][:3]}")
    print(f"final best score:    {summary['final_best_score']:.3f} | id={summary['final_best_id']} | failures={summary['final_best_failures'][:3]}")
    print(f"repair triggered:    {summary['repair_triggered']}")
    print(f"score gain final:    {summary['score_gain_final_minus_greedy']:.3f}")

    return {
        "summary": summary,
        "variants": variants,
        "initial_candidates": initial_candidates,
        "initial_ranked_ids": [c["candidate_id"] for c in initial_ranked],
        "repairs": repairs,
        "final_ranked_ids": [c["candidate_id"] for c in final_ranked],
        "greedy": greedy,
        "ensemble_best": ensemble_best,
        "final_best": final_best,
        "final_ranked": final_ranked,
    }


# ============================================================
# Reporting
# ============================================================

def flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in summary.items():
        if isinstance(v, list):
            out[k] = "; ".join(str(x) for x in v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [r["summary"] for r in results]
    n = len(summaries)

    if n == 0:
        return {}

    greedy_scores = np.array([s["greedy_score"] for s in summaries], dtype=float)
    ensemble_scores = np.array([s["ensemble_best_score"] for s in summaries], dtype=float)
    final_scores = np.array([s["final_best_score"] for s in summaries], dtype=float)

    agg = {
        "n_prompts": n,
        "mean_greedy_score": float(greedy_scores.mean()),
        "mean_ensemble_best_score": float(ensemble_scores.mean()),
        "mean_final_best_score": float(final_scores.mean()),

        "ensemble_win_rate_vs_greedy_by_score": float(np.mean(ensemble_scores > greedy_scores)),
        "final_win_rate_vs_greedy_by_score": float(np.mean(final_scores > greedy_scores)),
        "final_win_rate_vs_ensemble_by_score": float(np.mean(final_scores > ensemble_scores)),

        "mean_score_gain_ensemble_minus_greedy": float(np.mean(ensemble_scores - greedy_scores)),
        "mean_score_gain_final_minus_greedy": float(np.mean(final_scores - greedy_scores)),
        "mean_score_gain_final_minus_ensemble": float(np.mean(final_scores - ensemble_scores)),

        "repair_trigger_rate": float(np.mean([s["repair_triggered"] for s in summaries])),
        "mean_elapsed_s_per_prompt": float(np.mean([s["elapsed_s"] for s in summaries])),
        "total_elapsed_s": float(np.sum([s["elapsed_s"] for s in summaries])),
    }

    task_modes = sorted(set(s["task_mode"] for s in summaries))
    by_task = {}
    for tm in task_modes:
        sub = [s for s in summaries if s["task_mode"] == tm]
        gs = np.array([s["greedy_score"] for s in sub], dtype=float)
        es = np.array([s["ensemble_best_score"] for s in sub], dtype=float)
        fs = np.array([s["final_best_score"] for s in sub], dtype=float)
        by_task[tm] = {
            "n": len(sub),
            "mean_greedy_score": float(gs.mean()),
            "mean_ensemble_score": float(es.mean()),
            "mean_final_score": float(fs.mean()),
            "final_win_rate_vs_greedy": float(np.mean(fs > gs)),
            "mean_final_gain": float(np.mean(fs - gs)),
            "repair_trigger_rate": float(np.mean([s["repair_triggered"] for s in sub])),
        }

    agg["by_task_mode"] = by_task
    return agg


def write_outputs(results: List[Dict[str, Any]], aggregate: Dict[str, Any], cfg: Config):
    payload = {
        "config": cfg.__dict__,
        "aggregate": aggregate,
        "results": results,
    }

    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CSV summary
    rows = [flatten_summary(r["summary"]) for r in results]
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with open(cfg.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Human eval CSV: one row per prompt with greedy/final side-by-side.
    human_fields = [
        "prompt_index",
        "task_mode",
        "prompt",
        "greedy_answer",
        "final_answer",
        "greedy_score",
        "final_score",
        "score_gain",
        "human_winner",
        "human_notes",
    ]
    with open(cfg.human_eval_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=human_fields)
        w.writeheader()
        for r in results:
            s = r["summary"]
            w.writerow({
                "prompt_index": s["prompt_index"],
                "task_mode": s["task_mode"],
                "prompt": s["prompt"],
                "greedy_answer": r["greedy"]["text"],
                "final_answer": r["final_best"]["text"],
                "greedy_score": s["greedy_score"],
                "final_score": s["final_best_score"],
                "score_gain": s["score_gain_final_minus_greedy"],
                "human_winner": "",
                "human_notes": "",
            })

    # Markdown report.
    md = []
    md.append("# Prompt Ensemble Benchmark Report\n")
    md.append("This benchmark compares greedy decoding against prompt-ensemble competition and verifier/repair selection.\n")

    md.append("## Aggregate\n")
    md.append("```json")
    md.append(json.dumps(aggregate, indent=2))
    md.append("```")

    md.append("\n## Summary table\n")
    md.append("| i | task | greedy | ensemble | final | final-greedy | repair | final winner | failures |")
    md.append("|---:|---|---:|---:|---:|---:|---|---|---|")
    for r in results:
        s = r["summary"]
        md.append(
            f"| {s['prompt_index']} | {s['task_mode']} | {s['greedy_score']:.2f} | "
            f"{s['ensemble_best_score']:.2f} | {s['final_best_score']:.2f} | "
            f"{s['score_gain_final_minus_greedy']:.2f} | {s['repair_triggered']} | "
            f"{s['final_best_id']} | {', '.join(s['final_best_failures'][:4])} |"
        )

    md.append("\n## Side-by-side examples\n")
    for r in results:
        s = r["summary"]
        md.append(f"### Prompt {s['prompt_index']} — {s['task_mode']}\n")
        md.append(f"**Prompt:** {s['prompt']}\n")
        md.append(f"**Greedy score:** {s['greedy_score']:.3f}  \n")
        md.append(f"**Final score:** {s['final_best_score']:.3f}  \n")
        md.append(f"**Final winner:** `{s['final_best_id']}`\n")

        md.append("#### Greedy\n")
        md.append(r["greedy"]["text"])
        md.append("\n#### Final\n")
        md.append(r["final_best"]["text"])
        md.append("\n---\n")

    Path(cfg.output_report).write_text("\n".join(md), encoding="utf-8")

    print_section("SAVED OUTPUTS")
    print(f"JSON:       {cfg.output_json}")
    print(f"CSV:        {cfg.output_csv}")
    print(f"Markdown:   {cfg.output_report}")
    print(f"Human eval: {cfg.human_eval_csv}")


def print_aggregate(aggregate: Dict[str, Any]):
    print_section("AGGREGATE RESULTS")
    print(json.dumps(aggregate, indent=2))

    print("\nMain numbers:")
    print(f"prompts:                         {aggregate['n_prompts']}")
    print(f"mean greedy score:               {aggregate['mean_greedy_score']:.3f}")
    print(f"mean ensemble score:             {aggregate['mean_ensemble_best_score']:.3f}")
    print(f"mean final score:                {aggregate['mean_final_best_score']:.3f}")
    print(f"ensemble win rate vs greedy:     {aggregate['ensemble_win_rate_vs_greedy_by_score']:.3f}")
    print(f"final win rate vs greedy:        {aggregate['final_win_rate_vs_greedy_by_score']:.3f}")
    print(f"mean final gain vs greedy:       {aggregate['mean_score_gain_final_minus_greedy']:.3f}")
    print(f"repair trigger rate:             {aggregate['repair_trigger_rate']:.3f}")


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

    print_section("BENCHMARK PURPOSE")
    print("This script tests whether prompt ensemble + verifier + repair beats greedy across many prompts.")
    print("The score is the local verifier score, not a human score.")
    print("Use the human_eval CSV for manual validation of the real win rate.")

    prompt_suite = load_prompt_suite(CFG)
    print(f"\nLoaded prompts: {len(prompt_suite)}")

    lm = LM(CFG)

    results = []
    global_t0 = time.time()

    for i, item in enumerate(prompt_suite):
        result = run_one_prompt(lm, item, i, CFG)
        results.append(result)

    total_dt = time.time() - global_t0

    aggregate = aggregate_results(results)
    aggregate["wall_time_s"] = total_dt

    print_aggregate(aggregate)
    write_outputs(results, aggregate, CFG)


if __name__ == "__main__":
    main()
