"""Shared chat-style math evaluator.

Used by ``evaluate_math.py``, ``evaluate_amc.py``, ``evaluate_aime.py`` and
``evaluate_aime_2025.py``. Each of those scripts is now a thin wrapper that:

    1. parses CLI args
    2. loads a parquet via ``utils.parquet_loader.load_parquet_dataset``
    3. calls :func:`run_chat_eval` to handle generation + scoring + I/O

The parquet files in ``evaluation/data/`` are pre-replicated (``_x5``, ``_x10``,
``_x32``) — i.e. each problem appears N times so that single-sample greedy
generation reproduces self-consistency over N rollouts. We therefore pass
``n=1`` to vLLM and rely on the data layout for repetition.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Any, Dict, List

REPO_EVAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_EVAL_ROOT not in sys.path:
    sys.path.append(REPO_EVAL_ROOT)

from utils.grader import math_equal  # noqa: E402


def _last_boxed_only_string(string: str):
    """Return the contents of the last ``\\boxed{...}`` (or ``\\fbox{...}``)."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None
    return string[left_brace_idx + 1 : right_brace_idx].strip()


def match_answer(response: str):
    """Heuristically extract a final-answer string from a model completion."""
    is_matched = False
    for ans_marker in ["answer:", "answer is", "answers are"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("\n"):
                response = response[:-2]

    for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[:ans_idx].strip()
            if response.endswith("\n"):
                response = response[:-2]

    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    if ". " in response:
        dot_idx = response.lower().rfind(". ")
        if dot_idx != -1:
            response = response[:dot_idx].strip()

    for ans_marker in ["be ", "is ", "are ", "=", ": ", "get ",
                       "be\n", "is\n", "are\n", ":\n", "get\n"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("\n"):
                response = response[:-2]

    is_matched = is_matched if any(c.isdigit() for c in response) else False
    return is_matched, response


def make_chat_prompt(question: str, tokenizer) -> str:
    msg = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def is_correct(prediction: str, answer: str) -> bool:
    """Symbolic / numeric equivalence with a fallback for ``\\pi``."""
    try:
        if r"\pi" in prediction or r"\pi" in answer:
            return any(
                math_equal(prediction, answer, timeout=True, pi=pi_val)
                for pi_val in (math.pi, 3.14)
            )
        return math_equal(prediction, answer, timeout=True)
    except Exception:
        return False


def _generate(model_path: str, prompts: List[str],
              temperature: float, top_p: float, repetition_penalty: float,
              max_tokens: int = 4096,
              gpu_memory_utilization: float = 0.9,
              extra_stop: List[str] | None = None) -> List[str]:
    """vLLM inference. Imported lazily so static analyzers don't require vLLM."""
    from vllm import LLM, SamplingParams  # noqa: WPS433

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    stop = list(extra_stop) if extra_stop else []
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop=stop or None,
    )
    print(f"[chat_eval] sampling: temp={temperature} top_p={top_p} "
          f"repetition_penalty={repetition_penalty} n_prompts={len(prompts)}")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_chat_eval(
    *,
    benchmark: str,
    examples: List[Dict[str, Any]],
    model_path: str,
    save_dir: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_tokens: int = 4096,
    gpu_memory_utilization: float = 0.9,
    extra_stop: List[str] | None = None,
) -> Dict[str, Any]:
    """Run a chat-style benchmark end-to-end.

    Args:
        benchmark:               Display name (printed in the summary).
        examples:                Output of ``load_parquet_dataset``.
        model_path:              HF path or local checkpoint.
        save_dir:                Output directory (created if missing).
        temperature/top_p/...:   Standard vLLM sampling knobs.
        max_tokens:              Per-sample max new tokens.
        extra_stop:              Optional extra stop strings.

    Returns:
        ``{"benchmark", "total", "correct", "accuracy"}``.
    """
    from transformers import AutoTokenizer  # noqa: WPS433

    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    questions = [ex["question"] for ex in examples]
    prompts = [make_chat_prompt(q, tokenizer) for q in questions]

    completions = _generate(
        model_path=model_path,
        prompts=prompts,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        extra_stop=extra_stop,
    )

    completions_path = os.path.join(save_dir, "completions.jsonl")
    _write_jsonl(
        completions_path,
        [
            {"idx": ex["idx"], "question": ex["question"],
             "answer": ex["answer"], "completion": comp}
            for ex, comp in zip(examples, completions)
        ],
    )

    results: List[Dict[str, Any]] = []
    correct = 0
    for ex, completion in zip(examples, completions):
        gold = str(ex["answer"]).lstrip("0") or "0"
        matched, extracted = match_answer(completion)
        extracted = extracted.strip("The final answer is ").strip(". I hope it is correct.")
        ok = is_correct(extracted, gold)
        correct += int(ok)
        results.append({
            "idx": ex["idx"],
            "question": ex["question"],
            "answer": gold,
            "extracted": extracted,
            "matched": matched,
            "correct": ok,
            "completion": completion,
        })

    total = len(results)
    accuracy = correct / total if total else 0.0
    summary = {
        "benchmark": benchmark,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }

    _write_jsonl(os.path.join(save_dir, "results.jsonl"), results)
    with open(os.path.join(save_dir, "results_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(save_dir, "results_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"{benchmark}: total={total} correct={correct} "
                 f"accuracy={accuracy:.4f}\n")

    print(f"\n########## {benchmark} ##########")
    print(f"total={total} correct={correct} accuracy={accuracy:.4f}")
    print(f"results -> {save_dir}\n")
    return summary


__all__ = [
    "match_answer",
    "make_chat_prompt",
    "is_correct",
    "run_chat_eval",
]
