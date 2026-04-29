"""Shared parquet loader for math benchmark evaluation.

All math benchmark parquet files in CPMobius use the verl dataset schema:

    columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    prompt:        list[{"role": "user", "content": <question>}]
    reward_model:  {"ground_truth": <answer_str_or_array>, "style": "rule"}

The AIME-2025 parquet is the lone exception, with flat ``question`` / ``answer``
columns.

Verl prompts include a fixed CoT instruction suffix
``" Let's think step by step and output the final answer within \\boxed{}."``
that is injected at training time. We strip it here so evaluators see the bare
problem statement and can apply their own chat template / prompt format.

The parquet files are already repeated N times per problem (see filename
suffixes ``_x5``, ``_x10``, ``_x32``); that repetition encodes the desired
self-consistency sampling count, so the loader preserves duplicates by default.
Set ``dedupe=True`` to collapse duplicates when the consumer wants to control
sampling itself.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

COT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def _normalize_answer(ans: Any) -> str:
    """Collapse the various ``ground_truth`` representations into a single string."""
    if ans is None:
        return ""
    if hasattr(ans, "tolist"):  # numpy array
        ans = ans.tolist()
    if isinstance(ans, (list, tuple)):
        ans = ans[0] if ans else ""
    return str(ans).strip()


def _strip_cot_suffix(question: str) -> str:
    return question[: -len(COT_SUFFIX)].rstrip() if question.endswith(COT_SUFFIX) else question


def load_parquet_dataset(path: str, dedupe: bool = False) -> List[Dict[str, Any]]:
    """Load a parquet file and normalize each row to ``{idx, question, answer}``.

    Args:
        path:   Path to the parquet file.
        dedupe: If True, keep only the first occurrence of each (question, answer)
                pair. Defaults to False so the file's built-in repetition factor
                drives self-consistency sampling.

    Returns:
        List of dicts with keys ``idx``, ``question``, ``answer``. Each dict also
        carries ``raw`` (the original row as a dict) for evaluators that want to
        inspect extra columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")

    df = pd.read_parquet(path)
    cols = set(df.columns)

    if {"prompt", "reward_model"}.issubset(cols):
        rows = _iter_verl_rows(df)
    elif {"question", "answer"}.issubset(cols):
        rows = _iter_flat_rows(df)
    else:
        raise ValueError(
            f"Unsupported parquet schema for {path}: columns={list(df.columns)}. "
            "Expected verl-style (prompt + reward_model) or flat (question + answer)."
        )

    examples: List[Dict[str, Any]] = []
    seen: set = set()
    for example in rows:
        key = (example["question"], example["answer"])
        if dedupe and key in seen:
            continue
        seen.add(key)
        examples.append(example)

    if not examples:
        raise ValueError(f"No examples loaded from {path}.")
    return examples


def _iter_verl_rows(df: pd.DataFrame):
    for i, row in enumerate(df.to_dict(orient="records")):
        prompt_msgs = row["prompt"]
        question = _strip_cot_suffix(prompt_msgs[0]["content"])
        answer = _normalize_answer(row["reward_model"].get("ground_truth"))
        yield {"idx": i, "question": question, "answer": answer, "raw": row}


def _iter_flat_rows(df: pd.DataFrame):
    for i, row in enumerate(df.to_dict(orient="records")):
        question = str(row["question"]).strip()
        answer = _normalize_answer(row["answer"])
        yield {"idx": i, "question": question, "answer": answer, "raw": row}
