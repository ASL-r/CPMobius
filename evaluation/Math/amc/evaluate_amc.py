"""AMC evaluator.

Reads a verl-style parquet (``amc_x10.parquet``) and reports accuracy via the
shared chat-evaluation logic in ``utils.chat_eval``.
"""

from __future__ import annotations

import argparse
import os
import sys

EVAL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if EVAL_ROOT not in sys.path:
    sys.path.append(EVAL_ROOT)

from utils.chat_eval import run_chat_eval
from utils.parquet_loader import load_parquet_dataset

os.environ.setdefault("NCCL_IGNORE_DISABLED_P2P", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMC evaluation")
    parser.add_argument("--data_path", required=True,
                        help="Path to AMC parquet (e.g. data/amc_x10.parquet)")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_parquet_dataset(args.data_path)
    run_chat_eval(
        benchmark="AMC",
        examples=examples,
        model_path=args.model,
        save_dir=args.save_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
