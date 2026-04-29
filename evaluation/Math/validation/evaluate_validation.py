# Adapt from https://github.com/hendrycks/math/blob/main/modeling/evaluate_gpt3.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import traceback
import openai
import argparse
import numpy as np
import operator
import json
import tqdm
import pandas as pd
from utils.util import clean_numbers, last_boxed_only, last_boxed_only_string
from utils.math_equivalence import is_equiv
from utils.grader import math_equal
from collections import defaultdict
from vllm import LLM, SamplingParams
import torch
import re
import math
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from verl.utils.reward_score.evaluation_utils.math_util import match_answer, evaluate_math
from concurrent.futures import ProcessPoolExecutor
# import multiprocessing as mp

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_WORKER_START_METHOD"] = "spawn"


def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        # tensor_parallel_size=torch.cuda.device_count(),
        tensor_parallel_size=4,
        gpu_memory_utilization=0.90,
        worker_use_ray=True
    )
    sampling_params = SamplingParams(max_tokens=4096,
                                     temperature=0,
                                     best_of=1
                                     )
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = [output.outputs[0].text for output in outputs]
    # completions = [re.split(r'\$|###|\n', output.outputs[0].text)[0] for output in outputs]
    return completions


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def _last_boxed_only_string(string):
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

    return string[left_brace_idx + 1: right_brace_idx].strip()


# def match_answer_prime(response):
#     is_matched = False
#     ans_marker = 'The answer is: '
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]
            
#     ans_marker = 'answer:\n'
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     ans_marker = 'answer: '
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     # Find boxed
#     ans_boxed = _last_boxed_only_string(response)
#     if ans_boxed:
#         is_matched = True
#         response = ans_boxed

#     # Grade
#     return is_matched, response

# def match_answer_verl(response):
#     # breakpoint()
#     is_matched = False
#     for ans_marker in ['answer:', "answer is", "answers are"]:
#         ans_idx = response.lower().rfind(ans_marker)
#         if ans_idx != -1:
#             is_matched = True
#             response = response[ans_idx + len(ans_marker):].strip()
#             if response.endswith("\n"):
#                 response = response[:-2]

#     for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
#         ans_idx = response.lower().rfind(ans_marker)
#         if ans_idx != -1:
#             is_matched = True
#             response = response[:ans_idx].strip()
#             if response.endswith("\n"):
#                 response = response[:-2]

#     # Find boxed
#     ans_boxed = _last_boxed_only_string(response)
#     if ans_boxed:
#         is_matched = True
#         response = ans_boxed

#     if ". " in response:
#         dot_idx = response.lower().rfind(". ")
#         if dot_idx != -1:
#             response = response[:dot_idx].strip()

#     for ans_marker in ['be ', "is ", "are ", "=", ": ", "get ", 'be\n', "is\n", "are\n", ":\n", "get\n"]:
#         ans_idx = response.lower().rfind(ans_marker)
#         if ans_idx != -1:
#             is_matched = True
#             response = response[ans_idx + len(ans_marker):].strip()
#             if response.endswith("\n"):
#                 response = response[:-2]

#     is_matched = is_matched if any([c.isdigit() for c in response]) else False  # answer must have a digit
#     # Grade
#     return is_matched, response

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS





def make_conv_hf(question, tokenizer):
    # system_prompt = open("system_prompt.md").read()
    content = question
    msg = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

def equiv(problem_data, model_output):
    answer = str(problem_data["reward_model"]["ground_truth"])
    problem_data["completion"] = model_output
    equiv, _, __ = evaluate_math(model_output, answer)

    return equiv

def run(args, max=-1):
    from datasets import load_dataset
    if not args.data_path:
        raise ValueError("--data_path is required")
    dataset = load_dataset("parquet", data_files=args.data_path)
    print("reading problems done!")
    all_problems = [all_problem for all_problem in dataset["train"]]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    completions = generate_sample_batch(
        [make_conv_hf(problem_data["prompt"][0]["content"], tokenizer) for problem_data in all_problems])
    tmp_data = []
    for problem_data, model_output in zip(dataset["train"], completions):
        problem_data["completion"] = model_output
        tmp_data.append(problem_data)
    write_jsonl_file(os.path.join(args.save_dir, "completions.jsonl"), tmp_data)
    calculate_acc(args.save_dir)
    # total = len(dataset["train"])
    # total = len(all_problems)
    # correct = 0
    # save_data = []
    # for problem_data, model_output in zip(all_problems, completions):
    #     # breakpoint()
    #     answer = str(problem_data["reward_model"]["ground_truth"])
    #     problem_data["completion"] = model_output
    #     is_matched, model_output = match_answer(model_output)
    #     model_output = model_output.strip("The final answer is ").strip(". I hope it is correct.")
    #     equiv, _, __ = evaluate_math(model_output, answer)
    #     # try:
    #     #     if "\pi" in model_output or "\pi" in answer:
    #     #         equivs = []
    #     #         for pi in [math.pi, 3.14]:
    #     #             equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
    #     #         equiv = any(equivs)
    #     #     else:
    #     #         equiv = evaluate_math(model_output, answer, timeout=True)
    #     # except:
    #     #     equiv = False

    #     if equiv:
    #         correct += 1
    #     problem_data["success"] = equiv
    #     save_data.append(problem_data)

    # print("##########Validation")
    # print(f"total: {total}, success: {correct}, rate: {correct / total}")

    # output_file = os.path.join(args.save_dir, "results_total.txt")
    # with open(output_file, "w+") as f:
    #     f.write(f"total: {total}, success: {correct}, rate: {correct / total}")
    # write_jsonl_file(os.path.join(args.save_dir, "results.json"), save_data)

# def func(problem_data, model_output):
#     answer = str(problem_data["reward_model"]["ground_truth"])
#     problem_data["completion"] = model_output
#     equiv, _, __ = evaluate_math(model_output, answer)

#     return equiv

def calculate_acc(path: str):
    all_problems = read_jsonl_file(os.path.join(path, "completions.jsonl"))

    all_completions = [all_problem["completion"] for all_problem in all_problems]
    # breakpoint()
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    # completions = generate_sample_batch(
        # [make_conv_hf(problem_data["prompt"][0]["content"], tokenizer) for problem_data in all_problems])
    # tmp_data = []
    # for problem_data, model_output in zip(dataset["train"], completions):
    #     problem_data["completion"] = model_output
    #     tmp_data.append(problem_data)

    # total = len(dataset["train"])
    total = len(all_problems)
    correct = 0
    save_data = []

    with ProcessPoolExecutor(max_workers=100) as executor:
        results = executor.map(equiv, all_problems, all_completions)
        results = list(results)
    correct = sum(results)
    print("##########Validation")
    # breakpoint()
    print(f"total: {total}, success: {correct}, rate: {correct / total}")

    output_file = os.path.join(path, "results_total.txt")
    with open(output_file, "w+") as f:
        f.write(f"total: {total}, success: {correct}, rate: {correct / total}")
    # write_jsonl_file(os.path.join(args.save_dir, "results.json"), save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_dir", "-s", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()
    run(args)
    # run_saved(args)
