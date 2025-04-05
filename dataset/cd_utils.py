"""Adapted from https://github.com/McGill-NLP/nano-aha-moment/blob/main/nano_r1_script.py."""

import re
from datasets import load_dataset
from typing import List, Tuple
import os
import sys

import numpy as np

from algos.algo import Request
from ddp_utils import rank_zero_only
from gen_utils import GeneratorClient


SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process in the mind "
    "and then provide the user with the answer."
)

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)


def prepare_cd_dataset(subset="all", split="test", subsample=None, template="cot"):
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")["train"]
    if split == "test":
        data_indices = np.random.randint(0, len(data), 100)
        data = data.select(data_indices)
    else:
        data_indices = np.random.randint(0, len(data), 10000)
        data = data.select(data_indices)

    formatted_data = []
    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    numbers=example["nums"], target=example["target"]
                ),
            },
        ]
        # need to compute reward with the numbers and target
        formatted_data.append((prompt_str, (example["nums"], example["target"])))
    return zip(*formatted_data)


def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.

    Args:
        text (str): The model-generated text containing XML-style <answer> tags.

    Returns:
        str or None: The content inside the <answer> tags, or None if no valid answer is found.

    Explanation:
        1. Splits the text on the <answer> tag to isolate content after the tag.
        2. Checks if at least one <answer> tag exists in the text.
        3. For the last <answer> segment:
           - Verifies it contains a closing </answer> tag.
           - Extracts only the content between the tags.
        4. Returns None if the answer is empty (just "...") or if tags are missing.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


@rank_zero_only
def eval_cd(questions, answers, temperature=0.0, top_p=1.0, max_tokens=1024):
    if not isinstance(questions[0], list):
        # then create chat messages
        questions = [[{"role": "user", "content": q}] for q in questions]

    print("====================================")
    print(f"Number of questions: {len(questions)}")
    print(f"Example question: {questions[0]}")
    print(f"Example answer: {answers[0]}")
    print("====================================")

    results = GeneratorClient.get().chat(
        questions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    correct = []
    for i in range(len(results)):
        is_correct = correctness_reward(results[i][0], answers[i][0], answers[i][1])
        correct.append(is_correct)
    return correct


def compute_cd_reward(requests: List[Request]) -> List[float]:
    rewards = []
    for request in requests:
        response = request.response
        label = request.label
        answer_reward = correctness_reward(response, label[0], label[1])
        syntax_reward = format_reward(response)
        rewards.append(answer_reward + syntax_reward)
    return rewards


def format_reward(response, **kwargs):
    score = 0.0
    if "<think>" in response:
        score += 0.2
    if "</think>" in response:
        score += 0.2
    if "<answer>" in response:
        score += 0.2
    if "</answer>" in response:
        score += 0.2
    return score


def correctness_reward(answer: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # Extract all numbers from the equation
        answer = extract_answer_from_model_output(answer)
        used_numbers = [int(n) for n in re.findall(r"\d+", answer)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, answer):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(answer, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0
