"""Taken from https://github.com/aburkov/theLMbook/blob/main/GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.ipynb."""

import glob
import re
from datasets import load_dataset
from typing import List, Tuple
import os
import sys

from algos.algo import Request
from ddp_utils import rank_zero_only
from gen_utils import GenerationBackend

import os
from typing import List, Dict, Any
import json
import re
import yaml


PROMPT = """Find the common rule that maps an input grid to an output grid, given the examples below.

{input_train}


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Think carefully about the problem by enclosing your reasoning inside <reasoning> </reasoning>.
Then, output your finale answer within <answer> </answer>. Your final answer should just be the text output grid itself.

Input:
{input_test}
"""

EXAMPLE_PROMPT = """Example {example_id}:

Input:
{input}
Output:
{output}

"""


@rank_zero_only
def eval_arc(questions, answers, temperature=0.0, top_p=1.0, max_tokens=1024):
    if not isinstance(questions[0], list):
        # then create chat messages
        questions = [[{"role": "user", "content": q}] for q in questions]

    print("====================================")
    print(f"Number of questions: {len(questions)}")
    print(f"Example question: {questions[0]}")
    print(f"Example answer: {answers[0]}")
    print("====================================")

    results = GenerationBackend.get().chat(
        questions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    correct = []
    for i in range(len(results)):
        try:
            is_correct = bool(compute_arc_content_reward(results[i][0], answers[i]))
            correct.append(is_correct)
        except:
            continue
    return correct


def prepare_arc_dataset(subset="all", split="test", subsample=None, template='cot'):
    TMP_DIR = "/tmp/arc-agi"

    if not os.path.exists(TMP_DIR):
        os.system("git clone https://github.com/fchollet/ARC-AGI /tmp/arc-agi")

    if split == "train":
        data_dir = os.path.join(TMP_DIR, "data", "training")
    else:
        data_dir = os.path.join(TMP_DIR, "data", "evaluation")

    pairs = []
    for task_file in glob.glob(os.path.join(data_dir, "*.json")):
        with open(task_file, "r") as f:
            task_data = json.load(f)

            formatted_examples = ""
            for i, pair in enumerate(task_data["train"]):
                formatted_examples += EXAMPLE_PROMPT.format(
                    input=pair["input"], output=pair["output"], example_id=i
                )

            # for all test pairs, create a prompt
            for j, pair in enumerate(task_data["test"]):
                prompt = [
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            input_train=formatted_examples,
                            input_test=pair['input'],
                        ),
                    }
                ]
                target = str(pair["output"])
                pairs.append((prompt, target))
    return zip(*pairs)


def compute_arc_content_reward(response, label):
    start_tag = "<answer>"
    end_tag = "</answer>"

    answer_content = ""
    if start_tag in response and end_tag in response:
        start_idx = response.index(start_tag) + len(start_tag)
        end_idx = response.index(end_tag)
        if start_idx < end_idx:
            answer_content = response[start_idx:end_idx].strip()

    # Compute content reward based on answer presence and match with label
    content_reward = 0.0
    if answer_content:
        if answer_content.lower() == label.lower():
            content_reward += 1.0
    return content_reward


def compute_arc_reward(requests: List[Request]) -> List[float]:
    rewards = []
    for request in requests:
        response = request.response
        label = request.label.strip()
        syntax_reward = 0.2 * float("<reasoning>" in response)
        syntax_reward += 0.2 * float("</reasoning>" in response)
        syntax_reward += 0.2 * float("<answer>" in response)
        syntax_reward += 0.2 * float("</answer>" in response)
        content_reward = compute_arc_content_reward(response, label)
        total_reward = syntax_reward + content_reward
        rewards.append(total_reward)
    return rewards


def extract_json_grid_from_end(text):
    """
    Safely extracts JSON grid from the end of text.
    Returns a list of lists (grid) if successful, None otherwise.
    """
    try:
        # First, try to find a complete JSON array with nested arrays
        complete_grid_match = re.search(
            r"\[\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\s*\]",
            text,
            re.DOTALL,
        )
        if complete_grid_match:
            try:
                return json.loads(complete_grid_match.group(0))
            except json.JSONDecodeError:
                pass  # Continue with line-by-line approach if this fails

        # Handle the case where arrays are written without commas between rows
        no_comma_grid_match = re.search(
            r"\[\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\s*\]",
            text,
            re.DOTALL,
        )
        if no_comma_grid_match:
            # Add commas between the arrays
            fixed_json = re.sub(r"\]\s*\[", "],[", no_comma_grid_match.group(0))
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass  # Continue with line-by-line approach if this fails

        # Handle multi-line grid format without outer brackets and with variable number of rows
        multi_line_grid_match = re.search(
            r"\[\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*\n\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*",
            text,
            re.DOTALL,
        )
        if multi_line_grid_match:
            # Add outer brackets and commas between rows
            grid_text = multi_line_grid_match.group(0)
            fixed_json = "[" + re.sub(r"\]\s*\n\s*\[", "],[", grid_text) + "]"
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass  # Continue with line-by-line approach if this fails

        # Original line-by-line approach as fallback
        lines = text.strip().splitlines()
        extracted_lines = []

        # Iterate backwards to find JSON-like lines
        for line in reversed(lines):
            line = line.strip()
            if re.match(r"^\[\s*(\d+\s*,\s*)*\d+\s*\]$", line):
                extracted_lines.append(line)
            elif extracted_lines:
                # Once we encounter a non-matching line after capturing, break.
                break

        # Reverse to restore original order
        extracted_lines.reverse()

        # Convert lines to actual lists
        result = []
        for line in extracted_lines:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Skip invalid lines

        return result if result else None
    except Exception:
        return None


def save_submission(save_submission_dir: str, task_id: str, task_attempts) -> None:
    """
    Save the submission to a file with full attempt metadata.

    The save_submission_dir should be a directory path that includes the config name,
    e.g., 'submissions/o1_short_response' or 'submissions/gemini_pro'.
    """
    os.makedirs(save_submission_dir, exist_ok=True)
    submission_file = os.path.join(save_submission_dir, f"{task_id}.json")

    with open(submission_file, "w") as f:
        json.dump(task_attempts, f, indent=4)

    return submission_file
