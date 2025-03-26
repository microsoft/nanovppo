"""Taken from https://github.com/aburkov/theLMbook/blob/main/GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.ipynb.
"""

import re
from datasets import load_dataset
from typing import List, Tuple
import os
import sys

from algos.algo import Request
from ddp_utils import rank_zero_only
from gen_utils import GenerationBackend


GSM8K_COT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def prepare_gsm8k_dataset(subset="all", split="test", subsample=None, template='cot'):
   data = load_dataset('openai/gsm8k', 'main')[split]
   formatted_data = []
   for example in data:
       # Convert list of messages to a single string prompt.
       prompt_str = [
           {"role": "system", "content": GSM8K_COT},
           {"role": "user", "content": example["question"]}
       ]
       answer = extract_answer_from_dataset(example["answer"])
       formatted_data.append((prompt_str, answer))
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


def extract_answer_from_dataset(text):
   """
   Extracts the answer from the GSM8K dataset examples.

   Args:
       text (str): The dataset example text containing a question and answer.

   Returns:
       str or None: The extracted answer part after the '####' delimiter, or None if not found.

   Explanation:
       1. Checks if the text contains the '####' delimiter that separates question from answer.
       2. If found, splits the text at this delimiter and returns the second part (the answer).
       3. The answer is stripped of leading/trailing whitespace.
       4. Returns None if no delimiter is present.
   """
   if "####" not in text:
       return None
   return text.split("####")[1].strip()


def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.

   Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
   """
   text = text.replace('$', '').replace('%', '')
   pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
   match = re.search(pattern, text)
   return float(match.group(1)) if match else None


def extract_single_number(text):
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

   Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
   """
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None


@rank_zero_only
def eval_gsm8k(questions, answers, temperature=0.0, top_p=1.0, max_tokens=1024):
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
            is_correct = correctness_reward(results[i][0], answers[i]) > 0.
            correct.append(is_correct)
        except:
            continue
    return correct


def compute_gsm8k_reward(requests: List[Request]) -> List[float]:
    rewards = []
    for request in requests:
        response = request.response
        label = request.label
        answer_reward = correctness_reward(response, label)
        syntax_reward = format_reward(response)
        rewards.append(answer_reward + syntax_reward)
    return rewards


def correctness_reward(response, answer, **kwargs):
    score = 0.
    pred_answer = extract_answer_from_model_output(response)
    if pred_answer == answer:  # Exact match case
        score = 2.
    else:
        # Try numeric equivalence
        r_num = extract_single_number(str(response))
        a_num = extract_single_number(str(pred_answer))
        if r_num is not None and a_num is not None and r_num == a_num:
            score = 1.5
        else:
            score = 0.0
    return score


def format_reward(response, **kwargs):
    score = 0.0
    if "<reasoning>" in response: score += 0.2
    if "</reasoning>" in response: score += 0.2
    if "<answer>" in response: score += 0.2
    if "</answer>" in response: score += 0.2
    return score
