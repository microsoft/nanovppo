from concurrent.futures import ThreadPoolExecutor
import pprint
from typing import List, Optional
from datasets import load_dataset, concatenate_datasets
import numpy as np
import re
import regex
from tqdm import tqdm
from algos.algo import Request
from ddp_utils import rank_zero_only
from gen_utils import GenerationBackend


MATH_TEMPLATE = (
    """{}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."""
)


MATH_LCOT_TEMPLATE = (
    """{}\n\nThink about the reasoning process to arrive at the answer. """
    """First, reason step by step by enclosing your thinking into <reasoning> reasoning process here </reasoning>. """
    """Then, put your final answer within \\boxed{{}}."""
)


MATH_QUESTION_TEMPLATE = (
    """{}\n\nPlease reason step by step. Structure your thought by formulating questions and giving answers.
    Questions should be enclosed in <question> </question> and answers in <answer> </answer>. """
    """Once you are ready to give the final answer, put your final answer within \\boxed{{}}."""
)


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")

    # replace \\ with \
    # string = string.replace("\\\\", "\\")
    # string = string.replace("\\\\", "\\")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # if len(string.split("=")) == 2:
    #     if len(string.split("=")[0]) <= 2:
    #         string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string


def extract_boxed_answers(text):
    answers = []
    for piece in text.split("boxed{")[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == "{":
                n += 1
            elif piece[i] == "}":
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    if not answers:
        return [""]
    return answers


def extract_program_output(pred_str):
    """
    extract output between the last ```output\n...\n```
    """
    if "```output" not in pred_str:
        return ""
    if "```output" in pred_str:
        pred_str = pred_str.split("```output")[-1]
    if "```" in pred_str:
        pred_str = pred_str.split("```")[0]
    output = pred_str.strip()
    return output


def extract_answer(pred_str, exhaust=False):
    pred = []
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = [tmp.split("$. I hope", 1)[0].strip()]
    elif "boxed" in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif "he answer is" in pred_str:
        pred = [pred_str.split("he answer is")[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output != "":
            # fall back to program
            pred.append(program_output)
        else:  # use the last number
            pattern = "-?\d*\.?\d+"
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if len(ans) >= 1:
                ans = ans[-1]
            else:
                ans = ""
            if ans:
                pred.append(ans)

    # multiple line
    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return strip_string(answer)
    except:
        return answer


def extract_math_answer(question, reasoning):
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        if "separated by commas" in question and all(ch not in ans for ch in "()[]"):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend(
                [
                    a.strip()
                    for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split(
                        "[SEP]"
                    )
                ]
            )
        else:
            answer.append(ans.strip())
    return answer


def compute_math_reward(requests: List[Request]) -> List[float]:
    from math_grader import grade_answer

    rewards = []
    for request in requests:
        response = request.response
        label = request.label
        answer = extract_answer(response)
        rewards.append(grade_answer(given_answer=answer, ground_truth=label))
    return rewards


def compute_lcot_math_reward(requests: List[Request]) -> List[float]:
    from math_grader import grade_answer

    rewards = []
    for request in requests:
        response = request.response
        label = request.label
        answer = extract_answer(response)
        answer_reward = grade_answer(given_answer=answer, ground_truth=label)
        if answer_reward:
            syntax_reward = 0.2 * float("<reasoning>" in response)
            syntax_reward += 0.2 * float("</reasoning>" in response)
        else:
            syntax_reward = 0.
        rewards.append(answer_reward + syntax_reward)
    return rewards


def compute_qst_math_reward(requests: List[Request]) -> List[float]:
    from math_grader import grade_answer

    rewards = []
    for request in requests:
        response = request.response
        label = request.label
        answer = extract_answer(response)
        answer_reward = grade_answer(given_answer=answer, ground_truth=label)
        syntax_reward = re.search(f"<question>(.*)</question>(.*)<answer>(.*)</answer>", response, re.DOTALL)
        syntax_reward = 1. if syntax_reward else 0.0
        rewards.append(answer_reward + syntax_reward)
    return rewards


def prepare_math_dataset(subset="all", split="test", subsample=None, template='cot'):
    if split == "500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    else:
        subsets = ['algebra', 'counting_and_probability', 'geometry', 
                   'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        if subset in [None, "all"]:
            # download dataset for each subset then concatenate 
            subset_datasets = [
                load_dataset(
                    "EleutherAI/hendrycks_math", 
                    subset, 
                    split=split, 
                    download_mode="reuse_dataset_if_exists"
                ) for subset in subsets
            ]
            dataset = concatenate_datasets(subset_datasets)
        else:
            # get dataset for that one subset 
            dataset = load_dataset("EleutherAI/hendrycks_math", subset, split=split)
    if subsample is None or subsample == -1:
        subsample = len(dataset)

    if template == 'lcot':
        template = MATH_LCOT_TEMPLATE
    elif template == 'cot':
        template = MATH_TEMPLATE
    elif template == 'qst':
        template = MATH_QUESTION_TEMPLATE

    # create array of all questions in dataset
    questions = []
    for i in range(len(dataset)):
        questions.append(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": template.format(dataset[i]["problem"]),
                },
            ]
        )
    # same for answers
    if "answer" in dataset.column_names:
        answers = list(dataset["answer"])
    else:
        answers = []
        for i in range(len(dataset)):
            answers.append(extract_answer(dataset[i]["solution"]))

    print("Loaded dataset with {} questions.".format(len(questions)))
    return questions[:subsample], answers[:subsample]


@rank_zero_only
def eval_math(questions, answers, temperature=0.0, top_p=1.0, max_tokens=1024):
    from math_grader import grade_answer

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

    pred_answers = []
    for i in range(len(results)):
        pred_answers.append(extract_answer(results[i][0]))

    return list(
        [
            grade_answer(given_answer=pred_answers[i], ground_truth=answers[i])
            for i in range(len(answers))
        ]
    )
