from collections import defaultdict
import re
from typing import Dict, List, Union

import torch
import copy
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from ddp_utils import rank_zero_only, ddp_state
from utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    AccumulatorDict,
    get_logprobs,
    get_shifted_logprobs,
    masked_mean,
    masked_whiten,
    print_metrics,
    repeat,
    flatten,
    compute_kl_divergence,
)

from dataset.data_utils import (
    create_joint_tensors,
    get_ending_tokens,
    pad_query_and_response,
)
from gen_utils import GeneratorClient
from algos.algo import Algo, Request, RequestUtils
from algos.rft import RFT
import ast
from algos.vppo import (
    create_joint_tensors_for_vppo,
    vppo_balanced_chunk,
    vppo_treetune_chunk,
)


VERIFIER_TEMPLATE = """Read the following problem and solution chunked in steps.
Your task is to rate each step’s correctness and usefulness.
Return a JSON list of integers in the range 0‑5, one per step.

Problem:
{problem}

Solution:
{step_solution}

Think through the evaluation between <think> and </think>.
Put only the list of scores between <scores> and </scores>.

E.g. if there are 3 steps in the solution, your answer should look like this:

<think>your thinking here</think>
<scores>[score step 1, score step 2, score step 3]</scores>
"""


VERIFIER_TEMPLATE_2 = """Here is a problem and a partial solution chunked in steps. Your task is to determine whether the last step of the solution is correct and useful to solve the problem.

Problem:
{problem}

Solution:
{step_solution}

Format your answer as a single score (from 0 to 5), 0 being the worst and 5 being the best, where the score correspond to the quality last step of the solution.
Think about your answer between <think> and </think> and then write the score down for the last step between <answer> and </answer>.
"""

VERIFIER_TEMPLATE_ASS = "<think>"


def genrm_format_reward(completion: str, num_expected_scores: int) -> float:
    """
    Format: <think>...</think><scores>...</scores>

    Returns:
        float: Reward score
    """
    try:
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<scores>([\s\S]*?)<\/scores>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <scores>...</scores>
            try:
                answer = ast.literal_eval(match.group(2).strip())
                # Must be a list …
                if not isinstance(answer, list):
                    return 0.5

                if not all(isinstance(x, (int, float)) for x in answer):
                    return 0.5

                if not len(answer) == num_expected_scores:
                    return 0.5
            except:
                # If it doesn't match, reward is 0.5
                return 0.5    
        return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def genrm_extract_answer(completion: str):
    parts = completion.split("<scores>")
    if len(parts) < 2:
        return None
    last_part = parts[-1]

    if "</scores>" not in last_part:
        return None
    answer = last_part.split("</scores>")[0].strip()

    try:
        answer = ast.literal_eval(answer)
        # Must be a list …
        if not isinstance(answer, list):
            return None
        # … and every element must be numeric
        if not all(isinstance(x, (int, float)) for x in answer):
            return None
    except Exception:
        return None
    return answer


class GenRM(Algo):
    @classmethod
    def add_parser_args(self, parser):
        parser.add_argument(
            "--kl_ctl", type=float, default=0.0001, help="KL term control"
        )
        parser.add_argument(
            "--kl_max", type=int, default=10, help="Clip KL divergence to this value"
        )
        parser.add_argument(
            "--inference",
            type=int,
            default=0,
            help="If 0, infer all together, if 1 infer independently.",
        )
        parser.add_argument("--drgrpo", type=int, default=0, help="If 1, use DrGRPO")
        return parser

    def __init__(
        self,
        model_name,
        reward_func,
        k=5,
        temperature=DEFAULT_TEMP,
        max_tokens=DEFAULT_MAX_TOKENS,
        device="cuda",
        **algo_kwargs,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.k = k
        self.kl_ctl = algo_kwargs["kl_ctl"]
        self.kl_max = algo_kwargs["kl_max"]
        self.drgpo = algo_kwargs["drgrpo"]
        self.reward_func = reward_func
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stats = AccumulatorDict()

    def extract_genrm_scores(
        self,
        response_chunks: List[List[str]],
        response_rewards: List[float],
        genrm_responses: List[str],
    ):
        # Parse verifier responses to extract scores
        step_rewards = []
        format_rewards = []
        correctness_rewards = []
        for i, response in enumerate(genrm_responses):
            # append the first <think> which has been teacher forced
            response = "<think>" + response
            format_reward = genrm_format_reward(
                response, num_expected_scores=len(response_chunks[i])
            )
            answer = genrm_extract_answer(response)

            if not answer or len(answer) != len(response_chunks[i]):
                # correctness reward is 0 if we couldn't parse the answer
                answer = [0.0] * len(response_chunks[i])
                corr_reward = 0
            else:
                answer = [np.clip(float(s), 0, 5) / 5 for s in answer]
                corr_reward = 1.0 - np.abs(
                    np.mean(answer) - float(response_rewards[i]) / 2.0
                )

            step_rewards.append(answer)
            format_rewards.append(format_reward)
            correctness_rewards.append(corr_reward)

        return step_rewards, format_rewards, correctness_rewards

    @torch.no_grad
    def gather_episodes(self, messages, labels):
        engine = GeneratorClient.get()

        messages = copy.deepcopy(messages)

        # Gather first set of completions
        responses, finished = engine.chat(
            messages,
            temperature=self.temperature,
            n=self.k,
            max_tokens=self.max_tokens,
            return_finished=True,
        )
        responses = flatten(responses)
        finished = flatten(finished)
        qids = repeat(range(len(messages)), self.k)
        labels = repeat(labels, self.k)
        messages = repeat(messages, self.k)

        evaluation_requests = []
        for query_id, query_messages, query_response, label in zip(
            qids, messages, responses, labels
        ):
            evaluation_requests.append(
                Request(
                    query_id=query_id,
                    messages=query_messages,
                    response=query_response,
                    label=label,
                )
            )

        # compute rewards
        rewards = self.reward_func(evaluation_requests)
        if type(rewards) is tuple:
            for key, value in rewards[1].items():
                self.stats.accumulate("avg_" + key, np.mean(value))
            rewards = rewards[0]

        RequestUtils.populate(evaluation_requests, rewards, "reward")

        # now create genrm messages
        genrm_messages = []
        responses_steps, responses_indices = vppo_treetune_chunk(
            responses, max_chunks=6, merge_every_n_chunk=2
        )

        for i in range(len(messages)):
            user_message = messages[i][-1]["content"]
            assistant_message = responses[i]
            chunks_ = [
                "Step {i}:\n{chunk}\nEND OF STEP".format(i=j + 1, chunk=chunk)
                for j, chunk in enumerate(responses_steps[i])
            ]
            assistant_message = "\n\n".join(chunks_)
            genrm_messages.append(
                [
                    {
                        "role": "system",
                        "content": VERIFIER_TEMPLATE.format(
                            problem=user_message, step_solution=assistant_message
                        ),
                    },
                    {"role": "assistant", "content": VERIFIER_TEMPLATE_ASS},
                ]
            )

        genrm_responses, genrm_finished = engine.chat(
            genrm_messages,
            temperature=self.temperature,
            n=1,
            max_tokens=self.max_tokens,
            return_finished=True,
        )
        genrm_responses = flatten(genrm_responses)

        genrm_step_rewards, genrm_format_rewards, genrm_corr_rewards = (
            self.extract_genrm_scores(
                responses_steps,
                rewards,
                genrm_responses,
            )
        )

        genrm_rewards = [
            float(f) + float(f) * r
            for f, r in zip(genrm_format_rewards, genrm_corr_rewards)
        ]  # between 0 and 2

        step_rewards = [
            [step_reward + seq_reward for step_reward in step_rewards]
            for step_rewards, seq_reward in zip(genrm_step_rewards, rewards)
        ]

        # Compute per‑query, z‑scored advantages
        step_advantages = []
        for i in range(0, len(step_rewards), self.k):
            block = step_rewards[i : i + self.k]  # k responses of one query
            flat = np.concatenate(block)
            mean, std = flat.mean(), flat.std() + 1e-8  # avoid div‑by‑zero
            step_advantages.extend(
                [((np.array(r) - mean) / std).tolist() for r in block]
            )

        genrm_advantages = np.array(genrm_rewards)
        genrm_advantages = genrm_advantages.reshape(-1, self.k)
        genrm_advantages = genrm_advantages - genrm_advantages.mean(
            axis=1, keepdims=True
        )
        genrm_advantages = genrm_advantages / (
            genrm_advantages.std(axis=1, keepdims=True) + 1e-8
        )
        genrm_advantages = genrm_advantages.reshape(-1).tolist()

        genrm_response_indices = [[len(r)] for r in genrm_responses]
        genrm_advantages = [[r] for r in genrm_advantages]

        max_reward, avg_reward = RequestUtils.gather_max_avg_reward(evaluation_requests)

        self.stats.accumulate("avg_reward", avg_reward)
        self.stats.accumulate("avg_steps", np.mean([len(r) for r in responses_steps]))
        self.stats.accumulate("genrm_reward", np.mean(genrm_rewards))
        self.stats.accumulate("genrm_corr_reward", np.mean(genrm_corr_rewards))
        self.stats.accumulate("genrm_format_reward", np.mean(genrm_format_rewards))

        ddp_state.print("====================================")
        ddp_state.print("Problem:\n", evaluation_requests[0].messages[-1]["content"])
        ddp_state.print("\nAnswer:\n", evaluation_requests[0].response)
        ddp_state.print("\nGenRM:\n", genrm_responses[0])
        ddp_state.print("\nReward: ", evaluation_requests[0].reward)
        ddp_state.print("GenRM Scores: ", genrm_step_rewards[0])
        ddp_state.print("GenRM Valid: ", genrm_format_rewards[0])

        ddp_state.print("====================================")
        ddp_state.print("Average reward:", avg_reward)
        ddp_state.print("Average steps:", np.mean([len(r) for r in responses_steps]))
        ddp_state.print("Average GenRM corr. reward:", np.mean(genrm_corr_rewards))
        ddp_state.print("Average GenRM syn. reward:", np.mean(genrm_format_rewards))
        ddp_state.print(
            "Reward distribution:",
            print_metrics(
                [
                    np.mean(rewards)
                    for rewards in RequestUtils.group_by_query_id(
                        evaluation_requests, "reward"
                    ).values()
                ]
            ),
        )
        ddp_state.print("Finished: ", (100.0 * np.sum(finished)) / len(finished), "%")
        ddp_state.print("====================================")

        query_response, query_response_mask, response_mask, advantages = (
            create_joint_tensors_for_vppo(
                self.tokenizer,
                messages + genrm_messages,
                responses + genrm_responses,
                step_advantages=step_advantages + genrm_advantages,
                step_indices=responses_indices + genrm_response_indices,
                is_final=finished + genrm_finished,
            )
        )
        self.stats.accumulate(
            "avg_resp_length",
            response_mask[: len(messages)].sum(1).float().mean().item(),
        )

        # probabilities under the reference policy
        self.ref_model.to(self.model.device)
        ref_logprobs = get_logprobs(
            self.ref_model,
            query_response,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )
        self.ref_model.to("cpu")
        torch.cuda.empty_cache()

        # probabilities under the sampling policy
        old_logprobs = get_logprobs(
            self.model,
            query_response,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )

        outputs = (
            query_response,
            query_response_mask,
            response_mask,
            advantages,
            ref_logprobs,
            old_logprobs,
        )
        return outputs

    def compute_loss(self, episode_returns) -> float:
        (
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
            mb_advantage,
            mb_ref_logprobs,
            mb_old_logprobs,
        ) = episode_returns[:6]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            # trim from mb_query_response everything that is after the last 1 token in mb_query_response_mask
            max_tokens = mb_query_response_mask.sum(dim=1).max()
            mb_query_response = mb_query_response[:, :max_tokens]
            mb_query_response_mask = mb_query_response_mask[:, :max_tokens]
            mb_response_mask = mb_response_mask[:, :max_tokens]
            mb_advantage = mb_advantage[:, :max_tokens]
            # already shifted
            mb_ref_logprobs = mb_ref_logprobs[:, : max_tokens - 1]
            mb_old_logprobs = mb_old_logprobs[:, : max_tokens - 1]

            output = self.model(
                input_ids=mb_query_response,
                attention_mask=mb_query_response_mask,
                return_dict=True,
            )

            mb_logprobs = get_shifted_logprobs(
                output.logits,
                mb_query_response,
                mb_response_mask,
                temperature=self.temperature,
            )

            # Compute the PPO-clip loss
            mb_advantage = mb_advantage[:, 1:]
            log_ratio = mb_logprobs - mb_old_logprobs
            ratio = torch.exp(log_ratio)

            pg_losses1 = -mb_advantage * ratio
            pg_losses2 = -mb_advantage * torch.clamp(ratio, 0.9, 1.1)
            pg_losses = torch.max(pg_losses1, pg_losses2)

            labels_mask = mb_response_mask[:, 1:]
            per_token_kl = (
                torch.exp(mb_ref_logprobs - mb_logprobs)
                - (mb_ref_logprobs - mb_logprobs)
                - 1
            )

            per_token_loss = pg_losses + self.kl_ctl * per_token_kl
            pg_loss = masked_mean(per_token_loss, labels_mask, axis=1).mean()

            del output
            torch.cuda.empty_cache()

            self.stats.accumulate(
                "entropy",
                -((mb_logprobs * labels_mask).sum() / labels_mask.sum()).item(),
            )
            self.stats.accumulate(
                "kl_loss", masked_mean(per_token_kl, labels_mask, axis=1).mean().item()
            )
            return pg_loss
