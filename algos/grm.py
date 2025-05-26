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


VERIFIER_SYSTEM_MESSAGE = "You are a helpful assistant. Think about the usefulness of each step in the solution."


VERIFIER_TEMPLATE = """
Here is a problem and a solution chunked in steps.

Problem:
{problem}

Solution:
{step_solution}

Your task is to determine whether every step of the solution is correct and useful to solve the problem.
Format your answer as a list of scores (from 0 to 5), 0 being the worst and 5 being the best, where each score corresponds to a step of the solution.
Think about your scores between <think> and </think> and then write the scores down for every step between <answer> and </answer>.

e.g. if there are 3 steps in the solution, your answer should look like this:

<answer>[score step 1, score step 2, score step 3]</answer>
"""


VERIFIER_TEMPLATE_ASS = "<think>"


def extract_genrm_scores(
    response_chunks: List[List[str]],
    response_rewards: List[float],
    genrm_responses: List[str],
):
    # Parse verifier responses to extract scores
    step_rewards = []
    format_rewards = []
    correctness_rewards = []
    for i, response in enumerate(genrm_responses):
        response = "<think>" + response
        format_correct = False
        parsed_scores = [0.0] * len(response_chunks[i])
        try:
            answer_start = response.rfind("<answer>")
            answer_end = response.rfind("</answer>")
            if answer_start != -1:
                if answer_end == -1:
                    answer_end = len(response)
                answer_text = response[
                    answer_start + len("<answer>") : answer_end
                ].strip()
                # Extract the list of scores
                if answer_text.startswith("[") and answer_text.endswith("]"):
                    scores_text = answer_text[1:-1]
                    scores = [
                        np.clip(float(s.strip()), 0, 5) / 5
                        for s in scores_text.split(",")
                    ]
                    # Verify if the number of scores matches the number of chunks
                    if len(scores) == len(response_chunks[i]):
                        parsed_scores = scores
                        format_correct = True
        except Exception:
            pass

        format_rewards.append(format_correct)
        correctness_rewards.append(
            format_correct
            * (1.0 - np.abs(np.mean(parsed_scores) - float(response_rewards[i]) / 2))
        )
        step_rewards.append(parsed_scores)

    return step_rewards, format_rewards, correctness_rewards


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
        responses_steps, responses_indices = vppo_balanced_chunk(
            responses,
            max_chunks=6,
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
                    {"role": "system", "content": VERIFIER_SYSTEM_MESSAGE},
                    {
                        "role": "user",
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
            extract_genrm_scores(
                responses_steps,
                rewards,
                genrm_responses,
            )
        )

        genrm_rewards = [
            float(f) + r
            for f, r in zip(genrm_format_rewards, genrm_corr_rewards)
        ]  # between 0 and 2

        step_rewards = [
            [step_reward + seq_reward for step_reward in step_rewards]
            for step_rewards, seq_reward in zip(genrm_step_rewards, rewards)
        ]
        step_advantages = []
        # Compute per‑query, z‑scored advantages
        for i in range(0, len(step_rewards), self.k):
            fi = []
            for s in step_rewards[i : i + self.k]:
                fi.extend(s)

            fi = np.array(fi)
            avg = np.mean(fi)
            std = np.std(fi) + 1e-8
            for j, s in enumerate(step_rewards[i : i + self.k]):
                step_advantages.append([(s_val - avg) / std for s_val in s])

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
