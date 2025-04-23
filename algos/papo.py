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


GUESSER_SYSTEM_MESSAGE = "You are a helpful assistant. Your goal is to guess which model wrote the solution."


GUESSER_TEMPLATE = """Here is a problem and a solution. One out of four models wrote the solution. Your task is to guess which model wrote the solution.

Problem:
{problem}

Solution:
{solution}

Which model wrote the solution? Just write the model number (1, 2, 3, 4).
"""

MODEL_SYSTEM_TEMPLATE = """IMPORTANT: You are model {model_number} out of {total_models} models."""


class PAPO(Algo):
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

        messages_expand = []

        # Add model system messages
        for i in range(len(messages)):
            for j in range(self.k):
                messages_expand.append([
                    {
                        "role": "system",
                        "content": messages[i][0]['content'] + "\n\n" + MODEL_SYSTEM_TEMPLATE.format(
                            model_number=j + 1, total_models=self.k
                        ),
                    },
                    {
                        "role": "user",
                        "content": messages[i][1]['content'],
                    },
                ])

        # Gather first set of completions
        responses, finished = engine.chat(
            messages_expand,
            temperature=self.temperature,
            n=1,
            max_tokens=self.max_tokens,
            return_finished=True,
        )
        responses = flatten(responses)
        finished = flatten(finished)
        qids = repeat(range(len(messages)), self.k)
        labels = repeat(labels, self.k)
        messages = messages_expand

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

        guess_messages = []
        for i in range(len(messages)):
            user_message = messages[i][-1]["content"]
            assistant_message = responses[i]
            guess_messages.append(
                [
                    {
                        "role": "system",
                        "content": GUESSER_SYSTEM_MESSAGE,
                    },
                    {
                        "role": "user",
                        "content": GUESSER_TEMPLATE.format(
                            problem=user_message,
                            solution=assistant_message,
                        ),
                    },
                ]
            )

        guess_responses, guess_finished = engine.chat(
            guess_messages,
            temperature=0.,
            n=1,
            max_tokens=4,
            return_finished=True,
        )
        guess_responses = flatten(guess_responses)

        guess_responses = [g.strip() for g in guess_responses]
        guess_rewards = []
        for i in range(len(guess_responses)):
            guess_response = guess_responses[i].strip()
            try:
                guess_response = int(guess_response[0])
                guess_rewards.append(float(int(guess_response) == ((i % self.k) + 1)))
            except:
                guess_rewards.append(0)

        guess_advantages = np.array(guess_rewards)
        guess_advantages = guess_advantages.reshape(-1, self.k)
        guess_advantages = guess_advantages - guess_advantages.mean(
            axis=1, keepdims=True
        )
        if not self.drgpo:
            guess_advantages = guess_advantages / (
                guess_advantages.std(axis=1, keepdims=True) + 1e-8
            )
        guess_advantages = guess_advantages.reshape(-1).tolist()
        guess_advantages = np.ones(
            len(guess_advantages), dtype=np.float32
        ).tolist()

        # oracle responses, train the verifier with SFT!
        guess_responses = [str((i % self.k) + 1) for i in range(len(guess_responses))]

        rewards = (np.asarray(rewards) + 0.5 * np.asarray(guess_rewards)).reshape(-1, self.k)
        advantages = (rewards - rewards.mean(axis=1, keepdims=True))
        if not self.drgpo:
            advantages = advantages / (
                rewards.std(axis=1, keepdims=True) + 1e-8
            )
        advantages = advantages.reshape(-1).tolist()

        max_reward, avg_reward = RequestUtils.gather_max_avg_reward(evaluation_requests)

        self.stats.accumulate("avg_reward", avg_reward)
        self.stats.accumulate("guess_reward", np.mean(guess_rewards))

        ddp_state.print("====================================")
        ddp_state.print("Problem:\n", evaluation_requests[0].messages[-1]["content"])
        ddp_state.print("\nAnswer:\n", evaluation_requests[0].response)
        ddp_state.print("\nGuesser:\n", guess_responses[0])
        ddp_state.print("\nGuesser reward: ", guess_rewards[0])
        ddp_state.print("\nReward: ", evaluation_requests[0].reward)

        ddp_state.print("====================================")
        ddp_state.print("Average reward:", avg_reward)

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

        query_response, query_response_mask, response_mask = create_joint_tensors(
            self.tokenizer,
            messages + guess_messages,
            responses + guess_responses,
            is_final=finished + guess_finished,
        )
        advantages = torch.tensor(advantages + guess_advantages, dtype=torch.float32)
        self.stats.accumulate(
            "avg_resp_length",
            response_mask[: len(messages)].sum(1).float().mean().item(),
        )

        # probabilities under the reference policy
        # self.ref_model.to(self.model.device)
        ref_logprobs = get_logprobs(
            self.ref_model,
            query_response,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )
        # self.ref_model.to("cpu")
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

        is_guess = torch.zeros(
            query_response.shape[0], dtype=torch.bool, device=query_response.device
        )
        is_guess[len(messages):] = 1
        outputs = (
            query_response,
            query_response_mask,
            response_mask,
            is_guess,
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
            mb_is_guess,
            mb_advantage,
            mb_ref_logprobs,
            mb_old_logprobs,
        ) = episode_returns

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            # trim from mb_query_response everything that is after the last 1 token in mb_query_response_mask
            max_tokens = mb_query_response_mask.sum(dim=1).max()
            mb_query_response = mb_query_response[:, :max_tokens]
            mb_query_response_mask = mb_query_response_mask[:, :max_tokens]
            mb_response_mask = mb_response_mask[:, :max_tokens]
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
            log_ratio = mb_logprobs - mb_old_logprobs
            ratio = torch.exp(log_ratio)

            pg_losses1 = -mb_advantage * ratio
            pg_losses2 = -mb_advantage * torch.clamp(ratio, 0.9, 1.1)
            pg_losses = torch.max(pg_losses1, pg_losses2)

            pg_losses = pg_losses * ~mb_is_guess.unsqueeze(1) + (
                -mb_advantage * mb_logprobs
            ) * (mb_is_guess.unsqueeze(1))

            labels_mask = mb_response_mask[:, 1:]
            per_token_kl = (
                torch.exp(mb_ref_logprobs - mb_logprobs)
                - (mb_ref_logprobs - mb_logprobs)
                - 1
            )

            per_token_loss = pg_losses + self.kl_ctl * per_token_kl * mb_is_guess.unsqueeze(1)
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
