from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from gen_utils import GeneratorClient
from ddp_utils import ddp_state
from utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    AccumulatorDict,
    compute_kl_divergence,
    flatten,
    get_logprobs,
    print_metrics,
    repeat,
)
from dataset.data_utils import create_joint_tensors
from ddp_utils import rank_zero_only
from algos.algo import Algo, Request, RequestUtils


class RFT(Algo):
    @classmethod
    def add_parser_args(self, parser):
        parser.add_argument(
            "--kl_ctl",
            type=float,
            default=0.0,
            help="Target KL divergence between policy and reference policy.",
        )
        parser.add_argument(
            "--pfiltr",
            action="store_true",
            help="Filter prompts that are either too difficult or too easy (0.2 < r < 0.8).",
        )
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
        self.reward_func = reward_func
        self.k = k
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.kl_ctl = algo_kwargs.get("kl_ctl", 0.0)
        self.pfiltr = algo_kwargs.get("pfiltr", False)
        self.stats = AccumulatorDict()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        if self.kl_ctl:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map=device
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

    @torch.no_grad()
    def gather_episodes(
        self,
        messages: List[List[Dict[str, str]]],
        labels: List[str],
    ):
        engine = GeneratorClient.get()

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
        query_ids = repeat(range(len(messages)), self.k)
        labels = repeat(labels, self.k)
        messages = repeat(messages, self.k)

        print("====================================")
        print("Example response:")
        print(messages[0])
        print(responses[0])

        evaluation_requests = []
        for query_id, query_messages, query_response, label, finish in zip(
            query_ids, messages, responses, labels, finished
        ):
            evaluation_requests.append(
                Request(
                    query_id=query_id,
                    messages=query_messages,
                    response=query_response,
                    label=label,
                    finished=finish,
                )
            )

        rewards = list(self.reward_func(evaluation_requests))
        RequestUtils.populate(evaluation_requests, rewards, "reward")

        max_reward, avg_reward = RequestUtils.gather_max_avg_reward(evaluation_requests)

        self.stats.accumulate("avg_reward", avg_reward)
        self.stats.accumulate("max_reward", max_reward)
        self.stats.accumulate("finished", (100.0 * np.sum(finished) / len(finished)))

        ddp_state.print("====================================")
        ddp_state.print("Optimistic reward:", max_reward)
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

        if self.pfiltr:
            evaluation_requests = self.prompt_filtering(evaluation_requests)

        queries = []
        responses = []
        advantages = []
        finished = []
        for request in evaluation_requests:
            if request.reward <= 0:
                continue

            queries.append(request.messages)  # the query are the messages
            responses.append(request.response)
            advantages.append(request.reward)
            finished.append(request.finished)

        query_response_tensors, query_response_mask, response_mask = (
            create_joint_tensors(
                self.tokenizer,
                queries,
                responses,
                is_final=finished,  # superflous here given that all unfinished have rewards 0
            )
        )
        advantages = torch.tensor(advantages, dtype=torch.float32)

        outputs = (
            query_response_tensors,
            query_response_mask,
            response_mask,
            advantages,
        )

        if self.kl_ctl:
            ref_logprobs = get_logprobs(
                self.ref_model,
                query_response_tensors,
                query_response_mask,
                response_mask,
                temperature=self.temperature,
                reduction="none",
            )
            outputs += (ref_logprobs,)

        return outputs

    def prompt_filtering(self, evaluation_requests: List[Request]) -> List[Request]:
        """Filter prompts that are either too difficult or too easy (0.2 < r < 0.8)."""
        rewards_by_query_id = RequestUtils.group_by_query_id(
            evaluation_requests, "reward"
        )
        filtered_query_ids = [
            query_id
            for query_id, rewards in rewards_by_query_id.items()
            if 0.2 <= np.mean(rewards) <= 0.8
        ]
        evaluation_requests = [
            req for req in evaluation_requests if req.query_id in filtered_query_ids
        ]
        return evaluation_requests

    def compute_loss(self, episode_returns) -> float:
        (
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
            mb_advantage,
        ) = episode_returns[:4]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            # trim from mb_query_response everything that is after the last 1 token in mb_query_response_mask
            max_tokens = mb_query_response_mask.sum(dim=1).max()

            mb_query_response = mb_query_response[:, :max_tokens]
            mb_query_response_mask = mb_query_response_mask[:, :max_tokens]
            mb_response_mask = mb_response_mask[:, :max_tokens]

            output = self.model(
                input_ids=mb_query_response,
                attention_mask=mb_query_response_mask,
                return_dict=True,
            )
            logits = output.logits / (self.temperature + 1e-7)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)

            shift_logits = logits[:, :-1]
            shift_labels = mb_query_response[:, 1:]
            shift_labels_mask = mb_response_mask[:, 1:]
            mb_logprobs = torch.gather(
                shift_logits, 2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            loss = -mb_logprobs * mb_advantage.unsqueeze(-1)
            loss = (loss * shift_labels_mask).sum() / shift_labels_mask.sum()

            if self.kl_ctl:
                mb_ref_logprobs = episode_returns[-1]
                mb_ref_logprobs = mb_ref_logprobs[:, : max_tokens - 1]
                ref_kl = compute_kl_divergence(
                    mb_ref_logprobs, mb_logprobs, mask=shift_labels_mask
                )
                loss += self.kl_ctl * ref_kl
                self.stats.accumulate("kl_loss", ref_kl.item())
                del mb_ref_logprobs

            del (
                mb_logprobs,
                shift_logits,
                shift_labels,
                shift_labels_mask,
                logits,
                output,
            )
            torch.cuda.empty_cache()
            return loss
