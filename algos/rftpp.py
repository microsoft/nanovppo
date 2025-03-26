from collections import defaultdict
import copy
import torch
from typing import List, Dict
import numpy as np

from algos.algo import Request, RequestUtils
from data_utils import create_joint_tensors, pad_query_and_response
from ddp_utils import rank_zero_only
from gen_utils import GenerationBackend
from utils import (
    DEFAULT_TEMP,
    get_ending_tokens,
    get_logprobs,
    flatten,
    print_metrics,
    repeat,
)
from algos.vppo import vppo_balanced_chunk
from algos.rft import RFT


def chunk_and_compute_entropies(
    model,
    tokenizer,
    queries,
    responses,
    is_final,
    temperature=DEFAULT_TEMP,
):
    from utils import get_entropies
    from bisect import bisect_left

    if len(queries) != len(responses):
        raise ValueError("Queries and responses must be the same length.")

    is_final = is_final or [True] * len(queries)

    # if the latest message is not from the assistant, we need to add the assistant token
    ending_tokens = get_ending_tokens(tokenizer)
    tokenized_queries = []

    for query in queries:
        tok_query = torch.tensor(
            tokenizer.apply_chat_template(
                query,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        tokenized_queries.append(tok_query)

    # response is stripped by apply_chat_template...
    responses = [
        r.strip() + (ending_tokens if f else "") for r, f in zip(responses, is_final)
    ]

    # we report the character offsets, these will be compared to step offsets return by chunking
    tokenized_responses = []
    tokenized_offsets = []
    for response in responses:
        out = tokenizer.encode_plus(
            response, return_offsets_mapping=True, add_special_tokens=False
        )
        tokenized_responses.append(torch.tensor(out["input_ids"]))
        tokenized_offsets.append(out["offset_mapping"])

    steps, step_indices = vppo_balanced_chunk(responses, 4, "\n\n")

    query_response_tensors, query_response_mask, response_mask = pad_query_and_response(
        tokenized_queries,
        tokenized_responses,
        tokenizer.pad_token_id,
        padding_side="right",
    )

    entropies = get_entropies(
        model,
        query_response_tensors,
        query_response_mask,
        response_mask,
        temperature=temperature,
    )

    step_entropies = []
    for query_id in range(len(queries)):
        q_response_step_indices = step_indices[query_id]
        q_response_offsets = tokenized_offsets[query_id]
        total_len = query_response_mask[query_id].sum()
        query_len = total_len - response_mask[query_id].sum()

        q_step_entropies = [[] for _ in range(len(q_response_step_indices))]
        for token_pos, offset in enumerate(q_response_offsets):
            pos = bisect_left(q_response_step_indices, offset[0])

            # If pos == len(q_response_step_indices), it means offset[0] is greater
            # than all elements in q_response_step_indices, so we clamp to the last index.
            if pos == len(q_response_step_indices):
                pos -= 1

            q_step_entropies[pos].append(entropies[query_id][query_len + token_pos])

        step_entropies.append([np.mean(ents) for ents in q_step_entropies])

    return steps, step_indices, step_entropies


class ILPRequest(Request):
    policy_type: str = "ilp"
    orig_messages: List[Dict[str, str]] = None
    messages_no_prompt: List[Dict[str, str]] = None
    used_prompt: str = None
    full_response: str = None


class PSPRequest(Request):
    policy_type: str = "psp"
    orig_messages: List[Dict[str, str]] = None
    full_response: str = None


class ILP:
    """Inserts a reflection prompt before the last answer message, e.g.:

    x, s_1, s_2, s_3, s_4
                  |
             "wait a min", s'_4, s'_5, ...
    """

    reflect_prompts = [
        "Mmmh, wait a minute...",
        "Let's analyze alternatives. ",
        "Did I miss something? Let me check...",
        "This doesn't seem right, what if",
        "Let's rethink this. ",
    ]
    max_chunks: int = 4
    sep: str = "\n\n"
    pmode: int = "cp"

    def get_requests(self, requests: List[Request]):
        react_requests = []
        responses_steps, _ = vppo_balanced_chunk(
            [r.response for r in requests], self.max_chunks, self.sep
        )
        for request, steps in zip(requests, responses_steps):
            if request.reward > 0:
                continue

            prompt = np.random.choice(self.reflect_prompts)
            insert_pos = len(steps) - 1 if len(steps) > 1 else 1
            insert_msg = self.sep.join(steps[:insert_pos] + [prompt])
            insert_msg_no_prompt = self.sep.join(steps[:insert_pos]) + self.sep
            react_requests.append(
                ILPRequest(
                    query_id=request.query_id,
                    messages=request.messages
                    + [{"role": "assistant", "content": insert_msg}],
                    orig_messages=request.messages,
                    messages_no_prompt=request.messages
                    + [{"role": "assistant", "content": insert_msg_no_prompt}],
                    used_prompt=prompt,
                    label=request.label,
                )
            )
        return react_requests

    def process_responses(self, react_requests: List[ILPRequest]):
        i = 0
        for request in react_requests:
            # for visibility, build the full response
            request.full_response = request.messages[-1]["content"] + request.response

            if self.pmode == "cpp":  # predict react prompt + response
                request.response = request.used_prompt + request.response
                request.messages = request.messages_no_prompt
            elif (
                self.pmode == "all"
            ):  # predict everything, previous + prompt + response
                request.response = request.full_response
                request.messages = request.orig_messages
            else:
                raise ValueError(f"Invalid pmode {self.pmode}.")
        return react_requests


class PSP:
    """Process sampling policy, splits the answer and re-sample the suffix, e.g.:

    x, s_1, s_2, s_3, s_4
            |
            s'_2, s'_3, s'_4
    """

    max_chunks: int = 4
    sep: str = "\n\n"
    pmode: str = (
        "cp"  # cp: predict response given partial context, all: predict everything
    )

    def get_requests(self, requests: List[Request]):
        aug_requests = []
        responses_steps, _ = vppo_balanced_chunk(
            [r.response for r in requests], self.max_chunks, self.sep
        )
        for request, steps in zip(requests, responses_steps):
            if request.reward > 0:
                continue

            if len(steps) == 1:
                aug_messages = request.messages
            else:
                step = np.random.randint(1, len(steps))
                prefix = self.sep.join(steps[:step]) + self.sep
                aug_messages = request.messages + [
                    {"role": "assistant", "content": prefix}
                ]

            aug_requests.append(
                PSPRequest(
                    query_id=request.query_id,
                    messages=aug_messages,
                    orig_messages=request.messages,
                    label=request.label,
                )
            )
        return aug_requests

    def process_responses(self, requests: List[PSPRequest]):
        for request in requests:
            if request.messages[-1]["role"] == "assistant":
                request.full_response = (
                    request.messages[-1]["content"] + request.response
                )
            else:
                request.full_response = request.response

            if self.pmode == "all":  # predict everything context and completion
                request.response = request.full_response
                request.messages = request.orig_messages
            elif self.mode == "cp":
                continue
            else:
                raise ValueError(f"Invalid pmode {self.pmode}.")
        return requests


class RFTPP(RFT):
    @classmethod
    def add_parser_args(self, parser):
        parser = RFT.add_parser_args(parser)
        parser.add_argument(
            "--pols",
            type=str,
            default="ilp",
            help="Define augmentation policies to use for RFT",
        )
        parser.add_argument(
            "--pmode",
            type=str,
            default="cpr",
            help="Pmode for the policies, e.g. all, cp, cpr",
        )
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.policies = []

        for pol in kwargs["pols"].split(","):
            if pol == "ilp":
                ILP.pmode = kwargs["pmode"]
                self.policies.append(ILP)
            elif pol == "psp":
                PSP.pmode = kwargs["pmode"]
                self.policies.append(PSP)
            else:
                raise ValueError(f"Invalid policy {pol}")

    @rank_zero_only
    @torch.no_grad()
    def gather_episodes(
        self,
        messages: List[List[Dict[str, str]]],
        labels: List[str],
    ):
        vllm = GenerationBackend.get()

        # Gather first set of completions
        responses, finished = vllm.chat(
            messages,
            temperature=self.temperature,
            n=self.k,
            max_tokens=self.max_tokens,
            return_finished=True,
        )
        responses = flatten(responses)
        finished = flatten(finished)
        query_ids = repeat(range(len(messages)), self.k)
        messages = repeat(messages, self.k)
        labels = repeat(labels, self.k)

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
                    finished=finish,
                    label=label,
                )
            )

        # Compute rewards
        rewards = list(self.reward_func(evaluation_requests))
        RequestUtils.populate(evaluation_requests, rewards, "reward")

        # Run augmentation policies
        policy_requests_all = []
        for policy_class in self.policies:
            policy = policy_class()
            policy_requests = policy.get_requests(evaluation_requests, self)

            if policy_requests:
                print("====================================")
                print(
                    f"Processing {len(policy_requests)} policy {policy_class.__name__} requests"
                )
                print("====================================")

                policy_responses, policy_finished = vllm.chat(
                    [m.messages for m in policy_requests],
                    temperature=self.temperature,
                    n=1,  # we use 1 as doubles are already accounted for
                    max_tokens=self.max_tokens,
                    return_finished=True,
                )
                policy_responses = flatten(policy_responses)
                policy_finished = flatten(policy_finished)
                RequestUtils.populate(policy_requests, policy_responses, "response")
                RequestUtils.populate(policy_requests, policy_finished, "finished")
                policy_requests = policy.process_responses(policy_requests)
                policy_requests_all.extend(policy_requests)

        # compute rewards for the react messages
        policy_rewards = list(self.reward_func(policy_requests_all))
        RequestUtils.populate(policy_requests_all, policy_rewards, "reward")

        # do some printing
        i = 0
        for request in policy_requests_all:
            if request.reward > 0 and i < 2:
                print("====================================")
                print("Example response:")
                print(request.messages)
                print(request.full_response)
                print("Reward: ", request.reward)
                i += 1

        # combine the evaluation requests and the react requests
        all_eval_requests = evaluation_requests + policy_requests_all

        pre_max_reward, pre_avg_reward = RequestUtils.gather_max_avg_reward(
            evaluation_requests
        )
        post_max_reward, post_avg_reward = RequestUtils.gather_max_avg_reward(
            all_eval_requests
        )

        self.stats.accumulate("avg_reward", post_avg_reward)
        self.stats.accumulate("opt_reward", post_max_reward)
        self.stats.accumulate("pre_opt_reward", pre_max_reward)
        self.stats.accumulate("pre_avg_reward", pre_avg_reward)
        self.stats.accumulate("finished", (100.0 * np.sum(finished) / len(finished)))

        print("====================================")
        print("(Pre) Optimistic reward:", pre_max_reward)
        print("(Pre) Average reward:", pre_avg_reward)
        print("(Post) Optimistic reward:", post_max_reward)
        print("(Post) Average reward:", post_avg_reward)
        print(
            "(Pre) Avg Reward distribution: ",
            print_metrics(
                [
                    np.mean(rewards)
                    for rewards in RequestUtils.group_by_query_id(
                        evaluation_requests, "reward"
                    ).values()
                ],
                min_value=0,
                max_value=1,
            ),
        )
        print(
            "(Pre) Opt Reward distribution: ",
            print_metrics(
                [
                    np.max(rewards)
                    for rewards in RequestUtils.group_by_query_id(
                        evaluation_requests, "reward"
                    ).values()
                ],
                min_value=0,
                max_value=1,
            ),
        )
        print(
            "(Post) Opt Reward distribution:",
            print_metrics(
                [
                    np.max(rewards)
                    for rewards in RequestUtils.group_by_query_id(
                        all_eval_requests, "reward"
                    ).values()
                ],
                min_value=0,
                max_value=1,
            ),
        )
        print(
            "Num rewards per query:",
            print_metrics(
                [
                    len([r for r in requests if r.reward > 0])
                    for requests in RequestUtils.group_by_query_id(
                        all_eval_requests
                    ).values()
                ]
            ),
        )

        # group by policy type
        for pol_type, pol_requests in RequestUtils.group_by_field(
            policy_requests_all, "policy_type"
        ).items():
            print(
                f"Policy {pol_type}, "
                f"Avg. Reward: {np.mean([r.reward for r in pol_requests])}"
            )

        print("Finished: ", (100.0 * np.sum(finished)) / len(finished), "%")
        print("====================================")

        if self.pfiltr:
            all_eval_requests = self.prompt_filtering(all_eval_requests)

        queries = []
        responses = []
        advantages = []
        finished = []
        for request in all_eval_requests:
            if request.reward <= 0:
                continue

            queries.append(request.messages)
            responses.append(request.response)
            advantages.append(request.reward)
            finished.append(request.finished)

        query_response_tensors, query_response_mask, response_mask = (
            create_joint_tensors(
                self.tokenizer,
                queries,
                responses,
                is_final=finished,
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
