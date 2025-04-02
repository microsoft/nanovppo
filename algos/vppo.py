from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import torch
import copy
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from ddp_utils import rank_zero_only
from gen_utils import GenerationBackend
from utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    AccumulatorDict,
    get_logprobs,
    get_shifted_logprobs,
    masked_mean,
    masked_whiten,
    repeat,
    flatten,
    compute_kl_divergence,
)
from datasets.data_utils import create_joint_tensors, get_ending_tokens, pad_query_and_response
from algos.algo import Algo, Request, RequestUtils


def check_continuation_prompt(tokenizer: AutoTokenizer):
    """
    For VPPO, we need to make sure that the tokenizer chat template
    supports continuation prompts.
    """
    message = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is"},
    ]
    t1 = tokenizer.apply_chat_template(
        message, continue_final_message=True, tokenize=False
    )
    t2 = tokenizer.apply_chat_template(
        message, continue_final_message=False, tokenize=False
    )
    assert t1 != t2, "Continuation prompt should be different!"


def vppo_treetune_chunk(responses, max_chunks=32, merge_every_n_chunk=2):
    from algos.vppo_utils import split_solution_inplace

    final_steps = []
    final_indices = []
    for response in responses:
        try:
            indices = split_solution_inplace(response)
        except Exception as e:
            print(f"Error splitting solution: {e}")
            _, indices = vppo_balanced_chunk([response], max_chunks=max_chunks)
            indices = indices[0]

        # Merge every n steps
        merged_step_indices = []
        for i in range(0, len(indices), merge_every_n_chunk):
            if len(merged_step_indices) >= max_chunks:
                break
            merged_step_indices.append(indices[i])

        if merged_step_indices[-1] != indices[-1]:
            if len(merged_step_indices) >= max_chunks:
                merged_step_indices.pop()
            merged_step_indices.append(indices[-1])

        steps = [
            response[merged_step_indices[i] : merged_step_indices[i + 1]]
            for i in range(len(merged_step_indices) - 1)
        ]

        final_indices.append(merged_step_indices[1:])
        final_steps.append(steps)
    return final_steps, final_indices


def vppo_balanced_chunk(responses, max_chunks=4, sep="\n\n"):
    final_steps = []
    final_indices = []

    for response in responses:
        # Handle if response is a list
        if isinstance(response, list):
            response = response[0]

        steps = response.split(sep)

        # Merge leading empty steps into the first non-empty step
        first_non_empty_step_idx = None
        for i, step in enumerate(steps):
            if step.strip() != "":
                first_non_empty_step_idx = i
                break

        if first_non_empty_step_idx is not None and first_non_empty_step_idx > 0:
            new_first_step = sep.join(steps[: first_non_empty_step_idx + 1])
            steps = [new_first_step] + steps[first_non_empty_step_idx + 1 :]

        # Calculate total length including separators
        total_length = sum(len(s) for s in steps) + (len(steps) - 1) * len(sep)
        target_length = max(1, total_length // max_chunks)  # avoid div-by-zero

        merged = []
        cur_chunk = []
        cur_length = 0
        for i, s in enumerate(steps):
            # Length of s plus sep if not first in chunk
            add_len = len(s) + (len(sep) if cur_chunk else 0)
            # If adding s exceeds target and we haven't formed all but last chunk, close current
            if (
                cur_chunk
                and (cur_length + add_len > target_length)
                and (len(merged) < max_chunks - 1)
            ):
                merged.append(sep.join(cur_chunk))
                cur_chunk = [s]
                cur_length = len(s)
            else:
                if not cur_chunk:
                    cur_chunk = [s]
                    cur_length = len(s)
                else:
                    cur_chunk.append(s)
                    cur_length += add_len

        if cur_chunk:
            merged.append(sep.join(cur_chunk))

        steps = merged

        # Build character indices
        indices = [0]
        running_len = 0
        for i, chunk in enumerate(steps):
            running_len += len(chunk) + (0 if i == 0 else len(sep))
            indices.append(running_len)

        # Checks
        assert indices[-1] == len(response), f"{indices[-1]} != {len(response)}"
        assert (
            sep.join(steps) == response
        ), "Concatenating chunk steps should equal original!"

        final_steps.append([step + sep for step in steps[:-1]] + [steps[-1]])
        final_indices.append(indices[1:])

    return final_steps, final_indices


def print_chunk_statistics(chunks):
    num_responses = len(chunks)
    chunk_counts = []
    chunk_lengths = []

    for i, chunk_list in enumerate(chunks):
        num_chunks = len(chunk_list)
        total_len = sum(len(ch) for ch in chunk_list)
        avg_len = total_len / num_chunks if num_chunks > 0 else 0
        chunk_counts.append(num_chunks)
        chunk_lengths.append(avg_len)  # average chunk length for this response

    # Print overall summary
    overall_num_chunks = sum(chunk_counts)
    overall_avg_chunks = overall_num_chunks / num_responses if num_responses > 0 else 0
    overall_avg_len = sum(chunk_lengths) / num_responses if num_responses > 0 else 0
    min_chunks = min(chunk_counts) if chunk_counts else 0
    max_chunks = max(chunk_counts) if chunk_counts else 0

    print("\n=== Overall Chunk Statistics ===")
    print(f"Total responses: {num_responses}")
    print(f"Total chunks across all responses: {overall_num_chunks}")
    print(f"Average chunks per response: {overall_avg_chunks:.2f}")
    print(f"Min chunks in any response: {min_chunks}")
    print(f"Max chunks in any response: {max_chunks}")
    print(f"Average chunk length (across responses): {overall_avg_len:.2f} characters")


def create_joint_tensors_for_vppo(
    tokenizer,
    queries: Union[List[Dict[str, str]]],
    responses: List[str],
    step_advantages: List[List[float]],
    step_indices: List[List[int]],
    is_final=None,
):
    """
    Create joint tensors for VPPO by spreading advantages at the token level.
    To do so, we need to know the character indices of the reasoning steps in the response.
    """
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
        r + (ending_tokens if f else "") for r, f in zip(responses, is_final)
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

    query_response_tensors, query_response_mask, response_mask = pad_query_and_response(
        tokenized_queries,
        tokenized_responses,
        tokenizer.pad_token_id,
        padding_side="right",
    )

    advantages = torch.zeros_like(query_response_mask, dtype=torch.float32)
    for query_id in range(len(queries)):
        q_response_step_indices = step_indices[query_id]
        q_response_offsets = tokenized_offsets[query_id]
        q_step_advantages = step_advantages[query_id]
        q_response_text = responses[query_id]
        tokenized_query = tokenized_queries[query_id]
        tokenized_response = tokenized_responses[query_id]

        assert len(q_response_step_indices) == len(q_step_advantages)

        if len(q_response_step_indices) > 0:
            char_advantages = np.ones(len(q_response_text)) * -7777777
            q_response_step_indices = [0] + q_response_step_indices

            for i, (start, end) in enumerate(zip(q_response_step_indices[:-1], q_response_step_indices[1:])):
                char_advantages[start:end] = q_step_advantages[i]

            # fill the end of message tokens with the last step advantage
            if is_final[query_id]:
                char_advantages[q_response_step_indices[-1] :] = q_step_advantages[-1]

            assert np.all(char_advantages != -7777777)

            # Find the advantage of response tokens from the advantage of its characters
            token_advantages = [None] * len(tokenized_response)
            for i in range(len(token_advantages)):
                start_char_pos_of_token = q_response_offsets[i][0]
                token_advantages[i] = char_advantages[start_char_pos_of_token]

            response_indices = response_mask[query_id].nonzero(as_tuple=True)[0]
            advantages[query_id, response_indices] = torch.tensor(
                token_advantages, dtype=torch.float32
            )

    return (
        query_response_tensors,
        query_response_mask,
        response_mask,
        advantages,
    )


@dataclass
class VPPOResult:
    values: List[List[float]]
    advantages: List[List[float]]
    stats: Dict[str, float] = None
    requests: List[Dict] = None


@dataclass
class VPPORequest(Request):
    step_id: int = None


def compute_vppo_value_estimation(
    messages: List[List[Dict]],
    responses_steps: List[List[str]],
    labels: List[str],
    reward_func: Callable = None,
    num_samples: int = 3,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    return_requests: bool = False,
    return_stats: bool = False,
) -> VPPOResult:
    """Computes value estimation using MC sampling for a set of responses.

    Args:
        messages: List of messages (prompts).
        responses_steps: List of responses steps for each response.
        labels: List of ground-truth labels for each response.
        sep: Separator string used to split response steps.
        reward_func: Reward function to compute rewards for each response.
        num_samples: Number of samples to use for value estimation.
        temperature: Temperature to use for sampling.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        List of value estimates for each response and response step.
    """

    vppo_requests = []
    for query_id, (query_messages, query_steps, query_label) in enumerate(
        zip(messages, responses_steps, labels)
    ):
        for step_id in range(len(query_steps)):
            # append prefix so far if any
            if step_id > 0:
                past_messages = query_messages + [
                    {
                        "role": "assistant",
                        "content": "".join(query_steps[:step_id]),
                    }
                ]
            else:
                past_messages = query_messages

            vppo_requests.append(
                VPPORequest(
                    query_id=query_id,
                    step_id=step_id,
                    messages=past_messages,
                    label=query_label,
                )
            )

    vppo_responses, vppo_finished = GenerationBackend.get().chat(
        [r.messages for r in vppo_requests],
        temperature=temperature,
        n=num_samples,
        max_tokens=max_tokens,
        return_finished=True,
    )
    vppo_responses = flatten(vppo_responses)
    vppo_finished = flatten(vppo_finished)
    vppo_requests = repeat(vppo_requests, num_samples)
    RequestUtils.populate(vppo_requests, vppo_responses, "response")
    RequestUtils.populate(vppo_requests, vppo_finished, "finished")

    # one per query id
    orig_requests = [
        Request(response="".join(steps), label=label)
        for steps, label in zip(responses_steps, labels)
    ]

    # compute rewards
    rewards = list(reward_func(vppo_requests + orig_requests))
    vppo_rewards = rewards[: len(vppo_requests)]
    orig_rewards = rewards[len(vppo_requests) :]
    RequestUtils.populate(vppo_requests, vppo_rewards, "reward")

    # initialize value array for each message
    assert len(orig_rewards) == len(messages)

    vppo_values = []
    for i in range(len(messages)):
        # last value is the original response reward
        vppo_values.append([0.0] * len(responses_steps[i])  + [int(orig_rewards[i])])

    for request in vppo_requests:
        # rewards for same query id and step id are averaged across num_samples
        vppo_values[request.query_id][request.step_id] += request.reward / num_samples

    vppo_advantages = [
        [v_n - v_p for v_p, v_n in zip(v[:-1], v[1:])] for v in vppo_values
    ]
    for steps, adv in zip(responses_steps, vppo_advantages):
        assert len(adv) == len(steps)

    output = VPPOResult(vppo_values, vppo_advantages)
    if return_requests:
        output.requests = vppo_requests

    if return_stats:
        vppo_stats = {}
        vppo_stats["avg_full_reward"] = np.mean([v[-1] for v in vppo_values])
        vppo_stats["avg_reward"] = np.mean(
            [
                np.mean(v + [orig_rewards[id]])
                for id, v in RequestUtils.group_by_query_id(
                    vppo_requests, "reward"
                ).items()
            ]
        )
        vppo_stats["opt_reward"] = np.mean(
            [
                np.max(v + [orig_rewards[id]])
                for id, v in RequestUtils.group_by_query_id(
                    vppo_requests, "reward"
                ).items()
            ]
        )
        output.stats = vppo_stats
    return output


class VPPO(Algo):
    @classmethod
    def add_parser_args(self, parser):
        parser.add_argument(
            "--kl_ctl", type=float, default=0.0001, help="KL term control"
        )
        parser.add_argument(
            "--kl_max", type=int, default=10, help="Clip KL divergence to this value"
        )
        parser.add_argument(
            "--vppok", type=int, default=3, help="Samples for VPPO value estimation"
        )
        parser.add_argument(
            "--max_chunks",
            type=int,
            default=32,
            help="How many steps to use for VPPO chunking.",
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        # Check that the continuation prompt is working, as vppo needs completions
        # of partial assistant messages
        check_continuation_prompt(self.tokenizer)

        self.k = k
        self.vppok = algo_kwargs.get("vppok")
        self.kl_ctl = algo_kwargs.get("kl_ctl")
        self.kl_max = algo_kwargs.get("kl_max")
        self.max_chunks = algo_kwargs.get("max_chunks", 4)
        self.reward_func = reward_func
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stats = AccumulatorDict()

        print('KL control:', self.kl_ctl)
        print('KL max:', self.kl_max)
        print('VPPO samples:', self.vppok)
        print('Max chunks:', self.max_chunks)

    @torch.no_grad
    def gather_episodes(
        self,
        messages,
        labels,
    ):
        vllm = GenerationBackend.get()

        responses, finished = vllm.chat(
            messages,
            temperature=self.temperature,
            n=self.k,
            max_tokens=self.max_tokens,
            return_finished=True,
        )
        responses = flatten(responses)
        finished = flatten(finished)
        messages = repeat(messages, self.k)
        labels = repeat(labels, self.k)

        # Now we chunk the responses and build the VLLM requests for each partial prefix
        responses_steps, step_indices = vppo_treetune_chunk(
            responses, max_chunks=self.max_chunks,
        )

        # Print an example response!
        print("====================================")
        print("Example response:")
        print(messages[0])
        for i, step in enumerate(responses_steps[0]):
            print(f"Step {i}:")
            print(f"---\n{step}\n---")
        print("====================================")
        print_chunk_statistics(responses_steps)

        vppo_result = compute_vppo_value_estimation(
            messages,
            responses_steps,
            labels,
            reward_func=self.reward_func,
            num_samples=self.vppok,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            return_stats=True,
        )
        step_advantages = vppo_result.advantages

        # We tokenize the queries and responses and create per-token advantages
        query_response, query_response_mask, response_mask, advantages = (
            create_joint_tensors_for_vppo(
                self.tokenizer,
                messages,
                responses,
                step_advantages=step_advantages,
                step_indices=step_indices,
                is_final=finished,
            )
        )

        # print optimistic reward (max reward across generations for each query_id)
        self.stats.accumulate("opt_reward", vppo_result.stats["opt_reward"])
        self.stats.accumulate("avg_reward", vppo_result.stats["avg_full_reward"])

        print("====================================")
        print("Opt reward:", vppo_result.stats["opt_reward"])
        print("Average reward:", vppo_result.stats["avg_reward"])
        print("Average full reward:", vppo_result.stats["avg_full_reward"])
        print("====================================")

        # whiten the advantages
        advantages = masked_whiten(advantages, response_mask, unbiased_variance=True)

        # probabilities under the reference policy
        ref_logprobs = get_logprobs(
            self.ref_model,
            query_response,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )

        # probabilities under the sampling policy
        old_logprobs = get_logprobs(
            self.model,
            query_response,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )

        return (
            query_response,
            query_response_mask,
            response_mask,
            advantages,
            ref_logprobs,
            old_logprobs,
        )

    def compute_loss(self, episode_returns) -> float:
        (
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
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
            mb_advantage = mb_advantage[:, :max_tokens]

            # already shifted
            mb_ref_logprobs = mb_ref_logprobs[:, : max_tokens - 1]
            mb_old_logprobs = mb_old_logprobs[:, : max_tokens - 1]

            output = self.model(
                input_ids=mb_query_response,
                attention_mask=mb_query_response_mask,
                return_dict=True,
            )

            # response_mask:       0 0 1 1 1
            # query_response:      1 3 5 7 9
            # advantages:          0 0 0.3 0.3 0.3
            # logits:              1 3 5 7
            # labels:              3 5 7 9
            # shift_response_mask: 0 1 1 1
            # advantages:          0 0.3 0.3 0.3
            mb_logprobs = get_shifted_logprobs(
                output.logits,
                mb_query_response,
                mb_response_mask,
                temperature=self.temperature,
            )

            # Compute the PPO-clip loss
            action_mask = mb_response_mask[:, 1:]
            log_ratio = (mb_logprobs - mb_old_logprobs) * action_mask
            ratio = torch.exp(log_ratio)

            mb_advantage = mb_advantage[:, 1:]
            pg_losses1 = -mb_advantage * ratio
            pg_losses2 = -mb_advantage * torch.clamp(ratio, 0.8, 1.2)
            pg_losses = torch.max(pg_losses1, pg_losses2)
            pg_loss = masked_mean(pg_losses, action_mask)

            per_token_kl = (
                torch.exp(mb_ref_logprobs - mb_logprobs)
                - (mb_ref_logprobs - mb_logprobs)
                - 1
            )
            per_token_kl = torch.clamp(
                per_token_kl * action_mask,
                min=0,
                max=10,
            )
            kl_loss = per_token_kl.sum(dim=1).mean()
            pg_loss = pg_loss + self.kl_ctl * kl_loss

            self.stats.accumulate(
                "kl_loss", kl_loss.item()
            )
            return pg_loss
