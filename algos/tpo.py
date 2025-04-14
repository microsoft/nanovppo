from collections import defaultdict
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


TEACHER_PROMPT = (
    "Generate reasonings that satisfy the following criterion:\n\n{criterion}"
)

VERIFIER_TEMPLATE = "Answer with YES or NO whether the following satisfies the criterion for a good reasoning.\n\nCriterion:\n{criterion}\n\nReasoning:{reasoning}\n\nDoes the reasoning satisfy the criterion? (YES/NO):"

EXTRACTOR_TEMPLATE_USER = """
Analyze the two reasonings. The good reasoning was preferred to the bad reasoning. 

CRITERIA GENERATION: Create a *specific* criterion such that the criterion is reasonable, and reflects the preference
between the two reasonings. The criterion should be a single sentence, and should not be too long.

The criterion should be what makes the good reasoning better than the bad reasoning.

Good reasoning:\n\n
{good_reasoning}

Bad reasoning:\n\n
{bad_reasoning}
"""

EXTRACTOR_TEMPLATE_ASS = "Criterion:"


class TPO(Algo):
    @classmethod
    def add_parser_args(self, parser):
        parser.add_argument(
            "--kl_ctl", type=float, default=0.0001, help="KL term control"
        )
        parser.add_argument(
            "--tea_ctl",
            type=float,
            default=0.1,
            help="Teacher distillation term control",
        )
        parser.add_argument(
            "--kl_max", type=int, default=10, help="Clip KL divergence to this value"
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

        self.criterion = None

        self.k = k
        self.tea_ctl = algo_kwargs["tea_ctl"]
        self.kl_ctl = algo_kwargs["kl_ctl"]
        self.kl_max = algo_kwargs["kl_max"]
        self.drgpo = algo_kwargs["drgrpo"]
        self.reward_func = reward_func
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stats = AccumulatorDict()

    def build_tensors_for_criterion_verifier(
        self,
        tokenizer,
        verifier_messages,
        verifier_answers,
        num_examples_in_batch,
    ):
        ver_query_response, ver_query_response_mask, ver_response_mask = (
            create_joint_tensors(
                self.tokenizer,
                verifier_messages,
                verifier_answers,
                is_final=[1] * len(verifier_messages),
            )
        )
        if num_examples_in_batch - len(ver_query_response) > 0:
            ver_query_response = torch.cat(
                [
                    ver_query_response,
                    torch.zeros(
                        num_examples_in_batch - len(ver_query_response),
                        ver_query_response.shape[1],
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            ver_query_response_mask = torch.cat(
                [
                    ver_query_response_mask,
                    torch.zeros(
                        num_examples_in_batch - len(ver_query_response_mask),
                        ver_query_response_mask.shape[1],
                        dtype=ver_query_response_mask.dtype,
                    ),
                ],
                dim=0,
            )
            ver_response_mask = torch.cat(
                [
                    ver_response_mask,
                    torch.zeros(
                        num_examples_in_batch - len(ver_response_mask),
                        ver_response_mask.shape[1],
                        dtype=ver_response_mask.dtype,
                    ),
                ],
                dim=0,
            )
        else:
            # cut the tensors to the maximum length
            ver_query_response = ver_query_response[:num_examples_in_batch]
            ver_query_response_mask = ver_query_response_mask[:num_examples_in_batch]
            ver_response_mask = ver_response_mask[:num_examples_in_batch]

        return (
            ver_query_response,
            ver_query_response_mask,
            ver_response_mask,
        )

    def build_tensors_for_criterion_extractor(
        self,
        tokenizer,
        ext_messages,
        ext_criteria,
        criteria_scores,
        num_examples_in_batch,
    ):
        criterion_advantages = torch.tensor(
            (criteria_scores - criteria_scores.mean()).reshape(-1).tolist(),
            dtype=torch.float32,
        )
        ext_query_response, ext_query_response_mask, ext_response_mask = (
            create_joint_tensors(
                self.tokenizer,
                ext_messages,
                ext_criteria,
                is_final=[1] * len(ext_messages),
            )
        )
        if num_examples_in_batch - len(ext_query_response) > 0:
            # pad to query_response length
            criterion_advantages = torch.cat(
                [
                    criterion_advantages,
                    torch.zeros(num_examples_in_batch - len(criterion_advantages)),
                ],
            )
            ext_query_response = torch.cat(
                [
                    ext_query_response,
                    torch.zeros(
                        num_examples_in_batch - len(ext_query_response),
                        ext_query_response.shape[1],
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            ext_query_response_mask = torch.cat(
                [
                    ext_query_response_mask,
                    torch.zeros(
                        num_examples_in_batch - len(ext_query_response_mask),
                        ext_query_response_mask.shape[1],
                        dtype=ext_query_response_mask.dtype,
                    ),
                ],
                dim=0,
            )
            ext_response_mask = torch.cat(
                [
                    ext_response_mask,
                    torch.zeros(
                        num_examples_in_batch - len(ext_response_mask),
                        ext_response_mask.shape[1],
                        dtype=ext_response_mask.dtype,
                    ),
                ],
                dim=0,
            )
        else:
            # cut the tensors to the maximum length
            ext_query_response = ext_query_response[:num_examples_in_batch]
            ext_query_response_mask = ext_query_response_mask[:num_examples_in_batch]
            ext_response_mask = ext_response_mask[:num_examples_in_batch]
            criterion_advantages = criterion_advantages[:num_examples_in_batch]

        return (
            criterion_advantages,
            ext_query_response,
            ext_query_response_mask,
            ext_response_mask,
        )

    def create_best_worst_pairs(
        self,
        responses: List[str],
        rewards: List[float],
        k: int,
        n: int,
    ) -> List[tuple]:
        best_worst_pairs = []
        rewards_reshaped = np.array(rewards).reshape(-1, k)
        for i, example_rewards in enumerate(rewards_reshaped):
            # Create all pairs of (high, low) reward responses
            for k in range(self.k):
                for j in range(self.k):
                    if example_rewards[k] > example_rewards[j]:
                        absolute_best_idx = i * self.k + k
                        absolute_worst_idx = i * self.k + j
                        best_response = responses[absolute_best_idx]
                        worst_response = responses[absolute_worst_idx]
                        best_response_reward = example_rewards[k]
                        worst_response_reward = example_rewards[j]
                        best_worst_pairs.append(
                            (
                                absolute_best_idx,
                                absolute_worst_idx,
                                best_response_reward,
                                worst_response_reward,
                                best_response,
                                worst_response,
                            )
                        )

        best_worst_pairs = list(set(best_worst_pairs))
        np.random.shuffle(best_worst_pairs)
        best_worst_pairs = best_worst_pairs[:n]
        return list(set(best_worst_pairs))

    @torch.no_grad
    def gather_episodes(self, messages, labels):
        engine = GeneratorClient.get()

        messages = copy.deepcopy(messages)
        if self.criterion is not None:
            # add the criterion to the system messages
            for message in messages:
                message[0]["content"] = (
                    message[0]["content"]
                    + "\n\n"
                    + TEACHER_PROMPT.format(criterion=self.criterion)
                )

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

        max_reward, avg_reward = RequestUtils.gather_max_avg_reward(evaluation_requests)
        self.stats.accumulate("avg_reward", avg_reward)

        ddp_state.print("====================================")
        ddp_state.print("Problem: ", evaluation_requests[0].messages[-1]["content"])
        ddp_state.print("Answer: ", evaluation_requests[0].response)
        ddp_state.print("Reward: ", evaluation_requests[0].reward)

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
            messages,
            responses,
            is_final=finished,
        )
        self.stats.accumulate(
            "avg_resp_length", response_mask.sum(1).float().mean().item()
        )

        best_worst_pairs = self.create_best_worst_pairs(
            responses=responses,
            rewards=rewards,
            k=self.k,
            n=16,
        )

        extactor_messages = []
        for i, (bidx, widx, brw, wrw, br, wr) in enumerate(best_worst_pairs):
            user_message = EXTRACTOR_TEMPLATE_USER.format(
                good_reasoning=br,
                bad_reasoning=wr,
            )
            extactor_messages.append(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": EXTRACTOR_TEMPLATE_ASS},
                ]
            )

        criteria, _ = engine.chat(
            extactor_messages,
            temperature=self.temperature,
            n=1,
            max_tokens=self.max_tokens,
            return_finished=True,
        )
        criteria = flatten(criteria)

        # we need to verify the criteria now
        verifier_exec_messages = []
        verifier_sft_messages = []
        verifier_sft_answers = []

        for i, response in enumerate(responses):
            for j in range(len(criteria)):
                user_message = VERIFIER_TEMPLATE.format(
                    criterion=criteria[j],
                    reasoning=response,
                )
                verifier_exec_messages.append(
                    [{"role": "user", "content": user_message}]
                )

        for j in range(len(best_worst_pairs)):
            criterion = criteria[j]
            bidx, widx, br, wr, best_response, worst_response = best_worst_pairs[j]
            verifier_sft_messages.append(
                [
                    {
                        "role": "user",
                        "content": VERIFIER_TEMPLATE.format(
                            criterion=criteria[j], reasoning=best_response
                        ),
                    },
                ]
            )
            verifier_sft_messages.append(
                [
                    {
                        "role": "user",
                        "content": VERIFIER_TEMPLATE.format(
                            criterion=criteria[j], reasoning=worst_response
                        ),
                    },
                ]
            )
            verifier_sft_answers.append("YES")
            verifier_sft_answers.append("NO")

        scores, _ = engine.chat(
            verifier_exec_messages,
            temperature=0.35,
            n=1,
            max_tokens=4,
            return_finished=True,
        )
        scores = flatten(scores)
        scores = [1 if score.strip() in ["YES", "yes"] else 0 for score in scores]

        scores = np.array(scores).reshape(len(responses), len(criteria))
        scores_diff = np.asarray(
            [
                scores[bidx, :] - scores[widx, :]
                for bidx, widx, _, _, _, _ in best_worst_pairs
            ]
        )

        criteria_scores = scores_diff.mean(axis=0)
        best_criterion = np.argmax(criteria_scores)

        print("Best criterion:\n", criteria[best_criterion])
        print("Criterion score:\n", criteria_scores[best_criterion])

        self.criterion = criteria[best_criterion]

        rewards = np.asarray(rewards).reshape(-1, self.k)
        advantages = rewards - rewards.mean(axis=1, keepdims=True)
        if not self.drgpo:
            advantages = advantages / (rewards.std(axis=1, keepdims=True) + 1e-8)
        advantages = advantages.reshape(-1).tolist()
        advantages = torch.tensor(advantages, dtype=torch.float32)

        (
            extr_advantages,
            extr_query_response,
            extr_query_response_mask,
            extr_response_mask,
        ) = self.build_tensors_for_criterion_extractor(
            self.tokenizer,
            extactor_messages,
            criteria,
            criteria_scores,
            num_examples_in_batch=len(advantages),
        )

        ver_query_response, ver_query_response_mask, ver_response_mask = (
            self.build_tensors_for_criterion_verifier(
                self.tokenizer,
                verifier_sft_messages,
                verifier_sft_answers,
                num_examples_in_batch=len(advantages),
            )
        )

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
            extr_query_response,
            extr_query_response_mask,
            extr_response_mask,
            extr_advantages,
            ver_query_response,
            ver_query_response_mask,
            ver_response_mask,
        )

    def compute_loss(self, episode_returns) -> float:
        (
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
            mb_advantage,
            mb_ref_logprobs,
            mb_old_logprobs,
            mb_ext_query_response,
            mb_ext_query_response_mask,
            mb_ext_response_mask,
            mb_ext_advantage,
            mb_ver_query_response,
            mb_ver_query_response_mask,
            mb_ver_response_mask,
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

            pg_losses1 = -mb_advantage.unsqueeze(1) * ratio
            pg_losses2 = -mb_advantage.unsqueeze(1) * torch.clamp(ratio, 0.9, 1.1)
            pg_losses = torch.max(pg_losses1, pg_losses2)

            labels_mask = mb_response_mask[:, 1:]
            per_token_kl = (
                torch.exp(mb_ref_logprobs - mb_logprobs)
                - (mb_ref_logprobs - mb_logprobs)
                - 1
            )

            per_token_loss = pg_losses + self.kl_ctl * per_token_kl
            pg_loss = masked_mean(per_token_loss, labels_mask, axis=1).mean()

            max_ext_tokens = mb_ext_query_response_mask.sum(dim=1).max()
            if max_ext_tokens.item() > 0:
                mb_ext_query_response = mb_ext_query_response[:, :max_ext_tokens]
                mb_ext_query_response_mask = mb_ext_query_response_mask[
                    :, :max_ext_tokens
                ]
                mb_ext_response_mask = mb_ext_response_mask[:, :max_ext_tokens]

                output = self.model(
                    input_ids=mb_ext_query_response,
                    attention_mask=mb_ext_query_response_mask,
                    return_dict=True,
                )

                mb_v_logprobs = get_shifted_logprobs(
                    output.logits,
                    mb_ext_query_response,
                    mb_ext_response_mask,
                    temperature=self.temperature,
                )
                mb_v_label_mask = mb_ext_response_mask[:, 1:]

                # compute just the reinforce loss
                reinforce_loss = -mb_ext_advantage.unsqueeze(1) * mb_v_logprobs
                reinforce_loss = masked_mean(
                    reinforce_loss, mb_v_label_mask, axis=1
                ).mean()
                self.stats.accumulate("extractor_loss", reinforce_loss.item())
                pg_loss += reinforce_loss

            max_ver_tokens = mb_ver_query_response_mask.sum(dim=1).max()
            if max_ver_tokens.item() > 0:
                mb_ver_query_response = mb_ver_query_response[:, :max_ver_tokens]
                mb_ver_query_response_mask = mb_ver_query_response_mask[
                    :, :max_ver_tokens
                ]
                mb_ver_response_mask = mb_ver_response_mask[:, :max_ver_tokens]

                output = self.model(
                    input_ids=mb_ver_query_response,
                    attention_mask=mb_ver_query_response_mask,
                    return_dict=True,
                )

                mb_v_logprobs = get_shifted_logprobs(
                    output.logits,
                    mb_ver_query_response,
                    mb_ver_response_mask,
                    temperature=self.temperature,
                )
                mb_v_label_mask = mb_ver_response_mask[:, 1:]

                reinforce_loss = -mb_v_logprobs
                reinforce_loss = masked_mean(
                    reinforce_loss, mb_v_label_mask, axis=1
                ).mean()
                self.stats.accumulate("verifier_loss", reinforce_loss.item())
                pg_loss += reinforce_loss

            self.stats.accumulate(
                "entropy",
                -((mb_logprobs * labels_mask).sum() / labels_mask.sum()).item(),
            )
            self.stats.accumulate(
                "kl_loss", masked_mean(per_token_kl, labels_mask, axis=1).mean().item()
            )
            return pg_loss
