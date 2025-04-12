import concurrent
import logging
import os
import sys
import threading
import time
from asyncio import threads
import subprocess
from functools import partial
from typing import List, Optional

import docker
import psutil
import requests
import torch
import tqdm

from ddp_utils import ddp_state
from docker.errors import APIError, NotFound
from transformers import AutoTokenizer
from vllm.worker.worker import Worker

from vllm import LLM, SamplingParams
from utils import DEFAULT_MAX_TOKENS, DEFAULT_TEMP, pack, repeat


class VLLMGeneratorClient:
    _instance = None

    def __init__(self, model_name, seed):
        self.model_name = model_name
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.start()
        VLLMGeneratorClient._instance = self

    def shutdown(self):
        pass

    @ddp_state.on_main_process
    def start(self):
        self.llm = LLM(
            model=self.model_name,
            skip_tokenizer_init=False,
            gpu_memory_utilization=0.2,
            tensor_parallel_size=1,
            enable_prefix_caching=True,
            swap_space=1,
            scheduling_policy="fcfs",
            dtype=torch.bfloat16,
            enable_sleep_mode=True,
        )

    def sleep(self):
        self.llm.sleep(1)

    def wake_up(self):
        self.llm.wake_up()

    def shutdown(self):
        pass

    @classmethod
    def get(cls):
        assert cls._instance is not None, "VLLMGeneratorClient not initialized"
        return cls._instance

    @ddp_state.on_main_process
    def load_weights(self, model):
        self.wake_up()

        state_dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
            state_dict.items()
        )

    def chat(
        self,
        messages,
        temperature=DEFAULT_TEMP,
        top_p=1.0,
        max_tokens=DEFAULT_MAX_TOKENS,
        n=1,
        return_finished=False,
    ):
        import requests

        prompts = [
            self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=message[-1]["role"] == "user",
                continue_final_message=message[-1]["role"] == "assistant",
                tokenize=False,
            )
            for message in messages
        ]

        ddp_state.print("VLLM prompt:")
        ddp_state.print(prompts[0])

        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
        )

        outputs_ = []
        finished_ = []

        for output_completions in outputs:
            outputs_.append([g.text for g in output_completions.outputs])
            finished_.append(
                [g.finish_reason == "stop" for g in output_completions.outputs]
            )

        if return_finished:
            return outputs_, finished_
        return outputs_
