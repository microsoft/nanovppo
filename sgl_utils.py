import concurrent
import logging
import os
import sys
import threading
import time
from asyncio import threads
from functools import partial
from typing import List, Optional

import docker
import psutil
import requests
import torch
import tqdm
from docker.errors import APIError, NotFound
from transformers import AutoTokenizer
from vllm.worker.worker import Worker

from ddp_utils import ddp_state
from utils import DEFAULT_MAX_TOKENS, DEFAULT_TEMP, pack, repeat


def wait_for_server_shutdown(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(1)

            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Server did not die within timeout period")
        except requests.exceptions.RequestException:
            break


class SGLGeneratorClient:
    _instance = None

    def __init__(self, model_name, **kwargs):
        import os
        from sglang.utils import execute_shell_command, wait_for_server

        self.port = 30000
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Waiting for SGLang server...")
        wait_for_server(f"http://localhost:{self.port}")
        print("Server found!")

        SGLGeneratorClient._instance = self

    @ddp_state.on_main_process
    def save_model(self, model):
        if hasattr(model, "module"):
            model = model.module

        print("Serializing model...")
        model.save_pretrained(f"/tmp/saved_model")

    @classmethod
    def get(cls):
        assert cls._instance is not None, "SGLGeneratorClient not initialized"
        return cls._instance

    @ddp_state.on_main_process
    def load_weights(self, model):
        import requests

        if hasattr(model, "module"):
            model = model.module

        model.save_pretrained("/tmp/saved_model")
        response = requests.post(
            f"http://localhost:{self.port}/update_weights_from_disk",
            json={"model_path": "/tmp/saved_model"},
        )
        assert response.json()["success"] is True

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
        prompts = repeat(prompts, n)

        ddp_state.print("SGLang prompt:")
        ddp_state.print(prompts[0])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm.tqdm(
                    executor.map(
                        partial(
                            send_request, temperature, top_p, max_tokens, 1, self.port
                        ),
                        prompts,
                    ),
                    total=len(prompts),
                )
            )

        outputs, finished = zip(*results)
        outputs = pack([o[0] for o in outputs], n)
        finished = pack([f[0] for f in finished], n)
        results = zip(outputs, finished)

        outputs = []
        finished = []
        for outputs_, finished_ in results:
            outputs.append(outputs_)
            finished.append(finished_)

        if return_finished:
            return outputs, finished
        return outputs


def send_request(temperature, top_p, max_tokens, n, port, prompt):
    import requests

    response = requests.post(
        f"http://localhost:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "n": n,
                "top_p": top_p,
            },
        },
    )
    response = response.json()
    if type(response) != list:
        response = [response]

    outputs = [r["text"] for r in response]
    finished = [r["meta_info"]["finish_reason"]["type"] == "stop" for r in response]
    return outputs, finished
