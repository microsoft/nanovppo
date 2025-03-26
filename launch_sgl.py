import concurrent
import logging
import os
import subprocess
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
from utils import DEFAULT_MAX_TOKENS, DEFAULT_TEMP


def is_server_up(timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    try:
        response = requests.get(
            f"http://localhost:30000/v1/models",
            headers={"Authorization": "Bearer None"},
        )
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
    return False


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


def execute_shell_command(command: str, env=None) -> subprocess.Popen:
    """
    Execute a shell command and return the process handle

    Args:
        command: Shell command as a string (can include \\ line continuations)
    Returns:
        subprocess.Popen: Process handle
    """
    # Replace \ newline with space and split
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()

    return subprocess.Popen(parts, text=True, env=env, stderr=subprocess.STDOUT)


class SGLGenerator:
    _base_gpu_id = os.environ.get('SGL_BASE_GPU_ID', 1)
    _dp_size = os.environ.get('SGL_DP_SIZE', 1)
    _instance = None

    @classmethod
    def get(cls):
        assert cls._instance is not None, "SGLGenerator not initialized"
        return cls._instance

    def __init__(self, model_name, seed):
        self.model_name = model_name
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.start()
        SGLGenerator._instance = self

    def shutdown(self):
        from sglang.utils import terminate_process
        if self.process:
            terminate_process(self.process)
            wait_for_server_shutdown("http://localhost:30000")

    @ddp_state.on_main_process
    def start(self):
        from sglang.utils import wait_for_server
        import os

        # Create a clean environment for the subprocess
        env = os.environ.copy()

        # List of variables to unset (expanded to cover all torchrun/TorchElastic vars)
        # otherwise SGLang server doesn't start
        torch_vars = [
            "LOCAL_RANK", "RANK", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME",
            "LOCAL_WORLD_SIZE", "WORLD_SIZE", "GROUP_WORLD_SIZE", "ROLE_WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT",
            "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_ERROR_FILE", "TORCH_NCCL_ASYNC_ERROR_HANDLING"
        ]
        for var in torch_vars:
            if var in env:
                del env[var]

        server_process = execute_shell_command(
            f"""python3 -m sglang.launch_server \
--model-path {self.model_name} \
--host 0.0.0.0 --port 30000 \
--log-level warning \
--mem-fraction-static 0.6 \
--random-seed {self.seed} \
--base-gpu-id {self._base_gpu_id} \
--dp-size {self._dp_size}
""",
            env=env,
        )

        wait_for_server("http://localhost:30000")
        self.process = server_process


if __name__ == '__main__':
    SGLGenerator(model_name="meta-llama/Llama-3.1-8B-Instruct", seed=42)
    while True:
        time.sleep(5)
