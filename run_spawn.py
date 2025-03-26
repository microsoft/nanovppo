import argparse
from contextlib import contextmanager
import gc
import math
from multiprocessing import Process
import os
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from accelerate import Accelerator
from algos.vppo import VPPO
from math_utils import (
    compute_lcot_math_reward,
    compute_math_reward,
    eval_math,
    prepare_math_dataset,
)
from algos.rft import RFT
from algos.grpo import GRPO
from data_utils import (
    MultiTensorDataset,
    chunk_text,
    get_dataloader,
)
from launch_sgl import SGLGenerator, is_server_up
from transformers.trainer_pt_utils import get_parameter_names
from ddp_utils import init_ddp, gather_and_concatenate
from gen_utils import GenerationBackend
from utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    AccumulatorDict,
    CosineWarmupScheduler,
    print_metrics,
    save_args,
    setup_output_directory,
)
import bitsandbytes as bnb
from ddp_utils import ddp_state
from sglang.utils import wait_for_server
from run_torch import train
import torch.multiprocessing as mp


models = {
    "q1.5": "Qwen/Qwen2.5-1.5B",
    "q1.5i": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "ll8b": "meta-llama/Llama-3.1-8B-Instruct",
}


algos = {
    "rft": RFT,
    "grpo": GRPO,
    "vppo": VPPO,
}


def get_algo_kwargs(args, klass):
    parser = argparse.ArgumentParser()
    parser = klass.add_parser_args(parser)
    argnames = [action.dest for action in parser._actions]
    return {k: v for k, v in vars(args).items() if k in argnames}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Model name")
    parser.add_argument("-o", type=str, help="Output directory")
    parser.add_argument("-s", type=int, help="Seed", default=42)
    parser.add_argument("-a", type=str, help="Algorithm")
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-5)
    parser.add_argument("--lcot", action="store_true", help="Use long CoT")
    parser.add_argument("--epc", type=int, help="Number of epochs", default=10)
    parser.add_argument(
        "--onlbsz", type=int, help="Online batch size (per gpu)", default=32
    )
    parser.add_argument(
        "--offepc", type=int, help="Number of offline epochs", default=4
    )
    parser.add_argument(
        "--offbsz", type=int, help="Offline batch size (per gpu)", default=8
    )
    parser.add_argument(
        "--innbsz",
        type=int,
        help="Inner/Grad accumulation batch size (per gpu)",
        default=1,
    )
    parser.add_argument(
        "--opt",
        type=str,
        help="Optimizer to use for training",
        default="adamw",
        choices=["adamw", "adamw8bit"],
    )
    parser.add_argument(
        "--eval-every-n-epochs", type=int, help="Eval every N epochs", default=1
    )
    parser.add_argument("-k", type=int, help="Number of samples", default=5)
    parser.add_argument("-t", type=float, help="Temperature", default=DEFAULT_TEMP)
    parser.add_argument(
        "--maxtok", type=int, help="Number of tokens", default=DEFAULT_MAX_TOKENS
    )
    parser.add_argument("-b", type=str, help="Backend", default="sgl")
    parser.add_argument("--tpsz", type=int, help="Tensor parallel size", default=1)
    parser.add_argument("--ss", type=str, help="Math subset to consider", default="all")
    parser.add_argument("--subs", type=int, help="Subsample examples", default=-1)
    parser.add_argument(
        "--fast", action="store_true", help="Use fast mode (no eval on epoch 0)"
    )
    parser.add_argument(
        "--t500", action="store_true", help="Use 500 examples for MATH test."
    )
    parser.add_argument("-P", type=int, help="Number of processes", default=1)
    parser.add_argument("-d", type=str, help="Run description", default=None)

    # parse known args first
    partial_args, unknown_args = parser.parse_known_args()

    # conditionally add algorithm-specific args
    algos[partial_args.a].add_parser_args(parser)

    # Parse final args
    final_args = parser.parse_args()

    if is_server_up():
        raise ValueError("Terminate previous server before starting!")

    os.environ["OMP_NUM_THREADS"] = "1"
    mp.spawn(train, args=(final_args.P, final_args,), nprocs=final_args.P, join=True)
