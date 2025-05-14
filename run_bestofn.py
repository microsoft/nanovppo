import argparse
from contextlib import contextmanager
import os
import random
import numpy as np
import torch

from tqdm import tqdm

from algos.algo import Request
from algos.tpo import TPO
from algos.vppo import VPPO
from dataset.arc_utils import compute_arc_reward, eval_arc, prepare_arc_dataset
from dataset.cd_utils import compute_cd_reward, correctness_reward, eval_cd, prepare_cd_dataset
from dataset.gsm8k_utils import compute_gsm8k_reward, eval_gsm8k, prepare_gsm8k_dataset
from dataset.math_utils import (
    compute_lcot_math_reward,
    compute_math_reward,
    compute_qst_math_reward,
    eval_math,
    prepare_math_dataset,
)
from algos.rft import RFT
from algos.grm import GenRM
from algos.grpo import GRPO
from algos.grpod import GRPOD
from dataset.data_utils import (
    MultiTensorDataset,
    chunk_text,
    get_dataloader,
)
from sgl_utils import SGLGenerator, is_server_up
from ddp_utils import init_ddp, gather_and_concatenate
from gen_utils import GeneratorClient, GeneratorClient
from utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    AccumulatorDict,
    CosineWarmupScheduler,
    copy_model_param,
    param_ema,
    print_metrics,
    save_args,
    setup_output_directory,
)
from ddp_utils import ddp_state


def get_algo_kwargs(args, klass):
    parser = argparse.ArgumentParser()
    parser = klass.add_parser_args(parser)
    argnames = [action.dest for action in parser._actions]
    return {k: v for k, v in vars(args).items() if k in argnames}


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def run_inference(local_rank: int, world_size: int, args):
    """
    Generate ``args.maxN`` continuations per evaluation prompt and compute pass@N.
    Nothing is updated – we merely evaluate a previously trained / fine–tuned model.
    """
    print("=============================")
    print(f"[Inference] Bringing up {args.b} server")

    GeneratorClient.init(
        backend=args.b,
        model_name=args.load_model + "/model",
        seed=args.s,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )

    torch.manual_seed(args.s)
    random.seed(args.s)
    np.random.seed(args.s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.s)

    if args.dataset == "cd":
        prepare_dataset = prepare_cd_dataset
        reward_fn = correctness_reward
    else:
        raise ValueError("Unknown dataset")

    queries, labels = prepare_dataset(
        args.ss, split="test", subsample=args.subs, template=args.template
    )

    generator = GeneratorClient.get()

    responses = generator.chat(
        queries,
        temperature=args.t if hasattr(args, "t") else DEFAULT_TEMP,
        max_tokens=args.maxtok if hasattr(args, "maxtok") else DEFAULT_MAX_TOKENS,
        top_p=args.top_p if hasattr(args, "top_p") else 0.9,
        n=args.maxN,
    )
    corrects = [
        sum([reward_fn(response, gold[0], gold[1]) for response in response_array])
        for response_array, gold in zip(responses, labels)
    ]

    pass_at_k = [
        np.mean([estimator(args.maxN, c, k) for c in corrects])
        for k in range(1, args.maxN + 1)
    ]

    print(f"pass@k: {pass_at_k}")

    # store JSON result beside the model
    os.makedirs(args.o, exist_ok=True)
    with open(os.path.join(args.o, "inference_stats.json"), "w") as f:
        import json
        json.dump(
            {
                "ticks": range(1, args.maxN + 1),
                "pass_at_k": pass_at_k,
            },
            f,
        )

    GeneratorClient.get().shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, help="Model directory")
    parser.add_argument("-o", type=str, help="Output directory")
    parser.add_argument("-s", type=int, help="Seed", default=42)
    parser.add_argument("-maxN", type=int, help="Max pass@N", default=32)
    args = parser.parse_args()

    args_path = os.path.join(args.load_model, "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"Could not find {args_path}")

    with open(args_path, "r") as f:
        import json

        saved_args = json.load(f)

    # Re-create original Namespace then override with CLI flags
    final_args = argparse.Namespace(**saved_args)

    # CLI overrides (only if the user supplied them)
    if args.o is not None:
        final_args.o = args.o
    else:
        # default inference output folder next to the model
        final_args.o = os.path.join(args.load_model, "inference_output")

    final_args.load_model = args.load_model  # keep track of path
    final_args.s = args.s  # new/overridden seed
    final_args.maxN = args.maxN  # inference-time pass@N

    os.environ["OMP_NUM_THREADS"] = "1"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and final_args.b == "vllm":
        raise ValueError("VLLM backend does not support distributed training.")

    run_inference(local_rank, world_size, final_args)
