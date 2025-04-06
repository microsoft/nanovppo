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
from dataset.arc_utils import compute_arc_reward, eval_arc, prepare_arc_dataset
from dataset.cd_utils import compute_cd_reward, eval_cd, prepare_cd_dataset
from dataset.gsm8k_utils import compute_gsm8k_reward, eval_gsm8k, prepare_gsm8k_dataset
from dataset.math_utils import (
    compute_lcot_math_reward,
    compute_math_reward,
    compute_qst_math_reward,
    eval_math,
    prepare_math_dataset,
)
from algos.rft import RFT
from algos.grpo import GRPO
from dataset.data_utils import (
    MultiTensorDataset,
    chunk_text,
    get_dataloader,
)
from transformers.trainer_pt_utils import get_parameter_names
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
import bitsandbytes as bnb
from ddp_utils import ddp_state
from sglang.utils import wait_for_server
import torch.multiprocessing as mp


models = {
    "q1.5": "Qwen/Qwen2.5-1.5B",
    "q3": "Qwen/Qwen2.5-3B",
    "q3i": "Qwen/Qwen2.5-3B-Instruct",
    "q1.5m": "Qwen/Qwen2.5-Math-1.5B",
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


def train(local_rank, world_size, args):
    algo_kwargs = get_algo_kwargs(args, algos[args.a])
    print("Args kwargs:", args)
    print("Algo kwargs:", algo_kwargs)

    init_ddp(local_rank, world_size)

    if local_rank == 0:
        print("=============================")
        print("Bringing up SGL server")
        generator = SGLGenerator(models[args.m], args.s)

    # wait for generator to be up!
    GeneratorClient.init(args.b, model_name=models[args.m], seed=args.s)

    # output directory
    torch.manual_seed(args.s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.s)
    np.random.seed(args.s)
    random.seed(args.s)

    # dataset setup + rewards
    if args.dataset == "math":
        prepare_dataset = prepare_math_dataset
        eval_dataset = eval_math
        if args.template == "lcot":
            reward_func = compute_lcot_math_reward
        elif args.template == "cot":
            reward_func = compute_math_reward
        else:
            reward_func = compute_qst_math_reward
    elif args.dataset == "gsm8k":
        prepare_dataset = prepare_gsm8k_dataset
        eval_dataset = eval_gsm8k
        reward_func = compute_gsm8k_reward
    elif args.dataset == "arc":
        prepare_dataset = prepare_arc_dataset
        eval_dataset = eval_arc
        reward_func = compute_arc_reward
    elif args.dataset == "cd":
        prepare_dataset = prepare_cd_dataset
        eval_dataset = eval_cd
        reward_func = compute_cd_reward
    else:
        raise ValueError("Unknown dataset!")

    if args.o == "auto":
        # create a timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.o = f"runs_output/{args.a}_{args.m}_{args.s}_{timestamp}"

    max_epochs = args.maxepochs
    max_off_epochs = args.offepc
    # total batch size across devices
    onl_batch_size = args.onlbsz * ddp_state.num_processes
    # this is *per device*
    off_batch_size = args.offbsz
    inn_batch_size = args.onlbsz // args.offbsz

    assert (
        onl_batch_size % ddp_state.num_processes == 0
    ), "Batch size must be divisible by the number of GPUs!"

    if ddp_state.is_main_process:
        setup_output_directory(args.o)
        save_args(args, args.o)

    algo = algos[args.a](
        models[args.m],
        reward_func,
        k=args.k,
        temperature=args.t,
        max_tokens=args.maxtok,
        device=ddp_state.device,
        **algo_kwargs,
    )
    algo.model.gradient_checkpointing_enable()
    algo.model = DDP(
        algo.model,
        device_ids=[ddp_state.local_process_index],
        output_device=ddp_state.local_process_index,
    )

    decay_parameters = get_parameter_names(algo.model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in algo.model.named_parameters() if n in decay_parameters
            ],
            "weight_decay": 1e-6,
        },
        {
            "params": [
                p for n, p in algo.model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.opt == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif args.opt == "adamw8bit":
        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            lr=args.lr,
        )
    ddp_state.print(torch.cuda.mem_get_info())

    total_steps = (max_epochs * max_off_epochs)

    ddp_state.print("Batch size (total across GPUs):", onl_batch_size)
    ddp_state.print("Batch size (per GPU):", args.onlbsz)
    ddp_state.print("Gradient accumulation steps:", args.offbsz)
    ddp_state.print("Max epochs:", max_epochs)
    ddp_state.print("Max offline epochs:", max_off_epochs)
    ddp_state.print("Total steps:", total_steps)

    if args.sch == "cosine_with_warmup":
        scheduler = CosineWarmupScheduler(
            optimizer,
            max_lr=args.lr,
            min_lr=1e-3 * args.lr,
            warmup_steps=0.03 * total_steps,
            max_steps=total_steps,
        )
    elif args.sch == "constant_with_warmup":
        scheduler = CosineWarmupScheduler(
            optimizer,
            max_lr=args.lr,
            min_lr=args.lr,
            warmup_steps=0.03 * total_steps,
            max_steps=total_steps,
        )
    else:
        raise ValueError("Unknown scheduler.")

    if args.ema > 0.0:
        ema_params = copy_model_param(algo.model.module)

    ddp_state.print("Scheduler set! Total steps:", total_steps)

    with ddp_state.main_process_first():
        all_queries, all_labels = prepare_dataset(
            args.ss, split="train", subsample=args.subs, template=args.template
        )
        if args.t500:
            if args.dataset != "math":
                raise ValueError("t500 only available with MATH dataset!")

            test_queries, test_labels = prepare_dataset(
                args.ss,
                split="500",
                template=args.template,
            )
        else:
            test_queries, test_labels = prepare_dataset(
                args.ss, split="test", subsample=args.subs, template=args.template
            )

        # Form a small subset of the training data for evaluation
        train_subsample = np.random.choice(
            len(all_queries), args.subs if args.subs > -1 else 100, replace=False
        )
        train_queries = [all_queries[i] for i in train_subsample]
        train_labels = [all_labels[i] for i in train_subsample]

    training_stats = []

    assert (
        onl_batch_size >= args.subs
    ), "Batch size cannot be smaller than the dataset size!"

    global_step = 0
    eval_step = False
    best_acc = 0
    tracc_math = 0
    acc_math = 0

    for global_epoch in range(max_epochs):
        epoch_stats = AccumulatorDict()

        sample_indices = np.random.choice(
            len(all_queries), onl_batch_size, replace=False
        )
        queries_batch = [all_queries[i] for i in sample_indices]
        labels_batch = [all_labels[i] for i in sample_indices]

        if not args.fast and global_step == 0:
            acc_math = np.mean(
                eval_dataset(test_queries, test_labels, temperature=0.35, top_p=0.9)
            )
            tracc_math = np.mean(
                eval_dataset(train_queries, train_labels, temperature=0.35, top_p=0.9)
            )

            if ddp_state.is_main_process:
                print("====================================")
                print(f"Initial Accuracy: {tracc_math}, Test Accuracy: {acc_math}")
                print("====================================")

            training_stats.append(
                {
                    "epoch": global_epoch,
                    "step": global_step,
                    "train_accuracy": tracc_math,
                    "test_accuracy": acc_math,
                }
            )

        # create dataset out of gathered episodes
        with ddp_state.split_between_processes(
            list(zip(queries_batch, labels_batch))
        ) as partial_batch:
            part_queries, part_labels = zip(*partial_batch)
            episode_data = algo.gather_episodes(part_queries, part_labels)

        epoch_dataset = MultiTensorDataset(*episode_data)
        dataloader = get_dataloader(epoch_dataset, inn_batch_size)

        ddp_state.print("====================================")
        ddp_state.print("Beginning updating the policy")
        ddp_state.print("Length of the dataset:", len(epoch_dataset))

        torch.save(
            episode_data,
            f"{args.o}/episode_data_{ddp_state.local_process_index}_{global_epoch}.pt",
        )

        # offline steps
        train_iterator = iter(dataloader)
        acc_steps = len(dataloader)
        off_sampler = dataloader.sampler
        off_steps = max_off_epochs * len(dataloader)
        off_epoch = 0

        for off_epoch in range(max_off_epochs):
            algo.model.train()
            optimizer.zero_grad()
            train_iterator = iter(dataloader)
            off_sampler.set_epoch(off_epoch)

            for micro_step, batch in enumerate(
                tqdm(
                    train_iterator,
                    total=len(dataloader),
                    desc="Offline epoch {}".format(off_epoch),
                    disable=not ddp_state.is_main_process,
                )
            ):
                loss_batch = 0
                batch = [b.to(ddp_state.device) for b in batch]

                algo.model.require_backward_grad_sync = (
                    micro_step == len(dataloader) - 1
                )
                loss = algo.compute_loss(batch)

                (loss / acc_steps).backward()
                epoch_stats.accumulate("loss", loss.item())
                del batch
                del loss
                torch.cuda.empty_cache()

            # update the model
            torch.nn.utils.clip_grad_norm_(algo.model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            global_step += 1

            if args.ema:
                param_ema(ema_params, algo.model.module, args.ema)

            ddp_state.print(
                f"Iter: {iter}, Off Epoch: {off_epoch}, "
                f"Step: {global_step}, {algo.__class__.__name__}, "
                f"Loss: {epoch_stats.mean('loss'):.4f}, "
                f"Lr: {scheduler.get_last_lr()[0]:.6f}"
            )

        del dataloader
        del episode_data
        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        ddp_state.wait_for_everyone()

        eval_step = (global_epoch + 1) % args.evalevery == 0

        if eval_step:
            acc_math = np.mean(
                eval_dataset(test_queries, test_labels, temperature=0.35, top_p=0.9)
            )
            tracc_math = np.mean(
                eval_dataset(train_queries, train_labels, temperature=0.35, top_p=0.9)
            )
            eval_step = False
            epoch_stats.accumulate("test_accuracy", acc_math)
            epoch_stats.accumulate("train_accuracy", tracc_math)

        # append a bunch of training stats
        training_stats.append(
            {
                "epoch": global_epoch,
                "step": global_step,
                "lr": scheduler.get_last_lr()[0],
                **epoch_stats.get(),
                **algo.stats.get(),
            }
        )

        if ddp_state.is_main_process:
            print("====================================")
            print(
                f"Step {global_step}, Train Accuracy: {tracc_math}, Test Accuracy: {acc_math}, Max Test Acc: {best_acc}"
            )
            print(training_stats[-1])
            print(
                "Train Accuracy So Far:",
                print_metrics(
                    [
                        t["train_accuracy"]
                        for t in training_stats
                        if "train_accuracy" in t
                    ]
                ),
            )
            print(
                "Test Accuracy So Far:",
                print_metrics(
                    [t["test_accuracy"] for t in training_stats if "test_accuracy" in t]
                ),
            )
            print(
                "Reward So Far:",
                print_metrics([t.get("avg_reward", 0) for t in training_stats]),
            )
            print("====================================")

            # save stats
            with open(f"{args.o}/training_stats.json", "w") as f:
                import json

                json.dump(training_stats, f)

            # save best model
            if acc_math > best_acc:
                algo.model.module.save_pretrained(f"{args.o}/model")
                algo.tokenizer.save_pretrained(f"{args.o}/model")
                best_acc = acc_math
                print("Best model saved!")

            # update the weights of the data generator after the epoch
            GeneratorClient.get().load_weights(algo.model)

        ddp_state.wait_for_everyone()

    if ddp_state.local_process_index == 0:
        generator.shutdown()
    ddp_state.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Model name")
    parser.add_argument("-o", type=str, help="Output directory")
    parser.add_argument("-s", type=int, help="Seed", default=42)
    parser.add_argument("-a", type=str, help="Algorithm")
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-5)
    parser.add_argument("--template", default="cot", help="Template type COT/LCOT")
    parser.add_argument("--maxepochs", type=int, help="Max number of epochs", default=-1)
    parser.add_argument(
        "--onlbsz", type=int, help="Online batch size (per gpu)", default=32
    )
    parser.add_argument(
        "--offepc", type=int, help="Number of offline epochs", default=1
    )
    parser.add_argument(
        "--offbsz", type=int, help="Batch size (per gpu, gradient accumulation)", default=1
    )
    parser.add_argument(
        "--opt",
        type=str,
        help="Optimizer to use for training",
        default="adamw",
        choices=["adamw", "adamw8bit"],
    )
    parser.add_argument(
        "--ema",
        type=float,
        help="Apply ema to the model parameters",
        default=0.0,
    )
    parser.add_argument("--evalevery", type=int, help="Eval every N epochs", default=-1)
    parser.add_argument("-k", type=int, help="Number of samples", default=5)
    parser.add_argument("-t", type=float, help="Temperature", default=DEFAULT_TEMP)
    parser.add_argument(
        "--maxtok", type=int, help="Number of tokens", default=DEFAULT_MAX_TOKENS
    )
    parser.add_argument(
        "--sch", type=str, help="Scheduler", default="constant_with_warmup"
    )
    parser.add_argument("--dataset", type=str, help="Dataset", default="math")
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    train(local_rank, world_size, final_args)
