"""
Prototype two-GPU training loop for LISA/BBHx.

This script is intended for throughput benchmarking before integrating a full
multi-GPU pipeline into dingo_train.

Design:
- GPU0: batch generation / transform pipeline (DataLoader)
- GPU1: model forward/backward/optimizer

Usage:
    python -m dingo.gw.training.two_gpu_pipeline_prototype \
      --settings_file /path/to/more_params.yaml \
      --train_dir /path/to/train_dir \
      --steps 500
"""

import argparse
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml

from dingo.gw.training.train_pipeline import prepare_training_new, initialize_stage


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_file", required=True, type=str)
    parser.add_argument("--train_dir", required=True, type=str)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--warmup_steps", type=int, default=20)
    return parser.parse_args()


def _move_batch_to_device(batch, device):
    moved = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            if item.device == device:
                moved.append(item)
            else:
                moved.append(item.to(device, non_blocking=True))
        else:
            moved.append(torch.as_tensor(item, device=device))
    return moved


def _iter_with_background_prefetch(loader):
    iterator = iter(loader)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(next, iterator)
        while True:
            try:
                batch = future.result()
            except StopIteration:
                break
            future = executor.submit(next, iterator)
            yield batch


def main():
    args = _parse_args()

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Need at least 2 CUDA devices for this prototype; found {torch.cuda.device_count()}."
        )

    gen_device = torch.device("cuda:0")
    train_device = torch.device("cuda:1")

    # Make GPU0 default for generation path.
    torch.cuda.set_device(gen_device)

    with open(args.settings_file, "r") as fp:
        train_settings = yaml.safe_load(fp)

    local_settings = deepcopy(train_settings.pop("local"))
    # Build model/dataset with standard path first.
    pm, wfd = prepare_training_new(train_settings, args.train_dir, local_settings)

    # Move training network to GPU1 for compute.
    pm.network.to(train_device)
    pm.device = train_device

    stage0 = train_settings["training"]["stage_0"]
    train_loader, _ = initialize_stage(
        pm, wfd, stage0, local_settings["num_workers"], resume=False
    )

    pm.network.train()
    data_iter = _iter_with_background_prefetch(train_loader)

    timings = {
        "data": 0.0,
        "transfer": 0.0,
        "network": 0.0,
    }
    counted = 0

    for step_idx in range(args.steps):
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        t1 = time.perf_counter()

        batch = _move_batch_to_device(batch, train_device)
        t2 = time.perf_counter()

        pm.optimizer.zero_grad(set_to_none=True)
        loss = pm.loss(batch[0], *batch[1:])
        loss.backward()
        pm.optimizer.step()
        t3 = time.perf_counter()

        if step_idx >= args.warmup_steps:
            timings["data"] += t1 - t0
            timings["transfer"] += t2 - t1
            timings["network"] += t3 - t2
            counted += 1

        if (step_idx + 1) % args.print_every == 0:
            print(
                f"step={step_idx+1:5d} "
                f"loss={loss.item():.4f} "
                f"data={t1-t0:.3f}s "
                f"transfer={t2-t1:.3f}s "
                f"network={t3-t2:.3f}s"
            )

    if counted == 0:
        print("No post-warmup steps were collected.")
        return

    avg_data = timings["data"] / counted
    avg_transfer = timings["transfer"] / counted
    avg_network = timings["network"] / counted
    total = avg_data + avg_transfer + avg_network
    batch_size = stage0["batch_size"]
    samples_per_sec = batch_size / total if total > 0 else float("nan")

    print("\n=== Two-GPU Prototype Summary ===")
    print(f"generation gpu : {gen_device}")
    print(f"training gpu   : {train_device}")
    print(f"batch_size     : {batch_size}")
    print(f"avg_data       : {avg_data:.3f}s")
    print(f"avg_transfer   : {avg_transfer:.3f}s")
    print(f"avg_network    : {avg_network:.3f}s")
    print(f"avg_total      : {total:.3f}s")
    print(f"samples_per_s  : {samples_per_sec:.1f}")


if __name__ == "__main__":
    main()

