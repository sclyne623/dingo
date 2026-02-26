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
    parser.add_argument(
        "--backend_native_fused",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override local.backend_native_fused for this run.",
    )
    parser.add_argument(
        "--ab_backend_native_fused",
        action="store_true",
        help="Run A/B benchmark with backend_native_fused=False and True.",
    )
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


def _run_once(args, train_settings, local_settings, fused_flag, gen_device, train_device):
    run_local_settings = deepcopy(local_settings)
    run_local_settings["backend_native_fused"] = bool(fused_flag)

    # Make GPU0 default for generation path.
    torch.cuda.set_device(gen_device)
    pm, wfd = prepare_training_new(
        deepcopy(train_settings), args.train_dir, run_local_settings
    )

    # Move training network to GPU1 for compute.
    pm.network.to(train_device)
    pm.device = train_device

    stage0 = train_settings["training"]["stage_0"]
    train_loader, _ = initialize_stage(
        pm, wfd, stage0, run_local_settings["num_workers"], resume=False
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
        return None

    avg_data = timings["data"] / counted
    avg_transfer = timings["transfer"] / counted
    avg_network = timings["network"] / counted
    total = avg_data + avg_transfer + avg_network
    batch_size = stage0["batch_size"]
    samples_per_sec = batch_size / total if total > 0 else float("nan")

    summary = {
        "backend_native_fused": bool(fused_flag),
        "generation_gpu": str(gen_device),
        "training_gpu": str(train_device),
        "batch_size": int(batch_size),
        "avg_data": float(avg_data),
        "avg_transfer": float(avg_transfer),
        "avg_network": float(avg_network),
        "avg_total": float(total),
        "samples_per_s": float(samples_per_sec),
    }
    print("\n=== Two-GPU Prototype Summary ===")
    print(f"backend_native_fused: {summary['backend_native_fused']}")
    print(f"generation gpu      : {summary['generation_gpu']}")
    print(f"training gpu        : {summary['training_gpu']}")
    print(f"batch_size          : {summary['batch_size']}")
    print(f"avg_data            : {summary['avg_data']:.3f}s")
    print(f"avg_transfer        : {summary['avg_transfer']:.3f}s")
    print(f"avg_network         : {summary['avg_network']:.3f}s")
    print(f"avg_total           : {summary['avg_total']:.3f}s")
    print(f"samples_per_s       : {summary['samples_per_s']:.1f}")
    return summary


def main():
    args = _parse_args()

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Need at least 2 CUDA devices for this prototype; found {torch.cuda.device_count()}."
        )

    gen_device = torch.device("cuda:0")
    train_device = torch.device("cuda:1")

    with open(args.settings_file, "r") as fp:
        settings_blob = yaml.safe_load(fp)

    local_settings = deepcopy(settings_blob.pop("local"))
    train_settings = deepcopy(settings_blob)

    if args.ab_backend_native_fused:
        results = []
        for fused_flag in (False, True):
            print(f"\n--- A/B run: backend_native_fused={fused_flag} ---")
            result = _run_once(
                args,
                train_settings,
                local_settings,
                fused_flag,
                gen_device,
                train_device,
            )
            if result is not None:
                results.append(result)
            torch.cuda.empty_cache()

        if len(results) == 2:
            off = results[0]
            on = results[1]
            speedup = (
                on["samples_per_s"] / off["samples_per_s"]
                if off["samples_per_s"] > 0
                else float("nan")
            )
            print("\n=== A/B Comparison ===")
            print(
                f"fused=False samples_per_s: {off['samples_per_s']:.1f} "
                f"(avg_total={off['avg_total']:.3f}s)"
            )
            print(
                f"fused=True  samples_per_s: {on['samples_per_s']:.1f} "
                f"(avg_total={on['avg_total']:.3f}s)"
            )
            print(f"speedup (on/off): {speedup:.3f}x")
        return

    fused_flag = local_settings.get("backend_native_fused", False)
    if args.backend_native_fused is not None:
        fused_flag = args.backend_native_fused.lower() == "true"
    _run_once(args, train_settings, local_settings, fused_flag, gen_device, train_device)


if __name__ == "__main__":
    main()
