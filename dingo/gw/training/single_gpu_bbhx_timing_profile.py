"""
Single-GPU BBHx timing profiler for DINGO LISA training.

Profiles:
- Data pipeline time (DataLoader + transforms)
- Batch transfer time (to training GPU)
- Network time (forward + backward + optimizer)
- Optional internal timing breakdown from GenerateBBHxDirectResponse

Usage:
    python -m dingo.gw.training.single_gpu_bbhx_timing_profile \
      --settings_file /path/to/more_params.yaml \
      --train_dir /path/to/train_dir \
      --steps 500
"""

import argparse
import time
from copy import deepcopy

import torch
import yaml

from dingo.gw.training.train_pipeline import prepare_training_new, initialize_stage
from dingo.gw.transforms.detector_transforms import GenerateBBHxDirectResponse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_file", required=True, type=str)
    parser.add_argument("--train_dir", required=True, type=str)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument(
        "--backend_native_fused",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override backend_native_fused for this run.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override local.num_workers for this run.",
    )
    parser.add_argument(
        "--cuda_batch_prefetch",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override local.cuda_batch_prefetch for this run.",
    )
    parser.add_argument(
        "--transform_timing_print_every",
        type=int,
        default=0,
        help="If >0, print transform timing summary every N transform calls.",
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


def _find_base_dataset(ds):
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def _find_bbhx_transform(loader):
    base = _find_base_dataset(loader.dataset)
    transform = getattr(base, "transform", None)
    if not hasattr(transform, "transforms"):
        return None
    for t in transform.transforms:
        if isinstance(t, GenerateBBHxDirectResponse):
            return t
    return None


def main():
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    with open(args.settings_file, "r") as fp:
        settings_blob = yaml.safe_load(fp)

    local_settings = deepcopy(settings_blob.pop("local", {}))
    train_settings = deepcopy(settings_blob)

    # Force single-GPU mode for this profiler run.
    local_settings["device"] = "cuda"
    local_settings["num_gpus"] = 1
    local_settings["two_gpu_split"] = False
    local_settings.setdefault("num_workers", 0)
    local_settings.setdefault("cuda_batch_prefetch", False)

    if args.num_workers is not None:
        local_settings["num_workers"] = int(args.num_workers)
    if args.cuda_batch_prefetch is not None:
        local_settings["cuda_batch_prefetch"] = (
            args.cuda_batch_prefetch.lower() == "true"
        )
    if args.backend_native_fused is not None:
        local_settings["backend_native_fused"] = (
            args.backend_native_fused.lower() == "true"
        )

    pm, wfd = prepare_training_new(
        deepcopy(train_settings),
        args.train_dir,
        local_settings,
    )
    pm.network.to(device)
    pm.device = device

    # Enable transform-level timing instrumentation.
    waveform_generator_settings = wfd.settings.setdefault("waveform_generator", {})
    waveform_generator_settings["timing_profile"] = True
    waveform_generator_settings["timing_profile_print_every"] = int(
        args.transform_timing_print_every
    )
    if args.backend_native_fused is not None:
        fused_flag = args.backend_native_fused.lower() == "true"
        waveform_generator_settings["backend_native_fused"] = bool(fused_flag)
        if hasattr(wfd, "waveform_generator"):
            wfd.waveform_generator.backend_native_fused = bool(fused_flag)

    stage0 = train_settings["training"]["stage_0"]
    train_loader, _ = initialize_stage(
        pm,
        wfd,
        stage0,
        local_settings["num_workers"],
        resume=False,
    )

    bbhx_transform = _find_bbhx_transform(train_loader)
    if bbhx_transform is None:
        print("Warning: GenerateBBHxDirectResponse transform not found.")
    else:
        bbhx_transform.reset_timing_stats()

    pm.network.train()
    iterator = iter(train_loader)

    timings = {"data": 0.0, "transfer": 0.0, "network": 0.0}
    counted = 0

    for step_idx in range(args.steps):
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        t1 = time.perf_counter()

        batch = _move_batch_to_device(batch, device)
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
                f"step={step_idx + 1:5d} "
                f"loss={loss.item():.4f} "
                f"data={t1 - t0:.3f}s "
                f"transfer={t2 - t1:.3f}s "
                f"network={t3 - t2:.3f}s"
            )

    if counted == 0:
        print("No post-warmup steps collected.")
        return

    avg_data = timings["data"] / counted
    avg_transfer = timings["transfer"] / counted
    avg_network = timings["network"] / counted
    avg_total = avg_data + avg_transfer + avg_network
    batch_size = stage0["batch_size"]
    samples_per_sec = batch_size / avg_total if avg_total > 0 else float("nan")

    print("\n=== Single-GPU Timing Summary ===")
    print(f"steps_measured      : {counted}")
    print(f"batch_size          : {batch_size}")
    print(f"avg_data            : {avg_data:.4f}s")
    print(f"avg_transfer        : {avg_transfer:.4f}s")
    print(f"avg_network         : {avg_network:.4f}s")
    print(f"avg_total           : {avg_total:.4f}s")
    print(f"samples_per_s       : {samples_per_sec:.2f}")

    if bbhx_transform is not None:
        stats = bbhx_transform.get_timing_stats(reset=False)
        print("\n=== BBHx Transform Internal Timing (avg per transform call) ===")
        print(f"transform_calls     : {stats['count']}")
        print(f"copy_inputs         : {stats['avg_copy_inputs']:.6f}s")
        print(f"merge_params        : {stats['avg_merge_params']:.6f}s")
        print(f"waveform_generate   : {stats['avg_waveform_generate']:.6f}s")
        print(f"to_torch_or_norm    : {stats['avg_to_torch_or_normalize']:.6f}s")
        print(f"channel_pack        : {stats['avg_channel_pack']:.6f}s")
        print(f"param_bookkeeping   : {stats['avg_parameter_bookkeeping']:.6f}s")
        print(f"transform_total     : {stats['avg_total']:.6f}s")
        if avg_data > 0:
            frac = 100.0 * stats["avg_total"] / avg_data
            print(f"transform/data pct  : {frac:.1f}%")


if __name__ == "__main__":
    main()
