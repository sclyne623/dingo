"""
Step-limited DDP throughput prototype for BBHx/LISA training.

Run with torchrun, e.g.
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      -m dingo.gw.training.ddp_pipeline_prototype \
      --settings_file /path/to/more_params.yaml \
      --train_dir /path/to/run_dir \
      --steps 500
"""

import argparse
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from dingo.gw.training.train_pipeline import initialize_stage, prepare_training_new


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
    return parser.parse_args()


def _to_device_if_needed(x, device):
    if isinstance(x, torch.Tensor):
        if x.device == device:
            return x
        return x.to(device, non_blocking=True)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device, non_blocking=True)
    return torch.as_tensor(x, device=device)


def _init_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = _parse_args()
    rank, world_size, local_rank = _init_ddp()
    device = torch.device(f"cuda:{local_rank}")

    try:
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

        local_settings = deepcopy(train_settings.pop("local"))
        local_settings["device"] = str(device)
        local_settings["rank"] = rank
        local_settings["world_size"] = world_size
        local_settings["two_gpu_split"] = False
        local_settings.setdefault("distributed", {})
        local_settings["distributed"]["backend"] = "ddp"

        if args.backend_native_fused is not None:
            local_settings["backend_native_fused"] = (
                args.backend_native_fused.lower() == "true"
            )

        pm, wfd = prepare_training_new(train_settings, args.train_dir, local_settings)
        find_unused_parameters = bool(
            local_settings.get("distributed", {}).get("find_unused_parameters", True)
        )
        pm.network = DDP(
            pm.network,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )

        stage0 = train_settings["training"]["stage_0"]
        train_loader, _, _ = initialize_stage(
            pm,
            wfd,
            stage0,
            local_settings["num_workers"],
            world_size=world_size,
            rank=rank,
            resume=False,
        )

        pm.network.train()
        data_iter = iter(train_loader)

        data_sum = 0.0
        transfer_sum = 0.0
        network_sum = 0.0
        counted = 0
        last_loss = float("nan")

        for step_idx in range(args.steps):
            t0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            t1 = time.perf_counter()

            batch = [_to_device_if_needed(d, device) for d in batch]
            t2 = time.perf_counter()

            pm.optimizer.zero_grad(set_to_none=True)
            loss = pm.loss(batch[0], *batch[1:])
            loss.backward()
            pm.optimizer.step()
            dist.barrier()
            t3 = time.perf_counter()

            last_loss = float(loss.item())
            if step_idx >= args.warmup_steps:
                data_sum += t1 - t0
                transfer_sum += t2 - t1
                network_sum += t3 - t2
                counted += 1

            if rank == 0 and (step_idx + 1) % args.print_every == 0:
                print(
                    f"step={step_idx+1:5d} "
                    f"loss={last_loss:.4f} "
                    f"data={t1-t0:.3f}s "
                    f"transfer={t2-t1:.3f}s "
                    f"network={t3-t2:.3f}s"
                )

        stats = torch.tensor(
            [data_sum, transfer_sum, network_sum, float(counted), last_loss],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        if rank == 0:
            total_count = max(1.0, stats[3].item() / world_size)
            avg_data = (stats[0].item() / world_size) / total_count
            avg_transfer = (stats[1].item() / world_size) / total_count
            avg_network = (stats[2].item() / world_size) / total_count
            avg_total = avg_data + avg_transfer + avg_network
            per_rank_batch = train_loader.batch_size
            global_batch = per_rank_batch * world_size
            samples_per_s = global_batch / avg_total if avg_total > 0 else float("nan")
            avg_loss = stats[4].item() / world_size

            print("\n=== DDP Prototype Summary ===")
            print(f"world_size          : {world_size}")
            print(f"device(rank0)       : {device}")
            print(
                f"backend_native_fused: {bool(local_settings.get('backend_native_fused', False))}"
            )
            print(f"per_rank_batch      : {per_rank_batch}")
            print(f"global_batch        : {global_batch}")
            print(f"avg_data            : {avg_data:.3f}s")
            print(f"avg_transfer        : {avg_transfer:.3f}s")
            print(f"avg_network         : {avg_network:.3f}s")
            print(f"avg_total           : {avg_total:.3f}s")
            print(f"samples_per_s       : {samples_per_s:.1f}")
            print(f"mean_loss(last_step): {avg_loss:.4f}")

    finally:
        _cleanup_ddp()


if __name__ == "__main__":
    main()
