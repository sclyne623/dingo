import os
from socket import gethostname
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from typing import Any, Optional, Union, Tuple, Iterable
import bilby


def fix_random_seeds(_):
    """Utility function to set random seeds when using multiple workers for DataLoader."""
    np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1))
    try:
        bilby.core.utils.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1))
    except AttributeError:  # In case using an old version of Bilby.
        pass


def get_cuda_info() -> dict[str, Any]:
    """Get a small snapshot of visible CUDA resources."""
    if not torch.cuda.is_available():
        return {}
    return {
        "cuDNN version": torch.backends.cudnn.version(),
        "CUDA version": torch.version.cuda,
        "device count": torch.cuda.device_count(),
        "device name": torch.cuda.get_device_name(0),
        "memory (GB)": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        ),
    }


def document_gpus(target_dir: str) -> None:
    """Write GPU diagnostics into info_gpus.txt in target_dir."""
    cuda_info = get_cuda_info()
    with open(os.path.join(target_dir, "info_gpus.txt"), "w") as f:
        f.write(f"# Running on host:\n{gethostname()}\n")
        f.write("# CUDA information:\n")
        for k, v in cuda_info.items():
            f.write(f"{k}: {v}\n")


def set_seed_based_on_rank(rank: int) -> None:
    """Offset random seeds by rank so each DDP process gets unique draws."""
    base_seed = int(torch.initial_seed())
    torch.manual_seed(base_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed + rank)
    np.random.seed((base_seed % (2**32 - 1)) + rank)


def enable_tf32(print_output: bool = True) -> None:
    """
    Allow TF32 tensor-core math for float32 matmuls and cuDNN on Ampere+ GPUs.

    PyTorch < 1.12 enabled TF32 by default, so historical Dingo training on
    A100s used it implicitly; later PyTorch versions default to full-FP32
    matmuls, which are several times slower. TF32 keeps FP32 accumulation and
    only reduces the matmul input mantissa, leaving network weights, loss,
    optimizer state, and inference in FP32.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if print_output:
            print("Enabled TF32 tensor-core math for matmul and cuDNN.")


def setup_ddp(
    rank: int,
    world_size: int,
    port: int = 12355,
    timeout_s: float = 600.0,
) -> None:
    """
    Initialize NCCL process group.
    If launched via torchrun/srun env, honor existing MASTER_* values.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(port))
    if not dist.is_nccl_available():
        raise RuntimeError("NCCL backend unavailable; cannot run DDP on CUDA.")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=float(timeout_s)),
        device_id=torch.device("cuda", rank),
    )


def cleanup_ddp() -> None:
    """Destroy process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def replace_BatchNorm_with_SyncBatchNorm(network: nn.Module) -> nn.Module:
    """Convert BN layers to SyncBatchNorm for multi-GPU DDP."""
    return nn.SyncBatchNorm.convert_sync_batchnorm(network)


def print_number_of_model_parameters(network: nn.Module) -> None:
    """Print fixed/learnable parameter counts (DDP-aware)."""
    bare = network.module if isinstance(network, DDP) else network
    n_grad = get_number_of_model_parameters(network, (True,))
    n_nograd = get_number_of_model_parameters(network, (False,))
    print(f"Fixed parameters: {n_nograd}\nLearnable parameters: {n_grad}")
    try:
        if bare.name == "FlowWrapper":
            n_emb = get_number_of_model_parameters(bare.embedding_net, (True,))
            n_flow = get_number_of_model_parameters(bare.flow, (True,))
            print(
                f"   - learnable embedding network parameters: {n_emb} ({n_emb / n_grad * 100:.2f}%)\n"
                f"   - learnable flow parameters: {n_flow} ({n_flow / n_grad * 100:.2f}%)"
            )
    except Exception:
        pass


def get_activation_function_from_string(activation_name: str):
    """
    Returns an activation function, based on the name provided.

    :param activation_name: str
        name of the activation function, one of {'elu', 'relu', 'leaky_rely'}
    :return: function
        corresponding activation function
    """
    if activation_name.lower() == "elu":
        return F.elu
    elif activation_name.lower() == "relu":
        return F.relu
    elif activation_name.lower() == "leaky_relu":
        return F.leaky_relu
    elif activation_name.lower() == "gelu":
        return F.gelu
    else:
        raise ValueError("Invalid activation function.")


def get_number_of_model_parameters(
    model: nn.Module,
    requires_grad_flags: tuple = (True, False),
):
    """
    Counts parameters of the module. The list requires_grad_flag can be used
    to specify whether all parameters should be counted, or only those with
    requires_grad = True or False.
    :param model: nn.Module
        model
    :param requires_grad_flags: tuple
        tuple of bools, for requested requires_grad flags
    :return:
        number of parameters of the model with requested required_grad flags
    """
    num_params = 0
    for p in list(model.parameters()):
        if p.requires_grad in requires_grad_flags:
            n = 1
            for s in list(p.size()):
                n = n * s
            num_params += n
    return num_params


def get_optimizer_from_kwargs(
    model_parameters: Iterable,
    **optimizer_kwargs,
):
    """
    Builds and returns an optimizer for model_parameters. The type of the
    optimizer is determined by kwarg type, the remaining kwargs are passed to
    the optimizer.

    Parameters
    ----------
    model_parameters: Iterable
        iterable of parameters to optimize or dicts defining parameter groups
    optimizer_kwargs:
        kwargs for optimizer; type needs to be one of [adagrad, adam, adamw,
        lbfgs, RMSprop, sgd], the remaining kwargs are used for specific
        optimizer kwargs, such as learning rate and momentum

    Returns
    -------
    optimizer
    """
    optimizers_dict = {
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "lbfgs": torch.optim.LBFGS,
        "RMSprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD,
    }
    if not "type" in optimizer_kwargs:
        raise KeyError("Optimizer type needs to be specified.")
    if not optimizer_kwargs["type"].lower() in optimizers_dict:
        raise ValueError("No valid optimizer specified.")
    optimizer = optimizers_dict[optimizer_kwargs.pop("type")]
    return optimizer(model_parameters, **optimizer_kwargs)


def get_scheduler_from_kwargs(
    optimizer: torch.optim.Optimizer,
    **scheduler_kwargs,
):
    """
    Builds and returns an scheduler for optimizer. The type of the
    scheduler is determined by kwarg type, the remaining kwargs are passed to
    the scheduler.

    Parameters
    ----------
    optimizer: torch.optim.optimizer.Optimizer
        optimizer for which the scheduler is used
    scheduler_kwargs:
        kwargs for scheduler; type needs to be one of [step, cosine,
        reduce_on_plateau], the remaining kwargs are used for
        specific scheduler kwargs, such as learning rate and momentum

    Returns
    -------
    scheduler
    """
    schedulers_dict = {
        "step": torch.optim.lr_scheduler.StepLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    if not "type" in scheduler_kwargs:
        raise KeyError("Scheduler type needs to be specified.")
    if not scheduler_kwargs["type"].lower() in schedulers_dict:
        raise ValueError("No valid scheduler specified.")
    scheduler = schedulers_dict[scheduler_kwargs.pop("type")]
    return scheduler(optimizer, **scheduler_kwargs)


def perform_scheduler_step(
    scheduler,
    loss=None,
):
    """
    Wrapper for scheduler.step(). If scheduler is ReduceLROnPlateau,
    then scheduler.step(loss) is called, if not, scheduler.step().

    Parameters
    ----------

    scheduler:
        scheduler for learning rate
    loss:
        validation loss
    """
    if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(loss)
    else:
        scheduler.step()


def get_lr(optimizer):
    """Returns a list with the learning rates of the optimizer."""
    return [param_group["lr"] for param_group in optimizer.state_dict()["param_groups"]]


def split_dataset_into_train_and_test(dataset, train_fraction):
    """
    Splits dataset into a trainset of size int(train_fraction * len(dataset)),
    and a testset with the remainder. Uses fixed random seed for
    reproducibility.

    Parameters
    ----------
    dataset: torch.utils.data.Datset
        dataset to be split
    train_fraction: float
        fraction of the dataset to be used for trainset

    Returns
    -------
    trainset, testset
    """
    train_size = int(train_fraction * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )


def build_train_and_test_loaders(
    dataset: torch.utils.data.Dataset,
    train_fraction: float,
    batch_size: int,
    num_workers: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
):
    """
    Split the dataset into train and test sets, and build corresponding DataLoaders.
    The random split uses a fixed seed for reproducibility.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    train_fraction : float
        Fraction of dataset to use for training. The remainder is used for testing.
        Should lie between 0 and 1.
    batch_size : int
    num_workers : int

    Returns
    -------
    (train_loader, test_loader, train_sampler)
    """

    # Split the dataset. This function uses a fixed seed for reproducibility.
    train_dataset, test_dataset = split_dataset_into_train_and_test(
        dataset, train_fraction
    )

    # Build DataLoaders.
    # If dataset items are already CUDA tensors (BBHx GPU fast-path), avoid worker
    # multiprocessing and pinning, which are CPU tensor optimizations.
    output_on_cuda = bool(getattr(dataset, "output_on_cuda", False))
    if output_on_cuda and num_workers != 0:
        print(
            "Dataset outputs CUDA tensors. Forcing num_workers=0 to avoid "
            "CUDA multiprocessing instability in DataLoader workers."
        )
        num_workers = 0
    pin_memory = not output_on_cuda

    returns_batched_output = bool(getattr(dataset, "returns_batched_output", False))

    def collate_passthrough_or_default(batch):
        # If dataset.__getitems__ already returned a fully-batched object
        # (e.g., [theta_batch, waveform_batch]), keep it as-is.
        if (
            returns_batched_output
            and isinstance(batch, list)
            and len(batch) > 0
            and all(isinstance(x, (torch.Tensor, np.ndarray)) for x in batch)
        ):
            return batch
        return default_collate(batch)

    if rank is not None and world_size is not None:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, num_replicas=world_size, rank=rank
        )
        test_sampler = DistributedSampler(
            test_dataset, shuffle=False, num_replicas=world_size, rank=rank
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=fix_random_seeds,
            collate_fn=collate_passthrough_or_default if returns_batched_output else None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=fix_random_seeds,
            collate_fn=collate_passthrough_or_default if returns_batched_output else None,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=fix_random_seeds,
            collate_fn=collate_passthrough_or_default if returns_batched_output else None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=fix_random_seeds,
            collate_fn=collate_passthrough_or_default if returns_batched_output else None,
        )

    return train_loader, test_loader, train_sampler


def set_requires_grad_flag(
    model, name_startswith=None, name_contains=None, requires_grad=True
):
    """
    Set param.requires_grad of all model parameters with a name starting with
    name_startswith, or name containing name_contains, to requires_grad.
    """
    for name, param in model.named_parameters():
        if (
            name_startswith is not None
            and name.startswith(name_startswith)
            or name_contains is not None
            and name_contains in name
        ):
            param.requires_grad = requires_grad


def torch_detach_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x
