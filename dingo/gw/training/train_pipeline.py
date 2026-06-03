from typing import Optional, Tuple
import os

import numpy as np
import yaml
import argparse
import shutil
import textwrap
import time
from copy import deepcopy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from threadpoolctl import threadpool_limits

from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.gw.training.train_builders import (
    build_dataset,
    set_train_transforms,
    build_svd_for_embedding_network,
)
from dingo.gw.SVD import SVDBasis
from dingo.core.utils.trainutils import RuntimeLimits
from dingo.core.utils import (
    set_requires_grad_flag,
    get_number_of_model_parameters,
    build_train_and_test_loaders,
)
from dingo.core.utils.torchutils import (
    cleanup_ddp,
    replace_BatchNorm_with_SyncBatchNorm,
    set_seed_based_on_rank,
    setup_ddp,
)
from dingo.core.utils.trainutils import EarlyStopping
from dingo.gw.dataset import WaveformDataset
from dingo.core.posterior_models import BasePosteriorModel


def _resolve_svd_file_path(path: str, train_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(train_dir, path)


def _wait_for_files(
    file_paths,
    timeout_s: float = 7200.0,
    poll_s: float = 5.0,
):
    """
    Wait for all paths in file_paths to exist.

    This is used in DDP startup to let rank 0 build SVD files while other ranks wait
    without using NCCL collectives (which can timeout during long SVD construction).
    """
    start = time.time()
    missing = [p for p in file_paths if not os.path.exists(p)]
    while missing:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(
                "Timed out waiting for precomputed SVD files to appear: "
                f"{missing}. Waited {elapsed:.1f}s."
            )
        time.sleep(poll_s)
        missing = [p for p in file_paths if not os.path.exists(p)]


def _load_precomputed_v_rb_list(
    precomputed_files,
    detectors,
    domain_min_idx: int,
    train_dir: str,
):
    if isinstance(precomputed_files, dict):
        file_list = []
        for ifo in detectors:
            if ifo not in precomputed_files:
                raise KeyError(
                    f"Missing precomputed SVD file for detector '{ifo}'. "
                    f"Provided keys: {list(precomputed_files.keys())}"
                )
            file_list.append(precomputed_files[ifo])
    elif isinstance(precomputed_files, (list, tuple)):
        if len(precomputed_files) != len(detectors):
            raise ValueError(
                f"Expected {len(detectors)} precomputed SVD files, got "
                f"{len(precomputed_files)}."
            )
        file_list = list(precomputed_files)
    else:
        raise TypeError(
            "precomputed_files must be either a dict keyed by detector name "
            "or a list aligned with data.detectors."
        )

    v_rb_list = []
    sizes = []
    for ifo, path in zip(detectors, file_list):
        file_path = _resolve_svd_file_path(path, train_dir)
        print(f"Loading precomputed SVD for {ifo} from {file_path}")
        basis = SVDBasis(file_name=file_path)
        V = basis.V
        if V is None:
            raise ValueError(f"No V matrix found in {file_path}")
        V = V[domain_min_idx:]
        sizes.append(V.shape[1])
        v_rb_list.append(V)
    common_size = min(sizes)
    if len(set(sizes)) > 1:
        print(
            "Warning: precomputed SVD sizes differ across detectors: "
            f"{sizes}. Truncating all to common size {common_size}."
        )
        v_rb_list = [V[:, :common_size] for V in v_rb_list]
    return v_rb_list, common_size


def _parse_two_gpu_settings(local_settings: dict):
    enabled = bool(local_settings.get("two_gpu_split", False))
    gen_device = torch.device(local_settings.get("generation_device", "cuda:0"))
    train_device = torch.device(local_settings.get("training_device", "cuda:1"))

    if not enabled:
        return False, gen_device, train_device

    if not torch.cuda.is_available():
        raise RuntimeError("two_gpu_split=True requires CUDA, but CUDA is unavailable.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"two_gpu_split=True requires >=2 CUDA devices; found {torch.cuda.device_count()}."
        )
    if gen_device.type != "cuda" or train_device.type != "cuda":
        raise ValueError(
            "two_gpu_split requires CUDA devices. "
            f"Got generation_device={gen_device}, training_device={train_device}."
        )
    if gen_device.index == train_device.index:
        raise ValueError(
            "two_gpu_split requires distinct generation/training GPUs. "
            f"Both are set to {gen_device}."
        )
    return True, gen_device, train_device


def _configure_two_gpu_split_for_pm(pm: BasePosteriorModel, local_settings: dict):
    enabled, gen_device, train_device = _parse_two_gpu_settings(local_settings)
    pm.two_gpu_split = bool(enabled)

    if not enabled:
        pm.generation_device = None
        pm.training_device = None
        return

    # Keep data generation/transforms on generation GPU.
    torch.cuda.set_device(gen_device)
    pm.generation_device = gen_device
    pm.training_device = train_device

    # Put training network on training GPU.
    pm.network.to(train_device)
    pm.device = train_device
    print(
        "Enabled two-GPU split training: "
        f"generation on {pm.generation_device}, training on {pm.training_device}."
    )


def get_num_gpus(local_settings: dict) -> int:
    if "num_gpus" in local_settings:
        return int(local_settings["num_gpus"])
    condor_num_gpus = local_settings.get("condor", {}).get("num_gpus")
    if condor_num_gpus is not None:
        return int(condor_num_gpus)
    if torch.cuda.is_available():
        return int(torch.cuda.device_count())
    return 1


def _ddp_enabled(local_settings: dict) -> bool:
    backend = (
        local_settings.get("distributed", {})
        .get("backend", "")
        .strip()
        .lower()
    )
    return backend == "ddp"


def _run_training_ddp_worker(
    rank: int,
    world_size: int,
    train_settings: Optional[dict],
    local_settings: dict,
    train_dir: str,
    checkpoint_name: Optional[str],
    results,
):
    port = int(local_settings.get("distributed", {}).get("port", 12355))
    ddp_timeout_raw = local_settings.get("distributed", {}).get(
        "ddp_timeout_s",
        local_settings.get("distributed", {}).get(
            "svd_wait_timeout_s",
            local_settings.get("svd_wait_timeout_s", 7200),
        ),
    )
    ddp_timeout_s = float(ddp_timeout_raw)
    try:
        setup_ddp(rank, world_size, port=port, timeout_s=ddp_timeout_s)
        set_seed_based_on_rank(rank)

        local_settings_rank = deepcopy(local_settings)
        local_settings_rank["rank"] = rank
        local_settings_rank["world_size"] = world_size
        local_settings_rank["device"] = f"cuda:{rank}"
        local_settings_rank["two_gpu_split"] = False

        if checkpoint_name is None:
            train_settings_rank = deepcopy(train_settings)
            svd_cfg = (
                train_settings_rank.get("model", {})
                .get("embedding_kwargs", {})
                .get("svd", {})
            )
            needs_svd_build = bool(
                svd_cfg and ("precomputed_files" not in svd_cfg)
            )

            if needs_svd_build and rank != 0:
                detectors = train_settings_rank["data"]["detectors"]
                precomputed_files = {
                    ifo: os.path.join(train_dir, f"svd_{ifo}.hdf5")
                    for ifo in detectors
                }
                timeout_raw = local_settings.get("distributed", {}).get(
                    "svd_wait_timeout_s",
                    local_settings.get("svd_wait_timeout_s", 7200),
                )
                timeout_s = float(timeout_raw)
                print(
                    "DDP rank "
                    f"{rank}: waiting for precomputed SVD files with timeout {timeout_s:.0f}s: "
                    f"{list(precomputed_files.values())}"
                )
                _wait_for_files(
                    list(precomputed_files.values()),
                    timeout_s=timeout_s,
                    poll_s=5.0,
                )
                svd_cfg["precomputed_files"] = precomputed_files
                pm, wfd = prepare_training_new(
                    train_settings_rank, train_dir, local_settings_rank
                )
            else:
                pm, wfd = prepare_training_new(
                    train_settings_rank, train_dir, local_settings_rank
                )
        else:
            pm, wfd = prepare_training_resume(
                checkpoint_name, local_settings_rank, train_dir
            )

        # Keep ranks in lockstep before any DDP collectives. This avoids one rank
        # timing out in communicator setup while another is still finishing
        # expensive preparation (e.g., SVD initialization).
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        pm.network = replace_BatchNorm_with_SyncBatchNorm(pm.network)
        find_unused_parameters = bool(
            local_settings.get("distributed", {}).get("find_unused_parameters", True)
        )
        pm.network = torch.nn.parallel.DistributedDataParallel(
            pm.network,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=find_unused_parameters,
        )

        if local_settings.get("compile_network", False):
            pm.network = torch.compile(pm.network)

        with threadpool_limits(limits=1, user_api="blas"):
            complete = train_stages(pm, wfd, train_dir, local_settings_rank)

        if rank == 0:
            results["complete"] = bool(complete)
            results["error"] = ""
    except Exception as exc:
        if rank == 0:
            results["error"] = str(exc)
        raise
    finally:
        cleanup_ddp()


def _run_multi_gpu_training(
    train_settings: Optional[dict],
    local_settings: dict,
    train_dir: str,
    checkpoint_name: Optional[str],
) -> bool:
    if local_settings.get("two_gpu_split", False):
        raise ValueError("Cannot use local.distributed.backend=ddp with two_gpu_split.")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested but CUDA is unavailable.")

    world_size = get_num_gpus(local_settings)
    if world_size < 2:
        raise RuntimeError(
            f"DDP requested but num_gpus={world_size}. Set local.num_gpus >= 2."
        )

    manager = mp.Manager()
    results = manager.dict()
    results["complete"] = False
    results["error"] = ""

    mp.spawn(
        _run_training_ddp_worker,
        args=(
            world_size,
            train_settings,
            local_settings,
            train_dir,
            checkpoint_name,
            results,
        ),
        nprocs=world_size,
        join=True,
    )

    if results.get("error"):
        raise RuntimeError(f"DDP training failed: {results['error']}")
    return bool(results.get("complete", False))


def copy_files_to_local(
    file_path: str, local_dir: Optional[str], leave_keys_on_disk: bool, is_condor: bool = False,
) -> str:
    """
    Copy files to local node if local_dir is provided to minimize network traffic during training.

    Parameters
    ----------
    file_path: str
        Path to file that should be copied.
    local_dir: Optional[str]
        Directory where file should be copied. If None, file will not be copied.
    leave_keys_on_disk: bool
        Whether to leave keys on disk and load them during training. If dataset is not copied and
        leave_keys_on_disk is True, a warning will be raised.
    is_condor: bool
        Whether this is a condor job.

    Returns
    -------
    local_file_path: str
        Modified file path if file was copied to local node, else the original file path.
    """
    local_file_path = file_path
    if local_dir is not None:
        file_name = file_path.split("/")[-1]
        local_file_path = os.path.join(local_dir, file_name)
        print(f"Copying file to {local_file_path}")
        # Copy file
        start_time = time.time()
        shutil.copy(file_path, local_file_path)
        elapsed_time = time.time() - start_time
        print("Done. This took {:2.0f}:{:2.0f} min.".format(*divmod(elapsed_time, 60)))
    elif leave_keys_on_disk and is_condor:
        print(
            f"Warning: leave_waveforms_on_disk defaults to True, but local_cache_path is not specified. "
            f"This means that the waveforms will be loaded during training from {local_file_path} ."
            f"This can lead to unexpected long times for data loading during training due to network traffic. "
            f"To prevent this, specify 'local_cache_path = tmp' in the local settings or set "
            f"leave_waveforms_on_disk = False. However, the latter is not recommended for large datasets since "
            f"it can lead to memory issues when loading the entire dataset into RAM. "
        )

    return local_file_path


def prepare_training_new(
    train_settings: dict, train_dir: str, local_settings: dict
) -> Tuple[BasePosteriorModel, WaveformDataset]:
    """
    Based on a settings dictionary, initialize a WaveformDataset and PosteriorModel.

    For model type 'nsf+embedding' (the only acceptable type at this point) this also
    initializes the embedding network projection stage with SVD V matrices based on
    clean detector waveforms.

    Parameters
    ----------
    train_settings : dict
        Settings which ultimately come from train_settings.yaml file.
    train_dir : str
        This is only used to save diagnostics from the SVD.
    local_settings : dict
        Local settings (e.g., num_workers, device)

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """
    data_settings = deepcopy(train_settings["data"])
    two_gpu_enabled, gen_device, _ = _parse_two_gpu_settings(local_settings)
    if two_gpu_enabled:
        # Keep waveform generation/transforms on generation GPU.
        torch.cuda.set_device(gen_device)
    # Optionally copy files to local and update path
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )
    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        on_fly=local_settings.get("on_fly", True),
    )  # No transforms yet
    waveform_generator_settings = wfd.settings.setdefault("waveform_generator", {})
    if waveform_generator_settings.get("BBHx", False):
        if "gpu_fastpath" in local_settings:
            waveform_generator_settings["gpu_fastpath"] = bool(
                local_settings["gpu_fastpath"]
            )
        if "backend_native_fused" in local_settings:
            waveform_generator_settings["backend_native_fused"] = bool(
                local_settings["backend_native_fused"]
            )
        if "timing_profile" in local_settings:
            waveform_generator_settings["timing_profile"] = bool(
                local_settings["timing_profile"]
            )
        if "timing_profile_print_every" in local_settings:
            waveform_generator_settings["timing_profile_print_every"] = int(
                local_settings["timing_profile_print_every"]
            )
        if hasattr(wfd, "waveform_generator"):
            if "gpu_fastpath" in waveform_generator_settings:
                wfd.waveform_generator.gpu_fastpath = bool(
                    waveform_generator_settings["gpu_fastpath"]
                )
            if "backend_native_fused" in waveform_generator_settings:
                wfd.waveform_generator.backend_native_fused = bool(
                    waveform_generator_settings["backend_native_fused"]
                )
            if "timing_profile" in waveform_generator_settings:
                wfd.waveform_generator.timing_profile = bool(
                    waveform_generator_settings["timing_profile"]
                )
            if "timing_profile_print_every" in waveform_generator_settings:
                wfd.waveform_generator.timing_profile_print_every = int(
                    waveform_generator_settings["timing_profile_print_every"]
                )
    initial_weights = {}

    # The embedding network is assumed to have an SVD projection layer. If other types
    # of embedding networks are added in the future, update this code.

    _embedding_kwargs = train_settings["model"].get("embedding_kwargs", None)
    # The transformer embedding has no SVD/reduced-basis layer to seed, so skip
    # the SVD build entirely when embedding_kwargs has no `svd` block.
    if _embedding_kwargs and "svd" in _embedding_kwargs:
        svd_kwargs = deepcopy(train_settings["model"]["embedding_kwargs"]["svd"])
        precomputed_files = svd_kwargs.pop("precomputed_files", None)
        if precomputed_files is not None:
            print("\nUsing precomputed SVD matrices for embedding network.")
            v_rb_list, inferred_size = _load_precomputed_v_rb_list(
                precomputed_files=precomputed_files,
                detectors=train_settings["data"]["detectors"],
                domain_min_idx=wfd.domain.min_idx,
                train_dir=train_dir,
            )
            initial_weights["V_rb_list"] = v_rb_list
            # Ensure embedding network receives the required RB size even when
            # training YAML omits it for precomputed-file mode.
            train_settings["model"]["embedding_kwargs"].setdefault("svd", {})
            train_settings["model"]["embedding_kwargs"]["svd"]["size"] = int(
                inferred_size
            )
        else:
            # First, build the SVD for seeding the embedding network.
            print("\nBuilding SVD for initialization of embedding network.")
            initial_weights["V_rb_list"] = build_svd_for_embedding_network(
                wfd,
                train_settings["data"],
                train_settings["training"]["stage_0"]["asd_dataset_path"],
                num_workers=local_settings["num_workers"],
                batch_size=train_settings["training"]["stage_0"]["batch_size"],
                out_dir=train_dir,
                **svd_kwargs,
            )

    # Now set the transforms for training. We need to do this here so that we can (a)
    # get the data dimensions to configure the network, and (b) save the
    # parameter standardization dict in the PosteriorModel. In principle, (a) could
    # be done without generating data (by careful calculation) and (b) could also
    # be done outside the transform setup. But for now, this is convenient. The
    # transforms will be reset later by initialize_stage().

    set_train_transforms(
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset_path"],
    )

    # This modifies the model settings in-place.
    autocomplete_model_kwargs(train_settings["model"], wfd[0])
    full_settings = {
        "dataset_settings": wfd.settings,
        "train_settings": train_settings,
    }

    print("\nInitializing new posterior model.")
    print("Complete settings:")
    print(yaml.dump(full_settings, default_flow_style=False, sort_keys=False))

    pm = build_model_from_kwargs(
        settings=full_settings,
        initial_weights=initial_weights,
        device=local_settings["device"],
    )
    _configure_two_gpu_split_for_pm(pm, local_settings)
    pm.cuda_batch_prefetch = bool(local_settings.get("cuda_batch_prefetch", False))
    if pm.cuda_batch_prefetch:
        print("Enabled CUDA batch prefetch (background DataLoader iterator).")

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                config=full_settings,
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    return pm, wfd


def prepare_training_resume(
    checkpoint_name: str, local_settings: dict, train_dir: str
) -> Tuple[BasePosteriorModel, WaveformDataset]:
    """
    Loads a PosteriorModel from a checkpoint, as well as the corresponding
    WaveformDataset, in order to continue training. It initializes the saved optimizer
    and scheduler from the checkpoint.

    Parameters
    ----------
    checkpoint_name : str
        File name containing the checkpoint (.pt format).
    local_settings : dict
        Local settings (e.g., num_workers, device)
    train_dir: str
        Path to training directory where the wandb info is saved.

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """

    two_gpu_enabled, gen_device, _ = _parse_two_gpu_settings(local_settings)
    if two_gpu_enabled:
        torch.cuda.set_device(gen_device)

    pm = build_model_from_kwargs(
        filename=checkpoint_name, device=local_settings["device"]
    )
    _configure_two_gpu_split_for_pm(pm, local_settings)
    pm.cuda_batch_prefetch = bool(local_settings.get("cuda_batch_prefetch", False))
    if pm.cuda_batch_prefetch:
        print("Enabled CUDA batch prefetch (background DataLoader iterator).")
    data_settings = deepcopy(pm.metadata["train_settings"]["data"])
    # Optionally copy files to local and update path
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )
    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        on_fly=local_settings.get("on_fly", True),
    )
    waveform_generator_settings = wfd.settings.setdefault("waveform_generator", {})
    if waveform_generator_settings.get("BBHx", False) and hasattr(wfd, "waveform_generator"):
        if "gpu_fastpath" in local_settings:
            waveform_generator_settings["gpu_fastpath"] = bool(
                local_settings["gpu_fastpath"]
            )
            wfd.waveform_generator.gpu_fastpath = bool(local_settings["gpu_fastpath"])
        if "backend_native_fused" in local_settings:
            waveform_generator_settings["backend_native_fused"] = bool(
                local_settings["backend_native_fused"]
            )
            wfd.waveform_generator.backend_native_fused = bool(
                local_settings["backend_native_fused"]
            )
        if "timing_profile" in local_settings:
            waveform_generator_settings["timing_profile"] = bool(
                local_settings["timing_profile"]
            )
            wfd.waveform_generator.timing_profile = bool(local_settings["timing_profile"])
        if "timing_profile_print_every" in local_settings:
            waveform_generator_settings["timing_profile_print_every"] = int(
                local_settings["timing_profile_print_every"]
            )
            wfd.waveform_generator.timing_profile_print_every = int(
                local_settings["timing_profile_print_every"]
            )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                resume="must",
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    return pm, wfd


def initialize_stage(
    pm: BasePosteriorModel,
    wfd: WaveformDataset,
    stage: dict,
    num_workers: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    resume: bool = False,
):
    """
    Initializes training based on PosteriorModel metadata and current stage:
        * Builds transforms (based on noise settings for current stage);
        * Builds DataLoaders;
        * At the beginning of a stage (i.e., if not resuming mid-stage), initializes
        a new optimizer and scheduler;
        * Freezes / unfreezes SVD layer of embedding network

    Parameters
    ----------
    pm : BasePosteriorModel
    wfd : WaveformDataset
    stage : dict
        Settings specific to current stage of training
    num_workers : int
    resume : bool
        Whether training is resuming mid-stage. This controls whether the optimizer and
        scheduler should be re-initialized based on contents of stage dict.

    Returns
    -------
    (train_loader, test_loader, train_sampler)
    """

    train_settings = pm.metadata["train_settings"]
    print_output = rank is None or rank == 0

    # Ensure transform/data pipeline remains on generation GPU in split mode.
    if bool(getattr(pm, "two_gpu_split", False)):
        torch.cuda.set_device(pm.generation_device)

    # Rebuild transforms based on possibly different noise.
    set_train_transforms(
        wfd,
        train_settings["data"],
        stage["asd_dataset_path"],
        print_output=print_output,
    )

    if world_size is not None and world_size > 1:
        total_batch_size = stage["batch_size"]
        if total_batch_size % world_size != 0:
            raise ValueError(
                f"Total batch size {total_batch_size} is not divisible by "
                f"world_size={world_size}."
            )
        batch_size_per_gpu = total_batch_size // world_size
    else:
        batch_size_per_gpu = stage["batch_size"]

    # Allows for changes in batch size between stages.
    train_loader, test_loader, train_sampler = build_train_and_test_loaders(
        wfd,
        train_settings["data"]["train_fraction"],
        batch_size_per_gpu,
        num_workers,
        world_size=world_size,
        rank=rank,
    )

    if not resume:
        # New optimizer and scheduler. If we are resuming, these should have been
        # loaded from the checkpoint.
        if print_output:
            print("Initializing new optimizer and scheduler.")
        pm.optimizer_kwargs = stage["optimizer"]
        pm.scheduler_kwargs = stage["scheduler"]
        pm.initialize_optimizer_and_scheduler()

    # Freeze/unfreeze RB layer if necessary
    if "freeze_rb_layer" in stage:
        if stage["freeze_rb_layer"]:
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=False
            )
        else:
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=True
            )
    if print_output:
        n_grad = get_number_of_model_parameters(pm.network, (True,))
        n_nograd = get_number_of_model_parameters(pm.network, (False,))
        print(f"Fixed parameters: {n_nograd}\nLearnable parameters: {n_grad}\n")

    return train_loader, test_loader, train_sampler


def train_stages(
    pm: BasePosteriorModel, wfd: WaveformDataset, train_dir: str, local_settings: dict
) -> bool:
    """
    Train the network, iterating through the sequence of stages. Stages can change
    certain settings such as the noise characteristics, optimizer, and scheduler settings.

    Parameters
    ----------
    pm : BasePosteriorModel
    wfd : WaveformDataset
    train_dir : str
        Directory for saving checkpoints and train history.
    local_settings : dict

    Returns
    -------
    bool
        True if all stages are complete
        False otherwise
    """

    train_settings = pm.metadata["train_settings"]
    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch, **local_settings["runtime_limits"]
    )
    rank = local_settings.get("rank", None)
    world_size = local_settings.get("world_size", None)
    print_primary = rank is None or rank == 0

    # Extract list of stages from settings dict
    stages = []
    num_stages = 0
    while True:
        try:
            stages.append(train_settings["training"][f"stage_{num_stages}"])
            num_stages += 1
        except KeyError:
            break
    end_epochs = list(np.cumsum([stage["epochs"] for stage in stages]))

    num_starting_stage = np.searchsorted(end_epochs, pm.epoch + 1)
    for n in range(num_starting_stage, num_stages):
        stage = stages[n]

        if pm.epoch == end_epochs[n] - stage["epochs"]:
            if print_primary:
                print(f"\nBeginning training stage {n}. Settings:")
                print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader, train_sampler = initialize_stage(
                pm,
                wfd,
                stage,
                local_settings["num_workers"],
                world_size=world_size,
                rank=rank,
                resume=False,
            )
        else:
            if print_primary:
                print(f"\nResuming training in stage {n}. Settings:")
                print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader, train_sampler = initialize_stage(
                pm,
                wfd,
                stage,
                local_settings["num_workers"],
                world_size=world_size,
                rank=rank,
                resume=True,
            )
        early_stopping = None
        if stage.get("early_stopping"):
            try:
                early_stopping = EarlyStopping(**stage["early_stopping"])
            except Exception:
                print(
                    "Early stopping settings invalid. Please pass 'patience', 'delta', 'metric'"
                )
                raise

        runtime_limits.max_epochs_total = end_epochs[n]
        pm.train(
            train_loader,
            test_loader,
            train_dir=train_dir,
            train_sampler=train_sampler,
            runtime_limits=runtime_limits,
            checkpoint_epochs=local_settings["checkpoint_epochs"],
            use_wandb=local_settings.get("wandb", False),
            test_only=local_settings.get("test_only", False),
            early_stopping=early_stopping,
            train_print_freq=local_settings.get("train_print_freq", 50),
            test_print_freq=local_settings.get("test_print_freq", 50),
            gradient_updates_per_optimizer_step=stage.get(
                "gradient_updates_per_optimizer_step", 1
            ),
            automatic_mixed_precision=stage.get("automatic_mixed_precision", False),
        )
        # if test_only, model should not be saved, and run is complete
        if local_settings.get("test_only", False):
            return True

        if pm.epoch == end_epochs[n] and print_primary:
            save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
            print(f"Training stage complete. Saving to {save_file}.")
            pm.save_model(save_file, save_training_info=True)
        if runtime_limits.local_limits_exceeded(pm.epoch):
            if print_primary:
                print("Local runtime limits reached. Ending program.")
            break

    if pm.epoch == end_epochs[-1]:
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Train a neural network for gravitational-wave single-event inference.
        
        This program can be called in one of two ways:
            a) with a settings file. This will create a new network based on the 
            contents of the settings file.
            b) with a checkpoint file. This will resume training from the checkpoint.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        help="YAML file containing training settings.",
    )
    parser.add_argument(
        "--train_dir", required=True, help="Directory for Dingo training output."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file from which to resume training.",
    )
    parser.add_argument(
        "--exit_command",
        type=str,
        default="",
        help="Optional command to execute after completion of training.",
    )
    args = parser.parse_args()

    # The settings file and checkpoint are mutually exclusive.
    if args.checkpoint is None and args.settings_file is None:
        parser.error("Must specify either a checkpoint file or a settings file.")
    if args.checkpoint is not None and args.settings_file is not None:
        parser.error("Cannot specify both a checkpoint file and a settings file.")

    return args


def train_local():
    args = parse_args()

    os.makedirs(args.train_dir, exist_ok=True)

    if args.settings_file is not None:
        print("Beginning new training run.")
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

        # Extract the local settings from train settings file, save it separately. This
        # file can later be modified, and the settings take effect immediately upon
        # resuming.

        local_settings = train_settings.pop("local")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "w") as f:
            if (
                local_settings.get("wandb", False)
                and "id" not in local_settings["wandb"].keys()
            ):
                try:
                    import wandb

                    local_settings["wandb"]["id"] = wandb.util.generate_id()
                except ImportError:
                    print("wandb not installed, cannot generate run id.")
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

    else:
        print("Resuming training run.")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)
    use_ddp = _ddp_enabled(local_settings)

    if use_ddp:
        complete = _run_multi_gpu_training(
            train_settings if args.settings_file is not None else None,
            local_settings,
            args.train_dir,
            args.checkpoint if args.settings_file is None else None,
        )
    else:
        if args.settings_file is not None:
            pm, wfd = prepare_training_new(train_settings, args.train_dir, local_settings)
        else:
            pm, wfd = prepare_training_resume(
                args.checkpoint, local_settings, args.train_dir
            )
        if local_settings.get("compile_network", False):
            pm.network = torch.compile(pm.network)
        with threadpool_limits(limits=1, user_api="blas"):
            complete = train_stages(pm, wfd, args.train_dir, local_settings)

    if complete:
        if args.exit_command:
            print(
                f"All training stages complete. Executing exit command: {args.exit_command}."
            )
            os.system(args.exit_command)
        else:
            print("All training stages complete.")
    else:
        print("Program terminated due to runtime limit.")
