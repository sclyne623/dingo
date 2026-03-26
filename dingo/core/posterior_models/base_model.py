"""
This module contains the abstract base class for representing posterior models,
as well as functions for training and testing across an epoch.
"""

from abc import abstractmethod, ABC
import contextlib
import os
from os.path import join
import h5py
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist
import dingo.core.utils as utils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
import time
import numpy as np
import dingo.core.utils.trainutils
import json
from collections import OrderedDict
from typing import Optional
from dingo.core.utils.backward_compatibility import update_model_config
from dingo.core.utils.misc import get_version

from dingo.core.utils.trainutils import EarlyStopping

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    # PyTorch < 2.3 fallback
    from torch.cuda.amp import GradScaler as _CudaGradScaler, autocast

    class GradScaler:  # type: ignore[no-redef]
        def __new__(cls, device="cuda", **kwargs):
            return _CudaGradScaler(**kwargs)


class BasePosteriorModel(ABC):
    """
    Abstract base class for PosteriorModels. This is intended to construct and hold a
    neural network for estimating the posterior density, as well as saving / loading,
    and training.

    Subclasses must implement methods for constructing the specific network, sampling,
    density evaluation, and computing the loss during training.
    """

    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
    ):
        """
        Initialize a model for the posterior distribution.

        Parameters
        ----------
        model_filename: str
            If given, loads data from the given file.
        metadata: dict
            If given, initializes the model from these settings
        initial_weights: dict
            Initial weights for the model
        device: str
        load_training_info: bool
        """

        self.version = f"dingo={get_version()}"  # dingo version

        self.device = None
        self.rank = None
        self.optimizer_kwargs = None
        self.network_kwargs = None
        self.scheduler_kwargs = None
        self.initial_weights = initial_weights

        self.metadata = metadata
        if self.metadata is not None:
            self.model_kwargs = self.metadata["train_settings"]["model"]
            # Expect self.optimizer_settings and self.scheduler_settings to be set
            # separately, and before calling initialize_optimizer_and_scheduler().

        self.epoch = 0
        self.network = None
        self.optimizer = None
        self.scheduler = None
        self.context = None
        self.event_metadata = None

        # build model
        if model_filename is not None:
            self.load_model(
                model_filename, load_training_info=load_training_info, device=device
            )
        else:
            self.initialize_network()
            self.network_to_device(device)

    @abstractmethod
    def initialize_network(self):
        """
        Initialize the network backbone for the posterior model.
        """
        pass

    @abstractmethod
    def sample(self, *context: torch.Tensor, num_samples: int = 1):
        """
        Sample parameters theta from the posterior model,

        theta ~ p(theta | context)

        Parameters
        ----------
        context: torch.Tensor
            Context information (typically observed data). Should have a batch
            dimension (even if size B = 1).
        num_samples: int = 1
            Number of samples to generate.

        Returns
        -------
        samples: torch.Tensor
            Shape (B, num_samples, dim(theta))
        """
        pass

    @abstractmethod
    def sample_and_log_prob(self, *context: torch.Tensor, num_samples: int = 1):
        """
        Sample parameters theta from the posterior model,

        theta ~ p(theta | context)

        and also return the log_prob. For models such as normalizing flows, it is more
        economical to calculate the log_prob at the same time as sampling, rather than
        as a separate step.

        Parameters
        ----------
        context: torch.Tensor
            Context information (typically observed data). Should have a batch
            dimension (even if size B = 1).
        num_samples: int = 1
            Number of samples to generate.

        Returns
        -------
        samples, log_prob: torch.Tensor, torch.Tensor
            Shapes (B, num_samples, dim(theta)), (B, num_samples)
        """
        pass

    @abstractmethod
    def log_prob(self, theta: torch.Tensor, *context: torch.Tensor):
        """
        Evaluate the log posterior density,

        log p(theta | context)

        Parameters
        ----------
        theta: torch.Tensor
            Parameter values at which to evaluate the density. Should have a batch
            dimension (even if size B = 1).
        context: torch.Tensor
            Context information (typically observed data). Must have context.shape[0] = B.

        Returns
        -------
        log_prob: torch.Tensor
            Shape (B,)
        """
        pass

    @abstractmethod
    def loss(self, theta: torch.Tensor, *context: torch.Tensor):
        """
        Compute the loss for a batch of data.

        Parameters
        ----------
        theta: torch.Tensor
            Parameter values at which to evaluate the density. Should have a batch
            dimension (even if size B = 1).
        context: torch.Tensor
            Context information (typically observed data). Must have the same leading
            (batch) dimension as theta.

        Returns
        -------
        loss: torch.Tensor
            Mean loss across the batch (a scalar).
        """
        pass

    def network_to_device(self, device):
        """
        Put model to device, and set self.device accordingly.
        """
        if "cpu" not in device and "cuda" not in device:
            raise ValueError(f"Device should contain cpu or cuda, got {device}.")
        if ":" in device and "cuda" in device:
            self.rank = int(device.split(":")[1])
        self.device = torch.device(device)
        # Commented below so that code runs on first cuda device in the case of multiple.
        # if device == 'cuda' and torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs.")
        #     raise NotImplementedError('This needs testing!')
        #     # dim = 0 [512, ...] -> [256, ...], [256, ...] on 2 GPUs
        #     self.network = torch.nn.DataParallel(self.network)
        print(f"Putting posterior model to device {self.device}.")
        self.network.to(self.device)

    def initialize_optimizer_and_scheduler(self):
        """
        Initializes the optimizer and scheduler with self.optimizer_kwargs
        and self.scheduler_kwargs, respectively.
        """
        if self.optimizer_kwargs is not None:
            self.optimizer = utils.get_optimizer_from_kwargs(
                self.network.parameters(), **self.optimizer_kwargs
            )
        if self.scheduler_kwargs is not None:
            self.scheduler = utils.get_scheduler_from_kwargs(
                self.optimizer, **self.scheduler_kwargs
            )

    def save_model(
        self,
        model_filename: str,
        save_training_info: bool = True,
    ):
        """
        Save the posterior model to the disk.

        Parameters
        ----------
        model_filename: str
            filename for saving the model
        save_training_info: bool
            specifies whether information required to proceed with training is
            saved, e.g. optimizer state dict

        """
        model_dict = {
            "model_kwargs": self.model_kwargs,
            "model_state_dict": (
                self.network.module.state_dict()
                if isinstance(self.network, DDP)
                else self.network.state_dict()
            ),
            "epoch": self.epoch,
            "version": self.version,
        }

        if self.metadata is not None:
            model_dict["metadata"] = self.metadata

        if self.context is not None:
            model_dict["context"] = self.context

        if self.event_metadata is not None:
            model_dict["event_metadata"] = self.event_metadata

        if save_training_info:
            model_dict["optimizer_kwargs"] = self.optimizer_kwargs
            model_dict["scheduler_kwargs"] = self.scheduler_kwargs
            if self.optimizer is not None:
                model_dict["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                model_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(model_dict, model_filename)

    def _load_model_from_hdf5(self, model_filename: str):
        """
        Helper function to load a trained model that has been
        saved in HDF5 format using `dingo_pt_to_hdf5`.
        Parameters
        ----------
        model_filename: str
            path to saved model; must have extension '.hdf5'
        Returns
        -------
        d: dict
            A stripped down version of the dict saved by torch.save()
            Specifically, it does not include 'optimizer_state_dict'
            to save space at inference time.
        """
        d = {}
        with h5py.File(model_filename, "r") as fp:
            model_basename = os.path.basename(model_filename)
            if fp.attrs["CANONICAL_FILE_BASENAME"] != model_basename:
                raise ValueError(
                    "HDF5 attribute CANONICAL_FILE_BASENAME differs from model name",
                    model_basename,
                )

            # Load small nested dicts from json
            for k, v in fp["serialized_dicts"].items():
                d[k] = json.loads(v[()])

            # Load model weights
            model_state_dict = OrderedDict()
            for k, v in fp["model_weights"].items():
                model_state_dict[k] = torch.from_numpy(np.array(v, dtype=np.float32))
            d["model_state_dict"] = model_state_dict

        return d

    def load_model(
        self,
        model_filename: str,
        load_training_info: bool = True,
        device: str = "cuda",
    ):
        """
        Load a posterior model from the disk.

        Parameters
        ----------
        model_filename: str
            path to saved model
        load_training_info: bool #TODO: load information for training
            specifies whether information required to proceed with training is
            loaded, e.g. optimizer state dict
        device: str
        """

        # Make sure that when the model is loaded, the torch tensors are put on the
        # device indicated in the saved metadata. External routines run on a cpu
        # machine may have moved the model from 'cuda' to 'cpu'.
        ext = os.path.splitext(model_filename)[-1]
        if ext == ".pt":
            d = torch.load(model_filename, map_location=device)
        elif ext == ".hdf5":
            d = self._load_model_from_hdf5(model_filename)
        else:
            raise ValueError("Models should be ether in .pt or .hdf5 format.")

        self.version = d.get("version")

        self.model_kwargs = d["model_kwargs"]
        update_model_config(self.model_kwargs)  # For backward compatibility
        self.initialize_network()

        self.epoch = d["epoch"]

        self.metadata = d["metadata"]

        if "context" in d:
            self.context = d["context"]

        if "event_metadata" in d:
            self.event_metadata = d["event_metadata"]

        if device != "meta":
            self.network.load_state_dict(d["model_state_dict"])

            self.network_to_device(device)

            if load_training_info:
                if "optimizer_kwargs" in d:
                    self.optimizer_kwargs = d["optimizer_kwargs"]
                if "scheduler_kwargs" in d:
                    self.scheduler_kwargs = d["scheduler_kwargs"]
                # initialize optimizer and scheduler
                self.initialize_optimizer_and_scheduler()
                # load optimizer and scheduler state dict
                if "optimizer_state_dict" in d:
                    self.optimizer.load_state_dict(d["optimizer_state_dict"])
                if "scheduler_state_dict" in d:
                    self.scheduler.load_state_dict(d["scheduler_state_dict"])
            else:
                # put model in evaluation mode
                self.network.eval()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        train_dir: str,
        train_sampler: Optional[torch.utils.data.DistributedSampler] = None,
        runtime_limits: object = None,
        checkpoint_epochs: int = None,
        use_wandb=False,
        test_only=False,
        early_stopping: Optional[EarlyStopping] = None,
        train_print_freq: int = 50,
        test_print_freq: int = 50,
        gradient_updates_per_optimizer_step: int = 1,
        automatic_mixed_precision: bool = False,
    ):
        """

        Parameters
        ----------
        train_loader
        test_loader
        train_dir
        runtime_limits
        checkpoint_epochs
        use_wandb
        test_only: bool = False
            if True, training is skipped
        early_stopping: EarlyStopping
            Optional EarlyStopping instance.

        Returns
        -------

        """

        is_primary = self.rank is None or self.rank == 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        if test_only:
            test_loss = test_epoch(
                self, test_loader, print_freq=max(1, int(test_print_freq))
            )
            if dist.is_available() and dist.is_initialized():
                t = torch.tensor(test_loss, device=self.device, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                test_loss = (t / world_size).item()
            if is_primary:
                print(f"test loss: {test_loss:.3f}")

        else:
            while not runtime_limits.limits_exceeded(self.epoch):
                self.epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)

                # Training
                lr = utils.get_lr(self.optimizer)
                if is_primary:
                    print(f"\nStart training epoch {self.epoch} with lr {lr}")
                time_start = time.time()
                train_loss = train_epoch(
                    self,
                    train_loader,
                    print_freq=max(1, int(train_print_freq)) if is_primary else 0,
                    gradient_updates_per_optimizer_step=gradient_updates_per_optimizer_step,
                    automatic_mixed_precision=automatic_mixed_precision,
                )
                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor(train_loss, device=self.device, dtype=torch.float64)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    train_loss = (t / world_size).item()
                train_time = time.time() - time_start

                if is_primary:
                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(train_time, 60)
                        )
                    )

                # Testing
                if is_primary:
                    print(f"Start testing epoch {self.epoch}")
                time_start = time.time()
                test_loss = test_epoch(
                    self,
                    test_loader,
                    print_freq=max(1, int(test_print_freq)) if is_primary else 0,
                )
                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor(test_loss, device=self.device, dtype=torch.float64)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    test_loss = (t / world_size).item()
                test_time = time.time() - time_start

                if is_primary:
                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(time.time() - time_start, 60)
                        )
                    )

                # scheduler step for learning rate
                utils.perform_scheduler_step(self.scheduler, test_loss)

                # write history and save model
                if is_primary:
                    utils.write_history(train_dir, self.epoch, train_loss, test_loss, lr)
                    utils.save_model(self, train_dir, checkpoint_epochs=checkpoint_epochs)
                if use_wandb and is_primary:
                    try:
                        import wandb

                        wandb.define_metric("epoch")
                        wandb.define_metric("*", step_metric="epoch")
                        wandb.log(
                            {
                                "epoch": self.epoch,
                                "learning_rate": lr[0],
                                "train_loss": train_loss,
                                "test_loss": test_loss,
                                "train_time": train_time,
                                "test_time": test_time,
                            }
                        )
                    except ImportError:
                        print("wandb not installed. Skipping logging to wandb.")

                if early_stopping is not None:
                    # Whether to use train or test loss
                    stop_now = False
                    if is_primary:
                        early_stopping_loss = (
                            test_loss
                            if early_stopping.metric == "validation"
                            else train_loss
                        )
                        is_best_model = early_stopping(early_stopping_loss)
                        if is_best_model:
                            self.save_model(
                                join(train_dir, "best_model.pt"), save_training_info=False
                            )
                        stop_now = bool(early_stopping.early_stop)
                    if dist.is_available() and dist.is_initialized():
                        t = torch.tensor(
                            1 if stop_now else 0, device=self.device, dtype=torch.int32
                        )
                        dist.broadcast(t, src=0)
                        stop_now = bool(t.item())
                    if stop_now:
                        if is_primary:
                            print("Early stopping")
                        break
                if is_primary:
                    print(f"Finished training epoch {self.epoch}.\n")


def train_epoch(
    pm,
    dataloader,
    print_freq: int = 1,
    gradient_updates_per_optimizer_step: int = 1,
    automatic_mixed_precision: bool = False,
):
    pm.network.train()
    loss_info = dingo.core.utils.trainutils.LossInfo(
        pm.epoch,
        len(dataloader.dataset),
        dataloader.batch_size,
        mode="Train",
        print_freq=print_freq,
    )

    def _dataset_outputs_cuda(loader):
        ds = loader.dataset
        # random_split returns Subset objects
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        return bool(getattr(ds, "output_on_cuda", False))

    def _iter_with_background_prefetch(loader):
        cuda_device = pm.device if pm.device.type == "cuda" else None

        def _next_on_device(it):
            if cuda_device is not None:
                torch.cuda.set_device(cuda_device)
            batch = next(it)
            event = None
            if cuda_device is not None:
                event = torch.cuda.Event()
                event.record()  # records on bg thread's stream (after DLPack sync)
            return batch, event

        iterator = iter(loader)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_next_on_device, iterator)
            while True:
                try:
                    batch, event = future.result()
                except StopIteration:
                    break
                future = executor.submit(_next_on_device, iterator)
                yield batch, event

    def _plain_iter_with_events(loader):
        for batch in loader:
            yield batch, None

    def _to_device_if_needed(x):
        if isinstance(x, torch.Tensor):
            if x.device == pm.device:
                return x
            return x.to(pm.device, non_blocking=True)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(pm.device, non_blocking=True)
        return torch.as_tensor(x, device=pm.device)

    use_cuda_batch_prefetch = (
        bool(getattr(pm, "cuda_batch_prefetch", False))
        and pm.device.type == "cuda"
        and _dataset_outputs_cuda(dataloader)
    )
    data_iter = (
        _iter_with_background_prefetch(dataloader)
        if use_cuda_batch_prefetch
        else _plain_iter_with_events(dataloader)
    )

    _train_stream = None  # use default stream; priority isolation hurt BBHx more than it helped

    scaler = GradScaler("cuda") if automatic_mixed_precision else None

    for batch_idx, (data, prefetch_event) in enumerate(data_iter):
        loss_info.update_timer()
        with (torch.cuda.stream(_train_stream) if _train_stream else contextlib.nullcontext()):
            if prefetch_event is not None:
                torch.cuda.current_stream().wait_event(prefetch_event)
            if batch_idx % gradient_updates_per_optimizer_step == 0:
                pm.optimizer.zero_grad(set_to_none=True)
            # data to device
            data = [_to_device_if_needed(d) for d in data]
            # compute loss
            if automatic_mixed_precision:
                with autocast("cuda"):
                    loss = pm.loss(data[0], *data[1:])
                scaler.scale(loss).backward()
            else:
                loss = pm.loss(data[0], *data[1:])
                loss.backward()
            # optimizer step every N batches
            if (batch_idx + 1) % gradient_updates_per_optimizer_step == 0:
                if automatic_mixed_precision:
                    scaler.step(pm.optimizer)
                    scaler.update()
                else:
                    pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch(pm, dataloader, print_freq: int = 1):
    with torch.no_grad():
        pm.network.eval()
        loss_info = dingo.core.utils.trainutils.LossInfo(
            pm.epoch,
            len(dataloader.dataset),
            dataloader.batch_size,
            mode="Test",
            print_freq=print_freq,
        )

        def _dataset_outputs_cuda(loader):
            ds = loader.dataset
            while hasattr(ds, "dataset"):
                ds = ds.dataset
            return bool(getattr(ds, "output_on_cuda", False))

        def _iter_with_background_prefetch(loader):
            cuda_device = pm.device if pm.device.type == "cuda" else None

            def _next_on_device(it):
                if cuda_device is not None:
                    torch.cuda.set_device(cuda_device)
                batch = next(it)
                event = None
                if cuda_device is not None:
                    event = torch.cuda.Event()
                    event.record()  # records on bg thread's stream (after DLPack sync)
                return batch, event

            iterator = iter(loader)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_next_on_device, iterator)
                while True:
                    try:
                        batch, event = future.result()
                    except StopIteration:
                        break
                    future = executor.submit(_next_on_device, iterator)
                    yield batch, event

        def _plain_iter_with_events(loader):
            for batch in loader:
                yield batch, None

        def _to_device_if_needed(x):
            if isinstance(x, torch.Tensor):
                if x.device == pm.device:
                    return x
                return x.to(pm.device, non_blocking=True)
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(pm.device, non_blocking=True)
            return torch.as_tensor(x, device=pm.device)

        use_cuda_batch_prefetch = (
            bool(getattr(pm, "cuda_batch_prefetch", False))
            and pm.device.type == "cuda"
            and _dataset_outputs_cuda(dataloader)
        )
        data_iter = (
            _iter_with_background_prefetch(dataloader)
            if use_cuda_batch_prefetch
            else _plain_iter_with_events(dataloader)
        )

        _train_stream = None  # use default stream; priority isolation hurt BBHx more than it helped

        for batch_idx, (data, prefetch_event) in enumerate(data_iter):
            loss_info.update_timer()
            with (torch.cuda.stream(_train_stream) if _train_stream else contextlib.nullcontext()):
                if prefetch_event is not None:
                    torch.cuda.current_stream().wait_event(prefetch_event)
                # data to device
                data = [_to_device_if_needed(d) for d in data]
                # compute loss
                loss = pm.loss(data[0], *data[1:])
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()
