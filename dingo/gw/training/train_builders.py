from typing import List, Optional
import copy

import torch.multiprocessing
import torchvision
from threadpoolctl import threadpool_limits
from bilby.gw.detector import InterferometerList

from dingo.gw.SVD import SVDBasis

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.transforms import (
    ProjectOntoDetectors,
    ProjectOntoSpaceDetectors,
    GenerateBBHxDirectResponse,
    SampleNoiseASD,
    WhitenAndScaleStrain,
    AddWhiteNoiseComplex,
    SelectStandardizeRepackageParameters,
    RepackageStrainsAndASDS,
    UnpackDict,
    GNPECoalescenceTimes,
    SampleExtrinsicParameters,
    GetDetectorTimes,
)
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.prior import default_inference_parameters
from dingo.gw.gwutils import *
from dingo.core.utils import *


def build_dataset(
    data_settings: dict,
    leave_waveforms_on_disk: Optional[bool] = False,
    on_fly: Optional[bool] = False,
) -> WaveformDataset:
    """Build a dataset based on a settings dictionary. This should contain the path of
    a saved waveform dataset.

    This function also truncates the dataset as necessary.

    Parameters
    ----------
    data_settings : dict
    leave_waveforms_on_disk: bool
        If provided, the values associated with the waveforms will not be loaded into memory during initialization.
        Instead, they will be loaded from disk when the dataset is accessed. This is useful for reducing the memory
        load of large datasets, but can slow down data preprocessing.
    on_fly: bool
        if provided, will generate polarizations on the fly when calling get_item from the waveform dataset

    Returns
    -------
    WaveformDataset
    """

    # Build and truncate datasets
    domain_update = data_settings.get("domain_update", None)
    wfd = WaveformDataset(
        file_name=data_settings["waveform_dataset_path"],
        precision="double",
        domain_update=domain_update,
        svd_size_update=data_settings.get("svd_size_update"),
        leave_waveforms_on_disk=leave_waveforms_on_disk,
        on_fly=on_fly,
    )
    return wfd


def set_train_transforms(
    wfd,
    data_settings,
    asd_dataset_path,
    omit_transforms=None,
    print_output=True,
):
    """
    Set the transform attribute of a waveform dataset based on a settings dictionary.
    The transform takes waveform polarizations, samples random extrinsic parameters,
    projects to detectors, adds noise, and formats the data for input to the neural
    network. It also implements optional GNPE transformations.

    Note that the WaveformDataset is modified in-place, so this function returns nothing.

    Parameters
    ----------
    wfd : WaveformDataset
    data_settings : dict
    asd_dataset_path : str
        Path corresponding to the ASD dataset used to generate noise.
    omit_transforms :
        List of sub-transforms to omit from the full composition.
    """

    if print_output:
        print(f"Setting train transforms.")
        if omit_transforms is not None:
            print("Omitting \n\t" + "\n\t".join([t.__name__ for t in omit_transforms]))

    # By passing the wfd domain when instantiating the noise dataset, this ensures the
    # domains will match. In particular, it truncates the ASD dataset beyond the new
    # f_max, and sets it to 1 below f_min.
    asd_dataset = ASDDataset(
        asd_dataset_path,
        ifos=data_settings["detectors"],
        precision="double",
        domain_update=wfd.domain.domain_dict,
    )
    assert wfd.domain == asd_dataset.domain

    # Add window factor to domain, so that we can compute the noise variance.
    # TODO: we want to set `domain = wfd.domain`. This does not work at the moment,
    #  because this requires updating the window factor of the wfd.domain (instead of
    #  just the local domain object). This causes trouble if the
    #  set_train_transforms function is called multiple times, since the second time
    #  the domain_update = wfd.domain.domain_dict contains a window factor, which will
    #  cause an error in domain_update.
    domain = build_domain(wfd.domain.domain_dict)
    domain.window_factor = get_window_factor(data_settings["window"])

    extrinsic_prior_dict = get_extrinsic_prior_dict(data_settings["extrinsic_prior"])
    if data_settings["inference_parameters"] == "default":
        data_settings["inference_parameters"] = default_inference_parameters

    ref_time = data_settings["ref_time"]
    # Build detector objects

    #Needs to be a check later
    if data_settings["detector_type"] != "LISA":
        ifo_list = InterferometerList(data_settings["detectors"])
        transforms = [
        SampleExtrinsicParameters(extrinsic_prior_dict),
        GetDetectorTimes(ifo_list, ref_time),
    ]
    else:
         transforms = [SampleExtrinsicParameters(extrinsic_prior_dict)]


    # Build transforms.


    

    extra_context_parameters = []
    if "gnpe_time_shifts" in data_settings:
        if data_settings["detector_type"] == "LISA":
            raise NotImplementedError(
                "gnpe_time_shifts is currently only supported for ground-based "
                "detectors, not detector_type='LISA'."
            )
        d = data_settings["gnpe_time_shifts"]
        transforms.append(
            GNPECoalescenceTimes(
                ifo_list,
                d["kernel"],
                d["exact_equiv"],
                inference=False,
            )
        )
        extra_context_parameters += transforms[-1].context_parameters

    # Add the GNPE context to context_parameters the first time the transforms are
    # constructed. We do not want to overwrite the ordering of the parameters in
    # subsequent runs.
    if "context_parameters" not in data_settings:
        data_settings["context_parameters"] = []
    for p in extra_context_parameters:
        if p not in data_settings["context_parameters"]:
            data_settings["context_parameters"].append(p)

    # If the standardization factors have already been set, use those. Otherwise,
    # calculate them, and save them within the data settings.
    #
    # Standardizations are calculated at this point because the present set of
    # transforms is sufficient for generating samples of all regression and context
    # parameters.
    try:
        standardization_dict = data_settings["standardization"]
        if print_output:
            print("Using previously-calculated parameter standardizations.")
    except KeyError:
        if print_output:
            print("Calculating new parameter standardizations.")
        standardization_dict = get_standardization_dict(
            extrinsic_prior_dict,
            wfd,
            data_settings["inference_parameters"] + data_settings["context_parameters"],
            torchvision.transforms.Compose(transforms),
        )
        data_settings["standardization"] = standardization_dict
    waveform_generator_settings = wfd.settings.get("waveform_generator", {})
    bbhx_direct_response = (
        waveform_generator_settings.get("BBHx", False)
        and waveform_generator_settings.get("direct_response", False)
    )
    bbhx_gpu_fastpath = (
        bbhx_direct_response
        and waveform_generator_settings.get("gpu_fastpath", False)
        and hasattr(wfd, "waveform_generator")
        and getattr(wfd.waveform_generator, "use_gpu", False)
    )
    bbhx_backend_native_fused = (
        bbhx_gpu_fastpath
        and waveform_generator_settings.get("backend_native_fused", False)
    )
    bbhx_timing_profile = bool(
        waveform_generator_settings.get("timing_profile", False)
    )
    bbhx_timing_profile_print_every = int(
        waveform_generator_settings.get("timing_profile_print_every", 0)
    )
    # SVD initialization intentionally omits downstream formatting/noise transforms
    # and expects CPU numpy arrays. Keep GPU fast-path for full train/test transforms.
    use_gpu_fastpath_for_this_transform = bbhx_gpu_fastpath and omit_transforms is None
    use_backend_native_fused_for_this_transform = (
        bbhx_backend_native_fused and omit_transforms is None
    )
    if data_settings["detector_type"] == "LISA":
        if bbhx_direct_response:
            if not hasattr(wfd, "waveform_generator"):
                raise RuntimeError(
                    "BBHx direct-response training requires on-the-fly waveform "
                    "generation. Set local.on_fly=True."
                )
            transforms.append(
                GenerateBBHxDirectResponse(
                    wfd.waveform_generator,
                    data_settings["detectors"],
                    gpu_fastpath=use_gpu_fastpath_for_this_transform,
                    backend_native_fused=use_backend_native_fused_for_this_transform,
                    timing_profile=bbhx_timing_profile,
                    timing_profile_print_every=bbhx_timing_profile_print_every,
                )
            )
        else:
            transforms.append(ProjectOntoSpaceDetectors("TDIAET",domain, ref_time,data_settings["detectors"],data_settings["lisa_settings"])) #Hard Coded need to change
    else:
        transforms.append(ProjectOntoDetectors(ifo_list, domain, ref_time))
    transforms.append(
        SampleNoiseASD(
            asd_dataset,
            gpu_cache=use_gpu_fastpath_for_this_transform,
        )
    )
    transforms.append(WhitenAndScaleStrain(domain.noise_std))
    # We typically add white detector noise. For debugging purposes, this can be turned
    # off with zero_noise option in data_settings.
    if not data_settings.get("zero_noise", False):
        transforms.append(AddWhiteNoiseComplex())
    transforms.append(
        SelectStandardizeRepackageParameters(
            {
                k: data_settings[k]
                for k in ["inference_parameters", "context_parameters"]
            },
            standardization_dict,
        )
    )
    transforms.append(
        RepackageStrainsAndASDS(data_settings["detectors"], first_index=domain.min_idx)
    )

    if data_settings["context_parameters"]:
        selected_keys = ["inference_parameters", "waveform", "context_parameters"]
    else:
        selected_keys = ["inference_parameters", "waveform"]

    transforms.append(UnpackDict(selected_keys=selected_keys))

    # Drop transforms that are not desired. This is useful for generating, e.g.,
    # noise-free data, or for producing data not formatted for input to the network.
    if omit_transforms is not None:
        transforms = [t for t in transforms if type(t) not in omit_transforms]

    wfd.transform = torchvision.transforms.Compose(transforms)
    # Used by DataLoader builder to disable pin_memory and worker multiprocessing
    # when dataset items are already CUDA tensors.
    wfd.output_on_cuda = bool(use_gpu_fastpath_for_this_transform)
    # When transforms already produce full batched tensors, keep them batched and
    # bypass split/re-collate in WaveformDataset/DataLoader.
    wfd.returns_batched_output = bool(use_gpu_fastpath_for_this_transform)


def build_svd_for_embedding_network(
    wfd: WaveformDataset,
    data_settings: dict,
    asd_dataset_path: str,
    size: int,
    num_training_samples: int,
    num_validation_samples: int,
    num_workers: int = 0,
    batch_size: int = 1000,
    method: str = "svds",
    out_dir: Optional[str] = None,
) -> List:
    """
    Construct SVD matrices V based on clean waveforms in each interferometer. These
    will be used to seed the weights of the initial projection part of the embedding
    network.

    It first generates a number of training waveforms, and then produces the SVD.

    Parameters
    ----------
    wfd : WaveformDataset
    data_settings : dict
    asd_dataset_path : str
        Training waveforms will be whitened with respect to these ASDs.
    size : int
        Number of basis elements to include in the SVD projection.
    num_training_samples : int
    num_validation_samples : int
    num_workers : int
    batch_size : int
    method : str
        SVD backend passed to SVDBasis.generate_basis.
    out_dir : str
        SVD performance diagnostics are saved here.

    Returns
    -------
    list of numpy arrays
        The V matrices for each interferometer. They are ordered as in data_settings[
        'detectors'].
    """
    # Building the transforms can alter the data_settings dictionary. We do not want
    # the construction of the SVD to impact this, so begin with a fresh copy of this
    # dictionary.
    data_settings = copy.deepcopy(data_settings)

    # This is needed to prevent an occasional error when loading a large dataset into
    # memory using a dataloader. This removes a limitation on the number of "open files".
    old_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Fix the luminosity distance to a standard value, just in order to generate the SVD.
    data_settings["extrinsic_prior"]["luminosity_distance"] = "100.0"

    # Build the dataset, but with certain transforms omitted. In particular, we want to
    # build the SVD based on zero-noise waveforms. They should still be whitened though.
    set_train_transforms(
        wfd,
        data_settings,
        asd_dataset_path,
        omit_transforms=[
            AddWhiteNoiseComplex,
            RepackageStrainsAndASDS,
            SelectStandardizeRepackageParameters,
            UnpackDict,
        ],
    )

    print("Generating waveforms for embedding network SVD initialization.")
    time_start = time.time()
    ifos = list(wfd[0]["waveform"].keys())
    waveform_len = len(wfd[0]["waveform"][ifos[0]])
    num_waveforms = num_training_samples + num_validation_samples
    if num_waveforms > len(wfd):
        raise IndexError(
            f"Requested {num_waveforms} samples for generating SVD for embedding "
            f"network, but waveform dataset only contains {len(wfd)} samples."
        )
    waveforms = {
        ifo: np.empty((num_waveforms, waveform_len), dtype=np.complex128)
        for ifo in ifos
    }
    parameters = pd.DataFrame()

    loader = DataLoader(
        wfd,
        batch_size=batch_size,
        num_workers=0,
        worker_init_fn=fix_random_seeds,
    )
    with threadpool_limits(limits=1, user_api="blas"):
        for idx, data in enumerate(loader):
            # This is for handling the last batch, which may otherwise push the total
            # number of samples above the number requested.
            lower = idx * batch_size
            n = min(batch_size, num_waveforms - lower)

            parameters = pd.concat(
                [parameters, pd.DataFrame(data["parameters"]).iloc[:n]],
                ignore_index=True,
            )
            strain_data = data["waveform"]
            for ifo, strains in strain_data.items():
                waveforms[ifo][lower : lower + n] = strains[:n]
            if lower + n == num_waveforms:
                break
    print(f"...done. This took {time.time() - time_start:.0f} s.")

    # Reset the standard sharing strategy.
    torch.multiprocessing.set_sharing_strategy(old_sharing_strategy)

    print("Generating SVD basis for ifo:")
    time_start = time.time()
    basis_dict = {}
    print(f"...using SVD method: {method}")
    for ifo in ifos:
        basis = SVDBasis()
        basis.generate_basis(waveforms[ifo][:num_training_samples], size, method)
        basis_dict[ifo] = basis
        print(f"...{ifo} done.")
    print(f"...this took {time.time() - time_start:.0f} s.")

    if out_dir is not None:
        print(f"Testing SVD basis matrices.")
        for ifo, basis in basis_dict.items():
            print(f"...{ifo}:")
            basis.compute_test_mismatches(
                waveforms[ifo][num_training_samples:],
                parameters=parameters.iloc[num_training_samples:].reset_index(
                    drop=True
                ),
                verbose=True,
            )
            basis.to_file(os.path.join(out_dir, f"svd_{ifo}.hdf5"))
    print("Done")

    # Return V matrices in standard order. Drop the elements below domain.min_idx,
    # since the neural network expects data truncated below these. The dropped elements
    # should be 0.
    waveform_settings = {}
    if isinstance(getattr(wfd, "settings", None), dict):
        waveform_settings = wfd.settings.get("waveform_generator", {}) or {}
    lisa_like = bool(
        waveform_settings.get("LISA", False) or waveform_settings.get("BBHx", False)
    )
    svd_zero_tol = 1e-6 if lisa_like else 1e-12

    print(f"Truncating SVD matrices below index {wfd.domain.min_idx}.")
    print("...V matrix shapes:")
    V_rb_list = []
    for ifo in data_settings["detectors"]:
        V = basis_dict[ifo].V
        V_prefix = V[: wfd.domain.min_idx]
        if V_prefix.size and not np.allclose(
            V_prefix, 0, atol=svd_zero_tol, rtol=0.0
        ):
            max_abs = float(np.max(np.abs(V_prefix)))
            if lisa_like:
                print(
                    f"Warning: {ifo} SVD rows below min_idx are not exactly zero "
                    f"(max abs={max_abs:.3e}). Zeroing before truncation."
                )
                V = V.copy()
                V[: wfd.domain.min_idx] = 0
            else:
                raise AssertionError(
                    f"SVD rows below min_idx must be zero for {ifo}; "
                    f"max abs={max_abs:.3e}."
                )
        V = V[wfd.domain.min_idx :]
        print("      " + str(V.shape))
        V_rb_list.append(V)
    print("\n")
    return V_rb_list
