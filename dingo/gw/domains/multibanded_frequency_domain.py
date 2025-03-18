from typing import Iterable, Union, Optional
import numpy as np
import torch
import lal
from copy import copy

from .base_frequency_domain import BaseFrequencyDomain
from .uniform_frequency_domain import UniformFrequencyDomain


class MultibandedFrequencyDomain(BaseFrequencyDomain):
    r"""
    Defines a non-uniform frequency domain that is made up of a sequence of
    uniform-frequency domain bands. Each subsequent band in the sequence has double the
    bin-width of the previous one, i.e., delta_f is doubled each band as one moves up
    the bands. This is intended to allow for efficient representation of gravitational
    waveforms, which generally have slower oscillations at higher frequencies. Indeed,
    the leading order chirp has phase evolution [see
    https://doi.org/10.1103/PhysRevD.49.2658],
    $$
    \Psi(f) = \frac{3}{4}(8 \pi \mathcal{M} f)^{-5/3},
    $$
    hence a coarser grid can be used at higher f.

    The domain is partitioned into bands via a sequence of nodes that are specified at
    initialization.

    In comparison to the UniformFrequencyDomain, the MultibandedFrequencyDomain has the
    following key differences:

    * The sample frequencies start at the first node, rather than f = 0.0 Hz.

    * Quantities such as delta_f, noise_std, etc., are represented as arrays rather than
    scalars, as they vary depending on f.

    The MultibandedFrequencyDomain furthermore has an attribute base_domain,
    which holds an underlying UniformFrequencyDomain object. The decimate() method
    decimates data in the base_domain to the multi-banded domain.
    """

    def __init__(
        self,
        nodes: Iterable[float],
        delta_f_initial: float,
        base_domain: Union[UniformFrequencyDomain, dict],
    ):
        """
        Parameters
        ----------
        nodes: Iterable[float]
            Defines the partitioning of the underlying frequency domain into bands. In
            total, there are len(nodes) - 1 frequency bands. Band j consists of
            decimated data from the base domain in the range [nodes[j]:nodes[j+1]).
        delta_f_initial: float
            delta_f of band 0. The decimation factor doubles between adjacent bands,
            so delta_f is doubled as well.
        base_domain: Union[UniformFrequencyDomain, dict]
            Original (uniform frequency) domain of data, which is the starting point
            for the decimation. This determines the decimation details and the noise_std.
            Either provided as dict for build_domain, or as domain_object.
        """
        super().__init__()
        if isinstance(base_domain, dict):
            from .build_domain import build_domain

            base_domain = build_domain(base_domain)

        self.nodes = np.array(nodes, dtype=np.float32)
        self.base_domain = base_domain
        self._initialize_bands(delta_f_initial)
        if not isinstance(self.base_domain, UniformFrequencyDomain):
            raise ValueError(
                f"Expected domain type UniformFrequencyDomain, got {type(base_domain)}."
            )
        # truncation indices for domain update
        self._range_update_idx_lower = None
        self._range_update_idx_upper = None
        self._range_update_initial_length = None

    def _initialize_bands(self, delta_f_initial: float):
        if len(self.nodes.shape) != 1:
            raise ValueError(
                f"Expected format [num_bands + 1] for nodes, "
                f"got {self.nodes.shape}."
            )
        self.num_bands = len(self.nodes) - 1
        self._nodes_indices = (self.nodes / self.base_domain.delta_f).astype(int)

        self._delta_f_bands = (
            delta_f_initial * (2 ** np.arange(self.num_bands))
        ).astype(np.float32)
        self._decimation_factors_bands = (
            self._delta_f_bands / self.base_domain.delta_f
        ).astype(int)
        self._num_bins_bands = (
            (self._nodes_indices[1:] - self._nodes_indices[:-1])
            / self._decimation_factors_bands
        ).astype(int)

        self._band_assignment = np.concatenate(
            [
                np.ones(num_bins_band, dtype=int) * idx
                for idx, num_bins_band in enumerate(self._num_bins_bands)
            ]
        )
        self._delta_f = self._delta_f_bands[self._band_assignment]

        # For each bin, [self._f_base_lower, self._f_base_upper] describes the
        # frequency range in the base domain which is used for truncation.
        self._f_base_lower = np.concatenate(
            (self.nodes[:1], self.nodes[0] + np.cumsum(self._delta_f[:-1]))
        )
        self._f_base_upper = (
            self.nodes[0] + np.cumsum(self._delta_f) - self.base_domain.delta_f
        )

        # Set sample frequencies as mean of decimation range.
        self._sample_frequencies = (self._f_base_upper + self._f_base_lower) / 2
        self._sample_frequencies_torch = None
        self._sample_frequencies_torch_cuda = None
        # sample_frequencies should always be the decimation of the base domain
        # frequencies.

        if self.f_min not in self.base_domain() or self.f_max not in self.base_domain():
            raise ValueError(
                f"Endpoints ({self.f_min}, {self.f_max}) not in base "
                f"domain, {self.base_domain.domain_dict}"
            )

        # Update base domain to required range.
        self.base_domain.update({"f_min": self.f_min, "f_max": self.f_max})

    def decimate(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Decimate data from the base_domain to the multi-banded domain.

        Parameters
        ----------
        data : array-like (np.ndarray or torch.Tensor)
            Decimation is done along the trailing dimension of this array. This
            dimension should therefore be compatible with the base frequency domain,
            i.e., running from 0.0 Hz or f_min up to f_max, with uniform delta_f.

        Returns
        -------
        Decimated array of the same type as the input.
        """
        if data.shape[-1] == len(self.base_domain):
            offset_idx = 0
        elif data.shape[-1] == len(self.base_domain) - self.base_domain.min_idx:
            offset_idx = -self.base_domain.min_idx
        else:
            raise ValueError(
                f"Provided data has {data.shape[-1]} bins, which is incompatible with "
                f"the expected domain of length {len(self.base_domain)}"
            )
        if isinstance(data, np.ndarray):
            data_decimated = np.empty((*data.shape[:-1], len(self)), dtype=data.dtype)
        elif isinstance(data, torch.Tensor):
            data_decimated = torch.empty(
                (*data.shape[:-1], len(self)), dtype=data.dtype
            )
        else:
            raise NotImplementedError(
                f"Decimation not implemented for data of type {data}."
            )

        lower_out = 0  # running index for decimated band data
        for idx_band in range(self.num_bands):
            lower_in = self._nodes_indices[idx_band] + offset_idx
            upper_in = self._nodes_indices[idx_band + 1] + offset_idx
            decimation_factor = self._decimation_factors_bands[idx_band]
            num_bins = self._num_bins_bands[idx_band]

            data_decimated[..., lower_out : lower_out + num_bins] = decimate_uniform(
                data[..., lower_in:upper_in], decimation_factor
            )
            lower_out += num_bins

        assert lower_out == len(self)

        return data_decimated

    def update(self, new_settings: dict):
        """
        Update the domain by truncating the frequency range (by specifying new f_min,
        f_max).

        After calling this function, data from the original domain can be truncated to
        the new domain using self.update_data(). For simplicity, we do not allow for
        multiple updates of the domain.

        Parameters
        ----------
        new_settings : dict
            Settings dictionary. Keys must either be the keys contained in domain_dict, or
            a subset of ["f_min", "f_max"].
        """
        if set(new_settings.keys()).issubset(["f_min", "f_max"]):
            self._set_new_range(**new_settings)
        elif set(new_settings.keys()) == self.domain_dict.keys():
            if new_settings == self.domain_dict:
                return
            self._set_new_range(
                f_min=new_settings["base_domain"]["f_min"],
                f_max=new_settings["base_domain"]["f_max"],
            )
            if self.domain_dict != new_settings:
                raise ValueError(
                    f"Update settings {new_settings} are incompatible with "
                    f"domain settings {self.domain_dict}."
                )
        else:
            raise ValueError(
                f"Invalid argument for domain update {new_settings}. Must either be "
                f'{list(self.domain_dict.keys())} or a subset of ["f_min, f_max"].'
            )

    def _set_new_range(
        self, f_min: Optional[float] = None, f_max: Optional[float] = None
    ):
        """
        Set a new range [f_min, f_max] for the domain. This operation is only allowed
        if the new range is contained within the old one.

        Note: f_min, f_max correspond to the range in the *base_domain*.

        Parameters
        ----------
        f_min : float
            New minimum frequency (optional).
        f_max : float
            New maximum frequency (optional).
        """
        if f_min is None and f_max is None:
            return
        if self._range_update_initial_length is not None:
            raise ValueError(f"Can't update domain of type {type(self)} a second time.")
        if f_min is not None and f_max is not None and f_min >= f_max:
            raise ValueError("f_min must not be larger than f_max.")

        lower_bin, upper_bin = 0, len(self) - 1

        if f_min is not None:
            if self._f_base_lower[0] <= f_min <= self._f_base_lower[-1]:
                # find new starting bin (first element with f >= f_min)
                lower_bin = np.where(self._f_base_lower >= f_min)[0][0]
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_base_lower[-1], self._f_base_lower[-1]}]."
                )

        if f_max is not None:
            if self._f_base_upper[0] <= f_max <= self._f_base_upper[-1]:
                # find new final bin (last element where f <= f_max)
                upper_bin = np.where(self._f_base_upper <= f_max)[0][-1]
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_base_lower[-1], self._f_base_lower[-1]}]."
                )

        lower_band = self._band_assignment[lower_bin]
        upper_band = self._band_assignment[upper_bin]
        # new nodes extend to upper_band + 2: we have +1 from the exclusive end index
        # and +1, as we have num_bands + 1 elements in nodes
        nodes_new = copy(self.nodes)[lower_band : upper_band + 2]
        nodes_new[0] = self._f_base_lower[lower_bin]
        nodes_new[-1] = self._f_base_upper[upper_bin] + self.base_domain.delta_f

        self._range_update_initial_length = len(self)
        self._range_update_idx_lower = lower_bin
        self._range_update_idx_upper = upper_bin

        self.nodes = nodes_new
        self._initialize_bands(self._delta_f_bands[lower_band])
        assert self._range_update_idx_upper - self._range_update_idx_lower + 1 == len(
            self
        )

    def update_data(
        self, data: np.ndarray | torch.Tensor, axis: int = -1, **kwargs
    ) -> np.ndarray | torch.Tensor:
        """
        Truncates the data array to be compatible with the domain. This is used when
        changing f_min or f_max.

        update_data() will only have an effect after updating the domain to have a new
        frequency range using self.update().

        Parameters
        ----------
        data : array-like (np.ndarray or torch.Tensor)
            Array should be compatible with either the original or updated
            MultibandedFrequencyDomain along the specified axis. In the latter
            case, nothing is done. In the former, data are truncated appropriately.
        axis: int
            Axis along which to operate.

        Returns
        -------
        Updated data of the same type as input.
        """
        if data.shape[axis] == len(self):
            return data
        elif (
            self._range_update_initial_length is not None
            and data.shape[axis] == self._range_update_initial_length
        ):
            sl = [slice(None)] * data.ndim
            # First truncate beyond f_max.
            sl[axis] = slice(
                self._range_update_idx_lower, self._range_update_idx_upper + 1
            )
            data = data[tuple(sl)]
            return data
        else:
            raise ValueError(
                f"Data (shape {data.shape}) incompatible with the domain "
                f"(length {len(self)})."
            )

    @property
    def sample_frequencies(self) -> np.ndarray:
        return self._sample_frequencies

    @property
    def frequency_mask(self) -> np.ndarray:
        """Array of len(self) consisting of ones.

        As the MultibandedFrequencyDomain starts from f_min, no masking is generally
        required."""
        return np.ones_like(self.sample_frequencies)

    @property
    def frequency_mask_length(self) -> int:
        return len(self.frequency_mask)

    @property
    def min_idx(self):
        return 0

    @property
    def max_idx(self):
        return len(self) - 1

    @property
    def window_factor(self):
        return self.base_domain.window_factor

    @window_factor.setter
    def window_factor(self, value):
        """Set window factor of base domain."""
        self.base_domain.window_factor = float(value)

    @property
    def f_max(self) -> float:
        return self._f_base_upper[-1]

    @property
    def f_min(self) -> float:
        return self._f_base_lower[0]

    @property
    def duration(self) -> float:
        raise NotImplementedError()

    @property
    def sampling_rate(self) -> float:
        raise NotImplementedError()

    @property
    def domain_dict(self) -> dict:
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        # Call tolist() on self.bands, such that it can be saved as str for metadata.
        return {
            "type": "MultibandedFrequencyDomain",
            "nodes": self.nodes.tolist(),
            "delta_f_initial": self._delta_f_bands[0].item(),
            "base_domain": self.base_domain.domain_dict,
        }


######################
### util functions ###
######################


def get_decimation_bands_adaptive(
    base_domain: UniformFrequencyDomain,
    waveforms: np.ndarray,
    min_num_bins_per_period: int = 16,
    delta_f_max: float = np.inf,
):
    """
    Get frequency bands for decimation, which can be used to initialize a
    MultibandedFrequencyDomain object. This is based on the waveforms array. First,
    the oscillation periods are extracted from the waveforms. Next, frequency bands are
    set up such that each oscillation is captured by at least min_num_bins_per_period
    bins. The decimation factor increases by a factor of 2 between consecutive bands.

    Parameters
    ----------
    base_domain: UniformFrequencyDomain
        Original uniform frequency domain of the data to be decimated.
    waveforms: np.ndarray
        2D array with complex waveforms in the original uniform frequency domain. Used
        to determine the required resolution, and thereby the boundaries of the bands.
    min_num_bins_per_period: int = 8
        Minimum number of bins per oscillation period.
        Note: a period here describes the interval between two consecutive zero
        crossings, so it differs by a factor of 2 from the usual convention.
    delta_f_max: float = np.inf
        Maximum delta_f of the bands.

    Returns
    -------
    bands: list
        List of frequency bands. Can be used to initialize MultibandedFrequencyDomain.

    """
    if len(waveforms.shape) != 2 or waveforms.shape[-1] != len(base_domain):
        raise ValueError(
            f"Waveform array has shape {waveforms.shape}, expected, "
            f"(N, {len(base_domain)}): "
            f"N waveforms, each of the same length {len(base_domain)} as domain."
        )

    # For some reason, the last bin of a waveform is always zero, so we need to get rid
    # of that for the step below.
    x = waveforms[:, base_domain.min_idx : -1]

    # Ideally, we would just call
    #   periods = np.min(get_periods(x, upper_bound_for_monotonicity=True), axis=0)
    # here. However, get_periods does not work perfectly on phase-heterodyned BNS
    # waveforms. The reason is that get_periods assumes waveforms that oscillate
    # symmetrically around the origin. However, in practice there are some waveforms
    # for which the oscillation of the real part is not symmetric in some segment of the
    # frequency axis. Instead of an oscillation in the range (-a, a) [with a being the
    # local amplitude], we sometimes encounter oscillations in range (-eps, a) with
    # a >> eps > 0. I suspect that this behaviour is an artefact of the phase
    # heterodyning. Since get_periods infers the periods based on the zero crossings,
    # it will infer a very small period for that frequency segment. In these rare cases,
    # the inferred period is not a good approximation of the number of bins required to
    # capture the oscillation, as the rate of change of the signal is much smaller than
    # what the period suggests. So below, we remove these cases by using
    # np.percentile(_, 1) instead of np.min(_).
    periods = get_periods(x.real, upper_bound_for_monotonicity=False)
    # periods = get_period_for_complex_oscillation(x, upper_bound_for_monotonicity=False)
    periods = np.percentile(periods, 1, axis=0)
    periods = np.minimum.accumulate(periods[::-1])[::-1]

    max_dec_factor_array = periods / min_num_bins_per_period
    initial_downsampling, band_nodes_indices = get_band_nodes_for_adaptive_decimation(
        max_dec_factor_array,
        max_dec_factor_global=int(delta_f_max / base_domain.delta_f),
    )

    # transform downsampling factor and band nodes from indices to frequencies
    delta_f_initial = base_domain.delta_f * initial_downsampling
    nodes = base_domain()[np.array(band_nodes_indices) + base_domain.min_idx]

    return nodes, delta_f_initial


def get_decimation_bands_from_chirp_mass(
    base_domain: UniformFrequencyDomain,
    chirp_mass_min: float,
    alpha: int = 1,
    delta_f_max: float = np.inf,
):
    """
    Get frequency bands for decimation, which can be used to initialize a
    MultibandedFrequencyDomain object. This is based on the minimal chirp mass,
    which to leading order determines the required frequency resolution in each
    frequency band.

    Parameters
    ----------
    base_domain: UniformFrequencyDomain
        Original uniform frequency domain of the data to be decimated.
    chirp_mass_min: float
        Minimum chirp mass. Smaller chirp masses require larger resolution.
    alpha: int
        Factor by which to decrease the resolution. Needs to be a power of 2.
        The resolution can for instance be decreased when using heterodyning.
    delta_f_max: float = np.inf
        Maximum delta_f of the bands.

    Returns
    -------
    bands: list
        List of frequency bands. Can be used to initialize MultibandedFrequencyDomain.
    """
    if not is_power_of_2(1 / base_domain.delta_f):
        raise NotImplementedError(
            f"Decimation only implemented for domains with delta_f = 1 / k**2, "
            f"got {base_domain.delta_f}."
        )
    if not is_power_of_2(alpha):
        raise NotImplementedError(f"Alpha needs to be a power of 2, got {alpha}.")

    # delta_f and f_min for first band, derived from base_domain and chirp_mass_min
    delta_f_band = alpha / ceil_to_power_of_2(
        duration_LO(chirp_mass_min, base_domain.f_min)
    )
    # delta_f can't be smaller than base_domain.delta_f
    delta_f_band = max(delta_f_band, base_domain.delta_f)
    f = base_domain.f_min - base_domain.delta_f / 2.0 + delta_f_band / 2.0
    bands = []

    while f + delta_f_band / 2 < base_domain.f_max:
        f_min_band = f
        while is_within_band(
            f + delta_f_band,
            chirp_mass_min,
            delta_f_band,
            base_domain.f_max,
            alpha,
            delta_f_max,
        ):
            f += delta_f_band
        f_max_band = f
        bands.append([f_min_band, f_max_band, delta_f_band])

        delta_f_band *= 2
        f += (delta_f_band / 2 + delta_f_band) / 2

    return bands


def decimate_uniform(data, decimation_factor: int):
    """
    Reduce dimension of data by decimation_factor along last axis, by uniformly
    averaging sets of decimation_factor neighbouring bins.

    Parameters
    ----------
    data
        Array or tensor to be decimated.
    decimation_factor
        Factor by how much to compress. Needs to divide data.shape[-1].
    Returns
    -------
    data_decimated
        Uniformly decimated data, as array or tensor.
        Shape (*data.shape[:-1], data.shape[-1]/decimation_factor).
    """
    if data.shape[-1] % decimation_factor != 0:
        raise ValueError(
            f"data.shape[-1] ({data.shape[-1]} is not a multiple of decimation_factor "
            f"({decimation_factor})."
        )
    if isinstance(data, np.ndarray):
        return (
            np.sum(np.reshape(data, (*data.shape[:-1], -1, decimation_factor)), axis=-1)
            / decimation_factor
        )
    elif isinstance(data, torch.Tensor):
        return (
            torch.sum(
                torch.reshape(data, (*data.shape[:-1], -1, decimation_factor)), dim=-1
            )
            / decimation_factor
        )
    else:
        raise NotImplementedError(
            f"Decimation not implemented for data of type {data}."
        )


def ceil_to_power_of_2(x):
    return 2 ** (np.ceil(np.log2(x)))


def floor_to_power_of_2(x):
    return 2 ** (np.floor(np.log2(x)))


def is_power_of_2(x):
    return 2 ** int(np.log2(x)) == x


def duration_LO(chirp_mass, frequency):
    # Eq. (3) in https://arxiv.org/abs/1703.02062
    # in geometric units:
    f = frequency / lal.C_SI
    M = chirp_mass * lal.GMSUN_SI / lal.C_SI**2
    t = 5 * (8 * np.pi * f) ** (-8 / 3) * M ** (-5 / 3)
    return t / lal.C_SI


def is_within_band(f, chirp_mass_min, delta_f_band, f_max, alpha=1, delta_f_max=np.inf):
    # check if next frequency value would be larger than global f_max
    if f + delta_f_band / 2 > f_max:
        return False
    # check whether delta_f can be increased (if yes, return False)
    elif (
        duration_LO(chirp_mass_min, f) < 1 / (2 * delta_f_band * alpha)
        and 2 * delta_f_band <= delta_f_max
    ):
        return False
    else:
        return True


def number_of_zero_crossings(x):
    if np.iscomplex(x).any():
        raise ValueError("Only works for real arrays.")
    x = x / np.max(np.abs(x))
    return np.sum((x[..., :-1] * x[..., 1:]) < 0, axis=-1)


def get_period_for_complex_oscillation(
    x: np.ndarray, upper_bound_for_monotonicity: bool = False
):
    """
    Takes complex 1D or 2D array x as input. Returns array of the same shape,
    specifying the cycle length for each bin (axis=-1). This is done by looking at the
    local rate of change of the normalized array x / np.abs(x). Assuming sine-like
    osscillations, the period is related to the maximum rate of change via

        period = 2 pi / max_local(rate_of_change_per_bin).

    Note: this assumes a monotonically increasing period.

    Parameters
    ----------
    x: np.ndarray
        Complex array with oscillation signal. 1D or 2D, oscillation pattern on axis -1.
    upper_bound_for_monotonicity: bool = False
        If set, then the periods returned increase monotonically.

    Returns
    -------
    periods_expanded: np.ndarray
        Array with same shape as x, containing the period (as float) for each bin.
    """
    if not np.iscomplexobj(x):
        raise ValueError("This is only implemented for complex oscillations.")
    if not len(x.shape) in [1, 2]:
        raise ValueError(
            f"Expected shape (num_bins) or (num_waveforms, num_bins), got {x.shape}."
        )
    # Infer period from derivative
    y = x * 1
    if np.min(np.abs(x)) == 0:
        raise ValueError("This function requires |x| > 0.")
    # normalize x
    x = x / np.abs(x)
    # normalized derivative
    dx = np.concatenate(
        ((x[..., 1:] - x[..., :-1]).real, (x[..., 1:] - x[..., :-1]).imag)
    )
    # Infer period from the derivative, assuming sine-like oscillations.
    periods = 2 * np.pi / np.abs(dx)
    if upper_bound_for_monotonicity:
        periods = np.minimum.accumulate((periods)[..., ::-1], axis=-1)[..., ::-1]
    return periods


def get_periods(x: np.ndarray, upper_bound_for_monotonicity: bool = False):
    """
    Takes 1D or 2D array x as input. Returns array of the same shape, specifying the
    cycle length for each bin (axis=-1). This is done by checking for zero-crossings
    in x. The lower/upper boundaries are filled with the periods from the neighboring
    intervals.

    Note: This assumes an oscillatory behavior of x about 0.

    Parameters
    ----------
    x: np.ndarray
        Array with oscillation signal. 1D or 2D, oscillation pattern on axis -1.
    upper_bound_for_monotonicity: bool = False
        If set, then the periods returned increase monotonically.

    Returns
    -------
    periods_expanded: np.ndarray
        Array with same shape as x, containing the period (as int) for each bin.

    Examples
    --------
    >>> x = np.array([-1, 0, 1, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1])
    >>> get_periods(x)
    array([4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6])
    >>> get_periods(x, upper_bound_for_monotonicity=True)
    array([4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6])

    >>> x = np.array([-1, 0, 1, 2, 1, 0, -1, 0, 1])
    >>> get_periods(x)
    array([4, 4, 4, 4, 4, 2, 2, 2, 2])
    >>> get_periods(x, upper_bound_for_monotonicity=True)
    array([2, 2, 2, 2, 2, 2, 2, 2, 2])
    """
    if np.iscomplex(x).any():
        raise ValueError("Only works for real arrays.")

    # implementation for single arrays
    if len(x.shape) == 1:
        zero_crossings = np.where(
            (x[:-1] >= 0) & (x[1:] < 0) | (x[:-1] <= 0) & (x[1:] > 0)
        )[0]
        periods_expanded = np.zeros(len(x), dtype=int)
        for lower, upper in zip(zero_crossings[:-1], zero_crossings[1:]):
            periods_expanded[lower:upper] = upper - lower
        # fill in boundaries
        periods_expanded[: zero_crossings[0]] = periods_expanded[zero_crossings[0]]
        periods_expanded[zero_crossings[-1] :] = periods_expanded[
            zero_crossings[-1] - 1
        ]
        # multiply with 2, as a period includes 2 zero crossings
        periods_expanded *= 2
        # if monotonically increasing periods are requested, upper bound the periods
        # with periods_expanded[i] = min(periods_expanded[i:]).
        if upper_bound_for_monotonicity:
            periods_expanded = np.minimum.accumulate(periods_expanded[::-1])[::-1]
        return periods_expanded

    # batched arrays: recurse with single arrays (there might be a faster way to do this)
    elif len(x.shape) == 2:
        periods_expanded = np.empty(x.shape, dtype=int)
        for idx, x_single in enumerate(x):
            periods_expanded[idx, :] = get_periods(
                x_single, upper_bound_for_monotonicity
            )
        return periods_expanded

    else:
        raise ValueError(
            f"Only implemented for single or batched arrays, got {len(x.shape)} axes."
        )


def get_band_nodes_for_adaptive_decimation(
    max_dec_factor_array: np.ndarray, max_dec_factor_global: int = np.inf
):
    """
    Sets up adaptive multibanding for decimation. The 1D array max_dec_factor_array has
    the same length as the original, and contains the maximal acceptable decimation
    factors for each bin. max_dec_factor_global further specifies the maximum
    decimation factor.

    Parameters
    ----------
    max_dec_factor_array: np.ndarray
        Array with maximal decimation factor for each bin. Monotonically increasing.
    max_dec_factor_global: int = np.inf
        Global maximum for decimation factor.

    Returns
    -------
    initial_downsampling: int
        Downsampling factor of band 0.
    band_nodes: list[int]
        List with nodes for bands.
        Band j consists of indices [nodes[j]:nodes[j+1].
    """
    if len(max_dec_factor_array.shape) != 1:
        raise ValueError("max_dec_factor_array needs to be 1D array.")
    if not (max_dec_factor_array[1:] >= max_dec_factor_array[:-1]).all():
        raise ValueError("max_dec_factor_array needs to increase monotonically.")

    max_dec_factor_array = np.clip(max_dec_factor_array, None, max_dec_factor_global)
    N = len(max_dec_factor_array)
    dec_factor = int(max(1, floor_to_power_of_2(max_dec_factor_array[0])))
    band_nodes = [0]
    upper = dec_factor
    initial_downsampling = dec_factor
    while upper - 1 < N:
        if upper - 1 + dec_factor >= N:
            # conclude while loop, append upper as last node
            band_nodes.append(upper)
        elif dec_factor * 2 <= max_dec_factor_array[upper]:
            # conclude previous band
            band_nodes.append(upper)
            # enter new band
            dec_factor *= 2
        upper += dec_factor

    return initial_downsampling, band_nodes
