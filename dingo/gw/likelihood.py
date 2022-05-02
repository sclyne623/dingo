import numpy as np
from torchvision.transforms import Compose
from bilby.gw.detector.networks import InterferometerList
from scipy.fft import fft

from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.domains import build_domain, FrequencyDomain
from dingo.gw.inference.data_preparation import get_event_data_and_domain
from dingo.gw.transforms import (
    ProjectOntoDetectors,
    GetDetectorTimes,
    WhitenAndScaleStrain,
)


class StationaryGaussianGWLikelihood:
    """
    Implements GW likelihood for stationary, Gaussian noise.
    """

    def __init__(
        self,
        wfg_kwargs,
        data_domain,
        event_data,
        t_ref=None,
        wfg_frequency_range=None,
        time_marginalization_kwargs=None,
    ):
        """
        Initialize the likelihood.

        Parameters
        ----------
        wfg_kwargs: dict
            Waveform generator parameters (at least approximant and f_ref).
        data_domain: dingo.gw.domains.Domain
            Domain object for event data.
        event_data: dict
            GW data. Contains strain data in event_data["waveforms"] and asds in
            event_data["asds"].
        t_ref: float
            Reference time; true geocent time for GW is t_ref + theta["geocent_time"].
        wfg_frequency_range: dict
            Frequency range for waveform generator. If None, that of data domain is used,
            which corresponds to the bounds of the likelihood integral.
            Possible keys:
                'f_start': float
                    Frequency at which to start the waveform generation. Overrides f_start in
                    metadata["model"]["dataset_settings"]["waveform_generator"].
                'f_end': float
                    Frequency at which to start the waveform generation.
        time_marginalization_kwargs: dict
            Time marginalization parameters. If None, no time marginalization is used.
        """
        super().__init__()
        # set up waveform generator
        self.wfg_kwargs = wfg_kwargs
        self.data_domain = data_domain
        if type(self.data_domain) is not FrequencyDomain:
            raise NotImplementedError(
                f"Likelihood implemented for FrequencyDomain, "
                f"got {type(self.data_domain)}."
            )
        # The waveform generator potentially has a larger frequency range than the
        # data domain, which may e.g. be required for EOB waveforms to generate robustly.
        # When computing the likelihood the data will be truncated accordingly.
        self.waveform_generator = get_wfg(wfg_kwargs, data_domain, wfg_frequency_range)

        # set GW event data
        self.t_ref = t_ref
        self.whitened_strains = {
            k: v / event_data["asds"][k] / data_domain.noise_std
            for k, v in event_data["waveform"].items()
        }
        self.asds = event_data["asds"]
        if len(list(self.whitened_strains.values())[0]) != data_domain.max_idx + 1:
            raise ValueError("Strain data does not match domain.")
        # log noise evidence, independent of theta and waveform model
        self.log_Zn = sum(
            [
                -1 / 2.0 * inner_product(d_ifo, d_ifo)
                for d_ifo in self.whitened_strains.values()
            ]
        )

        # build transforms for detector projections
        self.ifo_list = InterferometerList(self.whitened_strains.keys())
        self.projection_transforms = Compose(
            [
                GetDetectorTimes(self.ifo_list, self.t_ref),
                ProjectOntoDetectors(self.ifo_list, self.data_domain, self.t_ref),
                WhitenAndScaleStrain(self.data_domain.noise_std),
            ]
        )

        # optionally initialize time marginalization
        self.time_marginalization = False
        if time_marginalization_kwargs is not None:
            self.initialize_time_marginalization(**time_marginalization_kwargs)

    def initialize_time_marginalization_old(self, time_prior, N_t=None):
        """
        Initialize time marginalization.

        Parameters
        ----------
        time_prior: bilby.core.prior.Prior
            Prior for time.
        N_t: int = None
            Number of time samples. If None, we use FFT to compute the time
            marginalized likelihood, which is fast but has a fixed time resolution of
            1/self.data_domain.f_max. Else, the time shifting will be performed manually.
        """
        self.time_marginalization = True
        self.time_prior = time_prior
        self.N_t = N_t
        t_lower, t_upper = self.time_prior._minimum, self.time_prior._maximum
        # if N_t is None, set up time marginalization via FFT
        if self.N_t is None:
            # FFT returns an array, where bin i corresponds to time i * dt, where dt is
            # 1 / self.domain.delta_f. The array wraps around, which we account for below.
            time_axis = np.arange(len(self.data_domain())) / self.data_domain.f_max
            self.time_prior_log = np.max(
                (
                    self.time_prior.ln_prob(time_axis),
                    self.time_prior.ln_prob(time_axis - max(time_axis)),
                ),
                axis=0,
            )
            # normalize the time prior
            self.time_prior_log -= np.log(np.sum(np.exp(self.time_prior_log)))
        else:
            # if N_t is not None, set up time marginalization via manual shifting.
            # To that end we evaluate the time prior on a uniform 1D grid.
            self.time_samples = np.linspace(t_lower, t_upper, self.N_t)
            self.time_prior_log = self.time_prior.ln_prob(self.time_samples)
            self.time_prior_log -= np.log(np.sum(np.exp(self.time_prior_log)))

    def initialize_time_marginalization(self, t_lower, t_upper, N_FFT=1):
        """
        Initialize time marginalization. Time marginalization can be performed via FFT,
        which is super fast. However, this limits the time resolution to delta_t =
        1/self.data_domain.f_max. In order to allow for a finer time resolution we
        compute the time marginalized likelihood N_FFT via FFT on a grid of N_FFT
        different time shifts [0, delta_t, 2*delta_t, ..., (N_FFT-1)*delta_t] and
        average over the time shifts. The effective time resolution is thus

            delta_t_eff = delta_t / N_FFT = 1 / (f_max * N_FFT).

        Note: Time marginalization in only implemented for uniform time priors.

        Parameters
        ----------
        t_lower: float
            Lower time bound of the uniform time prior.
        t_upper: float
            Upper time bound of the uniform time prior.
        N_FFT: int = 1
            Size of grid for FFT for time marginalization.
        """
        self.time_marginalization = True
        self.N_FFT = N_FFT
        delta_t = 1.0 / self.data_domain.f_max  # time resolution of FFT
        # time shifts for different FFTs
        self.t_FFT = np.arange(self.N_FFT) * delta_t / self.N_FFT

        self.shifted_strains = {}
        for idx, dt in enumerate(self.t_FFT):
            # Instead of shifting the waveform mu by + dt when computing the
            # time-marginalized likelihood, we shift the strain data by -dt. This saves
            # time for likelihood evaluations, since it can be precomputed.
            self.shifted_strains[dt] = {
                k: v * np.exp(-2j * np.pi * self.data_domain() * (-dt))
                for k, v in self.whitened_strains.items()
            }

        # Get the time prior. This will be multiplied with the result of the FFT.
        T = 1 / self.data_domain.delta_f
        time_axis = (np.arange(len(self.data_domain())) / self.data_domain.f_max)
        self.time_grid = time_axis[:, np.newaxis] + self.t_FFT[np.newaxis, :]
        active_indices = np.where(
            (self.time_grid >= t_lower) & (self.time_grid <= t_upper)
            | (self.time_grid - T >= t_lower) & (self.time_grid - T <= t_upper)
        )
        time_prior = np.zeros(self.time_grid.shape)
        time_prior[active_indices] = 1.0
        with np.errstate(divide='ignore'): # ignore warnings for log(0) = -inf
            self.time_prior_log = np.log(time_prior / np.sum(time_prior))

    def generate_signal(self, theta):
        """
        Compute the GW signal for a given set of parameters theta.

        Step 1: generate polarizations h_plus and h_cross
        Step 2: project h_plus and h_cross onto detectors,
                whiten the signal, scale to account for window factor

        Parameters
        ----------
        theta: dict
            BBH parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        gw_strain: dict
            GW signal for each detector.
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}

        # Step 1: generate polarizations h_plus and h_cross
        polarizations = self.waveform_generator.generate_hplus_hcross(theta_intrinsic)
        polarizations = {  # truncation, in case wfg has a larger frequency range
            k: self.data_domain.update_data(v) for k, v in polarizations.items()
        }

        # Step 2: project h_plus and h_cross onto detectors
        sample = {
            "parameters": theta_intrinsic,
            "extrinsic_parameters": theta_extrinsic,
            "waveform": polarizations,
            "asds": self.asds,
        }
        sample = self.projection_transforms(sample)

        return sample

    def log_likelihood(self, theta):
        """
        The likelihood is given by

                  log L(d|theta) = psi - 1/2. <d - mu(theta), d - mu(theta)>

        where psi is a waveform model independent (and thus irrelevant) constant. Here,
        we denote the strain data by d and the GW signal by mu(theta).
        [see e.g. arxiv.org/pdf/1809.02293, equation (44) for details]
        We expand this expression below to compute the log likelihood, and omit psi.
        The expaneded expression reads

                log L(d|theta) = log_Zn + kappa2(theta) - 1/2 rho2opt(theta),

                log_Zn = -1/2. <d, d>,
                kappa2(theta) = <d, mu(theta)>,
                rho2opt(theta) = <mu(theta), mu(theta)>.

        The noise-weighted inner product is defined as

                  <a, b> = 4 * delta_f * sum(a.conj() * b / PSD).real.

        Here, we work with data d and signals mu that are already whitened by
        1 / [sqrt(PSD) * domain.noise_std], where

                  noise_std = np.sqrt(window_factor) / np.sqrt(4 * delta_f).

        With this preprocessing, the inner products thus simply become

                  <a, b> = sum(a.conj() * b).real.

        ! Be careful with window factors here !


        Time marginalization:
        The above expansion of the likelihood is particularly useful for time
        marginalization, as only kappa2 depends on the time parameter.


        Parameters
        ----------
        theta: dict
            BBH parameters.

        Returns
        -------
        log_likelihood: float
        """

        # Step 1: Compute whitened GW strain mu(theta) for parameters theta.
        mu = self.generate_signal(theta)["waveform"]
        d = self.whitened_strains

        # Step 2: Compute likelihood. log_Zn is precomputed, so we only need to
        # compute the remaining terms rho2opt and kappa2
        rho2opt = sum([inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values()])
        kappa2 = sum(
            [
                inner_product(d_ifo, mu_ifo)
                for d_ifo, mu_ifo in zip(d.values(), mu.values())
            ]
        )
        return self.log_Zn + kappa2 - 1 / 2.0 * rho2opt

    def log_likelihood_time_marginalized_old(self, theta):
        """
        Compute log likelihood with time marginalization.

        Parameters
        ----------
        theta

        Returns
        -------
        log_likelihood: float
        """
        # Step 1: Compute whitened GW strain mu(theta) for parameters theta.
        # The geocent_time parameter needs to be set to 0.
        theta["geocent_time"] = 0.0
        mu = self.generate_signal(theta)["waveform"]
        d = self.whitened_strains

        # Step 2: Compute likelihood. log_Zn is precomputed, so we only need to
        # compute the remaining terms rho2opt and kappa2.
        # rho2opt is time independent, and thus same as in the log_likelihood method.
        rho2opt = sum([inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values()])

        # kappa2 is time dependent. We compute it for the discretized times k * delta_t
        if self.N_t is None:
            # Compute marginalized kappa2 with FFT
            kappa2_ = [
                fft(d_ifo.conj() * mu_ifo).real
                for d_ifo, mu_ifo in zip(d.values(), mu.values())
            ]
            # sum contributions of different ifos
            kappa2_ = np.sum(kappa2_, axis=0)
            # marginalize over time; this requires multiplying the likelihoods with the
            # prior (*not* in log space), summing over the time bins, and then taking
            # the log. See Eq. (52) in https://arxiv.org/pdf/1809.02293.pdf.
            # To prevent numerical issues, we use the logsumexp trick.
            alpha = np.max(kappa2_ + self.time_prior_log)
            kappa2 = alpha - np.log(
                np.sum(np.exp(kappa2_ + self.time_prior_log - alpha))
            )

        else:
            kappa2_ = np.zeros((len(d), self.N_t))
            for idx_ifo, (d_ifo, mu_ifo) in enumerate(zip(d.values(), mu.values())):
                for idx_t, t in enumerate(self.time_samples):
                    kappa2_[idx_ifo, idx_t] = inner_product(
                        d_ifo, mu_ifo * np.exp(-2j * np.pi * self.data_domain() * t)
                    )
            kappa2_ = np.sum(kappa2_, axis=0)
            # marginalize over time; this requires multiplying the likelihoods with the
            # prior (*not* in log space), summing over the time bins, and then taking
            # the log. See Eq. (52) in https://arxiv.org/pdf/1809.02293.pdf.
            # To prevent numerical issues, we use the logsumexp trick.
            alpha = np.max(kappa2_ + self.time_prior_log)
            kappa2 = alpha - np.log(
                np.sum(np.exp(kappa2_ + self.time_prior_log - alpha))
            )

        return self.log_Zn + kappa2 - 1 / 2.0 * rho2opt

    def log_likelihood_time_marginalized(self, theta):
        """
        Compute log likelihood with time marginalization.

        Parameters
        ----------
        theta

        Returns
        -------
        log_likelihood: float
        """
        # Step 1: Compute whitened GW strain mu(theta) for parameters theta.
        # The geocent_time parameter needs to be set to 0.
        theta["geocent_time"] = 0.0
        mu = self.generate_signal(theta)["waveform"]
        # d = self.whitened_strains

        # Step 2: Compute likelihood. log_Zn is precomputed, so we only need to
        # compute the remaining terms rho2opt and kappa2.
        # rho2opt is time independent, and thus same as in the log_likelihood method.
        rho2opt = sum([inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values()])

        # kappa2 is time dependent. We use FFT to compute it for the discretized times
        # k * (delta_t/N_FFT) and then sum over the time bins. The kappa2 contribution
        # is then given by
        #
        #       log sum_k exp(kappa2_k + log_prior_k),
        #
        # see Eq. (52) in https://arxiv.org/pdf/1809.02293.pdf. Here, kappa2_k is the
        # value of kappa2 and log_prior_k is the log_prior density at time
        # k * (delta_t/N_FFT). The sum over k is the discretized integration of t.
        # Note: the time is discretized in two ways; for each FFT j (N_FFT in total),
        # there are len(data_domain) time samples i, such that
        #
        #       t_ij = i * delta_t + j * (delta_t/N_FFT).
        #
        # Summing over the time bins corresponds to a sum across both axes i and j.
        kappa2_ij = np.zeros((len(self.data_domain), self.N_FFT))
        for j, dt in enumerate(self.t_FFT):
            # Get precomputed whitened strain, that is shifted by -dt.
            d = self.shifted_strains[dt]
            # Compute kappa2 contribution
            kappa2_ = [
                fft(d_ifo.conj() * mu_ifo).real
                for d_ifo, mu_ifo in zip(d.values(), mu.values())
            ]
            # sum contributions of different ifos
            kappa2_ij[:, j] = np.sum(kappa2_, axis=0)
        # Marginalize over time; this requires multiplying the likelihoods with the
        # prior (*not* in log space), summing over the time bins (both axes i and j!),
        # and then taking the log. See Eq. (52) in https://arxiv.org/pdf/1809.02293.pdf.
        # To prevent numerical issues, we use the logsumexp trick.
        assert kappa2_ij.shape == self.time_prior_log.shape
        exponent = kappa2_ij + self.time_prior_log
        alpha = np.max(exponent)
        kappa2 = alpha + np.log(np.sum(np.exp(exponent - alpha)))

        return self.log_Zn + kappa2 - 1 / 2.0 * rho2opt

    def log_prob(self, *args, **kwargs):
        """
        Wraps log_likelihood method, required since downstream methods call log_prob.
        """
        if not self.time_marginalization:
            return self.log_likelihood(*args, **kwargs)
        else:
            return self.log_likelihood_time_marginalized(*args, **kwargs)


def inner_product(a, b, min_idx=0, delta_f=None, psd=None):
    """
    Compute the inner product between two complex arrays. There are two modes: either,
    the data a and b are not whitened, in which case delta_f and the psd must be
    provided. Alternatively, if delta_f and psd are not provided, the data a and b are
    assumed to be whitened already (i.e., whitened as d -> d * sqrt(4 delta_f / psd)).

    Parameters
    ----------
    a: np.ndaarray
        First array with frequency domain data.
    b: np.ndaarray
        Second array with frequency domain data.
    min_idx: int = 0
        Truncation of likelihood integral, index of lowest frequency bin to consider.
    delta_f: float
        Frequency resolution of the data. If None, a and b are assumed to be whitened
        and the inner product is computed without further whitening.
    psd: np.ndarray = None
        PSD of the data. If None, a and b are assumed to be whitened and the inner
        product is computed without further whitening.

    Returns
    -------
    inner_product: float
    """
    #
    if psd is not None:
        if delta_f is None:
            raise ValueError(
                "If unwhitened data is provided, both delta_f and psd must be provided."
            )
        return 4 * delta_f * np.sum((a.conj() * b / psd)[min_idx:]).real
    else:
        return np.sum((a.conj() * b)[min_idx:]).real


def split_off_extrinsic_parameters(theta):
    """
    Split theta into intrinsic and extrinsic parameters.

    Parameters
    ----------
    theta: dict
        BBH parameters. Includes intrinsic parameters to be passed to waveform
        generator, and extrinsic parameters for detector projection.

    Returns
    -------
    theta_intrinsic: dict
        BBH intrinsic parameters.
    theta_extrinsic: dict
        BBH extrinsic parameters.
    """
    extrinsic_parameters = ["geocent_time", "luminosity_distance", "ra", "dec", "psi"]
    theta_intrinsic = {}
    theta_extrinsic = {}
    for k, v in theta.items():
        if k in extrinsic_parameters:
            theta_extrinsic[k] = v
        else:
            theta_intrinsic[k] = v
    # set fiducial values for time and distance
    theta_intrinsic["geocent_time"] = 0
    theta_intrinsic["luminosity_distance"] = 100
    return theta_intrinsic, theta_extrinsic


def build_stationary_gaussian_likelihood(
    metadata,
    event_dataset=None,
    wfg_frequency_range=None,
):
    """
    Build a StationaryGaussianLikelihoodBBH object from the metadata.

    Parameters
    ----------
    metadata: dict
        Metadata from stored dingo parameter samples file.
        Typially accessed via pd.read_pickle(/path/to/dingo-output.pkl).metadata.
    event_dataset: str = None
        Path to event dataset for caching. If None, don't cache.
    wfg_frequency_range: dict = None
        Frequency range for waveform generator. If None, that of data domain is used,
        which corresponds to the bounds of the likelihood integral.
        Possible keys:
            'f_start': float
                Frequency at which to start the waveform generation. Overrides f_start in
                metadata["model"]["dataset_settings"]["waveform_generator"].
            'f_end': float
                Frequency at which to start the waveform generation.

    Returns
    -------
    likelihood: StationaryGaussianGWLikelihood
        likelihood object
    """
    # get strain data
    event_data, data_domain = get_event_data_and_domain(
        metadata["model"], event_dataset=event_dataset, **metadata["event"]
    )

    # set up likelihood
    likelihood = StationaryGaussianGWLikelihood(
        metadata["model"]["dataset_settings"]["waveform_generator"],
        data_domain,
        event_data,
        t_ref=metadata["event"]["time_event"],
        wfg_frequency_range=wfg_frequency_range,
    )

    return likelihood


def get_wfg(wfg_kwargs, data_domain, frequency_range=None):
    """
    Set up waveform generator from wfg_kwargs. The domain of the wfg is primarily
    determined by the data domain, but a new (larger) frequency range can be
    specified if this is necessary for the waveforms to be generated successfully
    (e.g., for EOB waveforms which require a sufficiently small f_min and sufficiently
    large f_max).

    Parameters
    ----------
    wfg_kwargs: dict
        Waveform generator parameters.
    data_domain: dingo.gw.domains.Domain
        Domain of event data, with bounds determined by likelihood integral.
    frequency_range: dict = None
        Frequency range for waveform generator. If None, that of data domain is used,
        which corresponds to the bounds of the likelihood integral.
        Possible keys:
            'f_start': float
                Frequency at which to start the waveform generation. Overrides f_start in
                metadata["model"]["dataset_settings"]["waveform_generator"].
            'f_end': float
                Frequency at which to start the waveform generation.

    Returns
    -------
    wfg: dingo.gw.waveform_generator.WaveformGenerator
        Waveform generator object.

    """
    if frequency_range is None:
        return WaveformGenerator(domain=data_domain, **wfg_kwargs)

    else:
        if "f_start" in frequency_range and frequency_range["f_start"] is not None:
            if frequency_range["f_start"] > data_domain.f_min:
                raise ValueError("f_start must be less than f_min.")
            wfg_kwargs["f_start"] = frequency_range["f_start"]
        if "f_end" in frequency_range and frequency_range["f_end"] is not None:
            if frequency_range["f_end"] < data_domain.f_max:
                raise ValueError("f_end must be greater than f_max.")
            # get wfg domain, but care to not modify the original data_domain
            data_domain = build_domain(
                {**data_domain.domain_dict, "f_max": frequency_range["f_end"]}
            )
        return WaveformGenerator(domain=data_domain, **wfg_kwargs)


def main():
    import pandas as pd

    samples = pd.read_pickle(
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets/dingo_samples"
        "/02_XPHM/dingo_samples_GW150914.pkl"
    )
    event_dataset = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel"
        "/tutorials/02_gwpe/datasets/strain_data/events_dataset.hdf5"
    )

    likelihood = build_stationary_gaussian_likelihood(samples.attrs, event_dataset)

    from tqdm import tqdm

    log_likelihoods = []
    for idx in tqdm(range(1000)):
        theta = dict(samples.iloc[idx])
        try:
            l = likelihood.log_prob(theta)
        except:
            print(idx)
            l = float("nan")
        log_likelihoods.append(l)
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = log_likelihoods[~np.isnan(log_likelihoods)]
    print(f"mean: {np.mean(log_likelihoods)}")
    print(f"std: {np.std(log_likelihoods)}")
    print(f"max: {np.max(log_likelihoods)}")
    print(f"min: {np.min(log_likelihoods)}")


if __name__ == "__main__":
    main()
