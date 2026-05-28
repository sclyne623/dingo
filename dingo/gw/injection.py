import numpy as np
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.gw.domains.base_frequency_domain import BaseFrequencyDomain
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.domains import (
    UniformFrequencyDomain,
    Domain,
    MultibandedFrequencyDomain,
)
from dingo.gw.domains import build_domain, build_domain_from_model_metadata
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults, split_off_extrinsic_parameters
from dingo.gw.transforms import (
    GetDetectorTimes,
    ProjectOntoDetectors,
    ProjectOntoSpaceDetectors,
    WhitenAndScaleStrain,
    ApplyCalibrationUncertainty,
)
from dingo.gw.waveform_generator.waveform_generator import (
    WaveformGenerator,
    NewInterfaceWaveformGenerator,
    LISAWaveformGenerator,
    BBHxWaveformGenerator,
)
import lisabeta.tools.pytools as pytools


class GWSignal(object):
    """
    Base class for generating gravitational wave signals in interferometers. Generates
    waveform polarizations based on provided parameters, and then projects to detectors.

    Includes option for whitening the signal based on a provided ASD.
    """

    def __init__(
        self,
        wfg_kwargs: dict,
        wfg_domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        data_domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        ifo_list: list,
        t_ref: float,
        lisa_settings: dict | None = None,
    ):
        """
        Parameters
        ----------
        wfg_kwargs : dict
            Waveform generator parameters [approximant, f_ref, and (optionally) f_start].
        wfg_domain : UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain used for waveform generation. This can potentially deviate from the
            final domain, having a wider frequency range needed for waveform generation.
        data_domain : UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain object for final signal.
        ifo_list : list
            Names of interferometers for projection.
        t_ref : float
            Reference time that specifies ifo locations.
        """
        self._use_base_domain = False
        self._check_domains(wfg_domain, data_domain)
        self.data_domain = data_domain
        self.LISA_flag = wfg_kwargs.get("LISA", False)
        self.BBHx_flag = wfg_kwargs.get("BBHx", False)
        self.lisa_like_flag = self.LISA_flag or self.BBHx_flag
        self.lisa_settings = lisa_settings

        # The waveform generator potentially has a larger frequency range than the
        # domain of the trained network / requested injection / etc. This is typically
        # the case for EOB waveforms, which require the larger range to generate
        # robustly. For this reason we have two domains.

        new_interface_flag = wfg_kwargs.get("new_interface", False)
        if new_interface_flag:
            self.waveform_generator = NewInterfaceWaveformGenerator(
                domain=wfg_domain, **wfg_kwargs
            )
        elif self.BBHx_flag:
            self.waveform_generator = BBHxWaveformGenerator(
                domain=wfg_domain, **wfg_kwargs
            )
        elif self.LISA_flag:
            self.waveform_generator = LISAWaveformGenerator(domain = wfg_domain, **wfg_kwargs)
        else:
            self.waveform_generator = WaveformGenerator(domain=wfg_domain, **wfg_kwargs)

        self.t_ref = t_ref
        if not self.lisa_like_flag:
            self.ifo_list = InterferometerList(ifo_list)
        else:
            self.ifo_list = ifo_list

        # When we set self.whiten, the projection transforms are automatically prepared.
        self._calibration_envelope = None
        self._calibration_marginalization_kwargs = None
        self.whiten = False

        self.asd = None

    @staticmethod
    def _check_domains(
        domain_in: UniformFrequencyDomain | MultibandedFrequencyDomain,
        domain_out: UniformFrequencyDomain | MultibandedFrequencyDomain,
    ):
        if domain_in.f_min > domain_out.f_min or domain_in.f_max < domain_out.f_max:
            raise ValueError(
                "Output domain is not contained within WaveformGenerator domain."
            )
        if (
            domain_in.domain_dict["type"] == "UniformFrequencyDomain"
            and domain_out.domain_dict["type"] == "UniformFrequencyDomain"
        ):
            if domain_in.delta_f != domain_out.delta_f:
                raise ValueError("Domains must have same delta_f.")

    @property
    def use_base_domain(self):
        return self._use_base_domain

    @use_base_domain.setter
    def use_base_domain(self, value: bool):
        if value != self._use_base_domain:
            if value:
                if hasattr(self.data_domain, "base_domain"):
                    self.waveform_generator.domain = (
                        self.waveform_generator.full_domain.base_domain
                    )
                    self.data_domain = self.data_domain.base_domain
                    self._use_base_domain = True
                    self._initialize_transform()
                else:
                    print(
                        f"{type(self.data_domain)} has no base domain. Nothing to do."
                    )
            else:
                raise NotImplementedError(
                    "Cannot recover original domain from base domain alone."
                )

    @property
    def whiten(self):
        """
        Bool specifying whether to whiten (and scale) generated signals.
        """
        return self._whiten

    @whiten.setter
    def whiten(self, value):
        self._whiten = value
        self._initialize_transform()

    @property
    def calibration_marginalization_kwargs(self):
        """
        Dictionary with the following keys:

        calibration_envelope
            Dictionary of the form {"H1": filepath, "L1": filepath, ...} with locations of
            lookup tables for the calibration uncertainty curves.

        num_calibration_nodes
            Number of nodes for the calibration model.

        num_calibration_curves
            Number of calibration curves to use in marginalization.
        """
        return self._calibration_marginalization_kwargs

    @calibration_marginalization_kwargs.setter
    def calibration_marginalization_kwargs(self, value):
        self._calibration_marginalization_kwargs = value
        self._initialize_transform()

    def _initialize_transform(self):
        if self.lisa_like_flag:
            
            transforms = [ProjectOntoSpaceDetectors("TDIAET",self.data_domain,self.t_ref,self.ifo_list,self.lisa_settings)]
            
        else:
                                                    
            transforms = [
                GetDetectorTimes(self.ifo_list, self.t_ref),
                ProjectOntoDetectors(self.ifo_list, self.data_domain, self.t_ref),
            ]
        if self.calibration_marginalization_kwargs:
            transforms.append(
                ApplyCalibrationUncertainty(
                    self.ifo_list,
                    self.data_domain,
                    **self.calibration_marginalization_kwargs,
                )
            )
        if self.whiten:
            transforms.append(WhitenAndScaleStrain(self.data_domain.noise_std))
        self.projection_transforms = Compose(transforms)

    def _bbhx_direct_response(self, theta_intrinsic, theta_extrinsic):
        """Generate the detector-response waveform for a BBHx generator, mirroring
        the training-time DetectorTransform path so injections match what the
        network was trained on.

        Returns a dict ``{channel_name: 1d complex array}`` on the data domain.
        """
        wfg = self.waveform_generator
        if getattr(wfg, "gpu_fastpath", False) and getattr(
            wfg, "backend_native_fused", False
        ):
            h = wfg.generate_direct_response_backend_native(
                theta_intrinsic, theta_extrinsic, catch_waveform_errors=False
            )
        else:
            h = wfg.generate_amp_phase(
                {**theta_intrinsic, **theta_extrinsic}, catch_waveform_errors=False
            )
        waveform = h["waveform"] if isinstance(h, dict) else h

        # Backend array (CuPy / torch) -> numpy.
        if hasattr(waveform, "get"):
            waveform = np.asarray(waveform.get())
        else:
            waveform = np.asarray(waveform)
        waveform = np.squeeze(waveform)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

        strains = {
            ifo: waveform[i].reshape(-1) for i, ifo in enumerate(self.ifo_list)
        }

        # If the generator returned a decentered waveform (merger at t=0), the
        # training-time DetectorTransform re-applies exp(-i 2π f t_ref) to put the
        # merger back at the physical t_ref. We must reproduce that here, otherwise
        # the injected strain is time-shifted by t_ref relative to the network's
        # training distribution. See detector_transforms.DetectorTransform.
        if getattr(wfg, "decenter_waveform", False):
            t_ref_val = theta_extrinsic.get(
                "t_ref",
                theta_extrinsic.get(
                    "geocent_time",
                    theta_intrinsic.get(
                        "t_ref",
                        theta_intrinsic.get(
                            "geocent_time", wfg.default_t_ref_seconds
                        ),
                    ),
                ),
            )
            nf = next(iter(strains.values())).shape[0]
            freqs = np.asarray(self.data_domain.sample_frequencies, dtype=np.float64)
            if freqs.shape[0] != nf:
                # Fall back to the generator's output grid if the data-domain grid
                # does not line up with the returned waveform length.
                freqs = np.asarray(
                    wfg._get_output_frequency_grid(), dtype=np.float64
                )
            phasor = np.exp(-1j * 2.0 * np.pi * float(t_ref_val) * freqs)
            strains = {k: v * phasor for k, v in strains.items()}

        return strains

    def _bbhx_signal_m(self, theta_intrinsic, theta_extrinsic):
        """Per-azimuthal-m decomposition of the BBHx signal using the BBHx-internal
        response, consistent with ``signal()`` and the data.

        Each mode (l, m) transforms as ``exp(-i m phase)`` under reference-phase
        shifts, so we generate the detector response of the modes sharing an
        azimuthal index m at ``phase=0``; downstream code (e.g. the
        phase-marginalized likelihood) applies the m-dependent phase. This mirrors
        ``signal()`` (same ``_bbhx_direct_response`` path, including the re-center),
        unlike the generate_amp_phase_m + ProjectOntoSpaceDetectors path which uses
        a different response convention.

        Returns a dict ``{m: {"waveform": {channel: strain}, ...}}``.
        """
        wfg = self.waveform_generator
        full_modes = list(wfg.mode_list)

        groups = {}
        for lm in full_modes:
            groups.setdefault(int(lm[1]), []).append(tuple(lm))

        # The decomposition is defined at phase=0; the m-dependent phase factor is
        # re-applied by the consumer.
        theta_extrinsic_ref = {**theta_extrinsic, "phase": 0.0, "phi": 0.0}

        m_bins = {}
        try:
            for m, modes_m in groups.items():
                wfg.mode_list = modes_m
                waveform = self._bbhx_direct_response(
                    theta_intrinsic, theta_extrinsic_ref
                )
                sample = {
                    "parameters": theta_intrinsic,
                    "extrinsic_parameters": theta_extrinsic,
                    "waveform": waveform,
                }
                if self.asd is not None:
                    sample["asds"] = self.asd
                if self.whiten:
                    sample = WhitenAndScaleStrain(self.data_domain.noise_std)(sample)
                m_bins[m] = sample
        finally:
            wfg.mode_list = full_modes

        return m_bins

    def signal(self, theta):
        """
        Compute the GW signal for parameters theta.

        Step 1: Generate polarizations
        Step 2: Project polarizations onto detectors; optionally (depending on
        self.whiten) whiten and scale.

        Parameters
        ----------
        theta: dict
            Signal parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        dict
            keys:
                waveform: GW strain signal for each detector.
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}


        # Step 1: generate polarizations h_plus and h_cross or AMp/Phase for LISA
        if isinstance(self.waveform_generator, LISAWaveformGenerator):
            theta_intrinsic = pytools.complete_mass_params(theta_intrinsic)
            theta_intrinsic = pytools.complete_spin_params(theta_intrinsic)
            
            
            polarizations = self.waveform_generator.generate_amp_phase({**theta_extrinsic,**theta_intrinsic})
        elif isinstance(self.waveform_generator, BBHxWaveformGenerator):
            # Apply the BBHx-internal detector response here, exactly as training
            # does (see DetectorTransform in detector_transforms.py). The network
            # was trained on this response, so injections must reproduce it rather
            # than going through generate_amp_phase_m + ProjectOntoSpaceDetectors,
            # which uses a different TDI/response convention and would feed the
            # network out-of-distribution data.
            waveform = self._bbhx_direct_response(theta_intrinsic, theta_extrinsic)
            sample = {
                "parameters": theta_intrinsic,
                "extrinsic_parameters": theta_extrinsic,
                "waveform": waveform,
            }
            asd = self.asd
            if asd is not None:
                sample["asds"] = asd
            if self.whiten:
                sample = WhitenAndScaleStrain(self.data_domain.noise_std)(sample)
            return sample
        else:
            polarizations = self.waveform_generator.generate_hplus_hcross(theta_intrinsic)
            polarizations = {  # truncation, in case wfg has a larger frequency range
            k: self.data_domain.update_data(v) for k, v in polarizations.items()
            }

        # Step 2: project h_plus and h_cross onto detectors
        sample = {
            "parameters": theta_intrinsic,
            "extrinsic_parameters": theta_extrinsic,
            "waveform": polarizations,
        }
        
        
        asd = self.asd
        if asd is not None:
            sample["asds"] = asd

        return self.projection_transforms(sample)

    # It would be good to have an ASD class to handle all of this functionality,
    # namely storing ASDs from numpy arrays, from ASDDatasets, loading from files,
    # etc. For now this functionality is partially implemented here.

    def signal_m(self, theta):
        """
        Compute the GW signal for parameters theta. Same as self.signal(theta) method,
        but it does not sum the contributions of the individual modes, and instead
        returns a dict {m: pol_m for m in [-l_max,...,0,...,l_max]} where each
        contribution pol_m transforms as exp(-1j * m * phase_shift) under phase shifts.

        Step 1: Generate polarizations
        Step 2: Project polarizations onto detectors;
                optionally (depending on self.whiten) whiten and scale.

        Parameters
        ----------
        theta: dict
            Signal parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        dict
            keys:
                waveform:
                    GW strain signal for each detector, with individual contributions
                    {m: pol_m for m in [-l_max,...,0,...,l_max]}
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}
        if isinstance(self.waveform_generator, LISAWaveformGenerator):

            theta_intrinsic = pytools.complete_mass_params(theta_intrinsic)
            theta_intrinsic["phi"] = theta_intrinsic["phase"]
            
            theta_intrinsic = pytools.complete_spin_params(theta_intrinsic)
            
            
            pol_lm = self.waveform_generator.generate_amp_phase({**theta_extrinsic,**theta_intrinsic})
        elif isinstance(self.waveform_generator, BBHxWaveformGenerator):
            # Decompose using the BBHx-internal response (consistent with signal()
            # and the data), rather than generate_amp_phase_m +
            # ProjectOntoSpaceDetectors, which applies a different (lisabeta)
            # response and breaks importance sampling.
            return self._bbhx_signal_m(theta_intrinsic, theta_extrinsic)
        if isinstance(
            self.waveform_generator, (LISAWaveformGenerator, BBHxWaveformGenerator)
        ):
            m_bins = {}
            for lm, pol_data in pol_lm.items():
                m = int(lm[1]) # Ensure m is an integer key
                
                # Wrap for the Projector
                sample_in = {
                    "parameters": theta_intrinsic,
                    "extrinsic_parameters": theta_extrinsic,
                    "waveform": {lm: pol_data},
                }
                if self.asd is not None:
                    sample_in["asds"] = self.asd

                # Run projection (returns dict with 'waveform', 'parameters', etc.)
                projected_sample = self.projection_transforms(sample_in)
                
                # Sum into the m-bin
                if m not in m_bins:
                    # We need a deep copy of the structure
                    m_bins[m] = {
                        "parameters": projected_sample["parameters"],
                        "extrinsic_parameters": projected_sample["extrinsic_parameters"],
                        "waveform": {chan: strain.copy() for chan, strain in projected_sample["waveform"].items()},
                    }
                    if self.asd is not None:
                        m_bins[m]["asds"] = self.asd
                else:
                    # Accumulate the waveform for this m index
                    for chan, strain in projected_sample["waveform"].items():
                        m_bins[m]["waveform"][chan] += strain

            return m_bins
            


            

        # Step 1: generate m-contributions to polarizations h_plus and h_cross
        else:
            pol_m = self.waveform_generator.generate_hplus_hcross_m(theta_intrinsic)

        

        # truncation, in case wfg has a larger frequency range
            pol_m = {
                k_m: {
                    k_pol: self.data_domain.update_data(v_pol)
                    for k_pol, v_pol in v_m.items()
                }
                for k_m, v_m in pol_m.items()
            }

        # Step 2: project m-contributions to h_plus and h_cross onto detectors
            sample_out = {}
            for m, pol in pol_m.items():
                sample = {
                    "parameters": theta_intrinsic,
                    "extrinsic_parameters": theta_extrinsic,
                    "waveform": pol,
                }
                if self.asd is not None:
                    sample["asds"] = self.asd
                sample_out[m] = self.projection_transforms(sample)

        return sample_out

    @property
    def asd(self):
        """
        Amplitude spectral density.

        Either a single array, a dict (for individual interferometers),
        or an ASDDataset, from which random ASDs are drawn.
        """
        if isinstance(self._asd, np.ndarray):
            if self.lisa_like_flag:
                asd = {ifo: self._asd for ifo in self.ifo_list}
            else:
                asd = {ifo.name: self._asd for ifo in self.ifo_list}
        elif isinstance(self._asd, dict):
            asd = self._asd
        elif isinstance(self._asd, ASDDataset):
            asd = self._asd.sample_random_asds()
        elif self._asd is None:
            return None
        else:
            raise TypeError("Invalid ASD type.")
        asd = {
            k: self.data_domain.update_data(v, low_value=1e-20) for k, v in asd.items()
        }
        return asd

    @asd.setter
    def asd(self, asd):
        if isinstance(
            self.waveform_generator, (LISAWaveformGenerator, BBHxWaveformGenerator)
        ):
            ifo_names = [ifo for ifo in self.ifo_list]
        else:
            ifo_names = [ifo.name for ifo in self.ifo_list]
        if isinstance(asd, ASDDataset):
            if set(asd.asds.keys()) != set(ifo_names):
                raise KeyError("ASDDataset ifos do not match signal.")
            if asd.domain.domain_dict != self.data_domain.domain_dict:
                print("Updating ASDDataset domain to match data domain.")
                domain_dict = self.data_domain.domain_dict
                if "window_factor" in domain_dict:
                    print("Dropping window factor for update.")
                    del domain_dict["window_factor"]
                asd.update_domain(domain_dict)
        elif isinstance(asd, dict):
            if set(asd.keys()) != set(ifo_names):
                raise KeyError("ASD ifos do not match signal.")
        elif isinstance(asd, str):
            raise NotImplementedError(
                "Still need to implement injections with ASDs defined by file names."
            )
        self._asd = asd


class Injection(GWSignal):
    """
    Produces injections of signals (with random or specified parameters) into stationary
    Gaussian noise. Output is not whitened.
    """

    def __init__(self, prior, **gwsignal_kwargs):
        """
        Parameters
        ----------
        prior : PriorDict
            Prior used for sampling random parameters.
        gwsignal_kwargs
            Arguments to be passed to GWSignal base class.
        """
        super().__init__(**gwsignal_kwargs)
        self.prior = prior

    @classmethod
    def from_posterior_model_metadata(cls, metadata):
        """
        Instantiate an Injection based on a posterior model. The prior, waveform
        settings, etc., will all be consistent with what the model was trained with.

        Parameters
        ----------
        metadata : dict
            Dict which you can get via PosteriorModel.metadata
        """
        intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})
        if 'lisa_settings' in metadata["train_settings"]["data"].keys():
            lisa_settings = metadata["train_settings"]["data"]["lisa_settings"]
        else:
            lisa_settings = None
        return cls(
            prior=prior,
            wfg_kwargs=metadata["dataset_settings"]["waveform_generator"],
            wfg_domain=build_domain(metadata["dataset_settings"]["domain"]),
            data_domain=build_domain_from_model_metadata(metadata),
            ifo_list=metadata["train_settings"]["data"]["detectors"],
            t_ref=metadata["train_settings"]["data"]["ref_time"],
            lisa_settings = lisa_settings
        )

    def injection(self, theta):
        """
        Generate an injection based on specified parameters.

        This is a signal + noise  consistent with the amplitude spectral density in
        self.asd. If self.asd is an ASDDataset, then it uses a random ASD from this
        dataset.

        Data are not whitened.

        Parameters
        ----------
        theta : dict
            Parameters used for injection.

        Returns
        -------
        dict
            keys:
                waveform: data (signal + noise) in each detector
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        signal = self.signal(theta)
        try:
            # Be careful to use the ASD included with the signal, since each time
            # self.asd is accessed it gives a different ASD (if using an ASD dataset).
            asd = signal["asds"]
        except KeyError:
            raise ValueError("self.asd must be set in order to produce injections.")

        if self.whiten:
            print("self.whiten was set to True. Resetting to False.")
            self.whiten = False

        data = {}
        for ifo, s in signal["waveform"].items():
            noise = (
                (np.random.randn(len(s)) + 1j * np.random.randn(len(s)))
                * asd[ifo]
                * self.data_domain.noise_std
            )
            d = s + noise
            data[ifo] = self.data_domain.update_data(d, low_value=0.0)

        signal["waveform"] = data
        return signal

    def random_injection(self):
        """
        Generate a random injection.

        This is a signal + noise  consistent with the amplitude spectral density in
        self.asd. If self.asd is an ASDDataset, then it uses a random ASD from this
        dataset.

        Data are not whitened.

        Returns
        -------
        dict
            keys:
                waveform: data (signal + noise) in each detector
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta = self.prior.sample()
        theta = {
            k: float(v) for k, v in theta.items()
        }  # Some parameters are np.float64
        return self.injection(theta)
