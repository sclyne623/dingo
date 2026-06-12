from functools import partial
from multiprocessing import Pool
from math import isclose
import time

import numpy as np
import astropy.units as u
from typing import Dict, List, Tuple, Union, Callable
from numbers import Number
import warnings
import pandas as pd

import lal
import lalsimulation as LS

from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    bilby_to_lalsimulation_spins,
)
from bilby.gw.utils import (
    lalsim_SimInspiralWaveformParamsInsertTidalLambda1,
    lalsim_SimInspiralWaveformParamsInsertTidalLambda2,
)

import dingo.gw.waveform_generator.wfg_utils as wfg_utils
import dingo.gw.waveform_generator.frame_utils as frame_utils
from dingo.gw.domains import (
    Domain,
    UniformFrequencyDomain,
    MultibandedFrequencyDomain,
    TimeDomain,
)
from dingo.gw.transforms.waveform_transforms import DecimateAll

import lisabeta.lisa.lisatools as lisatools
import lisabeta.pyconstants as pyconstants
import lisabeta.waveforms.bbh.pyIMRPhenomHM as pyIMRPhenomHM
import lisabeta.waveforms.bbh.pyIMRPhenomXHM as pyIMRPhenomXHM
import lisabeta.waveforms.bbh.pyIMRPhenomD as pyIMRPhenomD
import lisabeta.tools.pytools as pytools

# BBHx imports for LISA gravitational wave waveforms
try:
    from bbhx.waveformbuild import BBHWaveformFD
    from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
    from bbhx.response.fastfdresponse import LISATDIResponse
    from bbhx.utils.constants import PC_SI, YRSID_SI
except ImportError:
    pass  # BBHx not installed



class WaveformGenerator:
    """Generate polarizations using LALSimulation routines in the specified domain for a
    single GW coalescence given a set of waveform parameters.
    """

    def __init__(
        self,
        approximant: str,
        domain: Domain,
        f_ref: float,
        f_start: float = None,
        mode_list: List[Tuple] = None,
        transform=None,
        spin_conversion_phase=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        approximant : str
            Waveform "approximant" string understood by lalsimulation
            This is defines which waveform model is used.
        domain : Domain
            Domain object that specifies on which physical domain the
            waveform polarizations will be generated, e.g. Fourier
            domain, time domain.
        f_ref : float
            Reference frequency for the waveforms
        f_start : float
            Starting frequency for waveform generation. This is optional, and if not
            included, the starting frequency will be set to f_min. This exists so that
            EOB waveforms can be generated starting from a lower frequency than f_min.
        mode_list : List[Tuple]
            A list of waveform (ell, m) modes to include when generating
            the polarizations.
        spin_conversion_phase : float = None
            Value for phiRef when computing cartesian spins from bilby spins via
            bilby_to_lalsimulation_spins. The common convention is to use the value of
            the phase parameter here, which is also used in the spherical harmonics
            when combining the different modes. If spin_conversion_phase = None,
            this default behavior is adapted.
            For dingo, this convention for the phase parameter makes it impossible to
            treat the phase as an extrinsic parameter, since we can only account for
            the change of phase in the spherical harmonics when changing the phase (in
            order to also change the cartesian spins -- specifically, to rotate the spins
            by phase in the sx-sy plane -- one would need to recompute the modes,
            which is expensive).
            By setting spin_conversion_phase != None, we impose the convention to always
            use phase = spin_conversion_phase when computing the cartesian spins.
        """
        if not isinstance(approximant, str):
            raise ValueError("approximant should be a string, but got", approximant)
        else:
            self.approximant_str = approximant
            self.lal_params = None
            if "SEOBNRv5" not in approximant:
                # This LAL function does not work with waveforms using the new interface. TODO: Improve the check.
                self.approximant = LS.GetApproximantFromString(approximant)
                if mode_list is not None:
                    self.lal_params = self.setup_mode_array(mode_list)

        if not issubclass(type(domain), Domain):
            raise ValueError(
                "domain should be an instance of a subclass of Domain, but got",
                type(domain),
            )
        else:
            self.domain = domain

        self.f_ref = f_ref
        self.f_start = f_start

        self.transform = transform
        self._spin_conversion_phase = None
        self.spin_conversion_phase = spin_conversion_phase

    @property
    def domain(self):
        if self._use_base_domain:
            return self._domain.base_domain
        else:
            return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value
        if isinstance(
            self._domain, MultibandedFrequencyDomain
        ) and not LS.SimInspiralImplementedFDApproximants(self.approximant):
            # For non-frequency domain approximants, generate waveforms in the base
            # UniformFrequencyDomain, and later decimate.
            self._use_base_domain = True
            self._domain_transform = DecimateAll(self._domain)
        else:
            # For frequency-domain approximants, generate waveforms directly in either
            # UFD or MFD.
            self._use_base_domain = False
            self._domain_transform = None

    @property
    def full_domain(self):
        return self._domain

    @property
    def spin_conversion_phase(self):
        return self._spin_conversion_phase

    @spin_conversion_phase.setter
    def spin_conversion_phase(self, value):
        if value is None:
            print(
                "Setting spin_conversion_phase = None. Using phase parameter for "
                "conversion to cartesian spins."
            )
        else:
            print(
                f"Setting spin_conversion_phase = {value}. Using this value for the "
                f"phase parameter for conversion to cartesian spins."
            )
        self._spin_conversion_phase = value

    def generate_hplus_hcross(
        self, parameters: Dict[str, float], catch_waveform_errors=True
    ) -> Dict[str, np.ndarray]:
        """Generate GW polarizations (h_plus, h_cross).

        If the generation of the lalsimulation waveform fails with an
        "Input domain error", we return NaN polarizations.

        Use the domain, approximant, and mode_list specified in the constructor
        along with the waveform parameters to generate the waveform polarizations.


        Parameters
        ----------
        parameters: Dict[str, float]
            A dictionary of parameter names and scalar values.
            The parameter dictionary must include the following keys.
            For masses, spins, and distance there are multiple options.

            Mass: (mass_1, mass_2) or a pair of quantities from
                ((chirp_mass, total_mass), (mass_ratio, symmetric_mass_ratio))
            Spin:
                (a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl) if precessing binary or
                (chi_1, chi_2) if the binary has aligned spins
            Reference frequency: f_ref at which spin vectors are defined
            Extrinsic:
                Distance: one of (luminosity_distance, redshift, comoving_distance)
                Inclination: theta_jn
                Reference phase: phase
                Geocentric time: geocent_time (GPS time)
            The following parameters are not required:
                Sky location: ra, dec,
                Polarization angle: psi
            Units:
                Masses should be given in units of solar masses.
                Distance should be given in megaparsecs (Mpc).
                Frequencies should be given in Hz and time in seconds.
                Spins should be dimensionless.
                Angles should be in radians.

        catch_waveform_errors: bool
            Whether to catch lalsimulation errors

        Returns
        -------
        wf_dict:
            A dictionary of generated waveform polarizations
        """
        if not isinstance(parameters, dict):
            raise ValueError("parameters should be a dictionary, but got", parameters)
        elif not isinstance(list(parameters.values())[0], float):
            raise ValueError("parameters dictionary must contain floats", parameters)

        # Include reference frequency with the parameters. Copy the dict first for safety.
        parameters = parameters.copy()
        parameters["f_ref"] = self.f_ref

        parameters_generator, target_function = self._convert_parameters(
            parameters,
            self.lal_params,
            return_target_function=True,
        )

        # Generate GW polarizations
        if isinstance(
            self.domain, (UniformFrequencyDomain, MultibandedFrequencyDomain)
        ):
            wf_generator = self.generate_FD_waveform
        elif isinstance(self.domain, TimeDomain):
            wf_generator = self.generate_TD_waveform
        else:
            raise ValueError(f"Unsupported domain type {type(self.domain)}.")

        try:
            if target_function is not None:
                wf_dict = wf_generator(parameters_generator, target_function)
            else:
                wf_dict = wf_generator(parameters_generator)
        except Exception as e:
            if not catch_waveform_errors:
                raise
            else:
                EDOM = e.args[0] == "Internal function call failed: Input domain error"
                if EDOM:
                    warnings.warn(
                        f"Evaluating the waveform failed with error: {e}\n"
                        f"The parameters were {parameters_generator}\n"
                    )
                    pol_nan = np.ones(len(self.domain), dtype=complex) * np.nan
                    wf_dict = {"h_plus": pol_nan, "h_cross": pol_nan}
                else:
                    raise

        if self._domain_transform is not None:
            wf_dict = self._domain_transform(wf_dict)

        if self.transform is not None:
            return self.transform(wf_dict)
        else:
            return wf_dict

    def _convert_to_scalar(self, x: Union[np.ndarray, float]) -> Number:
        """
        Convert a single element array to a number.

        Parameters
        ----------
        x:
            Array or number

        Returns
        -------
        A number
        """
        if isinstance(x, np.ndarray):
            if x.shape == () or x.shape == (1,):
                return x.item()
            else:
                raise ValueError(
                    f"Expected an array of length one, but shape = {x.shape}"
                )
        else:
            return x

    def _convert_parameters(
        self,
        parameter_dict: Dict,
        lal_params=None,
        target_function=None,
        return_target_function=False,
    ) -> Tuple:
        """Convert to lal source frame parameters

        Parameters
        ----------
        parameter_dict : Dict
            A dictionary of parameter names and 1-dimensional prior distribution
            objects. If None, we use a default binary black hole prior.
        lal_params : (None, or Swig Object of type 'tagLALDict *')
            Extra parameters which can be passed to lalsimulation calls.
        target_function: str = None
            Name of the lalsimulation function for which to prepare the parameters.
            If None, use SimInspiralFD if self.domain is FD, and SimInspiralTD if
            self.domain is TD.
            Choices:
                - SimInspiralFD (Also works for SimInspiralChooseFDWaveform)
                - SimInspiralTD (Also works for SimInspiralChooseTDWaveform)
                - SimInspiralChooseFDModes
                - SimInspiralChooseTDModes
        return_target_function: bool = False
            if set, also returns lal target function.
        Returns
        -------
        lal_parameter_tuple:
            A tuple of parameters for the lalsimulation waveform generator.
        target_function:
            Target function for waveform generation, only returned if
            return_target_function = True.
        """
        # check that the target_function is valid
        if target_function is None:
            if isinstance(self.domain, UniformFrequencyDomain):
                target_function = "SimInspiralFD"
            elif isinstance(self.domain, MultibandedFrequencyDomain):
                target_function = "SimInspiralChooseFDWaveformSequence"
            elif isinstance(self.domain, TimeDomain):
                target_function = "SimInspiralTD"
            else:
                raise ValueError(f"Unsupported domain type {type(self.domain)}.")
        target_functions_dict = {
            "SimInspiralFD": LS.SimInspiralFD,
            "SimInspiralTD": LS.SimInspiralTD,
            "SimInspiralChooseTDModes": LS.SimInspiralChooseTDModes,
            "SimInspiralChooseFDModes": LS.SimInspiralChooseFDModes,
            "SimInspiralChooseFDWaveformSequence": LS.SimInspiralChooseFDWaveformSequence,
            "SimIMRPhenomXPCalculateModelParametersFromSourceFrame": LS.SimIMRPhenomXPCalculateModelParametersFromSourceFrame,
        }
        if target_function not in target_functions_dict:
            raise ValueError(
                f"Unsupported lalsimulation waveform function {target_function}."
            )

        # Transform mass, spin, and distance parameters
        p, _ = convert_to_lal_binary_black_hole_parameters(parameter_dict)

        # Convert to SI units
        p["mass_1"] *= lal.MSUN_SI
        p["mass_2"] *= lal.MSUN_SI
        p["luminosity_distance"] *= 1e6 * lal.PC_SI

        # Transform to lal source frame: iota and Cartesian spin components
        param_keys_in = (
            "theta_jn",
            "phi_jl",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "a_1",
            "a_2",
            "mass_1",
            "mass_2",
            "f_ref",
            "phase",
        )
        param_values_in = [p[k] for k in param_keys_in]
        # if spin_conversion_phase is set, use this as fixed phiRef when computing the
        # cartesian spins instead of using the phase parameter
        if self.spin_conversion_phase is not None:
            param_values_in[-1] = self.spin_conversion_phase
        iota_and_cart_spins = bilby_to_lalsimulation_spins(*param_values_in)
        iota, s1x, s1y, s1z, s2x, s2y, s2z = [
            float(self._convert_to_scalar(x)) for x in iota_and_cart_spins
        ]

        # Construct argument list for FD and TD lal waveform generator wrappers
        spins_cartesian = s1x, s1y, s1z, s2x, s2y, s2z
        masses = (p["mass_1"], p["mass_2"])
        r = p["luminosity_distance"]
        phase = p["phase"]
        ecc_params = (0.0, 0.0, 0.0)  # longAscNodes, eccentricity, meanPerAno
        # for BNS/NSBH: insert tidal deformability
        if "lambda_1" in p or "lambda_2" in p:
            if lal_params is None:
                lal_params = lal.CreateDict()
            lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
                lal_params, p.get("lambda_1", 0)
            )
            lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
                lal_params, p.get("lambda_2", 0)
            )

        # Get domain parameters
        f_ref = p["f_ref"]
        if isinstance(self.domain, UniformFrequencyDomain):
            delta_f = self.domain.delta_f
            f_max = self.domain.f_max
            if self.f_start is not None:
                f_min = self.f_start
            else:
                f_min = self.domain.f_min
            # parameters needed for TD waveforms
            delta_t = 0.5 / self.domain.f_max
        elif isinstance(self.domain, TimeDomain):
            raise NotImplementedError("Time domain not supported yet.")
            # FIXME: compute f_min from duration or specify it if SimInspiralTD
            #  is used for a native FD waveform
            f_min = 20.0
            delta_t = self.domain.delta_t
            # parameters needed for FD waveforms
            f_max = 1.0 / self.domain.delta_t
            delta_f = 1.0 / self.domain.duration

        if target_function == "SimInspiralFD":
            # LS.SimInspiralFD takes parameters:
            #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
            #   distance, inclination, phiRef,
            #   longAscNodes, eccentricity, meanPerAno,
            #   deltaF, f_min, f_max, f_ref,
            #   lal_params, approximant
            domain_pars = (delta_f, f_min, f_max, f_ref)
            domain_pars = tuple(float(p) for p in domain_pars)
            lal_parameter_tuple = (
                masses
                + spins_cartesian
                + (r, iota, phase)
                + ecc_params
                + domain_pars
                + (lal_params, self.approximant)
            )
            lal_parameter_tuple = (
                tuple(float(p) for p in lal_parameter_tuple[:18])
                + lal_parameter_tuple[18:]
            )

        elif target_function == "SimInspiralChooseFDWaveformSequence":
            # LS.SimInspiralChooseFDWaveformSequence takes parameters:
            #   phiRef, m1, m2, S1x, S1y, S1z, S2x, S2y, S2z
            #   f_ref, distance, iota,
            #   lal_params, approximant, frequency_array
            lal_parameter_tuple = (phase, *masses, *spins_cartesian, f_ref, r, iota)
            lal_parameter_tuple = tuple(float(p) for p in lal_parameter_tuple)
            # create lal object for frequency array
            frequency_array = lal.CreateREAL8Vector(
                len(self.domain()[self.domain.min_idx :])
            )
            frequency_array.data = self.domain()[self.domain.min_idx :]
            lal_parameter_tuple = (
                *lal_parameter_tuple,
                lal_params,
                self.approximant,
                frequency_array,
            )

        elif target_function == "SimInspiralTD":
            # LS.SimInspiralTD takes parameters:
            #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
            #   distance, inclination, phiRef,
            #   longAscNodes, eccentricity, meanPerAno,
            #   delta_t, f_min, f_ref
            #   lal_params, approximant
            domain_pars = (delta_t, f_min, f_ref)
            domain_pars = tuple(float(p) for p in domain_pars)
            lal_parameter_tuple = (
                masses
                + spins_cartesian
                + (r, iota, phase)
                + ecc_params
                + domain_pars
                + (lal_params, self.approximant)
            )
        elif target_function == "SimInspiralChooseFDModes":
            domain_pars = (delta_f, f_min, f_max, f_ref)
            domain_pars = tuple(float(p) for p in domain_pars)
            lal_parameter_tuple = (
                masses
                + spins_cartesian
                + domain_pars
                + (phase, r, iota)
                + (lal_params, self.approximant)
            )

        elif target_function == "SimIMRPhenomXPCalculateModelParametersFromSourceFrame":
            lal_parameter_tuple = (
                masses + (f_ref,) + (phase, iota) + spins_cartesian + (lal_params,)
            )

        elif target_function == "SimInspiralChooseTDModes":
            # LS.SimInspiralChooseTDModes takes parameters:
            #   phiRef=0 (for lal legacy reasons), delta_t,
            #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
            #   f_min, f_ref
            #   distance,
            #   lal_params, l_max, approximant
            domain_pars = (delta_t, f_min, f_ref)
            domain_pars = tuple(float(p) for p in domain_pars)
            if "l_max" not in parameter_dict:
                l_max = 5  # hard code l_max for now
            lal_parameter_tuple = (
                (
                    0.0,
                    domain_pars[0],
                )  # domain_pars[0] = delta_t
                + masses
                + spins_cartesian
                + domain_pars[1:]  # domain_pars[1:] = f_min, f_ref
                + (r,)
                + (lal_params, l_max, self.approximant)
            )
            # also pass iota, since this is needed for recombination of the modes
            lal_parameter_tuple = (lal_parameter_tuple, iota)

        if return_target_function:
            return lal_parameter_tuple, target_functions_dict[target_function]
        else:
            return lal_parameter_tuple

    def setup_mode_array(self, mode_list: List[Tuple]) -> lal.Dict:
        """Define a mode array to select waveform modes
        to include in the polarizations from a list of modes.

        Parameters
        ----------
        mode_list : a list of (ell, m) modes

        Returns
        -------
        lal_params:
            A lal parameter dictionary
        """
        lal_params = lal.CreateDict()
        ma = LS.SimInspiralCreateModeArray()
        for ell, m in mode_list:
            LS.SimInspiralModeArrayActivateMode(ma, ell, m)
            # LS.SimInspiralModeArrayActivateMode(ma, ell, -m)
        LS.SimInspiralWaveformParamsInsertModeArray(lal_params, ma)
        return lal_params

    def generate_FD_waveform(
        self,
        parameters_lal: Tuple,
        target_function: Callable,
    ) -> Dict[str, np.ndarray]:
        """
        Generate Fourier domain GW polarizations (h_plus, h_cross).

        Parameters
        ----------
        parameters_lal:
            A tuple of parameters for the lalsimulation waveform generator
        target_function:
            Lalsimulation function for waveform generation.

        Returns
        -------
        pol_dict:
            A dictionary of generated waveform polarizations
        """
        # Note: SEOBNRv4PHM does not support the specification of spins at a
        # reference frequency different from the starting frequency. In addition,
        # waveform generation will fail if the orbital distance is smaller than ~ 10M.
        # To avoid this, we can start at a sufficiently low and consistent starting frequency
        # for the entire dataset. If the number of generation failures is a very small
        # fraction over the prior distribution then the dataset should be good to use.
        #
        # Note: XLALSimInspiralFD() internally calls XLALSimInspiralTD() to generate
        # a conditioned time-domain waveform. In the past, this function lowered
        # the starting frequency, but this is thankfully no longer the case
        # for models such as SEOBNRv4PHM where the reference frequency is equal
        # to the starting frequency. So, the TD waveform will be generated by
        # calling XLALSimInspiralChooseTDWaveform().
        # See https://git.ligo.org/waveforms/reviews/lalsuite/-/commit/195f9127682de19f5fce19cc5828116dd2d23461
        #
        # LS.SimInspiralFD takes parameters:
        #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
        #   distance, inclination, phiRef,
        #   longAscNodes, eccentricity, meanPerAno,
        #   deltaF, f_min, f_max, f_ref,
        #   lal_params, approximant

        # call the lalsimulation waveform generation function. For uniform frequency
        #   UniformFrequencyDomain:                LS.SimInspiralFD
        #   MultibandedFrequencyDomain:     SimInspiralChooseFDWaveformSequence
        hp, hc = target_function(*parameters_lal)

        # The check below filters for unphysical waveforms:
        # For IMRPhenomXPHM, the LS.SimInspiralFD result is numerically instable
        # for rare parameter configurations (~1 in 1M), leading to bins with very large
        # numbers if multibanding is used. If that happens, turn off multibanding to
        # fix this.
        if max(np.max(np.abs(hp.data.data)), np.max(np.abs(hc.data.data))) > 1e-20:
            print(
                f"Generation with parameters {parameters_lal} likely numerically "
                f"unstable due to multibanding, turn off multibanding."
            )
            if target_function == LS.SimInspiralFD:
                lal_dict_idx = 18
            elif target_function == LS.SimInspiralChooseFDWaveformSequence:
                lal_dict_idx = 12
            else:
                raise NotImplementedError(
                    f"Unsupported lal target function " f"{target_function}."
                )
            lal_dict = parameters_lal[lal_dict_idx]
            if lal_dict is None:
                lal_dict = lal.CreateDict()
            LS.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lal_dict, 0)
            LS.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lal_dict, 0)
            hp, hc = target_function(
                *parameters_lal[:lal_dict_idx],
                lal_dict,
                *parameters_lal[lal_dict_idx + 1 :],
            )
            if max(np.max(np.abs(hp.data.data)), np.max(np.abs(hc.data.data))) > 1e-20:
                print(
                    f"Warning: turning off multibanding for parameters {parameters_lal}"
                    f" likely numerically might not have fixed it, check manually."
                )

        # Postprocessing, specific for target_function.
        # For LS.SimInspiralFD, this includes sanity checks and possibly truncation.
        if target_function == LS.SimInspiralFD:
            # Ensure that the waveform agrees with the frequency grid defined in the domain.
            if not isclose(self.domain.delta_f, hp.deltaF, rel_tol=1e-6):
                raise ValueError(
                    f"Waveform delta_f is inconsistent with domain: {hp.deltaF} vs {self.domain.delta_f}!"
                    f"To avoid this, ensure that f_max = {self.domain.f_max} is a power of two"
                    "when you are using a native time-domain waveform model."
                )

            frequency_array = self.domain()
            h_plus = np.zeros_like(frequency_array, dtype=complex)
            h_cross = np.zeros_like(frequency_array, dtype=complex)
            # Ensure that length of wf agrees with length of domain. Enforce by truncating frequencies beyond f_max
            if len(hp.data.data) > len(frequency_array):
                warnings.warn(
                    "LALsimulation waveform longer than domain's `frequency_array`"
                    f"({len(hp.data.data)} vs {len(frequency_array)}). Truncating lalsim array."
                )
                h_plus = hp.data.data[: len(h_plus)]
                h_cross = hc.data.data[: len(h_cross)]
            else:
                h_plus[: len(hp.data.data)] = hp.data.data
                h_cross[: len(hc.data.data)] = hc.data.data

            # Undo the time shift done in SimInspiralFD to the waveform
            dt = 1 / hp.deltaF + (hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds * 1e-9)
            time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array)
            h_plus *= time_shift
            h_cross *= time_shift
            return {"h_plus": h_plus, "h_cross": h_cross}

        elif target_function == LS.SimInspiralChooseFDWaveformSequence:
            frequency_array = self.domain()[self.domain.min_idx :]
            h_plus = np.zeros_like(frequency_array, dtype=complex)
            h_cross = np.zeros_like(frequency_array, dtype=complex)
            h_plus[:] = hp.data.data[:]
            h_cross[:] = hc.data.data[:]
            return {"h_plus": h_plus, "h_cross": h_cross}

        else:
            raise NotImplementedError(
                f"Unsupported lal target function " f"{target_function}."
            )

    def generate_hplus_hcross_m(
        self, parameters: Dict[str, float]
    ) -> Dict[tuple, Dict[str, np.ndarray]]:
        """
        Generate GW polarizations (h_plus, h_cross), separated into contributions from
        the different modes. This method is identical to self.generate_hplus_hcross,
        except that it generates the individual contributions of the modes to the
        polarizations and sorts these according to their transformation behavior (see
        below), instead of returning the overall sum.

        This is useful in order to treat the phase as an extrinsic parameter. Instead of
        {"h_plus": hp, "h_cross": hc}, this method returns a dict in the form of
        {m: {"h_plus": hp_m, "h_cross": hc_m} for m in [-l_max,...,0,...,l_max]}. Each
        key m contains the contribution to the polarization that transforms according
        to exp(-1j * m * phase) under phase transformations (due to the spherical
        harmonics).

        Note:
            - pol_m[m] contains contributions of the m modes *and* and the -m modes.
              This is because the frequency domain (FD) modes have a positive frequency
              part which transforms as exp(-1j * m * phase), while the negative
              frequency part transforms as exp(+1j * m * phase). Typically, one of these
              dominates [e.g., the (2,2) mode is dominated by the negative frequency
              part and the (-2,2) mode is dominated by the positive frequency part]
              such that the sum of (l,|m|) and (l,-|m|) modes transforms approximately as
              exp(1j * |m| * phase), which is e.g. used for phase marginalization in
              bilby/lalinference. However, this is not exact. In this method we account
              for this effect, such that each contribution pol_m[m] transforms
              *exactly* as exp(-1j * m * phase).
            - Phase shifts contribute in two ways: Firstly via the spherical harmonics,
              which we account for with the exp(-1j * m * phase) transformation.
              Secondly, the phase determines how the PE spins transform to cartesian
              spins, by rotating (sx,sy) by phase. This is *not* accounted for in this
              function. Instead, the phase for computing the cartesian spins is fixed
              to self.spin_conversion_phase (if not None). This effectively changes the
              PE parameters {phi_jl, phi_12} to parameters {phi_jl_prime, phi_12_prime}.
              For parameter estimation, a postprocessing operation can be applied to
              account for this, {phi_jl_prime, phi_12_prime} -> {phi_jl, phi_12}.
              See also documentation of __init__ method for more information on
              self.spin_conversion_phase.

        Differences to self.generate_hplus_hcross:
        - We don't catch errors yet TODO
        - We don't apply transforms yet TODO

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters for the waveform.
            For details see see self.generate_hplus_hcross.

        Returns
        -------
        pol_m: dict
            Dictionary with contributions to h_plus and h_cross, sorted by their
            transformation behaviour under phase shifts:
            {m: {"h_plus": hp_m, "h_cross": hc_m} for m in [-l_max,...,0,...,l_max]}
            Each contribution h_m transforms as exp(-1j * m * phase) under phase shifts
            (for fixed self.spin_conversion_phase, see above).
        """
        if not isinstance(parameters, dict):
            raise ValueError("parameters should be a dictionary, but got", parameters)
        elif not isinstance(list(parameters.values())[0], float):
            raise ValueError("parameters dictionary must contain floats", parameters)

        if isinstance(self.domain, UniformFrequencyDomain):
            # Generate FD modes in for frequencies [-f_max, ..., 0, ..., f_max].
            if LS.SimInspiralImplementedFDApproximants(self.approximant):
                # Step 1: generate waveform modes in L0 frame in native domain of
                # approximant (here: FD)
                hlm_fd, iota = self.generate_FD_modes_LO(parameters)

                # Step 2: Transform modes to target domain.
                # Not required here, as approximant domain and target domain are both FD.

            else:
                assert LS.SimInspiralImplementedTDApproximants(self.approximant)
                # Step 1: generate waveform modes in L0 frame in native domain of
                # approximant (here: TD)
                hlm_td, iota = self.generate_TD_modes_L0(parameters)

                # Step 2: Transform modes to target domain.
                # This requires tapering of TD modes, and FFT to transform to FD.
                wfg_utils.taper_td_modes_in_place(hlm_td)
                hlm_fd = wfg_utils.td_modes_to_fd_modes(hlm_td, self.domain)

            # Step 3: Separate negative and positive frequency parts of the modes,
            # and add contributions according to their transformation behavior under
            # phase shifts.
            pol_m = wfg_utils.get_polarizations_from_fd_modes_m(
                hlm_fd, iota, parameters["phase"]
            )
            for h in pol_m.values():
                # Ensure that length of wf agrees with length of domain. Enforce by
                # truncating frequencies beyond f_max
                if len(h["h_plus"]) > len(self.domain):
                    warnings.warn(
                        "LALsimulation waveform longer than domain's `frequency_array`"
                        f"({len(h['h_plus'])} vs {len(self.domain)}). Truncating "
                        f"lalsim array."
                    )
                    h["h_plus"] = h["h_plus"][: len(self.domain)]
                    h["h_cross"] = h["h_cross"][: len(self.domain)]

        elif isinstance(self.domain, MultibandedFrequencyDomain):
            if LS.SimInspiralImplementedFDApproximants(self.approximant):
                # SimInspiralChooseFDModes does not work with multi-banding. Hence,
                # temporarily switch from MFD to FD, generate the modes, decimate to MFD,
                # and reset the domain to MFD.

                self._use_base_domain = True
                self._domain_transform = DecimateAll(self._domain)
                hlm_fd, iota = self.generate_FD_modes_LO(parameters)
                pol_m = wfg_utils.get_polarizations_from_fd_modes_m(
                    hlm_fd, iota, parameters["phase"]
                )
                for h in pol_m.values():
                    # Ensure that length of wf agrees with length of domain. Enforce by
                    # truncating frequencies beyond f_max
                    if len(h["h_plus"]) > len(self.domain):
                        warnings.warn(
                            "LALsimulation waveform longer than domain's `frequency_array`"
                            f"({len(h['h_plus'])} vs {len(self.domain)}). Truncating "
                            f"lalsim array."
                        )
                        h["h_plus"] = h["h_plus"][: len(self.domain)]
                        h["h_cross"] = h["h_cross"][: len(self.domain)]
                pol_m = self._domain_transform(pol_m)
                self.domain = self.full_domain

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError(
                f"Target domain of type {type(self.domain)} not yet implemented."
            )

        if self._domain_transform is not None:
            return self._domain_transform(pol_m)
        else:
            return pol_m

    def generate_FD_modes_LO(self, parameters):
        """
        Generate FD modes in the L0 frame.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters for the waveform.
            For details see see self.generate_hplus_hcross.

        Returns
        -------
        hlm_fd: dict
            Dictionary with (l,m) as keys and the corresponding FD modes in lal format as
            values.
        iota: float
        """
        # TD approximants that are implemented in J frame. Currently tested for:
        #   101: IMRPhenomXPHM
        if self.approximant in [101]:
            parameters_lal_fd_modes = self._convert_parameters(
                {**parameters, "f_ref": self.f_ref},
                target_function="SimInspiralChooseFDModes",
            )
            iota = parameters_lal_fd_modes[14]
            hlm_fd = LS.SimInspiralChooseFDModes(*parameters_lal_fd_modes)
            # unpack linked list, convert lal objects to arrays
            hlm_fd = wfg_utils.linked_list_modes_to_dict_modes(hlm_fd)
            hlm_fd = {k: v.data.data for k, v in hlm_fd.items()}
            # For the waveform models considered here (e.g., IMRPhenomXPHM), the modes
            # are returned in the J frame (where the observer is at inclination=theta_JN,
            # azimuth=0). In this frame, the dependence on the reference phase enters
            # via the modes themselves. We need to convert to the L0 frame so that the
            # dependence on phase enters via the spherical harmonics.
            hlm_fd = frame_utils.convert_J_to_L0_frame(
                hlm_fd,
                parameters,
                self,
                spin_conversion_phase=self.spin_conversion_phase,
            )
            return hlm_fd, iota
        else:
            raise NotImplementedError(
                f"Approximant {self.approximant_str} not "
                f"implemented. When adding this approximant to this method, make sure "
                f"the the output dict hlm_td contains the TD modes in the *L0 frame*. "
                f"In particular, adding an approximant that is implemented in the same "
                f"domain and frame as one of the approximants should just be a matter of "
                f"adding the approximant number (here: {self.approximant}) to the "
                f"corresponding if statement. However, when doing this please make sure "
                f"to test that this works as intended! Ideally, add some unit tests."
            )

    def generate_TD_modes_L0(self, parameters):
        """
        Generate TD modes in the L0 frame.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters for the waveform.
            For details see see self.generate_hplus_hcross.

        Returns
        -------
        hlm_td: dict
            Dictionary with (l,m) as keys and the corresponding TD modes in lal format as
            values.
        iota: float
        """
        # TD approximants that are implemented in L0 frame. Currently tested for:
        #   52: SEOBNRv4PHM
        if self.approximant in [52]:
            parameters_lal_td_modes, iota = self._convert_parameters(
                {**parameters, "f_ref": self.f_ref},
                target_function="SimInspiralChooseTDModes",
            )
            hlm_td = LS.SimInspiralChooseTDModes(*parameters_lal_td_modes)
            return wfg_utils.linked_list_modes_to_dict_modes(hlm_td), iota
        else:
            raise NotImplementedError(
                f"Approximant {LS.GetApproximantFromString(self.approximant)} not "
                f"implemented. When adding this approximant to this method, make sure "
                f"the the output dict hlm_td contains the TD modes in the *L0 frame*. "
                f"In particular, adding an approximant that is implemented in the same "
                f"domain and frame as one of the approximants should just be a matter of "
                f"adding the approximant number (here: {self.approximant}) to the "
                f"corresponding if statement. However, when doing this please make sure "
                f"to test that this works as intended! Ideally, add some unit tests."
            )

    def generate_TD_waveform(self, parameters_lal: Tuple) -> Dict[str, np.ndarray]:
        """
        Generate time domain GW polarizations (h_plus, h_cross)

        Parameters
        ----------
        parameters_lal:
            A tuple of parameters for the lalsimulation waveform generator

        Returns
        -------
        pol_dict:
            A dictionary of generated waveform polarizations
        """
        # Note: XLALSimInspiralTD() now calls XLALSimInspiralChooseTDWaveform()
        # for models such as SEOBNRv4PHM where the reference frequency is equal
        # to the starting frequency and thus leaves our choice of starting
        # frequency untouched.
        #
        # LS.SimInspiralTD takes parameters:
        #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
        #   distance, inclination, phiRef,
        #   longAscNodes, eccentricity, meanPerAno,
        #   deltaT, f_min, f_ref
        #   lal_params, approximant

        hp, hc = LS.SimInspiralTD(*parameters_lal)
        h_plus = hp.data.data
        h_cross = hc.data.data
        pol_dict = {"h_plus": h_plus, "h_cross": h_cross}
        return pol_dict


class NewInterfaceWaveformGenerator(WaveformGenerator):
    """Generate polarizations using GWSignal routines in the specified domain for a
    single GW coalescence given a set of waveform parameters.
    """

    def __init__(self, **kwargs):
        WaveformGenerator.__init__(self, **kwargs)

        # we want to import the new interface, but we don't want to import it
        # for all users because just importing the module causes unintended side effects
        # for example, you cannot pass the debugger through the import statement.
        # Thus we only import it when the class is instantiated.
        # However, we don't want to reimport the module every time we need to
        # use the function as this is costly. Therefore, we import it once
        # when the class is instantiated and store it in the global namespace
        from lalsimulation.gwsignal.core import waveform
        from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

        globals()["gws_wfm"] = waveform
        globals()[
            "new_interface_get_waveform_generator"
        ] = gwsignal_get_waveform_generator

        self.mode_list = kwargs.get("mode_list", None)

    @property
    def domain(self):
        if self._use_base_domain:
            return self._domain.base_domain
        else:
            return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value
        if isinstance(self._domain, MultibandedFrequencyDomain):
            # If using the MultibandedFrequencyDomain and an approximant implemented
            # in gwsignal, assume it's a time-domain approximant and decimate after
            # generating the waveform.
            self._use_base_domain = True
            self._domain_transform = DecimateAll(self._domain)
        else:
            self._use_base_domain = False
            self._domain_transform = None

    def _convert_parameters(
        self,
        parameter_dict: Dict,
        lal_params=None,
        target_function=None,
        return_target_function=False,
    ):
        # Transform mass, spin, and distance parameters
        p, _ = convert_to_lal_binary_black_hole_parameters(parameter_dict)

        # Transform to lal source frame: iota and Cartesian spin components
        param_keys_in = (
            "theta_jn",
            "phi_jl",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "a_1",
            "a_2",
            "mass_1",
            "mass_2",
            "f_ref",
            "phase",
        )
        param_values_in = [p[k] for k in param_keys_in]

        # Masses for spin conversion must be in SI units. However, for waveform generation, they must remain in solar
        # masses due to sensitive dependence of SEOBNRv5 waveforms to small changes in the mass. Hence, we only convert
        # units here.
        param_values_in[7] *= lal.MSUN_SI
        param_values_in[8] *= lal.MSUN_SI

        # if spin_conversion_phase is set, use this as fixed phiRef when computing the
        # cartesian spins instead of using the phase parameter
        if self.spin_conversion_phase is not None:
            param_values_in[-1] = self.spin_conversion_phase
        iota_and_cart_spins = bilby_to_lalsimulation_spins(*param_values_in)
        iota, s1x, s1y, s1z, s2x, s2y, s2z = [
            float(self._convert_to_scalar(x)) for x in iota_and_cart_spins
        ]

        f_ref = p["f_ref"]
        delta_f = self.domain.delta_f
        f_max = self.domain.f_max
        if self.f_start is not None:
            f_min = self.f_start
        else:
            f_min = self.domain.f_min
        # parameters needed for TD waveforms
        delta_t = 0.5 / self.domain.f_max

        params_gwsignal = {
            "mass1": p["mass_1"] * u.solMass,
            "mass2": p["mass_2"] * u.solMass,
            "spin1x": s1x * u.dimensionless_unscaled,
            "spin1y": s1y * u.dimensionless_unscaled,
            "spin1z": s1z * u.dimensionless_unscaled,
            "spin2x": s2x * u.dimensionless_unscaled,
            "spin2y": s2y * u.dimensionless_unscaled,
            "spin2z": s2z * u.dimensionless_unscaled,
            "deltaT": delta_t * u.s,
            "f22_start": f_min * u.Hz,
            "f22_ref": f_ref * u.Hz,
            "f_max": f_max * u.Hz,
            "deltaF": delta_f * u.Hz,
            "phi_ref": p["phase"] * u.rad,
            "distance": p["luminosity_distance"] * u.Mpc,
            "inclination": iota * u.rad,
            "ModeArray": self.mode_list,
            "condition": 1,
        }

        # SEOBNRv5 specific parameters
        if "postadiabatic" in p:
            params_gwsignal["postadiabatic"] = p["postadiabatic"]

            if "postadiabatic_type" in p:
                params_gwsignal["postadiabatic_type"] = p["postadiabatic_type"]

        if "lmax_nyquist" in p:
            params_gwsignal["lmax_nyquist"] = p["lmax_nyquist"]
        else:
            params_gwsignal["lmax_nyquist"] = 2

        if return_target_function:
            # This is a hack to make compatible with LAL version. Target functions for
            # new waveform generator are defined in generate_FD_waveform, etc.
            # TODO: Revamp this whole module.
            return params_gwsignal, None
        else:
            return params_gwsignal

    def generate_FD_waveform(self, parameters_gwsignal: Dict) -> Dict[str, np.ndarray]:
        """
        Generate Fourier domain GW polarizations (h_plus, h_cross).

        Parameters
        ----------
        parameters_lal:
            A tuple of parameters for the lalsimulation waveform generator

        Returns
        -------
        pol_dict:
            A dictionary of generated waveform polarizations
        """
        # Note: SEOBNRv4PHM does not support the specification of spins at a
        # reference frequency different from the starting frequency. In addition,
        # waveform generation will fail if the orbital distance is smaller than ~ 10M.
        # To avoid this, we can start at a sufficiently low and consistent starting frequency
        # for the entire dataset. If the number of generation failures is a very small
        # fraction over the prior distribution then the dataset should be good to use.
        #
        # Note: XLALSimInspiralFD() internally calls XLALSimInspiralTD() to generate
        # a conditioned time-domain waveform. In the past, this function lowered
        # the starting frequency, but this is thankfully no longer the case
        # for models such as SEOBNRv4PHM where the reference frequency is equal
        # to the starting frequency. So, the TD waveform will be generated by
        # calling XLALSimInspiralChooseTDWaveform().
        # See https://git.ligo.org/waveforms/reviews/lalsuite/-/commit/195f9127682de19f5fce19cc5828116dd2d23461
        #
        # LS.SimInspiralFD takes parameters:
        #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
        #   distance, inclination, phiRef,
        #   longAscNodes, eccentricity, meanPerAno,
        #   deltaF, f_min, f_max, f_ref,
        #   lal_params, approximant

        # Sanity check types of arguments
        # check_floats = all(map(lambda x: isinstance(x, float), parameters_lal[:18]))
        # check_int = isinstance(parameters_lal[19], int)
        # parameters_lal[18]  # lal_params could be None or a LALDict
        # if not (check_floats and check_int):
        #    raise ValueError(
        #        "SimInspiralFD received invalid argument(s)", parameters_lal
        #    )

        # Depending on whether the domain is uniform or non-uniform call the appropriate wf generator
        generator = new_interface_get_waveform_generator(self.approximant_str)
        hpc = gws_wfm.GenerateFDWaveform(parameters_gwsignal, generator)
        hp = hpc.hp
        hc = hpc.hc

        # Ensure that the waveform agrees with the frequency grid defined in the domain.
        if not isclose(self.domain.delta_f, hp.df.value, rel_tol=1e-6):
            raise ValueError(
                f"Waveform delta_f is inconsistent with domain: {hp.df.value} vs {self.domain.delta_f}!"
                f"To avoid this, ensure that f_max = {self.domain.f_max} is a power of two"
                "when you are using a native time-domain waveform model."
            )

        frequency_array = self.domain()
        h_plus = np.zeros_like(frequency_array, dtype=complex)
        h_cross = np.zeros_like(frequency_array, dtype=complex)
        # Ensure that length of wf agrees with length of domain. Enforce by truncating frequencies beyond f_max
        if len(hp) > len(frequency_array):
            warnings.warn(
                "GWSignal waveform longer than domain's `frequency_array`"
                f"({len(hp)} vs {len(frequency_array)}). Truncating gwsignal array."
            )
            h_plus = hp[: len(h_plus)].value
            h_cross = hc[: len(h_cross)].value
        else:
            h_plus = hp.value
            h_cross = hc.value

        # Undo the time shift done in SimInspiralFD to the waveform
        dt = 1 / hp.df.value + hp.epoch.value
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array)
        h_plus *= time_shift
        h_cross *= time_shift
        pol_dict = {"h_plus": h_plus, "h_cross": h_cross}
        return pol_dict

    def generate_hplus_hcross_m(
        self, parameters: Dict[str, float]
    ) -> Dict[tuple, Dict[str, np.ndarray]]:
        """
        Generate GW polarizations (h_plus, h_cross), separated into contributions from
        the different modes. This method is identical to self.generate_hplus_hcross,
        except that it generates the individual contributions of the modes to the
        polarizations and sorts these according to their transformation behavior (see
        below), instead of returning the overall sum.

        This is useful in order to treat the phase as an extrinsic parameter. Instead of
        {"h_plus": hp, "h_cross": hc}, this method returns a dict in the form of
        {m: {"h_plus": hp_m, "h_cross": hc_m} for m in [-l_max,...,0,...,l_max]}. Each
        key m contains the contribution to the polarization that transforms according
        to exp(-1j * m * phase) under phase transformations (due to the spherical
        harmonics).

        Note:
            - pol_m[m] contains contributions of the m modes *and* and the -m modes.
              This is because the frequency domain (FD) modes have a positive frequency
              part which transforms as exp(-1j * m * phase), while the negative
              frequency part transforms as exp(+1j * m * phase). Typically, one of these
              dominates [e.g., the (2,2) mode is dominated by the negative frequency
              part and the (-2,2) mode is dominated by the positive frequency part]
              such that the sum of (l,|m|) and (l,-|m|) modes transforms approximately as
              exp(1j * |m| * phase), which is e.g. used for phase marginalization in
              bilby/lalinference. However, this is not exact. In this method we account
              for this effect, such that each contribution pol_m[m] transforms
              *exactly* as exp(-1j * m * phase).
            - Phase shifts contribute in two ways: Firstly via the spherical harmonics,
              which we account for with the exp(-1j * m * phase) transformation.
              Secondly, the phase determines how the PE spins transform to cartesian
              spins, by rotating (sx,sy) by phase. This is *not* accounted for in this
              function. Instead, the phase for computing the cartesian spins is fixed
              to self.spin_conversion_phase (if not None). This effectively changes the
              PE parameters {phi_jl, phi_12} to parameters {phi_jl_prime, phi_12_prime}.
              For parameter estimation, a postprocessing operation can be applied to
              account for this, {phi_jl_prime, phi_12_prime} -> {phi_jl, phi_12}.
              See also documentation of __init__ method for more information on
              self.spin_conversion_phase.

        Differences to self.generate_hplus_hcross:
        - We don't catch errors yet TODO
        - We don't apply transforms yet TODO

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters for the waveform.
            For details see see self.generate_hplus_hcross.

        Returns
        -------
        pol_m: dict
            Dictionary with contributions to h_plus and h_cross, sorted by their
            transformation behaviour under phase shifts:
            {m: {"h_plus": hp_m, "h_cross": hc_m} for m in [-l_max,...,0,...,l_max]}
            Each contribution h_m transforms as exp(-1j * m * phase) under phase shifts
            (for fixed self.spin_conversion_phase, see above).
        """
        if not isinstance(parameters, dict):
            raise ValueError("parameters should be a dictionary, but got", parameters)
        elif not isinstance(list(parameters.values())[0], float):
            raise ValueError("parameters dictionary must contain floats", parameters)

        generator = new_interface_get_waveform_generator(self.approximant_str)
        if isinstance(self.domain, UniformFrequencyDomain):
            # Generate FD modes in for frequencies [-f_max, ..., 0, ..., f_max].
            if generator.domain == "freq":
                # Step 1: generate waveform modes in L0 frame in native domain of
                # approximant (here: FD)
                hlm_fd, iota = self.generate_FD_modes_LO(parameters)

                # Step 2: Transform modes to target domain.
                # Not required here, as approximant domain and target domain are both FD.

            elif (
                self.approximant_str == "SEOBNRv5PHM"
                or self.approximant_str == "SEOBNRv5HM"
            ):
                # Step 1: generate waveform modes in L0 frame in native domain of
                # approximant (here: TD), applying standard conditioning
                hlm_td, iota = self.generate_TD_modes_L0_conditioned_extra_time(
                    parameters
                )

                # Step 2: Transform modes to target domain.
                hlm_fd = wfg_utils.td_modes_to_fd_modes(hlm_td, self.domain)
            else:
                # assert LS.SimInspiralImplementedTDApproximants(self.approximant)
                # Step 1: generate waveform modes in L0 frame in native domain of
                # approximant (here: TD)
                hlm_td, iota = self.generate_TD_modes_L0(parameters)

                # Step 2: Transform modes to target domain.
                # This requires tapering of TD modes, and FFT to transform to FD.
                wfg_utils.taper_td_modes_in_place(hlm_td)
                hlm_fd = wfg_utils.td_modes_to_fd_modes(hlm_td, self.domain)

            # Step 3: Separate negative and positive frequency parts of the modes,
            # and add contributions according to their transformation behavior under
            # phase shifts.
            pol_m = wfg_utils.get_polarizations_from_fd_modes_m(
                hlm_fd, iota, parameters["phase"]
            )

        else:
            raise NotImplementedError(
                f"Target domain of type {type(self.domain)} not yet implemented."
            )

        if self._domain_transform is not None:
            return self._domain_transform(pol_m)
        else:
            return pol_m

    def generate_FD_modes_LO(self, parameters):  # Pending to adapt
        """
        Generate FD modes in the L0 frame.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters for the waveform.
            For details see see self.generate_hplus_hcross.

        Returns
        -------
        hlm_fd: dict
            Dictionary with (l,m) as keys and the corresponding FD modes in lal format as
            values.
        iota: float
        """
        # TD approximants that are implemented in J frame. Currently tested for:
        #   101: IMRPhenomXPHM
        if self.approximant_str in ["IMRPhenomXPHM"]:
            parameters_gwsignal = self._convert_parameters(
                {**parameters, "f_ref": self.f_ref}
            )
            iota = parameters_gwsignal["inclination"]
            generator = new_interface_get_waveform_generator(self.approximant_str)
            hlm_fd = gws_wfm.GenerateFDModes(parameters_gwsignal, generator)
            # unpack linked list, convert lal objects to arrays

            hlms_lal = {}
            for key, value in hlm_fd.items():
                if type(key) != str:
                    hlm_lal = lal.CreateCOMPLEX16TimeSeries(
                        "hplus",
                        value.epoch.value,
                        0,
                        value.dt.value,
                        lal.DimensionlessUnit,
                        len(value),
                    )
                    hlm_lal.data.data = value.value
                    hlms_lal[key] = hlm_lal

            hlm_fd = wfg_utils.linked_list_modes_to_dict_modes(hlms_lal)
            hlm_fd = {k: v.data.data for k, v in hlm_fd.items()}
            # For the waveform models considered here (e.g., IMRPhenomXPHM), the modes
            # are returned in the J frame (where the observer is at inclination=theta_JN,
            # azimuth=0). In this frame, the dependence on the reference phase enters
            # via the modes themselves. We need to convert to the L0 frame so that the
            # dependence on phase enters via the spherical harmonics.
            hlm_fd = frame_utils.convert_J_to_L0_frame(
                hlm_fd,
                parameters,
                self,
                spin_conversion_phase=self.spin_conversion_phase,
            )
            return hlm_fd, iota
        else:
            raise NotImplementedError(
                f"Approximant {LS.GetApproximantFromString(self.approximant)} not "
                f"implemented. When adding this approximant to this method, make sure "
                f"the the output dict hlm_td contains the TD modes in the *L0 frame*. "
                f"In particular, adding an approximant that is implemented in the same "
                f"domain and frame as one of the approximants should just be a matter of "
                f"adding the approximant number (here: {self.approximant}) to the "
                f"corresponding if statement. However, when doing this please make sure "
                f"to test that this works as intended! Ideally, add some unit tests."
            )

    def generate_TD_modes_L0(self, parameters):
        """
        Generate TD modes in the L0 frame.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters for the waveform.
            For details see see self.generate_hplus_hcross.

        Returns
        -------
        hlm_td: dict
            Dictionary with (l,m) as keys and the corresponding TD modes in lal format as
            values.
        iota: float
        """
        # TD approximants that are implemented in L0 frame. Currently tested for:
        #   52: SEOBNRv4PHM

        parameters_gwsignal = self._convert_parameters(
            {**parameters, "f_ref": self.f_ref}
        )

        generator = new_interface_get_waveform_generator(self.approximant_str)
        hlm_td = gws_wfm.GenerateTDModes(parameters_gwsignal, generator)
        hlms_lal = {}

        for key, value in hlm_td.items():
            if type(key) != str:
                hlm_lal = lal.CreateCOMPLEX16TimeSeries(
                    "hplus",
                    value.epoch.value,
                    0,
                    value.dt.value,
                    lal.DimensionlessUnit,
                    len(value),
                )
                hlm_lal.data.data = value.value
                hlms_lal[key] = hlm_lal

        return hlms_lal, parameters_gwsignal["inclination"].value

    def generate_TD_modes_L0_conditioned_extra_time(self, parameters):
        """
        Generate TD modes in the L0 frame applying a conditioning routine which mimics the behaviour of the standard
        LALSimulation conditioning
        (https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_generator_conditioning_8c.html#ac78b5fcdabf8922a3ac479da20185c85)

        Essentially, a new starting frequency is computed to have some extra cycles that will be tapered. Some extra
        buffer time is also added to ensure that the waveform at the requested starting frequency is not modified,
        while still having a tapered timeseries suited for clean FFT.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters for the waveform.
            For details see self.generate_hplus_hcross.

        Returns
        -------
        hlm_td: dict
            Dictionary with (l,m) as keys and the corresponding TD modes in lal format as
            values.
        iota: float
        """
        # TD approximants that are implemented in L0 frame. Currently tested for:
        # SEOBNRv5HM and SEOBNRv5PHM

        parameters_gwsignal = self._convert_parameters(
            {**parameters, "f_ref": self.f_ref}
        )

        (
            f_min,
            new_f_start,
            t_extra,
            original_f_min,
            f_isco,
        ) = wfg_utils.get_starting_frequency_for_SEOBRNRv5_conditioning(
            parameters_gwsignal
        )
        params = parameters_gwsignal.copy()
        params["f22_start"] = new_f_start * u.Hz

        generator = new_interface_get_waveform_generator(self.approximant_str)
        hlm_td = gws_wfm.GenerateTDModes(params, generator)
        hlms_lal = {}

        for key, value in hlm_td.items():
            if type(key) != str:
                hlm_lal = wfg_utils.taper_td_modes_for_SEOBRNRv5_extra_time(
                    value, t_extra, f_min, original_f_min, f_isco
                )
                hlms_lal[key] = hlm_lal

        return hlms_lal, parameters_gwsignal["inclination"].value

    def generate_TD_waveform(self, parameters_gwsignal: Dict) -> Dict[str, np.ndarray]:
        """
        Generate time domain GW polarizations (h_plus, h_cross)

        Parameters
        ----------
        parameters_gwsignal:
            A dict of parameters for the gwsignal waveform generator

        Returns
        -------
        pol_dict:
            A dictionary of generated waveform polarizations
        """
        # Note: XLALSimInspiralTD() now calls XLALSimInspiralChooseTDWaveform()
        # for models such as SEOBNRv4PHM where the reference frequency is equal
        # to the starting frequency and thus leaves our choice of starting
        # frequency untouched.
        #
        # LS.SimInspiralTD takes parameters:
        #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
        #   distance, inclination, phiRef,
        #   longAscNodes, eccentricity, meanPerAno,
        #   deltaT, f_min, f_ref
        #   lal_params, approximant

        generator = new_interface_get_waveform_generator(self.approximant_str)
        hpc = gws_wfm.GenerateTDWaveform(parameters_gwsignal, generator)

        h_plus = hpc.hp.value
        h_cross = hpc.hc.value
        pol_dict = {"h_plus": h_plus, "h_cross": h_cross}
        return pol_dict


def SEOBNRv4PHM_maximum_starting_frequency(
    total_mass: float, fudge: float = 0.99
) -> float:
    """
    Given a total mass return the largest possible starting frequency allowed
    for SEOBNRv4PHM and similar effective-one-body models.

    The intended use for this function is at the stage of designing
    a data set: after choosing a mass prior one can use it to figure out
    which prior samples would run into an issue when generating an EOB waveform,
    and tweak the parameters to reduce the number of failing configurations.

    Parameters
    ----------
    total_mass:
        Total mass in units of solar masses
    fudge:
        A fudge factor

    Returns
    -------
    f_max_Hz:
        The largest possible starting frequency in Hz
    """
    total_mass_sec = total_mass * lal.MTSUN_SI
    f_max_Hz = fudge * 10.5 ** (-1.5) / (np.pi * total_mass_sec)
    return f_max_Hz


def generate_waveforms_task_func(
    args: Tuple, waveform_generator: WaveformGenerator
) -> Dict[str, np.ndarray]:
    """
    Picklable wrapper function for parallel waveform generation.

    Parameters
    ----------
    args:
        A tuple of (index, pandas.core.series.Series)
    waveform_generator:
        A WaveformGenerator instance

    Returns
    -------
    The generated waveform polarization dictionary
    """
    parameters = args[1].to_dict()

    if isinstance(waveform_generator, BBHxWaveformGenerator):
        # For direct-response mode, generate detector-frame strains directly.
        if getattr(waveform_generator, "direct_response", False):
            return waveform_generator.generate_amp_phase(parameters)
        # Otherwise provide intrinsic mode dictionaries and apply response later.
        return waveform_generator.generate_amp_phase_m(parameters)

    if isinstance(waveform_generator, LISAWaveformGenerator):
        # LISABeta generator returns amplitude/phase-style dictionaries by mode.
        return waveform_generator.generate_amp_phase(parameters)
    
    else:
        return waveform_generator.generate_hplus_hcross(parameters)


def _stack_nested_waveforms(values):
    """Recursively stack nested waveform dictionaries."""
    first = values[0]
    if isinstance(first, dict):
        return {
            k: _stack_nested_waveforms([v[k] for v in values])
            for k in first.keys()
        }
    try:
        return np.stack(values)
    except ValueError:
        # Some LISA/LISABeta mode arrays are variable-length across samples.
        return list(values)


def generate_waveforms_parallel(
     waveform_generator: WaveformGenerator,
     parameter_samples: pd.DataFrame,
     pool: Pool = None,
 ) -> Dict[str, np.ndarray]:

    """
     
     Generate a waveform dataset, optionally in parallel.

     Parameters
     ----------
     waveform_generator: WaveformGenerator
         A WaveformGenerator instance
     parameter_samples: pd.DataFrame
         Intrinsic parameter samples
     pool: multiprocessing.Pool
         Optional pool of workers for parallel generation
 
     Returns
     -------
     polarizations:
         A dictionary of all generated polarizations stacked together
     """
     # logger.info('Generating waveform polarizations ...')
 
    task_func = partial(
         generate_waveforms_task_func, waveform_generator=waveform_generator
    )
    task_data = parameter_samples.iterrows()
 
    if pool is not None:
        waveform_dict_list = pool.map(task_func, task_data)
    else:
        waveform_dict_list = list(map(task_func, task_data))

    
    waveform_dict = {
        pol: _stack_nested_waveforms([wf[pol] for wf in waveform_dict_list])
        for pol in waveform_dict_list[0].keys()
    }

    return waveform_dict


def sum_contributions_m(x_m, phase_shift=0.0):
    """
    Sum the contributions over m-components, optionally introducing a phase shift.
    """
    keys = next(iter(x_m.values())).keys()
    result = {key: 0.0 for key in keys}
    for key in keys:
        for m, x in x_m.items():
            result[key] += x[key] * np.exp(-1j * m * phase_shift)
    return result


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from dingo.gw.domains import build_domain
    from dingo.gw.prior import build_prior_with_defaults

    domain_settings = {
        "type": "UniformFrequencyDomain",
        "f_min": 10.0,
        "f_max": 2048.0,
        "delta_f": 0.125,
    }
    domain = build_domain(domain_settings)
    intrinsic_dict = {
        "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
        "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
        "luminosity_distance": 1000.0,
        "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
        "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
        "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
        "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
        "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
        "geocent_time": 0.0,
    }
    prior = build_prior_with_defaults(intrinsic_dict)
    p = prior.sample()
    p = {
        "mass_ratio": 0.3501852584069329,
        "chirp_mass": 31.709276525188667,
        "luminosity_distance": 1000.0,
        "theta_jn": 1.3663250108421872,
        "phase": 2.3133395191342094,
        "a_1": 0.9082488389607664,
        "a_2": 0.23195443013657285,
        "tilt_1": 2.2991912365076708,
        "tilt_2": 2.2878677821511086,
        "phi_12": 2.3726027637572384,
        "phi_jl": 1.5356479043406908,
        "geocent_time": 0.0,
    }

    wfg = WaveformGenerator(
        # "SEOBNRv4PHM",
        "IMRPhenomXPHM",
        domain,
        20.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )

    pol_m = wfg.generate_hplus_hcross_m(p)

    phase_shift = np.random.uniform(high=2 * np.pi)
    print(f"{phase_shift:.2f}")
    pol = sum_contributions_m(pol_m, phase_shift=phase_shift)

    pol_ref = wfg.generate_hplus_hcross({**p, "phase": p["phase"] + phase_shift})
    # m = mismatch(
    #     apply_frequency_mask(pol, wfg.domain), apply_frequency_mask(pol_ref, wfg.domain)
    # )
    # print(f"mismatch {m:.1e}")

    import matplotlib.pyplot as plt

    x = wfg.domain()
    plt.xlim((10, 512))
    plt.xscale("log")
    plt.plot(x, pol_ref["h_plus"].real)
    plt.plot(x, pol["h_plus"].real)
    plt.plot(x, (pol_ref["h_plus"] - pol["h_plus"]).real)
    plt.show()


class LISAWaveformGenerator:
    """Generate Amplitude/Phase waveforms using lisabeta routines in the specified domain for a
    single GW coalescence given a set of waveform parameters.
    """

    def __init__(
        self,
        approximant: str,
        domain: Domain,
        f_ref: float,
        f_start: float = None,
        mode_list: list[Tuple] = None,
        transform=None,
        spin_conversion_phase=None,
        frozenLISA = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        approximant : str
            Waveform "approximant" string understood by lalsimulation
            This is defines which waveform model is used.
        domain : Domain
            Domain object that specifies on which physical domain the
            waveform polarizations will be generated, e.g. Fourier
            domain, time domain.
        f_ref : float
            Reference frequency for the waveforms
        f_start : float
            Starting frequency for waveform generation. This is optional, and if not
            included, the starting frequency will be set to f_min. This exists so that
            EOB waveforms can be generated starting from a lower frequency than f_min.
        mode_list : List[Tuple]
            A list of waveform (ell, m) modes to include when generating
            the polarizations.
        spin_conversion_phase : float = None
            Value for phiRef when computing cartesian spins from bilby spins via
            bilby_to_lalsimulation_spins. The common convention is to use the value of
            the phase parameter here, which is also used in the spherical harmonics
            when combining the different modes. If spin_conversion_phase = None,
            this default behavior is adapted.
            For dingo, this convention for the phase parameter makes it impossible to
            treat the phase as an extrinsic parameter, since we can only account for
            the change of phase in the spherical harmonics when changing the phase (in
            order to also change the cartesian spins -- specifically, to rotate the spins
            by phase in the sx-sy plane -- one would need to recompute the modes,
            which is expensive).
            By setting spin_conversion_phase != None, we impose the convention to always
            use phase = spin_conversion_phase when computing the cartesian spins.
        """
        if not isinstance(approximant, str):
            raise ValueError("approximant should be a string, but got", approximant)
        else:
            self.approximant_str = approximant
    
        if not issubclass(type(domain), Domain):
            raise ValueError(
                "domain should be an instance of a subclass of Domain, but got",
                type(domain),
            )
        else:
            self.domain = domain

        self.f_ref = f_ref
        self.f_start = f_start

        self.transform = transform
        self.frozenLISA = frozenLISA
    
    @property
    def domain(self):
        if self._use_base_domain:
            return self._domain.base_domain
        else:
            return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value
        if isinstance(
            self._domain, MultibandedFrequencyDomain
        ) and not LS.SimInspiralImplementedFDApproximants(LS.GetApproximantFromString(self.approximant_str)):
            # For non-frequency domain approximants, generate waveforms in the base
            # UniformFrequencyDomain, and later decimate.
            self._use_base_domain = True
            self._domain_transform = DecimateAll(self._domain)
        else:
            # For frequency-domain approximants, generate waveforms directly in either
            # UFD or MFD.
            self._use_base_domain = False
            self._domain_transform = None

    @property
    def full_domain(self):
        return self._domain
    
    
    
        
    def generate_amp_phase(
        self, parameters: Dict[str, float], catch_waveform_errors=True,
    ) -> Dict[str, np.ndarray]:
        """Generate GW polarizations (h_plus, h_cross).

        If the generation of the lalsimulation waveform fails with an
        "Input domain error", we return NaN polarizations.

        Use the domain, approximant, and mode_list specified in the constructor
        along with the waveform parameters to generate the waveform polarizations.


        Parameters
        ----------
        parameters: Dict[str, float]
            A dictionary of parameter names and scalar values.
            The parameter dictionary must include the following keys.
            For masses, spins, and distance there are multiple options.

            Mass: (mass_1, mass_2) or a pair of quantities from
                ((chirp_mass, total_mass), (mass_ratio, symmetric_mass_ratio))
            Spin:
                (a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl) if precessing binary or
                (chi_1, chi_2) if the binary has aligned spins
            Reference frequency: f_ref at which spin vectors are defined
            Extrinsic:
                Distance: one of (luminosity_distance, redshift, comoving_distance)
                Inclination: theta_jn
                Reference phase: phase
                Geocentric time: geocent_time (GPS time)
            The following parameters are not required:
                Sky location: ra, dec,
                Polarization angle: psi
            Units:
                Masses should be given in units of solar masses.
                Distance should be given in megaparsecs (Mpc).
                Frequencies should be given in Hz and time in seconds.
                Spins should be dimensionless.
                Angles should be in radians.

        catch_waveform_errors: bool
            Whether to catch lalsimulation errors

        Returns
        -------
        wf_dict:
            A dictionary of generated waveform polarizations
        """
    
        parameters = parameters.copy()

        # Extract geocent_time (seconds) and convert to years for LISA orbital position
        geocent_time_s = parameters.get("geocent_time", parameters.get("t_ref", 0.0))
        t0 = geocent_time_s / pyconstants.YRSID_SI

        #Convert everything to SSB frame
        parameters = self.convert_parameters(parameters, self.frozenLISA, t0=t0)

        #parameters["f_ref"] = self.f_ref

        gridfreq = self.Generate_coarse_freq_grid(parameters, t0=t0)
        
        if (self.approximant_str=='IMRPhenomD'):

        # Specifying fref_for_phiref, phiref defines the source frame
        # tref, fref_for_tref are not free, we set tf=0 (tSSB=t0) at fstart
            wfClass = pyIMRPhenomD.IMRPhenomDh22AmpPhase(gridfreq, parameters["m1"], parameters["m2"], parameters["chi1"], parameters["chi2"], parameters["dist"], tref=0., phiref=0., fref_for_tref=0., fref_for_phiref=0.,Deltat = parameters["Deltat"], force_phiref_fref=True, extra_params=None)
            wfhlm = wfClass.get_waveform()
            fpeak = wfClass.get_fpeak()
        elif (self.approximant_str=='IMRPhenomHM'):
            # IMRPhenomHM
            # For now tf=0 at fpeak hardcoded
            # fref means fref_for_phiref, default fpeak
            wfClass = pyIMRPhenomHM.IMRPhenomHMhlmAmpPhase(gridfreq,parameters["m1"], parameters["m2"], parameters["chi1"], parameters["chi2"],parameters["dist"], phiref=0., fref=0.,Deltat = parameters["Deltat"], scale_freq_hm=True, extra_params=None)
            wfhlm = wfClass.get_waveform()

        elif (self.approximant_str =='IMRPhenomXHM'):
        
            wfClass = pyIMRPhenomXHM.IMRPhenomXHMhlmAmpPhase(gridfreq, parameters["m1"], parameters["m2"], parameters["chi1"], parameters["chi2"],parameters["dist"], phiref=0., fref=0., Deltat=parameters["Deltat"], scale_freq_hm=True, extra_params=None)
            wfhlm = wfClass.get_waveform()
        
        return wfhlm
            
    def generate_amp_phase_m(self, parameters):
        # This calls your existing lisabeta-wrapped method
        # returns {(l, m): {'amp': ..., 'phase': ..., ...}}
        raw_modes = self.generate_amp_phase(parameters)
        
        pol_m = {}
        for (l, m), data in raw_modes.items():
            # Calculate complex frequency-domain strain: A * exp(i * phi)
            complex_strain = data['amp'] * np.exp(1j * data['phase'])
            
            # Group by 'm' index. We sum all 'l' contributions for the same 'm'.
            if m not in pol_m:
                pol_m[m] = {"waveform": complex_strain}
            else:
                pol_m[m]["waveform"] += complex_strain
                
        return pol_m
        
        
        
        
        
            
        
    
    
        
    
    def convert_parameters(self,parameters,frozenLISA, t0=0.):
        """ Convert parameters for LISA analysis.  
        We set all parameters to be set into the SSB frame if they aren't already.  
        We also complete mass and spin parameters 
        
        parameters
        ----------
        parameters: Dict[str, float]
            A dictionary of parameter names and scalar values
        t0: 
            Reference time (yr), so that t=0 for the waveform
            corresponds to t0 in the SSB-frame
        frozenLISA: 
            Whether to keep detector arms fixed
        
        """
        if parameters.get('Lframe', False):
            parameters = lisatools.convert_Lframe_to_SSBframe(parameters,
                                                          t0=t0,
                                                          frozenLISA=frozenLISA)

        parameters = pytools.complete_mass_params(parameters)
        parameters = pytools.complete_spin_params(parameters)
        
        return parameters
    
    
    def Generate_coarse_freq_grid(self, params, t0=0.):
        fLow, fHigh = wfg_utils.FrequencyBoundsLISATDI_SMBH(params, t0=t0, timetomerger_max=1., minf=self.domain.f_min, maxf=self.domain.f_max,
                                                  fstart22=None, fend22=None, tmin=None, tmax=None, Mfmax_model=0.3, 
                                                  DeltatL_cut=None, DeltatSSB_cut=None, scale_freq_hm=True, 
                                                  modes=None, f_t_acc=1e-06, approximant=self.approximant_str)
        if "M" not in params.keys():
            params = pytools.complete_mass_params(params)
        
        if isinstance(fLow, dict) and isinstance(fHigh, dict):
            gridfreqClass = pytools.FrequencyGrid(fLow[(2,2)], fHigh[(2,2)], params["M"], params["q"], acc=acc, DeltalnMf_max=0.025)
            gridfreq22 = gridfreqClass.get_freq()
            gridfreq = {}
            for lm in modes:
                gridfreq[lm] = pytools.log_affine_scaling(gridfreq22, fLow[lm], fHigh[lm])
        else:
            # For PhenomHM will be rescaled by m/2 for different modes hlm
            gridfreqClass = pytools.FrequencyGrid(fLow, fHigh, params["M"], params["q"], acc=1e-04, DeltalnMf_max=0.025)
            gridfreq = gridfreqClass.get_freq()

        return gridfreq


class BBHxWaveformGenerator:
    """Generate Amplitude/Phase waveforms using BBHx (GPU-accelerated MBHB waveforms) in the specified
    domain for a single GW coalescence given a set of waveform parameters.
    
    This class provides GPU-accelerated waveform generation for massive black hole binaries (MBHBs)
    as observed by LISA, using the BBHx package.
    """

    def __init__(
        self,
        approximant: str,
        domain: Domain,
        f_ref: float,
        f_start: float = None,
        mode_list: list[Tuple] = None,
        transform=None,
        spin_conversion_phase=None,
        frozenLISA=False,
        use_gpu=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        approximant : str
            Waveform approximant string. For BBHx this should be 'PhenomHM'.
        domain : Domain
            Domain object that specifies on which physical domain the
            waveform polarizations will be generated (frequency domain, time domain, etc.)
        f_ref : float
            Reference frequency for the waveforms (Hz)
        f_start : float, optional
            Starting frequency for waveform generation (Hz)
        mode_list : List[Tuple], optional
            A list of waveform (ell, m) modes to include when generating the polarizations.
        spin_conversion_phase : float, optional
            Phase value for spin parameter conversions
        frozenLISA : bool, optional
            Whether to keep detector arms fixed. Default is False.
        use_gpu : bool, optional
            Whether to use GPU acceleration. Default is False.
        kwargs : dict, optional
            Optional BBHx settings. Supported keys include:
            - default_t_ref_seconds: fallback BBHx reference time in seconds.
            - default_t_ref_years: fallback BBHx reference time in sidereal years
              (used only if default_t_ref_seconds is not provided).
        """
        if not isinstance(approximant, str):
            raise ValueError("approximant should be a string, but got", approximant)
        else:
            self.approximant_str = approximant
    
        if not issubclass(type(domain), Domain):
            raise ValueError(
                "domain should be an instance of a subclass of Domain, but got",
                type(domain),
            )
        else:
            self.domain = domain

        self.f_ref = f_ref
        self.f_start = f_start
        _default_mode_list = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        self.mode_list = [tuple(m) for m in (mode_list if mode_list is not None else _default_mode_list)]
        self.transform = transform
        self.frozenLISA = frozenLISA
        self.use_gpu = use_gpu
        self.direct_response = bool(kwargs.get("direct_response", False))
        self.gpu_fastpath = bool(kwargs.get("gpu_fastpath", False))
        self.backend_native_fused = bool(kwargs.get("backend_native_fused", False))
        self.timing_profile = bool(kwargs.get("timing_profile", False))
        self.timing_profile_print_every = int(kwargs.get("timing_profile_print_every", 0))
        self.bbhx_length = int(kwargs.get("bbhx_length", 1024))
        # BBHx waveform window in years (LISA frame), passed through to BBHWaveformFD.
        # Keep BBHx defaults unless explicitly overridden.
        self.bbhx_t_obs_start_years = float(kwargs.get("bbhx_t_obs_start_years", 0.0))
        self.bbhx_t_obs_end_years = float(kwargs.get("bbhx_t_obs_end_years", 1.0))
        # When True, ``bbhx_t_obs_start_years`` / ``bbhx_t_obs_end_years`` are
        # interpreted as ABSOLUTE SSB years (the bbhx ``shift_t_limits=True``
        # path), and bbhx evaluates the response on tf in absolute SSB time
        # rather than time-from-merger. Required when the bbhx response
        # checks the orbit table (which lives in absolute SSB) directly.
        self.bbhx_shift_t_limits = bool(kwargs.get("bbhx_shift_t_limits", False))
        # If True, multiply the bbhx output by exp(+i 2π f t_ref) so that the
        # merger lands at t=0 in the strain time axis (dingo / lisabeta
        # convention). The carrier exp(-i 2π f t_ref) is exact at numerical
        # precision (validated empirically); the response geometry is
        # unaffected because it is not a linear-in-f phase. Consumers that see
        # a decentered waveform must re-apply the shift via t_ref carried on
        # the output dict / sample.
        self.decenter_waveform = bool(kwargs.get("decenter_waveform", False))
        default_t_ref_seconds = kwargs.get("default_t_ref_seconds", None)
        if default_t_ref_seconds is None:
            default_t_ref_years = float(kwargs.get("default_t_ref_years", 1.0))
            default_t_ref_seconds = default_t_ref_years * YRSID_SI
        self.default_t_ref_seconds = float(default_t_ref_seconds)
        self.orbits = None
        self._cached_output_freqs_cpu = None
        self._cached_output_freqs_backend = None
        self.reset_timing_stats()
        
        # Initialize BBHx waveform generator
        try:
            # Initialize orbits for GPU if needed
            response_kwargs = {}
            if self.use_gpu:
                try:
                    from lisatools.detector import EqualArmlengthOrbits
                    self.orbits = EqualArmlengthOrbits(use_gpu=True)
                    self.orbits.configure(linear_interp_setup=True)
                    response_kwargs = dict(orbits=self.orbits)
                except ImportError as e:
                    warnings.warn(f"GPU mode requires lisatools for orbits. Falling back to CPU response calculation. Error: {e}")
                    self.use_gpu = False
            
            self.waveform_gen = BBHWaveformFD(
                amp_phase_kwargs=dict(run_phenomd=False),
                response_kwargs=response_kwargs,
                use_gpu=self.use_gpu,
            )
        except (NameError, ImportError) as e:
            raise ImportError(f"BBHx is not installed or cannot be imported. Please ensure BBHx is in PYTHONPATH. Error: {e}")
    
    @property
    def domain(self):
        if self._use_base_domain:
            return self._domain.base_domain
        else:
            return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value
        # For now, assume BBHx always generates in frequency domain
        self._use_base_domain = False
        self._domain_transform = None
        self._cached_output_freqs_cpu = None
        self._cached_output_freqs_backend = None

    @property
    def full_domain(self):
        return self._domain

    @staticmethod
    def _to_numpy(x):
        return x.get() if hasattr(x, "get") else np.asarray(x)

    def _decenter_waveform(self, waveform, freqs, t_ref):
        """Multiply by exp(+i 2π f t_ref) so the merger sits at the canonical
        t=0 of the strain time axis. ``freqs`` runs along the last axis of
        ``waveform``. ``t_ref`` is a scalar or 1-D array of length B.

        Supports both bbhx native layouts: (channels, length) for B=1 squeezed,
        (channels, B, length), or (B, channels, length). The batch axis is
        identified as the one whose size matches ``len(t_ref)`` (or = 1 when
        ``t_ref`` is a scalar).
        """
        xp = self.waveform_gen.xp if hasattr(self.waveform_gen, "xp") else np
        f = xp.asarray(freqs).reshape(-1)
        t = xp.atleast_1d(xp.asarray(t_ref))
        two_pi_i = 1j * 2.0 * np.pi
        Nf = f.shape[0]
        B  = t.shape[0]

        if waveform.ndim == 2:
            # (channels, length)
            phasor = xp.exp(two_pi_i * float(t.ravel()[0]) * f)
            return waveform * phasor[None, :]

        if waveform.ndim == 3:
            shape = waveform.shape
            assert shape[-1] == Nf, (
                f"_decenter_waveform: waveform last axis {shape[-1]} != Nf={Nf}"
            )
            # Identify the batch axis by matching dimension to B.
            if B == 1:
                # Scalar t_ref; broadcast as a single phasor along the freq axis.
                phasor = xp.exp(two_pi_i * float(t.ravel()[0]) * f)
                return waveform * phasor[None, None, :]
            if shape[0] == B and shape[1] != B:
                # (B, channels, length)
                phasor = xp.exp(two_pi_i * t[:, None] * f[None, :])  # (B, Nf)
                return waveform * phasor[:, None, :]
            if shape[1] == B:
                # (channels, B, length)
                phasor = xp.exp(two_pi_i * t[:, None] * f[None, :])  # (B, Nf)
                return waveform * phasor[None, :, :]
            raise ValueError(
                f"_decenter_waveform: cannot locate batch axis of size {B} "
                f"in waveform shape {tuple(shape)}"
            )
        raise ValueError(
            f"_decenter_waveform: unsupported shape {tuple(waveform.shape)}"
        )

    def set_timing_profile(self, enabled: bool, print_every: int = None):
        self.timing_profile = bool(enabled)
        if print_every is not None:
            self.timing_profile_print_every = int(print_every)

    def reset_timing_stats(self):
        self._timing_count_total = 0
        self._timing_count_direct = 0
        self._timing_count_amp_phase = 0
        self._timing_totals = {
            "all_total": 0.0,
            "direct_total": 0.0,
            "direct_freq_grid": 0.0,
            "direct_param_pick": 0.0,
            "direct_batch_infer": 0.0,
            "direct_mass_from_mcq": 0.0,
            "direct_scalar_pick": 0.0,
            "direct_coerce": 0.0,
            "direct_t_ref_fix": 0.0,
            "direct_distance_convert": 0.0,
            "direct_waveform_call": 0.0,
            "direct_decenter": 0.0,
            "direct_return_convert": 0.0,
            "amp_total": 0.0,
            "amp_parse_parameters": 0.0,
            "amp_freq_grid": 0.0,
            "amp_waveform_call": 0.0,
            "amp_package": 0.0,
        }

    def _timing_add(self, key: str, dt: float):
        if self.timing_profile:
            self._timing_totals[key] += dt

    def _timing_finalize(self, path: str, total_dt: float):
        if not self.timing_profile:
            return
        self._timing_count_total += 1
        self._timing_totals["all_total"] += total_dt
        if path == "direct":
            self._timing_count_direct += 1
            self._timing_totals["direct_total"] += total_dt
        elif path == "amp":
            self._timing_count_amp_phase += 1
            self._timing_totals["amp_total"] += total_dt

        if (
            self.timing_profile_print_every > 0
            and self._timing_count_total % self.timing_profile_print_every == 0
        ):
            stats = self.get_timing_stats()
            if self._timing_count_direct > 0:
                print(
                    "[BBHxGenTiming] "
                    f"n_direct={stats['direct_calls']} "
                    f"direct_total={stats['avg_direct_total']:.4f}s "
                    f"waveform_call={stats['avg_direct_waveform_call']:.4f}s "
                    f"coerce={stats['avg_direct_coerce']:.4f}s"
                )

    def get_timing_stats(self, reset: bool = False):
        out = {
            "enabled": self.timing_profile,
            "total_calls": int(self._timing_count_total),
            "direct_calls": int(self._timing_count_direct),
            "amp_calls": int(self._timing_count_amp_phase),
        }
        dcount = max(self._timing_count_direct, 1)
        acount = max(self._timing_count_amp_phase, 1)
        tcount = max(self._timing_count_total, 1)
        for key, total in self._timing_totals.items():
            out[f"total_{key}"] = float(total)
            if key.startswith("direct_"):
                out[f"avg_{key}"] = float(total) / dcount
            elif key.startswith("amp_"):
                out[f"avg_{key}"] = float(total) / acount
            else:
                out[f"avg_{key}"] = float(total) / tcount
        if reset:
            self.reset_timing_stats()
        return out

    @staticmethod
    def _get_first(parameters: Dict[str, float], keys, default=None):
        for key in keys:
            if key in parameters:
                return parameters[key]
        return default

    @staticmethod
    def _masses_from_chirp_mass_and_q(chirp_mass: float, q: float) -> Tuple[float, float]:
        # Bilby/Dingo convention: q = m2 / m1 <= 1.
        m1 = chirp_mass * (1.0 + q) ** (1.0 / 5.0) / (q ** (3.0 / 5.0))
        m2 = q * m1
        return m1, m2

    @staticmethod
    def _broadcast_to_length(x, n: int, name: str):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=np.float64)
        if arr.size == n:
            return arr.astype(np.float64, copy=False)
        if arr.size == 1:
            return np.full(n, float(arr.ravel()[0]), dtype=np.float64)
        raise ValueError(
            f"BBHx parameter '{name}' has incompatible length {arr.size}, expected 1 or {n}."
        )

    def _parse_parameters(self, parameters: Dict[str, float]) -> Dict[str, float]:
        m1 = self._get_first(parameters, ["mass_1", "m1"])
        m2 = self._get_first(parameters, ["mass_2", "m2"])
        if m1 is None or m2 is None:
            chirp_mass = self._get_first(parameters, ["chirp_mass", "Mchirp"])
            q = self._get_first(parameters, ["mass_ratio", "q"])
            if chirp_mass is None or q is None:
                raise ValueError(
                    "BBHxWaveformGenerator requires either (mass_1, mass_2) or "
                    "(chirp_mass/Mchirp, mass_ratio/q)."
                )
            m1, m2 = self._masses_from_chirp_mass_and_q(chirp_mass, q)

        chi1z = self._get_first(
            parameters, ["chi_1z", "chi1z", "chi_1", "chi1"], 0.0
        )
        chi2z = self._get_first(
            parameters, ["chi_2z", "chi2z", "chi_2", "chi2"], 0.0
        )

        distance_mpc = self._get_first(
            parameters,
            # Prefer LISA naming first to avoid default bilby aliases silently
            # overriding explicitly provided LISA extrinsics.
            ["dist", "luminosity_distance", "redshift_distance"],
            None,
        )
        if distance_mpc is None:
            raise ValueError(
                "BBHxWaveformGenerator requires luminosity_distance/dist (Mpc)."
            )

        inc = self._get_first(parameters, ["theta_jn", "inc"], 0.0)
        phase = self._get_first(parameters, ["phase", "phi"], 0.0)
        lam = self._get_first(parameters, ["lambda", "ra", "lambd"], 0.0)
        beta = self._get_first(parameters, ["beta", "dec"], 0.0)
        psi = self._get_first(parameters, ["psi"], 0.0)

        # BBHx expects t_ref in seconds (SSB frame). When the caller explicitly
        # provides a time parameter, preserve it exactly.
        if "t_ref" in parameters:
            t_ref = parameters["t_ref"]
        elif "t_ref_years" in parameters:
            t_ref = parameters["t_ref_years"] * YRSID_SI
        elif "geocent_time" in parameters:
            t_ref = parameters["geocent_time"]
        else:
            t_ref = self.default_t_ref_seconds

        m1_arr = np.asarray(m1, dtype=np.float64)
        n = m1_arr.size if m1_arr.ndim > 0 else 1

        return {
            "m1": self._broadcast_to_length(m1, n, "m1"),
            "m2": self._broadcast_to_length(m2, n, "m2"),
            "chi1z": self._broadcast_to_length(chi1z, n, "chi1z"),
            "chi2z": self._broadcast_to_length(chi2z, n, "chi2z"),
            "distance_mpc": self._broadcast_to_length(distance_mpc, n, "distance_mpc"),
            "inc": self._broadcast_to_length(inc, n, "inc"),
            "phase": self._broadcast_to_length(phase, n, "phase"),
            "lam": self._broadcast_to_length(lam, n, "lam"),
            "beta": self._broadcast_to_length(beta, n, "beta"),
            "psi": self._broadcast_to_length(psi, n, "psi"),
            "t_ref": self._broadcast_to_length(t_ref, n, "t_ref"),
        }

    def _build_bbhx_frequency_grid(self) -> np.ndarray:
        """Build a sparse log-spaced frequency grid for BBHx interpolation.

        For multibanded domains, use base-domain bounds. This avoids relying on
        nonuniform ``delta_f`` arrays.
        """
        if hasattr(self.domain, "base_domain"):
            f_min = float(self.domain.base_domain.f_min)
            f_max = float(self.domain.base_domain.f_max)
        else:
            f_min = float(self.domain.f_min)
            f_max = float(self.domain.f_max)
        return np.logspace(np.log10(f_min), np.log10(f_max), self.bbhx_length)

    def _get_output_frequency_grid(self) -> np.ndarray:
        """Frequency grid for final waveforms, aligned with the Dingo domain."""
        if hasattr(self.domain, "sample_frequencies"):
            return np.asarray(self.domain.sample_frequencies, dtype=np.float64)
        if hasattr(self.domain, "base_domain"):
            return np.asarray(self.domain.base_domain.sample_frequencies, dtype=np.float64)
        return self._build_bbhx_frequency_grid()

    @staticmethod
    def _pick_value(extrinsic_parameters: Dict[str, float], intrinsic_parameters: Dict[str, float], keys, default=None):
        for key in keys:
            if key in extrinsic_parameters:
                return extrinsic_parameters[key]
            if key in intrinsic_parameters:
                return intrinsic_parameters[key]
        return default

    @staticmethod
    def _to_float64_array(x):
        if np.isscalar(x):
            return np.asarray(float(x), dtype=np.float64)
        return np.asarray(x, dtype=np.float64)

    def _infer_batch_size(self, *values) -> int:
        for v in values:
            if v is None:
                continue
            arr = np.asarray(v)
            if arr.ndim > 0 and arr.size > 1:
                return int(arr.size)
        return 1

    def _coerce_batch_value(self, x, n: int, name: str):
        arr = self._to_float64_array(x)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=np.float64)
        if arr.size == n:
            return arr.astype(np.float64, copy=False).reshape(n)
        if arr.size == 1:
            return np.full(n, float(arr.ravel()[0]), dtype=np.float64)
        raise ValueError(
            f"BBHx parameter '{name}' has incompatible length {arr.size}, expected 1 or {n}."
        )

    def generate_direct_response_backend_native(
        self,
        intrinsic_parameters: Dict[str, float],
        extrinsic_parameters: Dict[str, float],
        catch_waveform_errors: bool = False,
    ):
        """
        Backend-native fused direct-response path for BBHx GPU fast training.
        Avoids merged parameter dicts and extra Python bookkeeping in transforms.
        """
        total_t0 = time.perf_counter() if self.timing_profile else None
        t0 = time.perf_counter() if self.timing_profile else None
        freqs = self._get_cached_backend_frequency_grid()
        if self.timing_profile:
            self._timing_add("direct_freq_grid", time.perf_counter() - t0)
        try:
            t0 = time.perf_counter() if self.timing_profile else None
            m1 = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["mass_1", "m1"])
            m2 = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["mass_2", "m2"])
            chirp_mass = self._pick_value(
                extrinsic_parameters, intrinsic_parameters, ["chirp_mass", "Mchirp"]
            )
            q = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["mass_ratio", "q"])
            if self.timing_profile:
                self._timing_add("direct_param_pick", time.perf_counter() - t0)

            t0 = time.perf_counter() if self.timing_profile else None
            n = self._infer_batch_size(m1, m2, chirp_mass, q)
            if self.timing_profile:
                self._timing_add("direct_batch_infer", time.perf_counter() - t0)

            if m1 is None or m2 is None:
                if chirp_mass is None or q is None:
                    raise ValueError(
                        "BBHxWaveformGenerator requires either (mass_1, mass_2) or "
                        "(chirp_mass/Mchirp, mass_ratio/q)."
                    )
                t0 = time.perf_counter() if self.timing_profile else None
                chirp_arr = self._coerce_batch_value(chirp_mass, n, "chirp_mass")
                q_arr = self._coerce_batch_value(q, n, "q")
                m1, m2 = self._masses_from_chirp_mass_and_q(chirp_arr, q_arr)
                if self.timing_profile:
                    self._timing_add("direct_mass_from_mcq", time.perf_counter() - t0)

            t0 = time.perf_counter() if self.timing_profile else None
            chi1z = self._pick_value(
                extrinsic_parameters, intrinsic_parameters, ["chi_1z", "chi1z", "chi_1", "chi1"], 0.0
            )
            chi2z = self._pick_value(
                extrinsic_parameters, intrinsic_parameters, ["chi_2z", "chi2z", "chi_2", "chi2"], 0.0
            )
            distance_mpc = self._pick_value(
                extrinsic_parameters,
                intrinsic_parameters,
                ["dist", "luminosity_distance", "redshift_distance"],
                None,
            )
            if distance_mpc is None:
                raise ValueError(
                    "BBHxWaveformGenerator requires luminosity_distance/dist (Mpc)."
                )
            inc = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["theta_jn", "inc"], 0.0)
            phase = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["phase", "phi"], 0.0)
            lam = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["lambda", "ra", "lambd"], 0.0)
            beta = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["beta", "dec"], 0.0)
            psi = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["psi"], 0.0)

            if "t_ref" in extrinsic_parameters or "t_ref" in intrinsic_parameters:
                t_ref = self._pick_value(extrinsic_parameters, intrinsic_parameters, ["t_ref"])
            elif "t_ref_years" in extrinsic_parameters or "t_ref_years" in intrinsic_parameters:
                t_ref = self._pick_value(
                    extrinsic_parameters, intrinsic_parameters, ["t_ref_years"]
                )
                t_ref = self._to_float64_array(t_ref) * YRSID_SI
            elif "geocent_time" in extrinsic_parameters or "geocent_time" in intrinsic_parameters:
                t_ref = self._pick_value(
                    extrinsic_parameters, intrinsic_parameters, ["geocent_time"]
                )
            else:
                t_ref = self.default_t_ref_seconds
            if self.timing_profile:
                self._timing_add("direct_scalar_pick", time.perf_counter() - t0)

            t0 = time.perf_counter() if self.timing_profile else None
            m1 = self._coerce_batch_value(m1, n, "m1")
            m2 = self._coerce_batch_value(m2, n, "m2")
            chi1z = self._coerce_batch_value(chi1z, n, "chi1z")
            chi2z = self._coerce_batch_value(chi2z, n, "chi2z")
            distance_mpc = self._coerce_batch_value(distance_mpc, n, "distance_mpc")
            inc = self._coerce_batch_value(inc, n, "inc")
            phase = self._coerce_batch_value(phase, n, "phase")
            lam = self._coerce_batch_value(lam, n, "lam")
            beta = self._coerce_batch_value(beta, n, "beta")
            psi = self._coerce_batch_value(psi, n, "psi")
            t_ref = self._coerce_batch_value(t_ref, n, "t_ref")
            if self.timing_profile:
                self._timing_add("direct_coerce", time.perf_counter() - t0)

            t0 = time.perf_counter() if self.timing_profile else None
            distance_si = distance_mpc * PC_SI * 1e6
            if self.timing_profile:
                self._timing_add("direct_distance_convert", time.perf_counter() - t0)

            t0 = time.perf_counter() if self.timing_profile else None
            waveform_data = self.waveform_gen(
                m1,
                m2,
                chi1z,
                chi2z,
                distance_si,
                phase,
                self.f_ref,
                inc,
                lam,
                beta,
                psi,
                t_ref,
                t_obs_start=self.bbhx_t_obs_start_years,
                t_obs_end=self.bbhx_t_obs_end_years,
                shift_t_limits=self.bbhx_shift_t_limits,
                freqs=freqs,
                modes=self.mode_list,
                direct=False,
                fill=True,
                squeeze=True,
                length=self.bbhx_length,
            )
            if self.timing_profile:
                self._timing_add("direct_waveform_call", time.perf_counter() - t0)

            if self.decenter_waveform:
                t0 = time.perf_counter() if self.timing_profile else None
                waveform_data = self._decenter_waveform(waveform_data, freqs, t_ref)
                if self.timing_profile:
                    self._timing_add(
                        "direct_decenter", time.perf_counter() - t0
                    )

            t0 = time.perf_counter() if self.timing_profile else None
            if self.use_gpu:
                if self.timing_profile:
                    self._timing_add("direct_return_convert", time.perf_counter() - t0)
                    self._timing_finalize("direct", time.perf_counter() - total_t0)
                return waveform_data
            out = self._to_numpy(waveform_data)
            if self.timing_profile:
                self._timing_add("direct_return_convert", time.perf_counter() - t0)
                self._timing_finalize("direct", time.perf_counter() - total_t0)
            return out
        except Exception:
            if not catch_waveform_errors:
                raise
            nan_shape = (3, len(freqs))
            return np.full(nan_shape, np.nan, dtype=np.complex128)

    def _get_cached_backend_frequency_grid(self):
        """Return cached output frequencies in backend array type."""
        if self._cached_output_freqs_cpu is None:
            self._cached_output_freqs_cpu = self._get_output_frequency_grid()
        if not self.use_gpu:
            return self._cached_output_freqs_cpu
        if self._cached_output_freqs_backend is None:
            self._cached_output_freqs_backend = self.waveform_gen.xp.asarray(
                self._cached_output_freqs_cpu
            )
        return self._cached_output_freqs_backend

    def generate_amp_phase(
        self, parameters: Dict[str, float], catch_waveform_errors=False,
    ) -> Dict[str, np.ndarray]:
        """Generate GW amplitude and phase using BBHx.

        Parameters
        ----------
        parameters: Dict[str, float]
            A dictionary of parameter names and scalar values.
            Required keys:
                - mass_1, mass_2: Component masses (solar masses)
                - chi_1z, chi_2z: Aligned-frame spins
                - luminosity_distance: Distance to the source (Mpc)
                - theta_jn: Inclination angle (radians)
                - phase: Reference phase (radians)
                - geocent_time: Geocentric time (GPS seconds)
                - (optional) ra, dec, psi: Sky location and polarization

        catch_waveform_errors: bool
            Whether to catch waveform generation errors (default False for debugging)

        Returns
        -------
        wf_dict: Dict
            Dictionary of waveform data with frequency-dependent amplitudes and phases
        """
        total_t0 = time.perf_counter() if self.timing_profile else None
        try:
            parameters = parameters.copy()
            
            # Convert LISA detector parameters if needed
            if self.frozenLISA:
                parameters = lisatools.convert_Lframe_to_SSBframe(
                    parameters, t0=0., frozenLISA=True
                )
            
            t0 = time.perf_counter() if self.timing_profile else None
            parsed = self._parse_parameters(parameters)
            if self.timing_profile:
                self._timing_add("amp_parse_parameters", time.perf_counter() - t0)
            m1 = parsed["m1"]
            m2 = parsed["m2"]
            chi1z = parsed["chi1z"]
            chi2z = parsed["chi2z"]
            distance_si = parsed["distance_mpc"] * PC_SI * 1e6
            inc = parsed["inc"]
            phase = parsed["phase"]
            lam = parsed["lam"]
            beta = parsed["beta"]
            psi = parsed["psi"]
            t_ref = parsed["t_ref"]
            
            # Interpolate BBHx output onto the Dingo domain grid.
            t0 = time.perf_counter() if self.timing_profile else None
            freqs = self._get_cached_backend_frequency_grid()
            if self.timing_profile:
                self._timing_add("amp_freq_grid", time.perf_counter() - t0)
            
            # Generate waveform using BBHx
            t0 = time.perf_counter() if self.timing_profile else None
            waveform_data = self.waveform_gen(
                m1, m2, chi1z, chi2z,
                distance_si,
                phase, self.f_ref,
                inc, lam, beta, psi,
                t_ref,
                t_obs_start=self.bbhx_t_obs_start_years,
                t_obs_end=self.bbhx_t_obs_end_years,
                shift_t_limits=self.bbhx_shift_t_limits,
                freqs=freqs,
                modes=self.mode_list,
                direct=False,
                fill=True,
                squeeze=True,
                length=self.bbhx_length
            )
            if self.timing_profile:
                self._timing_add("amp_waveform_call", time.perf_counter() - t0)

            if self.decenter_waveform:
                waveform_data = self._decenter_waveform(waveform_data, freqs, t_ref)

            # Package waveform data. Keep all returned channels (A/E/T) rather than
            # indexing a single channel.
            t0 = time.perf_counter() if self.timing_profile else None
            if self.direct_response and self.gpu_fastpath:
                # Keep backend array type (e.g., CuPy) for CUDA fast-path transforms.
                waveform_payload = waveform_data
            else:
                waveform_payload = self._to_numpy(waveform_data)

            if self.direct_response:
                # Training direct-response path consumes detector-frame waveform only.
                # Skip unused amp/phase construction to reduce per-batch overhead.
                wf_dict = {
                    "waveform": waveform_payload,
                    "freqs": freqs,
                }
            else:
                waveform_payload = self._to_numpy(waveform_data)
                wf_dict = {
                    "waveform": waveform_payload,
                    "amp": np.abs(waveform_payload),
                    "phase": np.angle(waveform_payload),
                    "freqs": freqs,
                }
            if self.decenter_waveform:
                # Carry t_ref so the projection / detector transform can
                # re-apply exp(-i 2π f t_ref) and recover the physical strain.
                wf_dict["t_ref"] = self._to_numpy(t_ref)
                wf_dict["decentered"] = True
            if self.timing_profile:
                self._timing_add("amp_package", time.perf_counter() - t0)
                self._timing_finalize("amp", time.perf_counter() - total_t0)
            
            return wf_dict
            
        except Exception as e:
            if catch_waveform_errors:
                warnings.warn(f"Waveform generation failed: {e}")
                # Default to three LISA channels when shape inference is not available.
                nan_shape = (3, len(freqs))
                return {
                    "waveform": np.full(nan_shape, np.nan, dtype=np.complex128),
                    "amp": np.full(nan_shape, np.nan),
                    "phase": np.full(nan_shape, np.nan),
                    "freqs": freqs,
                }
            else:
                raise

    def generate_amp_phase_m(
        self, parameters: Dict[str, float]
    ) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        """Generate intrinsic amplitude/phase/tf data per (l, m) mode.

        Parameters
        ----------
        parameters: Dict[str, float]
            Waveform parameters

        Returns
        -------
        pol_lm: Dict[Tuple[int, int], Dict[str, np.ndarray]]
            Mode dictionary compatible with ``ProjectOntoSpaceDetectors``.
        """
        parameters = parameters.copy()

        # Convert LISA detector parameters if needed
        if self.frozenLISA:
            parameters = lisatools.convert_Lframe_to_SSBframe(
                parameters, t0=0.0, frozenLISA=True
            )

        parsed = self._parse_parameters(parameters)
        m1 = parsed["m1"]
        m2 = parsed["m2"]
        chi1z = parsed["chi1z"]
        chi2z = parsed["chi2z"]
        distance_si = parsed["distance_mpc"] * PC_SI * 1e6
        t_ref = parsed["t_ref"]

        # Match the LISABeta path: generate mode data on a coarse grid here, then let
        # ProjectOntoSpaceDetectors interpolate onto the full domain frequencies.
        freqs = self._build_bbhx_frequency_grid()
        if self.use_gpu:
            freqs = self.waveform_gen.xp.asarray(freqs)
        phi_ref_amp_phase = np.zeros_like(np.atleast_1d(m1), dtype=float)
        self.waveform_gen.amp_phase_gen(
            m1,
            m2,
            chi1z,
            chi2z,
            distance_si,
            phi_ref_amp_phase,
            self.f_ref,
            t_ref,
            length=self.bbhx_length,
            freqs=freqs,
            modes=self.mode_list,
            direct=True,
        )

        amp = self._to_numpy(self.waveform_gen.amp_phase_gen.amp)
        phase = self._to_numpy(self.waveform_gen.amp_phase_gen.phase)
        tf = self._to_numpy(self.waveform_gen.amp_phase_gen.tf)
        freqs_shaped = self._to_numpy(self.waveform_gen.amp_phase_gen.freqs_shaped)

        # Remove the binary dimension for single-sample generation.
        if amp.ndim == 3 and amp.shape[0] == 1:
            amp = amp[0]
            phase = phase[0]
            tf = tf[0]
            freqs_shaped = freqs_shaped[0]

        pol_lm: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        for mode_idx, lm in enumerate(self.mode_list):
            pol_lm[lm] = {
                "freq": np.asarray(freqs_shaped[mode_idx]),
                "amp": np.asarray(amp[mode_idx]),
                "phase": np.asarray(phase[mode_idx]),
                "tf": np.asarray(tf[mode_idx]),
            }

        return pol_lm
