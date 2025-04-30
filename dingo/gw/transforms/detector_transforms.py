import math
import numpy as np
import torch
import pandas as pd
from bilby.gw.detector.interferometer import Interferometer
from lal import GreenwichMeanSiderealTime
from typing import Union
from bilby.gw.detector import calibration
from bilby.gw.prior import CalibrationPriorDict

import lisabeta.lisa.pyresponse as pyresponse
import lisabeta.tools.pyspline as pyspline
import ast


CC = 299792458.0


def time_delay_from_geocenter(
    ifo: Interferometer,
    ra: Union[float, np.ndarray, torch.Tensor],
    dec: Union[float, np.ndarray, torch.Tensor],
    time: float,
):
    """
    Calculate time delay between ifo and geocenter. Identical to method
    ifo.time_delay_from_geocenter(ra, dec, time), but the present implementation allows
    for batched computation, i.e., it also accepts arrays and tensors for ra and dec.

    Implementation analogous to bilby-cython implementation
    https://git.ligo.org/colm.talbot/bilby-cython/-/blob/main/bilby_cython/geometry.pyx,
    which is in turn based on XLALArrivaTimeDiff in TimeDelay.c.

    Parameters
    ----------
    ifo: bilby.gw.detector.interferometer.Interferometer
        bilby interferometer object.
    ra: Union[float, np.array, torch.Tensor]
        Right ascension of the source in radians. Either float, or float array/tensor.
    dec: Union[float, np.array, torch.Tensor]
        Declination of the source in radians. Either float, or float array/tensor.
    time: float
        GPS time in the geocentric frame.

    Returns
    -------
    float: Time delay between the two detectors in the geocentric frame
    """
    # check that ra and dec are of same type and length
    if type(ra) != type(dec):
        raise ValueError(
            f"ra type ({type(ra)}) and dec type ({type(dec)}) don't match."
        )
    if isinstance(ra, (np.ndarray, torch.Tensor)):
        if len(ra.shape) != 1:
            raise ValueError(f"Only one axis expected for ra and dec, got multiple.")
        if ra.shape != dec.shape:
            raise ValueError(
                f"Shapes of ra ({ra.shape}) and dec ({dec.shape}) don't match."
            )

    if isinstance(ra, (float, np.float32, np.float64)):
        return ifo.time_delay_from_geocenter(ra, dec, time)

    elif isinstance(ra, (np.ndarray, torch.Tensor)) and len(ra) == 1:
        return ifo.time_delay_from_geocenter(ra[0], dec[0], time)

    else:
        if isinstance(ra, np.ndarray):
            sin = np.sin
            cos = np.cos
        elif isinstance(ra, torch.Tensor):
            sin = torch.sin
            cos = torch.cos
        else:
            raise NotImplementedError(
                "ra, dec must be either float, np.ndarray, or torch.Tensor."
            )

        gmst = math.fmod(GreenwichMeanSiderealTime(float(time)), 2 * np.pi)
        phi = ra - gmst
        theta = np.pi / 2 - dec
        sintheta = sin(theta)
        costheta = cos(theta)
        sinphi = sin(phi)
        cosphi = cos(phi)
        detector_1 = ifo.vertex
        detector_2 = np.zeros(3)
        return (
            (detector_2[0] - detector_1[0]) * sintheta * cosphi
            + (detector_2[1] - detector_1[1]) * sintheta * sinphi
            + (detector_2[2] - detector_1[2]) * costheta
        ) / CC

def process_transfer(freq_grid,amp,phase,tf,t0,l,m,inc,phi,lambd, beta, psi,interp_freqs,f_min,detector_type,LISAconst, 
                    responseapprox, frozenLISA,TDIrescaled):
                    """helper function for the ProjectontoSpaceDetector class.  The class is designed to work with a
                     multidimensional array that comes with batching, but the actual transfer function code only 
                     likes 1D arrays, so that part gets broken out here.  Currently needs f_min argument that should be addressed
                     elsewhere.
                    """
                    
                    
                    #Class called for each waveform dependent on extrinsic parameters 
                    #Also where detector settings are added
                    tdiClass = pyresponse.LISAFDresponseTDI3Chan(freq_grid, tf, 
                                                         t0, l, m, inc, phi, lambd, beta, psi, 
                                                         detector_type,LISAconst, responseapprox, frozenLISA, 
                                                         TDIrescaled)
                    
                    #Get Transfer function that amp-phase waveform gets multiplied with
                    phaseRdelay, transferL1, transferL2, transferL3 = tdiClass.get_response() 
                    
                    #calculate complex amplitudes and new phase

                    camp1 = amp * transferL1
                    camp2 = amp * transferL2
                    camp3 = amp * transferL3
                    phasetot = phase + phaseRdelay #PhaseRDelay to account for detector length delays
                    
                    #Break each amplitude into real and complex for each channel and interpolate
                    amp_real_chan1 = np.copy(np.real(camp1))
                    amp_imag_chan1 = np.copy(np.imag(camp1))
                    amp_real_chan2 = np.copy(np.real(camp2))
                    amp_imag_chan2 = np.copy(np.imag(camp2))
                    amp_real_chan3 = np.copy(np.real(camp3))
                    amp_imag_chan3 = np.copy(np.imag(camp3))
                    #Interpolation.  Need to initialize classes to do this currently.
                    spline_amp_real_chan1Class = pyspline.CubicSpline(freq_grid, amp_real_chan1)
                    spline_amp_imag_chan1Class = pyspline.CubicSpline(freq_grid, amp_imag_chan1)
                    spline_amp_real_chan2Class = pyspline.CubicSpline(freq_grid, amp_real_chan2)
                    spline_amp_imag_chan2Class = pyspline.CubicSpline(freq_grid, amp_imag_chan2)
                    spline_amp_real_chan3Class = pyspline.CubicSpline(freq_grid, amp_real_chan3)
                    spline_amp_imag_chan3Class = pyspline.CubicSpline(freq_grid, amp_imag_chan3)
                    spline_phaseClass = pyspline.CubicSpline(freq_grid, phasetot)
                    
                    spline_amp_real_chan1 = spline_amp_real_chan1Class.get_spline()
                    spline_amp_imag_chan1 = spline_amp_imag_chan1Class.get_spline()
                    spline_amp_real_chan2 = spline_amp_real_chan2Class.get_spline()
                    spline_amp_imag_chan2 = spline_amp_imag_chan2Class.get_spline()
                    spline_amp_real_chan3 = spline_amp_real_chan3Class.get_spline()
                    spline_amp_imag_chan3 = spline_amp_imag_chan3Class.get_spline()
                    spline_phase = spline_phaseClass.get_spline()
                    # Evaluate splines
                    ampreal_chan1 = pyspline.spline_eval_vector(spline_amp_real_chan1, interp_freqs, extrapol_zero=True)
                    ampimag_chan1 = pyspline.spline_eval_vector(spline_amp_imag_chan1, interp_freqs, extrapol_zero=True)
                    ampreal_chan2 = pyspline.spline_eval_vector(spline_amp_real_chan2, interp_freqs, extrapol_zero=True)
                    ampimag_chan2 = pyspline.spline_eval_vector(spline_amp_imag_chan2, interp_freqs, extrapol_zero=True)
                    ampreal_chan3 = pyspline.spline_eval_vector(spline_amp_real_chan3, interp_freqs, extrapol_zero=True)
                    ampimag_chan3 = pyspline.spline_eval_vector(spline_amp_imag_chan3, interp_freqs, extrapol_zero=True)
                    phase = pyspline.spline_eval_vector(spline_phase, interp_freqs, extrapol_zero=True)
                    # Get complex values for the TDI freqseries
                    eiphase = np.exp(1j*phase)
                    tdi_chan1_vals = (ampreal_chan1 + 1j*ampimag_chan1) * eiphase
                    tdi_chan2_vals = (ampreal_chan2 + 1j*ampimag_chan2) * eiphase
                    tdi_chan3_vals = (ampreal_chan3 + 1j*ampimag_chan3) * eiphase

                    #Set waveforms = 0 below fmin.  This should happen elsewhere
                    tdi_chan1_vals[interp_freqs < f_min] = 0.
                    tdi_chan2_vals[interp_freqs < f_min] = 0.
                    tdi_chan3_vals[interp_freqs < f_min] = 0.


                    return tdi_chan1_vals, tdi_chan2_vals, tdi_chan3_vals


class GetDetectorTimes(object):
    """
    Compute the time shifts in the individual detectors based on the sky
    position (ra, dec), the geocent_time and the ref_time.
    """

    def __init__(self, ifo_list, ref_time):
        self.ifo_list = ifo_list
        self.ref_time = ref_time

    def __call__(self, input_sample):
        sample = input_sample.copy()
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        ra = extrinsic_parameters["ra"]
        dec = extrinsic_parameters["dec"]
        geocent_time = extrinsic_parameters["geocent_time"]
        for ifo in self.ifo_list:
            if type(ra) == torch.Tensor:
                # computation does not work on gpu, so do it on cpu
                ra = ra.cpu()
                dec = dec.cpu()
            dt = time_delay_from_geocenter(ifo, ra, dec, self.ref_time)
            if type(dt) == torch.Tensor:
                dt = dt.to(geocent_time.device)
            ifo_time = geocent_time + dt
            extrinsic_parameters[f"{ifo.name}_time"] = ifo_time
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class ProjectOntoSpaceDetectors(object):
    """
    Project the GW onto the detectors in ifo_list (AET for LISA). This does
    not sample any new parameters, but relies on the parameters provided in
    sample['extrinsic_parameters']. Specifically, this transform applies the
    following operations:
    
    Also Changed ProjectOntoDetectors to take lisa_settings as init.  
    This is a way for me to add extra settings needed for transfer function.
    
    Also instead of ifo_list takes detector_type

    (1) Rescale GW amplitudes to account for sampled luminosity distance
    (2) Generate Transfer function for each detector using the extrinsic parameters
    (3) Project each mode onto the LISA detectors as T*A*np.exp(1j*phase)
    (4) Sum modes to get final waveform for each detector
    
    """

    def __init__(self, detector_type, domain, ref_time, lisa_settings):
        self.detector_type = detector_type
        self.domain = domain
        self.ref_time = ref_time
        self.LISAconst = lisa_settings["LISAconst"]
        self.responseapprox = lisa_settings["responseapprox"]
        self.frozenLISA = lisa_settings["frozenLISA"]
        self.TDIrescaled = lisa_settings["TDIrescaled"]
    

    def __call__(self, input_sample):
        
        sample = input_sample.copy()
        for lm in sample["waveform"].keys():
            l = lm[0]
            m = lm[1]
            
        
        
        
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        
        try:
            d_ref = parameters["dist"]
            d_new = extrinsic_parameters.pop("dist")
            beta = extrinsic_parameters.pop("beta")
            inc = extrinsic_parameters.pop("inc")
            lambd = extrinsic_parameters.pop("lambda")
            psi = extrinsic_parameters.pop("psi")
            tc_ref = parameters["geocent_time"]
            phi = parameters["phi"]
            assert np.allclose(tc_ref, 0.0), (
                "This should always be 0. If for some reason "
                "you want to save time shifted polarizations,"
                " then remove this assert statement."
            )
            tc_new = extrinsic_parameters.pop("geocent_time")
        except:
            raise ValueError("Missing parameters.")
        
        #Hard Code for now need to confirm this is geocent_time
        t0=0.
        arr_len = len(d_new)
        
        # (1) rescale polarizations and set distance parameter to sampled value
        if np.isscalar(d_ref) or np.isscalar(d_new):
            d_ratio = d_ref / d_new
        elif isinstance(d_ref, np.ndarray) and isinstance(d_new, np.ndarray):
            d_ratio = (d_ref / d_new)[:, np.newaxis]
        else:
            raise ValueError("luminosity_distance should be a float or a numpy array.")
        
        
        
        
        interp_freqs = self.domain.sample_frequencies
                

        
        interp_freqs = interp_freqs.astype(np.float64)
     

        #Everything gets done in a single loop here
        # 1. Rescale waveform amplitude 
        # 2. Loop through each waveform and calculate transfer function.  Each transfer function is 
        #        multiplied with the waveform amplitude to generate a complex and real amplitude.
        #        PhaseRDelay is also added to phase to address delay from signal arriving at SSB to 
        #        Signal arriving at LISA detector
        # 3. Interpolate each part of the signal (Re(A), im(A), phase) for each detector 
        # 4. compute mode strain as (Re(A) + i*im(A))*exp(i*phase)
        # 5. Add mode strain to total strain array to create a single waveform as sum of modes.
        
        chan1 = np.zeros((arr_len,len(interp_freqs)), dtype=np.complex128)
        chan2 = np.zeros((arr_len,len(interp_freqs)), dtype=np.complex128)
        chan3 = np.zeros((arr_len,len(interp_freqs)), dtype=np.complex128)
        
        for lm in sample["waveform"].keys():
            l, m = ast.literal_eval(lm)

            if len(d_ratio) ==1:
                sample["waveform"][lm]["amp"]=sample["waveform"][lm]["amp"]*d_ratio
            else:
                for i in range(len(d_ratio)): #Scale waveform according to distance
                
                    sample["waveform"][lm]["amp"][i] = sample["waveform"][lm]["amp"][i]*d_ratio[i] 
                
            
            #l = lm[0]
            #m = lm[1]

            #seperate workflows for single vs batched waveform
            if any(len(np.array(x)) == 1 for x in [inc,phi,lambd,beta]):
                
                freq_grid = sample["waveform"][lm]["freq"]
                amp = sample["waveform"][lm]["amp"][0]
                phase = sample["waveform"][lm]["phase"]
                #print("phase",phase)
                if isinstance(sample["waveform"][lm]["tf"], tuple):
                    tf = np.array(sample["waveform"][lm]["tf"][0])
                else:
                    tf = sample["waveform"][lm]["tf"]
                

                
                chan1_mode,chan2_mode,chan3_mode  = process_transfer(freq_grid,amp, phase,tf,t0,l,m,inc[0],phi[0],lambd[0], beta[0], psi[0],interp_freqs,self.domain.f_min,self.detector_type,self.LISAconst, 
                    self.responseapprox, self.frozenLISA,self.TDIrescaled)
                
                sample["waveform"][lm]["Chan1"] = chan1_mode
                sample["waveform"][lm]["Chan2"] = chan2_mode
                sample["waveform"][lm]["Chan3"] = chan3_mode
            else:
            
                #Calculate Transfer Functions using list comprehension
                mode_strains = [process_transfer(freq_grid,amp, phase,tf,t0,l,m,inc_,phi_,lambd_, beta_, psi_,interp_freqs,self.domain.f_min,self.detector_type,self.LISAconst, 
                        self.responseapprox, self.frozenLISA,self.TDIrescaled) for freq_grid,amp,phase, tf, inc_,phi_,lambd_, beta_, psi_ in zip(sample["waveform"][lm]["freq"],sample["waveform"][lm]["amp"],sample["waveform"][lm]["phase"],sample["waveform"][lm]["tf"],inc,phi,lambd,beta,psi)]

                #Probably dont need this but useful for checking
                sample["waveform"][lm]["Chan1"] = [i[0] for i in mode_strains]
                sample["waveform"][lm]["Chan2"] = [i[1] for i in mode_strains]
                sample["waveform"][lm]["Chan3"] = [i[2] for i in mode_strains]
            

            chan1+=sample["waveform"][lm]["Chan1"]
            chan2+=sample["waveform"][lm]["Chan2"]
            chan3+=sample["waveform"][lm]["Chan3"]

       
        strains = {"chan1": chan1,
                "chan2": chan2,
                "chan3":chan3
        }

        # Add extrinsic parameters corresponding to the transformations
        # applied in the loop above to parameters. These have all been popped off of
        # extrinsic_parameters, so they only live one place.
        parameters["inc"] = inc
        parameters["beta"] = beta
        parameters["lambda"] = lambd
        parameters["psi"] = psi
        parameters["geocent_time"] = tc_new
        parameters["dist"] = d_new
     
        sample["waveform"] = strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters
        
        return sample


class ProjectOntoDetectors(object):
    """
    Project the GW polarizations onto the detectors in ifo_list. This does
    not sample any new parameters, but relies on the parameters provided in
    sample['extrinsic_parameters']. Specifically, this transform applies the
    following operations:

    (1) Rescale polarizations to account for sampled luminosity distance
    (2) Project polarizations onto the antenna patterns using the ref_time and
        the extrinsic parameters (ra, dec, psi)
    (3) Time shift the strains in the individual detectors according to the
        times <ifo.name>_time provided in the extrinsic parameters.
    """

    def __init__(self, ifo_list, domain, ref_time):
        self.ifo_list = ifo_list
        self.domain = domain
        self.ref_time = ref_time

    def __call__(self, input_sample):
        sample = input_sample.copy()
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        try:
            d_ref = parameters["luminosity_distance"]
            d_new = extrinsic_parameters.pop("luminosity_distance")
            ra = extrinsic_parameters.pop("ra")
            dec = extrinsic_parameters.pop("dec")
            psi = extrinsic_parameters.pop("psi")
            tc_ref = parameters["geocent_time"]
            assert np.allclose(tc_ref, 0.0), (
                "This should always be 0. If for some reason "
                "you want to save time shifted polarizations,"
                " then remove this assert statement."
            )
            tc_new = extrinsic_parameters.pop("geocent_time")
        except:
            raise ValueError("Missing parameters.")

        # (1) rescale polarizations and set distance parameter to sampled value
        if np.isscalar(d_ref) or np.isscalar(d_new):
            d_ratio = d_ref / d_new
        elif isinstance(d_ref, np.ndarray) and isinstance(d_new, np.ndarray):
            d_ratio = (d_ref / d_new)[:, np.newaxis]
        else:
            raise ValueError("luminosity_distance should be a float or a numpy array.")
        hc = sample["waveform"]["h_cross"] * d_ratio
        hp = sample["waveform"]["h_plus"] * d_ratio
        parameters["luminosity_distance"] = d_new
        

        strains = {}
        for ifo in self.ifo_list:
            # (2) project strains onto the different detectors
            # TODO the Bilby cython functions are not vectorized, so for now
            # we just loop over the extrinsic parameters. This is not ideal
            # and eventually one should also vectorize these functions to
            # achieve optimal batching capabilities.
            if any(np.isscalar(x) for x in [ra, dec, psi]):
                fp = ifo.antenna_response(ra, dec, self.ref_time, psi, mode="plus")
                fc = ifo.antenna_response(ra, dec, self.ref_time, psi, mode="cross")
            else:
                fp = np.array(
                    [
                        ifo.antenna_response(ra, dec, self.ref_time, psi, mode="plus")
                        for ra, dec, psi in zip(ra, dec, psi)
                    ],
                    dtype=np.float32,
                )
                fc = np.array(
                    [
                        ifo.antenna_response(ra, dec, self.ref_time, psi, mode="cross")
                        for ra, dec, psi in zip(ra, dec, psi)
                    ],
                    dtype=np.float32,
                )
                fp = fp[..., np.newaxis]
                fc = fc[..., np.newaxis]
            strain = fp * hp + fc * hc

            # (3) time shift the strain. If polarizations are timeshifted by
            #     tc_ref != 0, undo this here by subtracting it from dt.
            dt = extrinsic_parameters[f"{ifo.name}_time"] - tc_ref
            strains[ifo.name] = self.domain.time_translate_data(strain, dt)

        # Add extrinsic parameters corresponding to the transformations
        # applied in the loop above to parameters. These have all been popped off of
        # extrinsic_parameters, so they only live one place.
        parameters["ra"] = ra
        parameters["dec"] = dec
        parameters["psi"] = psi
        parameters["geocent_time"] = tc_new
        for ifo in self.ifo_list:
            param_name = f"{ifo.name}_time"
            parameters[param_name] = extrinsic_parameters.pop(param_name)

        sample["waveform"] = strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample


class TimeShiftStrain(object):
    """
    Time shift the strains in the individual detectors according to the
    times <ifo.name>_time provided in the extrinsic parameters.
    """

    def __init__(self, ifo_list, domain):
        self.ifo_list = ifo_list
        self.domain = domain

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = input_sample["extrinsic_parameters"].copy()

        strains = {}

        if isinstance(input_sample["waveform"], dict):
            for ifo in self.ifo_list:
                # time shift the strain
                strain = input_sample["waveform"][ifo.name]
                dt = extrinsic_parameters.pop(f"{ifo.name}_time")
                strains[ifo.name] = self.domain.time_translate_data(strain, dt)

        elif isinstance(input_sample["waveform"], torch.Tensor):
            strains = input_sample["waveform"]
            dt = [extrinsic_parameters.pop(f"{ifo.name}_time") for ifo in self.ifo_list]
            dt = torch.stack(dt, 1)
            strains = self.domain.time_translate_data(strains, dt)

        else:
            raise NotImplementedError(
                f"Unexpected type {type(input_sample['waveform'])}, expected dict or "
                f"torch.Tensor"
            )

        sample["waveform"] = strains
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample


class ApplyCalibrationUncertainty(object):
    r"""
    Expand out a waveform using several detector calibration draws. These multiple
    draws are intended to be used for marginalizing over calibration uncertainty.

    Detector calibration uncertainty is modeled as described in
    https://dcc.ligo.org/LIGO-T1400682/public

    Gravitational wave data $d$ is assumed to be of the form

    $$d(f) = h_{obs}(f) + n(f),$$

    where $h_{obs}$ is the observed waveform and $n$ is the noise. Since the detector
    is not perfectly calibrated, the observed waveform is not identical to the true
    waveform $h(f)$. Rather, it is assumed to have corrections of the form

    $$h_{obs}(f) = h(f) * (1 + \delta A(f)) * \exp(i \delta \phi(f)) = h(f) * \alpha(f),$$

    where $\delta A(f)$ and $\delta \phi(f)$ are frequency-dependent amplitude and
    phase errors. Under the calibration model, these are parametrized with cubic
    splines, defined in terms of calibration parameters $A_i$ and $\phi_i$, defined
    at log-spaced frequency nodes,

    $$
    \delta A(f) &= \mathrm{spline}(f; {f_i, \delta A_i}), \\
    \delta \phi(f) &= \mathrm{spline}(f; {f_i, \delta \phi_i}).
    $$

    The calibration parameters are not known precisely, rather they are assumed to be
    normally distributed, with mean 0 and standard deviation  determined by the
    "calibration envelope", which varies from event to event.

    For each detector waveform, this transform draws a collection of $N$
    calibration curves $\{(\delta A^n(f), \delta \phi^n(f))\}_{n=1}^N$ according to a
    calibration envelope, and applies them to generate $N$ observed waveforms $\{h^n_{
    obs}(f)\}$. This is intended to be used for marginalizing over the calibration
    uncertainty when evaluating the likelihood for importance sampling.

    """

    def __init__(
        self,
        ifo_list,
        data_domain,
        calibration_envelope,
        num_calibration_curves,
        num_calibration_nodes,
        correction_type="data",
    ):
        r"""
        Parameters
        ---------

        ifo_list : InterferometerList
            List of Interferometers present in the analysis.
        data_domain : Domain
            Domain on which data is defined.
        calibration_envelope : dict
            Dictionary of the form ``{"H1": filepath, "L1": filepath}``,
            where the filepaths are strings pointing to ".txt" files containing
            calibration envelopes. The calibration envelope depends on the event analyzed,
            and therefore  remains fixed for all applications of the transform. The
            calibration envelope is used to define the variances $(\sigma_{\delta A_i},
            \sigma_{\delta \phi_i})$ of the calibration paramters.
        num_calibration_curves : int
            Number of calibration curves $N$ to produce and apply to the
            waveform. Ultimately, this will translate to the number of samples in the
            Monte Carlo estimate of the marginalized likelihood integral.
        num_calibration_nodes : int
            Number of log-spaced frequency nodes $f_i$ to use in defining the spline.
        correction_type : str = "data"
            It was discovered in Oct. 2024 that the calibration envelopes specified by
            the detchar group were not being used correctly by PE codes. According to
            the detchar group, envelopes are over $\eta$ which is defined as:

            $$
            h_{obs}(f) = \frac{1}{\eta} * h(f).
            $$

            Of course, $\frac{1}{\eta} = \alpha$. Previously, the envelopes were
            being used as if $\eta = \alpha$ which is wrong. Therefore, there is
            now an additional option where one can specify correction_type = "data"
            if the calibration envelopes are over $\eta$ and correction_type = "template"
            if the calibration envelopes are over $\alpha$.
        """

        self.ifo_list = ifo_list
        self.num_calibration_curves = num_calibration_curves

        self.data_domain = data_domain
        self.calibration_prior = {}
        if all([s.endswith(".txt") for s in calibration_envelope.values()]):
            # Generating .h5 lookup table from priors in .txt file
            self.calibration_envelope = calibration_envelope
            for ifo in self.ifo_list:
                # Setting calibration model to cubic spline
                ifo.calibration_model = calibration.CubicSpline(
                    f"recalib_{ifo.name}_",
                    minimum_frequency=data_domain.f_min,
                    maximum_frequency=data_domain.f_max,
                    n_points=num_calibration_nodes,
                )

                # Setting priors
                # What this will do is take the the calibration envelope and set
                # a spline on the median and sigma of the amplitude and phase.
                # Then in log frequency it will setup node points say at
                # frequency points, $f_i$.  Then for each node point f_i, it
                # will create a gaussian prior according to the spline of the
                # median and sigma found earlier
                self.calibration_prior[
                    ifo.name
                ] = CalibrationPriorDict.from_envelope_file(
                    self.calibration_envelope[ifo.name],
                    self.data_domain.f_min,
                    self.data_domain.f_max,
                    num_calibration_nodes,
                    ifo.name,
                    correction_type=correction_type,
                )

        else:
            raise Exception("Calibration envelope must be specified in a .txt file!")

    def __call__(self, input_sample):
        sample = input_sample.copy()
        for ifo in self.ifo_list:
            calibration_parameter_draws, calibration_draws = {}, {}
            # Sampling from prior
            calibration_parameter_draws[ifo.name] = pd.DataFrame(
                self.calibration_prior[ifo.name].sample(self.num_calibration_curves)
            )
            calibration_draws[ifo.name] = np.zeros(
                (
                    self.num_calibration_curves,
                    len(self.data_domain.sample_frequencies),
                ),
                dtype=complex,
            )

            for i in range(self.num_calibration_curves):
                calibration_draws[ifo.name][
                    i, self.data_domain.frequency_mask
                ] = ifo.calibration_model.get_calibration_factor(
                    self.data_domain.sample_frequencies[
                        self.data_domain.frequency_mask
                    ],
                    prefix="recalib_{}_".format(ifo.name),
                    **calibration_parameter_draws[ifo.name].iloc[i],
                )

            # Multiplying the sample waveform in the interferometer according to
            # the calibration curve.  This is done by following the perscription
            # here:
            #
            # https://dcc.ligo.org/LIGO-T1400682 Eq 3 and 4
            #
            # We take the waveform h(f) and multiply it by C = (1 + \delta A(f))
            # \exp(i \delta \psi) i.e. h_obs(f) = C * h(f)
            # Here C is "calibration_draws"

            # Padding 0's to everything in the calibration array which is below f_min

            sample["waveform"][ifo.name] = (
                sample["waveform"][ifo.name] * calibration_draws[ifo.name]
            )

        return sample
