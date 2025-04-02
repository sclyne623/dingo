import numpy as np
import lal
import lalsimulation as LS
import lisabeta.tools.pytools as pytools
import lisabeta.pyconstants as pyconstants


def linked_list_modes_to_dict_modes(hlm_ll):
    """Convert linked list of modes into dictionary with keys (l,m)."""
    hlm_dict = {}

    mode = hlm_ll.this
    while mode is not None:
        l, m = mode.l, mode.m
        hlm_dict[(l, m)] = mode.mode
        mode = mode.next

    return hlm_dict


def get_tapering_window_for_complex_time_series(h, tapering_flag: int = 1):
    """
    Get window for tapering of a complex time series from the lal backend. This is done
    by  tapering the time series with lal, and dividing tapered output by untapered
    input. lal does not support tapering of complex time series objects, so as a
    workaround we taper only the real part of the array and extract the window based on
    this.

    Parameters
    ----------
    h:
        complex lal time series object
    tapering_flag: int = 1
        Flag for tapering. See e.g. lines 2773-2777 in
            https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
            _l_a_l_sim_inspiral_waveform_taper_8c_source.html#l00222
        tapering_flag = 1 corresponds to LAL_SIM_INSPIRAL_TAPER_START

    Returns
    -------
    window: np.ndarray
        Array of length h.data.length, with the window used for tapering.
    """
    h_tapered = lal.CreateREAL8TimeSeries(
        "h_tapered", h.epoch, 0, h.deltaT, None, h.data.length
    )
    h_tapered.data.data = h.data.data.copy().real
    LS.SimInspiralREAL8WaveTaper(h_tapered.data, tapering_flag)
    eps = 1e-20 * np.max(np.abs(h.data.data))
    window = (np.abs(h_tapered.data.data) + eps) / (np.abs(h.data.data.real) + eps)
    # FIXME: using eps for numerical stability is not really robust here
    return window


def taper_td_modes_in_place(hlm_td, tapering_flag: int = 1):
    """
    Taper the time domain modes in place.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex lal time series objects for the
        corresponding modes.
    tapering_flag: int = 1
        Flag for tapering. See e.g. lines 2773-2777 in
            https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
            _l_a_l_sim_inspiral_waveform_taper_8c_source.html#l00222
        tapering_flag = 1 corresponds to LAL_SIM_INSPIRAL_TAPER_START
    """
    for _, h in hlm_td.items():
        window = get_tapering_window_for_complex_time_series(h, tapering_flag)
        h.data.data *= window


def td_modes_to_fd_modes(hlm_td, domain):
    """
    Transform dict of td modes to dict of fd modes via FFT. The td modes are expected
    to be tapered.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex lal time series objects for the
        corresponding tapered modes.
    domain: dingo.gw.domains.UniformFrequencyDomain
        Target domain after FFT.

    Returns
    -------
    hlm_fd: dict
        Dictionary with (l,m) keys and numpy arrays with the corresponding modes as
        values.
    """
    hlm_fd = {}

    delta_f = domain.delta_f
    delta_t = 0.5 / domain.f_max
    f_nyquist = domain.f_max  # use f_max as f_nyquist
    chirplen = int(2 * f_nyquist / delta_f)
    # sample frequencies, -f_max,...,-f_min,...0,...,f_min,...,f_max
    freqs = np.concatenate((-domain()[::-1], domain()[1:]), axis=0)
    # For even chirplength, we get chirplen + 1 output frequencies. However, the f_max
    # and -f_max bins are redundant, so we have chirplen unique bins.
    assert len(freqs) == chirplen + 1

    lal_fft_plan = lal.CreateForwardCOMPLEX16FFTPlan(chirplen, 0)
    for lm, h_td in hlm_td.items():
        assert np.abs(h_td.deltaT - delta_t) < 1e-12

        # resize data to chirplen by zero-padding or truncating
        # if chirplen < h_td.data.length:
        #     print(
        #         f"Specified frequency interval of {delta_f} Hz is too large "
        #         f"for a chirp of duration {h_td.data.length * delta_t} s with "
        #         f"Nyquist frequency {f_nyquist} Hz. The inspiral will be "
        #         f"truncated."
        #     )
        lal.ResizeCOMPLEX16TimeSeries(h_td, h_td.data.length - chirplen, chirplen)

        # Initialize a lal frequency series. We choose length chirplen + 1, while h_td is
        # only of length chirplen. This means, that the last bin h_fd.data.data[-1]
        # will not be modified by the lal FFT, and we have to copy over h_fd.data.data[0]
        # to h_fd.data.data[-1]. This corresponds to setting h(-f_max) = h(f_max).
        h_fd = lal.CreateCOMPLEX16FrequencySeries(
            "h_fd", h_td.epoch, 0, delta_f, None, chirplen + 1
        )
        # apply FFT
        lal.COMPLEX16TimeFreqFFT(h_fd, h_td, lal_fft_plan)
        assert np.abs(h_fd.deltaF - delta_f) < 1e-10
        assert np.abs(h_fd.f0 + domain.f_max) < 1e-6

        # time shift
        dt = (
            1.0 / h_fd.deltaF + h_fd.epoch.gpsSeconds + h_fd.epoch.gpsNanoSeconds * 1e-9
        )
        hlm_fd[lm] = h_fd.data.data * np.exp(-1j * 2 * np.pi * dt * freqs)
        # Set h(-f_max) = h(f_max), see above
        hlm_fd[lm][-1] = hlm_fd[lm][0]

    return hlm_fd


def get_polarizations_from_fd_modes_m(hlm_fd, iota, phase):
    pol_m = {}
    polarizations = ["h_plus", "h_cross"]

    for (l, m), h in hlm_fd.items():
        if m not in pol_m:
            pol_m[m] = {k: 0.0 for k in polarizations}
            pol_m[-m] = {k: 0.0 for k in polarizations}

        # In the L0 frame, we compute the polarizations from the modes using the
        # spherical harmonics below.
        ylm = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m)
        ylmstar = ylm.conjugate()

        # Modes (l,m) are defined on domain -f_max,...,-f_min,...0,...,f_min,...,f_max.
        # This splits up the frequency series into positive and negative frequency parts.
        if len(h) % 2 != 1:
            raise ValueError(
                "Even number of bins encountered, should be odd: -f_max,...,0,...,f_max."
            )
        offset = len(h) // 2
        h1 = h[offset:]
        h2 = h[offset::-1].conj()

        # Organize the modes such that pol_m[m] transforms as e^{- 1j * m * phase}.
        # This differs from the usual way, e.g.,
        #   https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
        #   _l_a_l_sim_inspiral_8c_source.html#l04801
        pol_m[m]["h_plus"] += 0.5 * h1 * ylm
        pol_m[-m]["h_plus"] += 0.5 * h2 * ylmstar
        pol_m[m]["h_cross"] += 0.5 * 1j * h1 * ylm
        pol_m[-m]["h_cross"] += -0.5 * 1j * h2 * ylmstar

    return pol_m


def get_starting_frequency_for_SEOBRNRv5_conditioning(parameters):
    """
    Compute starting frequency needed for having 3 extra cycles for tapering the TD modes.
    It returns the needed quantities to apply the standard LALSimulation conditioning routines to the TD modes.

    Parameters
    ----------
    parameters: dict
        Dictionary of parameters suited for GWSignal (obtained with NewInterfaceWaveformGenerator._convert_parameters)

    Returns
    ----------
    f_min: float
      Waveform starting frequency
    f_start: float
      New waveform starting frequency
    extra_time: float
      Extra time to take care of situations where the frequency is close to merger
    original_f_min: float
      Initial waveform starting frequency
    f_isco: float
      ISCO frequency
    """

    extra_time_fraction = (
        0.1  # fraction of waveform duration to add as extra time for tapering
    )
    extra_cycles = 3.0  # more extra time measured in cycles at the starting frequency

    f_min = parameters["f22_start"].value
    m1 = parameters["mass1"].value
    m2 = parameters["mass2"].value
    S1z = parameters["spin1z"].value
    S2z = parameters["spin2z"].value
    original_f_min = f_min

    f_isco = 1.0 / (pow(9.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)
    if f_min > f_isco:
        f_min = f_isco

    # upper bound on the chirp time starting at f_min
    tchirp = LS.SimInspiralChirpTimeBound(
        f_min, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, S1z, S2z
    )
    # upper bound on the final black hole spin */
    spinkerr = LS.SimInspiralFinalBlackHoleSpinBound(S1z, S2z)
    # upper bound on the final plunge, merger, and ringdown time */
    tmerge = LS.SimInspiralMergeTimeBound(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
    ) + LS.SimInspiralRingdownTimeBound((m1 + m2) * lal.MSUN_SI, spinkerr)

    # extra time to include for all waveforms to take care of situations where the frequency is close to merger (and is sweeping rapidly): this is a few cycles at the low frequency
    textra = extra_cycles / f_min
    # compute a new lower frequency
    f_start = LS.SimInspiralChirpStartFrequencyBound(
        (1.0 + extra_time_fraction) * tchirp + tmerge + textra,
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
    )

    f_isco = 1.0 / (pow(6.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)

    return f_min, f_start, extra_time_fraction * tchirp + textra, original_f_min, f_isco


def taper_td_modes_for_SEOBRNRv5_extra_time(
    h, extra_time, f_min, original_f_min, f_isco
):
    """
    Apply standard tapering procedure mimicking LALSimulation routine (https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_generator_conditioning_8c.html#ac78b5fcdabf8922a3ac479da20185c85)

    Parameters
    ----------
    h:
        complex gwpy TimeSeries object
    extra_time: float
        Extra time to take care of situations where the frequency is close to merger
    f_min: float
        Starting frequency employed in waveform generation
    original_f_min: float
        Initial starting frequency requested by the user
    f_isco:
        ISCO frequency

    Returns
    ----------
    h_return
        complex lal timeseries object
    """

    # Split in real and imaginary parts, since LAL conditioning routines are for real timeseries
    h_tapered_re = lal.CreateREAL8TimeSeries(
        "h_tapered", h.epoch.value, 0, h.dt.value, None, len(h)
    )
    h_tapered_re.data.data = h.value.copy().real

    h_tapered_im = lal.CreateREAL8TimeSeries(
        "h_tapered_im", h.epoch.value, 0, h.dt.value, None, len(h)
    )
    h_tapered_im.data.data = h.value.copy().imag

    # condition the time domain waveform by tapering in the extra time at the beginning and high-pass filtering above original f_min
    LS.SimInspiralTDConditionStage1(
        h_tapered_re, h_tapered_im, extra_time, original_f_min
    )
    # final tapering at the beginning and at the end to remove filter transients
    # waveform should terminate at a frequency >= Schwarzschild ISCO
    # so taper one cycle at this frequency at the end; should not make
    # any difference to IMR waveforms */
    LS.SimInspiralTDConditionStage2(h_tapered_re, h_tapered_im, f_min, f_isco)

    # Construct complex timeseries
    h_return = lal.CreateCOMPLEX16TimeSeries(
        "h_return",
        h_tapered_re.epoch,
        0,
        h_tapered_re.deltaT,
        None,
        h_tapered_re.data.length,
    )

    h_return.data.data = h_tapered_re.data.data + 1j * h_tapered_im.data.data

    # return timeseries
    return h_return


def FrequencyBoundsLISATDI_SMBH(params, timetomerger_max=1., minf=1e-5, maxf=1., fstart22=None, fend22=None, tmin=None, tmax=None, Mfmax_model=0.3, DeltatL_cut=None, DeltatSSB_cut=None, t0=0., tref=0., phiref=0., fref_for_phiref=0., fref_for_tref=0., force_phiref_fref=True, toffset=0., frozenLISA=False, scale_freq_hm=True, modes=None, f_t_acc=1e-6, approximant='IMRPhenomD', **kwargs):
    """PULLED DIRECTLY FROM LISABETA CODE.  COULD PROB GO IN WFG_UTILS.PY"""
    
    
    """Helper function to get the (2,2) frequency bounds in GenerateLISATDI_SMBH
    Args:
      params              # Dictionary of input parameters for the signal
                            Can be in SSB-frame or L-frame, if the latter it
                            will be converted back to SSB-frame
      Params dictionary keys format:
       [m1,               # Redshifted mass of body 1 (solar masses)
        m2,               # Redshifted mass of body 2 (solar masses)
        chi1,             # Dimensionless spin of body 1 (in [-1, 1])
        chi2,             # Dimensionless spin of body 2 (in [-1, 1])

        Deltat,           # Time shift (s)
        dist,             # Luminosity distance (Mpc)
        inc,              # Inclination angle (rad)
        phi,              # Observer's azimuthal phase (rad)
        lambda,           # Source longitude (rad)
        beta,             # Source latitude (rad)
        psi]              # Polarization angle (rad)

    Keyword args:
      timetomerger_max    # Time to merger (yr) to set an fstart cutting the
                            waveform (None to ignore, default 1.)
      minf                # Minimal frequency (Hz)
      maxf                # Maximal frequency (Hz)
      fstart22            # Starting frequency for 22 mode (Hz) (None to ignore)
      fend22              # Ending frequency for 22 mode (Hz) (None to ignore)
      tmin                # Starting time of observation window (yr),
                            SSB time absolute (default None, ignore)
      tmax                # Ending time of observation window (yr),
                            SSB time absolute (default None, ignore)
      Mfmax_model         # Max geometric frequency generated by the waveform
                            model (default 0.3) -- for 22, scaled for HM if
                            scale_freq_hm
      DeltatL_cut         # Ending time of observations (s), NOTE: L-frame
                            time measured from t0 (default None, ignore)
      DeltatSSB_cut       # [For testing only] Ending time of observations (s),
                            NOTE: SSB-frame time measured from t0
                            (default None,ignore)
      scale_freq_hm       # Scale freq cuts by m/2 (default True)
                            If False, compute f_lm(t) by inversion separately
                            the output will then be a dictionary across modes
      modes               # Set of modes, used when scale_freq_hm is False
      t0                  # Reference time (yr), so that t=0 for the waveform
                            corresponds to t0 in the SSB-frame
      tref                # Time at fref_for_tref (s) (default 0)
      phiref              # Orbital phase at fref_for_phiref (rad) (default 0)
      fref_for_tref       # Ref. frequency (Hz) for tref (default 0 for fpeak)
      fref_for_phiref     # Ref. frequency (Hz) for phiref (default 0 for fpeak)
      force_phiref_fref   # Flag to force phiref at fref (default True)
      toffset             # Extra time shift applied to the waveform (s)
      f_t_acc             # Target accuracy in time (s) for f(t) inversion
      frozenLISA          # Freeze LISA motion
    """
    
    dict_approximants_modes = {
    'IMRPhenomD': [(2,2)],
    'IMRPhenomHM': [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)],
    'IMRPhenomX': [(2,2)],
    'IMRPhenomXHM': [(2,2), (2,1), (3,3), (3,2), (4,4)],
    'EOBNRv2HMROM': [(2,2), (2,1), (3,3), (4,4), (5,5)],
    'SEOBNRv4HMROM': [(2,2), (2,1), (3,3), (4,4), (5,5)],
    'SEOBNRv5HMROM': [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (5,5)],
}

    # If input parameters are given in the Lframe, convert to SSBframe
    if params.get('Lframe', False):
        params = lisatools.convert_Lframe_to_SSBframe(params,
                                                      t0=t0,
                                                      frozenLISA=frozenLISA)

    params = pytools.complete_mass_params(params)
    params = pytools.complete_spin_params(params)

    m1 = params['m1']
    m2 = params['m2']
    chi1 = params['chi1']
    chi2 = params['chi2']
    Deltat = params['Deltat']
    dist = params['dist']
    inc = params['inc']
    phi = params['phi']
    lambd = params['lambda']
    beta = params['beta']
    psi = params['psi']

    # Units
    M = m1 + m2
    Ms = M * pyconstants.MTSUN_SI

    # Set of harmonics to be returned, default for approximants
    # TODO: add error raising if incompatibility with mode content of approx.
    if modes is None:
        modes = dict_approximants_modes[approximant]

    # Check logic of the arguments
    # Presence of higher harmonics
    hm = (not (modes==[(2,2)]))
    # Presence of a time cut with numerical inversion f(t)
    time_cut_foft = (tmin is not None) or (tmax is not None) or (DeltatL_cut is not None) or (DeltatSSB_cut is not None)
    # If True result will be separate as a dict across modes
    separate_hm = hm and not scale_freq_hm
    # Presence of a cut in 22-mode frequency
    f22_cut = ((not fstart22 is None) or (not fend22 is None))
    # NOTE: when HM are present, cut in f22 and cut in time incompatible, unless using scaling relation m/2*f22
    if f22_cut and not scale_freq_hm:
        raise ValueError('Cut in f22 can only be used with the scaling relation m/2*f22, no input for separate flm yet.')

    # Will remain a number if not separate_hm, else will be a dict over modes
    # We start from minf, maxf and will update the bounds
    # We will come back and re-enforce minf,maxf across all modes at the end
    fLow = minf
    fHigh = maxf

    # Maximal frequency covered by the waveform model
    fHigh = np.fmin(fHigh, Mfmax_model / Ms)

    # Cuts expressed directly in the 22-mode frequency
    # minf, maxf are global across modes while fstart22, fend22 will be scaled
    # Incompatible with separate_hm for now (need input fstart_lm, fend_lm)
    if not fstart22 is None:
        fLow = np.fmax(fLow, fstart22)
    if not fend22 is None:
        fHigh = np.fmin(fHigh, fend22)

    # Newtonian estimate for the starting frequency given a max duration of signal
    # If output is going to be separate for each mode, scale this by m/2
    if timetomerger_max is not None:
        f_timetomerger_max = pytools.funcNewtonianfoft(m1, m2, timetomerger_max * pyconstants.YRSID_SI)
        fLow = np.fmax(f_timetomerger_max, fLow)

    # Take into account time cuts if specified - we need a mock waveform class for f(t) inversion
    # TODO: this repeats the initialization of the waveform, not optimal
    # NOTE: for now, here we use PhenomD/HM regardless of the approximant asked for
    # NOTE: PhenomD has more alignment options than PhenomHM, risk for inconsistency
    if time_cut_foft:
        mock_gridfreq = np.array([fLow, fHigh])
        if approximant in ['IMRPhenomD', 'IMRPhenomHM', 'EOBNRv2HMROM', 'SEOBNRv4HMROM', 'SEOBNRv5HMROM']: # NOTE: for EOB/SEOB, we use PhenomHM here
            if (not hm) or scale_freq_hm:
                mock_wfClass = pyIMRPhenomD.IMRPhenomDh22AmpPhase(mock_gridfreq, m1, m2, chi1, chi2, dist, tref=tref, phiref=phiref, fref_for_tref=fref_for_tref, fref_for_phiref=fref_for_phiref, force_phiref_fref=force_phiref_fref, Deltat=Deltat)
            else:
                mock_wfClass = pyIMRPhenomHM.IMRPhenomHMhlmAmpPhase(mock_gridfreq, m1, m2, chi1, chi2, dist, phiref=phiref, fref=fref_for_phiref, Deltat=Deltat)
        if approximant in ['IMRPhenomX', 'IMRPhenomXHM']:
            if (not hm) or scale_freq_hm:
                mock_wfClass = pyIMRPhenomX.IMRPhenomXh22AmpPhase(mock_gridfreq, m1, m2, chi1, chi2, dist, Deltat=Deltat, fref=fref_for_phiref, phiref=phiref)
            else:
                mock_wfClass = pyIMRPhenomXHM.IMRPhenomXHMhlmAmpPhase(mock_gridfreq, m1, m2, chi1, chi2, dist, Deltat=Deltat, fref=fref_for_phiref, phiref=phiref)
        mock_tpeak = mock_wfClass.get_tpeak()
        mock_fpeak = mock_wfClass.get_fpeak()

    # At this stage fLow,fHigh are still single numbers
    # After this fLow,fHigh are either single numbers or a dict depending on separate_hm
    if separate_hm:
        fLow = dict([(lm, lm[1]/2. * fLow) for lm in modes])
        fHigh = dict([(lm, lm[1]/2. * fHigh) for lm in modes])

    # NOTE: times in the waveform are relative to t0, hence [tmin-t0, tmax-t0]
    # NOTE: DeltatL_cut is given in the L-frame, convert to SSB-frame
    # NOTE: DeltatSSB_cut in SSB-frame, for testing only (not physical)
    Deltatmin_s = None
    Deltatmax_s = None
    if tmin is not None:
        Deltatmin_s = (tmin - t0) * pyconstants.YRSID_SI
    if (tmax is not None) or (DeltatL_cut is not None) or (DeltatSSB_cut is not None):
        Deltatmax_s = np.infty
        if tmax is not None:
            Deltatmax_s = np.fmin(Deltatmax_s, (tmax-t0) * pyconstants.YRSID_SI)
        if (DeltatL_cut is not None):
            if (DeltatSSB_cut is not None):
                raise ValueError('DeltatL_cut and DeltatSSB_cut are exclusive.')
            tL_cut = t0*pyconstants.YRSID_SI + DeltatL_cut
            Deltat_cut = lisatools.tSSBfromLframe(tL_cut, lambd, beta, frozenLISA=frozenLISA, tfrozenLISA=t0) - t0*pyconstants.YRSID_SI
            Deltatmax_s = np.fmin(Deltatmax_s, Deltat_cut)
        if (DeltatSSB_cut is not None):
            if (DeltatL_cut is not None):
                raise ValueError('DeltatL_cut and DeltatSSB_cut are exclusive.')
            Deltatmax_s = np.fmin(Deltatmax_s, DeltatSSB_cut)

    # Compute f(t) for time cut at the beginning
    if Deltatmin_s is not None:
        if not separate_hm:
            fLow_tmin = mock_wfClass.compute_foft(Deltatmin_s, fLow, f_t_acc)
            fLow = np.fmax(fLow, fLow_tmin)
        else:
            tf_lm = dict([(lm, Deltatmin_s) for lm in modes])
            fLow_tmin_lm = mock_wfClass.compute_foft_lm(tf_lm, fLow[(2,2)], f_t_acc)
            fLow = dict([(lm, np.fmax(fLow_tmin_lm[lm], fLow[lm])) for lm in modes])

    # Compute f(t) for time cut at the end
    # NOTE: the cut in tmax is ignored if tmax > tpeak
    if Deltatmax_s is not None and (Deltatmax_s < mock_tpeak):
        fHigh_tmax_guess = pytools.funcNewtonianfoft(m1, m2, mock_tpeak - Deltatmax_s)
        fHigh_tmax_guess = np.fmin(fHigh_tmax_guess, mock_fpeak)
        if not separate_hm:
            fHigh_tmax = mock_wfClass.compute_foft(Deltatmax_s, fHigh_tmax_guess, f_t_acc)
            fHigh = np.fmin(fHigh, fHigh_tmax)
        else:
            tf_lm = dict([(lm, Deltatmax_s) for lm in modes])
            fHigh_tmax_lm = mock_wfClass.compute_foft_lm(tf_lm, fHigh_tmax_guess, f_t_acc)
            fHigh = dict([(lm, np.fmin(fHigh_tmax_lm[lm], fHigh[lm])) for lm in modes])

    if not separate_hm:
        if fLow >= fHigh:
            print("fLow > fHigh [%f > %f]"%(fLow, fHigh))
            raise SignalEmptyValueError
    else:
        if fLow[(2,2)] >= fHigh[(2,2)]:
            print("fLow > fHigh [%f > %f]"%(fLow, fHigh))
            raise SignalEmptyValueError

    # Re-enforce minf, maxf limits when separating modes
    if separate_hm:
        fLow = dict([(lm, np.fmax(minf, fLow[lm])) for lm in modes])
        fHigh = dict([(lm, np.fmin(maxf, fHigh[lm])) for lm in modes])

    return fLow, fHigh
