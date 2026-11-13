"""
Coalescence-time handling for the lisabeta LISA response.

Conventions (identical to lisabeta, see lisabeta/lisa/lisa.py and lisatools.py):

* ``t0_s``: the reference SSB time in seconds. DINGO takes it from
  ``geocent_time`` (a fixed value in the LISA priors, ~1.5 yr). It places
  LISA on its orbit; it is NOT a free parameter.
* ``Deltat``: SSB-frame time offset, t_SSB = t0_s + Deltat. This is exactly
  lisabeta's ``Deltat`` waveform argument.
* ``DeltatL``: LISA-frame time offset, t_L = t0_s + DeltatL, where t_L is the
  arrival time at the centre of the constellation (lisabeta's L-frame time).

A time shift Deltat in lisabeta adds 2*pi*f*Deltat to every mode's phase and
Deltat to its t(f). The response depends on t(f) only through t(f) + t0_s, so
the same signal is obtained by generating at Deltat = 0, moving the orbital
reference time to t0_s + Deltat (moving LISA only), and multiplying the TDI
channels by exp(+2*pi*i*f*Deltat). The phase convention is lisabeta's
(h = A exp(+i phase)), which is the opposite sign to the ground-detector
``time_translate_data`` in DINGO's frequency domains.

With frozenLISA=True, lisabeta keeps LISA at t0_s whatever Deltat is, so only
the phase factor is applied and t_L uses the orientation at t0_s.

This module only depends on numpy, so it can be tested without lisabeta. The
constants are copied from lisabeta/constants.h; the cluster validation script
misc_scripts/lisa_time_shift_validation.py checks them against
lisabeta.pyconstants.
"""

import numpy as np

# lisabeta/constants.h
C_SI = 299792458.0
AU_SI = 149597870700.0
YRSID_SI = 31558149.763545600
EARTH_ORBIT_OMEGA_SI = 1.9909865927683785e-07
R_OVER_C = AU_SI / C_SI

TIME_SHIFT_KEYS = ("Deltat", "DeltatL")


def _orbital_phase(t0_s, deltat, frozen_lisa):
    """Orbital phase alpha = Omega * t of the constellation used by lisabeta."""
    if frozen_lisa:
        return EARTH_ORBIT_OMEGA_SI * t0_s + 0.0 * deltat
    return EARTH_ORBIT_OMEGA_SI * (t0_s + deltat)


def deltatL_from_deltat(deltat, lambd, beta, t0_s, frozen_lisa=False):
    """LISA-frame offset DeltatL for an SSB-frame offset Deltat.

    Same formula as lisabeta.lisa.lisatools.tLfromSSBframe with
    constellation_ini_phase = 0 and tfrozenLISA = t0_s, written in offsets from
    t0_s to keep full float64 precision.
    """
    deltat = np.asarray(deltat, dtype=np.float64)
    alpha = _orbital_phase(t0_s, deltat, frozen_lisa)
    out = deltat - R_OVER_C * np.cos(beta) * np.cos(alpha - lambd)
    return out if out.ndim else float(out)


def deltat_from_deltatL(deltatL, lambd, beta, t0_s, frozen_lisa=False, n_newton=3):
    """SSB-frame offset Deltat for a LISA-frame offset DeltatL.

    For a frozen LISA the relation is explicit. For a moving LISA it is
    implicit. The starting point is lisabeta's tSSBfromLframe (accurate to
    O(Omega^2 (R/c)^3) ~ 1e-5 s), refined by Newton steps on
    deltatL_from_deltat, so the round trip is exact to float64 precision.
    """
    deltatL = np.asarray(deltatL, dtype=np.float64)
    rc = R_OVER_C * np.cos(beta)
    if frozen_lisa:
        alpha = EARTH_ORBIT_OMEGA_SI * t0_s
        out = deltatL + rc * np.cos(alpha - lambd)
        return out if out.ndim else float(out)

    phase = EARTH_ORBIT_OMEGA_SI * (t0_s + deltatL) - lambd
    deltat = (
        deltatL
        + rc * np.cos(phase)
        - 0.5 * EARTH_ORBIT_OMEGA_SI * rc**2 * np.sin(2.0 * phase)
    )
    for _ in range(n_newton):
        phase = EARTH_ORBIT_OMEGA_SI * (t0_s + deltat) - lambd
        residual = deltat - rc * np.cos(phase) - deltatL
        derivative = 1.0 + rc * EARTH_ORBIT_OMEGA_SI * np.sin(phase)
        deltat = deltat - residual / derivative
    return deltat if deltat.ndim else float(deltat)


def time_shift_factor(frequencies, deltat):
    """exp(+2 pi i f Deltat), shaped to multiply strains of shape (n_f,) or (batch, n_f)."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    deltat = np.asarray(deltat, dtype=np.float64)
    if deltat.ndim == 0:
        return np.exp(2j * np.pi * frequencies * deltat)
    return np.exp(2j * np.pi * frequencies[np.newaxis, :] * deltat[:, np.newaxis])


def _is_zero(x):
    return x is None or np.all(np.asarray(x, dtype=np.float64) == 0.0)


def pop_time_shift(extrinsic_parameters, lambd, beta, t0_s, frozen_lisa=False):
    """Pop Deltat / DeltatL from extrinsic_parameters and return (Deltat, DeltatL).

    Returns (None, None) when neither key is present, so existing fixed-time
    configurations behave exactly as before. If both are present, Deltat must be
    zero (a fiducial value), and DeltatL defines the shift.
    """
    has_ssb = "Deltat" in extrinsic_parameters
    has_l = "DeltatL" in extrinsic_parameters
    if not (has_ssb or has_l):
        return None, None
    deltat = extrinsic_parameters.pop("Deltat", None)
    deltatL = extrinsic_parameters.pop("DeltatL", None)
    if has_ssb and has_l:
        if not _is_zero(deltat):
            raise ValueError(
                "Both Deltat and DeltatL were given with a nonzero Deltat; "
                "pass only one time offset."
            )
        deltat = None
    if deltatL is not None:
        deltat = deltat_from_deltatL(deltatL, lambd, beta, t0_s, frozen_lisa)
    else:
        deltatL = deltatL_from_deltat(deltat, lambd, beta, t0_s, frozen_lisa)
    deltat = np.asarray(deltat, dtype=np.float64)
    deltatL = np.asarray(deltatL, dtype=np.float64)
    if deltat.ndim == 0:
        return float(deltat), float(deltatL)
    return deltat, deltatL


def strip_time_shift_for_generator(parameters):
    """Generator inputs with the time shift removed (Deltat = 0, no DeltatL).

    The shift is applied by ProjectOntoSpaceDetectors; generating the modes
    with a nonzero Deltat as well would apply it twice.
    """
    parameters = dict(parameters)
    parameters.pop("DeltatL", None)
    parameters["Deltat"] = 0.0
    return parameters
