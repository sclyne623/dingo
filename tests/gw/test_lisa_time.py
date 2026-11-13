"""Tests for dingo.gw.lisa_time (numpy only; no lisabeta needed).

The module is loaded from its file so these tests also run where the full dingo
package (which imports lisabeta) cannot be imported.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "dingo" / "gw" / "lisa_time.py"
_spec = importlib.util.spec_from_file_location("lisa_time", _PATH)
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)

T0_S = 47337224.645318404  # geocent_time of the current LISA training prior
RNG = np.random.default_rng(20261113)
N = 2000
LAMBD = RNG.uniform(-np.pi, np.pi, N)
BETA = np.arcsin(RNG.uniform(-1, 1, N))


def lisabeta_tL_from_tSSB(tSSB, lambd, beta, frozen=False, tfrozen=None):
    """Verbatim port of lisabeta.lisa.lisatools.tLfromSSBframe (ini phase 0)."""
    alpha = lt.EARTH_ORBIT_OMEGA_SI * (tfrozen if frozen else tSSB)
    return tSSB - lt.R_OVER_C * np.cos(beta) * np.cos(alpha - lambd)


def lisabeta_tSSB_from_tL(tL, lambd, beta, frozen=False, tfrozen=None):
    """Verbatim port of lisabeta.lisa.lisatools.tSSBfromLframe (ini phase 0)."""
    alpha = lt.EARTH_ORBIT_OMEGA_SI * (tfrozen if frozen else tL)
    phase = alpha - lambd
    rc = lt.R_OVER_C
    corr = 0.0 if frozen else -0.5 * lt.EARTH_ORBIT_OMEGA_SI * (rc * np.cos(beta)) ** 2 * np.sin(2 * phase)
    return tL + rc * np.cos(beta) * np.cos(phase) + corr


def test_constants_consistent():
    assert lt.EARTH_ORBIT_OMEGA_SI == pytest.approx(2 * np.pi / lt.YRSID_SI, rel=1e-15)
    assert lt.R_OVER_C == pytest.approx(499.00478383615643, rel=1e-14)


@pytest.mark.parametrize("frozen", [False, True])
def test_deltatL_matches_lisabeta_formula(frozen):
    dt = RNG.uniform(-1800, 1800, N)
    ours = lt.deltatL_from_deltat(dt, LAMBD, BETA, T0_S, frozen_lisa=frozen)
    ref = lisabeta_tL_from_tSSB(T0_S + dt, LAMBD, BETA, frozen, T0_S) - T0_S
    # The reference subtracts two ~5e7 s numbers, so it is only good to ~1e-8 s.
    np.testing.assert_allclose(ours, ref, atol=5e-8, rtol=0)
    assert np.max(np.abs(ours - dt)) <= lt.R_OVER_C


@pytest.mark.parametrize("frozen", [False, True])
def test_round_trip_exact(frozen):
    dtL = RNG.uniform(-1800, 1800, N)
    dt = lt.deltat_from_deltatL(dtL, LAMBD, BETA, T0_S, frozen_lisa=frozen)
    back = lt.deltatL_from_deltat(dt, LAMBD, BETA, T0_S, frozen_lisa=frozen)
    np.testing.assert_allclose(back, dtL, atol=1e-9, rtol=0)


def test_inverse_close_to_lisabeta_explicit_formula():
    # lisabeta's explicit inverse is second order in Omega R/c; ours is exact.
    dtL = RNG.uniform(-1800, 1800, N)
    ours = lt.deltat_from_deltatL(dtL, LAMBD, BETA, T0_S)
    ref = lisabeta_tSSB_from_tL(T0_S + dtL, LAMBD, BETA) - T0_S
    assert np.max(np.abs(ours - ref)) < 1e-4


def test_scalar_inputs_return_floats():
    dtL = lt.deltatL_from_deltat(10.0, 0.3, -0.2, T0_S)
    dt = lt.deltat_from_deltatL(dtL, 0.3, -0.2, T0_S)
    assert isinstance(dtL, float) and isinstance(dt, float)
    assert dt == pytest.approx(10.0, abs=1e-9)


def test_time_shift_moves_tf_by_deltat():
    """lisabeta defines t(f) = (1/2 pi) d phase / df, with h = A exp(+i phase).
    The factor must increase t(f) by Deltat (signal arrives later)."""
    f = np.linspace(1e-4, 1e-2, 20001)
    phase = 2 * np.pi * f * 37.0 - 3e4 * np.sqrt(f)  # arbitrary smooth phase
    h = np.exp(1j * phase)
    for deltat in (-250.0, 0.0, 123.4):
        hs = h * lt.time_shift_factor(f, deltat)
        tf = np.gradient(np.unwrap(np.angle(hs)), f) / (2 * np.pi)
        tf0 = np.gradient(phase, f) / (2 * np.pi)
        np.testing.assert_allclose(tf - tf0, deltat, atol=1e-6)


def test_time_shift_factor_shapes():
    f = np.linspace(1e-4, 1e-2, 7)
    assert lt.time_shift_factor(f, 5.0).shape == (7,)
    fac = lt.time_shift_factor(f, np.array([0.0, 5.0, -5.0]))
    assert fac.shape == (3, 7)
    np.testing.assert_allclose(fac[0], 1.0)
    np.testing.assert_allclose(fac[1] * fac[2], 1.0)


def test_pop_time_shift_variants():
    lam, bet = 0.7, 0.2
    # Neither key: unchanged behaviour.
    ext = {"psi": 1.0}
    assert lt.pop_time_shift(ext, lam, bet, T0_S) == (None, None)
    assert ext == {"psi": 1.0}

    # DeltatL only.
    ext = {"DeltatL": 42.0}
    dt, dtL = lt.pop_time_shift(ext, lam, bet, T0_S)
    assert "DeltatL" not in ext and dtL == 42.0
    assert lt.deltatL_from_deltat(dt, lam, bet, T0_S) == pytest.approx(42.0, abs=1e-9)

    # Deltat only.
    ext = {"Deltat": -17.0}
    dt, dtL = lt.pop_time_shift(ext, lam, bet, T0_S)
    assert dt == -17.0
    assert dtL == pytest.approx(lt.deltatL_from_deltat(-17.0, lam, bet, T0_S))

    # Fiducial Deltat = 0 next to DeltatL (fixed intrinsic prior + sampled DeltatL).
    ext = {"Deltat": 0.0, "DeltatL": 5.0}
    dt, dtL = lt.pop_time_shift(ext, lam, bet, T0_S)
    assert dtL == 5.0 and ext == {}

    # Two nonzero offsets are ambiguous.
    with pytest.raises(ValueError):
        lt.pop_time_shift({"Deltat": 1.0, "DeltatL": 5.0}, lam, bet, T0_S)


def test_pop_time_shift_batched():
    dtL = RNG.uniform(-150, 150, N)
    ext = {"DeltatL": dtL}
    dt, dtL_out = lt.pop_time_shift(ext, LAMBD, BETA, T0_S)
    assert dt.shape == (N,)
    np.testing.assert_array_equal(dtL_out, dtL)
    # Fixing DeltatL across the sky spreads Deltat by up to R/c.
    assert np.ptp(dt - dtL) > 900.0


def test_strip_time_shift_for_generator():
    p = {"Deltat": 12.0, "DeltatL": 3.0, "Mchirp": 1e6}
    out = lt.strip_time_shift_for_generator(p)
    assert out == {"Deltat": 0.0, "Mchirp": 1e6}
    assert p["Deltat"] == 12.0  # input not modified
