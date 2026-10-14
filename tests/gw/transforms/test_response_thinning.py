"""LISA response thinning and chan3 skipping in process_transfer / ProjectOntoSpaceDetectors.

lisabeta's response and spline modules are replaced by smooth fakes that count calls, so
these tests run without lisabeta. If dingo cannot be imported (e.g. a broken lal/torch
stack), detector_transforms.py is loaded on its own with stubs for the missing packages.
"""
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.interpolate import CubicSpline as ScipySpline


def _load_detector_transforms():
    try:
        return importlib.import_module("dingo.gw.transforms.detector_transforms")
    except Exception:
        pass
    stubbed = [
        "torch", "pandas", "lal", "bilby", "bilby.gw", "bilby.gw.detector",
        "bilby.gw.detector.interferometer", "bilby.gw.prior", "lisabeta",
        "lisabeta.lisa", "lisabeta.lisa.pyresponse", "lisabeta.tools",
        "lisabeta.tools.pyspline", "lisabeta.pyconstants",
    ]
    saved = {name: sys.modules.get(name) for name in stubbed}
    for name in stubbed:
        try:
            importlib.import_module(name)
        except Exception:
            module = MagicMock(name=name)
            sys.modules[name] = module
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], child, module)
    path = (Path(__file__).resolve().parents[3]
            / "dingo" / "gw" / "transforms" / "detector_transforms.py")
    spec = importlib.util.spec_from_file_location("_detector_transforms_isolated", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


dt = _load_detector_transforms()

LISA_SETTINGS = {
    "LISAconst": "Proposal",
    "responseapprox": "full",
    "frozenLISA": False,
    "TDIrescaled": True,
}


# --------------------------------------------------------------------------------------
# Fake lisabeta modules
# --------------------------------------------------------------------------------------


class _Untouchable:
    """Stands in for transferL3 when chan3 must not be computed: any use raises."""

    __array_ufunc__ = None

    def _fail(self, *args, **kwargs):
        raise AssertionError("transferL3 was used although chan3 is not a detector")

    __mul__ = __rmul__ = __add__ = __radd__ = __getitem__ = __array__ = _fail
    real = imag = property(_fail)


class FakeLisabeta:
    def __init__(self, poison_chan3=False):
        self.poison_chan3 = poison_chan3
        self.response_grids = []  # number of points per response construction
        self.spline_grids = []  # number of points per spline build
        self.spline_evals = 0
        fake = self

        class LISAFDresponseTDI3Chan:
            def __init__(self, freq, tf, t0, l, m, inc, phi, lambd, beta, psi, *settings):
                self.freq, self.tf = np.asarray(freq), np.asarray(tf)
                fake.response_grids.append(len(self.freq))

            def get_response(self):
                f, t = self.freq / 6e-2, self.tf
                delay = 40.0 * np.sin(2e-7 * t) * f
                transfers = [
                    (1 + 0.2 * c * f) * np.exp(1j * (c + 3e-8 * t + 0.5 * f))
                    for c in (1, 2, 3)
                ]
                if fake.poison_chan3:
                    transfers[2] = _Untouchable()
                return (delay, *transfers)

        class CubicSpline:
            def __init__(self, x, y):
                self.x, self.y = np.asarray(x), np.asarray(y)
                fake.spline_grids.append(len(self.x))

            def get_spline(self):
                return ScipySpline(self.x, self.y, bc_type="natural")

        def spline_eval_vector(spline, xs, extrapol_zero=True):
            fake.spline_evals += 1
            out = spline(xs)
            if extrapol_zero:
                out[(xs < spline.x[0]) | (xs > spline.x[-1])] = 0.0
            return out

        self.pyresponse = types.SimpleNamespace(LISAFDresponseTDI3Chan=LISAFDresponseTDI3Chan)
        self.pyspline = types.SimpleNamespace(
            CubicSpline=CubicSpline, spline_eval_vector=spline_eval_vector
        )
        self.pyconstants = types.SimpleNamespace(YRSID_SI=31558149.763545603)


@pytest.fixture
def fake(monkeypatch):
    fake = FakeLisabeta()
    monkeypatch.setattr(dt, "pyresponse", fake.pyresponse)
    monkeypatch.setattr(dt, "pyspline", fake.pyspline)
    monkeypatch.setattr(dt, "pyconstants", fake.pyconstants)
    return fake


# --------------------------------------------------------------------------------------
# Reference algorithms (transcriptions, run on the same fakes)
# --------------------------------------------------------------------------------------


def unthinned_reference(fake, freq_grid, amp, phase, tf, t0, l, m, inc, phi, lambd, beta,
                        psi, interp_freqs, f_min, *settings):
    """process_transfer before response thinning (always computes all three channels)."""
    tdi = fake.pyresponse.LISAFDresponseTDI3Chan(
        freq_grid, tf, t0, l, m, inc, phi, lambd, beta, psi, *settings)
    delay, t1, t2, t3 = tdi.get_response()

    def ev(y):
        spline = fake.pyspline.CubicSpline(freq_grid, y).get_spline()
        return fake.pyspline.spline_eval_vector(spline, interp_freqs, extrapol_zero=True)

    eiphase = np.exp(1j * ev(phase + delay))
    out = []
    for t in (t1, t2, t3):
        camp = amp * t
        v = (ev(np.copy(np.real(camp))) + 1j * ev(np.copy(np.imag(camp)))) * eiphase
        v[interp_freqs < f_min] = 0.0
        out.append(v)
    return tuple(out)


def thinned_reference(fake, k, freq_grid, amp, phase, tf, t0, l, m, inc, phi, lambd, beta,
                      psi, interp_freqs, f_min, *settings):
    """make_process_transfer from notebooks/response_decimation_cluster.py (cluster job
    65674178), with chan3 computed."""

    def evaluate(x, y):
        spline = fake.pyspline.CubicSpline(
            x, np.ascontiguousarray(y, dtype=np.float64)).get_spline()
        return fake.pyspline.spline_eval_vector(spline, interp_freqs, extrapol_zero=True)

    freq_grid = np.ascontiguousarray(freq_grid, dtype=np.float64)
    sel = dt.response_thinning_indices(len(freq_grid), k)
    grid_k = np.ascontiguousarray(freq_grid[sel])
    tdi = fake.pyresponse.LISAFDresponseTDI3Chan(
        grid_k, np.ascontiguousarray(np.asarray(tf)[sel], dtype=np.float64), t0, l, m, inc,
        phi, lambd, beta, psi, *settings)
    delay, t1, t2, t3 = tdi.get_response()
    eiphase = np.exp(1j * (evaluate(freq_grid, phase) + evaluate(grid_k, delay)))
    out = []
    for t in (t1, t2, t3):
        camp = np.asarray(amp)[sel] * t
        v = (evaluate(grid_k, camp.real) + 1j * evaluate(grid_k, camp.imag)) * eiphase
        v[interp_freqs < f_min] = 0.0
        out.append(v)
    return tuple(out)


def mode_inputs(n=2000, f_lo=1e-4, f_hi=5e-2, n_interp=6000):
    freq = np.geomspace(f_lo, f_hi, n)
    amp = 1e-18 * (freq / 1e-4) ** (-7 / 6) * np.exp(-((freq / 5e-2) ** 4))
    phase = 2e4 * (freq / 1e-4) ** (-5 / 3)
    tf = -3e6 * (freq / 1e-4) ** (-8 / 3)
    interp = np.linspace(9e-5, 6e-2, n_interp)
    return [freq, amp, phase, tf, 0.1, 2, 2, 0.3, 0.0, 1.0, 0.2, 1.2, interp, 1.1e-4,
            "TDIAET", "Proposal", "full", False, True]


def whitened_mismatch(a, b):
    inner = np.sum(np.conj(a) * b).real
    return 1.0 - inner / np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))


# --------------------------------------------------------------------------------------
# process_transfer
# --------------------------------------------------------------------------------------


def test_thinning_indices():
    np.testing.assert_array_equal(dt.response_thinning_indices(10, 1), np.arange(10))
    np.testing.assert_array_equal(dt.response_thinning_indices(10, 4), [0, 4, 8, 9])
    np.testing.assert_array_equal(dt.response_thinning_indices(13, 4), [0, 4, 8, 12])
    # fewer than 4 points would remain, or the grid is tiny: full grid
    np.testing.assert_array_equal(dt.response_thinning_indices(9, 4), np.arange(9))
    np.testing.assert_array_equal(dt.response_thinning_indices(6, 4), np.arange(6))
    np.testing.assert_array_equal(dt.response_thinning_indices(4, 2), np.arange(4))
    for n in (7505, 7885, 100):
        idx = dt.response_thinning_indices(n, 8)
        assert idx[0] == 0 and idx[-1] == n - 1
        assert np.all(np.diff(idx) <= 8) and np.all(np.diff(idx)[:-1] == 8)


@pytest.mark.parametrize("compute_chan3", [True, False])
def test_k1_bit_identical_to_unthinned(fake, compute_chan3):
    args = mode_inputs()
    ref = unthinned_reference(fake, *args)
    new = dt.process_transfer(*args, compute_chan3=compute_chan3)
    np.testing.assert_array_equal(new[0], ref[0])
    np.testing.assert_array_equal(new[1], ref[1])
    if compute_chan3:
        np.testing.assert_array_equal(new[2], ref[2])
    else:
        assert new[2] is None


def test_defaults_are_unthinned_with_chan3(fake):
    args = mode_inputs()
    ref = unthinned_reference(fake, *args)
    new = dt.process_transfer(*args)
    assert len(new) == 3
    for r, n in zip(ref, new):
        np.testing.assert_array_equal(n, r)


@pytest.mark.parametrize("k", [2, 8, 16])
def test_thinned_matches_reference(fake, k):
    args = mode_inputs()
    ref = thinned_reference(fake, k, *args)
    new = dt.process_transfer(*args, response_thinning=k, compute_chan3=True)
    for r, n in zip(ref, new):
        np.testing.assert_array_equal(n, r)
    new2 = dt.process_transfer(*args, response_thinning=k, compute_chan3=False)
    np.testing.assert_array_equal(new2[0], ref[0])
    np.testing.assert_array_equal(new2[1], ref[1])
    assert new2[2] is None


def test_thinned_close_to_unthinned(fake):
    args = mode_inputs()
    full = dt.process_transfer(*args)
    thin = dt.process_transfer(*args, response_thinning=8)
    for ch in range(3):
        assert whitened_mismatch(full[ch], thin[ch]) < 1e-8
        # last point kept: same support, still zero below f_min and above the grid
        np.testing.assert_array_equal(thin[ch] != 0, full[ch] != 0)
    interp, f_min, freq = args[12], args[13], args[0]
    assert np.all(thin[0][(interp < f_min) | (interp > freq[-1])] == 0)


@pytest.mark.parametrize("k, chan3, n_response, n_full_splines, n_thin_splines", [
    (1, False, 2000, 5, 0),
    (1, True, 2000, 7, 0),
    (8, False, 251, 1, 5),
    (8, True, 251, 1, 7),
])
def test_spline_and_response_counts(fake, k, chan3, n_response, n_full_splines,
                                    n_thin_splines):
    args = mode_inputs(n=2000)
    dt.process_transfer(*args, response_thinning=k, compute_chan3=chan3)
    assert fake.response_grids == [n_response]
    assert fake.spline_grids.count(2000) == n_full_splines
    assert len(fake.spline_grids) == n_full_splines + n_thin_splines
    assert fake.spline_evals == n_full_splines + n_thin_splines


@pytest.mark.parametrize("k", [1, 8])
def test_chan3_never_touched_when_skipped(fake, k):
    fake.poison_chan3 = True
    out = dt.process_transfer(*mode_inputs(), response_thinning=k, compute_chan3=False)
    assert out[2] is None
    with pytest.raises(AssertionError, match="transferL3 was used"):
        dt.process_transfer(*mode_inputs(), response_thinning=k, compute_chan3=True)


def test_fallback_to_full_grid(fake):
    args = mode_inputs(n=9)  # k=4 would leave 3 points
    ref = unthinned_reference(fake, *args)
    fake.response_grids.clear()
    new = dt.process_transfer(*args, response_thinning=4)
    assert fake.response_grids == [9]
    for r, n in zip(ref, new):
        np.testing.assert_array_equal(n, r)


# --------------------------------------------------------------------------------------
# ProjectOntoSpaceDetectors
# --------------------------------------------------------------------------------------


MODES = {"(2, 2)": (1e-4, 5e-2, 1.0), "(3, 3)": (1.5e-4, 5.5e-2, 0.3)}


def _domain():
    return types.SimpleNamespace(sample_frequencies=np.linspace(9e-5, 6e-2, 6000), f_min=1.1e-4)


def _sample(batch=None):
    waveform = {}
    for lm, (f_lo, f_hi, scale) in MODES.items():
        freq, amp, phase, tf = mode_inputs(n=1500, f_lo=f_lo, f_hi=f_hi)[:4]
        mode = {"freq": freq, "amp": scale * amp, "phase": phase, "tf": tf}
        if batch:
            mode = {key: np.stack([v * (1 + 0.01 * i) if key == "amp" else v
                                   for i in range(batch)]) for key, v in mode.items()}
        waveform[lm] = mode
    pick = (lambda v: np.full(batch, v) + 0.01 * np.arange(batch)) if batch else (lambda v: v)
    return {
        "parameters": {"dist": pick(1e4), "inc": pick(0.3), "phi": pick(0.1)},
        "extrinsic_parameters": {"dist": pick(2e4), "beta": pick(0.2), "lambda": pick(1.0),
                                 "psi": pick(1.2), "geocent_time": pick(0.0)},
        "waveform": waveform,
    }


def _project(channels, **lisa_extra):
    return dt.ProjectOntoSpaceDetectors(
        "TDIAET", _domain(), 0.0, channels, {**LISA_SETTINGS, **lisa_extra})


def test_settings_default_and_validation():
    assert _project(["chan1", "chan2"]).response_thinning == 1
    assert _project(["chan1", "chan2"], response_thinning=8).response_thinning == 8
    assert _project(["chan1", "chan2"], response_thinning=np.int64(4)).response_thinning == 4
    for bad in (0, -1, 2.5, 8.0, "8", True, None):
        with pytest.raises(ValueError, match="response_thinning"):
            _project(["chan1", "chan2"], response_thinning=bad)


@pytest.mark.parametrize("k", [1, 8])
def test_projection_sums_modes_and_skips_chan3(fake, k):
    fake.poison_chan3 = True
    out = _project(["chan1", "chan2"], response_thinning=k)(_sample())
    assert set(out["waveform"]) == {"chan1", "chan2"}
    fake.poison_chan3 = False
    expected = [0, 0]
    for lm, mode in _sample()["waveform"].items():
        l, m = eval(lm)
        args = [mode["freq"], 0.5 * mode["amp"], mode["phase"], mode["tf"], 0.0, l, m, 0.3,
                0.1, 1.0, 0.2, 1.2, _domain().sample_frequencies, _domain().f_min,
                "TDIAET", *LISA_SETTINGS.values()]
        ref = (unthinned_reference(fake, *args) if k == 1
               else thinned_reference(fake, k, *args))
        expected = [expected[0] + ref[0], expected[1] + ref[1]]
    np.testing.assert_allclose(out["waveform"]["chan1"], expected[0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(out["waveform"]["chan2"], expected[1], rtol=1e-12, atol=0)


def test_projection_chan3_detector(fake):
    with_chan3 = _project(["chan1", "chan2", "chan3"], response_thinning=8)(_sample())
    without = _project(["chan1", "chan2"], response_thinning=8)(_sample())
    assert set(with_chan3["waveform"]) == {"chan1", "chan2", "chan3"}
    assert np.abs(with_chan3["waveform"]["chan3"]).max() > 0
    for ch in ("chan1", "chan2"):
        np.testing.assert_array_equal(with_chan3["waveform"][ch], without["waveform"][ch])


@pytest.mark.parametrize("channels", [["chan1", "chan2"], ["chan1", "chan2", "chan3"]])
def test_projection_batched_matches_single(fake, channels):
    batch = 3
    fake.poison_chan3 = "chan3" not in channels
    project = _project(channels, response_thinning=8)
    batched = project(_sample(batch))
    assert set(batched["waveform"]) == set(channels)
    single_inputs = _sample(batch)
    for i in range(batch):
        sample = {
            "parameters": {k: v[i] for k, v in single_inputs["parameters"].items()},
            "extrinsic_parameters": {k: v[i] for k, v in
                                     single_inputs["extrinsic_parameters"].items()},
            "waveform": {lm: {k: v[i] for k, v in mode.items()}
                         for lm, mode in single_inputs["waveform"].items()},
        }
        single = project(sample)
        for ch in channels:
            np.testing.assert_allclose(batched["waveform"][ch][i], single["waveform"][ch],
                                       rtol=1e-12, atol=0)
