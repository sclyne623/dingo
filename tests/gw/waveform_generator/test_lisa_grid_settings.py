"""LISA grid configuration tests; mock waveform/grid evaluation, not the generator."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator import waveform_generator as wfg


@pytest.fixture
def domain():
    return UniformFrequencyDomain(f_min=1e-4, f_max=0.06, delta_f=1e-4)


@pytest.fixture
def grid_backend(monkeypatch):
    frequencies = np.array([1e-4, 1e-3, 0.06])
    grid = Mock(return_value=SimpleNamespace(get_freq=lambda: frequencies.copy()))
    bounds = Mock(return_value=(1e-4, 0.06))
    monkeypatch.setattr(wfg.pytools, "FrequencyGrid", grid)
    monkeypatch.setattr(wfg.wfg_utils, "FrequencyBoundsLISATDI_SMBH", bounds)
    return grid, bounds, frequencies


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, {"acc": 1e-4, "DeltalnMf_max": 0.025}),
        ({"acc": 1e-6}, {"acc": 1e-6, "DeltalnMf_max": 0.025}),
        ({"DeltalnMf_max": 0.0015625}, {"acc": 1e-4, "DeltalnMf_max": 0.0015625}),
        (
            {"acc": 1e-6, "DeltalnMf_max": 0.003125},
            {"acc": 1e-6, "DeltalnMf_max": 0.003125},
        ),
    ],
)
def test_grid_settings_reach_lisabeta(domain, grid_backend, overrides, expected):
    grid, bounds, frequencies = grid_backend
    generator = wfg.LISAWaveformGenerator("IMRPhenomXHM", domain, 0.0, **overrides)
    params = {"M": 1e6, "q": 0.8}
    result = generator.Generate_coarse_freq_grid(params, t0=1.5)
    grid.assert_called_once_with(1e-4, 0.06, 1e6, 0.8, **expected)
    np.testing.assert_array_equal(result, frequencies)
    assert bounds.call_args.kwargs["t0"] == 1.5
    assert bounds.call_args.kwargs["minf"] == domain.f_min
    assert bounds.call_args.kwargs["maxf"] == domain.f_max
    assert generator.domain is domain


def test_dataset_yaml_drives_waveform_grid(domain, grid_backend, monkeypatch):
    """Use the same settings expansion as dataset and on-the-fly constructors."""
    settings = yaml.safe_load(
        """
waveform_generator:
  approximant: IMRPhenomXHM
  f_ref: 0.0
  LISA: true
  on_fly: true
  acc: 1.0e-6
  DeltalnMf_max: 0.0015625
"""
    )
    generator = wfg.LISAWaveformGenerator(
        domain=domain, **settings["waveform_generator"]
    )
    parameters = dict(
        M=1e6,
        q=0.8,
        m1=6e5,
        m2=4e5,
        chi1=0.2,
        chi2=0.3,
        dist=100.0,
        Deltat=0.0,
        geocent_time=0.0,
    )
    monkeypatch.setattr(generator, "convert_parameters", lambda p, *args, **kwargs: p)
    modes = {(3, 3): {"test": "native output"}}
    wrapper = Mock(return_value=SimpleNamespace(get_waveform=lambda: modes))
    monkeypatch.setattr(wfg.pyIMRPhenomXHM, "IMRPhenomXHMhlmAmpPhase", wrapper)
    assert generator.generate_amp_phase(parameters) is modes
    grid, _, frequencies = grid_backend
    assert grid.call_args.kwargs == {"acc": 1e-6, "DeltalnMf_max": 0.0015625}
    np.testing.assert_array_equal(wrapper.call_args.args[0], frequencies)
    assert wrapper.call_args.kwargs["scale_freq_hm"] is True
    assert generator.domain is domain


@pytest.mark.parametrize("setting", ["acc", "DeltalnMf_max"])
@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        "1e-4",
        None,
        [0.01],
        0.01 + 0j,
    ],
)
def test_invalid_grid_settings_fail_early(domain, grid_backend, setting, value):
    grid, bounds, _ = grid_backend
    with pytest.raises(
        ValueError, match=setting + " must be a finite positive real number"
    ):
        wfg.LISAWaveformGenerator("IMRPhenomXHM", domain, 0.0, **{setting: value})
    grid.assert_not_called()
    bounds.assert_not_called()


def test_mode_specific_bounds_use_settings_and_returned_modes(
    domain, grid_backend, monkeypatch
):
    grid, bounds, frequencies = grid_backend
    low = {(2, 2): 1e-4, (3, 3): 1.5e-4}
    high = {(2, 2): 0.04, (3, 3): 0.06}
    bounds.return_value = low, high
    scaling = Mock(side_effect=lambda f, lo, hi: np.array([lo, hi]))
    monkeypatch.setattr(wfg.pytools, "log_affine_scaling", scaling)
    generator = wfg.LISAWaveformGenerator(
        "IMRPhenomXHM", domain, 0.0, acc=1e-6, DeltalnMf_max=0.003125
    )
    result = generator.Generate_coarse_freq_grid({"M": 1e6, "q": 0.8})
    grid.assert_called_once_with(1e-4, 0.04, 1e6, 0.8, acc=1e-6, DeltalnMf_max=0.003125)
    assert set(result) == set(low)
    for lm in low:
        np.testing.assert_array_equal(result[lm], [low[lm], high[lm]])
    assert scaling.call_count == len(low)
