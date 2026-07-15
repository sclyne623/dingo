"""
Shared fixtures for gw tests.

Provides a small, self-contained likelihood object (synthetic strain data, flat
ASDs, IMRPhenomXPHM on a reduced frequency range) and parameter samples for the
phase-grid tests.
"""

import numpy as np
import pandas as pd
import pytest

from dingo.gw.domains import build_domain
from dingo.gw.likelihood import StationaryGaussianGWLikelihood


@pytest.fixture(scope="session")
def likelihood_object():
    domain = build_domain(
        {
            "type": "FrequencyDomain",
            "f_min": 20.0,
            "f_max": 256.0,
            "delta_f": 0.25,
            "window_factor": 1.0,
        }
    )
    n = domain.max_idx + 1
    rng = np.random.default_rng(42)
    strain = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-23
    asd = np.full(n, 1e-23)
    event_data = {
        "waveform": {"H1": strain, "L1": strain.copy()},
        "asds": {"H1": asd, "L1": asd.copy()},
    }
    return StationaryGaussianGWLikelihood(
        wfg_kwargs={
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0,
        },
        wfg_domain=domain,
        data_domain=domain,
        event_data=event_data,
        t_ref=1126259462.4,
    )


@pytest.fixture(scope="session")
def test_samples():
    n = 12
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "mass_1": 35 + rng.uniform(-2, 2, n),
            "mass_2": 30 + rng.uniform(-2, 2, n),
            "luminosity_distance": 440 + rng.uniform(-50, 50, n),
            "a_1": rng.uniform(0.1, 0.5, n),
            "a_2": rng.uniform(0.1, 0.5, n),
            "tilt_1": rng.uniform(0.2, 1.0, n),
            "tilt_2": rng.uniform(0.2, 1.0, n),
            "phi_12": rng.uniform(0, 6, n),
            "phi_jl": rng.uniform(0, 6, n),
            "theta_jn": rng.uniform(0.1, 1.0, n),
            "geocent_time": rng.uniform(-0.01, 0.01, n),
            "ra": rng.uniform(0, 6, n),
            "dec": rng.uniform(-1.0, 1.0, n),
            "psi": rng.uniform(0, 3, n),
            "phase": np.zeros(n),
        }
    )
