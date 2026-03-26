#!/usr/bin/env python3
"""
Generate a fiducial LISA ASD dataset (HDF5) for the inspiral-only MBHB pipeline.

The ASD is computed from the ESA Proposal LISA noise budget evaluated on the
base UniformFrequencyDomain grid, then saved in the ASDDataset format expected
by Dingo's training pipeline with ifos=['chan1', 'chan2', 'chan3'].

Usage
-----
python misc_scripts/generate_lisa_asd.py [--output PATH] [--domain-settings PATH]

Defaults to writing  training_data/asds_lisa_fiducial.hdf5  in the current
working directory.  The domain settings must match the base_domain used in
waveform_dataset_settings.yaml.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dingo.gw.domains import build_domain
from dingo.gw.noise.asd_dataset import ASDDataset

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_SI        = 2.998_924_58e8   # m/s
LISA_ARM_M  = 2.5e9            # m  (ESA Proposal arm length)
F_STAR      = C_SI / (2.0 * math.pi * LISA_ARM_M)  # ≈ 19.1 mHz


# ---------------------------------------------------------------------------
# Noise budget (ESA LISA Proposal, Amaro-Seoane et al. 2017)
# ---------------------------------------------------------------------------

def _single_link_psd(f: np.ndarray, L: float = LISA_ARM_M) -> np.ndarray:
    """Single-link fractional-frequency noise PSD S_link(f).

    Combines optical-metrology-system (OMS) and test-mass (TM) acceleration noise.
    """
    f = np.asarray(f, dtype=float)
    # OMS position noise: S_x = (1.5e-11 m)^2/Hz * (1 + (2 mHz/f)^4)
    S_oms = (1.5e-11) ** 2 * (1.0 + (2e-3 / np.maximum(f, 1e-12)) ** 4)  # m^2/Hz
    # TM acceleration noise: S_a = (3e-15 m/s^2)^2/Hz * (1 + (0.4 mHz/f)^2) * ...
    S_acc = (3e-15) ** 2 * (
        1.0 + (4e-4 / np.maximum(f, 1e-12)) ** 2
    ) * (
        1.0 + (f / 8e-3) ** 4
    )
    # Convert acceleration → displacement: divide by (2πf)^4
    S_acc /= (2.0 * math.pi * np.maximum(f, 1e-12)) ** 4  # m^2/Hz
    # Fractional frequency: divide by L^2
    return (S_oms + 2.0 * S_acc) / L ** 2  # Hz^{-1}


def lisa_psd_A(f: np.ndarray, L: float = LISA_ARM_M) -> np.ndarray:
    """TDI-A channel one-sided PSD (Hz^{-1}).

    Uses the 3-arm equal-armlength approximation.
    """
    f      = np.asarray(f, dtype=float)
    S_link = _single_link_psd(f, L)
    x      = 2.0 * math.pi * f * L / C_SI
    # TDI-A: sum of two Michelson channels with 60° opening angle
    return 8.0 * np.sin(x) ** 2 * (2.0 * (1.0 + np.cos(x) ** 2) * S_link)


def lisa_psd_E(f: np.ndarray, L: float = LISA_ARM_M) -> np.ndarray:
    """TDI-E channel one-sided PSD (Hz^{-1}).

    For equal armlengths the E channel PSD is identical to TDI-A.
    """
    return lisa_psd_A(f, L)


def lisa_psd_T(f: np.ndarray, L: float = LISA_ARM_M) -> np.ndarray:
    """TDI-T (breathing) channel one-sided PSD (Hz^{-1}).

    The T channel is the null channel at low frequencies; it has suppressed
    sensitivity to GWs but is included for completeness.
    """
    f      = np.asarray(f, dtype=float)
    S_link = _single_link_psd(f, L)
    x      = 2.0 * math.pi * f * L / C_SI
    return 16.0 * (1.0 - np.cos(x)) ** 2 * np.sin(x) ** 2 * S_link


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_DOMAIN = dict(
    type="UniformFrequencyDomain",
    f_min=1e-4,
    f_max=1e-1,
    delta_f=5e-6,
)


def generate_lisa_asd_dataset(
    output_path: str | Path,
    domain_settings: dict | None = None,
) -> ASDDataset:
    """Compute fiducial LISA ASDs and save as an ASDDataset HDF5 file.

    Parameters
    ----------
    output_path : str or Path
        Destination HDF5 file path.
    domain_settings : dict, optional
        Dingo domain dictionary.  Defaults to the base domain used by
        examples/lisa_inspiral_npe_model/waveform_dataset_settings.yaml.

    Returns
    -------
    ASDDataset
    """
    if domain_settings is None:
        domain_settings = DEFAULT_DOMAIN

    domain = build_domain(domain_settings)
    freqs  = np.array(domain.sample_frequencies, dtype=np.float64)
    n_bins = len(freqs)

    print(f"Domain: {n_bins} bins, {freqs[0]:.2e}–{freqs[-1]:.2e} Hz")
    print(f"LISA characteristic frequency f* ≈ {F_STAR * 1e3:.1f} mHz")

    # Compute ASD = sqrt(PSD) for each TDI channel.
    # Shape: (1, n_bins) — single fiducial realisation.
    psd_A = lisa_psd_A(freqs)
    psd_E = lisa_psd_E(freqs)
    psd_T = lisa_psd_T(freqs)

    # Guard against zero/negative PSD at f=0 (not a physical frequency bin)
    for psd in (psd_A, psd_E, psd_T):
        psd[freqs == 0] = np.inf

    asd_A = np.sqrt(psd_A)[None, :]  # (1, n_bins)
    asd_E = np.sqrt(psd_E)[None, :]
    asd_T = np.sqrt(psd_T)[None, :]

    # gps_times is a placeholder (no physical GPS time for a theoretical curve).
    gps_placeholder = np.array([0.0])

    dataset_dict = {
        "settings": {
            "domain_dict": domain.domain_dict,
            "description": (
                "Fiducial LISA TDI-A/E/T ASD dataset computed from the ESA Proposal "
                "noise budget (Amaro-Seoane et al. 2017).  Single realisation — suitable "
                "for stage_0 (fixed-noise) training of the MBHB inspiral-only network."
            ),
        },
        "asds": {
            "chan1": asd_A.astype(np.float64),
            "chan2": asd_E.astype(np.float64),
            "chan3": asd_T.astype(np.float64),
        },
        "gps_times": {
            "chan1": gps_placeholder,
            "chan2": gps_placeholder,
            "chan3": gps_placeholder,
        },
        "asd_parameterizations": {},
    }

    asd_dataset = ASDDataset(dictionary=dataset_dict)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asd_dataset.to_file(str(output_path))
    print(f"Saved ASD dataset → {output_path}")

    return asd_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", "-o",
        default="training_data/asds_lisa_fiducial.hdf5",
        help="Output HDF5 file path (default: training_data/asds_lisa_fiducial.hdf5)",
    )
    p.add_argument(
        "--f-min", type=float, default=DEFAULT_DOMAIN["f_min"],
        help="Minimum frequency [Hz]",
    )
    p.add_argument(
        "--f-max", type=float, default=DEFAULT_DOMAIN["f_max"],
        help="Maximum frequency [Hz]",
    )
    p.add_argument(
        "--delta-f", type=float, default=DEFAULT_DOMAIN["delta_f"],
        help="Frequency resolution [Hz]",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    domain_settings = dict(
        type="UniformFrequencyDomain",
        f_min=args.f_min,
        f_max=args.f_max,
        delta_f=args.delta_f,
    )
    generate_lisa_asd_dataset(args.output, domain_settings)
