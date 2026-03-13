#!/usr/bin/env python3
"""
Compare LISA waveforms generated with LISABeta and BBHx.

This script benchmarks:
1. LISABeta intrinsic mode generation
2. BBHx intrinsic mode generation
3. LISABeta response projection on LISABeta modes
4. LISABeta response projection on BBHx modes
5. Optional BBHx direct A/E/T generation

It reports detector-frame agreement on a common Dingo frequency grid so you can
check that the implementations are numerically identical while also timing the
waveform and response stages separately.

This is intended to run in the URI cluster environment after bootstrapping the
modules/conda/PYTHONPATH stack, for example via
``misc_scripts/lisa_cluster_bootstrap.sh``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_local_bbhx() -> None:
    """
    Allow Dingo's lowercase ``bbhx`` imports to resolve against a local ``BBhx``
    checkout in the repository root when the package is not installed.
    """
    try:
        import bbhx  # noqa: F401

        return
    except ImportError:
        pass

    local_pkg = REPO_ROOT / "BBhx"
    init_file = local_pkg / "__init__.py"
    if not init_file.exists():
        return

    spec = importlib.util.spec_from_file_location(
        "bbhx",
        init_file,
        submodule_search_locations=[str(local_pkg)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules["bbhx"] = module
    spec.loader.exec_module(module)


_bootstrap_local_bbhx()

IMPORT_ERROR = None
PYCONSTANTS_YRSID_SI = 31558149.763545603

try:
    import lisabeta.pyconstants as pyconstants
    from bbhx.utils.transform import tSSBfromLframe as bbhx_tSSBfromLframe
    from dingo.gw.domains import UniformFrequencyDomain
    from dingo.gw.transforms.detector_transforms import (
        GenerateBBHxDirectResponse,
        ProjectOntoSpaceDetectors,
    )
    from dingo.gw.waveform_generator.waveform_generator import (
        BBHxWaveformGenerator,
        LISAWaveformGenerator,
    )

    PYCONSTANTS_YRSID_SI = pyconstants.YRSID_SI
except Exception as exc:  # pragma: no cover - exercised only in missing-env cases
    IMPORT_ERROR = exc
    pyconstants = None
    bbhx_tSSBfromLframe = None
    UniformFrequencyDomain = None
    ProjectOntoSpaceDetectors = None
    GenerateBBHxDirectResponse = None
    BBHxWaveformGenerator = None
    LISAWaveformGenerator = None


DEFAULT_MODE_LIST = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
DEFAULT_LISA_SETTINGS = {
    "detector_type": "TDIAET",
    "LISAconst": "Proposal",
    "responseapprox": "full",
    "frozenLISA": False,
    "TDIrescaled": False,
}
DEFAULT_PARAMS = {
    "Mchirp": 7.0e5,
    "q": 0.9,
    "chi1": 0.3,
    "chi2": 0.3,
    "phi": 0.0,
    "inc": math.pi / 6.0,
    "geocent_time": 0.5 * PYCONSTANTS_YRSID_SI,
    "Deltat": 0.0,
    "dist": 4.8e5,
    "lambda": 0.05,
    "beta": 0.1,
    "psi": 1.239264,
}


def environment_report() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cwd": os.getcwd(),
        "import_error": None if IMPORT_ERROR is None else repr(IMPORT_ERROR),
    }


def _require_waveform_stack() -> None:
    if IMPORT_ERROR is None:
        return
    report = environment_report()
    raise RuntimeError(
        "Unable to import the Dingo/LISABeta/BBHx waveform stack.\n"
        "Active Python executable: {python_executable}\n"
        "Python version: {python_version}\n"
        "Conda env: {conda_env}\n"
        "Import error: {import_error}\n"
        "On the cluster, this usually means the notebook kernel is not using the "
        "`lisa_310_build` environment from `misc_scripts/lisa_cluster_bootstrap.sh`."
        .format(**report)
    ) from IMPORT_ERROR


@dataclass
class ComparisonConfig:
    f_min: float = 1.0e-5
    f_max: float = 5.0e-1
    delta_f: float = 1.0e-5
    f_ref: float = 0.0
    approximant: str = "IMRPhenomHM"
    bbhx_approximant: str = "PhenomHM"
    modes: list[tuple[int, int]] = field(
        default_factory=lambda: list(DEFAULT_MODE_LIST)
    )
    bbhx_length: int = 1024
    warmup: int = 1
    repeats: int = 5
    channels: str = "chan1,chan2,chan3"
    use_gpu: bool = False
    include_bbhx_direct: bool = False
    include_alignment_diagnostics: bool = True
    include_bbhx_selfcheck: bool = True
    json: bool = False
    rtol: float = 1.0e-3
    atol: float = 1.0e-20
    Mchirp: float = DEFAULT_PARAMS["Mchirp"]
    q: float = DEFAULT_PARAMS["q"]
    chi1: float = DEFAULT_PARAMS["chi1"]
    chi2: float = DEFAULT_PARAMS["chi2"]
    phi: float = DEFAULT_PARAMS["phi"]
    inc: float = DEFAULT_PARAMS["inc"]
    geocent_time: float = DEFAULT_PARAMS["geocent_time"]
    Deltat: float = DEFAULT_PARAMS["Deltat"]
    dist: float = DEFAULT_PARAMS["dist"]
    lambda_: float = DEFAULT_PARAMS["lambda"]
    beta: float = DEFAULT_PARAMS["beta"]
    psi: float = DEFAULT_PARAMS["psi"]

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            f_min=self.f_min,
            f_max=self.f_max,
            delta_f=self.delta_f,
            f_ref=self.f_ref,
            approximant=self.approximant,
            bbhx_approximant=self.bbhx_approximant,
            modes=list(self.modes),
            bbhx_length=self.bbhx_length,
            warmup=self.warmup,
            repeats=self.repeats,
            channels=self.channels,
            use_gpu=self.use_gpu,
            include_bbhx_direct=self.include_bbhx_direct,
            include_alignment_diagnostics=self.include_alignment_diagnostics,
            include_bbhx_selfcheck=self.include_bbhx_selfcheck,
            json=self.json,
            rtol=self.rtol,
            atol=self.atol,
            Mchirp=self.Mchirp,
            q=self.q,
            chi1=self.chi1,
            chi2=self.chi2,
            phi=self.phi,
            inc=self.inc,
            geocent_time=self.geocent_time,
            Deltat=self.Deltat,
            dist=self.dist,
            beta=self.beta,
            psi=self.psi,
            **{"lambda": self.lambda_},
        )


def _parse_mode_list(text: str) -> list[tuple[int, int]]:
    modes: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if len(item) != 2 or not item.isdigit():
            raise argparse.ArgumentTypeError(
                f"Invalid mode '{item}'. Use a comma-separated list like 22,21,33."
            )
        modes.append((int(item[0]), int(item[1])))
    if not modes:
        raise argparse.ArgumentTypeError("At least one mode is required.")
    return modes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-min", type=float, default=1.0e-5)
    parser.add_argument("--f-max", type=float, default=5.0e-1)
    parser.add_argument("--delta-f", type=float, default=1.0e-5)
    parser.add_argument("--f-ref", type=float, default=0.0)
    parser.add_argument("--approximant", type=str, default="IMRPhenomHM")
    parser.add_argument("--bbhx-approximant", type=str, default="PhenomHM")
    parser.add_argument(
        "--modes",
        type=_parse_mode_list,
        default=DEFAULT_MODE_LIST,
        help="Comma-separated mode list, e.g. 22,21,33,32,44,43",
    )
    parser.add_argument("--bbhx-length", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--channels", type=str, default="chan1,chan2,chan3")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--include-bbhx-direct", action="store_true")
    parser.add_argument("--skip-alignment-diagnostics", action="store_true")
    parser.add_argument("--skip-bbhx-selfcheck", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--atol", type=float, default=1.0e-20)

    for key, value in DEFAULT_PARAMS.items():
        parser.add_argument(
            f"--{key}",
            dest=key,
            type=float,
            default=value,
        )
    return parser


def _canonicalize_channels(text: str) -> list[str]:
    channels = [item.strip() for item in text.split(",") if item.strip()]
    valid = {"chan1", "chan2", "chan3"}
    invalid = [item for item in channels if item not in valid]
    if invalid:
        raise ValueError(f"Invalid channels: {invalid}")
    if not channels:
        raise ValueError("At least one channel is required.")
    return channels


def _build_domain(args: argparse.Namespace) -> UniformFrequencyDomain:
    _require_waveform_stack()
    return UniformFrequencyDomain(
        f_min=args.f_min,
        f_max=args.f_max,
        delta_f=args.delta_f,
    )


def _build_generators(
    domain: UniformFrequencyDomain,
    args: argparse.Namespace,
) -> tuple[LISAWaveformGenerator, BBHxWaveformGenerator]:
    _require_waveform_stack()
    lisa_gen = LISAWaveformGenerator(
        approximant=args.approximant,
        domain=domain,
        f_ref=args.f_ref,
        mode_list=args.modes,
        frozenLISA=DEFAULT_LISA_SETTINGS["frozenLISA"],
    )
    bbhx_gen = BBHxWaveformGenerator(
        approximant=args.bbhx_approximant,
        domain=domain,
        f_ref=args.f_ref,
        mode_list=args.modes,
        frozenLISA=DEFAULT_LISA_SETTINGS["frozenLISA"],
        use_gpu=bool(args.use_gpu),
        bbhx_length=args.bbhx_length,
    )
    return lisa_gen, bbhx_gen


def _build_projector(
    domain: UniformFrequencyDomain,
    channels: list[str],
) -> ProjectOntoSpaceDetectors:
    _require_waveform_stack()
    return ProjectOntoSpaceDetectors(
        detector_type=DEFAULT_LISA_SETTINGS["detector_type"],
        domain=domain,
        ref_time=0.0,
        channels=channels,
        lisa_settings=DEFAULT_LISA_SETTINGS,
    )


def _make_projection_sample(waveform: dict[str, Any], params: dict[str, float]) -> dict[str, Any]:
    return {
        "waveform": copy.deepcopy(waveform),
        "parameters": {
            "dist": params["dist"],
            "inc": params["inc"],
            "phi": params["phi"],
            "lambda": params["lambda"],
            "beta": params["beta"],
            "psi": params["psi"],
            "geocent_time": params["geocent_time"],
        },
        "extrinsic_parameters": {
            "dist": params["dist"],
            "inc": params["inc"],
            "phi": params["phi"],
            "lambda": params["lambda"],
            "beta": params["beta"],
            "psi": params["psi"],
            "geocent_time": params["geocent_time"],
        },
    }


def _make_bbhx_training_sample(params: dict[str, float]) -> dict[str, Any]:
    return {
        "parameters": {
            "Mchirp": params["Mchirp"],
            "q": params["q"],
            "chi1": params["chi1"],
            "chi2": params["chi2"],
        },
        "extrinsic_parameters": {
            "phi": params["phi"],
            "inc": params["inc"],
            "geocent_time": params["geocent_time"],
            "dist": params["dist"],
            "lambda": params["lambda"],
            "beta": params["beta"],
            "psi": params["psi"],
        },
    }


def _benchmark(
    name: str,
    func,
    warmup: int,
    repeats: int,
) -> tuple[Any, dict[str, float]]:
    for _ in range(warmup):
        func()

    output = None
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        output = func()
        durations.append(time.perf_counter() - t0)

    if output is None:
        raise RuntimeError(f"{name} did not produce an output.")

    metrics = {
        "count": float(len(durations)),
        "min_s": float(min(durations)),
        "max_s": float(max(durations)),
        "mean_s": float(statistics.mean(durations)),
        "median_s": float(statistics.median(durations)),
    }
    if len(durations) > 1:
        metrics["stdev_s"] = float(statistics.stdev(durations))
    else:
        metrics["stdev_s"] = 0.0
    return output, metrics


def _summarize_durations(durations: list[float]) -> dict[str, float]:
    if not durations:
        raise RuntimeError("No durations to summarize.")
    metrics = {
        "count": float(len(durations)),
        "min_s": float(min(durations)),
        "max_s": float(max(durations)),
        "mean_s": float(statistics.mean(durations)),
        "median_s": float(statistics.median(durations)),
    }
    if len(durations) > 1:
        metrics["stdev_s"] = float(statistics.stdev(durations))
    else:
        metrics["stdev_s"] = 0.0
    return metrics


def _benchmark_bbhx_native_components(
    bbhx_gen: BBHxWaveformGenerator,
    params: dict[str, float],
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, float]]:
    wave_builder = bbhx_gen.waveform_gen

    def _run_once() -> dict[str, float]:
        parsed = bbhx_gen._parse_parameters(params.copy())
        m1 = np.atleast_1d(parsed["m1"])
        m2 = np.atleast_1d(parsed["m2"])
        chi1z = np.atleast_1d(parsed["chi1z"])
        chi2z = np.atleast_1d(parsed["chi2z"])
        distance_si = np.atleast_1d(parsed["distance_mpc"] * 1e6 * 3.085677581491367e16)
        inc = np.atleast_1d(parsed["inc"])
        phase = np.atleast_1d(parsed["phase"])
        lam = np.atleast_1d(parsed["lam"])
        beta = np.atleast_1d(parsed["beta"])
        psi = np.atleast_1d(parsed["psi"])
        t_ref = np.atleast_1d(parsed["t_ref"])

        freqs = bbhx_gen._get_cached_backend_frequency_grid()
        length = int(bbhx_gen.bbhx_length)
        num_modes = len(bbhx_gen.mode_list)
        num_bin_all = len(m1)
        out_buffer = wave_builder.xp.zeros(
            wave_builder.num_interp_params * length * num_modes * num_bin_all
        )

        t_start = np.atleast_1d(bbhx_tSSBfromLframe(0.0, lam, beta, 0.0))
        t_end = np.atleast_1d(
            bbhx_tSSBfromLframe(float(PYCONSTANTS_YRSID_SI), lam, beta, 0.0)
        )
        t_obs = t_end - t_start
        phi_ref_amp_phase = np.zeros_like(m1)

        t0 = time.perf_counter()
        wave_builder.amp_phase_gen(
            m1,
            m2,
            chi1z,
            chi2z,
            distance_si,
            phi_ref_amp_phase,
            bbhx_gen.f_ref,
            t_ref,
            length,
            freqs=None,
            out_buffer=out_buffer,
            modes=bbhx_gen.mode_list,
            Tobs=t_obs,
            direct=False,
        )
        amp_dt = time.perf_counter() - t0

        t1 = time.perf_counter()
        wave_builder.response_gen(
            wave_builder.amp_phase_gen.freqs,
            inc,
            lam,
            beta,
            psi,
            phase,
            length,
            out_buffer=out_buffer,
            modes=wave_builder.amp_phase_gen.modes,
            direct=False,
        )
        response_dt = time.perf_counter() - t1

        return {
            "amp_phase": amp_dt,
            "response": response_dt,
        }

    for _ in range(warmup):
        _run_once()

    amp_durations = []
    response_durations = []
    for _ in range(repeats):
        result = _run_once()
        amp_durations.append(result["amp_phase"])
        response_durations.append(result["response"])

    return {
        "bbhx_native_amp_phase": _summarize_durations(amp_durations),
        "bbhx_native_response": _summarize_durations(response_durations),
    }


def _complex_overlap(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 1.0
    return float(np.abs(np.vdot(a, b)) / denom)


def _best_fit_complex_scale(reference: np.ndarray, candidate: np.ndarray) -> complex:
    denom = np.vdot(candidate, candidate)
    if np.abs(denom) == 0.0:
        return 0.0 + 0.0j
    return np.vdot(candidate, reference) / denom


def _interp_complex(
    source_freqs: np.ndarray,
    source_values: np.ndarray,
    target_freqs: np.ndarray,
) -> np.ndarray:
    real = np.interp(target_freqs, source_freqs, np.real(source_values))
    imag = np.interp(target_freqs, source_freqs, np.imag(source_values))
    return real + 1j * imag


def _mode_complex_series(
    mode_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(mode_payload["freq"], dtype=np.float64).ravel()
    amp = np.asarray(mode_payload["amp"])
    phase = np.asarray(mode_payload["phase"])
    amp = np.asarray(amp).reshape(-1)
    phase = np.asarray(phase).reshape(-1)
    return freqs, amp * np.exp(1j * phase)


def _as_frequency_series(values: Any, min_idx: int = 0) -> np.ndarray:
    arr = np.asarray(values)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr[min_idx:]
    raise ValueError(
        f"Expected a single frequency series after squeezing, got shape {arr.shape}."
    )


def _channel_metrics(
    lhs: dict[str, np.ndarray],
    rhs: dict[str, np.ndarray],
    min_idx: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for channel in lhs.keys() & rhs.keys():
        x = _as_frequency_series(lhs[channel], min_idx=min_idx)
        y = _as_frequency_series(rhs[channel], min_idx=min_idx)
        diff = x - y
        y_norm = np.linalg.norm(y)
        diff_peak = float(np.max(np.abs(diff))) if diff.size else 0.0
        y_peak = float(np.max(np.abs(y))) if y.size else 0.0
        scale = _best_fit_complex_scale(x, y)
        rel_l2 = float(np.linalg.norm(diff) / y_norm) if y_norm else 0.0
        out[channel] = {
            "max_abs": diff_peak,
            "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
            "rel_l2": rel_l2,
            "overlap": _complex_overlap(x, y),
            "ref_l2": float(y_norm),
            "ref_peak_abs": y_peak,
            "max_abs_over_ref_peak": float(diff_peak / y_peak) if y_peak else 0.0,
            "best_fit_scale_abs": float(np.abs(scale)),
            "best_fit_scale_phase": float(np.angle(scale)),
        }
    return out


def _mode_metrics(
    lhs_modes: dict[Any, dict[str, Any]],
    rhs_modes: dict[Any, dict[str, Any]],
    domain: UniformFrequencyDomain,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    common_modes = sorted(set(lhs_modes.keys()) & set(rhs_modes.keys()), key=str)
    full_grid = np.asarray(domain.sample_frequencies[domain.min_idx :], dtype=np.float64)

    for mode in common_modes:
        lhs_freqs, lhs_complex = _mode_complex_series(lhs_modes[mode])
        rhs_freqs, rhs_complex = _mode_complex_series(rhs_modes[mode])
        lo = max(float(np.min(lhs_freqs)), float(np.min(rhs_freqs)), float(full_grid[0]))
        hi = min(float(np.max(lhs_freqs)), float(np.max(rhs_freqs)), float(full_grid[-1]))
        mask = (full_grid >= lo) & (full_grid <= hi)
        target_freqs = full_grid[mask]
        if target_freqs.size < 2:
            out[str(mode)] = {
                "available": False,
                "reason": "no overlapping frequency support",
            }
            continue

        lhs_interp = _interp_complex(lhs_freqs, lhs_complex, target_freqs)
        rhs_interp = _interp_complex(rhs_freqs, rhs_complex, target_freqs)
        diff = lhs_interp - rhs_interp
        rhs_norm = np.linalg.norm(rhs_interp)
        rhs_peak = float(np.max(np.abs(rhs_interp))) if rhs_interp.size else 0.0
        diff_peak = float(np.max(np.abs(diff))) if diff.size else 0.0
        scale = _best_fit_complex_scale(lhs_interp, rhs_interp)
        out[str(mode)] = {
            "available": True,
            "n_freqs": int(target_freqs.size),
            "f_min": float(target_freqs[0]),
            "f_max": float(target_freqs[-1]),
            "rel_l2": float(np.linalg.norm(diff) / rhs_norm) if rhs_norm else 0.0,
            "max_abs": diff_peak,
            "max_abs_over_ref_peak": float(diff_peak / rhs_peak) if rhs_peak else 0.0,
            "overlap": _complex_overlap(lhs_interp, rhs_interp),
            "best_fit_scale_abs": float(np.abs(scale)),
            "best_fit_scale_phase": float(np.angle(scale)),
        }
    return out


def _fit_global_alignment(
    lhs: dict[str, np.ndarray],
    rhs: dict[str, np.ndarray],
    freqs: np.ndarray,
    min_idx: int,
) -> dict[str, float] | None:
    fit_freqs = []
    fit_phases = []
    fit_weights = []

    for channel in sorted(lhs.keys() & rhs.keys()):
        x = _as_frequency_series(lhs[channel], min_idx=min_idx)
        y = _as_frequency_series(rhs[channel], min_idx=min_idx)
        amp = np.abs(x) * np.abs(y)
        if amp.size == 0:
            continue
        threshold = float(np.max(amp)) * 1.0e-6 if np.max(amp) > 0 else 0.0
        mask = np.isfinite(x) & np.isfinite(y) & (amp > threshold)
        if np.count_nonzero(mask) < 8:
            continue
        phase_diff = np.unwrap(np.angle(y[mask] * np.conj(x[mask])))
        fit_freqs.append(freqs[min_idx:][mask])
        fit_phases.append(phase_diff)
        fit_weights.append(np.sqrt(amp[mask]))

    if not fit_freqs:
        return None

    fit_freqs_arr = np.concatenate(fit_freqs)
    fit_phases_arr = np.concatenate(fit_phases)
    fit_weights_arr = np.concatenate(fit_weights)
    slope, intercept = np.polyfit(fit_freqs_arr, fit_phases_arr, deg=1, w=fit_weights_arr)
    return {
        "dt_seconds": float(slope / (2.0 * np.pi)),
        "phase_radians": float(intercept),
    }


def _apply_alignment(
    waveform: dict[str, np.ndarray],
    freqs: np.ndarray,
    alignment: dict[str, float] | None,
) -> dict[str, np.ndarray]:
    if alignment is None:
        return {k: np.asarray(v).copy() for k, v in waveform.items()}

    phase = 2.0 * np.pi * freqs * alignment["dt_seconds"] + alignment["phase_radians"]
    factor = np.exp(-1j * phase)
    return {
        channel: np.asarray(values) * factor
        for channel, values in waveform.items()
    }


def _all_close(metrics: dict[str, dict[str, float]], rtol: float, atol: float) -> bool:
    for values in metrics.values():
        ref_is_effectively_zero = (
            values["ref_l2"] <= atol and values["ref_peak_abs"] <= atol
        )
        if ref_is_effectively_zero:
            if values["max_abs"] > atol:
                return False
            continue
        if values["rel_l2"] > rtol:
            return False
        if values["overlap"] < 1.0 - rtol:
            return False
        if values["max_abs_over_ref_peak"] > rtol:
            return False
    return True


def _build_params(args: argparse.Namespace) -> dict[str, float]:
    return {
        "Mchirp": args.Mchirp,
        "q": args.q,
        "chi1": args.chi1,
        "chi2": args.chi2,
        "phi": args.phi,
        "inc": args.inc,
        "geocent_time": args.geocent_time,
        "Deltat": args.Deltat,
        "dist": args.dist,
        "lambda": args.__dict__["lambda"],
        "beta": args.beta,
        "psi": args.psi,
    }


def _format_report(report: dict[str, Any]) -> str:
    lines = []
    lines.append("Configuration")
    lines.append(
        f"  domain: f_min={report['config']['f_min']:.6g} Hz, "
        f"f_max={report['config']['f_max']:.6g} Hz, "
        f"delta_f={report['config']['delta_f']:.6g} Hz"
    )
    lines.append(
        f"  modes: {','.join(f'{l}{m}' for l, m in report['config']['modes'])}"
    )
    lines.append(
        f"  channels: {','.join(report['config']['channels'])}, "
        f"use_gpu={report['config']['use_gpu']}"
    )
    lines.append("")
    lines.append("Timing")
    for name, values in report["timings"].items():
        lines.append(
            f"  {name}: mean={values['mean_s']:.6f}s "
            f"median={values['median_s']:.6f}s "
            f"min={values['min_s']:.6f}s max={values['max_s']:.6f}s"
        )
    lines.append("")
    lines.append("Detector-Frame Agreement")
    for name, channel_metrics in report["comparisons"].items():
        status = "PASS" if channel_metrics["all_close"] else "FAIL"
        lines.append(f"  {name}: {status}")
        if "alignment" in channel_metrics and channel_metrics["alignment"] is not None:
            lines.append(
                "    alignment: "
                f"dt={channel_metrics['alignment']['dt_seconds']:.6e}s "
                f"phase0={channel_metrics['alignment']['phase_radians']:.6e}rad"
            )
        for channel, values in channel_metrics["channels"].items():
            lines.append(
                f"    {channel}: rel_l2={values['rel_l2']:.3e} "
                f"max_abs={values['max_abs']:.3e} "
                f"max_abs/ref_peak={values['max_abs_over_ref_peak']:.3e} "
                f"overlap={values['overlap']:.12f} "
                f"|scale|={values['best_fit_scale_abs']:.6e} "
                f"arg(scale)={values['best_fit_scale_phase']:.6e}"
            )
    if "intrinsic_modes" in report:
        lines.append("")
        lines.append("Intrinsic Mode Agreement")
        for mode, values in report["intrinsic_modes"].items():
            if not values.get("available", False):
                lines.append(f"  {mode}: unavailable ({values['reason']})")
                continue
            lines.append(
                f"  {mode}: rel_l2={values['rel_l2']:.3e} "
                f"max_abs/ref_peak={values['max_abs_over_ref_peak']:.3e} "
                f"overlap={values['overlap']:.12f} "
                f"|scale|={values['best_fit_scale_abs']:.6e} "
                f"arg(scale)={values['best_fit_scale_phase']:.6e}"
            )
    return "\n".join(lines)


def format_report(report: dict[str, Any]) -> str:
    return _format_report(report)


def run_comparison(args: argparse.Namespace | ComparisonConfig | None = None, **overrides):
    _require_waveform_stack()
    if args is None:
        args = ComparisonConfig(**overrides)
    elif overrides:
        if isinstance(args, ComparisonConfig):
            base = args.to_namespace()
        else:
            base = argparse.Namespace(**vars(args))
        for key, value in overrides.items():
            setattr(base, "lambda" if key == "lambda_" else key, value)
        args = base

    if isinstance(args, ComparisonConfig):
        args = args.to_namespace()
    if hasattr(args, "skip_alignment_diagnostics") and args.skip_alignment_diagnostics:
        args.include_alignment_diagnostics = False
    if hasattr(args, "skip_bbhx_selfcheck") and args.skip_bbhx_selfcheck:
        args.include_bbhx_selfcheck = False

    channels = _canonicalize_channels(args.channels)
    params = _build_params(args)
    domain = _build_domain(args)
    lisa_gen, bbhx_gen = _build_generators(domain, args)
    projector = _build_projector(domain, channels)

    lisa_modes, lisa_gen_timing = _benchmark(
        "lisabeta_generate_modes",
        lambda: lisa_gen.generate_amp_phase(params),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    bbhx_modes, bbhx_gen_timing = _benchmark(
        "bbhx_generate_modes",
        lambda: bbhx_gen.generate_amp_phase_m(params),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    bbhx_training_transform = GenerateBBHxDirectResponse(
        waveform_generator=bbhx_gen,
        channels=channels,
        gpu_fastpath=False,
        backend_native_fused=False,
    )

    lisa_strain, lisa_resp_timing = _benchmark(
        "lisabeta_response_on_lisabeta_modes",
        lambda: projector(_make_projection_sample(lisa_modes, params))["waveform"],
        warmup=args.warmup,
        repeats=args.repeats,
    )
    bbhx_training_strain, bbhx_training_timing = _benchmark(
        "bbhx_training_direct_response",
        lambda: bbhx_training_transform(_make_bbhx_training_sample(params))["waveform"],
        warmup=args.warmup,
        repeats=args.repeats,
    )

    direct_bbhx_raw, direct_bbhx_timing = _benchmark(
        "bbhx_direct_aet",
        lambda: bbhx_gen.generate_amp_phase(params),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    direct_bbhx = _build_projector(domain, channels)(
        _make_projection_sample(direct_bbhx_raw, params)
    )["waveform"]
    bbhx_native_component_timings = _benchmark_bbhx_native_components(
        bbhx_gen,
        params,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    bbhx_strain_via_response, bbhx_resp_timing = _benchmark(
        "lisabeta_response_on_bbhx_modes",
        lambda: projector(_make_projection_sample(bbhx_modes, params))["waveform"],
        warmup=args.warmup,
        repeats=args.repeats,
    )

    comparisons: dict[str, Any] = {}
    lisa_vs_bbhx_training = _channel_metrics(
        lhs=lisa_strain,
        rhs=bbhx_training_strain,
        min_idx=domain.min_idx,
    )
    comparisons["lisabeta_training_vs_bbhx_training_direct_response"] = {
        "channels": lisa_vs_bbhx_training,
        "all_close": _all_close(lisa_vs_bbhx_training, args.rtol, args.atol),
    }

    lisa_vs_bbhx_direct = _channel_metrics(
        lhs=lisa_strain,
        rhs=direct_bbhx,
        min_idx=domain.min_idx,
    )
    comparisons["lisabeta_vs_bbhx_direct"] = {
        "channels": lisa_vs_bbhx_direct,
        "all_close": _all_close(lisa_vs_bbhx_direct, args.rtol, args.atol),
    }

    if getattr(args, "include_alignment_diagnostics", True):
        freqs = np.asarray(domain.sample_frequencies, dtype=np.float64)
        training_alignment = _fit_global_alignment(
            lhs=lisa_strain,
            rhs=bbhx_training_strain,
            freqs=freqs,
            min_idx=domain.min_idx,
        )
        bbhx_training_aligned = _apply_alignment(
            bbhx_training_strain, freqs, training_alignment
        )
        lisa_vs_bbhx_training_aligned = _channel_metrics(
            lhs=lisa_strain,
            rhs=bbhx_training_aligned,
            min_idx=domain.min_idx,
        )
        comparisons["lisabeta_training_vs_bbhx_training_direct_response_after_global_alignment"] = {
            "alignment": training_alignment,
            "channels": lisa_vs_bbhx_training_aligned,
            "all_close": _all_close(
                lisa_vs_bbhx_training_aligned, args.rtol, args.atol
            ),
        }

        training_direct_alignment = _fit_global_alignment(
            lhs=bbhx_training_strain,
            rhs=direct_bbhx,
            freqs=freqs,
            min_idx=domain.min_idx,
        )
        direct_bbhx_aligned_to_training = _apply_alignment(
            direct_bbhx, freqs, training_direct_alignment
        )
        bbhx_training_vs_direct_aligned = _channel_metrics(
            lhs=bbhx_training_strain,
            rhs=direct_bbhx_aligned_to_training,
            min_idx=domain.min_idx,
        )
        comparisons["bbhx_training_direct_response_vs_bbhx_direct_after_global_alignment"] = {
            "alignment": training_direct_alignment,
            "channels": bbhx_training_vs_direct_aligned,
            "all_close": _all_close(
                bbhx_training_vs_direct_aligned, args.rtol, args.atol
            ),
        }

        alignment = _fit_global_alignment(
            lhs=lisa_strain,
            rhs=direct_bbhx,
            freqs=freqs,
            min_idx=domain.min_idx,
        )
        bbhx_direct_aligned = _apply_alignment(direct_bbhx, freqs, alignment)
        lisa_vs_bbhx_direct_aligned = _channel_metrics(
            lhs=lisa_strain,
            rhs=bbhx_direct_aligned,
            min_idx=domain.min_idx,
        )
        comparisons["lisabeta_vs_bbhx_direct_after_global_alignment"] = {
            "alignment": alignment,
            "channels": lisa_vs_bbhx_direct_aligned,
            "all_close": _all_close(
                lisa_vs_bbhx_direct_aligned, args.rtol, args.atol
            ),
        }

    lisa_vs_bbhx_response = _channel_metrics(
        lhs=lisa_strain,
        rhs=bbhx_strain_via_response,
        min_idx=domain.min_idx,
    )
    comparisons["lisabeta_vs_bbhx_via_common_response_bridge"] = {
        "channels": lisa_vs_bbhx_response,
        "all_close": _all_close(lisa_vs_bbhx_response, args.rtol, args.atol),
    }

    bbhx_training_vs_direct = _channel_metrics(
        lhs=bbhx_training_strain,
        rhs=direct_bbhx,
        min_idx=domain.min_idx,
    )
    comparisons["bbhx_training_direct_response_vs_bbhx_direct"] = {
        "channels": bbhx_training_vs_direct,
        "all_close": _all_close(bbhx_training_vs_direct, args.rtol, args.atol),
    }

    if getattr(args, "include_alignment_diagnostics", True):
        freqs = np.asarray(domain.sample_frequencies, dtype=np.float64)
        bridge_alignment = _fit_global_alignment(
            lhs=lisa_strain,
            rhs=bbhx_strain_via_response,
            freqs=freqs,
            min_idx=domain.min_idx,
        )
        bbhx_bridge_aligned = _apply_alignment(
            bbhx_strain_via_response, freqs, bridge_alignment
        )
        lisa_vs_bbhx_bridge_aligned = _channel_metrics(
            lhs=lisa_strain,
            rhs=bbhx_bridge_aligned,
            min_idx=domain.min_idx,
        )
        comparisons["lisabeta_vs_bbhx_via_common_response_bridge_after_global_alignment"] = {
            "alignment": bridge_alignment,
            "channels": lisa_vs_bbhx_bridge_aligned,
            "all_close": _all_close(
                lisa_vs_bbhx_bridge_aligned, args.rtol, args.atol
            ),
        }

    if getattr(args, "include_bbhx_selfcheck", True):
        bbhx_selfcheck = _channel_metrics(
            lhs=direct_bbhx,
            rhs=bbhx_strain_via_response,
            min_idx=domain.min_idx,
        )
        comparisons["bbhx_direct_vs_bbhx_modes_via_lisabeta_response"] = {
            "channels": bbhx_selfcheck,
            "all_close": _all_close(bbhx_selfcheck, args.rtol, args.atol),
        }
        if getattr(args, "include_alignment_diagnostics", True):
            freqs = np.asarray(domain.sample_frequencies, dtype=np.float64)
            self_alignment = _fit_global_alignment(
                lhs=direct_bbhx,
                rhs=bbhx_strain_via_response,
                freqs=freqs,
                min_idx=domain.min_idx,
            )
            bbhx_modes_aligned = _apply_alignment(
                bbhx_strain_via_response, freqs, self_alignment
            )
            bbhx_selfcheck_aligned = _channel_metrics(
                lhs=direct_bbhx,
                rhs=bbhx_modes_aligned,
                min_idx=domain.min_idx,
            )
            comparisons["bbhx_direct_vs_bbhx_modes_after_global_alignment"] = {
                "alignment": self_alignment,
                "channels": bbhx_selfcheck_aligned,
                "all_close": _all_close(
                    bbhx_selfcheck_aligned, args.rtol, args.atol
                ),
            }

    report = {
        "config": {
            "f_min": args.f_min,
            "f_max": args.f_max,
            "delta_f": args.delta_f,
            "f_ref": args.f_ref,
            "modes": args.modes,
            "channels": channels,
            "use_gpu": bool(args.use_gpu),
            "parameters": params,
            "lisa_settings": DEFAULT_LISA_SETTINGS,
        },
        "timings": {
            "lisabeta_generate_modes": lisa_gen_timing,
            "bbhx_training_direct_response": bbhx_training_timing,
            "bbhx_direct_aet": direct_bbhx_timing,
            "bbhx_generate_modes": bbhx_gen_timing,
            "lisabeta_response_on_lisabeta_modes": lisa_resp_timing,
            "lisabeta_response_on_bbhx_modes": bbhx_resp_timing,
            **bbhx_native_component_timings,
        },
        "comparisons": comparisons,
        "intrinsic_modes": _mode_metrics(lisa_modes, bbhx_modes, domain),
    }

    return report


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = run_comparison(args)

    if args.json:
        print(json.dumps(report, indent=2, default=float))
    else:
        print(format_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
