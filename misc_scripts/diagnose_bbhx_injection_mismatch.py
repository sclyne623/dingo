#!/usr/bin/env python
"""Diagnose the BBHx training-vs-injection waveform path mismatch.

Background
----------
With direct_response / gpu_fastpath / backend_native_fused enabled, TRAINING
builds the network input via
    waveform_generator.generate_direct_response_backend_native(...)
(BBHx's own LISA response, with decenter_waveform applied).

INJECTION (GWSignal.signal) instead builds it via
    waveform_generator.generate_amp_phase_m(...)        # intrinsic modes, NO response, NO decenter
    + ProjectOntoSpaceDetectors("TDIAET", ...)          # a *different* response model

If those two disagree, the network sees out-of-distribution data at inference
time and returns near-prior / confidently-wrong posteriors even though the
training loss is healthy.

This script regenerates the injection both ways for the SAME parameters and
overlays them against the waveform actually stored in the result file
(context/waveform), which is what the network was conditioned on.

Run on the cluster (needs the BBHx + GPU env that produced the result):
    python diagnose_bbhx_injection_mismatch.py \
        --model   /path/to/your/trained_model.pt \
        --result  /path/to/BBHX_injection.hdf5 \
        --out      mismatch.png
"""
import argparse

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dingo.gw.injection import Injection
from dingo.core.posterior_models.build_model import build_model_from_kwargs


def to_np(x):
    """CuPy/torch/numpy -> numpy."""
    if hasattr(x, "get"):
        return np.asarray(x.get())
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def channels(waveform_dict_or_array, names):
    """Return a dict {name: 1d complex array} from either a dict of channels
    or a stacked (n_channels, length) array."""
    if isinstance(waveform_dict_or_array, dict):
        return {n: to_np(waveform_dict_or_array[n]).reshape(-1) for n in names}
    arr = to_np(waveform_dict_or_array)
    arr = np.squeeze(arr)
    if arr.ndim == 1:  # single channel
        return {names[0]: arr}
    # assume first axis indexes channels
    return {n: arr[i].reshape(-1) for i, n in enumerate(names) if i < arr.shape[0]}


def inner(a, b, asd):
    """Noise-weighted inner product Re<a|b> using a one-sided ASD."""
    w = 1.0 / asd**2
    return np.real(np.sum(np.conj(a) * b * w))


def overlap(a, b, asd):
    aa = inner(a, a, asd)
    bb = inner(b, b, asd)
    if aa <= 0 or bb <= 0:
        return float("nan")
    return inner(a, b, asd) / np.sqrt(aa * bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained PosteriorModel .pt")
    ap.add_argument("--result", required=True, help="BBHX_injection.hdf5")
    ap.add_argument("--out", default="bbhx_mismatch.png")
    args = ap.parse_args()

    # --- rebuild the exact injection setup from the trained model -------------
    pm = build_model_from_kwargs(filename=args.model, device="cpu")
    inj = Injection.from_posterior_model_metadata(pm.metadata)
    wfg = inj.waveform_generator

    print("waveform_generator settings of interest:")
    for k in (
        "direct_response",
        "gpu_fastpath",
        "backend_native_fused",
        "decenter_waveform",
        "frozenLISA",
    ):
        print(f"  {k} = {getattr(wfg, k, '<missing>')}")
    print(f"  mode_list = {wfg.mode_list}")

    # --- read the result file: truth params, stored (injected) waveform, ASD --
    with h5py.File(args.result, "r") as f:
        chan_names = list(f["context/waveform"].keys())
        stored = {n: f[f"context/waveform/{n}"][:] for n in chan_names}
        asds = {n: f[f"context/asds/{n}"][:] for n in chan_names}

    # Injection parameters live in the model metadata under the run that made
    # the file; if your file carries them differently, set theta by hand here.
    theta = dict(pm.metadata.get("injection_parameters", {}))
    if not theta:
        raise SystemExit(
            "No injection_parameters in metadata; set `theta` manually in the script."
        )
    theta = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in theta.items()}
    print("\ninjection parameters:")
    for k, v in theta.items():
        print(f"  {k} = {v}")

    # --- PATH B: the injection path (what actually fed this inference) --------
    inj.whiten = False
    sig = inj.signal(theta)
    path_injection = channels(sig["waveform"], chan_names)

    # --- PATH A: the training path (direct backend-native response) ----------
    # Mirror detector_transforms: split params the way the generator expects.
    from dingo.gw.injection import split_off_extrinsic_parameters

    theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
    theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}
    h_train = wfg.generate_direct_response_backend_native(
        theta_intrinsic, theta_extrinsic, catch_waveform_errors=False
    )
    h_train = h_train["waveform"] if isinstance(h_train, dict) and "waveform" in h_train else h_train
    path_training = channels(h_train, chan_names)

    stored = {n: to_np(v).reshape(-1) for n, v in stored.items()}

    # --- compare -------------------------------------------------------------
    print("\nnoise-weighted overlaps (1.0 == identical up to amplitude):")
    for n in chan_names:
        o_bt = overlap(path_injection[n], path_training[n], asds[n])
        o_sb = overlap(stored[n], path_injection[n], asds[n])
        o_st = overlap(stored[n], path_training[n], asds[n])
        print(f"  {n}:  injection-vs-training = {o_bt:+.4f} | "
              f"stored-vs-injection = {o_sb:+.4f} | stored-vs-training = {o_st:+.4f}")

    print(
        "\nExpectation if this is the bug:\n"
        "  stored-vs-injection  ~ 1.0  (regeneration reproduces the fed data)\n"
        "  injection-vs-training << 1.0 (the two pipelines disagree -> OOD input)\n"
    )

    # --- plot ----------------------------------------------------------------
    fig, axes = plt.subplots(len(chan_names), 1, figsize=(11, 4 * len(chan_names)),
                             squeeze=False)
    for ax, n in zip(axes[:, 0], chan_names):
        ax.plot(np.abs(stored[n]), label="stored (fed to network)", lw=2.5, alpha=0.6)
        ax.plot(np.abs(path_injection[n]), label="injection path (regen)", ls="--")
        ax.plot(np.abs(path_training[n]), label="TRAINING path", ls=":")
        ax.set_title(f"{n}  |amplitude|")
        ax.set_xlabel("frequency bin")
        ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
