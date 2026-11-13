#!/usr/bin/env python3
"""
Build the embedding-network SVD for a training run on CPU, ahead of the GPU job.

Runs the same code as dingo_train (build_dataset + build_svd_for_embedding_network,
same training YAML), but:
  * generates the SVD training waveforms with DataLoader workers (dingo_train makes
    them serially), and
  * leaves the full BLAS thread pool to the SVD (do not export OMP/MKL/OPENBLAS
    _NUM_THREADS=1 for this job; the workers are forked under a 1-thread limit).

It writes <train_dir>/svd_<detector>.hdf5 (plus the SVD validation mismatches in the
log). Point the training YAML at them with

    model:
      embedding_kwargs:
        svd:
          precomputed_files: {chan1: svd_chan1.hdf5, chan2: svd_chan2.hdf5}

and dingo_train loads them (paths relative to --train_dir) instead of rebuilding.

Usage:
    python build_embedding_svd.py --settings_file cpu_8_wk_psi_dt.yaml \
        --train_dir 8wk_free_time --num_workers 30
"""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_info

from dingo.gw.training.train_builders import build_dataset, build_svd_for_embedding_network


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings_file", required=True, help="Training YAML (the one dingo_train will use).")
    ap.add_argument("--train_dir", required=True, help="Training directory; svd_<detector>.hdf5 are written here.")
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="DataLoader workers for waveform generation.")
    ap.add_argument("--batch_size", type=int, default=250,
                    help="Waveforms per worker batch (memory per batch ~ batch x channels x bins x 16 B).")
    ap.add_argument("--method", default=None,
                    help="SVD backend (scipy, svds, random). Default: the YAML's svd.method, else scipy.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="Overwrite existing svd_*.hdf5 in train_dir.")
    return ap.parse_args()


def dingo_commit():
    import dingo

    d = os.path.dirname(os.path.dirname(os.path.abspath(dingo.__file__)))
    try:
        return subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    args = parse_args()
    t0 = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.settings_file) as f:
        train_settings = yaml.safe_load(f)
    local_settings = train_settings.pop("local", {}) or {}
    detectors = train_settings["data"]["detectors"]
    svd_kwargs = deepcopy(train_settings["model"]["embedding_kwargs"]["svd"])
    svd_kwargs.pop("precomputed_files", None)
    svd_kwargs.pop("generation_num_workers", None)
    method = args.method or svd_kwargs.pop("method", None) or "scipy"
    svd_kwargs.pop("method", None)

    os.makedirs(args.train_dir, exist_ok=True)
    out_files = [os.path.join(args.train_dir, f"svd_{ifo}.hdf5") for ifo in detectors]
    existing = [p for p in out_files if os.path.exists(p)]
    if existing and not args.force:
        raise SystemExit(f"Refusing to overwrite {existing}; pass --force.")

    print("dingo:", dingo_commit())
    print("BLAS:", [(i.get("internal_api"), i.get("num_threads")) for i in threadpool_info()])
    print(f"SVD: size={svd_kwargs.get('size')} method={method} "
          f"train={svd_kwargs.get('num_training_samples')} val={svd_kwargs.get('num_validation_samples')} "
          f"workers={args.num_workers} batch={args.batch_size}", flush=True)

    # Same dataset construction as prepare_training_new (no local caching).
    wfd = build_dataset(
        data_settings=deepcopy(train_settings["data"]),
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        on_fly=local_settings.get("on_fly", True),
    )

    v_list = build_svd_for_embedding_network(
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset_path"],
        batch_size=args.batch_size,
        method=method,
        out_dir=args.train_dir,
        generation_num_workers=args.num_workers,
        **svd_kwargs,
    )

    missing = [p for p in out_files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"SVD finished but files are missing: {missing}")
    record = {
        "settings_file": os.path.abspath(args.settings_file),
        "train_dir": os.path.abspath(args.train_dir),
        "files": out_files,
        "V_shapes_after_min_idx": [list(v.shape) for v in v_list],
        "method": method,
        "svd_settings": svd_kwargs,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dingo_commit": dingo_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_time_s": time.time() - t0,
    }
    with open(os.path.join(args.train_dir, "svd_build_info.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nWrote {out_files} in {time.time() - t0:.0f} s.")
    print("Training YAML: model.embedding_kwargs.svd.precomputed_files = "
          + json.dumps({ifo: os.path.basename(p) for ifo, p in zip(detectors, out_files)}))


if __name__ == "__main__":
    sys.exit(main())
