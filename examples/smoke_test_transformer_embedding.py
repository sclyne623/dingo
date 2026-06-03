"""Smoke test for the transformer-embedding wiring.

Run in an environment where `torch` and `dingo` are installed.

Default (synthetic, no data / no lisabeta needed):
    python examples/smoke_test_transformer_embedding.py

  Stage A (critical): build the transformer-embedding normalizing flow from
  synthetic *tokenized* tensors and run one forward + backward. Validates
  autocomplete_model_kwargs (transformer branch), the
  create_nsf_with_rb_projection_embedding_net dispatch, TransformerEmbeddingAdapter,
  TransformerModel.forward, and the flow context wiring.

  Stage B (best-effort): run StrainTokenization on a synthetic uniform domain and
  print/assert the output shapes (imported directly from the submodule to avoid
  the lisabeta/GSL import chain in dingo.gw.transforms.__init__).

Full pipeline mode (needs your real dataset + lisabeta/GSL working):
    python examples/smoke_test_transformer_embedding.py --full examples/transformer_lisa_train.yaml [--n 2]

  Loads N waveforms from the config's waveform_dataset_path, builds the REAL
  set_train_transforms (LISA response -> whiten -> repackage -> StrainTokenization)
  and prints the tokenized sample shapes. This is the test that exercises your
  actual multibanded grid and will surface the MFD/token-alignment constraint.
  A dummy parameter standardization is injected so it does NOT run the slow
  standardization sweep -- it only checks shapes / the tokenization assert.
"""
import argparse
import traceback

import numpy as np
import torch


def stage_a():
    print("\n=== STAGE A: model build + forward/backward (synthetic tokens) ===")
    from dingo.core.posterior_models.build_model import autocomplete_model_kwargs
    from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net

    batch, num_blocks, n_params = 4, 2, 8
    num_bins_per_token, num_channels = 16, 3
    n_tok_per_block = 50
    num_tokens = num_blocks * n_tok_per_block
    num_features = num_channels * num_bins_per_token

    waveform = torch.randn(batch, num_tokens, num_features)
    position = torch.zeros(batch, num_tokens, 3)
    position[..., 0] = torch.linspace(1e-4, 1e-1, num_tokens)
    position[..., 1] = position[..., 0] + 1e-4
    position[:, :n_tok_per_block, 2] = 0
    position[:, n_tok_per_block:, 2] = 1
    mask = torch.zeros(batch, num_tokens, dtype=torch.bool)
    theta = torch.randn(batch, n_params)

    embedding_kwargs = {
        "transformer_kwargs": {
            "d_model": 64, "nhead": 4, "num_layers": 2,
            "dim_feedforward": 128, "dropout": 0.1, "norm_first": True,
        },
        "tokenizer_kwargs": {"hidden_dims": 64, "activation": "gelu", "batch_norm": True},
        "positional_encoder_kwargs": {"positional_encoding_type": "continuous"},
        "block_encoder_kwargs": {"num_blocks": num_blocks, "block_encoding_type": "sine"},
        "final_net_kwargs": {"output_dim": 128, "hidden_dims": [128, 128], "activation": "gelu"},
    }
    posterior_kwargs = {
        "num_flow_steps": 4,
        "base_transform_kwargs": {
            "hidden_dim": 128, "num_transform_blocks": 2, "activation": "elu",
            "dropout_probability": 0.0, "batch_norm": True, "num_bins": 8,
            "base_transform_type": "rq-coupling",
        },
    }
    model_kwargs = {
        "posterior_model_type": "normalizing_flow",
        "embedding_kwargs": embedding_kwargs,
        "posterior_kwargs": posterior_kwargs,
    }

    data_sample = [theta[0], waveform[0], position[0], mask[0]]
    autocomplete_model_kwargs(model_kwargs, data_sample)
    print(
        f"  autocomplete -> context_dim={model_kwargs['posterior_kwargs']['context_dim']} "
        f"input_dim={model_kwargs['posterior_kwargs']['input_dim']} "
        f"tokenizer.input_dims={model_kwargs['embedding_kwargs']['tokenizer_kwargs']['input_dims']}"
    )

    model = create_nsf_with_rb_projection_embedding_net(
        posterior_kwargs=model_kwargs["posterior_kwargs"],
        embedding_kwargs=model_kwargs["embedding_kwargs"],
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"  model built: {n:,} trainable parameters")

    loss = -model.log_prob(theta, waveform, position, mask).mean()
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"  forward+backward OK: loss={loss.item():.4f}, finite grads={grad_ok}")
    assert torch.isfinite(loss), "loss is not finite"
    assert grad_ok, "no finite gradients"
    print("  STAGE A PASS")


def stage_b():
    print("\n=== STAGE B: StrainTokenization shape check (uniform domain) ===")
    from dingo.gw.domains.build_domain import build_domain
    from dingo.gw.transforms.tokenization_transforms import StrainTokenization

    domain = build_domain({"type": "FD", "f_min": 0.0, "f_max": 0.01, "delta_f": 1e-5})
    num_bins = len(domain.sample_frequencies) - domain.min_idx
    strain = np.random.randn(2, 3, num_bins).astype(np.float64)
    sample = {"waveform": strain, "asds": {"chan1": None, "chan2": None}}

    tok = StrainTokenization(domain=domain, token_size=16, drop_last_token=True,
                             print_output=False)
    out = tok(sample)
    wf = np.asarray(out["waveform"])
    pos = np.asarray(out["position"])
    m = np.asarray(out["drop_token_mask"])
    print(f"  waveform tokens: {wf.shape}  (expect [num_tokens, num_features])")
    print(f"  position:        {pos.shape}  (expect [num_tokens, 3])")
    print(f"  drop_token_mask: {m.shape}")
    assert pos.shape[-1] == 3 and pos.ndim == 2, "position should be [num_tokens, 3]"
    assert wf.shape[0] == pos.shape[0], "token count mismatch between waveform and position"
    print("  STAGE B PASS")


def _print_domain_bands(domain):
    """Print multibanded-domain band structure to help choose a token_size that
    keeps band nodes between tokens (the StrainTokenization MFD constraint)."""
    print(f"  domain: {type(domain).__name__}")
    for attr in ("nodes", "nodes_indices", "_nodes_indices", "delta_f_bands",
                 "_delta_f_bands", "num_bins_bands", "_num_bins_bands"):
        v = getattr(domain, attr, None)
        if v is not None:
            arr = np.asarray(v)
            tail = " ..." if arr.size > 16 else ""
            print(f"    {attr} = {arr.ravel()[:16]}{tail}")
    sf = getattr(domain, "sample_frequencies", None)
    if sf is not None:
        print(f"    num bins (full) = {len(sf)}, min_idx = {getattr(domain, 'min_idx', '?')}")


def full_mode(config_path, n=2):
    print(f"\n=== FULL: real set_train_transforms on {config_path} ===")
    import yaml
    from dingo.gw.training.train_builders import build_dataset, set_train_transforms

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_settings = cfg["data"]
    on_fly = bool(cfg.get("local", {}).get("on_fly", False))
    asd_path = cfg["training"]["stage_0"]["asd_dataset_path"]

    if "tokenization" not in data_settings:
        print("  WARNING: config has no `data: tokenization:` block -> this would "
              "test the SVD/RB path, not the transformer.")

    # Inject a dummy parameter standardization so set_train_transforms skips the
    # slow standardization sweep (we only care about shapes / the MFD assert).
    params = list(data_settings.get("inference_parameters", [])) + list(
        data_settings.get("context_parameters", [])
    )
    data_settings.setdefault(
        "standardization",
        {"mean": {p: 0.0 for p in params}, "std": {p: 1.0 for p in params}},
    )

    print("  loading waveform dataset ...")
    wfd = build_dataset(data_settings, on_fly=on_fly)
    print(f"  dataset: {len(wfd)} samples")
    _print_domain_bands(wfd.domain)

    try:
        set_train_transforms(wfd, data_settings, asd_path, print_output=False)
    except AssertionError as e:
        print("\n  !! MFD/token-alignment assert tripped while building transforms:")
        print(f"     {e}")
        print("  -> pick a token_size (or num_tokens) so band nodes land between "
              "tokens; use the band indices printed above.")
        raise

    for i in range(n):
        sample = wfd[i]
        print(f"  sample[{i}]: {len(sample)} tensors (order = selected_keys)")
        for j, t in enumerate(sample):
            shp = getattr(t, "shape", None)
            dt = getattr(t, "dtype", type(t).__name__)
            print(f"     [{j}] shape={tuple(shp) if shp is not None else '-'} dtype={dt}")
    print("  FULL PASS -- tokenized shapes look consistent; ready for a short train.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", metavar="CONFIG_YAML", default=None,
                        help="Run the real pipeline on the dataset in this config.")
    parser.add_argument("--n", type=int, default=2, help="Number of samples to inspect in --full.")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    if args.full:
        try:
            full_mode(args.full, args.n)
        except Exception:
            print("  FULL FAILED:")
            traceback.print_exc()
    else:
        ok = True
        try:
            stage_a()
        except Exception:
            ok = False
            print("  STAGE A FAILED:")
            traceback.print_exc()
        try:
            stage_b()
        except Exception:
            print("  STAGE B FAILED (non-fatal, inspect shapes/domain):")
            traceback.print_exc()
        print("\nDONE." + ("" if ok else "  (Stage A failed -- fix wiring before training.)"))
