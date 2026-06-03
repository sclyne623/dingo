"""Fast smoke test for the transformer-embedding wiring.

Run this in an environment where `torch` and `dingo` are installed (it does NOT
need lisabeta or any data files). It exercises the parts that could not be
runtime-tested in the sandbox where the integration was written.

    python examples/smoke_test_transformer_embedding.py

Stage A (critical): build the transformer-embedding normalizing flow from
synthetic *tokenized* tensors and run one forward + backward. This validates:
  - autocomplete_model_kwargs (transformer branch: input/context dims)
  - the create_nsf_with_rb_projection_embedding_net dispatch on transformer_kwargs
  - TransformerEmbeddingAdapter (tuple -> single-tensor return)
  - TransformerModel.forward(x, position, src_key_padding_mask)
  - the flow context wiring and a backward pass

Stage B (best-effort): run StrainTokenization on a synthetic uniform-domain
sample and print the produced shapes, so you can confirm the transform output
matches what the model consumes. Uses a UniformFrequencyDomain to avoid the
multibanded-node/token-alignment constraint (that is a separate TODO for your
real MFD grid).
"""
import copy
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

    # Synthetic tokenized batch (mimics StrainTokenization output, batched).
    waveform = torch.randn(batch, num_tokens, num_features)
    position = torch.zeros(batch, num_tokens, 3)
    position[..., 0] = torch.linspace(1e-4, 1e-1, num_tokens)        # f_min per token
    position[..., 1] = position[..., 0] + 1e-4                       # f_max per token
    position[:, :n_tok_per_block, 2] = 0                             # block 0 (chan1)
    position[:, n_tok_per_block:, 2] = 1                             # block 1 (chan2)
    mask = torch.zeros(batch, num_tokens, dtype=torch.bool)         # no tokens dropped
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

    # autocomplete_model_kwargs uses an *unbatched* sample (like wfd[0]).
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
    from dingo.gw.domains import build_domain
    from dingo.gw.transforms import StrainTokenization

    domain = build_domain({"type": "FD", "f_min": 0.0, "f_max": 0.01, "delta_f": 1e-5})
    num_bins = len(domain.sample_frequencies) - domain.min_idx
    strain = np.random.randn(2, 3, num_bins).astype(np.float64)  # [blocks, channels, bins]
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


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
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
        # Non-fatal: Stage B depends on domain/shape conventions that may differ;
        # the printed traceback tells you exactly what to adjust.
        print("  STAGE B FAILED (non-fatal, inspect shapes/domain):")
        traceback.print_exc()
    print("\nDONE." + ("" if ok else "  (Stage A failed -- fix wiring before training.)"))
