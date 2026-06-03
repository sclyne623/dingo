# Transformer embedding for LISA — lift from `dingo-t1`

This documents the transformer-embedding integration lifted from the
`dingo-t1` branch of `dingo-gw/dingo` into this LISA fork. It replaces the
frozen SVD reduced-basis embedding (which hit a ~1e-3 noise-weighted
compression wall at size 500 for full-sky LISA data) with a learned global
transformer encoder that has no linear n-width bottleneck.

## What was lifted (verbatim from `dingo-t1`)

| file | notes |
|---|---|
| `dingo/core/nn/transformer.py` | `Tokenizer, PositionalEncoding, BlockEncoding, MultiPositionalEncoding, TransformerModel, PoolingTransformer, create_transformer_enet` |
| `dingo/core/nn/resnet.py` | `DenseResidualNet, MLP, LinearLayer` (dependency of transformer.py; this fork lacked a `resnet.py`) |
| `dingo/gw/transforms/tokenization_transforms.py` | `StrainTokenization` + MFD-compat check |
| `dingo.gw.gwutils.add_defaults_for_missing_ifos` | small helper needed by `StrainTokenization` |

## LISA edits applied

- **`tokenization_transforms.py`**: `DETECTOR_DICT` changed from LIGO
  `{"H1":0,"L1":1,"V1":2}` to `{"chan1":0,"chan2":1}` (your TDI A/E channels).
  TODO: generalize if you add channel T.

## Wiring added (this fork's files)

1. `dingo/gw/transforms/__init__.py` — export `StrainTokenization`.
2. `dingo/gw/training/train_builders.py` (`set_train_transforms`) — appends a
   tokenizer and adds `position` / `drop_token_mask` to `selected_keys` when a
   `data: tokenization:` block is present. The data pipeline before it
   (response → whiten → repackage) is unchanged. Two tokenizers:
   - `StrainTokenization` (global stride) — needs every MFD band's bin count
     divisible by `token_size`.
   - `MultibandStrainTokenization` (`per_band: true`) — tokenizes each band
     independently; boundaries always align (last token per band zero-padded).
     **Required for the LISA 8-week grid**, whose band counts
     `[2000,3750,5000,2500,3125,1562,781,781,390]` are coprime (GCD 1), so no
     global `token_size > 1` exists. At `token_size=16` → 1247 tokens/channel,
     63 zero-pad bins out of 19889. (Added in `tokenization_transforms.py`.)
3. `dingo/core/nn/nsf.py` — `TransformerEmbeddingAdapter` (drops the
   `(embedding, logging_info)` tuple so it matches this fork's single-return
   `FlowWrapper` convention) + dispatch in
   `create_nsf_with_rb_projection_embedding_net` keyed on the presence of
   `transformer_kwargs`.
4. `dingo/core/posterior_models/build_model.py` (`autocomplete_model_kwargs`) —
   transformer branch that sets `tokenizer_kwargs.input_dims/output_dim`,
   `final_net_kwargs.input_dim`, and the flow `context_dim`. (GNPE is not
   supported with the transformer, so the RB path's `data_sample[2]` GNPE
   detection is bypassed — `data_sample[2]` is the `position` tensor here.)
5. `dingo/gw/training/train_pipeline.py` — skips `build_svd_for_embedding_network`
   when `embedding_kwargs` has no `svd` block.

Config: `examples/transformer_lisa_train.yaml`.

## What is NOT included (left for later)

- **Masking / flexibility training** (variable detectors, frequency cuts,
  random token drops). `dingo-t1` has it; this lift intentionally omits it for a
  first working LISA model. Add the drop transforms from `dingo-t1`'s
  `set_train_transforms` later if you want missing-data robustness.
- **Pretraining** (`enets_pretraining.py`).
- **GNPE with the transformer** (not supported upstream either).

## TODOs to verify before/at first run (I could not run torch/lisabeta here)

1. **MFD/token alignment.** `StrainTokenization` asserts your multibanded band
   nodes fall *between* tokens (uniform spacing within a token). If the assert
   trips, choose `token_size` (or `num_tokens`) compatible with your MFD bands.
2. **`num_blocks`.** Set to your channel count (2 for A/E) in
   `block_encoder_kwargs` (and `tokenizer_kwargs` if you enable
   `condition_on_position`). Not auto-derived.
3. **`position` tensor shape vs `data_sample` indexing.** Confirm that after
   `UnpackDict`, the per-sample context order is
   `(waveform_tokens, position, drop_token_mask)` so that
   `TransformerModel.forward(x, position, src_key_padding_mask)` lines up. The
   adapter passes them positionally.
4. **`context_dim` / `output_dim`.** The flow's `context_dim` is set to
   `final_net_kwargs.output_dim` (=128). Make sure that matches what you want.
5. **Config values** (`d_model`, `num_layers`, `nhead`, `token_size`) are
   starting points — tune. Reference values: clone the companion repo
   `dingo-gw/dingo-T1` (`01_paper_settings/01_training/03_training/`).

## Suggested validation order

1. Smoke test: build the model + one forward/backward on a tiny batch.
2. Short train on the **narrow** chirp prior; compare IS sample efficiency and a
   few posteriors against the frozen-SVD baseline.
3. Then widen the chirp prior — the regime where the SVD provably failed.

## Reference

Kofler et al., "Flexible Gravitational-Wave Parameter Estimation with
Transformers" (Dingo-T1). Code: `github.com/dingo-gw/dingo` branch `dingo-t1`;
companion `github.com/dingo-gw/dingo-T1`.
