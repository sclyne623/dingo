# Configure lisabeta's internal frequency grid

For `LISA: true` datasets using `LISAWaveformGenerator`, set the following keys
in the **dataset-generation YAML**, under `waveform_generator`:

```yaml
waveform_generator:
  approximant: IMRPhenomXHM
  f_ref: 0.0
  spin_conversion_phase: 0
  LISA: true
  on_fly: true
  acc: 1.0e-4
  DeltalnMf_max: 0.0015625
```

Keep your existing domain, priors, and other dataset settings. The denser spacing
above is an example to validate against your waveform population, not a recommended
accuracy threshold. Both values must be finite, positive real numbers. Use YAML
numeric values (for example `1.0e-4`), not quoted strings.

- `acc` defaults to `1e-4`. It controls the approximate inspiral phase interpolation
  error criterion used to build lisabeta's grid. It does not bound amplitude error,
  detector-waveform mismatch, or SVD mismatch.
- `DeltalnMf_max` defaults to `0.025`. It limits logarithmic frequency spacing;
  lowering it requests a denser internal grid and can increase generation and
  response-evaluation costs.

Omitting both keys preserves the previous grid settings. These controls change the
frequencies at which lisabeta generates native modes; the final DINGO frequency
domain and the higher-mode frequency-scaling convention remain the same. They do
not configure the BBHx or ordinary LAL waveform generators.

Dataset generation and on-the-fly waveform generation both pass the dataset's
`waveform_generator` settings to this constructor. On-the-fly generation reads
settings stored in the dataset: adding these keys only to a training YAML does
not change an existing dataset. Generate a dataset with the desired settings and
rebuild its SVD basis consistently, then point training at the new dataset.
Existing cached waveforms and bases are not updated by this configuration change.

Check convergence against more densely evaluated waveforms on outliers and fresh
validation sources before choosing a production value. Good SVD reconstruction
of the old waveforms is not evidence that their interpolation is accurate.
