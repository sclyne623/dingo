# LISA: free coalescence time and polarization (plain NPE)

Branch `claude/lisa-free-time`. Implementation: `dingo/gw/lisa_time.py`, applied in
`ProjectOntoSpaceDetectors` (lisabeta response only; BBHx raises).

## Time parameters

| Name           | Meaning                                                     |
|----------------|-------------------------------------------------------------|
| `geocent_time` | Fixed orbital reference `t0_s` (s). Keep it a DeltaFunction. |
| `Deltat`       | SSB-frame offset: t_SSB = t0_s + Deltat (lisabeta `Deltat`). |
| `DeltatL`      | LISA-frame offset: t_L = t0_s + DeltatL (arrival at LISA).   |

Sample one of `Deltat` / `DeltatL` in the extrinsic prior; the other is derived
from the sky position and written to `sample["parameters"]`. The waveform modes
must be generated with `Deltat = 0` (keep the intrinsic `Deltat` prior fixed at 0);
the projection refuses a nonzero intrinsic `Deltat` combined with an extrinsic shift.

Do not use the absolute `geocent_time` as the free time: at ~4.7e7 s, float32
destandardization resolves only ~4 s.

## Training settings (changes to the current config)

```yaml
data:
  extrinsic_prior:
    # unchanged: dist, lambda, beta
    psi: bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic")
    geocent_time: bilby.core.prior.analytical.DeltaFunction(47337224.645318404)  # unchanged
    DeltatL: bilby.core.prior.Uniform(minimum=-150.0, maximum=150.0)
  inference_parameters:
    # ... existing ...
    - psi
    - DeltatL
```

* Infer `DeltatL` (the sampled parameter), not the derived `Deltat`: its prior
  factorizes from the sky prior and its standardization is analytic.
* This changes the data settings and standardization: start a new training run;
  do not resume an existing checkpoint (resumes read the checkpoint metadata).
* Check the SVD residuals on a sample of the new data before reusing an SVD
  basis built with psi and time fixed.

## Validation

* Local, numpy only: `pytest tests/gw/test_lisa_time.py`.
* Cluster (needs lisabeta): `sbatch misc_scripts/lisa_time_shift_validation.sh`.
  Compares the new path against lisabeta generating the modes with `Deltat`
  directly, checks `DeltatL` against lisabeta's own response delay, and runs the
  training transforms with a `DeltatL` + free `psi` prior.
