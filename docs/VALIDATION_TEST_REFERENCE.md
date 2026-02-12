# Validation Test Script - Quick Reference

## Purpose

The validation test script (`test_optimization_validation.py`) verifies that all LISA performance optimizations maintain **exact numerical equivalence** with the original implementation while providing significant **performance improvements**.

## Quick Usage

### Single command validation:
```bash
cd /Users/samclyne/Downloads/dingo-LISA
./validate_optimizations.sh
```

### Manual run:
```bash
python tests/gw/transforms/test_optimization_validation.py
```

### With pytest:
```bash
pytest tests/gw/transforms/test_optimization_validation.py -v -s
```

## What Gets Tested

| Test | What It Checks | Expected Speedup |
|------|----------------|------------------|
| **Spline Interpolation** | `interpolate_complex_array()` accuracy & speed | 1.3-1.8x |
| **Process Transfer** | Full `process_transfer()` with TDI response | 1.5-2.5x |
| **Distance Scaling** | Vectorized vs loop implementation | 3-10x |

## Test Structure

### 1. Old Implementation (Baseline)
Each test includes a faithful copy of the **pre-optimization** code:
- `interpolate_complex_array_OLD()` - Uses `np.copy()` 
- `process_transfer_OLD()` - Separate spline instantiations
- `scale_distance_OLD()` - Python for-loop

### 2. New Implementation (Optimized)
Tests import from the actual optimized code:
- `interpolate_complex_array()` - No copies, efficient views
- `process_transfer()` - Streamlined splines
- Direct vectorized operations in transform class

### 3. Validation Checks
For each optimization:
1. **Numerical Equivalence**
   - Runs both old and new implementations
   - Compares outputs element-wise
   - Checks absolute and relative differences
   - Passes if: `max_diff < 1e-10` (machine precision)

2. **Performance Improvement**
   - Times multiple iterations of both methods
   - Calculates speedup factor
   - Passes if: new method is faster than old

## Example Output

```
================================================================================
LISA OPTIMIZATION VALIDATION TESTS
================================================================================

--------------------------------------------------------------------------------
TEST 1: Spline Interpolation
--------------------------------------------------------------------------------

Spline Interpolation:
  Max absolute difference: 1.23e-12
  Max relative difference: 2.45e-13
✓ Numerical equivalence verified

Spline Interpolation Performance:
  Old method: 2.450 ms per call
  New method: 1.523 ms per call
  Speedup: 1.61x
✓ Performance improvement verified

--------------------------------------------------------------------------------
TEST 2: Process Transfer Function
--------------------------------------------------------------------------------

Process Transfer chan1:
  Max absolute difference: 3.45e-12
  Max relative difference: 1.23e-09

Process Transfer chan2:
  Max absolute difference: 2.89e-12
  Max relative difference: 9.87e-10

Process Transfer chan3:
  Max absolute difference: 3.12e-12
  Max relative difference: 1.05e-09
✓ All channels numerically equivalent

Process Transfer Performance:
  Old method: 125.3 ms per call
  New method: 78.6 ms per call
  Speedup: 1.59x
✓ Performance improvement verified

--------------------------------------------------------------------------------
TEST 3: Distance Scaling
--------------------------------------------------------------------------------

Distance Scaling:
  Max absolute difference: 0.00e+00
  Max relative difference: 0.00e+00
✓ Perfect numerical equivalence

Distance Scaling Performance:
  Old method: 1.234 ms per call
  New method: 0.234 ms per call
  Speedup: 5.27x
✓ Significant performance improvement

================================================================================
SUMMARY
================================================================================
✓ All numerical equivalence tests passed
✓ All performance tests passed
✓ Optimizations are verified correct and faster
```

## Pass/Fail Criteria

### Numerical Accuracy Thresholds
- **Absolute difference**: < 1e-10
  - At typical GW amplitudes (~1e-22), this is ~1e-32 (negligible)
- **Relative difference**: < 1e-8
  - 0.00001% difference (well below measurement precision)

### Performance Thresholds
- **Spline interpolation**: > 1.0x speedup
- **Process transfer**: > 1.0x speedup  
- **Distance scaling**: > 2.0x speedup (vectorization expected to be much faster)

## Why This Test Is Important

### Scientific Correctness
- Ensures no bugs introduced during optimization
- Verifies floating-point arithmetic remains stable
- Confirms complex number handling is correct

### Performance Verification
- Proves optimizations actually improve speed
- Quantifies speedup for documentation
- Helps identify regressions in future changes

### Regression Testing
- Can be run after any code changes
- Detects breaking changes immediately
- Safe refactoring confidence

## Interpreting Results

### ✅ All Tests Pass
- Optimizations are working correctly
- Safe to use in production/training
- Expected speedups are being achieved

### ⚠️ Numerical Test Fails
**Symptoms:** "Results differ by X" where X > 1e-10

**Possible Causes:**
- Bug in optimization implementation
- Changed external library (lisabeta)
- Floating-point precision issue

**Actions:**
1. Check the exact difference value
2. Review optimization code changes
3. Verify lisabeta version hasn't changed
4. Consider if tolerance needs adjustment (scientific justification required)

### ⚠️ Performance Test Fails
**Symptoms:** Speedup < expected threshold

**Possible Causes:**
- System under heavy load
- Changed CPU governor (power saving)
- Insufficient iterations for stable timing
- Actual performance regression

**Actions:**
1. Re-run test multiple times
2. Check system load (`top` or `htop`)
3. Increase iteration count in test
4. Profile to find actual bottleneck

## Integration with CI/CD

To add to continuous integration:

```yaml
# .github/workflows/test.yml
- name: Validate LISA Optimizations
  run: |
    python tests/gw/transforms/test_optimization_validation.py
```

Or with pytest:
```yaml
- name: Run Optimization Tests
  run: |
    pytest tests/gw/transforms/test_optimization_validation.py -v
```

## Updating the Tests

When modifying optimizations:

1. **Update the OLD implementation** if API changed
   ```python
   def process_transfer_OLD(...):
       # Keep this as the pre-optimization baseline
       # Even if the API changes
   ```

2. **Adjust thresholds** only if scientifically justified
   ```python
   # If new precision analysis shows threshold should change
   assert max_abs_diff < NEW_THRESHOLD
   ```

3. **Document changes** in test docstrings
   ```python
   def test_numerical_equivalence(self):
       """
       Updated 2026-02-15: Changed threshold due to...
       """
   ```

4. **Run full test suite** before committing
   ```bash
   pytest tests/gw/transforms/ -v
   ```

## Related Files

- **Main optimizations**: [dingo/gw/transforms/detector_transforms.py](../../../dingo/gw/transforms/detector_transforms.py)
- **This test**: [test_optimization_validation.py](test_optimization_validation.py)
- **Benchmarks**: [misc_scripts/benchmark_optimizations.py](../../../misc_scripts/benchmark_optimizations.py)
- **Profiler**: [misc_scripts/profile_lisa_transforms.py](../../../misc_scripts/profile_lisa_transforms.py)
- **Guide**: [docs/LISA_OPTIMIZATION_GUIDE.md](../../../docs/LISA_OPTIMIZATION_GUIDE.md)

## FAQ

**Q: Why keep the old implementations in the test?**
A: To have a ground truth reference that we know was correct. This is safer than comparing to hardcoded values.

**Q: Can I remove the old implementations after testing?**
A: No! Keep them for regression testing. Future changes need the baseline.

**Q: What if I get slightly different numbers each run?**
A: This is normal due to floating-point non-determinism. The thresholds account for this.

**Q: Should I run this before every training run?**
A: Not necessary. Run it:
- After pulling new code
- After modifying optimizations
- Before important production runs
- As part of CI/CD

**Q: How long does the test take?**
A: ~30-60 seconds depending on system. Process transfer test is the slowest.

## Support

If tests fail unexpectedly:
1. Check the test output for specific error messages
2. Review recent code changes
3. Verify system configuration
4. Check [LISA_OPTIMIZATION_GUIDE.md](../../../docs/LISA_OPTIMIZATION_GUIDE.md) troubleshooting section
5. Open an issue with full test output
