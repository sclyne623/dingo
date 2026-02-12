# Optimization Validation Test

This test script verifies that the LISA performance optimizations produce numerically identical results to the original implementation while being significantly faster.

## What It Tests

1. **Spline Interpolation** (`interpolate_complex_array`)
   - Verifies new method without `np.copy()` gives identical results
   - Measures speedup (expected: 1.3-1.8x)

2. **Process Transfer** (`process_transfer`)
   - Compares full transfer function calculation
   - Tests all 3 TDI channels (chan1, chan2, chan3)
   - Measures speedup (expected: 1.5-2.5x)

3. **Distance Scaling**
   - Compares vectorized vs loop-based implementation
   - Measures speedup (expected: 3-10x)

## Running the Tests

### As a standalone script:
```bash
cd /Users/samclyne/Downloads/dingo-LISA
python tests/gw/transforms/test_optimization_validation.py
```

### With pytest:
```bash
cd /Users/samclyne/Downloads/dingo-LISA
pytest tests/gw/transforms/test_optimization_validation.py -v
```

### Run individual tests:
```bash
pytest tests/gw/transforms/test_optimization_validation.py::TestSplineInterpolation -v
pytest tests/gw/transforms/test_optimization_validation.py::TestProcessTransfer -v
pytest tests/gw/transforms/test_optimization_validation.py::TestDistanceScaling -v
```

## Expected Output

```
================================================================================
LISA OPTIMIZATION VALIDATION TESTS
================================================================================

This script verifies that optimizations produce identical results
while being significantly faster.

--------------------------------------------------------------------------------
TEST 1: Spline Interpolation
--------------------------------------------------------------------------------

Spline Interpolation:
  Max absolute difference: 1.23e-12
  Max relative difference: 2.45e-13

Spline Interpolation Performance:
  Old method: 2.450 ms per call
  New method: 1.523 ms per call
  Speedup: 1.61x

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

Process Transfer Performance:
  Old method: 125.3 ms per call
  New method: 78.6 ms per call
  Speedup: 1.59x

--------------------------------------------------------------------------------
TEST 3: Distance Scaling
--------------------------------------------------------------------------------

Distance Scaling:
  Max absolute difference: 0.00e+00
  Max relative difference: 0.00e+00

Distance Scaling Performance:
  Old method: 1.234 ms per call
  New method: 0.234 ms per call
  Speedup: 5.27x

================================================================================
SUMMARY
================================================================================
✓ All numerical equivalence tests passed
✓ All performance tests passed
✓ Optimizations are verified correct and faster

The optimized code produces identical results while being significantly faster!
```

## Validation Criteria

### Numerical Accuracy
- **Absolute difference** < 1e-10 (sub-femtometer scale)
- **Relative difference** < 1e-8 (0.00001%)

These thresholds ensure the optimizations don't introduce any meaningful numerical errors.

### Performance
- **Spline interpolation**: Must be faster (>1.0x speedup)
- **Process transfer**: Must be faster (>1.0x speedup)
- **Distance scaling**: Must be significantly faster (>2.0x speedup)

## Troubleshooting

### Test fails with "Results differ by X"
- This indicates a numerical accuracy issue
- Check if lisabeta library version changed
- Verify the optimization implementation

### Test fails with "should be faster"
- This indicates performance regression
- Check if system is under load
- Try running multiple times for consistent results
- May need to adjust iteration counts

### Import errors
- Ensure you're in the correct directory
- Check that dingo is in your Python path
- Install required dependencies: `lisabeta`, `numpy`

## Files Tested

- `/dingo/gw/transforms/detector_transforms.py`
  - `interpolate_complex_array()` - Helper function
  - `process_transfer()` - Main transformation function
  - Distance scaling in `ProjectOntoSpaceDetectors.__call__()`

## Maintenance

When updating optimizations:
1. Run this test to ensure numerical equivalence
2. Update the "old" implementation if the API changes
3. Adjust tolerance thresholds only if scientifically justified
4. Document any changes to expected speedups

## See Also

- [LISA_OPTIMIZATION_GUIDE.md](../../../docs/LISA_OPTIMIZATION_GUIDE.md) - Full optimization guide
- [LISA_OPTIMIZATION_SUMMARY.md](../../../docs/LISA_OPTIMIZATION_SUMMARY.md) - Quick reference
- [benchmark_optimizations.py](../../../misc_scripts/benchmark_optimizations.py) - Micro-benchmarks
- [profile_lisa_transforms.py](../../../misc_scripts/profile_lisa_transforms.py) - End-to-end profiling
