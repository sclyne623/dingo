#!/usr/bin/env python3
"""
Quick benchmark to verify LISA transformation speedups.

This script provides a simple comparison of key operations
before and after optimization.
"""

import numpy as np
import time
import lisabeta.tools.pyspline as pyspline


def benchmark_spline_operations():
    """Compare old vs new spline interpolation approach."""
    print("\n" + "="*70)
    print("BENCHMARK: Spline Interpolation")
    print("="*70)
    
    # Setup test data
    freq_grid = np.linspace(1e-4, 1e-1, 1000)
    complex_data = np.random.rand(1000) + 1j * np.random.rand(1000)
    interp_freqs = np.linspace(1e-4, 1e-1, 2048)
    n_iterations = 100
    
    # OLD METHOD: Separate real/imag with copies
    print("\nOLD METHOD (with np.copy):")
    times_old = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        
        # Simulate old approach with copies
        amp_real = np.copy(np.real(complex_data))
        amp_imag = np.copy(np.imag(complex_data))
        
        spline_real_class = pyspline.CubicSpline(freq_grid, amp_real)
        spline_imag_class = pyspline.CubicSpline(freq_grid, amp_imag)
        
        spline_real = spline_real_class.get_spline()
        spline_imag = spline_imag_class.get_spline()
        
        real_interp = pyspline.spline_eval_vector(spline_real, interp_freqs, extrapol_zero=True)
        imag_interp = pyspline.spline_eval_vector(spline_imag, interp_freqs, extrapol_zero=True)
        
        result_old = real_interp + 1j * imag_interp
        
        end = time.perf_counter()
        times_old.append(end - start)
    
    mean_old = np.mean(times_old) * 1000  # Convert to ms
    print(f"  Mean time: {mean_old:.3f} ms")
    print(f"  Std dev:   {np.std(times_old)*1000:.3f} ms")
    
    # NEW METHOD: Direct access without copies
    print("\nNEW METHOD (without np.copy):")
    times_new = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        
        # New optimized approach
        real_part = complex_data.real  # View, not copy
        imag_part = complex_data.imag  # View, not copy
        
        spline_real = pyspline.CubicSpline(freq_grid, real_part).get_spline()
        spline_imag = pyspline.CubicSpline(freq_grid, imag_part).get_spline()
        
        real_interp = pyspline.spline_eval_vector(spline_real, interp_freqs, extrapol_zero=True)
        imag_interp = pyspline.spline_eval_vector(spline_imag, interp_freqs, extrapol_zero=True)
        
        result_new = real_interp + 1j * imag_interp
        
        end = time.perf_counter()
        times_new.append(end - start)
    
    mean_new = np.mean(times_new) * 1000
    print(f"  Mean time: {mean_new:.3f} ms")
    print(f"  Std dev:   {np.std(times_new)*1000:.3f} ms")
    
    # Verify numerical equivalence
    print("\nNUMERICAL VERIFICATION:")
    max_diff = np.max(np.abs(result_old - result_new))
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Numerically equivalent: {max_diff < 1e-10}")
    
    # Summary
    speedup = mean_old / mean_new
    time_saved = mean_old - mean_new
    print("\nSUMMARY:")
    print(f"  Speedup factor: {speedup:.2f}x")
    print(f"  Time saved per call: {time_saved:.3f} ms")
    print(f"  Time saved per 1000 calls: {time_saved:.1f} seconds")
    
    return speedup


def benchmark_distance_scaling():
    """Compare loop vs vectorized distance scaling."""
    print("\n" + "="*70)
    print("BENCHMARK: Distance Scaling")
    print("="*70)
    
    # Setup test data
    batch_size = 100
    num_freqs = 1000
    amp_array = [np.random.rand(num_freqs) for _ in range(batch_size)]
    d_ratio = np.random.rand(batch_size)[:, np.newaxis]
    n_iterations = 1000
    
    # OLD METHOD: Loop
    print("\nOLD METHOD (for loop):")
    times_old = []
    for _ in range(n_iterations):
        amp_copy = [a.copy() for a in amp_array]
        start = time.perf_counter()
        
        for i in range(len(d_ratio)):
            amp_copy[i] = amp_copy[i] * d_ratio[i]
        
        end = time.perf_counter()
        times_old.append(end - start)
    
    mean_old = np.mean(times_old) * 1000
    print(f"  Mean time: {mean_old:.3f} ms")
    
    # NEW METHOD: Vectorized
    print("\nNEW METHOD (vectorized):")
    times_new = []
    for _ in range(n_iterations):
        amp_array_np = np.array(amp_array)
        start = time.perf_counter()
        
        amp_scaled = amp_array_np * d_ratio
        
        end = time.perf_counter()
        times_new.append(end - start)
    
    mean_new = np.mean(times_new) * 1000
    print(f"  Mean time: {mean_new:.3f} ms")
    
    # Summary
    speedup = mean_old / mean_new
    print("\nSUMMARY:")
    print(f"  Speedup factor: {speedup:.2f}x")
    print(f"  Time saved per call: {mean_old - mean_new:.3f} ms")
    
    return speedup


def benchmark_frequency_masking():
    """Compare assignment vs multiplication for frequency masking."""
    print("\n" + "="*70)
    print("BENCHMARK: Frequency Masking")
    print("="*70)
    
    # Setup
    interp_freqs = np.linspace(1e-4, 1e-1, 2048)
    data = np.random.rand(2048) + 1j * np.random.rand(2048)
    f_min = 5e-4
    n_iterations = 10000
    
    # OLD METHOD: Boolean indexing with assignment
    print("\nOLD METHOD (boolean indexing):")
    times_old = []
    for _ in range(n_iterations):
        data_copy = data.copy()
        start = time.perf_counter()
        
        data_copy[interp_freqs < f_min] = 0.
        
        end = time.perf_counter()
        times_old.append(end - start)
    
    mean_old = np.mean(times_old) * 1e6  # microseconds
    print(f"  Mean time: {mean_old:.2f} μs")
    
    # NEW METHOD: Multiplication with mask
    print("\nNEW METHOD (multiplication):")
    times_new = []
    for _ in range(n_iterations):
        data_copy = data.copy()
        start = time.perf_counter()
        
        freq_mask = interp_freqs >= f_min
        data_copy *= freq_mask
        
        end = time.perf_counter()
        times_new.append(end - start)
    
    mean_new = np.mean(times_new) * 1e6
    print(f"  Mean time: {mean_new:.2f} μs")
    
    # Summary
    speedup = mean_old / mean_new
    print("\nSUMMARY:")
    print(f"  Speedup factor: {speedup:.2f}x")
    
    return speedup


def main():
    print("\n" + "#"*70)
    print("# LISA WAVEFORM OPTIMIZATION BENCHMARK")
    print("#"*70)
    print("\nThis benchmark tests individual optimizations in isolation.")
    print("For full pipeline benchmarking, use profile_lisa_transforms.py")
    
    speedups = []
    
    # Run benchmarks
    speedups.append(benchmark_spline_operations())
    speedups.append(benchmark_distance_scaling())
    speedups.append(benchmark_frequency_masking())
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Spline interpolation speedup:  {speedups[0]:.2f}x")
    print(f"Distance scaling speedup:      {speedups[1]:.2f}x")
    print(f"Frequency masking speedup:     {speedups[2]:.2f}x")
    print(f"\nGeometric mean speedup:        {np.prod(speedups)**(1/len(speedups)):.2f}x")
    
    print("\nNOTE: These are micro-benchmarks of individual operations.")
    print("Overall speedup depends on:")
    print("  - Proportion of time spent in each operation")
    print("  - Batch size (affects multiprocessing benefit)")
    print("  - Number of CPU cores")
    print("  - System architecture")
    
    print("\nFor realistic end-to-end benchmarking:")
    print("  python profile_lisa_transforms.py --batch-size 8")


if __name__ == '__main__':
    main()
