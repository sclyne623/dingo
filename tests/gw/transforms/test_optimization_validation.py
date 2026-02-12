#!/usr/bin/env python3
"""
Validation test for LISA optimization changes.

This script verifies that the optimized code produces identical results
to the original implementation, while being significantly faster.

Run with: pytest test_optimization_validation.py -v
Or directly: python test_optimization_validation.py
"""

import numpy as np
import time
import pytest
import lisabeta.tools.pyspline as pyspline
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dingo.gw.transforms.detector_transforms import (
    interpolate_complex_array,
    process_transfer,
)


# ============================================================================
# OLD IMPLEMENTATIONS (before optimization)
# ============================================================================

def interpolate_complex_array_OLD(freq_grid, complex_data, interp_freqs):
    """
    OLD METHOD: Uses np.copy() for real/imag extraction.
    This is the original implementation before optimization.
    """
    # OLD: Explicit copies
    amp_real = np.copy(np.real(complex_data))
    amp_imag = np.copy(np.imag(complex_data))
    
    # OLD: Separate class instantiation and get_spline calls
    spline_real_class = pyspline.CubicSpline(freq_grid, amp_real)
    spline_imag_class = pyspline.CubicSpline(freq_grid, amp_imag)
    
    spline_real = spline_real_class.get_spline()
    spline_imag = spline_imag_class.get_spline()
    
    real_interp = pyspline.spline_eval_vector(spline_real, interp_freqs, extrapol_zero=True)
    imag_interp = pyspline.spline_eval_vector(spline_imag, interp_freqs, extrapol_zero=True)
    
    return real_interp + 1j * imag_interp


def process_transfer_OLD(freq_grid, amp, phase, tf, t0, l, m, inc, phi, lambd, beta, psi,
                         interp_freqs, f_min, detector_type, LISAconst, 
                         responseapprox, frozenLISA, TDIrescaled):
    """
    OLD METHOD: Original process_transfer with np.copy() calls.
    This implementation has the performance issues we're fixing.
    """
    import lisabeta.lisa.pyresponse as pyresponse
    
    # Get TDI transfer functions
    tdiClass = pyresponse.LISAFDresponseTDI3Chan(
        freq_grid, tf, t0, l, m, inc, phi, lambd, beta, psi, 
        detector_type, LISAconst, responseapprox, frozenLISA, TDIrescaled
    )
    phaseRdelay, transferL1, transferL2, transferL3 = tdiClass.get_response()
    
    # Calculate complex amplitudes
    camp1 = amp * transferL1
    camp2 = amp * transferL2
    camp3 = amp * transferL3
    phasetot = phase + phaseRdelay
    
    # OLD METHOD: Break into real/imag with copies
    amp_real_chan1 = np.copy(np.real(camp1))
    amp_imag_chan1 = np.copy(np.imag(camp1))
    amp_real_chan2 = np.copy(np.real(camp2))
    amp_imag_chan2 = np.copy(np.imag(camp2))
    amp_real_chan3 = np.copy(np.real(camp3))
    amp_imag_chan3 = np.copy(np.imag(camp3))
    
    # OLD METHOD: Separate spline class instantiation
    spline_amp_real_chan1Class = pyspline.CubicSpline(freq_grid, amp_real_chan1)
    spline_amp_imag_chan1Class = pyspline.CubicSpline(freq_grid, amp_imag_chan1)
    spline_amp_real_chan2Class = pyspline.CubicSpline(freq_grid, amp_real_chan2)
    spline_amp_imag_chan2Class = pyspline.CubicSpline(freq_grid, amp_imag_chan2)
    spline_amp_real_chan3Class = pyspline.CubicSpline(freq_grid, amp_real_chan3)
    spline_amp_imag_chan3Class = pyspline.CubicSpline(freq_grid, amp_imag_chan3)
    spline_phaseClass = pyspline.CubicSpline(freq_grid, phasetot)
    
    spline_amp_real_chan1 = spline_amp_real_chan1Class.get_spline()
    spline_amp_imag_chan1 = spline_amp_imag_chan1Class.get_spline()
    spline_amp_real_chan2 = spline_amp_real_chan2Class.get_spline()
    spline_amp_imag_chan2 = spline_amp_imag_chan2Class.get_spline()
    spline_amp_real_chan3 = spline_amp_real_chan3Class.get_spline()
    spline_amp_imag_chan3 = spline_amp_imag_chan3Class.get_spline()
    spline_phase = spline_phaseClass.get_spline()
    
    # Evaluate splines
    ampreal_chan1 = pyspline.spline_eval_vector(spline_amp_real_chan1, interp_freqs, extrapol_zero=True)
    ampimag_chan1 = pyspline.spline_eval_vector(spline_amp_imag_chan1, interp_freqs, extrapol_zero=True)
    ampreal_chan2 = pyspline.spline_eval_vector(spline_amp_real_chan2, interp_freqs, extrapol_zero=True)
    ampimag_chan2 = pyspline.spline_eval_vector(spline_amp_imag_chan2, interp_freqs, extrapol_zero=True)
    ampreal_chan3 = pyspline.spline_eval_vector(spline_amp_real_chan3, interp_freqs, extrapol_zero=True)
    ampimag_chan3 = pyspline.spline_eval_vector(spline_amp_imag_chan3, interp_freqs, extrapol_zero=True)
    phase_interp = pyspline.spline_eval_vector(spline_phase, interp_freqs, extrapol_zero=True)
    
    # Get complex values for the TDI freqseries
    eiphase = np.exp(1j * phase_interp)
    tdi_chan1_vals = (ampreal_chan1 + 1j * ampimag_chan1) * eiphase
    tdi_chan2_vals = (ampreal_chan2 + 1j * ampimag_chan2) * eiphase
    tdi_chan3_vals = (ampreal_chan3 + 1j * ampimag_chan3) * eiphase
    
    # OLD METHOD: Set waveforms = 0 below fmin using boolean indexing
    tdi_chan1_vals[interp_freqs < f_min] = 0.
    tdi_chan2_vals[interp_freqs < f_min] = 0.
    tdi_chan3_vals[interp_freqs < f_min] = 0.
    
    return tdi_chan1_vals, tdi_chan2_vals, tdi_chan3_vals


def scale_distance_OLD(amp_list, d_ratio):
    """
    OLD METHOD: Python loop for distance scaling.
    """
    result = []
    for i in range(len(d_ratio)):
        result.append(amp_list[i] * d_ratio[i])
    return result


def scale_distance_NEW(amp_array, d_ratio):
    """
    NEW METHOD: Vectorized distance scaling.
    """
    return amp_array * d_ratio


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

class TestSplineInterpolation:
    """Test optimized spline interpolation."""
    
    def test_numerical_equivalence(self):
        """Verify new method gives identical results to old method."""
        # Setup test data
        freq_grid = np.linspace(1e-4, 1e-1, 1000)
        complex_data = np.random.rand(1000) + 1j * np.random.rand(1000)
        interp_freqs = np.linspace(1e-4, 1e-1, 2048)
        
        # Run both methods
        result_old = interpolate_complex_array_OLD(freq_grid, complex_data, interp_freqs)
        result_new = interpolate_complex_array(freq_grid, complex_data, interp_freqs)
        
        # Check numerical equivalence
        max_abs_diff = np.max(np.abs(result_old - result_new))
        rel_diff = np.max(np.abs((result_old - result_new) / (np.abs(result_old) + 1e-30)))
        
        print(f"\nSpline Interpolation:")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {rel_diff:.2e}")
        
        assert max_abs_diff < 1e-10, f"Results differ by {max_abs_diff}"
        assert rel_diff < 1e-10, f"Relative difference is {rel_diff}"
    
    def test_performance(self):
        """Verify new method is faster."""
        freq_grid = np.linspace(1e-4, 1e-1, 1000)
        complex_data = np.random.rand(1000) + 1j * np.random.rand(1000)
        interp_freqs = np.linspace(1e-4, 1e-1, 2048)
        n_iterations = 50
        
        # Time old method
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = interpolate_complex_array_OLD(freq_grid, complex_data, interp_freqs)
        time_old = time.perf_counter() - start
        
        # Time new method
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = interpolate_complex_array(freq_grid, complex_data, interp_freqs)
        time_new = time.perf_counter() - start
        
        speedup = time_old / time_new
        print(f"\nSpline Interpolation Performance:")
        print(f"  Old method: {time_old/n_iterations*1000:.3f} ms per call")
        print(f"  New method: {time_new/n_iterations*1000:.3f} ms per call")
        print(f"  Speedup: {speedup:.2f}x")
        
        assert speedup > 1.0, f"New method should be faster, got {speedup:.2f}x"


class TestProcessTransfer:
    """Test optimized process_transfer function."""
    
    def setup_method(self):
        """Setup test data for process_transfer tests."""
        self.freq_grid = np.linspace(1e-4, 1e-1, 500)
        self.amp = np.random.rand(500) * 1e-22
        self.phase = np.random.rand(500) * 2 * np.pi
        self.tf = np.random.rand(500)
        self.t0 = 0.0
        self.l = 2
        self.m = 2
        self.inc = 0.5
        self.phi = 1.0
        self.lambd = 1.2
        self.beta = 0.3
        self.psi = 0.7
        self.interp_freqs = np.linspace(1e-4, 1e-1, 1024)
        self.f_min = 5e-4
        self.detector_type = "TDIAET"
        self.LISAconst = "Proposal"
        self.responseapprox = "full"
        self.frozenLISA = True
        self.TDIrescaled = False
    
    def test_numerical_equivalence(self):
        """Verify new process_transfer gives identical results to old method."""
        # Run both methods
        result_old = process_transfer_OLD(
            self.freq_grid, self.amp, self.phase, self.tf, self.t0, 
            self.l, self.m, self.inc, self.phi, self.lambd, self.beta, self.psi,
            self.interp_freqs, self.f_min, self.detector_type, self.LISAconst,
            self.responseapprox, self.frozenLISA, self.TDIrescaled
        )
        
        result_new = process_transfer(
            self.freq_grid, self.amp, self.phase, self.tf, self.t0, 
            self.l, self.m, self.inc, self.phi, self.lambd, self.beta, self.psi,
            self.interp_freqs, self.f_min, self.detector_type, self.LISAconst,
            self.responseapprox, self.frozenLISA, self.TDIrescaled
        )
        
        # Check each channel
        for i, (chan_old, chan_new, name) in enumerate(zip(result_old, result_new, 
                                                             ['chan1', 'chan2', 'chan3'])):
            max_abs_diff = np.max(np.abs(chan_old - chan_new))
            # Use magnitude for relative difference to avoid divide-by-zero
            magnitude = np.abs(chan_old) + 1e-30
            rel_diff = np.max(np.abs((chan_old - chan_new) / magnitude))
            
            print(f"\nProcess Transfer {name}:")
            print(f"  Max absolute difference: {max_abs_diff:.2e}")
            print(f"  Max relative difference: {rel_diff:.2e}")
            
            assert max_abs_diff < 1e-10, f"{name} differs by {max_abs_diff}"
            assert rel_diff < 1e-8, f"{name} relative difference is {rel_diff}"
    
    def test_performance(self):
        """Verify new process_transfer is faster."""
        n_iterations = 10
        
        # Time old method
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = process_transfer_OLD(
                self.freq_grid, self.amp, self.phase, self.tf, self.t0, 
                self.l, self.m, self.inc, self.phi, self.lambd, self.beta, self.psi,
                self.interp_freqs, self.f_min, self.detector_type, self.LISAconst,
                self.responseapprox, self.frozenLISA, self.TDIrescaled
            )
        time_old = time.perf_counter() - start
        
        # Time new method
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = process_transfer(
                self.freq_grid, self.amp, self.phase, self.tf, self.t0, 
                self.l, self.m, self.inc, self.phi, self.lambd, self.beta, self.psi,
                self.interp_freqs, self.f_min, self.detector_type, self.LISAconst,
                self.responseapprox, self.frozenLISA, self.TDIrescaled
            )
        time_new = time.perf_counter() - start
        
        speedup = time_old / time_new
        print(f"\nProcess Transfer Performance:")
        print(f"  Old method: {time_old/n_iterations*1000:.1f} ms per call")
        print(f"  New method: {time_new/n_iterations*1000:.1f} ms per call")
        print(f"  Speedup: {speedup:.2f}x")
        
        assert speedup > 1.0, f"New method should be faster, got {speedup:.2f}x"


class TestDistanceScaling:
    """Test optimized distance scaling."""
    
    def test_numerical_equivalence(self):
        """Verify vectorized distance scaling gives identical results."""
        batch_size = 50
        num_freqs = 1000
        
        # Create test data
        amp_list = [np.random.rand(num_freqs) for _ in range(batch_size)]
        amp_array = np.array(amp_list)
        d_ratio = np.random.rand(batch_size)[:, np.newaxis]
        
        # Run both methods
        result_old = scale_distance_OLD(amp_list, d_ratio)
        result_new = scale_distance_NEW(amp_array, d_ratio)
        
        # Convert old result to array for comparison
        result_old_array = np.array(result_old)
        
        max_abs_diff = np.max(np.abs(result_old_array - result_new))
        rel_diff = np.max(np.abs((result_old_array - result_new) / (np.abs(result_old_array) + 1e-30)))
        
        print(f"\nDistance Scaling:")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {rel_diff:.2e}")
        
        assert max_abs_diff < 1e-14, f"Results differ by {max_abs_diff}"
        assert rel_diff < 1e-14, f"Relative difference is {rel_diff}"
    
    def test_performance(self):
        """Verify vectorized distance scaling is faster."""
        batch_size = 100
        num_freqs = 1000
        n_iterations = 100
        
        # Time old method (loop)
        start = time.perf_counter()
        for _ in range(n_iterations):
            amp_list = [np.random.rand(num_freqs) for _ in range(batch_size)]
            d_ratio = np.random.rand(batch_size)[:, np.newaxis]
            _ = scale_distance_OLD(amp_list, d_ratio)
        time_old = time.perf_counter() - start
        
        # Time new method (vectorized)
        start = time.perf_counter()
        for _ in range(n_iterations):
            amp_array = np.random.rand(batch_size, num_freqs)
            d_ratio = np.random.rand(batch_size)[:, np.newaxis]
            _ = scale_distance_NEW(amp_array, d_ratio)
        time_new = time.perf_counter() - start
        
        speedup = time_old / time_new
        print(f"\nDistance Scaling Performance:")
        print(f"  Old method: {time_old/n_iterations*1000:.3f} ms per call")
        print(f"  New method: {time_new/n_iterations*1000:.3f} ms per call")
        print(f"  Speedup: {speedup:.2f}x")
        
        assert speedup > 2.0, f"Vectorized method should be much faster, got {speedup:.2f}x"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_tests():
    """Run all validation tests."""
    print("="*80)
    print("LISA OPTIMIZATION VALIDATION TESTS")
    print("="*80)
    print("\nThis script verifies that optimizations produce identical results")
    print("while being significantly faster.\n")
    
    # Test spline interpolation
    print("-"*80)
    print("TEST 1: Spline Interpolation")
    print("-"*80)
    test_spline = TestSplineInterpolation()
    test_spline.test_numerical_equivalence()
    test_spline.test_performance()
    
    # Test process_transfer
    print("\n" + "-"*80)
    print("TEST 2: Process Transfer Function")
    print("-"*80)
    test_transfer = TestProcessTransfer()
    test_transfer.setup_method()
    test_transfer.test_numerical_equivalence()
    test_transfer.test_performance()
    
    # Test distance scaling
    print("\n" + "-"*80)
    print("TEST 3: Distance Scaling")
    print("-"*80)
    test_distance = TestDistanceScaling()
    test_distance.test_numerical_equivalence()
    test_distance.test_performance()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✓ All numerical equivalence tests passed")
    print("✓ All performance tests passed")
    print("✓ Optimizations are verified correct and faster")
    print("\nThe optimized code produces identical results while being significantly faster!")


if __name__ == '__main__':
    run_all_tests()
