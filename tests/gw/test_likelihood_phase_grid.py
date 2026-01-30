"""
Test script to compare the optimized log_likelihood_phase_grid implementation
against the reference (original) implementation.
"""

import numpy as np
import pytest


def test_log_likelihood_phase_grid_comparison():
    """
    Compare optimized and reference implementations of log_likelihood_phase_grid.
    """
    from dingo.gw.likelihood import build_stationary_gaussian_likelihood
    from dingo.gw.waveform_generator import WaveformGenerator
    from dingo.gw.domains import build_domain
    
    # Set up a simple test case
    # You'll need to adjust these parameters based on your actual setup
    domain_dict = {
        "type": "FrequencyDomain",
        "f_min": 20.0,
        "f_max": 1024.0,
        "delta_f": 0.125,
    }
    
    wfg_kwargs = {
        "approximant": "IMRPhenomXPHM",
        "f_ref": 20.0,
    }
    
    # Define test parameters
    theta = {
        "mass_1": 35.0,
        "mass_2": 30.0,
        "luminosity_distance": 440.0,
        "a_1": 0.0,
        "a_2": 0.0,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "theta_jn": 0.0,
        "phase": 0.0,
        "geocent_time": 0.0,
        "ra": 0.0,
        "dec": 0.0,
        "psi": 0.0,
    }
    
    # Define phase grid to test
    phases = np.linspace(0, 2 * np.pi, 50)
    
    print("Note: This test requires a proper likelihood object with event data.")
    print("You'll need to modify this test with actual event data to run it.")
    print("\nExample usage with a likelihood object:")
    print("=" * 70)
    print("""
# Assuming you have a likelihood object:
# likelihood = build_stationary_gaussian_likelihood(...)

import numpy as np
import time

# Define phase grid
phases = np.linspace(0, 2 * np.pi, 100)

# Time the reference implementation
start = time.time()
log_like_ref = likelihood.log_likelihood_phase_grid_reference(theta, phases)
time_ref = time.time() - start

# Time the optimized implementation
start = time.time()
log_like_opt = likelihood.log_likelihood_phase_grid(theta, phases)
time_opt = time.time() - start

# Compare results
max_diff = np.max(np.abs(log_like_ref - log_like_opt))
rel_diff = np.max(np.abs((log_like_ref - log_like_opt) / log_like_ref))

print(f"Reference time: {time_ref:.4f} s")
print(f"Optimized time: {time_opt:.4f} s")
print(f"Speedup: {time_ref/time_opt:.2f}x")
print(f"Maximum absolute difference: {max_diff:.2e}")
print(f"Maximum relative difference: {rel_diff:.2e}")
print(f"Results match: {np.allclose(log_like_ref, log_like_opt, rtol=1e-10)}")
    """)


def compare_implementations(likelihood, theta, phases):
    """
    Utility function to compare both implementations.
    
    Parameters
    ----------
    likelihood : StationaryGaussianGWLikelihood
        Likelihood object to test
    theta : dict
        Parameter dictionary
    phases : array-like
        Phase values to evaluate
        
    Returns
    -------
    dict
        Dictionary containing comparison results
    """
    import time
    
    # Time and compute reference implementation
    start = time.time()
    log_like_ref = likelihood.log_likelihood_phase_grid_reference(theta, phases)
    time_ref = time.time() - start
    
    # Time and compute optimized implementation
    start = time.time()
    log_like_opt = likelihood.log_likelihood_phase_grid(theta, phases)
    time_opt = time.time() - start
    
    # Compute differences
    abs_diff = np.abs(log_like_ref - log_like_opt)
    max_abs_diff = np.max(abs_diff)
    
    # Avoid division by zero in relative difference
    rel_diff = np.abs((log_like_ref - log_like_opt) / (log_like_ref + 1e-100))
    max_rel_diff = np.max(rel_diff)
    
    results = {
        "log_like_ref": log_like_ref,
        "log_like_opt": log_like_opt,
        "time_ref": time_ref,
        "time_opt": time_opt,
        "speedup": time_ref / time_opt if time_opt > 0 else np.inf,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "allclose": np.allclose(log_like_ref, log_like_opt, rtol=1e-10, atol=1e-10),
    }
    
    return results


def print_comparison(results):
    """
    Pretty print comparison results.
    
    Parameters
    ----------
    results : dict
        Results from compare_implementations
    """
    print("\n" + "=" * 70)
    print("Log Likelihood Phase Grid Comparison")
    print("=" * 70)
    print(f"Reference implementation time: {results['time_ref']:.4f} s")
    print(f"Optimized implementation time: {results['time_opt']:.4f} s")
    print(f"Speedup factor: {results['speedup']:.2f}x")
    print("-" * 70)
    print(f"Maximum absolute difference: {results['max_abs_diff']:.2e}")
    print(f"Maximum relative difference: {results['max_rel_diff']:.2e}")
    print(f"Results match (atol=1e-10, rtol=1e-10): {results['allclose']}")
    print("=" * 70)
    
    if results['allclose']:
        print("✓ Implementations agree!")
    else:
        print("✗ Warning: Implementations differ!")


if __name__ == "__main__":
    test_log_likelihood_phase_grid_comparison()
