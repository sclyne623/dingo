"""
Example script demonstrating how to test the optimized log_likelihood_phase_grid
implementation against the reference implementation.

This script shows how to:
1. Create a likelihood object
2. Run the comparison test
3. Visualize the results
"""

import numpy as np
import matplotlib.pyplot as plt


def example_test_with_likelihood(likelihood, theta):
    """
    Example of how to test the phase grid implementations with an existing
    likelihood object.
    
    Parameters
    ----------
    likelihood : StationaryGaussianGWLikelihood
        A configured likelihood object
    theta : dict
        Parameter dictionary for the test
    """
    
    # Test with different phase grid sizes to see scaling
    grid_sizes = [10, 50, 100, 500, 1000]
    
    print("\nTesting different phase grid sizes...")
    print("=" * 80)
    
    speedups = []
    for n_phases in grid_sizes:
        phases = np.linspace(0, 2*np.pi, n_phases)
        results = likelihood.test_phase_grid_implementations(
            theta, phases, verbose=False
        )
        speedups.append(results['speedup'])
        
        print(f"n_phases={n_phases:5d}: "
              f"speedup={results['speedup']:6.2f}x, "
              f"match={results['allclose']}, "
              f"max_diff={results['max_abs_diff']:.2e}")
    
    print("=" * 80)
    
    # Plot speedup vs grid size
    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, speedups, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Phase Points', fontsize=12)
    plt.ylabel('Speedup Factor', fontsize=12)
    plt.title('Optimized vs Reference Implementation Speedup', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('phase_grid_speedup.png', dpi=150)
    print("\nSaved speedup plot to 'phase_grid_speedup.png'")
    
    # Detailed test with a moderate grid
    print("\n\nDetailed test with 100 phase points:")
    print("=" * 80)
    phases = np.linspace(0, 2*np.pi, 100)
    results = likelihood.test_phase_grid_implementations(theta, phases, verbose=True)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot log likelihoods
    axes[0].plot(phases, results['log_like_ref'], 'o-', label='Reference', alpha=0.7)
    axes[0].plot(phases, results['log_like_opt'], 'x--', label='Optimized', alpha=0.7)
    axes[0].set_xlabel('Phase', fontsize=12)
    axes[0].set_ylabel('Log Likelihood', fontsize=12)
    axes[0].set_title('Log Likelihood vs Phase', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot difference
    diff = results['log_like_ref'] - results['log_like_opt']
    axes[1].plot(phases, diff, 'r-', linewidth=2)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Phase', fontsize=12)
    axes[1].set_ylabel('Difference (Reference - Optimized)', fontsize=12)
    axes[1].set_title(f'Difference (max = {np.max(np.abs(diff)):.2e})', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('phase_grid_comparison.png', dpi=150)
    print("\nSaved comparison plot to 'phase_grid_comparison.png'")
    
    return results


def quick_test_example():
    """
    Quick example showing the test in action.
    Run this with your own likelihood object.
    """
    print("""
To test the phase grid implementations, use the following pattern:

# Create or load your likelihood object
from dingo.gw.likelihood import build_stationary_gaussian_likelihood

# Assuming you have metadata and event data...
# likelihood = build_stationary_gaussian_likelihood(
#     metadata=metadata,
#     event_dataset=event_dataset,
# )

# Define your test parameters
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

# Run the test
import numpy as np
phases = np.linspace(0, 2*np.pi, 100)

# Option 1: Use the built-in test method (recommended)
results = likelihood.test_phase_grid_implementations(theta, phases)

# Option 2: Manual comparison
import time

start = time.time()
log_like_ref = likelihood.log_likelihood_phase_grid_reference(theta, phases)
time_ref = time.time() - start

start = time.time()
log_like_opt = likelihood.log_likelihood_phase_grid(theta, phases)
time_opt = time.time() - start

print(f"Reference: {time_ref:.4f} s")
print(f"Optimized: {time_opt:.4f} s")
print(f"Speedup: {time_ref/time_opt:.2f}x")
print(f"Match: {np.allclose(log_like_ref, log_like_opt, rtol=1e-10)}")
    """)


if __name__ == "__main__":
    quick_test_example()
