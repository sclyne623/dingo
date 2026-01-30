"""
Tests for the optimized batch processing in log_likelihood_phase_grid_batch
and its usage in sample_synthetic_phase.
"""

import numpy as np
import pandas as pd
import pytest


class TestPhaseGridBatchOptimization:
    """
    Test suite for the optimized batch processing implementation.
    """
    
    def test_batch_vs_individual(self, likelihood_object, test_samples):
        """
        Test that batch processing produces identical results to individual processing.
        
        Parameters
        ----------
        likelihood_object : StationaryGaussianGWLikelihood
            A configured likelihood object
        test_samples : pd.DataFrame
            Test parameter samples (without phase)
        """
        phases = np.linspace(0, 2*np.pi, 50)
        n_samples = len(test_samples)
        
        # Compute using batch method
        result_batch = likelihood_object.log_likelihood_phase_grid_batch(
            test_samples, phases=phases, num_processes=1
        )
        
        # Compute individually
        result_individual = np.zeros((n_samples, len(phases)))
        for i, (_, row) in enumerate(test_samples.iterrows()):
            theta = row.to_dict()
            result_individual[i] = likelihood_object.log_likelihood_phase_grid(
                theta, phases=phases
            )
        
        # Check they match
        assert result_batch.shape == (n_samples, len(phases))
        assert np.allclose(result_batch, result_individual, rtol=1e-10, atol=1e-10), \
            f"Batch and individual results differ by up to {np.max(np.abs(result_batch - result_individual))}"
    
    def test_batch_timing(self, likelihood_object, test_samples):
        """
        Verify that batch processing is faster than individual processing.
        
        Parameters
        ----------
        likelihood_object : StationaryGaussianGWLikelihood
            A configured likelihood object  
        test_samples : pd.DataFrame
            Test parameter samples (at least 10 samples recommended)
        """
        import time
        
        phases = np.linspace(0, 2*np.pi, 100)
        
        # Time individual processing
        start = time.time()
        for _, row in test_samples.iterrows():
            theta = row.to_dict()
            _ = likelihood_object.log_likelihood_phase_grid(theta, phases=phases)
        time_individual = time.time() - start
        
        # Time batch processing
        start = time.time()
        _ = likelihood_object.log_likelihood_phase_grid_batch(
            test_samples, phases=phases, num_processes=1
        )
        time_batch = time.time() - start
        
        speedup = time_individual / time_batch
        
        print(f"\nTiming comparison:")
        print(f"  Individual: {time_individual:.3f} s")
        print(f"  Batch: {time_batch:.3f} s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Batch should be faster (at least for multiple samples)
        if len(test_samples) >= 5:
            assert speedup > 1.0, f"Batch processing should be faster, got speedup {speedup:.2f}x"
    
    def test_batch_with_multiprocessing(self, likelihood_object, test_samples):
        """
        Test that batch method works correctly with multiprocessing.
        
        Parameters
        ----------
        likelihood_object : StationaryGaussianGWLikelihood
            A configured likelihood object
        test_samples : pd.DataFrame
            Test parameter samples
        """
        phases = np.linspace(0, 2*np.pi, 50)
        
        # Single process
        result_single = likelihood_object.log_likelihood_phase_grid_batch(
            test_samples, phases=phases, num_processes=1
        )
        
        # Multiple processes
        result_multi = likelihood_object.log_likelihood_phase_grid_batch(
            test_samples, phases=phases, num_processes=2
        )
        
        # Results should be identical
        assert np.allclose(result_single, result_multi, rtol=1e-10, atol=1e-10)
    
    def test_empty_input(self, likelihood_object):
        """
        Test handling of edge cases like empty input.
        
        Parameters
        ----------
        likelihood_object : StationaryGaussianGWLikelihood
            A configured likelihood object
        """
        phases = np.linspace(0, 2*np.pi, 10)
        empty_df = pd.DataFrame()
        
        # Should handle empty input gracefully
        result = likelihood_object.log_likelihood_phase_grid_batch(
            empty_df, phases=phases
        )
        
        assert result.shape == (0, len(phases))
    
    def test_single_sample_batch(self, likelihood_object, test_samples):
        """
        Test that batch method works correctly with a single sample.
        
        Parameters
        ----------
        likelihood_object : StationaryGaussianGWLikelihood
            A configured likelihood object
        test_samples : pd.DataFrame
            Test parameter samples
        """
        phases = np.linspace(0, 2*np.pi, 30)
        single_sample = test_samples.iloc[:1]
        
        # Batch method
        result_batch = likelihood_object.log_likelihood_phase_grid_batch(
            single_sample, phases=phases
        )
        
        # Individual method
        theta = single_sample.iloc[0].to_dict()
        result_individual = likelihood_object.log_likelihood_phase_grid(
            theta, phases=phases
        )
        
        assert result_batch.shape == (1, len(phases))
        assert np.allclose(result_batch[0], result_individual, rtol=1e-10, atol=1e-10)


def example_usage():
    """
    Example showing how to use the optimized batch processing.
    """
    print("""
Example Usage:
==============

# Setup likelihood
from dingo.gw.likelihood import build_stationary_gaussian_likelihood

likelihood = build_stationary_gaussian_likelihood(...)

# Prepare samples (without phase)
import pandas as pd
theta_batch = pd.DataFrame({
    'mass_1': [35.0, 36.0, 37.0],
    'mass_2': [30.0, 31.0, 29.0],
    # ... other parameters
})

# Define phase grid
import numpy as np
phases = np.linspace(0, 2*np.pi, 100)

# Use batch processing (optimized)
log_like_batch = likelihood.log_likelihood_phase_grid_batch(
    theta_batch, 
    phases=phases,
    num_processes=4,  # For waveform generation
)

# Result shape: (n_samples, n_phases)
print(f"Shape: {log_like_batch.shape}")

# Compare to old approach (slower)
from dingo.core.multiprocessing import apply_func_with_multiprocessing

likelihood.phase_grid = phases
log_like_old = apply_func_with_multiprocessing(
    likelihood.log_likelihood_phase_grid,
    theta_batch,
    num_processes=4,
)

# Verify they match
assert np.allclose(log_like_batch, log_like_old)
print("Results match! But batch processing is much faster.")
""")


if __name__ == "__main__":
    example_usage()
