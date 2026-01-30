"""
Example script demonstrating the optimization of sample_synthetic_phase
using batch processing instead of multiprocessing.

The new implementation uses log_likelihood_phase_grid_batch which:
1. Computes waveforms for all samples in batch
2. Vectorizes operations across both samples and phases
3. Reduces function call overhead

Expected speedup: 2-10x depending on number of samples and phase grid size.
"""

import numpy as np
import pandas as pd


def example_test_synthetic_phase_optimization(result_object):
    """
    Example of how to test the synthetic phase optimization.
    
    Parameters
    ----------
    result_object : GWResult
        A GWResult object with samples
    """
    
    # Get a subset of samples without phase
    param_keys = [k for k in result_object.samples.columns 
                  if k != 'phase' and k != 'log_prob']
    theta_sample = result_object.samples[param_keys].iloc[:50]
    
    # Test with different configurations
    print("\n" + "=" * 80)
    print("Testing Sample Synthetic Phase Optimization")
    print("=" * 80)
    
    # Test 1: Small phase grid
    print("\n\nTest 1: Small phase grid (n_grid=50)")
    print("-" * 80)
    results1 = result_object.test_sample_synthetic_phase_optimization(
        theta_sample.iloc[:10],
        n_grid=50,
        approximation_22_mode=False,
        num_processes=1,
    )
    
    # Test 2: Larger phase grid
    print("\n\nTest 2: Larger phase grid (n_grid=200)")
    print("-" * 80)
    results2 = result_object.test_sample_synthetic_phase_optimization(
        theta_sample.iloc[:10],
        n_grid=200,
        approximation_22_mode=False,
        num_processes=1,
    )
    
    # Test 3: More samples
    print("\n\nTest 3: More samples (n_samples=50)")
    print("-" * 80)
    results3 = result_object.test_sample_synthetic_phase_optimization(
        theta_sample,
        n_grid=100,
        approximation_22_mode=False,
        num_processes=1,
    )
    
    # Visualize results
    try:
        import matplotlib.pyplot as plt
        
        configs = ['50 phases\n10 samples', '200 phases\n10 samples', 
                   '100 phases\n50 samples']
        speedups = [results1['speedup'], results2['speedup'], results3['speedup']]
        times_orig = [results1['time_orig'], results2['time_orig'], results3['time_orig']]
        times_opt = [results1['time_opt'], results2['time_opt'], results3['time_opt']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Speedup comparison
        x = np.arange(len(configs))
        bars = ax1.bar(x, speedups, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax1.axhline(1, color='k', linestyle='--', alpha=0.3, label='No speedup')
        ax1.set_ylabel('Speedup Factor', fontsize=12)
        ax1.set_xlabel('Configuration', fontsize=12)
        ax1.set_title('Speedup: Batch vs Multiprocessing', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x',
                    ha='center', va='bottom', fontweight='bold')
        
        # Time comparison
        width = 0.35
        ax2.bar(x - width/2, times_orig, width, label='Original (multiprocessing)', 
                color='#E63946')
        ax2.bar(x + width/2, times_opt, width, label='Optimized (batch)', 
                color='#06A77D')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_xlabel('Configuration', fontsize=12)
        ax2.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('synthetic_phase_optimization.png', dpi=150)
        print("\n\n✓ Saved plot to 'synthetic_phase_optimization.png'")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")
    
    return {
        'test1': results1,
        'test2': results2,
        'test3': results3,
    }


def quick_usage_example():
    """
    Print usage instructions.
    """
    print("""
Usage Example:
==============

# Load your GWResult object
from dingo.gw.result import GWResult

result = GWResult.from_file('path/to/result.hdf5')

# Method 1: Use built-in test method
import pandas as pd

# Get samples without phase
param_keys = [k for k in result.samples.columns if k not in ['phase', 'log_prob']]
theta_sample = result.samples[param_keys].iloc[:20]

# Run test
results = result.test_sample_synthetic_phase_optimization(
    theta_sample,
    n_grid=100,
    approximation_22_mode=False,
    num_processes=1,
)

# Method 2: Actually use the optimized version
# The optimization is automatically used when you call sample_synthetic_phase
# with approximation_22_mode=False

synthetic_phase_kwargs = {
    'n_grid': 100,
    'approximation_22_mode': False,
    'num_processes': 4,  # For waveform generation
    'uniform_weight': 0.01,
}

result.sample_synthetic_phase(synthetic_phase_kwargs)

# The batch processing will automatically be used, providing significant speedup!

Performance Notes:
==================
- The batch processing approach is most beneficial for:
  * Large number of samples (>10)
  * Large phase grids (>50 points)
  * When approximation_22_mode=False
  
- For approximation_22_mode=True, a different (already fast) code path is used

- Expected speedup: 2-10x depending on configuration
  * More samples = better speedup
  * Larger phase grids = better speedup
  * Overhead reduction becomes significant with many samples
""")


if __name__ == "__main__":
    quick_usage_example()
