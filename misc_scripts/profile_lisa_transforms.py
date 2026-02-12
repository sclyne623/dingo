#!/usr/bin/env python3
"""
Performance profiling script for LISA waveform transformations.

This script helps measure the speedup from optimizations in:
- process_transfer function (spline operations)
- ProjectOntoSpaceDetectors batch processing
- Distance scaling vectorization

Usage:
    python profile_lisa_transforms.py --batch-size 8 --num-modes 5
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add dingo to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from dingo.gw.transforms.detector_transforms import ProjectOntoSpaceDetectors
from dingo.gw.domains import build_domain
from dingo.gw.waveform_generator import LISAWaveformGenerator
from dingo.gw.prior import build_prior_with_defaults


def create_test_sample(batch_size=1, num_freq_points=1000, num_modes=5):
    """
    Create a mock waveform sample for testing.
    
    Parameters
    ----------
    batch_size : int
        Number of waveforms in batch
    num_freq_points : int
        Number of frequency points per waveform
    num_modes : int
        Number of (l,m) modes
        
    Returns
    -------
    dict
        Sample dictionary in DINGO format
    """
    freq_grid = np.linspace(1e-4, 1e-1, num_freq_points)
    
    # Create mock waveform data
    waveform = {}
    modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)][:num_modes]
    
    if batch_size == 1:
        for l, m in modes:
            waveform[(l, m)] = {
                'freq': freq_grid,
                'amp': np.random.rand(num_freq_points),
                'phase': np.random.rand(num_freq_points),
                'tf': np.random.rand(num_freq_points),
            }
    else:
        for l, m in modes:
            waveform[(l, m)] = {
                'freq': [freq_grid for _ in range(batch_size)],
                'amp': [np.random.rand(num_freq_points) for _ in range(batch_size)],
                'phase': [np.random.rand(num_freq_points) for _ in range(batch_size)],
                'tf': [np.random.rand(num_freq_points) for _ in range(batch_size)],
            }
    
    # Create mock parameters
    if batch_size == 1:
        parameters = {
            'dist': 100.0,
            'inc': 0.5,
            'geocent_time': 0.0,
            'phi': 1.0,
        }
        extrinsic_parameters = {
            'dist': 110.0,
            'inc': 0.5,
            'beta': 0.3,
            'lambda': 1.2,
            'psi': 0.7,
            'geocent_time': 0.0,
            'phi': 1.0,
        }
    else:
        parameters = {
            'dist': np.ones(batch_size) * 100.0,
            'inc': np.random.rand(batch_size),
            'geocent_time': np.zeros(batch_size),
            'phi': np.random.rand(batch_size),
        }
        extrinsic_parameters = {
            'dist': np.ones(batch_size) * 110.0,
            'inc': np.random.rand(batch_size),
            'beta': np.random.rand(batch_size),
            'lambda': np.random.rand(batch_size),
            'psi': np.random.rand(batch_size),
            'geocent_time': np.zeros(batch_size),
            'phi': np.random.rand(batch_size),
        }
    
    return {
        'waveform': waveform,
        'parameters': parameters,
        'extrinsic_parameters': extrinsic_parameters,
    }


def profile_transform(transform, sample, num_iterations=10):
    """
    Profile a transform by timing multiple iterations.
    
    Parameters
    ----------
    transform : callable
        Transform to profile
    sample : dict
        Sample to transform
    num_iterations : int
        Number of iterations to average over
        
    Returns
    -------
    dict
        Timing statistics
    """
    times = []
    
    # Warmup
    _ = transform(sample)
    
    # Timed runs
    for _ in range(num_iterations):
        sample_copy = {
            'waveform': sample['waveform'].copy(),
            'parameters': sample['parameters'].copy(),
            'extrinsic_parameters': sample['extrinsic_parameters'].copy(),
        }
        
        start = time.perf_counter()
        _ = transform(sample_copy)
        end = time.perf_counter()
        
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Profile LISA waveform transformation performance'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for testing (default: 8)'
    )
    parser.add_argument(
        '--num-modes',
        type=int,
        default=5,
        help='Number of (l,m) modes (default: 5)'
    )
    parser.add_argument(
        '--num-freq',
        type=int,
        default=1000,
        help='Number of frequency points (default: 1000)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations for averaging (default: 10)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LISA Waveform Transform Performance Profiling")
    print("=" * 70)
    print(f"Batch size: {args.batch_size}")
    print(f"Number of modes: {args.num_modes}")
    print(f"Frequency points: {args.num_freq}")
    print(f"Iterations: {args.iterations}")
    print()
    
    # Create LISA domain and transform
    domain_dict = {
        'type': 'FrequencyDomain',
        'f_min': 1e-4,
        'f_max': 1e-1,
        'delta_f': 1e-5,
    }
    domain = build_domain(domain_dict)
    
    lisa_settings = {
        'LISAconst': "Proposal",
        'responseapprox': "full",
        'frozenLISA': True,
        'TDIrescaled': False,
    }
    
    transform = ProjectOntoSpaceDetectors(
        detector_type="TDIAET",
        domain=domain,
        ref_time=0.0,
        channels=["chan1", "chan2", "chan3"],
        lisa_settings=lisa_settings
    )
    
    # Profile single waveform
    print("Profiling single waveform...")
    sample_single = create_test_sample(
        batch_size=1,
        num_freq_points=args.num_freq,
        num_modes=args.num_modes
    )
    stats_single = profile_transform(transform, sample_single, args.iterations)
    print(f"  Mean time: {stats_single['mean']*1000:.2f} ms")
    print(f"  Std dev:   {stats_single['std']*1000:.2f} ms")
    print(f"  Min time:  {stats_single['min']*1000:.2f} ms")
    print(f"  Max time:  {stats_single['max']*1000:.2f} ms")
    print()
    
    # Profile batch
    print(f"Profiling batch of {args.batch_size} waveforms...")
    sample_batch = create_test_sample(
        batch_size=args.batch_size,
        num_freq_points=args.num_freq,
        num_modes=args.num_modes
    )
    stats_batch = profile_transform(transform, sample_batch, args.iterations)
    print(f"  Mean time: {stats_batch['mean']*1000:.2f} ms")
    print(f"  Std dev:   {stats_batch['std']*1000:.2f} ms")
    print(f"  Min time:  {stats_batch['min']*1000:.2f} ms")
    print(f"  Max time:  {stats_batch['max']*1000:.2f} ms")
    print()
    
    # Compute speedup metrics
    time_per_waveform_single = stats_single['mean']
    time_per_waveform_batch = stats_batch['mean'] / args.batch_size
    speedup = time_per_waveform_single / time_per_waveform_batch
    
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Time per waveform (sequential): {time_per_waveform_single*1000:.2f} ms")
    print(f"Time per waveform (batched):    {time_per_waveform_batch*1000:.2f} ms")
    print(f"Speedup factor:                 {speedup:.2f}x")
    print()
    
    if speedup > 1.5:
        print("✓ Good speedup achieved with batching!")
    elif speedup > 1.0:
        print("✓ Some speedup achieved with batching")
    else:
        print("⚠ No speedup with batching - overhead may be too high")
    
    print()
    print("To compare with old version:")
    print("1. Checkout the commit before optimizations")
    print("2. Run this script with same parameters")
    print("3. Compare the times")


if __name__ == '__main__':
    main()
