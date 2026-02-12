#!/bin/bash
# Quick validation script for LISA optimizations
# Run this to verify that optimizations work correctly

set -e  # Exit on error

echo "=================================="
echo "LISA Optimization Validation"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -d "dingo/gw/transforms" ]; then
    echo "Error: Please run this script from the dingo-LISA root directory"
    exit 1
fi

echo "Step 1: Checking Python syntax..."
python -m py_compile dingo/gw/transforms/detector_transforms.py
echo "✓ Syntax check passed"
echo ""

echo "Step 2: Running validation tests..."
python tests/gw/transforms/test_optimization_validation.py
echo ""

echo "=================================="
echo "Validation Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Run benchmark: python misc_scripts/benchmark_optimizations.py"
echo "  2. Profile pipeline: python misc_scripts/profile_lisa_transforms.py"
echo "  3. Test on real training workload"
echo ""
