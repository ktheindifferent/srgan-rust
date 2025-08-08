#!/bin/bash

# Benchmarking script for SRGAN-Rust
# Compare performance of different models and configurations

echo "SRGAN-Rust Performance Benchmark"
echo "================================="

# Check if srgan-rust is available
if ! command -v srgan-rust &> /dev/null; then
    echo "srgan-rust not found. Please build and install it first."
    exit 1
fi

# Test different image sizes
echo -e "\n1. Testing different image sizes with natural model:"
for size in 256 512 1024; do
    echo -e "\n  Testing ${size}x${size} input:"
    srgan-rust benchmark \
        --width "$size" \
        --height "$size" \
        --iterations 10 \
        --warmup 3
done

# Compare all models
echo -e "\n2. Comparing all available models (512x512 input):"
srgan-rust benchmark \
    --width 512 \
    --height 512 \
    --compare \
    --iterations 10

# Test different upscaling factors
echo -e "\n3. Testing different upscaling factors:"
for factor in 2 4 8; do
    echo -e "\n  Testing ${factor}x upscaling:"
    srgan-rust benchmark \
        --factor "$factor" \
        --width 512 \
        --height 512 \
        --iterations 10
done

# Generate CSV output for analysis
echo -e "\n4. Generating CSV report:"
srgan-rust benchmark \
    --compare \
    --csv \
    --iterations 20 \
    > benchmark_results.csv

echo "CSV results saved to benchmark_results.csv"

# Memory usage test
echo -e "\n5. Memory usage test with large image:"
srgan-rust benchmark \
    --width 2048 \
    --height 2048 \
    --iterations 1 \
    --warmup 0

echo -e "\nBenchmark complete!"