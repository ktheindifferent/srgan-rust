#!/bin/bash

# Batch processing example script for SRGAN-Rust
# This script demonstrates various batch processing scenarios

# Check if srgan-rust is available
if ! command -v srgan-rust &> /dev/null; then
    echo "srgan-rust not found. Please build and install it first."
    exit 1
fi

# Function to process a directory
process_directory() {
    local input_dir=$1
    local output_dir=$2
    local model=$3
    
    echo "Processing images in $input_dir with $model model..."
    
    srgan-rust batch \
        "$input_dir" \
        "$output_dir" \
        -p "$model" \
        -r \
        --skip-existing
}

# Example 1: Process photos with natural model
echo "Example 1: Processing photos with natural model"
mkdir -p output/photos
process_directory "input/photos" "output/photos" "natural"

# Example 2: Process anime images with anime model
echo "Example 2: Processing anime images with anime model"
mkdir -p output/anime
process_directory "input/anime" "output/anime" "anime"

# Example 3: Process with custom model
if [ -f "custom_model.rsr" ]; then
    echo "Example 3: Processing with custom model"
    mkdir -p output/custom
    srgan-rust batch \
        input/test \
        output/custom \
        -c custom_model.rsr \
        -r \
        --skip-existing
fi

# Example 4: Process with different upscaling factors
echo "Example 4: Multi-scale processing"
for factor in 2 4 8; do
    echo "Processing with ${factor}x upscaling..."
    mkdir -p "output/scale_${factor}x"
    srgan-rust batch \
        input/samples \
        "output/scale_${factor}x" \
        -f "$factor" \
        -r
done

# Example 5: Sequential processing for large images
echo "Example 5: Sequential processing (memory-efficient)"
srgan-rust batch \
    input/large_images \
    output/large_images \
    -s \
    -r \
    --skip-existing

echo "Batch processing complete!"