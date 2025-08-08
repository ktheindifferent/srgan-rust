#!/bin/bash

# SRGAN-Rust Example Datasets Download Script
# Downloads and prepares various datasets for training and testing

set -e

echo "SRGAN-Rust Dataset Downloader"
echo "=============================="
echo ""

# Create datasets directory
DATASET_DIR="./datasets"
mkdir -p "$DATASET_DIR"

# Function to download and extract dataset
download_dataset() {
    local name=$1
    local url=$2
    local target_dir="$DATASET_DIR/$name"
    
    if [ -d "$target_dir" ]; then
        echo "⚠️  Dataset '$name' already exists, skipping..."
        return
    fi
    
    echo "📥 Downloading $name dataset..."
    mkdir -p "$target_dir"
    
    if [[ $url == *.zip ]]; then
        wget -q --show-progress -O "$target_dir/temp.zip" "$url"
        echo "📦 Extracting..."
        unzip -q "$target_dir/temp.zip" -d "$target_dir"
        rm "$target_dir/temp.zip"
    elif [[ $url == *.tar.gz ]] || [[ $url == *.tgz ]]; then
        wget -q --show-progress -O "$target_dir/temp.tar.gz" "$url"
        echo "📦 Extracting..."
        tar -xzf "$target_dir/temp.tar.gz" -C "$target_dir"
        rm "$target_dir/temp.tar.gz"
    else
        echo "❌ Unsupported archive format for $url"
        return 1
    fi
    
    echo "✅ $name dataset ready!"
    echo ""
}

# DIV2K Dataset (High Quality Images for Super-Resolution)
echo "1. DIV2K Dataset (800 training images)"
echo "   High-resolution images for super-resolution tasks"
read -p "   Download DIV2K? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_dataset "div2k_train" "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    download_dataset "div2k_valid" "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
fi

# Set5 Benchmark Dataset
echo "2. Set5 Benchmark Dataset"
echo "   Classic benchmark for super-resolution evaluation"
read -p "   Download Set5? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$DATASET_DIR/set5"
    echo "📥 Downloading Set5 images..."
    wget -q --show-progress -P "$DATASET_DIR/set5" \
        "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/baby.png" \
        "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/bird.png" \
        "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/butterfly.png" \
        "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/head.png" \
        "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/woman.png"
    echo "✅ Set5 dataset ready!"
    echo ""
fi

# Set14 Benchmark Dataset
echo "3. Set14 Benchmark Dataset"
echo "   Extended benchmark for super-resolution evaluation"
read -p "   Download Set14? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$DATASET_DIR/set14"
    echo "📥 Downloading Set14 images..."
    # Note: These URLs are examples - actual Set14 images would need proper sources
    echo "⚠️  Set14 requires manual download from research sources"
    echo "   Please visit: https://sites.google.com/site/romanzeyde/research-interests"
    echo ""
fi

# BSD100 Dataset
echo "4. BSD100 Dataset"
echo "   100 natural images from Berkeley Segmentation Dataset"
read -p "   Download BSD100? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$DATASET_DIR/bsd100"
    echo "⚠️  BSD100 requires manual download"
    echo "   Please visit: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/"
    echo ""
fi

# Urban100 Dataset
echo "5. Urban100 Dataset"
echo "   100 urban scene images with structured patterns"
read -p "   Download Urban100? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$DATASET_DIR/urban100"
    echo "⚠️  Urban100 requires manual download"
    echo "   Please visit: https://sites.google.com/site/jbhuang0604/publications/struct_sr"
    echo ""
fi

# Create sample low-resolution versions
echo ""
echo "Creating low-resolution training pairs..."
echo "========================================="

create_lr_versions() {
    local dataset=$1
    local scale=$2
    local input_dir="$DATASET_DIR/$dataset"
    local output_dir="$DATASET_DIR/${dataset}_LR_x${scale}"
    
    if [ ! -d "$input_dir" ]; then
        echo "⚠️  Dataset $dataset not found, skipping LR generation..."
        return
    fi
    
    if [ -d "$output_dir" ]; then
        echo "⚠️  LR version for $dataset already exists, skipping..."
        return
    fi
    
    mkdir -p "$output_dir"
    echo "📉 Generating x${scale} downscaled versions for $dataset..."
    
    # Use the srgan-rust downscale command if available
    if command -v cargo &> /dev/null && [ -f "../Cargo.toml" ]; then
        for img in "$input_dir"/*.{png,jpg,jpeg} 2>/dev/null; do
            if [ -f "$img" ]; then
                filename=$(basename "$img")
                cargo run --release -- downscale "$img" "$output_dir/$filename" -f $scale 2>/dev/null || true
            fi
        done
    else
        echo "⚠️  srgan-rust not found, using ImageMagick for downscaling..."
        for img in "$input_dir"/*.{png,jpg,jpeg} 2>/dev/null; do
            if [ -f "$img" ]; then
                filename=$(basename "$img")
                convert "$img" -resize $((100/$scale))% "$output_dir/$filename" 2>/dev/null || true
            fi
        done
    fi
    
    echo "✅ LR versions created for $dataset"
}

# Generate LR versions for downloaded datasets
for scale in 2 4; do
    create_lr_versions "div2k_train" $scale
    create_lr_versions "div2k_valid" $scale
    create_lr_versions "set5" $scale
done

echo ""
echo "Dataset preparation complete!"
echo "============================="
echo ""
echo "Downloaded datasets are in: $DATASET_DIR"
echo ""
echo "Training example:"
echo "  cargo run --release -- train \\"
echo "    --train-folder $DATASET_DIR/div2k_train \\"
echo "    --val-folder $DATASET_DIR/div2k_valid \\"
echo "    --output my_model.rsr"
echo ""
echo "Testing example:"
echo "  cargo run --release -- $DATASET_DIR/set5_LR_x4/baby.png output.png -p my_model.rsr"
echo ""