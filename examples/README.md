# SRGAN-Rust Example Datasets and Scripts

This directory contains scripts and tools for preparing datasets for training and testing SRGAN-Rust models.

## 📥 Download Datasets

### Quick Start
```bash
# Make the script executable
chmod +x download_datasets.sh

# Run the download script
./download_datasets.sh
```

This script will download popular super-resolution datasets including:
- **DIV2K**: 800 high-quality training images and 100 validation images
- **Set5**: Classic 5-image benchmark dataset
- **Set14**: Extended 14-image benchmark dataset
- **BSD100**: 100 natural images from Berkeley Segmentation Dataset
- **Urban100**: 100 urban scene images with structured patterns

## 🔧 Create Custom Training Datasets

### Python Dataset Creator

The `create_training_dataset.py` script helps you create training datasets from your own images.

#### Installation
```bash
# Install required Python packages
pip install pillow numpy
```

#### Basic Usage
```bash
# Create dataset from a directory of images
python create_training_dataset.py input_images/ output_dataset/ \
    --patch-size 96 \
    --scale 4 \
    --augment flip_h flip_v rotate_90
```

#### Options
- `--patch-size`: Size of extracted patches (default: 96)
- `--scale`: Downscaling factor for LR images (default: 4)
- `--stride`: Stride for patch extraction (default: same as patch_size)
- `--augment`: Data augmentation methods (flip_h, flip_v, rotate_90)
- `--train-split`: Train/validation split ratio (default: 0.8)
- `--min-std`: Minimum standard deviation to filter uniform patches (default: 10.0)
- `--create-test`: Generate synthetic test images

#### Example: Create Dataset with Augmentation
```bash
# Create augmented dataset with overlapping patches
python create_training_dataset.py \
    ~/Pictures/high_res_photos/ \
    ./datasets/my_dataset/ \
    --patch-size 128 \
    --scale 4 \
    --stride 64 \
    --augment flip_h flip_v \
    --train-split 0.9
```

## 📊 Dataset Structure

After running the scripts, your dataset directory will be organized as:

```
datasets/
├── div2k_train/           # Original HR images
│   └── *.png
├── div2k_train_LR_x4/     # 4x downscaled versions
│   └── *.png
├── my_dataset/            # Custom dataset
│   ├── train/
│   │   ├── HR/           # High-resolution patches
│   │   └── LR_x4/        # Low-resolution patches
│   ├── validation/
│   │   ├── HR/
│   │   └── LR_x4/
│   └── dataset_info.txt  # Dataset statistics
└── ...
```

## 🚀 Training with Datasets

### Using Downloaded Datasets
```bash
# Train with DIV2K dataset
cargo run --release -- train \
    --train-folder ./datasets/div2k_train \
    --val-folder ./datasets/div2k_valid \
    --output models/my_div2k_model.rsr \
    --batch-size 4 \
    --patch-size 96
```

### Using Custom Datasets
```bash
# Train with custom dataset
cargo run --release -- train \
    --train-folder ./datasets/my_dataset/train/HR \
    --val-folder ./datasets/my_dataset/validation/HR \
    --output models/my_custom_model.rsr \
    --factor 4
```

### Using Configuration Files
```bash
# Generate a configuration file
cargo run --release -- generate-config training_config.toml --preset advanced

# Train with configuration
cargo run --release -- train --config training_config.toml
```

## 🧪 Testing Models

### Test on Benchmark Datasets
```bash
# Test on Set5
for img in ./datasets/set5_LR_x4/*.png; do
    cargo run --release -- "$img" "output/$(basename $img)" -p models/my_model.rsr
done

# Calculate PSNR
cargo run --release -- psnr ./datasets/set5/*.png output/*.png
```

### Batch Processing
```bash
# Process entire directory
cargo run --release -- batch \
    --input ./datasets/set5_LR_x4 \
    --output ./results/set5_upscaled \
    --model models/my_model.rsr \
    --recursive
```

## 🔍 Benchmarking

```bash
# Compare models on test set
cargo run --release -- benchmark \
    --input ./datasets/set5_LR_x4 \
    --models natural anime models/my_model.rsr \
    --output benchmark_results.csv
```

## 💡 Tips

1. **Dataset Size**: For effective training, aim for at least 1000 training patches
2. **Patch Size**: Larger patches (128-256) capture more context but require more memory
3. **Augmentation**: Use augmentation to increase dataset diversity
4. **Validation Set**: Keep 10-20% of data for validation to monitor training progress
5. **Image Quality**: Use high-quality source images without compression artifacts

## 📚 Additional Resources

- [DIV2K Dataset Paper](https://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf)
- [Super-Resolution Benchmarks](https://paperswithcode.com/task/image-super-resolution)
- [SRGAN Paper](https://arxiv.org/abs/1609.04802)

## 🐛 Troubleshooting

### Download Issues
- Some datasets require manual download due to licensing
- Check your internet connection and available disk space
- Use a VPN if certain sources are blocked in your region

### Memory Issues
- Reduce `--batch-size` if training runs out of memory
- Use smaller `--patch-size` for limited GPU memory
- Process images in smaller batches with the batch command

### Performance
- Use `--release` flag when building for optimal performance
- Enable GPU acceleration with `upscale-gpu` command if available
- Profile memory usage with `profile-memory` command