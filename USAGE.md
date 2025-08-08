# SRGAN-Rust Usage Guide

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Command Reference](#command-reference)
- [Training Custom Models](#training-custom-models)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/yourusername/srgan-rust.git
cd srgan-rust

# Build with optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# The binary will be at target/release/srgan-rust
```

### Using Docker
```bash
# Build the Docker image
docker build -t srgan-rust .

# Run with Docker
docker run -v $(pwd):/workspace srgan-rust input.png output.png
```

## Basic Usage

### Upscaling Images

The simplest way to upscale an image using the default natural image model:

```bash
srgan-rust input.png output.png
```

### Using Different Pre-trained Models

```bash
# For anime/cartoon images
srgan-rust -p anime input.png output.png

# For bilinear upscaling (no neural network)
srgan-rust -p bilinear input.png output.png

# Using a custom trained model
srgan-rust -c my_model.rsr input.png output.png
```

### Specifying Upscaling Factor

```bash
# Upscale by 2x (default is 4x)
srgan-rust -f 2 input.png output.png

# Upscale by 8x
srgan-rust -f 8 input.png output.png
```

## Command Reference

### Main Command: Upscale

```bash
srgan-rust [OPTIONS] <INPUT_FILE> <OUTPUT_FILE>
```

**Options:**
- `-p, --parameters <PARAMETERS>`: Choose pre-trained model [natural, anime, bilinear] (default: natural)
- `-c, --custom <PARAMETER_FILE>`: Use custom trained parameters file (.rsr)
- `-f, --factor <FACTOR>`: Integer upscaling factor (default: 4)

**Examples:**
```bash
# Basic upscaling
srgan-rust photo.jpg photo_4x.png

# Anime image with 2x upscaling
srgan-rust -p anime -f 2 anime.png anime_2x.png

# Custom model
srgan-rust -c models/my_trained_model.rsr input.png output.png
```

### Downscale Command

Downscale images by an integer factor:

```bash
srgan-rust downscale <FACTOR> <INPUT_FILE> <OUTPUT_FILE>
```

**Options:**
- `-c, --colourspace <COLOURSPACE>`: Colorspace for downsampling [sRGB, RGB] (default: sRGB)

**Examples:**
```bash
# Downscale by 4x
srgan-rust downscale 4 large.png small.png

# Downscale in linear RGB space
srgan-rust downscale -c RGB 2 input.png output.png
```

### PSNR Command

Calculate Peak Signal-to-Noise Ratio between two images:

```bash
srgan-rust psnr <IMAGE1> <IMAGE2>
```

**Example:**
```bash
srgan-rust psnr original.png upscaled.png
# Output: PSNR: 32.5 dB
```

### Training Command

Train a custom model on your dataset:

```bash
srgan-rust train <TRAINING_FOLDER> <PARAMETER_FILE>
```

**Options:**
- `-R, --rate <LEARNING_RATE>`: Learning rate for Adam optimizer (default: 3e-3)
- `-q, --quantise`: Quantise weights to reduce file size
- `-l, --loss <LOSS>`: Loss function [L1, L2] (default: L1)
- `-c, --colourspace <COLOURSPACE>`: Downsampling colorspace [sRGB, RGB] (default: sRGB)
- `-r, --recurse`: Recurse into subfolders for training images
- `-s, --start <START_PARAMETERS>`: Start from existing parameters file
- `-f, --factor <FACTOR>`: Upscaling factor for the network (default: 4)
- `-w, --width <WIDTH>`: Minimum channels in hidden layers (default: 16)
- `-d, --log_depth <LOG_DEPTH>`: Network depth: 2^(log_depth)-1 layers (default: 4)
- `-p, --patch_size <PATCH_SIZE>`: Training patch size (default: 48)
- `-b, --batch_size <BATCH_SIZE>`: Training batch size (default: 4)
- `-v, --val_folder <VALIDATION_FOLDER>`: Validation image folder
- `-m, --val_max <N>`: Maximum validation images per pass

**Example:**
```bash
# Basic training
srgan-rust train ./training_images model.rsr

# Advanced training with validation
srgan-rust train \
  -R 1e-4 \
  -l L2 \
  -f 4 \
  -w 32 \
  -b 8 \
  -r \
  -v ./validation_images \
  ./training_images \
  model.rsr

# Continue training from checkpoint
srgan-rust train -s checkpoint.rsr ./training_images model.rsr
```

## Training Custom Models

### Preparing Training Data

1. **Image Requirements:**
   - Use high-quality images as training data
   - Images should be at least 192x192 pixels (for default patch_size=48, factor=4)
   - Supported formats: PNG, JPG, GIF, BMP, TIFF, WebP

2. **Directory Structure:**
   ```
   training_data/
   ├── category1/
   │   ├── image1.png
   │   ├── image2.png
   │   └── ...
   ├── category2/
   │   └── ...
   └── image3.png
   ```

3. **Dataset Size:**
   - Minimum: 100 images
   - Recommended: 1000+ images
   - More diverse data = better generalization

### Training Process

1. **Start Training:**
   ```bash
   srgan-rust train -r ./training_data model.rsr
   ```

2. **Monitor Progress:**
   - Training saves checkpoints every 1000 steps
   - Watch the loss decrease over time
   - Use validation folder to track quality

3. **Fine-tuning:**
   ```bash
   # Start with pre-trained model
   srgan-rust train -s L1_x4_UCID_x1node.rsr ./my_images custom_model.rsr
   ```

### Training Tips

- **Learning Rate:** Start with default (3e-3), reduce if training becomes unstable
- **Batch Size:** Larger batches (8-16) can improve stability but require more memory
- **Network Size:** Increase width for more capacity, but slower inference
- **Loss Function:** L1 for sharper results, L2 for smoother results
- **Validation:** Always use validation set to detect overfitting

## Performance Tips

### Compilation Optimizations

```bash
# Maximum performance build
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release

# Link-time optimization (slower build, faster runtime)
RUSTFLAGS="-C target-cpu=native -C lto=fat" cargo build --release
```

### Memory Usage

- Batch processing uses more memory but is faster
- Reduce batch_size if running out of memory during training
- For inference, process large images in tiles if needed

### Speed Considerations

- Smaller networks (width=8) are 2-4x faster than default
- Bilinear mode is 100x+ faster but lower quality
- Factor=2 is ~4x faster than factor=4

## Troubleshooting

### Common Issues

**"File not found" error:**
- Check file path is correct
- Ensure file extension is included
- Use absolute paths if relative paths don't work

**"Out of memory" during training:**
- Reduce batch_size (try 1 or 2)
- Reduce patch_size (try 32 or 24)
- Use smaller network width

**Poor upscaling quality:**
- Try different pre-trained model (anime vs natural)
- Train custom model on similar images
- Check input image quality (avoid compressed JPEGs)

**Training loss not decreasing:**
- Reduce learning rate (try 1e-4 or 1e-5)
- Check training data quality
- Increase network capacity (width/depth)

### Getting Help

- Check the [README](README.md) for basic information
- Open an issue on GitHub for bugs
- See [examples/](examples/) for sample scripts

## Examples Gallery

### Natural Images
```bash
# Landscape photo
srgan-rust landscape_480p.jpg landscape_1920p.png

# Portrait with custom model
srgan-rust -c portrait_model.rsr portrait.jpg portrait_hd.png
```

### Anime/Artwork
```bash
# Anime upscaling
srgan-rust -p anime anime_360p.png anime_1440p.png

# Pixel art (use nearest neighbor for best results)
srgan-rust -p bilinear -f 8 pixel_art.png pixel_art_8x.png
```

### Batch Processing
```bash
# Process all PNG files in a directory
for file in input/*.png; do
    name=$(basename "$file" .png)
    srgan-rust "$file" "output/${name}_4x.png"
done
```

### Quality Comparison
```bash
# Compare different models
srgan-rust -p natural photo.jpg photo_natural.png
srgan-rust -p anime photo.jpg photo_anime.png
srgan-rust -p bilinear photo.jpg photo_bilinear.png

# Calculate PSNR
srgan-rust psnr photo.jpg photo_natural.png
srgan-rust psnr photo.jpg photo_anime.png
```

---

For more information, visit the [project repository](https://github.com/yourusername/srgan-rust).