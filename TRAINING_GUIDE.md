# SRGAN-Rust Training Guide

This guide will help you train your own super-resolution models using SRGAN-Rust.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training Process](#training-process)
- [Monitoring Progress](#monitoring-progress)
- [Fine-tuning](#fine-tuning)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before training, ensure you have:
- A collection of high-quality images (minimum 100, recommended 1000+)
- Sufficient disk space (models are ~40-100MB, checkpoints save frequently)
- Time (training can take hours to days depending on dataset size)
- Optional: CUDA-capable GPU (currently CPU-only, GPU support planned)

## Dataset Preparation

### Image Requirements

Your training images should be:
- **High resolution**: At least 256x256 pixels, preferably larger
- **High quality**: Avoid compressed JPEGs, use PNG when possible
- **Diverse**: Include various subjects, lighting conditions, and styles
- **Relevant**: Match the type of images you plan to upscale

### Directory Structure

Organize your images like this:

```
training_data/
├── landscapes/
│   ├── mountains_001.png
│   ├── mountains_002.png
│   └── ...
├── portraits/
│   ├── person_001.png
│   ├── person_002.png
│   └── ...
└── objects/
    ├── car_001.png
    ├── building_001.png
    └── ...

validation_data/  (optional but recommended)
├── test_landscape.png
├── test_portrait.png
└── ...
```

### Dataset Size Guidelines

| Dataset Size | Training Time | Quality | Use Case |
|-------------|--------------|---------|----------|
| 100-500 images | 1-4 hours | Basic | Quick experiments |
| 500-2000 images | 4-12 hours | Good | Domain-specific models |
| 2000-10000 images | 12-48 hours | Excellent | Production models |
| 10000+ images | 2+ days | Best | General-purpose models |

## Configuration

### Quick Start with Default Settings

```bash
# Basic training with default parameters
srgan-rust train ./training_data model.rsr

# With validation
srgan-rust train -v ./validation_data ./training_data model.rsr
```

### Using Configuration Files

1. Generate a configuration file:
```bash
# Generate example config with comments
srgan-rust generate-config --example training_config.toml

# Generate minimal config
srgan-rust generate-config --format toml my_config.toml
```

2. Edit the configuration file:
```toml
[network]
factor = 4          # Upscaling factor (2, 4, or 8)
width = 32          # Network width (16-64, higher = better quality but slower)
log_depth = 4       # Network depth (3-5, 4 is good default)

[training]
learning_rate = 0.001   # Lower for fine-tuning, higher for scratch
batch_size = 4          # Increase if you have more memory
patch_size = 48         # Size of training patches
loss_type = "L1"        # L1 for sharp, L2 for smooth

[data]
training_folder = "./training_data"
recurse = true          # Search subdirectories

[validation]
validation_folder = "./validation_data"
frequency = 100         # Validate every N steps
```

3. Train with config:
```bash
srgan-rust train-config training_config.toml
```

### Key Parameters Explained

#### Network Architecture
- **factor**: Upscaling factor (2x, 4x, 8x)
  - 4x is most common and well-tested
  - 2x trains faster, good for iterative upscaling
  - 8x requires more training data and time

- **width**: Number of channels in hidden layers
  - 16: Fast but lower quality
  - 32: Good balance (recommended)
  - 64: Best quality but slow

- **log_depth**: Controls network depth
  - 3: 7 layers (fast, lower quality)
  - 4: 15 layers (recommended)
  - 5: 31 layers (slow, marginal improvement)

#### Training Hyperparameters
- **learning_rate**: How fast the model learns
  - 0.003: Default, good for training from scratch
  - 0.001: Safer, less likely to diverge
  - 0.0001: For fine-tuning existing models

- **batch_size**: Images processed together
  - 1-2: Low memory usage, slower
  - 4-8: Good balance
  - 16+: Faster but requires more memory

- **patch_size**: Size of training patches
  - 32: Faster training, less context
  - 48: Good default
  - 64+: More context but slower

- **loss_type**: Optimization target
  - L1: Produces sharper results (recommended)
  - L2: Smoother, less artifacts

## Training Process

### Starting Training

1. **From scratch**:
```bash
srgan-rust train \
  -R 0.003 \           # Learning rate
  -f 4 \               # 4x upscaling
  -w 32 \              # Network width
  -b 8 \               # Batch size
  -r \                 # Recurse subdirectories
  ./training_data \
  model.rsr
```

2. **From pre-trained model** (recommended):
```bash
srgan-rust train \
  -s res/L1_x4_UCID_x1node.rsr \  # Start from existing model
  -R 0.0001 \                      # Lower learning rate for fine-tuning
  ./training_data \
  custom_model.rsr
```

3. **With validation**:
```bash
srgan-rust train \
  -v ./validation_data \
  -m 50 \              # Max 50 validation images
  ./training_data \
  model.rsr
```

### Training Stages

Training typically progresses through these stages:

1. **Initial Stage (0-1000 steps)**
   - Loss drops rapidly
   - Output is blurry/noisy
   - Basic structures emerge

2. **Refinement Stage (1000-10000 steps)**
   - Loss decreases slowly
   - Details become sharper
   - Textures improve

3. **Fine-tuning Stage (10000+ steps)**
   - Loss plateaus
   - Subtle improvements
   - Risk of overfitting

## Monitoring Progress

### Reading Training Output

```
step 1000    err:0.0234    change:0.0012
```
- **step**: Current training iteration
- **err**: Current loss (lower is better)
- **change**: Parameter change magnitude

### Checkpoints

Checkpoints are saved every 1000 steps by default:
- Location: Same directory as output file
- Naming: `model.rsr.checkpoint_1000`, etc.
- Can resume from any checkpoint

### Validation Metrics

If using validation:
```
Validation - PSNR: 28.3 dB | Loss: 0.0198
```
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better, 30+ is good)
- **Loss**: Validation loss (should decrease over time)

### When to Stop Training

Stop training when:
- Validation loss stops improving for 5000+ steps
- Visual quality is satisfactory
- Training loss < 0.01 (for L1 loss)

## Fine-tuning

### Domain-Specific Models

For best results, fine-tune for your specific use case:

1. **Anime/Artwork**:
```bash
# Start from anime model
srgan-rust train \
  -s res/L1_x4_Anime_x1node.rsr \
  -R 0.0001 \
  ./anime_dataset \
  anime_custom.rsr
```

2. **Faces/Portraits**:
```bash
# Use larger patches for faces
srgan-rust train \
  -p 64 \
  -R 0.0003 \
  ./face_dataset \
  face_model.rsr
```

3. **Text/Documents**:
```bash
# Use L2 loss for smoother text
srgan-rust train \
  -l L2 \
  -R 0.001 \
  ./document_dataset \
  text_model.rsr
```

### Progressive Training

Train models progressively for better results:

1. Train 2x model first
2. Use 2x model output as training data for 4x model
3. Combine for 8x upscaling

## Best Practices

### Do's
- ✅ Use high-quality training images
- ✅ Start from pre-trained models when possible
- ✅ Use validation data to prevent overfitting
- ✅ Save checkpoints frequently
- ✅ Experiment with small datasets first
- ✅ Monitor both loss and visual quality

### Don'ts
- ❌ Don't use heavily compressed JPEGs for training
- ❌ Don't use too high learning rate (causes divergence)
- ❌ Don't train on images smaller than patch_size × factor
- ❌ Don't ignore validation metrics
- ❌ Don't overtrain (causes overfitting)

## Troubleshooting

### Common Issues and Solutions

**Loss is NaN or increasing**:
- Reduce learning rate (try 1/10 of current)
- Check for corrupted images in dataset
- Reduce batch size

**Out of memory**:
- Reduce batch_size (try 1 or 2)
- Reduce patch_size (try 32)
- Reduce network width

**Training is too slow**:
- Increase batch_size if memory allows
- Reduce patch_size
- Use smaller network (width=16)
- Ensure release build: `cargo build --release`

**Poor quality results**:
- Train longer (20000+ steps)
- Use more training data
- Try different loss function
- Adjust learning rate

**Overfitting** (good on training, bad on new images):
- Use more diverse training data
- Add validation set
- Reduce learning rate
- Stop training earlier

## Example Training Runs

### Quick Test Model (1 hour)
```bash
srgan-rust train \
  -f 2 \        # 2x for faster training
  -w 16 \       # Small network
  -b 8 \        # Larger batch
  -p 32 \       # Small patches
  ./test_data \
  test_model.rsr
```

### High-Quality Model (overnight)
```bash
srgan-rust train \
  -s res/L1_x4_UCID_x1node.rsr \
  -f 4 \
  -w 32 \
  -b 4 \
  -p 48 \
  -R 0.001 \
  -v ./validation \
  ./training_data \
  high_quality.rsr
```

### Production Model (weekend)
```bash
# Create config file
cat > production.toml << EOF
[network]
factor = 4
width = 48
log_depth = 4

[training]
learning_rate = 0.0003
batch_size = 8
patch_size = 64
loss_type = "L1"
checkpoint_interval = 500

[data]
training_folder = "./full_dataset"
recurse = true

[validation]
validation_folder = "./validation"
frequency = 100
max_images = 100

[output]
parameter_file = "./production_model.rsr"
checkpoint_dir = "./checkpoints"
save_best_model = true
EOF

# Run training
srgan-rust train-config production.toml
```

## Additional Resources

- [SRGAN Paper](https://arxiv.org/abs/1609.04802) - Original research paper
- [Dataset Sources](#) - Where to find training data
- [Community Models](#) - Pre-trained models from the community

---

For more help, open an issue on [GitHub](https://github.com/yourusername/srgan-rust).