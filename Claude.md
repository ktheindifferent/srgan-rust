# SRGAN-Rust Codebase Guide

## Project Overview
SRGAN-Rust is a Rust implementation of Super-Resolution Generative Adversarial Networks (SRGAN) for image upscaling using deep learning techniques. The project provides both pre-trained neural networks for immediate use and the ability to train custom models on specialized datasets.

## Core Capabilities
- **Image Upscaling**: Neural network-based upscaling of low-resolution images to high-resolution
- **Video Processing**: Frame-by-frame video upscaling with codec support
- **Model Training**: Custom model training with configurable parameters
- **Batch Processing**: Parallel processing of multiple images
- **Web API Server**: RESTful API for remote upscaling services
- **Benchmarking**: Performance testing and model comparison tools
- **Memory Profiling**: Memory usage analysis and optimization tools

## Technology Stack
```toml
[core]
language = "Rust (Edition 2018)"
version = "0.2.0"

[dependencies]
deep_learning = "alumina 0.3.0"
image_processing = "image 0.19"
linear_algebra = "ndarray 0.11.2"
cli = "clap 2"
serialization = ["serde 1", "bincode 1.0.1", "serde_json 1.0", "toml 0.5"]
compression = "xz2 0.1"
parallelization = "rayon 1.5"
logging = ["log 0.4", "env_logger 0.10", "indicatif 0.17"]
utilities = ["chrono 0.4", "lazy_static 1.4", "ctrlc 3.4"]
```

## Project Structure
```
srgan-rust/
├── src/
│   ├── main.rs                 # Application entry point
│   ├── cli.rs                  # CLI argument parsing and configuration
│   ├── lib.rs                  # Core library exports
│   ├── commands/               # Command implementations
│   │   ├── mod.rs             # Command module exports
│   │   ├── upscale.rs         # Standard CPU upscaling
│   │   ├── upscale_gpu.rs     # GPU-accelerated upscaling
│   │   ├── train.rs           # Model training
│   │   ├── train_prescaled.rs # Pre-scaled training
│   │   ├── batch.rs           # Batch processing
│   │   ├── benchmark.rs       # Performance benchmarking
│   │   ├── downscale.rs       # Image downscaling
│   │   ├── psnr.rs            # PSNR calculation
│   │   ├── quantise.rs        # Model quantization
│   │   ├── set_width.rs       # Image resizing
│   │   ├── video.rs           # Video processing
│   │   ├── server.rs          # Web server mode
│   │   ├── convert_model.rs   # Model format conversion
│   │   ├── generate_config.rs # Config file generation
│   │   └── profile_memory.rs  # Memory profiling
│   ├── training/              # Training infrastructure
│   │   ├── mod.rs            # Training module exports
│   │   ├── trainer_simple.rs # Training implementation
│   │   ├── data_loader.rs    # Dataset loading
│   │   ├── checkpoint.rs     # Checkpoint management
│   │   └── validation_simple.rs # Validation logic
│   ├── utils/                 # Utility functions
│   │   ├── mod.rs            # Utils module exports
│   │   ├── command_helpers.rs # Command utilities
│   │   ├── error_helpers.rs  # Error handling utilities
│   │   └── file_io.rs        # File I/O utilities
│   ├── res/                   # Pre-trained models
│   │   ├── L1_x4_Anime_x1node.rsr # Anime-optimized model
│   │   └── L1_x4_UCID_x1node.rsr  # Natural image model
│   ├── network.rs             # Neural network architecture
│   ├── aligned_crop.rs        # Image cropping utilities
│   ├── config.rs              # Configuration structures
│   ├── config_file.rs         # Config file parsing
│   ├── constants.rs           # Application constants
│   ├── error.rs               # Error types and handling
│   ├── gpu.rs                 # GPU acceleration layer
│   ├── logging.rs             # Logging and progress tracking
│   ├── model_converter.rs     # Model format conversion
│   ├── profiling.rs           # Memory profiling
│   ├── psnr.rs               # PSNR calculation
│   ├── validation.rs         # Input validation
│   ├── video.rs              # Video processing
│   └── web_server.rs         # Web API server
├── tests/                     # Test suite
│   ├── lib.rs                # Test module exports
│   ├── integration_tests.rs  # End-to-end tests
│   ├── batch_tests.rs        # Batch processing tests
│   ├── benchmark_tests.rs    # Benchmarking tests
│   ├── config_tests.rs       # Configuration tests
│   ├── error_tests.rs        # Error handling tests
│   ├── model_converter_tests.rs # Model conversion tests
│   ├── profiling_tests.rs    # Profiling tests
│   ├── psnr_tests.rs         # PSNR calculation tests
│   ├── video_tests.rs        # Video processing tests
│   └── web_server_tests.rs   # Web server tests
├── examples/                  # Example scripts and configs
│   ├── benchmark.sh          # Benchmarking script
│   ├── batch_process.sh      # Batch processing script
│   ├── create_training_dataset.py # Dataset creation tool
│   ├── download_datasets.sh  # Dataset download script
│   └── training_config.toml  # Example training config
├── docs/                      # Documentation and images
│   └── [sample images]       # Example upscaling results
├── train/                     # Sample training images
└── [configuration files]      # Docker, Cargo, formatting configs
```

## Available Commands

### Primary Commands
- `upscale` - Upscale an image using CPU (default command)
- `upscale-gpu` - GPU-accelerated image upscaling
- `train` - Train a new model from scratch
- `train_prescaled` - Train using pre-scaled images
- `batch` - Process multiple images in batch

### Utility Commands
- `benchmark` - Run performance benchmarks
- `downscale` - Downscale images
- `psnr` - Calculate Peak Signal-to-Noise Ratio
- `quantise` - Quantize model weights
- `set_width` - Resize image to specific width
- `video` - Process video files
- `server` - Start web API server
- `convert-model` - Convert models between formats
- `generate-config` - Generate configuration templates
- `profile-memory` - Profile memory usage
- `analyze-memory` - Analyze memory of any command
- `list-gpus` - List available GPU devices

## Key Features Implementation

### Error Handling
- Comprehensive error types in `error.rs`
- Result-based error propagation throughout
- User-friendly error messages
- Graceful recovery mechanisms

### Performance Optimizations
- Parallel processing with Rayon
- GPU acceleration support (foundation complete)
- Memory-efficient streaming for large datasets
- Optimized tensor operations with ndarray

### Testing Strategy
- Unit tests for core functionality
- Integration tests for CLI commands
- Performance benchmarks
- Memory profiling tests

### Configuration Management
- TOML/JSON config file support
- Command-line argument parsing
- Environment variable support
- Default configurations with overrides

## Development Guidelines

### Building the Project
```bash
# Development build
cargo build

# Optimized release build
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run tests
cargo test

# Run with verbose logging
RUST_LOG=debug ./target/release/srgan-rust [command]
```

### Common Development Tasks

#### Adding a New Command
1. Create command module in `src/commands/`
2. Implement command logic with proper error handling
3. Add CLI argument parsing in `cli.rs`
4. Wire command in `main.rs`
5. Add tests in `tests/`

#### Training a Model
```bash
# Generate training config
./srgan-rust generate-config --type training > my_config.toml

# Start training
./srgan-rust train --config my_config.toml

# Monitor with tensorboard (if implemented)
tensorboard --logdir=./logs
```

#### Processing Images
```bash
# Single image
./srgan-rust input.jpg output.png

# Batch processing
./srgan-rust batch ./input_dir ./output_dir --recursive

# With specific model
./srgan-rust -p anime image.jpg output.png
```

## Critical Areas for Attention

### Known Issues
1. **Unwrap Calls**: 45+ unwrap() calls that need proper error handling
2. **Memory Management**: Potential memory leaks in long-running operations
3. **GPU Integration**: Kernel implementation pending for full GPU support

### Performance Bottlenecks
- Large image loading in memory
- Sequential processing in some commands
- Model loading overhead for batch operations

### Security Considerations
- Input validation for file paths
- Web server authentication (API keys)
- Rate limiting for API endpoints
- CORS configuration for web access

## Testing Commands
```bash
# Run all tests
cargo test

# Run specific test module
cargo test --test integration_tests

# Run with coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html

# Benchmark performance
./srgan-rust benchmark --input test.jpg --iterations 10
```

## Deployment

### Docker
```bash
# Build image
docker build -t srgan-rust .

# Run container
docker run -v $(pwd)/images:/images srgan-rust upscale /images/input.jpg /images/output.png
```

### Web Server Mode
```bash
# Start server
./srgan-rust server --port 8080 --workers 4

# Test endpoint
curl -X POST -F "image=@test.jpg" http://localhost:8080/upscale
```

## Recent Enhancements (Sessions 1-8)

### Completed
- ✅ Comprehensive error handling system
- ✅ Unit and integration test suite
- ✅ Input validation framework
- ✅ Docker containerization
- ✅ Logging and progress tracking
- ✅ Batch processing capabilities
- ✅ Configuration file support
- ✅ Benchmarking infrastructure
- ✅ GPU acceleration foundation
- ✅ Memory profiling tools
- ✅ Model conversion utilities
- ✅ Video processing support
- ✅ Web API server
- ✅ Critical fixes documentation

### In Progress
- ⚠️ CI/CD pipeline (blocked by permissions)
- ⚠️ GPU kernel implementation
- ⚠️ Remaining unwrap() fixes

## Quick Reference

### Environment Variables
```bash
RUST_LOG=debug              # Logging level
SRGAN_MODEL_PATH=/path      # Custom model directory
SRGAN_CACHE_DIR=/cache      # Cache directory
SRGAN_MAX_WORKERS=8         # Max parallel workers
```

### Configuration Files
- `training_config.toml` - Training parameters
- `rustfmt.toml` - Code formatting rules
- `Cargo.toml` - Dependencies and metadata
- `docker-compose.yml` - Docker services

### Important Files
- `USAGE.md` - Detailed usage guide
- `TRAINING_GUIDE.md` - Model training documentation
- `CRITICAL_FIXES_GUIDE.md` - Production stability fixes
- `ENHANCEMENT_PLAN.md` - Future development roadmap
- `project_description.md` - Development progress log

## Contact and Support
- GitHub Issues: Report bugs and request features
- Documentation: See docs/ directory
- Examples: See examples/ directory

---
*Last Updated: Current Session*
*Version: 0.2.0*