# srgan-rust
![LogoNN](docs/logo_nn.png)![LogoLin](docs/logo_lin.png)![Logo](docs/logo_rs.png)

A Rust implementation of SRGAN (Super-Resolution Generative Adversarial Networks), which when given a low resolution image utilises deep learning to infer the corresponding high resolution image. 
Use the included pre-trained neural networks to upscale your images, or easily train your own specialised neural network!  
Feel free to open an issue for general discussion or to raise any problems.  

## Quick Start

### Installation
```bash
# Build from source with optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/srgan-rust --help
```

### Basic Usage
```bash
# Upscale an image using default settings
srgan-rust input.jpg output.png

# Use anime-optimized model
srgan-rust -p anime anime.png anime_4x.png

# Custom upscaling factor
srgan-rust -f 2 photo.jpg photo_2x.png
```

For detailed usage instructions, see [USAGE.md](USAGE.md).  

## Setup
To get the rust compiler (rustc) use [rustup](https://rustup.rs). For best performance compile using environmental variable `RUSTFLAGS="-C target-cpu=native" ` and a release mode build `cargo build --release`.  
Or in one line: `cargo rustc --release -- -C target-cpu=native`.  

## Examples
Set14 Cartoon  
![CartoonLowRes](docs/cartoon_nn.png)![Cartoon](docs/cartoon_rsa.png)

Set14 Butterfly  
![ButterflyLowRes](docs/butterfly_nn.png)![Butterfly](docs/butterfly_rs.png)

Bank Lobby (test image for [Neural Enhance](https://github.com/alexjc/neural-enhance))  
CC-BY-SA @benarent  
![BankLowRes](docs/bank_nn.png)![Bank](docs/bank_rs.png)

## Features

- 🚀 Fast image upscaling using SRGAN neural networks
- 🎨 Pre-trained models for natural images and anime/artwork  
- 🔧 Train custom models on your own datasets
- 🔄 Import PyTorch, TensorFlow, ONNX, and Keras models
- 📊 PSNR calculation for quality metrics
- ✅ Comprehensive input validation and error handling
- 🧪 CI/CD pipeline with GitHub Actions
- 📖 Detailed documentation and usage examples

## Documentation

- [Usage Guide](USAGE.md) - Detailed usage instructions and examples
- [Model Converter](docs/MODEL_CONVERTER.md) - Import models from PyTorch, TensorFlow, ONNX, and Keras
- [Training Guide](USAGE.md#training-custom-models) - How to train custom models
- [Project Development](project_description.md) - Development progress and todo list

## Notes

- Best results with high-quality input images
- Attempting to upscale images with significant noise or JPEG artifacts may produce poor results
- Input and output colorspace are nominally sRGB
- PNG output format recommended for best quality

## License
MIT


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the SRGAN paper by Ledig et al.
- Uses the Alumina deep learning framework