# srgan-rust
![LogoNN](docs/logo_nn.png)![LogoLin](docs/logo_lin.png)![Logo](docs/logo_rs.png)

A Rust implementation of SRGAN (Super-Resolution Generative Adversarial Networks), which when given a low resolution image utilises deep learning to infer the corresponding high resolution image. 
Use the included pre-trained neural networks to upscale your images, or easily train your own specialised neural network!  
Feel free to open an issue for general discussion or to raise any problems.  

## Usage
To upscale an image:  
`srgan-rust <INPUT_FILE> <OUTPUT_FILE>`  

PNG output files are recommended.

For further options:  
`srgan-rust --help`  
`srgan-rust train --help`  

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

## Note
Attemping to upscale images with significant noise or jpeg artefacts is likely to produce poor results. Input and output colorspace are nominally sRGB.

## License
MIT


## Notes

To train:
./srgan-rust train ./train ./test.rs