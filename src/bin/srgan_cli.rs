extern crate srgan_rust;

use std::path::Path;
use std::time::Instant;

use clap::{App, Arg, SubCommand};
use image::GenericImage;
use log::error;

use srgan_rust::error::{Result, SrganError};
use srgan_rust::thread_safe_network::ThreadSafeNetwork;

fn main() {
    env_logger::init();

    let matches = App::new("srgan")
        .version("0.2.0")
        .about("SRGAN image upscaling CLI")
        .subcommand(
            SubCommand::with_name("upscale")
                .about("Upscale an image locally without the API server")
                .arg(
                    Arg::with_name("input")
                        .required(true)
                        .index(1)
                        .help("Input image path"),
                )
                .arg(
                    Arg::with_name("output")
                        .required(true)
                        .index(2)
                        .help("Output image path"),
                )
                .arg(
                    Arg::with_name("scale")
                        .long("scale")
                        .takes_value(true)
                        .default_value("4")
                        .possible_values(&["2", "4"])
                        .help("Scale factor"),
                )
                .arg(
                    Arg::with_name("model")
                        .long("model")
                        .takes_value(true)
                        .default_value("srgan")
                        .possible_values(&["srgan", "waifu2x"])
                        .help("Model to use for upscaling"),
                )
                .arg(
                    Arg::with_name("quality")
                        .long("quality")
                        .help("Print PSNR/SSIM quality metrics after upscaling"),
                ),
        )
        .get_matches();

    let result = match matches.subcommand() {
        ("upscale", Some(sub_m)) => run_upscale(sub_m),
        _ => {
            eprintln!("Usage: srgan upscale <input> <output> [--scale 2|4] [--model srgan|waifu2x] [--quality]");
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        error!("{}", e);
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_upscale(matches: &clap::ArgMatches) -> Result<()> {
    let input = matches.value_of("input").unwrap();
    let output = matches.value_of("output").unwrap();
    let scale: usize = matches.value_of("scale").unwrap().parse().unwrap();
    let model_label = match matches.value_of("model").unwrap() {
        "srgan" => "natural",
        "waifu2x" => "waifu2x",
        other => other,
    };
    let quality = matches.is_present("quality");

    let input_path = Path::new(input);
    if !input_path.exists() {
        return Err(SrganError::InvalidInput(format!(
            "Input file does not exist: {}",
            input
        )));
    }

    println!("Loading {} model ({}x)...", matches.value_of("model").unwrap(), scale);
    let network = ThreadSafeNetwork::from_label(model_label, Some(scale)).map_err(|e| {
        SrganError::InvalidInput(format!("Failed to load model '{}': {}", model_label, e))
    })?;

    println!("Opening {}...", input);
    let img = image::open(input_path).map_err(SrganError::Image)?;
    let (in_w, in_h) = (img.width(), img.height());
    let in_file_size = std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);

    println!("Input: {}x{} ({} bytes)", in_w, in_h, in_file_size);
    println!("Upscaling...");

    let start = Instant::now();
    let upscaled = network.upscale_image(&img)?;
    let elapsed = start.elapsed();

    let (out_w, out_h) = (upscaled.width(), upscaled.height());

    upscaled
        .save(output)
        .map_err(SrganError::Io)?;

    let out_file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);

    println!("Output: {}x{} ({} bytes)", out_w, out_h, out_file_size);
    println!("Scale factor: {}x", scale);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Saved to {}", output);

    if quality {
        let orig_tensor = srgan_rust::image_to_data(&img);
        let downscaled_back = upscaled.resize_exact(in_w, in_h, image::FilterType::Lanczos3);
        let down_tensor = srgan_rust::image_to_data(&downscaled_back);

        let (rgb_err, _luma_err, pix) =
            srgan_rust::psnr::psnr_calculation(orig_tensor.view(), down_tensor.view());
        let psnr = if pix > 0.0 && rgb_err > 0.0 {
            -10.0 * (rgb_err / pix).log10()
        } else {
            f32::INFINITY
        };
        let ssim = srgan_rust::ssim::ssim_calculation(orig_tensor.view(), down_tensor.view());

        println!("PSNR: {:.2} dB", psnr);
        println!("SSIM: {:.4}", ssim);
    }

    Ok(())
}
