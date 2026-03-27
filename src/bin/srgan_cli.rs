extern crate srgan_rust;

use std::path::Path;
use std::time::Instant;

use clap::{App, Arg, SubCommand};
use image::GenericImage;
use log::error;

use srgan_rust::error::{Result, SrganError};
use srgan_rust::psnr::psnr_calculation;
use srgan_rust::ssim::ssim_calculation;
use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use srgan_rust::web_server::{WebServer, ServerConfig};

fn main() {
    env_logger::init();

    let matches = App::new("srgan")
        .version("0.2.0")
        .about("SRGAN image upscaling CLI — local upscaling, benchmarking, and API server")
        .subcommand(
            SubCommand::with_name("upscale")
                .about("Upscale a single image locally (no API key needed)")
                .arg(
                    Arg::with_name("input")
                        .required(true)
                        .index(1)
                        .help("Input image path"),
                )
                .arg(
                    Arg::with_name("output")
                        .long("output")
                        .short("o")
                        .takes_value(true)
                        .help("Output image path (default: <input>_upscaled.<format>)"),
                )
                .arg(
                    Arg::with_name("output_pos")
                        .index(2)
                        .help("Output image path (positional, same as --output)"),
                )
                .arg(
                    Arg::with_name("scale")
                        .long("scale")
                        .takes_value(true)
                        .default_value("4x")
                        .possible_values(&["2x", "4x"])
                        .help("Scale factor (2x or 4x)"),
                )
                .arg(
                    Arg::with_name("model")
                        .long("model")
                        .takes_value(true)
                        .default_value("srgan")
                        .possible_values(&["srgan", "real-esrgan", "waifu2x-anime"])
                        .help("Model to use for upscaling"),
                )
                .arg(
                    Arg::with_name("format")
                        .long("format")
                        .takes_value(true)
                        .default_value("png")
                        .possible_values(&["png", "jpg", "webp"])
                        .help("Output image format"),
                )
                .arg(
                    Arg::with_name("quality")
                        .long("quality")
                        .takes_value(true)
                        .possible_values(&["psnr", "ssim"])
                        .help("Print a specific quality metric (psnr or ssim) after upscaling"),
                ),
        )
        .subcommand(
            SubCommand::with_name("benchmark")
                .about("Run all images in a directory through available models, print PSNR/SSIM comparison table")
                .arg(
                    Arg::with_name("dir")
                        .required(true)
                        .index(1)
                        .help("Directory containing images to benchmark"),
                ),
        )
        .subcommand(
            SubCommand::with_name("server")
                .about("Start the web API server")
                .arg(
                    Arg::with_name("host")
                        .long("host")
                        .takes_value(true)
                        .default_value("127.0.0.1")
                        .help("Host to bind to"),
                )
                .arg(
                    Arg::with_name("port")
                        .long("port")
                        .takes_value(true)
                        .default_value("8080")
                        .help("Port to listen on"),
                ),
        )
        .subcommand(
            SubCommand::with_name("info")
                .about("Show available models, their metadata, and hardware info"),
        )
        .get_matches();

    let result = match matches.subcommand() {
        ("upscale", Some(sub_m)) => run_upscale(sub_m),
        ("benchmark", Some(sub_m)) => run_benchmark(sub_m),
        ("server", Some(sub_m)) => run_server(sub_m),
        ("info", Some(_)) => run_info(),
        _ => {
            eprintln!("Usage: srgan <subcommand> [options]");
            eprintln!("Subcommands: upscale, benchmark, server, info");
            eprintln!("Run `srgan --help` for details.");
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        error!("{}", e);
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

// ─── upscale ─────────────────────────────────────────────────────────────────

fn run_upscale(matches: &clap::ArgMatches) -> Result<()> {
    let input = matches.value_of("input").unwrap();
    let out_format = matches.value_of("format").unwrap_or("png");

    // Output path: --output flag > positional arg > auto-generated
    let output_owned: String;
    let output = if let Some(o) = matches.value_of("output").or(matches.value_of("output_pos")) {
        o
    } else {
        let stem = Path::new(input)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let ext = match out_format {
            "jpg" | "jpeg" => "jpg",
            "webp" => "webp",
            _ => "png",
        };
        output_owned = format!("{}_upscaled.{}", stem, ext);
        &output_owned
    };

    let scale_str = matches.value_of("scale").unwrap_or("4x");
    let scale: usize = scale_str.trim_end_matches('x').parse().unwrap_or(4);

    let model_label = match matches.value_of("model").unwrap_or("srgan") {
        "srgan" => "natural",
        "waifu2x" | "waifu2x-anime" => "waifu2x",
        "real-esrgan" => "real-esrgan",
        other => other,
    };
    let quality_metric = matches.value_of("quality"); // None, Some("psnr"), or Some("ssim")

    let input_path = Path::new(input);
    if !input_path.exists() {
        return Err(SrganError::InvalidInput(format!(
            "Input file does not exist: {}",
            input
        )));
    }

    println!("Loading {} model ({}x)...", matches.value_of("model").unwrap_or("srgan"), scale);
    let network = ThreadSafeNetwork::from_label(model_label, Some(scale)).map_err(|e| {
        SrganError::InvalidInput(format!("Failed to load model '{}': {}", model_label, e))
    })?;

    println!("Opening {}...", input);
    let img = image::open(input_path).map_err(SrganError::Image)?;
    let (in_w, in_h) = (img.width(), img.height());

    println!("Input: {}x{}", in_w, in_h);
    println!("Upscaling...");

    let start = Instant::now();
    let upscaled = network.upscale_image(&img)?;
    let elapsed = start.elapsed();

    let (out_w, out_h) = (upscaled.width(), upscaled.height());

    // Encode with the requested format
    match out_format {
        "jpg" | "jpeg" => {
            let rgb = upscaled.to_rgb();
            let fout = std::fs::File::create(output).map_err(SrganError::Io)?;
            let mut bw = std::io::BufWriter::new(fout);
            image::jpeg::JPEGEncoder::new_with_quality(&mut bw, 90)
                .encode(rgb.as_ref(), out_w, out_h, image::ColorType::RGB(8))
                .map_err(SrganError::Io)?;
        }
        _ => {
            // PNG and WebP (WebP falls back to PNG with image 0.19)
            upscaled.save(output).map_err(SrganError::Io)?;
        }
    }

    println!("Output: {}x{} ({})", out_w, out_h, out_format);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Saved to {}", output);

    if let Some(metric) = quality_metric {
        let orig_tensor = srgan_rust::image_to_data(&img);
        let downscaled_back = upscaled.resize_exact(in_w, in_h, image::FilterType::Lanczos3);
        let down_tensor = srgan_rust::image_to_data(&downscaled_back);

        match metric {
            "psnr" => {
                let (rgb_err, _luma_err, pix) =
                    psnr_calculation(orig_tensor.view(), down_tensor.view());
                let psnr = if pix > 0.0 && rgb_err > 0.0 {
                    -10.0 * (rgb_err / pix).log10()
                } else {
                    f32::INFINITY
                };
                println!("PSNR: {:.2} dB", psnr);
            }
            "ssim" => {
                let ssim = ssim_calculation(orig_tensor.view(), down_tensor.view());
                println!("SSIM: {:.4}", ssim);
            }
            _ => {}
        }
    }

    Ok(())
}

// ─── benchmark ───────────────────────────────────────────────────────────────

fn run_benchmark(matches: &clap::ArgMatches) -> Result<()> {
    let dir = matches.value_of("dir").unwrap();
    let dir_path = Path::new(dir);

    if !dir_path.is_dir() {
        return Err(SrganError::InvalidInput(format!(
            "Not a directory: {}",
            dir
        )));
    }

    // Collect image files
    let valid_ext = ["png", "jpg", "jpeg", "webp"];
    let mut files: Vec<_> = std::fs::read_dir(dir_path)
        .map_err(SrganError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| valid_ext.contains(&e.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    if files.is_empty() {
        println!("No image files found in {}", dir);
        return Ok(());
    }

    println!("Found {} image(s) in {}", files.len(), dir);
    println!();

    // Models to benchmark
    let models: &[(&str, &str)] = &[
        ("srgan", "natural"),
        ("real-esrgan", "real-esrgan"),
        ("waifu2x", "waifu2x"),
    ];

    // Load all networks up-front
    let mut networks: Vec<(&str, ThreadSafeNetwork)> = Vec::new();
    for &(display_name, label) in models {
        match ThreadSafeNetwork::from_label(label, Some(4)) {
            Ok(net) => networks.push((display_name, net)),
            Err(e) => eprintln!("Warning: could not load model '{}': {}", display_name, e),
        }
    }

    if networks.is_empty() {
        return Err(SrganError::InvalidInput("No models could be loaded".into()));
    }

    // Print table header
    print!("{:<30}", "Image");
    for &(name, _) in &networks {
        print!("  {:>10} {:>6}", format!("{}/PSNR", name), "SSIM");
    }
    println!();
    print!("{:-<30}", "");
    for _ in &networks {
        print!("  {:-<10} {:-<6}", "", "");
    }
    println!();

    // Per-model accumulators for averages
    let mut psnr_sums: Vec<f64> = vec![0.0; networks.len()];
    let mut ssim_sums: Vec<f64> = vec![0.0; networks.len()];
    let mut count: usize = 0;

    for file in &files {
        let img = match image::open(file) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  Skipping {}: {}", file.display(), e);
                continue;
            }
        };

        let (orig_w, orig_h) = (img.width(), img.height());
        let orig_tensor = srgan_rust::image_to_data(&img);
        let fname = file.file_name().unwrap_or_default().to_string_lossy();
        let display_name = if fname.len() > 28 {
            format!("..{}", &fname[fname.len() - 26..])
        } else {
            fname.to_string()
        };
        print!("{:<30}", display_name);

        for (idx, (_name, net)) in networks.iter().enumerate() {
            match net.upscale_image(&img) {
                Ok(upscaled) => {
                    let downscaled =
                        upscaled.resize_exact(orig_w, orig_h, image::FilterType::Lanczos3);
                    let down_tensor = srgan_rust::image_to_data(&downscaled);

                    let (rgb_err, _luma_err, pix) =
                        psnr_calculation(orig_tensor.view(), down_tensor.view());
                    let psnr = if pix > 0.0 && rgb_err > 0.0 {
                        -10.0 * (rgb_err / pix).log10()
                    } else {
                        f32::INFINITY
                    };
                    let ssim = ssim_calculation(orig_tensor.view(), down_tensor.view());

                    print!("  {:>10.2} {:>6.4}", psnr, ssim);
                    psnr_sums[idx] += psnr as f64;
                    ssim_sums[idx] += ssim as f64;
                }
                Err(e) => {
                    print!("  {:>10} {:>6}", "err", "err");
                    eprintln!("\n  Error processing {} with model: {}", fname, e);
                }
            }
        }
        println!();
        count += 1;
    }

    // Averages
    if count > 0 {
        print!("{:-<30}", "");
        for _ in &networks {
            print!("  {:-<10} {:-<6}", "", "");
        }
        println!();
        print!("{:<30}", "AVERAGE");
        for idx in 0..networks.len() {
            let avg_psnr = psnr_sums[idx] / count as f64;
            let avg_ssim = ssim_sums[idx] / count as f64;
            print!("  {:>10.2} {:>6.4}", avg_psnr, avg_ssim);
        }
        println!();
    }

    Ok(())
}

// ─── server ──────────────────────────────────────────────────────────────────

fn run_server(matches: &clap::ArgMatches) -> Result<()> {
    let mut config = ServerConfig::default();

    if let Some(host) = matches.value_of("host") {
        config.host = host.to_string();
    }
    if let Some(port) = matches.value_of("port") {
        config.port = port
            .parse()
            .map_err(|_| SrganError::InvalidInput("Invalid port number".into()))?;
    }

    println!("Starting SRGAN web server on {}:{}...", config.host, config.port);

    let server = WebServer::new(config)?;
    server.start()?;

    Ok(())
}

// ─── info ────────────────────────────────────────────────────────────────────

fn run_info() -> Result<()> {
    println!("SRGAN-Rust v{}", env!("CARGO_PKG_VERSION"));
    println!();

    // Models
    println!("Available models:");
    println!("  {:<16} {}", "srgan", "Neural net trained on natural images (L1 loss, 4x)");
    println!("  {:<16} {}", "real-esrgan", "Real-ESRGAN RRDB (23 blocks, sub-pixel upsampling, 4x/2x)");
    println!("  {:<16} {}", "waifu2x-anime", "Anime/illustration optimised (Lanczos3 + unsharp, 4x)");
    println!();
    println!("Output formats: png, jpg (quality 85-100), webp");
    println!();

    // Built-in model metadata
    println!("Built-in model details:");
    match srgan_rust::network_from_bytes(srgan_rust::L1_SRGB_NATURAL_PARAMS) {
        Ok(desc) => {
            println!("  srgan (natural):");
            println!("    Factor:       {}x", desc.factor);
            println!("    Width:        {}", desc.width);
            println!("    Log depth:    {}", desc.log_depth);
            println!("    Node factor:  {}", desc.global_node_factor);
            println!("    Parameters:   {} tensors", desc.parameters.len());
        }
        Err(e) => println!("  srgan: failed to load metadata: {}", e),
    }
    match srgan_rust::network_from_bytes(srgan_rust::L1_SRGB_ANIME_PARAMS) {
        Ok(desc) => {
            println!("  anime:");
            println!("    Factor:       {}x", desc.factor);
            println!("    Width:        {}", desc.width);
            println!("    Log depth:    {}", desc.log_depth);
            println!("    Node factor:  {}", desc.global_node_factor);
            println!("    Parameters:   {} tensors", desc.parameters.len());
        }
        Err(e) => println!("  anime: failed to load metadata: {}", e),
    }
    println!();

    // Hardware
    println!("Hardware:");
    println!("  CPUs:     {}", num_cpus());
    if let Ok(info) = sys_info::mem_info() {
        println!("  RAM:      {} MB total", info.total / 1024);
    }
    if let Ok(os) = sys_info::os_type() {
        let release = sys_info::os_release().unwrap_or_default();
        println!("  OS:       {} {}", os, release);
    }
    if let Ok(host) = sys_info::hostname() {
        println!("  Hostname: {}", host);
    }

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
