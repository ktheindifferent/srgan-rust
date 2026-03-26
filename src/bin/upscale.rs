extern crate srgan_rust;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::{App, Arg};
use glob::glob;
use image::GenericImage;
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, warn};
use rayon::prelude::*;

use srgan_rust::commands::batch::collect_image_files;
use srgan_rust::error::{Result, SrganError};
use srgan_rust::psnr::psnr_calculation;
use srgan_rust::ssim::ssim_calculation;
use srgan_rust::thread_safe_network::ThreadSafeNetwork;

fn main() {
    env_logger::init();

    let matches = App::new("srgan-upscale")
        .version("0.2.0")
        .about("Batch image upscaling CLI tool")
        .arg(
            Arg::with_name("input")
                .long("input")
                .short("i")
                .takes_value(true)
                .required(true)
                .help("Input file, directory, or glob pattern (e.g. \"*.jpg\")"),
        )
        .arg(
            Arg::with_name("output")
                .long("output")
                .short("o")
                .takes_value(true)
                .required(true)
                .help("Output directory"),
        )
        .arg(
            Arg::with_name("scale")
                .long("scale")
                .short("s")
                .takes_value(true)
                .default_value("4")
                .possible_values(&["2", "4", "8"])
                .help("Scale factor"),
        )
        .arg(
            Arg::with_name("model")
                .long("model")
                .short("m")
                .takes_value(true)
                .default_value("srgan")
                .possible_values(&["srgan", "waifu2x"])
                .help("Model to use for upscaling"),
        )
        .arg(
            Arg::with_name("quality")
                .long("quality")
                .short("q")
                .help("Enable PSNR/SSIM quality comparison (requires keeping originals in memory)"),
        )
        .arg(
            Arg::with_name("recursive")
                .long("recursive")
                .short("r")
                .help("Recurse into subdirectories"),
        )
        .arg(
            Arg::with_name("concurrency")
                .long("concurrency")
                .short("j")
                .takes_value(true)
                .help("Number of parallel jobs (default: CPU count)"),
        )
        .get_matches();

    if let Err(e) = run(&matches) {
        error!("{}", e);
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(matches: &clap::ArgMatches) -> Result<()> {
    let input = matches.value_of("input").unwrap();
    let output_dir = matches.value_of("output").unwrap();
    let scale: usize = matches.value_of("scale").unwrap().parse().unwrap();
    let model_label = match matches.value_of("model").unwrap() {
        "srgan" => "natural",
        "waifu2x" => "waifu2x",
        other => other,
    };
    let quality = matches.is_present("quality");
    let recursive = matches.is_present("recursive");
    let concurrency: Option<usize> = matches
        .value_of("concurrency")
        .and_then(|s| s.parse().ok());

    // Set up thread pool
    if let Some(n) = concurrency {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .unwrap_or_else(|e| warn!("Failed to set thread pool size: {}", e));
    }

    // Create output directory
    let output_path = Path::new(output_dir);
    std::fs::create_dir_all(output_path)
        .map_err(|e| SrganError::Io(e))?;

    // Collect input files (supports glob patterns like "*.jpg", "photos/**/*.png")
    let input_path = Path::new(input);
    let image_files = if input_path.is_file() {
        if is_supported_image(input_path) {
            vec![input_path.to_path_buf()]
        } else {
            return Err(SrganError::InvalidInput(format!(
                "Unsupported image format: {}",
                input
            )));
        }
    } else if input_path.is_dir() {
        collect_image_files(input_path, "", recursive)?
    } else if input.contains('*') || input.contains('?') || input.contains('[') {
        // Treat as glob pattern
        let mut files: Vec<PathBuf> = Vec::new();
        match glob(input) {
            Ok(paths) => {
                for entry in paths {
                    match entry {
                        Ok(path) if path.is_file() && is_supported_image(&path) => {
                            files.push(path);
                        }
                        Ok(_) => {} // skip non-image files
                        Err(e) => warn!("Glob entry error: {}", e),
                    }
                }
            }
            Err(e) => {
                return Err(SrganError::InvalidInput(format!(
                    "Invalid glob pattern '{}': {}",
                    input, e
                )));
            }
        }
        if files.is_empty() {
            return Err(SrganError::InvalidInput(format!(
                "No image files matched the pattern: {}",
                input
            )));
        }
        files
    } else {
        return Err(SrganError::InvalidInput(format!(
            "Input path does not exist: {}",
            input
        )));
    };

    if image_files.is_empty() {
        println!("No supported image files found.");
        return Ok(());
    }

    println!(
        "Upscaling {} file(s) with {} model at {}x scale",
        image_files.len(),
        matches.value_of("model").unwrap(),
        scale
    );

    // Load network
    let network = Arc::new(ThreadSafeNetwork::from_label(model_label, Some(scale)).map_err(|e| {
        eprintln!("Error: Failed to load '{}' model weights.", model_label);
        eprintln!();
        eprintln!("The built-in models (srgan/waifu2x) ship with the binary.");
        eprintln!("If you see this error, the binary may have been built incorrectly");
        eprintln!("or model data is corrupt.");
        eprintln!();
        eprintln!("To rebuild with embedded weights:");
        eprintln!("  cargo build --release --bin srgan-upscale");
        eprintln!();
        eprintln!("For custom models, use the main srgan-rust binary:");
        eprintln!("  srgan-rust -c /path/to/model.rsr input.jpg output.png");
        e
    })?);

    // Progress bar
    let pb = ProgressBar::new(image_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
    );

    let start_time = Instant::now();
    let successful = Arc::new(AtomicUsize::new(0));
    let failed_count = Arc::new(AtomicUsize::new(0));
    let total_output_bytes = Arc::new(AtomicUsize::new(0));
    let psnr_sum = Arc::new(Mutex::new(0.0f64));
    let ssim_sum = Arc::new(Mutex::new(0.0f64));
    let quality_count = Arc::new(AtomicUsize::new(0));
    let errors: Arc<Mutex<Vec<(PathBuf, String)>>> = Arc::new(Mutex::new(Vec::new()));

    let input_base = if input_path.is_dir() {
        input_path
    } else {
        input_path.parent().unwrap_or(Path::new("."))
    };

    image_files.par_iter().for_each(|file| {
        let relative = file.strip_prefix(input_base).unwrap_or(file);
        let out_file = output_path.join(relative).with_extension("png");

        // Create parent dirs
        if let Some(parent) = out_file.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                if let Ok(mut errs) = errors.lock() {
                    errs.push((file.clone(), format!("mkdir: {}", e)));
                }
                failed_count.fetch_add(1, Ordering::Relaxed);
                pb.inc(1);
                return;
            }
        }

        pb.set_message(format!("{}", relative.display()));

        match process_file(file, &out_file, &network, quality) {
            Ok(metrics) => {
                successful.fetch_add(1, Ordering::Relaxed);
                if let Ok(meta) = std::fs::metadata(&out_file) {
                    total_output_bytes.fetch_add(meta.len() as usize, Ordering::Relaxed);
                }
                if let Some((psnr, ssim)) = metrics {
                    if let Ok(mut p) = psnr_sum.lock() {
                        *p += psnr as f64;
                    }
                    if let Ok(mut s) = ssim_sum.lock() {
                        *s += ssim as f64;
                    }
                    quality_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(e) => {
                warn!("Failed to process {}: {}", file.display(), e);
                if let Ok(mut errs) = errors.lock() {
                    errs.push((file.clone(), e.to_string()));
                }
                failed_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        pb.inc(1);
    });

    pb.finish_and_clear();

    // Summary
    let elapsed = start_time.elapsed();
    let ok = successful.load(Ordering::Relaxed);
    let fail = failed_count.load(Ordering::Relaxed);
    let out_bytes = total_output_bytes.load(Ordering::Relaxed);

    println!("\n--- Summary ---");
    println!("Files processed: {}/{}", ok, image_files.len());
    if fail > 0 {
        println!("Files failed:    {}", fail);
    }
    println!("Total time:      {:.2}s", elapsed.as_secs_f64());
    if ok > 0 {
        println!(
            "Avg time/image:  {:.2}s",
            elapsed.as_secs_f64() / ok as f64
        );
    }
    println!("Output size:     {}", format_bytes(out_bytes));

    if quality {
        let qc = quality_count.load(Ordering::Relaxed);
        if qc > 0 {
            let avg_psnr = *psnr_sum.lock().unwrap_or_else(|e| e.into_inner()) / qc as f64;
            let avg_ssim = *ssim_sum.lock().unwrap_or_else(|e| e.into_inner()) / qc as f64;
            println!("Avg PSNR:        {:.2} dB", avg_psnr);
            println!("Avg SSIM:        {:.4}", avg_ssim);
        }
    }

    // Print errors
    if let Ok(errs) = errors.lock() {
        if !errs.is_empty() {
            eprintln!("\nErrors:");
            for (path, msg) in errs.iter() {
                eprintln!("  {}: {}", path.display(), msg);
            }
        }
    }

    Ok(())
}

fn process_file(
    input: &Path,
    output: &Path,
    network: &ThreadSafeNetwork,
    compute_quality: bool,
) -> Result<Option<(f32, f32)>> {
    let img = image::open(input).map_err(SrganError::Image)?;
    let upscaled = network.upscale_image(&img)?;
    upscaled.save(output).map_err(SrganError::Io)?;

    if compute_quality {
        // Compare: downscale the upscaled image back to original size, then measure
        // PSNR/SSIM against the original. This gives a round-trip quality metric.
        let orig_tensor = srgan_rust::image_to_data(&img);

        // Downscale the upscaled result back to original dims and compare
        let (orig_w, orig_h) = (img.width(), img.height());
        let downscaled_back = upscaled.resize_exact(orig_w, orig_h, image::FilterType::Lanczos3);
        let down_tensor = srgan_rust::image_to_data(&downscaled_back);

        let (rgb_err, _luma_err, pix) = psnr_calculation(orig_tensor.view(), down_tensor.view());
        let psnr = if pix > 0.0 && rgb_err > 0.0 {
            -10.0 * (rgb_err / pix).log10()
        } else {
            f32::INFINITY
        };

        let ssim = ssim_calculation(orig_tensor.view(), down_tensor.view());

        Ok(Some((psnr, ssim)))
    } else {
        Ok(None)
    }
}

fn is_supported_image(path: &Path) -> bool {
    let valid = ["png", "jpg", "jpeg", "webp"];
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| valid.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
