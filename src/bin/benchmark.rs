//! Model benchmarking binary.
//!
//! Runs a sample image through all available model variants (srgan, real-esrgan,
//! waifu2x-anime, waifu2x-photo), measures inference time, and computes
//! PSNR/SSIM quality metrics against the original high-res image.
//!
//! Usage:
//!   cargo run --bin srgan-benchmark -- --input sample.png [--reference hr.png]

use std::path::PathBuf;
use std::time::Instant;

use image::GenericImage;

use srgan_rust::UpscalingNetwork;
use srgan_rust::quality::compute_quality;

/// Models to benchmark.
const MODEL_LABELS: &[&str] = &[
    "natural",
    "anime",
    "real-esrgan",
    "real-esrgan-anime",
    "waifu2x-anime",
    "waifu2x-photo",
];

struct BenchmarkEntry {
    model: String,
    load_time_ms: f64,
    inference_time_ms: f64,
    psnr_db: Option<f32>,
    psnr_luma_db: Option<f32>,
    ssim: Option<f32>,
    status: String,
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let input_path = parse_arg(&args, "--input")
        .expect("Usage: srgan-benchmark --input <image> [--reference <hr_image>]");
    let reference_path = parse_arg(&args, "--reference");

    let input_path = PathBuf::from(&input_path);
    if !input_path.exists() {
        eprintln!("Error: input file not found: {}", input_path.display());
        std::process::exit(1);
    }

    // Load input image
    let input_image = image::open(&input_path).expect("Failed to load input image");
    let input_tensor = alumina::data::image_folder::image_to_data(&input_image);
    let shape = input_tensor.shape().to_vec();
    let input_4d = input_tensor
        .into_shape(ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]]))
        .expect("Failed to reshape input tensor");

    // Load reference image if provided (for quality metrics)
    let reference_tensor = reference_path.as_ref().map(|p| {
        let ref_path = PathBuf::from(p);
        if !ref_path.exists() {
            eprintln!("Warning: reference file not found: {}", ref_path.display());
            return None;
        }
        let ref_img = image::open(&ref_path).ok()?;
        Some(alumina::data::image_folder::image_to_data(&ref_img))
    }).flatten();

    println!("SRGAN-Rust Model Benchmark");
    println!("==========================");
    println!("Input: {} ({}x{})", input_path.display(), input_image.width(), input_image.height());
    if let Some(ref p) = reference_path {
        println!("Reference: {}", p);
    }
    println!();

    let mut results: Vec<BenchmarkEntry> = Vec::new();

    for label in MODEL_LABELS {
        print!("Benchmarking {:20}... ", label);

        // Load model
        let load_start = Instant::now();
        let network = match UpscalingNetwork::from_label(label, None) {
            Ok(n) => n,
            Err(e) => {
                println!("SKIP ({})", e);
                results.push(BenchmarkEntry {
                    model: label.to_string(),
                    load_time_ms: 0.0,
                    inference_time_ms: 0.0,
                    psnr_db: None,
                    psnr_luma_db: None,
                    ssim: None,
                    status: format!("skip: {}", e),
                });
                continue;
            }
        };
        let load_time = load_start.elapsed();

        // Run inference
        let infer_start = Instant::now();
        let output = match srgan_rust::upscale(input_4d.clone(), &network) {
            Ok(o) => o,
            Err(e) => {
                println!("FAIL ({})", e);
                results.push(BenchmarkEntry {
                    model: label.to_string(),
                    load_time_ms: load_time.as_secs_f64() * 1000.0,
                    inference_time_ms: 0.0,
                    psnr_db: None,
                    psnr_luma_db: None,
                    ssim: None,
                    status: format!("fail: {}", e),
                });
                continue;
            }
        };
        let infer_time = infer_start.elapsed();

        // Compute quality metrics if reference is available
        let (psnr_db, psnr_luma_db, ssim_val) = if let Some(ref ref_t) = reference_tensor {
            let output_3d = output.view();
            let score = compute_quality(ref_t.view(), output_3d);
            (Some(score.psnr_db), Some(score.psnr_luma_db), Some(score.ssim))
        } else {
            (None, None, None)
        };

        let infer_ms = infer_time.as_secs_f64() * 1000.0;
        let load_ms = load_time.as_secs_f64() * 1000.0;

        println!(
            "{:.0}ms (load: {:.0}ms){}",
            infer_ms,
            load_ms,
            if let Some(p) = psnr_db {
                format!(" | PSNR: {:.2}dB | SSIM: {:.4}", p, ssim_val.unwrap_or(0.0))
            } else {
                String::new()
            }
        );

        results.push(BenchmarkEntry {
            model: label.to_string(),
            load_time_ms: load_ms,
            inference_time_ms: infer_ms,
            psnr_db,
            psnr_luma_db,
            ssim: ssim_val,
            status: "ok".to_string(),
        });
    }

    // Print comparison table
    println!();
    println!("Comparison Table");
    println!("================");
    print_table(&results, reference_tensor.is_some());

    // Output JSON for programmatic consumption
    if std::env::var("BENCHMARK_JSON").is_ok() {
        let json: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "model": r.model,
                    "load_time_ms": r.load_time_ms,
                    "inference_time_ms": r.inference_time_ms,
                    "psnr_db": r.psnr_db,
                    "psnr_luma_db": r.psnr_luma_db,
                    "ssim": r.ssim,
                    "status": r.status,
                })
            })
            .collect();
        println!("\nJSON:\n{}", serde_json::to_string_pretty(&json).unwrap_or_default());
    }
}

fn print_table(results: &[BenchmarkEntry], has_quality: bool) {
    if has_quality {
        println!(
            "{:<22} {:>10} {:>10} {:>10} {:>12} {:>8}  {}",
            "Model", "Load(ms)", "Infer(ms)", "PSNR(dB)", "Luma PSNR", "SSIM", "Status"
        );
        println!("{:-<90}", "");
        for r in results {
            println!(
                "{:<22} {:>10.1} {:>10.1} {:>10} {:>12} {:>8}  {}",
                r.model,
                r.load_time_ms,
                r.inference_time_ms,
                r.psnr_db.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
                r.psnr_luma_db.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
                r.ssim.map(|v| format!("{:.4}", v)).unwrap_or_else(|| "-".into()),
                r.status,
            );
        }
    } else {
        println!(
            "{:<22} {:>10} {:>10}  {}",
            "Model", "Load(ms)", "Infer(ms)", "Status"
        );
        println!("{:-<60}", "");
        for r in results {
            println!(
                "{:<22} {:>10.1} {:>10.1}  {}",
                r.model, r.load_time_ms, r.inference_time_ms, r.status,
            );
        }
    }
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
