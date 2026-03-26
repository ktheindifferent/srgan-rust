use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;
use clap::ArgMatches;
use log::info;
use ndarray::ArrayD;
use std::time::{Duration, Instant};

// ── Resolutions to test ───────────────────────────────────────────────────────

const TEST_RESOLUTIONS: &[(usize, usize)] = &[
    (256,  256),
    (512,  512),
    (1024, 1024),
];

// ── Result types ──────────────────────────────────────────────────────────────

/// Per-resolution benchmark result for one model.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub model_name: String,
    pub input_size: (usize, usize),
    pub output_size: (usize, usize),
    pub factor: usize,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    /// Output megapixels per second.
    pub throughput_mpx_per_sec: f64,
    /// images per second (= 1 / avg_time_s)
    pub images_per_sec: f64,
    /// Estimated peak memory in MiB.
    pub memory_usage_mb: f64,
}

impl BenchmarkResult {
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{}x{},{}x{},{},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.1}",
            self.model_name,
            self.factor,
            self.input_size.0, self.input_size.1,
            self.output_size.0, self.output_size.1,
            self.iterations,
            self.total_time.as_secs_f64(),
            self.avg_time.as_secs_f64() * 1000.0,
            self.min_time.as_secs_f64() * 1000.0,
            self.max_time.as_secs_f64() * 1000.0,
            self.throughput_mpx_per_sec,
            self.images_per_sec,
            self.memory_usage_mb,
        )
    }

    pub fn csv_header() -> &'static str {
        "Model,Factor,In_W,In_H,Out_W,Out_H,Iters,Total_s,Avg_ms,Min_ms,Max_ms,Throughput_MPx/s,Img/s,Mem_MB"
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn benchmark(app_m: &ArgMatches) -> Result<()> {
    let iterations: usize = app_m
        .value_of("iterations")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let warmup: usize = app_m
        .value_of("warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let models_str = app_m.value_of("models").unwrap_or("natural,anime,bilinear");
    let model_names: Vec<&str> = models_str.split(',').map(|s| s.trim()).collect();

    let output_path = app_m.value_of("output");

    info!(
        "Benchmark: {} iterations (+{} warmup), models: [{}]",
        iterations, warmup, models_str
    );
    info!("Resolutions: {:?}", TEST_RESOLUTIONS);

    let mut all_results: Vec<BenchmarkResult> = Vec::new();

    for model_name in &model_names {
        match run_model_benchmarks(model_name, iterations, warmup) {
            Ok(results) => all_results.extend(results),
            Err(e) => eprintln!("Skipping '{}': {}", model_name, e),
        }
    }

    if all_results.is_empty() {
        return Err(SrganError::InvalidParameter(
            "No models could be benchmarked".to_string(),
        ));
    }

    print_table(&all_results);

    if let Some(path) = output_path {
        save_json(&all_results, path)?;
        println!("\nResults written to: {}", path);
    }

    Ok(())
}

// ── Per-model runner ──────────────────────────────────────────────────────────

fn run_model_benchmarks(
    model_name: &str,
    iterations: usize,
    warmup: usize,
) -> Result<Vec<BenchmarkResult>> {
    info!("Loading model '{}'…", model_name);

    let network = load_network(model_name)?;
    let factor = 4usize; // all built-in models are 4×

    let mut results = Vec::new();

    for &(w, h) in TEST_RESOLUTIONS {
        match run_single_benchmark(&network, model_name, factor, w, h, iterations, warmup) {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("  {}×{} failed: {}", w, h, e),
        }
    }

    Ok(results)
}

fn run_single_benchmark(
    network: &UpscalingNetwork,
    model_name: &str,
    factor: usize,
    width: usize,
    height: usize,
    iterations: usize,
    warmup: usize,
) -> Result<BenchmarkResult> {
    info!("  {}×{} — {} warmup + {} iterations…", width, height, warmup, iterations);

    let input = synthetic_image(width, height);
    let out_w = width * factor;
    let out_h = height * factor;

    // Warmup
    for _ in 0..warmup {
        let _ = crate::upscale(input.clone(), network)?;
    }

    // Timed iterations
    let mut times = Vec::with_capacity(iterations);
    let total_start = Instant::now();

    for _ in 0..iterations {
        let t = Instant::now();
        let _ = crate::upscale(input.clone(), network)?;
        times.push(t.elapsed());
    }

    let total_time = total_start.elapsed();
    let avg_time   = total_time / iterations as u32;
    let min_time   = times.iter().copied().min().unwrap_or(Duration::ZERO);
    let max_time   = times.iter().copied().max().unwrap_or(Duration::ZERO);

    let output_pixels      = (out_w * out_h) as f64;
    let throughput_mpx     = output_pixels / avg_time.as_secs_f64() / 1_000_000.0;
    let images_per_sec     = 1.0 / avg_time.as_secs_f64();
    let memory_mb          = estimate_memory(width, height, factor, model_name);

    Ok(BenchmarkResult {
        model_name: model_name.to_string(),
        input_size: (width, height),
        output_size: (out_w, out_h),
        factor,
        iterations,
        warmup_iterations: warmup,
        total_time,
        avg_time,
        min_time,
        max_time,
        throughput_mpx_per_sec: throughput_mpx,
        images_per_sec,
        memory_usage_mb: memory_mb,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn load_network(model_name: &str) -> Result<UpscalingNetwork> {
    if model_name.starts_with("custom:") {
        let path = &model_name[7..];
        let mut file = std::fs::File::open(path)?;
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut data)?;
        let desc = crate::network_from_bytes(&data)
            .map_err(|e| SrganError::Network(e))?;
        UpscalingNetwork::new(desc, "custom model")
            .map_err(|e| SrganError::Network(e))
    } else {
        UpscalingNetwork::from_label(model_name, Some(4))
            .map_err(|e| SrganError::Network(e))
    }
}

fn synthetic_image(width: usize, height: usize) -> ArrayD<f32> {
    let mut rng = rand::thread_rng();
    let mut img = ArrayD::<f32>::zeros(vec![1, height, width, 3]);
    for e in img.iter_mut() {
        *e = rand::Rng::gen::<f32>(&mut rng);
    }
    img
}

fn estimate_memory(width: usize, height: usize, factor: usize, model_name: &str) -> f64 {
    let input_mb  = (width * height * 3 * 4) as f64 / (1024.0 * 1024.0);
    let output_mb = (width * factor * height * factor * 3 * 4) as f64 / (1024.0 * 1024.0);
    let model_mb  = match model_name {
        "bilinear" => 0.1,
        "anime"    => 50.0,
        _          => 40.0,
    };
    input_mb + output_mb + model_mb
}

// ── Output ────────────────────────────────────────────────────────────────────

fn print_table(results: &[BenchmarkResult]) {
    println!();
    println!("╔══════════════╦══════════════╦══════════════╦══════════════╦══════════════╦══════════════╗");
    println!("║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║",
             "Model", "Resolution", "Avg (ms)", "Img/s", "Throughput", "Mem (MB)");
    println!("║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║",
             "", "", "", "", "MPx/s", "");
    println!("╠══════════════╬══════════════╬══════════════╬══════════════╬══════════════╬══════════════╣");

    for r in results {
        println!(
            "║ {:12} ║ {:>5}×{:<5} ║ {:12.1} ║ {:12.2} ║ {:12.2} ║ {:12.1} ║",
            r.model_name,
            r.input_size.0, r.input_size.1,
            r.avg_time.as_secs_f64() * 1000.0,
            r.images_per_sec,
            r.throughput_mpx_per_sec,
            r.memory_usage_mb,
        );
    }

    println!("╚══════════════╩══════════════╩══════════════╩══════════════╩══════════════╩══════════════╝");

    // Summary: fastest at each resolution
    println!();
    for &(w, h) in TEST_RESOLUTIONS {
        let at_res: Vec<_> = results.iter().filter(|r| r.input_size == (w, h)).collect();
        if let Some(fastest) = at_res.iter().min_by(|a, b| a.avg_time.partial_cmp(&b.avg_time).unwrap()) {
            println!(
                "  {}×{}: fastest = {} ({:.1} ms/img, {:.2} img/s)",
                w, h,
                fastest.model_name,
                fastest.avg_time.as_secs_f64() * 1000.0,
                fastest.images_per_sec,
            );
        }
    }
    println!();
}

fn save_json(results: &[BenchmarkResult], path: &str) -> Result<()> {
    let rows: Vec<serde_json::Value> = results.iter().map(|r| {
        serde_json::json!({
            "model": r.model_name,
            "input_width":  r.input_size.0,
            "input_height": r.input_size.1,
            "factor": r.factor,
            "iterations": r.iterations,
            "avg_ms": r.avg_time.as_secs_f64() * 1000.0,
            "min_ms": r.min_time.as_secs_f64() * 1000.0,
            "max_ms": r.max_time.as_secs_f64() * 1000.0,
            "images_per_sec": r.images_per_sec,
            "throughput_mpx_per_sec": r.throughput_mpx_per_sec,
            "memory_mb": r.memory_usage_mb,
        })
    }).collect();

    let json = serde_json::to_string_pretty(&serde_json::json!({ "results": rows }))
        .map_err(|e| SrganError::Serialization(e.to_string()))?;

    std::fs::write(path, json)?;
    Ok(())
}
