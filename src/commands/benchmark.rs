use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;
use base64::Engine as _;
use clap::ArgMatches;
use indicatif::ProgressBar;
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
    /// Input data throughput in MB/sec.
    pub mb_per_sec: f64,
}

impl BenchmarkResult {
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{}x{},{}x{},{},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.1},{:.2}",
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
            self.mb_per_sec,
        )
    }

    pub fn csv_header() -> &'static str {
        "Model,Factor,In_W,In_H,Out_W,Out_H,Iters,Total_s,Avg_ms,Min_ms,Max_ms,Throughput_MPx/s,Img/s,Mem_MB,MB/s"
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn benchmark(app_m: &ArgMatches) -> Result<()> {
    // Quality-comparison mode: real image in, per-model outputs + HTML/JSON report out.
    if app_m.value_of("output-dir").is_some() {
        return run_quality_benchmark(app_m);
    }

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
    let input_bytes        = (width * height * 3 * 4) as f64; // f32 RGB
    let mb_per_sec         = (input_bytes / (1024.0 * 1024.0)) / avg_time.as_secs_f64();

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
        mb_per_sec,
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
    println!("╔══════════════╦══════════════╦══════════════╦══════════════╦══════════════╦══════════════╦══════════════╗");
    println!("║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║",
             "Model", "Resolution", "Avg (ms)", "Img/s", "Throughput", "Mem (MB)", "MB/s");
    println!("║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║ {:12} ║",
             "", "", "", "", "MPx/s", "", "");
    println!("╠══════════════╬══════════════╬══════════════╬══════════════╬══════════════╬══════════════╬══════════════╣");

    for r in results {
        println!(
            "║ {:12} ║ {:>5}×{:<5} ║ {:12.1} ║ {:12.2} ║ {:12.2} ║ {:12.1} ║ {:12.2} ║",
            r.model_name,
            r.input_size.0, r.input_size.1,
            r.avg_time.as_secs_f64() * 1000.0,
            r.images_per_sec,
            r.throughput_mpx_per_sec,
            r.memory_usage_mb,
            r.mb_per_sec,
        );
    }

    println!("╚══════════════╩══════════════╩══════════════╩══════════════╩══════════════╩══════════════╩══════════════╝");

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
            "mb_per_sec": r.mb_per_sec,
        })
    }).collect();

    let json = serde_json::to_string_pretty(&serde_json::json!({ "results": rows }))
        .map_err(|e| SrganError::Serialization(e.to_string()))?;

    std::fs::write(path, json)?;
    Ok(())
}

// ── Quality comparison mode ────────────────────────────────────────────────────

/// Per-model quality result (real-image benchmark).
#[derive(Debug, Clone)]
pub struct QualityResult {
    pub model_name:        String,
    pub processing_ms:     f64,
    pub output_file:       String,
    pub file_size_bytes:   u64,
    pub psnr_vs_bilinear:  Option<f64>,
    pub ssim_vs_bilinear:  Option<f64>,
    pub scale:             usize,
}

/// Entry point for quality-comparison mode (`--output-dir` is present).
pub fn run_quality_benchmark(app_m: &ArgMatches) -> Result<()> {
    let input_path = app_m
        .value_of("input-img")
        .or_else(|| app_m.value_of("input"))
        .ok_or_else(|| SrganError::InvalidParameter(
            "--input <IMAGE> is required when using --output-dir".to_string(),
        ))?;

    let output_dir = app_m.value_of("output-dir").unwrap();

    let scale: usize = app_m
        .value_of("scale")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let models_str = app_m.value_of("models").unwrap_or("natural,anime,bilinear");
    let mut model_names: Vec<String> = models_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Ensure bilinear is first — it serves as the quality baseline.
    if let Some(idx) = model_names.iter().position(|m| m == "bilinear") {
        model_names.remove(idx);
    }
    model_names.insert(0, "bilinear".to_string());

    std::fs::create_dir_all(output_dir)?;

    println!(
        "Benchmarking {} models on {} ({}×) → {}",
        model_names.len(),
        input_path,
        scale,
        output_dir
    );

    let pb = ProgressBar::new(model_names.len() as u64);
    let mut results: Vec<QualityResult> = Vec::new();
    let mut bilinear_tensor: Option<ArrayD<f32>> = None;

    for model_name in &model_names {
        pb.set_message(format!("Running {}...", model_name));

        let network = match load_network_scaled(model_name, scale) {
            Ok(n) => n,
            Err(e) => {
                pb.println(format!("  Skipping '{}': {}", model_name, e));
                pb.inc(1);
                continue;
            }
        };

        let input = {
            let mut f = std::fs::File::open(input_path)?;
            crate::read(&mut f).map_err(SrganError::Image)?
        };

        let t0 = Instant::now();
        let output = crate::upscale(input, &network)
            .map_err(|e| SrganError::GraphExecution(e.to_string()))?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let safe_name = model_name.replace(['/', '\\', ':'], "_");
        let out_path = format!("{}/{}.png", output_dir, safe_name);
        {
            let mut f = std::fs::File::create(&out_path)?;
            crate::save(output.clone(), &mut f).map_err(|e| {
                SrganError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)))
            })?;
        }
        let file_size = std::fs::metadata(&out_path)?.len();

        let (psnr, ssim_val) = if model_name == "bilinear" {
            bilinear_tensor = Some(output.clone());
            (None, None)
        } else if let Some(ref baseline) = bilinear_tensor {
            let (rgb_err, _y_err, pix) =
                crate::psnr::psnr_calculation(output.view(), baseline.view());
            let psnr_db = if pix > 0.0 {
                if rgb_err <= 0.0 {
                    Some(f64::INFINITY)
                } else {
                    Some(-10.0 * (rgb_err / pix).log10() as f64)
                }
            } else {
                None
            };
            let ssim =
                Some(crate::ssim::ssim_calculation(output.view(), baseline.view()) as f64);
            (psnr_db, ssim)
        } else {
            (None, None)
        };

        results.push(QualityResult {
            model_name:       model_name.clone(),
            processing_ms:    elapsed_ms,
            output_file:      out_path,
            file_size_bytes:  file_size,
            psnr_vs_bilinear: psnr,
            ssim_vs_bilinear: ssim_val,
            scale,
        });

        pb.inc(1);
    }

    pb.finish_with_message("Processing complete.");

    if results.is_empty() {
        return Err(SrganError::InvalidParameter(
            "No models produced results.".to_string(),
        ));
    }

    let json_path = format!("{}/benchmark_report.json", output_dir);
    save_quality_json(&results, input_path, scale, &json_path)?;

    let html_path = format!("{}/benchmark_report.html", output_dir);
    generate_html_report(&results, input_path, &html_path)?;

    print_quality_table(&results);
    println!("\nReports saved:");
    println!("  {}", json_path);
    println!("  {}", html_path);

    Ok(())
}

fn load_network_scaled(model_name: &str, scale: usize) -> Result<UpscalingNetwork> {
    if model_name.starts_with("custom:") {
        let path = &model_name[7..];
        let mut file = std::fs::File::open(path)?;
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut data)?;
        let desc = crate::network_from_bytes(&data).map_err(SrganError::Network)?;
        UpscalingNetwork::new(desc, "custom model").map_err(SrganError::Network)
    } else {
        UpscalingNetwork::from_label(model_name, Some(scale)).map_err(SrganError::Network)
    }
}

fn save_quality_json(
    results: &[QualityResult],
    input_path: &str,
    scale: usize,
    path: &str,
) -> Result<()> {
    let rows: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "model":               r.model_name,
                "processing_time_ms":  r.processing_ms,
                "output_file":         r.output_file,
                "file_size_bytes":     r.file_size_bytes,
                "psnr_vs_bilinear_db": r.psnr_vs_bilinear,
                "ssim_vs_bilinear":    r.ssim_vs_bilinear,
                "scale":               r.scale,
            })
        })
        .collect();

    let json = serde_json::to_string_pretty(&serde_json::json!({
        "input":        input_path,
        "scale":        scale,
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "models":       rows,
    }))
    .map_err(|e| SrganError::Serialization(e.to_string()))?;

    std::fs::write(path, json)?;
    Ok(())
}

fn read_file_as_base64(path: &str) -> Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&bytes))
}

fn mime_type(path: &str) -> &'static str {
    match std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif")  => "image/gif",
        Some("webp") => "image/webp",
        _            => "image/png",
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn print_quality_table(results: &[QualityResult]) {
    println!();
    println!("╔══════════════════╦══════════════╦══════════════════╦══════════════╦══════════════╗");
    println!(
        "║ {:16} ║ {:12} ║ {:16} ║ {:12} ║ {:12} ║",
        "Model", "Time (ms)", "PSNR vs Bilinear", "SSIM", "File Size"
    );
    println!("╠══════════════════╬══════════════╬══════════════════╬══════════════╬══════════════╣");
    for r in results {
        let psnr_s = match r.psnr_vs_bilinear {
            Some(v) if v.is_infinite() => "∞ dB".to_string(),
            Some(v) => format!("{:.2} dB", v),
            None    => "—".to_string(),
        };
        let ssim_s = match r.ssim_vs_bilinear {
            Some(v) => format!("{:.4}", v),
            None    => "—".to_string(),
        };
        println!(
            "║ {:16} ║ {:12.0} ║ {:16} ║ {:12} ║ {:12} ║",
            r.model_name,
            r.processing_ms,
            psnr_s,
            ssim_s,
            format_size(r.file_size_bytes)
        );
    }
    println!("╚══════════════════╩══════════════╩══════════════════╩══════════════╩══════════════╝");
    println!();
}

fn generate_html_report(
    results: &[QualityResult],
    input_path: &str,
    html_path: &str,
) -> Result<()> {
    // ── CSS (no external dependencies) ────────────────────────────────────────
    let css = r#"
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0e0e10;color:#e0e0e0;padding:32px 24px;line-height:1.5}
h1{font-size:1.9em;margin-bottom:6px;color:#fff}
.meta{color:#666;font-size:.88em;margin-bottom:36px}
h2{font-size:1.25em;margin:36px 0 14px;color:#bbb;border-bottom:1px solid #222;padding-bottom:8px}
h3{font-size:1.05em;margin:24px 0 10px;color:#aaa}
.metrics{font-size:.82em;color:#5af;font-weight:400;margin-left:10px}
table{width:100%;border-collapse:collapse;background:#16161a;border-radius:8px;overflow:hidden;margin-bottom:28px}
th{background:#1e1e24;padding:10px 16px;text-align:left;font-weight:600;color:#999;font-size:.82em;text-transform:uppercase;letter-spacing:.06em}
td{padding:10px 16px;border-top:1px solid #1e1e24;font-size:.9em}
td.model{font-weight:700;color:#5af}
tr:hover td{background:#1a1a1f}
.input-wrap{margin-bottom:28px}
.input-img{max-width:360px;border-radius:6px;border:1px solid #2a2a30;display:block}
.thumbnails{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:36px}
figure{flex:1 1 200px;max-width:340px;background:#16161a;border-radius:8px;overflow:hidden;border:1px solid #1e1e24}
figure img{width:100%;display:block}
figcaption{padding:7px 12px;font-size:.82em;color:#666;text-align:center}
.compare-section{margin-bottom:48px}
.compare-widget{background:#16161a;border:1px solid #1e1e24;border-radius:10px;overflow:hidden;max-width:960px}
.compare-inner{position:relative;overflow:hidden;cursor:ew-resize;line-height:0;user-select:none}
.img-base{display:block;width:100%;height:auto}
.img-top{position:absolute;top:0;left:0;width:100%;height:auto;pointer-events:none}
.compare-divider{position:absolute;top:0;bottom:0;width:2px;background:rgba(255,255,255,.85);transform:translateX(-50%);pointer-events:none}
.divider-handle{position:absolute;top:50%;left:0;transform:translate(-50%,-50%);width:36px;height:36px;background:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 10px rgba(0,0,0,.5);color:#111;font-size:13px;font-weight:700}
.slider-row{display:flex;align-items:center;gap:12px;padding:10px 16px;background:#111115}
.cmp-slider{flex:1;-webkit-appearance:none;appearance:none;height:4px;background:#2a2a30;border-radius:2px;outline:none;cursor:pointer}
.cmp-slider::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:#5af;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.4)}
.cmp-slider::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:#5af;cursor:pointer;border:none}
.label{font-size:.78em;color:#555;white-space:nowrap}
"#;

    // ── JavaScript ────────────────────────────────────────────────────────────
    let js = r#"
(function () {
  document.querySelectorAll('.compare-widget').forEach(function (widget) {
    var inner   = widget.querySelector('.compare-inner');
    var imgTop  = inner.querySelector('.img-top');
    var divider = inner.querySelector('.compare-divider');
    var slider  = widget.querySelector('.cmp-slider');

    function update(pct) {
      imgTop.style.clipPath = 'inset(0 ' + (100 - pct) + '% 0 0)';
      divider.style.left    = pct + '%';
      if (slider) slider.value = pct;
    }

    if (slider) {
      slider.addEventListener('input', function () { update(parseInt(this.value, 10)); });
    }

    var dragging = false;
    inner.addEventListener('mousedown', function (e) { dragging = true; e.preventDefault(); });
    document.addEventListener('mouseup',  function () { dragging = false; });
    document.addEventListener('mousemove', function (e) {
      if (!dragging) return;
      var rect = inner.getBoundingClientRect();
      update(Math.max(0, Math.min(100, (e.clientX - rect.left) / rect.width * 100)));
    });
    inner.addEventListener('touchmove', function (e) {
      e.preventDefault();
      var rect = inner.getBoundingClientRect();
      update(Math.max(0, Math.min(100, (e.touches[0].clientX - rect.left) / rect.width * 100)));
    }, { passive: false });

    update(50);
  });
}());
"#;

    // ── Build table rows ──────────────────────────────────────────────────────
    let table_rows: String = results
        .iter()
        .map(|r| {
            let psnr_s = match r.psnr_vs_bilinear {
                Some(v) if v.is_infinite() => "&#8734; dB".to_string(),
                Some(v) => format!("{:.2} dB", v),
                None    => "&mdash;".to_string(),
            };
            let ssim_s = match r.ssim_vs_bilinear {
                Some(v) => format!("{:.4}", v),
                None    => "&mdash;".to_string(),
            };
            format!(
                "  <tr><td class=\"model\">{}</td><td>{:.0} ms</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                r.model_name,
                r.processing_ms,
                psnr_s,
                ssim_s,
                format_size(r.file_size_bytes)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // ── Encode input image ────────────────────────────────────────────────────
    let input_b64  = read_file_as_base64(input_path)?;
    let input_mime = mime_type(input_path);

    // ── Encode all output images ──────────────────────────────────────────────
    let bilinear = results.iter().find(|r| r.model_name == "bilinear");
    let bilinear_b64: Option<String> = bilinear
        .map(|r| read_file_as_base64(&r.output_file))
        .transpose()?;

    // Thumbnails
    let thumbnails: String = results
        .iter()
        .map(|r| match read_file_as_base64(&r.output_file) {
            Ok(b64) => format!(
                "  <figure><img src=\"data:image/png;base64,{}\" alt=\"{}\"><figcaption>{}</figcaption></figure>",
                b64, r.model_name, r.model_name
            ),
            Err(_) => format!("  <figure><p style=\"padding:12px;color:#666\">{} (error)</p></figure>", r.model_name),
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Comparison sliders
    let mut sliders = String::new();
    if let Some(ref blin_b64) = bilinear_b64 {
        for r in results.iter().filter(|r| r.model_name != "bilinear") {
            let model_b64 = read_file_as_base64(&r.output_file)?;
            let psnr_info = match r.psnr_vs_bilinear {
                Some(v) if v.is_infinite() => "PSNR: &#8734; dB".to_string(),
                Some(v) => format!("PSNR: {:.2} dB", v),
                None    => String::new(),
            };
            let ssim_info = match r.ssim_vs_bilinear {
                Some(v) => format!("SSIM: {:.4}", v),
                None    => String::new(),
            };
            sliders.push_str(&format!(
                "<section class=\"compare-section\">\n\
                 <h3>{model} vs Bilinear <span class=\"metrics\">{psnr}&nbsp;&nbsp;{ssim}</span></h3>\n\
                 <div class=\"compare-widget\">\n\
                   <div class=\"compare-inner\">\n\
                     <img class=\"img-base\" src=\"data:image/png;base64,{blin}\" alt=\"Bilinear\">\n\
                     <img class=\"img-top\" src=\"data:image/png;base64,{model_img}\" alt=\"{model}\">\n\
                     <div class=\"compare-divider\"><div class=\"divider-handle\">&#x2194;</div></div>\n\
                   </div>\n\
                   <div class=\"slider-row\">\n\
                     <span class=\"label\">&laquo; Bilinear</span>\n\
                     <input type=\"range\" min=\"0\" max=\"100\" value=\"50\" class=\"cmp-slider\">\n\
                     <span class=\"label\">{model} &raquo;</span>\n\
                   </div>\n\
                 </div>\n\
                 </section>\n",
                model     = r.model_name,
                psnr      = psnr_info,
                ssim      = ssim_info,
                blin      = blin_b64,
                model_img = model_b64,
            ));
        }
    }

    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    let scale = results.first().map(|r| r.scale).unwrap_or(4);

    // ── Assemble HTML ─────────────────────────────────────────────────────────
    let html = format!(
        "<!DOCTYPE html>\n\
         <html lang=\"en\">\n\
         <head>\n\
         <meta charset=\"UTF-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n\
         <title>SRGAN Benchmark Report</title>\n\
         <style>{css}</style>\n\
         </head>\n\
         <body>\n\
         <h1>SRGAN Multi-Model Benchmark Report</h1>\n\
         <p class=\"meta\">Input: <strong>{input}</strong>&nbsp;&nbsp;|&nbsp;&nbsp;Scale: <strong>{scale}&times;</strong>&nbsp;&nbsp;|&nbsp;&nbsp;Generated: {now}</p>\n\
         <h2>Quality Metrics</h2>\n\
         <table>\n\
           <thead><tr><th>Model</th><th>Time</th><th>PSNR vs Bilinear</th><th>SSIM vs Bilinear</th><th>File Size</th></tr></thead>\n\
           <tbody>\n{rows}\n  </tbody>\n\
         </table>\n\
         <h2>Input Image</h2>\n\
         <div class=\"input-wrap\"><img class=\"input-img\" src=\"data:{imime};base64,{ib64}\" alt=\"Input\"></div>\n\
         <h2>Model Outputs</h2>\n\
         <div class=\"thumbnails\">\n{thumbs}\n</div>\n\
         <h2>Side-by-Side Comparison</h2>\n\
         {sliders}\
         <script>{js}</script>\n\
         </body>\n\
         </html>\n",
        css     = css,
        input   = input_path,
        scale   = scale,
        now     = now,
        rows    = table_rows,
        imime   = input_mime,
        ib64    = input_b64,
        thumbs  = thumbnails,
        sliders = sliders,
        js      = js,
    );

    std::fs::write(html_path, &html)?;
    Ok(())
}
