use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;
use clap::ArgMatches;
use log::info;
use ndarray::ArrayD;
use std::time::{Duration, Instant};

/// Benchmark structure to hold results
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
    pub throughput_mpx_per_sec: f64,
    pub memory_usage_mb: f64,
}

impl BenchmarkResult {
    pub fn print_summary(&self) {
        println!("\n========== Benchmark Results ==========");
        println!("Model: {}", self.model_name);
        println!("Input size: {}x{}", self.input_size.0, self.input_size.1);
        println!("Output size: {}x{}", self.output_size.0, self.output_size.1);
        println!("Upscaling factor: {}x", self.factor);
        println!();
        println!("Iterations: {} (+ {} warmup)", self.iterations, self.warmup_iterations);
        println!("Total time: {:.3}s", self.total_time.as_secs_f64());
        println!("Average time: {:.3}ms", self.avg_time.as_secs_f64() * 1000.0);
        println!("Min time: {:.3}ms", self.min_time.as_secs_f64() * 1000.0);
        println!("Max time: {:.3}ms", self.max_time.as_secs_f64() * 1000.0);
        println!();
        println!("Throughput: {:.2} Megapixels/sec", self.throughput_mpx_per_sec);
        println!("Estimated memory: {:.1} MB", self.memory_usage_mb);
        println!("========================================");
    }
    
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{}x{},{}x{},{},{},{:.3},{:.3},{:.3},{:.3},{:.2},{:.1}",
            self.model_name,
            self.factor,
            self.input_size.0, self.input_size.1,
            self.output_size.0, self.output_size.1,
            self.iterations,
            self.warmup_iterations,
            self.total_time.as_secs_f64(),
            self.avg_time.as_secs_f64() * 1000.0,
            self.min_time.as_secs_f64() * 1000.0,
            self.max_time.as_secs_f64() * 1000.0,
            self.throughput_mpx_per_sec,
            self.memory_usage_mb
        )
    }
    
    pub fn csv_header() -> String {
        "Model,Factor,Input_Width,Input_Height,Output_Width,Output_Height,Iterations,Warmup,Total_Time_s,Avg_Time_ms,Min_Time_ms,Max_Time_ms,Throughput_MPx_s,Memory_MB".to_string()
    }
}

pub fn benchmark(app_m: &ArgMatches) -> Result<()> {
    // Parse arguments
    let width = app_m
        .value_of("WIDTH")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(512);
    
    let height = app_m
        .value_of("HEIGHT")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(512);
    
    let iterations = app_m
        .value_of("ITERATIONS")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    
    let warmup = app_m
        .value_of("WARMUP")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    
    let compare_all = app_m.is_present("COMPARE");
    let csv_output = app_m.is_present("CSV");
    
    info!("Starting benchmark with {}x{} input image", width, height);
    info!("Iterations: {} (+ {} warmup)", iterations, warmup);
    
    let mut results = Vec::new();
    
    if compare_all {
        // Benchmark all available models
        let models = vec![
            ("natural", 4),
            ("anime", 4),
            ("bilinear", 4),
        ];
        
        for (model_name, factor) in models {
            match benchmark_model(model_name, factor, width, height, iterations, warmup) {
                Ok(result) => results.push(result),
                Err(e) => eprintln!("Failed to benchmark {}: {}", model_name, e),
            }
        }
    } else {
        // Benchmark specified model
        let factor = app_m
            .value_of("FACTOR")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4);
        
        let model_name = if let Some(custom_path) = app_m.value_of("CUSTOM") {
            format!("custom:{}", custom_path)
        } else {
            app_m.value_of("PARAMETERS").unwrap_or("natural").to_string()
        };
        
        let result = benchmark_model(&model_name, factor, width, height, iterations, warmup)?;
        results.push(result);
    }
    
    // Output results
    if csv_output {
        println!("{}", BenchmarkResult::csv_header());
        for result in &results {
            println!("{}", result.to_csv_row());
        }
    } else {
        for result in &results {
            result.print_summary();
        }
        
        if results.len() > 1 {
            print_comparison(&results);
        }
    }
    
    Ok(())
}

fn benchmark_model(
    model_name: &str,
    factor: usize,
    width: usize,
    height: usize,
    iterations: usize,
    warmup: usize,
) -> Result<BenchmarkResult> {
    info!("Benchmarking {} model...", model_name);
    
    // Load the network
    let network = if model_name.starts_with("custom:") {
        let path = &model_name[7..];
        let mut file = std::fs::File::open(path)?;
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut data)?;
        let network_desc = crate::network_from_bytes(&data)?;
        UpscalingNetwork::new(network_desc, "custom model")
            .map_err(|e| SrganError::Network(e))?
    } else {
        UpscalingNetwork::from_label(model_name, Some(factor))
            .map_err(|e| SrganError::Network(e))?
    };
    
    // Create test input
    let input = create_test_image(width, height);
    let output_width = width * factor;
    let output_height = height * factor;
    
    // Warmup runs
    info!("Running {} warmup iterations...", warmup);
    for _ in 0..warmup {
        let _ = crate::upscale(input.clone(), &network)?;
    }
    
    // Benchmark runs
    info!("Running {} benchmark iterations...", iterations);
    let mut times = Vec::with_capacity(iterations);
    
    let total_start = Instant::now();
    for i in 0..iterations {
        let start = Instant::now();
        let _ = crate::upscale(input.clone(), &network)?;
        let elapsed = start.elapsed();
        times.push(elapsed);
        
        if (i + 1) % 10 == 0 {
            info!("  Completed {}/{} iterations", i + 1, iterations);
        }
    }
    let total_time = total_start.elapsed();
    
    // Calculate statistics
    let avg_time = total_time / iterations as u32;
    let min_time = times.iter().min().copied()
        .unwrap_or(Duration::from_secs(0));
    let max_time = times.iter().max().copied()
        .unwrap_or(Duration::from_secs(0));
    
    // Calculate throughput in megapixels per second
    let output_pixels = (output_width * output_height) as f64;
    let throughput_mpx_per_sec = output_pixels / avg_time.as_secs_f64() / 1_000_000.0;
    
    // Estimate memory usage (rough approximation)
    let input_memory = (width * height * 3 * 4) as f64 / 1_048_576.0; // float32
    let output_memory = (output_width * output_height * 3 * 4) as f64 / 1_048_576.0;
    let network_memory = estimate_network_memory(&network);
    let memory_usage_mb = input_memory + output_memory + network_memory;
    
    Ok(BenchmarkResult {
        model_name: model_name.to_string(),
        input_size: (width, height),
        output_size: (output_width, output_height),
        factor,
        iterations,
        warmup_iterations: warmup,
        total_time,
        avg_time,
        min_time,
        max_time,
        throughput_mpx_per_sec,
        memory_usage_mb,
    })
}

fn create_test_image(width: usize, height: usize) -> ArrayD<f32> {
    // Create a random test image
    let mut rng = rand::thread_rng();
    let shape = vec![1, height, width, 3];
    let mut image = ArrayD::<f32>::zeros(shape.clone());
    
    // Fill with random values
    for elem in image.iter_mut() {
        *elem = rand::Rng::gen::<f32>(&mut rng);
    }
    
    image.into_dyn()
}

fn estimate_network_memory(network: &UpscalingNetwork) -> f64 {
    // Rough estimate based on network description
    // This is a simplified estimation
    // Estimate based on network type
    let desc = network.to_string();
    if desc.contains("bilinear") {
        0.1
    } else if desc.contains("anime") {
        50.0
    } else {
        40.0 // Default for natural model
    }
}

fn print_comparison(results: &[BenchmarkResult]) {
    println!("\n========== Performance Comparison ==========");
    println!("{:<15} {:>12} {:>12} {:>15}",
        "Model", "Avg Time (ms)", "Min Time (ms)", "Throughput (MP/s)");
    println!("{:-<54}", "");
    
    for result in results {
        println!("{:<15} {:>12.3} {:>12.3} {:>15.2}",
            result.model_name,
            result.avg_time.as_secs_f64() * 1000.0,
            result.min_time.as_secs_f64() * 1000.0,
            result.throughput_mpx_per_sec
        );
    }
    
    // Find fastest model
    if let Some(fastest) = results.iter().min_by_key(|r| r.avg_time) {
        println!("\nFastest model: {} ({:.3}ms average)",
            fastest.model_name,
            fastest.avg_time.as_secs_f64() * 1000.0
        );
    }
    
    // Find highest throughput
    if let Some(highest) = results.iter()
        .max_by(|a, b| a.throughput_mpx_per_sec.partial_cmp(&b.throughput_mpx_per_sec)
            .unwrap_or(std::cmp::Ordering::Equal)) {
        println!("Highest throughput: {} ({:.2} MP/s)",
            highest.model_name,
            highest.throughput_mpx_per_sec
        );
    }
    
    println!("=============================================");
}