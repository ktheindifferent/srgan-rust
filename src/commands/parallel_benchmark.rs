use crate::benchmarks::{run_benchmark_suite, print_benchmark_table};
use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;
use clap::ArgMatches;
use log::info;

pub fn run_parallel_benchmark(app_m: &ArgMatches) -> Result<()> {
    info!("Starting parallel processing benchmark...");
    
    // Parse model
    let model_type = app_m.value_of("MODEL").unwrap_or("natural");
    let network = UpscalingNetwork::from_label(model_type, None)
        .map_err(|e| SrganError::Network(e))?;
    
    info!("Using {} model for benchmarking", model_type);
    
    // Run benchmark suite
    let results = run_benchmark_suite(&network);
    
    // Print results table
    print_benchmark_table(&results);
    
    // Print performance analysis
    print_performance_analysis(&results);
    
    Ok(())
}

fn print_performance_analysis(results: &[crate::benchmarks::BenchmarkResult]) {
    println!("\n\nPerformance Analysis");
    println!("====================");
    
    // Find best speedup for each batch size
    let mut batch_sizes = std::collections::HashSet::new();
    for result in results {
        batch_sizes.insert(result.num_images);
    }
    
    for batch_size in batch_sizes {
        let batch_results: Vec<_> = results
            .iter()
            .filter(|r| r.num_images == batch_size)
            .collect();
        
        if let Some(best) = batch_results.iter().max_by(|a, b| {
            a.speedup.partial_cmp(&b.speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            println!("\nBatch size {}:", batch_size);
            println!("  Best configuration: {} threads", best.num_threads);
            println!("  Speedup: {:.2}x", best.speedup);
            println!("  Throughput: {:.2} images/second", best.images_per_second);
            
            // Calculate efficiency
            if best.num_threads > 1 {
                let efficiency = (best.speedup / best.num_threads as f64) * 100.0;
                println!("  Parallel efficiency: {:.1}%", efficiency);
            }
        }
    }
    
    // Overall recommendations
    println!("\n\nRecommendations");
    println!("===============");
    
    let max_speedup = results.iter()
        .map(|r| r.speedup)
        .fold(0.0f64, f64::max);
    
    if max_speedup > 1.5 {
        println!("✓ Parallel processing provides significant performance improvements");
        println!("✓ Maximum speedup achieved: {:.2}x", max_speedup);
        
        if max_speedup > 3.0 {
            println!("✓ Near-linear scaling observed - excellent parallel efficiency");
        } else if max_speedup > 2.0 {
            println!("✓ Good parallel scaling observed");
        }
    } else {
        println!("⚠ Limited parallel speedup observed");
        println!("  Consider checking for bottlenecks in I/O or memory bandwidth");
    }
    
    // Thread recommendations
    let optimal_threads = find_optimal_thread_count(results);
    println!("\nOptimal thread count for this system: {}", optimal_threads);
    
    // Memory usage estimate
    println!("\nEstimated memory usage:");
    println!("  Per thread: ~200-300 MB (depending on image size)");
    println!("  Total with {} threads: ~{}-{} MB", 
        optimal_threads, 
        optimal_threads * 200,
        optimal_threads * 300
    );
}

fn find_optimal_thread_count(results: &[crate::benchmarks::BenchmarkResult]) -> usize {
    // Find thread count with best average speedup across all batch sizes
    let mut thread_speedups = std::collections::HashMap::new();
    
    for result in results {
        if result.num_threads > 1 {
            let speedups = thread_speedups.entry(result.num_threads).or_insert(Vec::new());
            speedups.push(result.speedup);
        }
    }
    
    let mut best_threads = 1;
    let mut best_avg_speedup = 1.0;
    
    for (threads, speedups) in thread_speedups {
        let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
        if avg_speedup > best_avg_speedup {
            best_avg_speedup = avg_speedup;
            best_threads = threads;
        }
    }
    
    best_threads
}