use crate::parallel::ThreadSafeNetwork;
use crate::UpscalingNetwork;
use alumina::data::image_folder::image_to_data;
use image::{DynamicImage, ImageBuffer, Rgb, GenericImage};
use std::time::{Duration, Instant};

/// Benchmark results for parallel processing
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub num_images: usize,
    pub num_threads: usize,
    pub total_time: Duration,
    pub images_per_second: f64,
    pub speedup: f64,
}

/// Generate synthetic test images for benchmarking
fn generate_test_images(count: usize, width: u32, height: u32) -> Vec<DynamicImage> {
    (0..count)
        .map(|i| {
            let img = ImageBuffer::from_fn(width, height, |x, y| {
                let r = ((x as f32 / width as f32) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = ((i as f32 / count as f32) * 255.0) as u8;
                Rgb([r, g, b])
            });
            DynamicImage::ImageRgb8(img)
        })
        .collect()
}

/// Benchmark sequential processing
pub fn benchmark_sequential(
    network: &UpscalingNetwork,
    images: &[DynamicImage],
) -> BenchmarkResult {
    let start = Instant::now();
    
    for img in images {
        let _ = network.upscale_image(img);
    }
    
    let duration = start.elapsed();
    let images_per_second = images.len() as f64 / duration.as_secs_f64();
    
    BenchmarkResult {
        name: "Sequential".to_string(),
        num_images: images.len(),
        num_threads: 1,
        total_time: duration,
        images_per_second,
        speedup: 1.0,
    }
}

/// Benchmark parallel processing
pub fn benchmark_parallel(
    network: &UpscalingNetwork,
    images: Vec<DynamicImage>,
    num_threads: usize,
) -> BenchmarkResult {
    let thread_safe_network = ThreadSafeNetwork::new(network.clone());
    let start = Instant::now();
    
    let _results = thread_safe_network.process_batch_parallel(
        images.clone(),
        |img, net| {
            let tensor = image_to_data(&img);
            let shape = tensor.shape().to_vec();
            let input = tensor.into_shape(ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]]))
                .map_err(|_| crate::error::SrganError::ShapeError("Shape error".to_string()))?;
            crate::upscale(input, net)
                .map_err(|e| crate::error::SrganError::GraphExecution(e.to_string()))
        },
        Some(num_threads),
    );
    
    let duration = start.elapsed();
    let images_per_second = images.len() as f64 / duration.as_secs_f64();
    
    BenchmarkResult {
        name: format!("Parallel ({}t)", num_threads),
        num_images: images.len(),
        num_threads,
        total_time: duration,
        images_per_second,
        speedup: 0.0, // Will be calculated later
    }
}

/// Run comprehensive benchmark suite
pub fn run_benchmark_suite(network: &UpscalingNetwork) -> Vec<BenchmarkResult> {
    println!("Running parallel processing benchmarks...");
    println!("========================================");
    
    let mut results = Vec::new();
    
    // Test different batch sizes
    let batch_sizes = vec![10, 50, 100];
    let thread_counts = vec![1, 2, 4, 8];
    
    for batch_size in batch_sizes {
        println!("\nBatch size: {} images", batch_size);
        println!("------------------------");
        
        // Generate test images (256x256)
        let images = generate_test_images(batch_size, 256, 256);
        
        // Run sequential benchmark
        let seq_result = benchmark_sequential(network, &images);
        println!("Sequential: {:.2}s ({:.2} img/s)", 
            seq_result.total_time.as_secs_f64(),
            seq_result.images_per_second
        );
        
        let seq_time = seq_result.total_time.as_secs_f64();
        results.push(seq_result);
        
        // Run parallel benchmarks with different thread counts
        for threads in &thread_counts {
            if *threads == 1 {
                continue; // Skip 1 thread (already done with sequential)
            }
            
            let mut par_result = benchmark_parallel(network, images.clone(), *threads);
            par_result.speedup = seq_time / par_result.total_time.as_secs_f64();
            
            println!("{}: {:.2}s ({:.2} img/s, {:.2}x speedup)",
                par_result.name,
                par_result.total_time.as_secs_f64(),
                par_result.images_per_second,
                par_result.speedup
            );
            
            results.push(par_result);
        }
    }
    
    results
}

/// Print benchmark comparison table
pub fn print_benchmark_table(results: &[BenchmarkResult]) {
    println!("\n\nBenchmark Results Summary");
    println!("==========================");
    println!("{:<20} {:>10} {:>10} {:>12} {:>15} {:>10}",
        "Method", "Images", "Threads", "Time (s)", "Images/sec", "Speedup");
    println!("{:-<87}", "");
    
    for result in results {
        println!("{:<20} {:>10} {:>10} {:>12.2} {:>15.2} {:>10.2}x",
            result.name,
            result.num_images,
            result.num_threads,
            result.total_time.as_secs_f64(),
            result.images_per_second,
            result.speedup
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImage;
    
    #[test]
    fn test_generate_images() {
        let images = generate_test_images(5, 128, 128);
        assert_eq!(images.len(), 5);
        for img in images {
            assert_eq!(img.width(), 128);
            assert_eq!(img.height(), 128);
        }
    }
    
    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult {
            name: "Test".to_string(),
            num_images: 10,
            num_threads: 4,
            total_time: Duration::from_secs(2),
            images_per_second: 5.0,
            speedup: 2.5,
        };
        
        assert_eq!(result.num_images, 10);
        assert_eq!(result.num_threads, 4);
        assert_eq!(result.images_per_second, 5.0);
    }
}