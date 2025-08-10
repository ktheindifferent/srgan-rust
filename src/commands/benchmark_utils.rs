use std::time::{Duration, Instant};
use log::info;
use crate::error::{SrganError, Result};
use crate::UpscalingNetwork;
use image::DynamicImage;

pub struct BenchmarkRunner {
    network: UpscalingNetwork,
    input: DynamicImage,
    warmup_iterations: usize,
    benchmark_iterations: usize,
}

impl BenchmarkRunner {
    pub fn new(
        network: UpscalingNetwork,
        input: DynamicImage,
        warmup_iterations: usize,
        benchmark_iterations: usize,
    ) -> Self {
        Self {
            network,
            input,
            warmup_iterations,
            benchmark_iterations,
        }
    }
    
    pub fn run(&self) -> Result<BenchmarkStatistics> {
        self.run_warmup()?;
        let times = self.run_benchmark()?;
        Ok(BenchmarkStatistics::from_times(times))
    }
    
    fn run_warmup(&self) -> Result<()> {
        log::info!("Running {} warmup iterations...", self.warmup_iterations);
        
        for _ in 0..self.warmup_iterations {
            let _ = crate::upscale(self.input.clone(), &self.network)?;
        }
        
        Ok(())
    }
    
    fn run_benchmark(&self) -> Result<Vec<Duration>> {
        log::info!("Running {} benchmark iterations...", self.benchmark_iterations);
        
        let mut times = Vec::with_capacity(self.benchmark_iterations);
        
        for i in 0..self.benchmark_iterations {
            let start = Instant::now();
            let _ = crate::upscale(self.input.clone(), &self.network)?;
            let elapsed = start.elapsed();
            times.push(elapsed);
            
            if (i + 1) % 10 == 0 {
                log::info!("  Completed {}/{} iterations", i + 1, self.benchmark_iterations);
            }
        }
        
        Ok(times)
    }
}

pub struct BenchmarkStatistics {
    pub mean: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
    pub throughput: f64,
}

impl BenchmarkStatistics {
    pub fn from_times(mut times: Vec<Duration>) -> Self {
        times.sort();
        
        let mean = Self::calculate_mean(&times);
        let median = Self::calculate_median(&times);
        let min = *times.first().unwrap_or(&Duration::ZERO);
        let max = *times.last().unwrap_or(&Duration::ZERO);
        let std_dev = Self::calculate_std_dev(&times, mean);
        let throughput = if mean.as_secs_f64() > 0.0 {
            1.0 / mean.as_secs_f64()
        } else {
            0.0
        };
        
        Self {
            mean,
            median,
            min,
            max,
            std_dev,
            throughput,
        }
    }
    
    fn calculate_mean(times: &[Duration]) -> Duration {
        if times.is_empty() {
            return Duration::ZERO;
        }
        
        let total: Duration = times.iter().sum();
        total / times.len() as u32
    }
    
    fn calculate_median(times: &[Duration]) -> Duration {
        if times.is_empty() {
            return Duration::ZERO;
        }
        
        let mid = times.len() / 2;
        if times.len() % 2 == 0 {
            (times[mid - 1] + times[mid]) / 2
        } else {
            times[mid]
        }
    }
    
    fn calculate_std_dev(times: &[Duration], mean: Duration) -> Duration {
        if times.len() <= 1 {
            return Duration::ZERO;
        }
        
        let mean_ms = mean.as_secs_f64() * 1000.0;
        let variance: f64 = times.iter()
            .map(|t| {
                let t_ms = t.as_secs_f64() * 1000.0;
                (t_ms - mean_ms).powi(2)
            })
            .sum::<f64>() / (times.len() - 1) as f64;
            
        Duration::from_secs_f64(variance.sqrt() / 1000.0)
    }
    
    pub fn print_summary(&self, model_name: &str) {
        println!("\n===== Benchmark Results for {} =====", model_name);
        println!("Mean:       {:.2} ms", self.mean.as_secs_f64() * 1000.0);
        println!("Median:     {:.2} ms", self.median.as_secs_f64() * 1000.0);
        println!("Min:        {:.2} ms", self.min.as_secs_f64() * 1000.0);
        println!("Max:        {:.2} ms", self.max.as_secs_f64() * 1000.0);
        println!("Std Dev:    {:.2} ms", self.std_dev.as_secs_f64() * 1000.0);
        println!("Throughput: {:.2} images/sec", self.throughput);
        println!("=====================================\n");
    }
}

pub fn load_benchmark_network(model_name: &str, factor: usize) -> Result<UpscalingNetwork> {
    if model_name.starts_with("custom:") {
        let path = &model_name[7..];
        let data = crate::utils::file_io::read_file_bytes(path)?;
        let network_desc = crate::network_from_bytes(&data)?;
        UpscalingNetwork::new(network_desc, "custom model")
            .map_err(|e| SrganError::Network(e))
    } else {
        UpscalingNetwork::from_label(model_name, Some(factor))
            .map_err(|e| SrganError::Network(e))
    }
}

pub fn create_test_image(width: usize, height: usize) -> DynamicImage {
    use image::{RgbImage, Rgb};
    use rand::Rng;
    
    let mut rng = rand::thread_rng();
    let mut img = RgbImage::new(width as u32, height as u32);
    
    for y in 0..height {
        for x in 0..width {
            let r = rng.gen_range(0..256) as u8;
            let g = rng.gen_range(0..256) as u8;
            let b = rng.gen_range(0..256) as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    
    DynamicImage::ImageRgb8(img)
}