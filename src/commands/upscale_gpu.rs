use crate::error::{Result, SrganError};
use crate::gpu::{GpuBackend, GpuContext, GpuDevice, GpuCompute};
use crate::validation;
use crate::{read, save, upscale, UpscalingNetwork};
use clap::ArgMatches;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use std::fs::File;
use std::path::Path;
use std::time::Instant;

pub fn upscale_gpu(app_m: &ArgMatches) -> Result<()> {
    let input_path = app_m.value_of("input")
        .ok_or_else(|| SrganError::InvalidParameter("Input path is required".to_string()))?;
    let output_path = app_m.value_of("output")
        .ok_or_else(|| SrganError::InvalidParameter("Output path is required".to_string()))?;
    let network_label = app_m.value_of("network").unwrap_or("natural");
    let gpu_backend = app_m.value_of("gpu").unwrap_or("auto");
    
    // Validate input
    validation::validate_input_file(input_path)?;
    validation::validate_output_path(output_path)?;
    
    // Select GPU backend
    let backend = if gpu_backend == "auto" {
        select_best_backend()
    } else {
        GpuBackend::from_str(gpu_backend)?
    };
    
    info!("Using GPU backend: {}", backend);
    
    // Create GPU context
    let gpu_context = GpuContext::new(backend)?;
    let device = gpu_context.device();
    
    info!("Selected device: {} (Memory: {} MB)", 
          device.name(), 
          device.memory_mb());
    
    // Create progress bar
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    
    // Load network
    pb.set_message("Loading network...");
    let network = UpscalingNetwork::from_label(network_label, None)?;
    
    // Load image
    pb.set_message("Loading image...");
    let mut input_file = File::open(input_path)
        .map_err(|_e| SrganError::FileNotFound(Path::new(input_path).to_path_buf()))?;
    let mut image = read(&mut input_file)?;
    
    // Get image dimensions
    let shape = image.shape();
    let height = shape[1];
    let width = shape[2];
    let _channels = shape[3];
    let total_pixels = height * width;
    
    info!("Input image: {}x{} ({:.2} megapixels)", 
          width, height, total_pixels as f64 / 1_000_000.0);
    
    // Transfer to GPU if available
    if backend != GpuBackend::None {
        pb.set_message("Transferring to GPU...");
        image.to_device(device)?;
    }
    
    // Perform upscaling
    pb.set_message("Upscaling image...");
    let start = Instant::now();
    
    let output = if backend != GpuBackend::None {
        // GPU-accelerated upscaling
        upscale_gpu_accelerated(image, &network, device)?
    } else {
        // CPU fallback
        upscale(image, &network)?
    };
    
    let elapsed = start.elapsed();
    let throughput = (total_pixels as f64) / elapsed.as_secs_f64() / 1_000_000.0;
    
    pb.set_message("Saving output...");
    
    // Save output
    let mut output_file = File::create(output_path)
        .map_err(|e| SrganError::Io(e))?;
    save(output, &mut output_file)?;
    
    pb.finish_with_message(format!(
        "✓ Upscaling complete in {:.2}s ({:.2} MP/s)",
        elapsed.as_secs_f64(),
        throughput
    ));
    
    info!("Output saved to: {}", output_path);
    
    // Print performance stats
    if backend != GpuBackend::None {
        println!("\nGPU Performance Statistics:");
        println!("  Backend: {}", backend);
        println!("  Device: {}", device.name());
        println!("  Processing time: {:.3}s", elapsed.as_secs_f64());
        println!("  Throughput: {:.2} MP/s", throughput);
        println!("  Memory used: {} MB", gpu_context.allocated_mb());
    }
    
    Ok(())
}

fn select_best_backend() -> GpuBackend {
    // Try backends in order of preference
    let backends = [
        GpuBackend::Cuda,
        GpuBackend::Metal,
        GpuBackend::Vulkan,
        GpuBackend::OpenCL,
    ];
    
    for backend in &backends {
        if backend.is_available() {
            info!("Auto-selected GPU backend: {}", backend);
            return *backend;
        }
    }
    
    warn!("No GPU backend available, falling back to CPU");
    GpuBackend::None
}

fn upscale_gpu_accelerated(
    mut image: ndarray::ArrayD<f32>,
    network: &UpscalingNetwork,
    _device: &GpuDevice,
) -> Result<ndarray::ArrayD<f32>> {
    // For now, fall back to CPU implementation
    // In a real implementation, this would use GPU kernels
    warn!("GPU acceleration not fully implemented, using CPU fallback");
    image.to_cpu()?;
    upscale(image, network).map_err(|e| SrganError::GraphExecution(e.to_string()))
}

pub fn list_gpu_devices(_app_m: &ArgMatches) -> Result<()> {
    println!("Available GPU devices:");
    println!("----------------------");
    
    let devices = GpuDevice::list_devices();
    
    if devices.is_empty() {
        println!("No GPU devices found. CPU mode will be used.");
    } else {
        for (i, device) in devices.iter().enumerate() {
            println!("Device {}: {}", i, device.name());
            println!("  Backend: {}", device.backend());
            if device.memory_mb() > 0 {
                println!("  Memory: {} MB", device.memory_mb());
            }
            println!();
        }
    }
    
    // Check backend availability
    println!("\nBackend availability:");
    println!("--------------------");
    let backends = [
        ("CUDA", GpuBackend::Cuda),
        ("OpenCL", GpuBackend::OpenCL),
        ("Metal", GpuBackend::Metal),
        ("Vulkan", GpuBackend::Vulkan),
    ];
    
    for (name, backend) in &backends {
        let status = if backend.is_available() {
            "✓ Available"
        } else {
            "✗ Not available"
        };
        println!("{:10} {}", name, status);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_select_best_backend() {
        // Should always return something, even if just CPU
        let backend = select_best_backend();
        assert!(backend == GpuBackend::None || backend.is_available());
    }
    
    #[test]
    fn test_backend_from_string() {
        assert!(GpuBackend::from_str("cuda").is_ok());
        assert!(GpuBackend::from_str("opencl").is_ok());
        assert!(GpuBackend::from_str("invalid").is_err());
    }
}