use crate::profiling::MemoryProfiler;
use crate::memory_scope;
use crate::error::{Result, SrganError};
use std::fs::File;
use std::thread;
use std::time::Duration;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, error};
use image::GenericImage;

pub fn profile_memory_command(
    input: &str,
    model: Option<&str>,
    output: Option<&str>,
    report_path: Option<&str>,
    sampling_interval_ms: u64,
) -> Result<()> {
    info!("Starting memory profiling for upscale operation");
    
    let report_path = report_path.unwrap_or("memory_profile.txt");
    let csv_path = report_path.replace(".txt", ".csv");
    
    let mut profiler = MemoryProfiler::new(sampling_interval_ms);
    profiler.start();
    
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
    );
    pb.set_message("Profiling memory usage...");
    
    let profiler_clone = std::sync::Arc::new(std::sync::Mutex::new(profiler));
    let profiler_thread = profiler_clone.clone();
    
    let sampling_handle = thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(sampling_interval_ms));
            if let Ok(mut p) = profiler_thread.lock() {
                p.sample();
            } else {
                break;
            }
        }
    });
    
    let result = {
        memory_scope!("upscale_operation");
        
        // Load the network
        use crate::UpscalingNetwork;
        let param_type = model.unwrap_or("natural");
        let network = UpscalingNetwork::from_label(param_type, Some(4))
            .map_err(|e| SrganError::Network(e))?;
        
        // Read input image
        let mut input_file = File::open(input)?;
        let input_data = crate::read(&mut input_file)?;
        
        // Perform upscaling
        let output_data = crate::upscale(input_data, &network)?;
        
        // Save output if specified
        if let Some(output_path) = output {
            let mut output_file = File::create(output_path)?;
            crate::save(output_data, &mut output_file)?;
        }
        
        Ok(())
    };
    
    pb.finish_with_message("Profiling complete");
    
    if let Ok(mut p) = profiler_clone.lock() {
        p.stop();
        
        let usage = p.get_current_usage();
        println!("\n📊 Memory Usage Summary:");
        println!("  Total Allocated: {:.2} MB", usage.allocated as f64 / 1_048_576.0);
        println!("  Total Deallocated: {:.2} MB", usage.deallocated as f64 / 1_048_576.0);
        println!("  Current Usage: {:.2} MB", usage.current as f64 / 1_048_576.0);
        println!("  Peak Usage: {:.2} MB", usage.peak as f64 / 1_048_576.0);
        
        if let Err(e) = p.save_report(report_path) {
            error!("Failed to save memory report: {}", e);
        } else {
            println!("\n📄 Memory report saved to: {}", report_path);
        }
        
        if let Err(e) = p.save_csv(&csv_path) {
            error!("Failed to save CSV data: {}", e);
        } else {
            println!("📈 CSV data saved to: {}", csv_path);
        }
    }
    
    drop(sampling_handle);
    
    result
}

pub fn analyze_memory_usage(
    command: &str,
    args: Vec<&str>,
    report_path: Option<&str>,
    sampling_interval_ms: u64,
) -> Result<()> {
    info!("Starting memory analysis for command: {}", command);
    
    let report_path = report_path.unwrap_or("memory_analysis.txt");
    let csv_path = report_path.replace(".txt", ".csv");
    
    let mut profiler = MemoryProfiler::new(sampling_interval_ms);
    profiler.start();
    
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg} [{elapsed_precise}]")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
    );
    pb.set_message(format!("Analyzing memory for '{}'...", command));
    
    let profiler_clone = std::sync::Arc::new(std::sync::Mutex::new(profiler));
    let profiler_thread = profiler_clone.clone();
    
    let sampling_handle = thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(sampling_interval_ms));
            if let Ok(mut p) = profiler_thread.lock() {
                p.sample();
            } else {
                break;
            }
        }
    });
    
    let result = match command {
        "upscale" => {
            if args.len() >= 1 {
                let input = args[0];
                let model = if args.len() > 1 { Some(args[1]) } else { None };
                let output = if args.len() > 2 { Some(args[2]) } else { None };
                
                // Load the network
                use crate::UpscalingNetwork;
                let param_type = model.unwrap_or("natural");
                let network = UpscalingNetwork::from_label(param_type, Some(4))
                    .map_err(|e| SrganError::Network(e))?;
                
                // Read input image
                let mut input_file = File::open(input)?;
                let input_data = crate::read(&mut input_file)?;
                
                // Perform upscaling
                let output_data = crate::upscale(input_data, &network)?;
                
                // Save output if specified
                if let Some(output_path) = output {
                    let mut output_file = File::create(output_path)?;
                    crate::save(output_data, &mut output_file)?;
                }
                
                Ok(())
            } else {
                Err(SrganError::InvalidParameter("Not enough arguments for upscale".into()))
            }
        }
        "downscale" => {
            if args.len() >= 2 {
                use image::{open, imageops::FilterType};
                
                let input = args[0];
                let factor = args[1].parse().unwrap_or(2);
                let output = if args.len() > 2 { Some(args[2]) } else { None };
                
                let img = open(input)?;
                let (width, height) = img.dimensions();
                let new_width = width / factor;
                let new_height = height / factor;
                
                let downscaled = image::imageops::resize(&img, new_width, new_height, FilterType::Lanczos3);
                
                if let Some(output_path) = output {
                    downscaled.save(output_path)?;
                }
                
                Ok(())
            } else {
                Err(SrganError::InvalidParameter("Not enough arguments for downscale".into()))
            }
        }
        _ => Err(SrganError::InvalidParameter(format!("Unknown command: {}", command)))
    };
    
    pb.finish_with_message("Analysis complete");
    
    if let Ok(mut p) = profiler_clone.lock() {
        p.stop();
        
        let report = p.report();
        println!("\n📊 Memory Analysis Results:");
        println!("  Command: {}", command);
        println!("  Duration: {:.2}s", report.duration.as_secs_f64());
        println!("  Peak Memory: {:.2} MB", report.usage.peak as f64 / 1_048_576.0);
        println!("  Final Memory: {:.2} MB", report.usage.current as f64 / 1_048_576.0);
        
        if !report.allocations.is_empty() {
            println!("\n  Allocations by Category:");
            for (category, stats) in &report.allocations {
                println!("    {}: {:.2} MB (peak: {:.2} MB)", 
                    category,
                    stats.current_bytes as f64 / 1_048_576.0,
                    stats.peak_bytes as f64 / 1_048_576.0
                );
            }
        }
        
        if let Err(e) = p.save_report(report_path) {
            error!("Failed to save analysis report: {}", e);
        } else {
            println!("\n📄 Analysis report saved to: {}", report_path);
        }
        
        if let Err(e) = p.save_csv(&csv_path) {
            error!("Failed to save CSV data: {}", e);
        } else {
            println!("📈 CSV data saved to: {}", csv_path);
        }
    }
    
    drop(sampling_handle);
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_memory_profiling_command() {
        let dir = tempdir().unwrap();
        let report_path = dir.path().join("test_profile.txt");
        
        let test_image = dir.path().join("test.png");
        let img = image::RgbImage::new(64, 64);
        img.save(&test_image).unwrap();
        
        let result = profile_memory_command(
            test_image.to_str().unwrap(),
            None,
            None,
            Some(report_path.to_str().unwrap()),
            100,
        );
        
        if result.is_ok() {
            assert!(report_path.exists());
            let csv_path = report_path.with_extension("csv");
            assert!(csv_path.exists());
        }
    }
}