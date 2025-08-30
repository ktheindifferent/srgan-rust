use crate::error::{Result, SrganError};
use crate::error_recovery::{
    CircuitBreaker, EnhancedError, ErrorAggregator, ErrorContext, 
    RetryConfig, RetryExecutor, IsRetryable
};
use crate::logging::{OperationLogger, PerformanceTracker};
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::validation;
use clap::ArgMatches;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn, span, Level};
use crate::metrics_wrapper::{increment_counter, record_histogram, set_gauge, increment_gauge, decrement_gauge};

/// Enhanced batch upscaling with retry logic and error recovery
pub async fn batch_upscale_enhanced(app_m: &ArgMatches) -> Result<()> {
    let span = span!(Level::INFO, "batch_upscale");
    let _enter = span.enter();
    
    let operation_logger = OperationLogger::new("batch_upscale");
    let performance_tracker = Arc::new(PerformanceTracker::new());
    let error_aggregator = Arc::new(ErrorAggregator::new());
    
    let input_dir = app_m
        .value_of("INPUT_DIR")
        .ok_or_else(|| SrganError::InvalidParameter("No input directory given".to_string()))?;
    let output_dir = app_m
        .value_of("OUTPUT_DIR")
        .ok_or_else(|| SrganError::InvalidParameter("No output directory given".to_string()))?;

    // Validate directories
    let input_path = validation::validate_directory(input_dir)?;
    let output_path = validation::validate_directory(output_dir)?;

    operation_logger.add_metadata("input_dir", input_path.display().to_string());
    operation_logger.add_metadata("output_dir", output_path.display().to_string());

    // Parse options
    let recursive = app_m.is_present("RECURSIVE");
    let parallel = !app_m.is_present("SEQUENTIAL");
    let skip_existing = app_m.is_present("SKIP_EXISTING");
    let pattern = app_m.value_of("PATTERN").unwrap_or("*.{png,jpg,jpeg,gif,bmp}");
    let max_retries = app_m.value_of("MAX_RETRIES")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(3);
    
    // Configure retry policy
    let retry_config = RetryConfig {
        max_attempts: max_retries,
        initial_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(5),
        exponential_base: 2.0,
        jitter: true,
    };
    
    // Set up circuit breaker for protection against cascading failures
    let circuit_breaker = Arc::new(CircuitBreaker::new(
        5,  // failure threshold
        3,  // success threshold to recover
        Duration::from_secs(30),  // timeout before retry
    ));
    
    // Parse thread configuration
    let num_threads = app_m.value_of("THREADS")
        .and_then(|s| s.parse::<usize>().ok());
    
    // Configure thread pool if specified
    if let Some(threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap_or_else(|e| {
                warn!("Failed to set thread pool size: {}. Using default.", e);
            });
    }
    
    // Load thread-safe network
    let factor = parse_factor(app_m);
    let thread_safe_network = Arc::new(
        performance_tracker.track("load_network", || {
            load_network(app_m, factor).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        })?
    );

    info!("Starting enhanced batch processing with error recovery");
    info!("Input directory: {}", input_path.display());
    info!("Output directory: {}", output_path.display());
    info!("Mode: {}", if parallel { "Parallel" } else { "Sequential" });
    info!("Max retries per image: {}", max_retries);

    // Collect image files
    let image_files = performance_tracker.track("collect_files", || {
        collect_image_files(&input_path, pattern, recursive)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    })?;
    
    if image_files.is_empty() {
        warn!("No image files found matching pattern: {}", pattern);
        return Ok(());
    }

    info!("Found {} images to process", image_files.len());
    increment_counter("srgan_batch_total_images", image_files.len() as u64);
    record_histogram("srgan_batch_size", image_files.len() as f64);

    // Create progress tracking
    let multi_progress = Arc::new(MultiProgress::new());
    let overall_pb = Arc::new(multi_progress.add(ProgressBar::new(image_files.len() as u64)));
    overall_pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
    );
    overall_pb.set_message("Processing images");

    let start_time = Instant::now();
    let successful = Arc::new(AtomicUsize::new(0));
    let skipped = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));

    set_gauge("srgan_active_jobs", 0.0);

    if parallel {
        let thread_count = num_threads.unwrap_or_else(|| rayon::current_num_threads());
        info!("Using parallel processing with {} threads", thread_count);
        
        // Process images in parallel with error recovery
        image_files.par_iter().for_each(|image_file| {
            let _span = span!(Level::DEBUG, "process_image", path = ?image_file);
            process_single_image_with_retry(
                image_file,
                &input_path,
                &output_path,
                &thread_safe_network,
                skip_existing,
                &overall_pb,
                &successful,
                &skipped,
                &failed,
                &retry_config,
                &circuit_breaker,
                &error_aggregator,
                &performance_tracker,
            );
        });
    } else {
        info!("Using sequential processing");
        
        // Process images sequentially with error recovery
        for image_file in &image_files {
            let _span = span!(Level::DEBUG, "process_image", path = ?image_file);
            process_single_image_with_retry(
                image_file,
                &input_path,
                &output_path,
                &thread_safe_network,
                skip_existing,
                &overall_pb,
                &successful,
                &skipped,
                &failed,
                &retry_config,
                &circuit_breaker,
                &error_aggregator,
                &performance_tracker,
            );
        }
    }

    overall_pb.finish_with_message("Batch processing complete");

    // Report results
    let duration = start_time.elapsed();
    let successful_count = successful.load(Ordering::Relaxed);
    let skipped_count = skipped.load(Ordering::Relaxed);
    let failed_count = failed.load(Ordering::Relaxed);
    
    info!(
        "Batch processing complete in {:.2}s",
        duration.as_secs_f32()
    );
    info!(
        "Results: {} successful, {} skipped, {} failed (of {} total)",
        successful_count, skipped_count, failed_count, image_files.len()
    );

    // Log error summary
    let error_summary = error_aggregator.get_summary();
    if error_summary.total_errors > 0 {
        error!("Error Summary:\n{}", error_summary);
    }

    // Log performance metrics
    performance_tracker.report();
    
    // Update final metrics
    increment_counter("srgan_images_processed", successful_count as u64);
    increment_counter("srgan_images_skipped", skipped_count as u64);
    increment_counter("srgan_images_failed", failed_count as u64);
    record_histogram("srgan_batch_duration_seconds", duration.as_secs_f64);
    
    operation_logger.add_metadata("successful", successful_count.to_string());
    operation_logger.add_metadata("failed", failed_count.to_string());
    operation_logger.add_metadata("skipped", skipped_count.to_string());
    operation_logger.complete();

    if failed_count > 0 {
        Err(SrganError::InvalidInput(
            format!("Failed to process {} images", failed_count)
        ))
    } else {
        Ok(())
    }
}

fn process_single_image_with_retry(
    image_path: &Path,
    input_base: &Path,
    output_base: &Path,
    network: &Arc<ThreadSafeNetwork>,
    skip_existing: bool,
    progress: &Arc<ProgressBar>,
    successful: &Arc<AtomicUsize>,
    skipped: &Arc<AtomicUsize>,
    failed: &Arc<AtomicUsize>,
    retry_config: &RetryConfig,
    circuit_breaker: &Arc<CircuitBreaker>,
    error_aggregator: &Arc<ErrorAggregator>,
    performance_tracker: &Arc<PerformanceTracker>,
) {
    increment_gauge("srgan_active_jobs", 1.0);
    
    let relative_path = image_path.strip_prefix(input_base).unwrap_or(image_path);
    let output_path = output_base.join(relative_path);
    
    // Check if we should skip
    if skip_existing && output_path.exists() {
        debug!("Skipping existing file: {}", output_path.display());
        skipped.fetch_add(1, Ordering::Relaxed);
        progress.inc(1);
        decrement_gauge("srgan_active_jobs", 1.0);
        return;
    }
    
    // Create output directory if needed
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                error_aggregator.record_error(
                    "directory_creation",
                    image_path.to_path_buf(),
                    format!("Failed to create directory: {}", e)
                );
                failed.fetch_add(1, Ordering::Relaxed);
                progress.inc(1);
                decrement_gauge("srgan_active_jobs", 1.0);
                return;
            }
        }
    }
    
    // Process with retry logic and circuit breaker
    let mut context = ErrorContext::new("image_upscale")
        .with_file(image_path.to_path_buf())
        .with_attempts(retry_config.max_attempts);
    
    let retry_executor = RetryExecutor::new(retry_config.clone());
    
    let process_result = tokio::runtime::Handle::current().block_on(async {
        circuit_breaker.call(|| {
            performance_tracker.track("process_image", || {
                process_image_internal(image_path, &output_path, network)
                    .map_err(|e| EnhancedError::transient(
                        format!("Failed to process image: {}", e),
                        Some(Duration::from_millis(500))
                    ))
            })
        }).await
    });
    
    match process_result {
        Ok(_) => {
            debug!("Successfully processed: {}", image_path.display());
            successful.fetch_add(1, Ordering::Relaxed);
            increment_counter("srgan_images_processed", 1);
        }
        Err(e) => {
            error_aggregator.record_error(
                "processing",
                image_path.to_path_buf(),
                format!("{}", e)
            );
            failed.fetch_add(1, Ordering::Relaxed);
            increment_counter("srgan_errors_total", 1);
        }
    }
    
    progress.inc(1);
    metrics::decrement_gauge!("srgan_active_jobs", 1.0);
}

fn process_image_internal(
    input_path: &Path,
    output_path: &Path,
    network: &Arc<ThreadSafeNetwork>,
) -> Result<()> {
    use image::{DynamicImage, ImageFormat};
    
    // Load the image
    let img = image::open(input_path)
        .map_err(|e| SrganError::Image(e))?;
    
    // Process with the network
    let upscaled = network.upscale_image(&img)?;
    
    // Save the result
    upscaled.save_with_format(output_path, ImageFormat::Png)
        .map_err(|e| SrganError::Image(e))?;
    
    Ok(())
}

fn collect_image_files(
    dir: &Path,
    pattern: &str,
    recursive: bool,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files_recursive(dir, pattern, recursive, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_recursive(
    dir: &Path,
    pattern: &str,
    recursive: bool,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    let entries = fs::read_dir(dir)
        .map_err(|e| SrganError::Io(e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| SrganError::Io(e))?;
        let path = entry.path();
        
        if path.is_dir() && recursive {
            collect_files_recursive(&path, pattern, recursive, files)?;
        } else if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if pattern.contains(&ext_str) {
                    files.push(path);
                }
            }
        }
    }
    
    Ok(())
}

fn parse_factor(app_m: &ArgMatches) -> Option<usize> {
    app_m.value_of("FACTOR")
        .and_then(|s| s.parse::<usize>().ok())
}

fn load_network(app_m: &ArgMatches, factor: Option<usize>) -> Result<ThreadSafeNetwork> {
    let network_type = app_m.value_of("NETWORK").unwrap_or("natural");
    
    match network_type {
        "natural" => ThreadSafeNetwork::load_builtin_natural(),
        "anime" => ThreadSafeNetwork::load_builtin_anime(),
        path if Path::new(path).exists() => {
            ThreadSafeNetwork::load_from_file(Path::new(path))
        }
        _ => Err(SrganError::InvalidParameter(
            format!("Unknown network type: {}", network_type)
        ))
    }
}

// Make EnhancedError implement IsRetryable for the existing SrganError
impl IsRetryable for SrganError {
    fn is_retryable(&self) -> bool {
        matches!(
            self,
            SrganError::Io(_) | SrganError::Network(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    #[test]
    fn test_collect_image_files() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();
        
        // Create test files
        fs::write(base_path.join("test1.png"), b"").unwrap();
        fs::write(base_path.join("test2.jpg"), b"").unwrap();
        fs::write(base_path.join("test3.txt"), b"").unwrap();
        
        let files = collect_image_files(base_path, "png,jpg", false).unwrap();
        assert_eq!(files.len(), 2);
    }
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(2, 2, Duration::from_millis(100));
        let mut failures = 0;
        
        // First two calls fail
        for _ in 0..2 {
            let result = breaker.call(|| {
                failures += 1;
                Err::<(), _>(EnhancedError::permanent("Test error"))
            }).await;
            assert!(result.is_err());
        }
        
        // Circuit should be open now
        let result = breaker.call(|| Ok::<_, EnhancedError>(42)).await;
        assert!(matches!(result, Err(EnhancedError::CircuitBreakerOpen { .. })));
        
        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Circuit should be half-open, call should succeed
        let result = breaker.call(|| Ok::<_, EnhancedError>(42)).await;
        assert!(result.is_ok());
    }
}