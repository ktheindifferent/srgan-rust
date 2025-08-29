use crate::error::{Result, SrganError};
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::validation;
use clap::ArgMatches;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{error, info, warn};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub fn batch_upscale(app_m: &ArgMatches) -> Result<()> {
    let input_dir = app_m
        .value_of("INPUT_DIR")
        .ok_or_else(|| SrganError::InvalidParameter("No input directory given".to_string()))?;
    let output_dir = app_m
        .value_of("OUTPUT_DIR")
        .ok_or_else(|| SrganError::InvalidParameter("No output directory given".to_string()))?;

    // Validate directories
    let input_path = validation::validate_directory(input_dir)?;
    let output_path = validation::validate_directory(output_dir)?;

    // Parse options
    let recursive = app_m.is_present("RECURSIVE");
    let parallel = !app_m.is_present("SEQUENTIAL");
    let skip_existing = app_m.is_present("SKIP_EXISTING");
    let pattern = app_m.value_of("PATTERN").unwrap_or("*.{png,jpg,jpeg,gif,bmp}");
    
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
    
    // Load thread-safe network - no Arc<Mutex<>> needed!
    let factor = parse_factor(app_m);
    let thread_safe_network = Arc::new(load_network(app_m, factor)?);

    info!("Starting batch processing");
    info!("Input directory: {}", input_path.display());
    info!("Output directory: {}", output_path.display());
    info!("Mode: {}", if parallel { "Parallel" } else { "Sequential" });

    // Collect image files
    let image_files = collect_image_files(&input_path, pattern, recursive)?;
    
    if image_files.is_empty() {
        warn!("No image files found matching pattern: {}", pattern);
        return Ok(());
    }

    info!("Found {} images to process", image_files.len());

    // Create progress tracking
    let multi_progress = Arc::new(MultiProgress::new());
    let overall_pb = Arc::new(multi_progress.add(ProgressBar::new(image_files.len() as u64)));
    overall_pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    overall_pb.set_message("Processing images");

    let start_time = Instant::now();
    let errors = Arc::new(Mutex::new(Vec::<(PathBuf, String)>::new()));
    let successful = Arc::new(AtomicUsize::new(0));

    if parallel {
        let thread_count = num_threads.unwrap_or_else(|| rayon::current_num_threads());
        info!("Using parallel processing with {} threads", thread_count);
        
        // Process images in parallel
        image_files.par_iter().for_each(|image_file| {
            process_single_image_parallel(
                image_file,
                &input_path,
                &output_path,
                &thread_safe_network,
                skip_existing,
                &overall_pb,
                &errors,
                &successful,
            );
        });
    } else {
        info!("Using sequential processing");
        
        // Process images sequentially
        for image_file in &image_files {
            process_single_image(
                image_file,
                &input_path,
                &output_path,
                &thread_safe_network,
                skip_existing,
                &overall_pb,
                &errors,
                &successful,
            );
        }
    }

    overall_pb.finish_with_message("Batch processing complete");

    // Report results
    let duration = start_time.elapsed();
    let successful_count = successful.load(Ordering::Relaxed);
    let error_list = errors.lock()
        .map_err(|_| SrganError::InvalidInput("Failed to acquire error lock".to_string()))?;
    
    info!(
        "Processed {} of {} images in {:.2}s",
        successful_count,
        image_files.len(),
        duration.as_secs_f32()
    );

    if !error_list.is_empty() {
        error!("Failed to process {} images:", error_list.len());
        for (path, err) in error_list.iter() {
            error!("  {}: {}", path.display(), err);
        }
    }

    Ok(())
}

fn process_single_image(
    image_path: &Path,
    input_base: &Path,
    output_base: &Path,
    network: &Arc<ThreadSafeNetwork>,
    skip_existing: bool,
    progress: &Arc<ProgressBar>,
    errors: &Arc<Mutex<Vec<(PathBuf, String)>>>,
    successful: &Arc<AtomicUsize>,
) {
    // Calculate relative path and output path
    let relative_path = image_path
        .strip_prefix(input_base)
        .unwrap_or(image_path);
    
    let output_path = output_base.join(relative_path);
    let output_path = output_path.with_extension("png"); // Always save as PNG

    // Skip if exists and skip_existing is true
    if skip_existing && output_path.exists() {
        progress.inc(1);
        progress.set_message(format!("Skipped: {}", relative_path.display()));
        return;
    }

    // Create output directory if needed
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                if let Ok(mut errs) = errors.lock() {
                    errs.push((
                    image_path.to_path_buf(),
                        format!("Failed to create output directory: {}", e),
                    ));
                }
                progress.inc(1);
                return;
            }
        }
    }

    // Process the image
    match process_image(image_path, &output_path, network) {
        Ok(_) => {
            successful.fetch_add(1, Ordering::Relaxed);
            progress.inc(1);
            progress.set_message(format!("Processed: {}", relative_path.display()));
        }
        Err(e) => {
            if let Ok(mut errs) = errors.lock() {
                errs.push((image_path.to_path_buf(), e.to_string()));
            }
            progress.inc(1);
            progress.set_message(format!("Failed: {}", relative_path.display()));
        }
    }
}

fn process_single_image_parallel(
    image_path: &Path,
    input_base: &Path,
    output_base: &Path,
    thread_safe_network: &Arc<ThreadSafeNetwork>,
    skip_existing: bool,
    progress: &Arc<ProgressBar>,
    errors: &Arc<Mutex<Vec<(PathBuf, String)>>>,
    successful: &Arc<AtomicUsize>,
) {
    // Calculate relative path and output path
    let relative_path = image_path
        .strip_prefix(input_base)
        .unwrap_or(image_path);
    
    let output_path = output_base.join(relative_path);
    let output_path = output_path.with_extension("png"); // Always save as PNG

    // Skip if exists and skip_existing is true
    if skip_existing && output_path.exists() {
        progress.inc(1);
        progress.set_message(format!("Skipped: {}", relative_path.display()));
        return;
    }

    // Create output directory if needed
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                if let Ok(mut errs) = errors.lock() {
                    errs.push((
                        image_path.to_path_buf(),
                        format!("Failed to create output directory: {}", e),
                    ));
                }
                progress.inc(1);
                return;
            }
        }
    }

    // Process the image
    match process_image(image_path, &output_path, thread_safe_network) {
        Ok(_) => {
            successful.fetch_add(1, Ordering::Relaxed);
            progress.inc(1);
            progress.set_message(format!("Processed: {}", relative_path.display()));
        }
        Err(e) => {
            if let Ok(mut errs) = errors.lock() {
                errs.push((image_path.to_path_buf(), e.to_string()));
            }
            progress.inc(1);
            progress.set_message(format!("Failed: {}", relative_path.display()));
        }
    }
}

fn process_image(input_path: &Path, output_path: &Path, network: &ThreadSafeNetwork) -> Result<()> {
    // Load image
    let img = image::open(input_path)
        .map_err(|e| SrganError::Image(e))?;
    
    // Upscale using thread-safe network
    let upscaled = network.upscale_image(&img)?;
    
    // Save output (convert io::Error to string)
    upscaled.save(output_path)
        .map_err(|e| SrganError::Io(e))?;
    
    Ok(())
}

fn collect_image_files(dir: &Path, pattern: &str, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    
    if recursive {
        collect_files_recursive(dir, pattern, &mut files)?;
    } else {
        collect_files_in_dir(dir, pattern, &mut files)?;
    }
    
    files.sort();
    Ok(files)
}

fn collect_files_recursive(dir: &Path, pattern: &str, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            collect_files_recursive(&path, pattern, files)?;
        } else if is_image_file(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn collect_files_in_dir(dir: &Path, _pattern: &str, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && is_image_file(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_image_file(path: &Path) -> bool {
    let valid_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"];
    
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| valid_extensions.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn parse_factor(app_m: &ArgMatches) -> usize {
    app_m
        .value_of("FACTOR")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4)
}

fn load_network(app_m: &ArgMatches, factor: usize) -> Result<ThreadSafeNetwork> {
    if let Some(file_str) = app_m.value_of("CUSTOM") {
        let param_path = validation::validate_input_file(file_str)?;
        ThreadSafeNetwork::load_from_file(&param_path)
    } else {
        let param_type = app_m.value_of("PARAMETERS").unwrap_or("natural");
        ThreadSafeNetwork::from_label(param_type, Some(factor))
    }
}