use crate::checkpoint::{
    checkpoint_path, load_checkpoint, save_checkpoint,
    global_checkpoint_path, load_global_checkpoint, save_global_checkpoint,
    BatchCheckpoint, BatchOptions,
};
use crate::error::{Result, SrganError};
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::validation;
use clap::ArgMatches;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{error, info, warn};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use uuid::Uuid;

// ── In-memory run state ────────────────────────────────────────────────────────

struct BatchRunState {
    checkpoint: BatchCheckpoint,
    /// Fast O(1) membership test for already-completed output paths.
    processed_set: HashSet<String>,
}

impl BatchRunState {
    fn is_done(&self, key: &str) -> bool {
        self.processed_set.contains(key)
    }

    fn mark_completed(&mut self, key: String) {
        self.processed_set.insert(key.clone());
        self.checkpoint.completed_images.push(key);
    }

    fn mark_failed(&mut self, key: String) {
        self.checkpoint.failed_images.push(key);
    }
}

// ── Main batch command ─────────────────────────────────────────────────────────

pub fn batch_upscale(app_m: &ArgMatches) -> Result<()> {
    let input_dir = app_m
        .value_of("INPUT_DIR")
        .ok_or_else(|| SrganError::InvalidParameter("No input directory given".to_string()))?;
    let output_dir = app_m
        .value_of("OUTPUT_DIR")
        .ok_or_else(|| SrganError::InvalidParameter("No output directory given".to_string()))?;

    let input_path = validation::validate_directory(input_dir)?;
    let output_path = validation::validate_directory(output_dir)?;

    let recursive = app_m.is_present("RECURSIVE");
    let parallel = !app_m.is_present("SEQUENTIAL");
    let skip_existing = app_m.is_present("SKIP_EXISTING");
    let resume_id = app_m.value_of("RESUME");
    let pattern = app_m.value_of("PATTERN").unwrap_or("*.{png,jpg,jpeg,gif,bmp}");
    let errors_log = app_m.value_of("ERRORS_LOG");
    let webhook_url = app_m.value_of("WEBHOOK");

    let num_threads = app_m
        .value_of("THREADS")
        .and_then(|s| s.parse::<usize>().ok());

    if let Some(threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap_or_else(|e| warn!("Failed to set thread pool size: {}", e));
    }

    // ── Determine batch ID ────────────────────────────────────────────────────
    let batch_id = app_m
        .value_of("BATCH_ID")
        .map(|s| s.to_string())
        .or_else(|| resume_id.map(|s| s.to_string()))
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    info!("Batch ID: {}", batch_id);

    let factor = parse_factor(app_m);
    let options = BatchOptions {
        parameters: app_m.value_of("PARAMETERS").map(|s| s.to_string()),
        custom_model: app_m.value_of("CUSTOM").map(|s| s.to_string()),
        factor,
        recursive,
        parallel,
        skip_existing,
    };

    // ── Load or create checkpoint ─────────────────────────────────────────────
    let run_state = if let Some(id) = resume_id {
        match load_checkpoint(id, &output_path)? {
            Some(cp) => {
                let n = cp.completed_images.len();
                info!("Resuming batch '{}': {} files already completed", id, n);
                let set = cp.completed_images.iter().cloned().collect();
                BatchRunState {
                    checkpoint: cp,
                    processed_set: set,
                }
            }
            None => {
                warn!("No checkpoint found for batch '{}'; starting fresh", id);
                BatchRunState {
                    checkpoint: BatchCheckpoint::new(
                        id.to_string(),
                        input_path.to_string_lossy().to_string(),
                        output_path.to_string_lossy().to_string(),
                        0,
                        options,
                    ),
                    processed_set: HashSet::new(),
                }
            }
        }
    } else {
        BatchRunState {
            checkpoint: BatchCheckpoint::new(
                batch_id.clone(),
                input_path.to_string_lossy().to_string(),
                output_path.to_string_lossy().to_string(),
                0,
                options,
            ),
            processed_set: HashSet::new(),
        }
    };

    let thread_safe_network = Arc::new(load_network(app_m, factor)?);

    // ── Collect files ─────────────────────────────────────────────────────────
    let image_files = collect_image_files(&input_path, pattern, recursive)?;

    if image_files.is_empty() {
        warn!("No image files found matching pattern: {}", pattern);
        return Ok(());
    }

    info!("Found {} images to process", image_files.len());

    let state_arc: Arc<Mutex<BatchRunState>> = Arc::new(Mutex::new({
        let mut s = run_state;
        s.checkpoint.total_images = image_files.len();
        s
    }));

    // ── Progress tracking ─────────────────────────────────────────────────────
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
    let errors: Arc<Mutex<Vec<(PathBuf, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let successful = Arc::new(AtomicUsize::new(0));
    let skipped = Arc::new(AtomicUsize::new(0));
    let timing: Arc<Mutex<(f64, usize)>> = Arc::new(Mutex::new((0.0, 0)));

    if parallel {
        let thread_count = num_threads.unwrap_or_else(|| rayon::current_num_threads());
        info!("Using parallel processing with {} threads", thread_count);

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
                &skipped,
                &timing,
                &state_arc,
            );
        });
    } else {
        info!("Using sequential processing");
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
                &skipped,
                &timing,
                &state_arc,
            );
        }
    }

    overall_pb.finish_with_message("Batch processing complete");

    let duration = start_time.elapsed();
    let successful_count = successful.load(Ordering::Relaxed);
    let skipped_count = skipped.load(Ordering::Relaxed);
    let error_list = errors
        .lock()
        .map_err(|_| SrganError::InvalidInput("Failed to acquire error lock".to_string()))?;

    info!(
        "Processed {} of {} images in {:.2}s ({} skipped, {} failed)",
        successful_count,
        image_files.len(),
        duration.as_secs_f32(),
        skipped_count,
        error_list.len()
    );

    // ── Write errors log ──────────────────────────────────────────────────────
    let log_target = errors_log
        .map(|s| s.to_string())
        .unwrap_or_else(|| output_path.join("batch_errors.log").to_string_lossy().to_string());

    if !error_list.is_empty() {
        write_errors_log(&log_target, &error_list);
        error!(
            "{} images failed — details written to {}",
            error_list.len(),
            log_target
        );
    }

    // ── Final checkpoint flush, then cleanup on success ───────────────────────
    if let Ok(mut s) = state_arc.lock() {
        let _ = save_checkpoint(&mut s.checkpoint, &output_path);
    }

    if error_list.is_empty() {
        let cp_file = checkpoint_path(&batch_id, &output_path);
        let _ = fs::remove_file(&cp_file);
        info!("Batch '{}' completed successfully; checkpoint removed", batch_id);
    }

    // ── Webhook notification ──────────────────────────────────────────────────
    if let Some(url) = webhook_url {
        send_webhook(
            url,
            successful_count,
            skipped_count,
            error_list.len(),
            image_files.len(),
            duration,
        );
    }

    Ok(())
}

// ── batch-status command ──────────────────────────────────────────────────────

/// Print progress for a running or interrupted batch job.
pub fn batch_status(app_m: &ArgMatches) -> Result<()> {
    let dir = app_m.value_of("DIR").unwrap_or(".");
    let dir_path = Path::new(dir);

    if let Some(id) = app_m.value_of("BATCH_ID") {
        match load_checkpoint(id, dir_path)? {
            Some(cp) => print_checkpoint_status(&cp),
            None => println!("No checkpoint found for batch '{}' in '{}'", id, dir),
        }
        return Ok(());
    }

    // Fallback: scan for any checkpoint file in the directory.
    let mut found = false;
    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(".srgan_checkpoint_") && name_str.ends_with(".json") {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if let Ok(cp) = serde_json::from_str::<BatchCheckpoint>(&content) {
                        print_checkpoint_status(&cp);
                        found = true;
                    }
                }
            }
        }
    }

    if !found {
        println!("No active batch job found in '{}'", dir);
        println!("(Expected: .srgan_checkpoint_<batch-id>.json)");
    }

    Ok(())
}

fn print_checkpoint_status(cp: &BatchCheckpoint) {
    let processed = cp.completed_images.len();
    let total = cp.total_images;
    let remaining = total.saturating_sub(processed);
    let pct = if total > 0 {
        processed as f32 / total as f32 * 100.0
    } else {
        0.0
    };

    println!("Batch ID:     {}", cp.batch_id);
    println!("Status:       {}/{} completed ({:.1}%)", processed, total, pct);
    println!("Failed:       {}", cp.failed_images.len());
    println!("Remaining:    {}", remaining);
    println!("Started at:   {}", cp.started_at);
    println!("Last updated: {}", cp.last_updated);
}

// ── Per-image processing (sequential) ────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn process_single_image(
    image_path: &Path,
    input_base: &Path,
    output_base: &Path,
    network: &Arc<ThreadSafeNetwork>,
    skip_existing: bool,
    progress: &Arc<ProgressBar>,
    errors: &Arc<Mutex<Vec<(PathBuf, String)>>>,
    successful: &Arc<AtomicUsize>,
    skipped: &Arc<AtomicUsize>,
    timing: &Arc<Mutex<(f64, usize)>>,
    state_arc: &Arc<Mutex<BatchRunState>>,
) {
    let relative = image_path.strip_prefix(input_base).unwrap_or(image_path);
    let output_path = output_base.join(relative).with_extension("png");
    let output_key = output_path.to_string_lossy().to_string();

    // Resume: skip if already in checkpoint
    if let Ok(s) = state_arc.lock() {
        if s.is_done(&output_key) {
            skipped.fetch_add(1, Ordering::Relaxed);
            progress.inc(1);
            return;
        }
    }

    if skip_existing && output_path.exists() {
        skipped.fetch_add(1, Ordering::Relaxed);
        progress.inc(1);
        progress.set_message(format!("Skipped: {}", relative.display()));
        return;
    }

    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                push_error(errors, image_path, &format!("mkdir failed: {}", e));
                record_failed(state_arc, output_base, image_path.to_string_lossy().to_string());
                progress.inc(1);
                return;
            }
        }
    }

    let t0 = Instant::now();
    match process_image(image_path, &output_path, network) {
        Ok(_) => {
            let elapsed = t0.elapsed().as_secs_f64();
            successful.fetch_add(1, Ordering::Relaxed);
            update_timing(timing, elapsed);
            record_completed(state_arc, output_base, output_key, elapsed);
            progress.inc(1);
            progress.set_message(format!("Processed: {}", relative.display()));
        }
        Err(e) => {
            push_error(errors, image_path, &e.to_string());
            record_failed(state_arc, output_base, image_path.to_string_lossy().to_string());
            progress.inc(1);
            progress.set_message(format!("Failed: {}", relative.display()));
        }
    }
}

// ── Per-image processing (parallel) ──────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn process_single_image_parallel(
    image_path: &Path,
    input_base: &Path,
    output_base: &Path,
    thread_safe_network: &Arc<ThreadSafeNetwork>,
    skip_existing: bool,
    progress: &Arc<ProgressBar>,
    errors: &Arc<Mutex<Vec<(PathBuf, String)>>>,
    successful: &Arc<AtomicUsize>,
    skipped: &Arc<AtomicUsize>,
    timing: &Arc<Mutex<(f64, usize)>>,
    state_arc: &Arc<Mutex<BatchRunState>>,
) {
    let relative = image_path.strip_prefix(input_base).unwrap_or(image_path);
    let output_path = output_base.join(relative).with_extension("png");
    let output_key = output_path.to_string_lossy().to_string();

    if let Ok(s) = state_arc.lock() {
        if s.is_done(&output_key) {
            skipped.fetch_add(1, Ordering::Relaxed);
            progress.inc(1);
            return;
        }
    }

    if skip_existing && output_path.exists() {
        skipped.fetch_add(1, Ordering::Relaxed);
        progress.inc(1);
        return;
    }

    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                push_error(errors, image_path, &format!("mkdir failed: {}", e));
                record_failed(state_arc, output_base, image_path.to_string_lossy().to_string());
                progress.inc(1);
                return;
            }
        }
    }

    let t0 = Instant::now();
    match process_image(image_path, &output_path, thread_safe_network) {
        Ok(_) => {
            let elapsed = t0.elapsed().as_secs_f64();
            successful.fetch_add(1, Ordering::Relaxed);
            update_timing(timing, elapsed);
            record_completed(state_arc, output_base, output_key, elapsed);
            progress.inc(1);
            progress.set_message(format!("Processed: {}", relative.display()));
        }
        Err(e) => {
            push_error(errors, image_path, &e.to_string());
            record_failed(state_arc, output_base, image_path.to_string_lossy().to_string());
            progress.inc(1);
            progress.set_message(format!("Failed: {}", relative.display()));
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn process_image(input_path: &Path, output_path: &Path, network: &ThreadSafeNetwork) -> Result<()> {
    let img = image::open(input_path).map_err(SrganError::Image)?;
    let upscaled = network.upscale_image(&img)?;
    upscaled.save(output_path).map_err(SrganError::Io)?;
    Ok(())
}

fn push_error(errors: &Arc<Mutex<Vec<(PathBuf, String)>>>, path: &Path, msg: &str) {
    if let Ok(mut errs) = errors.lock() {
        errs.push((path.to_path_buf(), msg.to_string()));
    }
}

fn update_timing(timing: &Arc<Mutex<(f64, usize)>>, elapsed: f64) {
    if let Ok(mut t) = timing.lock() {
        t.0 += elapsed;
        t.1 += 1;
    }
}

fn record_completed(
    state_arc: &Arc<Mutex<BatchRunState>>,
    checkpoint_dir: &Path,
    key: String,
    _elapsed: f64,
) {
    if let Ok(mut s) = state_arc.lock() {
        s.mark_completed(key);
        // Flush checkpoint every 10 completions to limit I/O.
        if s.checkpoint.completed_images.len() % 10 == 0 {
            let _ = save_checkpoint(&mut s.checkpoint, checkpoint_dir);
        }
    }
}

fn record_failed(
    state_arc: &Arc<Mutex<BatchRunState>>,
    checkpoint_dir: &Path,
    key: String,
) {
    if let Ok(mut s) = state_arc.lock() {
        s.mark_failed(key);
        let _ = save_checkpoint(&mut s.checkpoint, checkpoint_dir);
    }
}

fn write_errors_log(log_path: &str, errors: &[(PathBuf, String)]) {
    match fs::File::create(log_path) {
        Ok(mut f) => {
            for (path, err) in errors {
                let _ = writeln!(f, "{}\t{}", path.display(), err);
            }
        }
        Err(e) => error!("Could not write errors log '{}': {}", log_path, e),
    }
}

fn send_webhook(
    url: &str,
    successful: usize,
    skipped: usize,
    failed: usize,
    total: usize,
    duration: Duration,
) {
    let payload = format!(
        r#"{{"event":"batch_complete","total":{},"successful":{},"skipped":{},"failed":{},"duration_secs":{:.2}}}"#,
        total,
        successful,
        skipped,
        failed,
        duration.as_secs_f64()
    );
    match ureq::post(url)
        .set("Content-Type", "application/json")
        .send_string(&payload)
    {
        Ok(_) => info!("Webhook notification sent to {}", url),
        Err(e) => warn!("Webhook POST to {} failed: {}", url, e),
    }
}

pub fn collect_image_files(dir: &Path, pattern: &str, recursive: bool) -> Result<Vec<PathBuf>> {
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

// ── batch start ───────────────────────────────────────────────────────────────

/// `srgan-rust batch start --input-dir DIR --output-dir DIR [--model MODEL] [--scale N]`
///
/// Creates a checkpoint in `~/.srgan-rust/checkpoints/<uuid>.json` and begins
/// processing.  The job can be resumed after a crash with `batch resume <ID>`.
pub fn batch_start(app_m: &ArgMatches) -> Result<()> {
    let input_dir = app_m
        .value_of("input-dir")
        .ok_or_else(|| SrganError::InvalidParameter("No --input-dir given".to_string()))?;
    let output_dir = app_m
        .value_of("output-dir")
        .ok_or_else(|| SrganError::InvalidParameter("No --output-dir given".to_string()))?;

    let model = app_m.value_of("model").unwrap_or("natural");
    let factor = app_m
        .value_of("scale")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);
    let recursive = app_m.is_present("recursive");
    let parallel = !app_m.is_present("sequential");

    if let Some(t) = app_m.value_of("threads").and_then(|s| s.parse::<usize>().ok()) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .unwrap_or_else(|e| warn!("Failed to set thread pool size: {}", e));
    }

    let options = BatchOptions {
        parameters: Some(model.to_string()),
        custom_model: None,
        factor,
        recursive,
        parallel,
        skip_existing: false,
    };

    let batch_id = Uuid::new_v4().to_string();
    let input_path = validation::validate_directory(input_dir)?;
    let output_path = validation::validate_directory(output_dir)?;
    let image_files = collect_image_files(&input_path, "", recursive)?;

    let mut checkpoint = BatchCheckpoint::new(
        batch_id.clone(),
        input_dir.to_string(),
        output_dir.to_string(),
        image_files.len(),
        options,
    );
    save_global_checkpoint(&mut checkpoint)?;

    println!("Batch job started:  {}", batch_id);
    println!(
        "Images:             {} (model={}, scale={}x)",
        image_files.len(), model, factor
    );
    println!(
        "Checkpoint:         {}",
        global_checkpoint_path(&batch_id).display()
    );

    let network = Arc::new(ThreadSafeNetwork::from_label(model, Some(factor))?);
    run_global_batch_job(checkpoint, &input_path, &output_path, &image_files, &network, parallel)
}

// ── batch resume ──────────────────────────────────────────────────────────────

/// `srgan-rust batch resume <BATCH_ID>`
pub fn batch_resume(app_m: &ArgMatches) -> Result<()> {
    let batch_id = app_m
        .value_of("BATCH_ID")
        .ok_or_else(|| SrganError::InvalidParameter("No batch ID given".to_string()))?;

    let checkpoint = load_global_checkpoint(batch_id)?
        .ok_or_else(|| SrganError::FileNotFound(global_checkpoint_path(batch_id)))?;

    let input_path = validation::validate_directory(&checkpoint.input_dir)?;
    let output_path = validation::validate_directory(&checkpoint.output_dir)?;

    let recursive = checkpoint.options.recursive;
    let parallel = checkpoint.options.parallel;
    let factor = checkpoint.options.factor;
    let model = checkpoint.options.parameters.as_deref().unwrap_or("natural").to_string();

    let completed_set: HashSet<String> =
        checkpoint.completed_images.iter().cloned().collect();

    let image_files = collect_image_files(&input_path, "", recursive)?;
    let pending: Vec<PathBuf> = image_files
        .into_iter()
        .filter(|f| {
            let relative = f.strip_prefix(&input_path).unwrap_or(f);
            let key = output_path
                .join(relative)
                .with_extension("png")
                .to_string_lossy()
                .to_string();
            !completed_set.contains(&key)
        })
        .collect();

    println!(
        "Resuming batch {}: {}/{} done, {} remaining",
        batch_id,
        checkpoint.completed_images.len(),
        checkpoint.total_images,
        pending.len()
    );

    if pending.is_empty() {
        println!("All images already processed.");
        return Ok(());
    }

    let network = Arc::new(ThreadSafeNetwork::from_label(&model, Some(factor))?);
    run_global_batch_job(checkpoint, &input_path, &output_path, &pending, &network, parallel)
}

// ── batch status (by global checkpoint ID) ────────────────────────────────────

/// `srgan-rust batch status <BATCH_ID>`
pub fn batch_status_by_id(app_m: &ArgMatches) -> Result<()> {
    let batch_id = app_m
        .value_of("BATCH_ID")
        .ok_or_else(|| SrganError::InvalidParameter("No batch ID given".to_string()))?;

    let cp = load_global_checkpoint(batch_id)?
        .ok_or_else(|| SrganError::FileNotFound(global_checkpoint_path(batch_id)))?;

    let done = cp.completed_images.len();
    let failed = cp.failed_images.len();
    let total = cp.total_images;
    let remaining = total.saturating_sub(done + failed);
    let pct = if total > 0 { done as f32 / total as f32 * 100.0 } else { 0.0 };

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:50.cyan/blue}] {pos}/{len} ({percent}%)")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("##-"),
    );
    pb.set_position(done as u64);
    pb.abandon();

    println!("Batch ID:   {}", cp.batch_id);
    println!("Input:      {}", cp.input_dir);
    println!("Output:     {}", cp.output_dir);
    println!(
        "Model:      {} ({}x)",
        cp.options.parameters.as_deref().unwrap_or("natural"),
        cp.options.factor
    );
    println!("Progress:   {}/{} ({:.1}%)", done, total, pct);
    if failed > 0 {
        println!("Failed:     {}", failed);
    }
    if remaining > 0 {
        println!("Remaining:  {}", remaining);
        if done > 0 {
            let elapsed = unix_secs_elapsed(cp.started_at);
            let secs_per_img = elapsed as f64 / done as f64;
            let eta = secs_per_img * remaining as f64;
            if eta < 120.0 {
                println!("ETA:        {:.0}s", eta);
            } else {
                println!("ETA:        {:.1} min", eta / 60.0);
            }
        }
    } else {
        println!("Status:     complete");
    }

    Ok(())
}

fn unix_secs_elapsed(started_at: u64) -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    now.saturating_sub(started_at)
}

// ── batch list ────────────────────────────────────────────────────────────────

/// `srgan-rust batch list`
pub fn batch_list() -> Result<()> {
    use crate::checkpoint::{global_checkpoint_dir, list_global_checkpoints};

    let entries = list_global_checkpoints()?;
    if entries.is_empty() {
        println!(
            "No batch checkpoints found in {}.",
            global_checkpoint_dir().display()
        );
        return Ok(());
    }

    println!(
        "{:<38} {:>7} {:>7} {:>7}  {}",
        "ID", "Total", "Done", "Failed", "Input Dir"
    );
    println!("{}", "-".repeat(84));
    for cp in &entries {
        let done = cp.completed_images.len();
        let failed = cp.failed_images.len();
        let status = if done + failed >= cp.total_images && cp.total_images > 0 {
            "complete"
        } else {
            "running"
        };
        println!(
            "{:<38} {:>7} {:>7} {:>7}  {} [{}]",
            cp.batch_id, cp.total_images, done, failed, cp.input_dir, status,
        );
    }

    Ok(())
}

// ── Shared processing loop for batch start / resume ───────────────────────────

fn run_global_batch_job(
    checkpoint: BatchCheckpoint,
    input_base: &Path,
    output_base: &Path,
    image_files: &[PathBuf],
    network: &Arc<ThreadSafeNetwork>,
    parallel: bool,
) -> Result<()> {
    if image_files.is_empty() {
        println!("Nothing to process.");
        return Ok(());
    }

    let pb = Arc::new(ProgressBar::new(image_files.len() as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
    );

    let cp_arc: Arc<Mutex<BatchCheckpoint>> = Arc::new(Mutex::new(checkpoint));
    let errors: Arc<Mutex<Vec<(PathBuf, String)>>> = Arc::new(Mutex::new(Vec::new()));

    if parallel {
        image_files.par_iter().for_each(|f| {
            run_global_batch_one(f, input_base, output_base, network, &pb, &cp_arc, &errors);
        });
    } else {
        for f in image_files {
            run_global_batch_one(f, input_base, output_base, network, &pb, &cp_arc, &errors);
        }
    }

    pb.finish_with_message("done");

    let error_list = errors
        .lock()
        .map_err(|_| SrganError::InvalidInput("Failed to acquire error lock".to_string()))?;
    let done_count = cp_arc.lock().map(|c| c.completed_images.len()).unwrap_or(0);

    println!("Finished: {} processed, {} failed", done_count, error_list.len());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_global_batch_one(
    image_file: &Path,
    input_base: &Path,
    output_base: &Path,
    network: &Arc<ThreadSafeNetwork>,
    pb: &Arc<ProgressBar>,
    cp_arc: &Arc<Mutex<BatchCheckpoint>>,
    errors: &Arc<Mutex<Vec<(PathBuf, String)>>>,
) {
    let relative = image_file.strip_prefix(input_base).unwrap_or(image_file);
    let output_path = output_base.join(relative).with_extension("png");
    let output_key = output_path.to_string_lossy().to_string();

    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                push_error(errors, image_file, &format!("mkdir failed: {}", e));
                pb.inc(1);
                return;
            }
        }
    }

    match process_image(image_file, &output_path, network) {
        Ok(_) => {
            pb.set_message(format!("OK: {}", relative.display()));
            pb.inc(1);
            if let Ok(mut cp) = cp_arc.lock() {
                cp.completed_images.push(output_key);
                let _ = save_global_checkpoint(&mut cp);
            }
        }
        Err(e) => {
            pb.set_message(format!("FAIL: {}", relative.display()));
            pb.inc(1);
            push_error(errors, image_file, &e.to_string());
            if let Ok(mut cp) = cp_arc.lock() {
                cp.failed_images.push(image_file.to_string_lossy().to_string());
                let _ = save_global_checkpoint(&mut cp);
            }
        }
    }
}
