use crate::error::{Result, SrganError};
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::validation;
use clap::ArgMatches;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{error, info, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ── BatchState — persisted to batch_state.json for resume support ─────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchState {
    /// Absolute output paths that have been successfully processed.
    pub processed: HashSet<String>,
    /// Total files in this run.
    pub total: usize,
    /// Unix timestamp (seconds) when the run started.
    pub started_at: u64,
    /// Running average seconds-per-image (updated as files complete).
    pub avg_secs_per_image: f64,
}

impl BatchState {
    fn load(state_path: &Path) -> Self {
        if state_path.exists() {
            if let Ok(s) = fs::read_to_string(state_path) {
                if let Ok(state) = serde_json::from_str::<BatchState>(&s) {
                    return state;
                }
            }
        }
        BatchState::default()
    }

    fn save(&self, state_path: &Path) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(state_path, json);
        }
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
    let resume = app_m.is_present("RESUME");
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

    let factor = parse_factor(app_m);
    let thread_safe_network = Arc::new(load_network(app_m, factor)?);

    // ── State file for resume ─────────────────────────────────────────────────
    let state_path = output_path.join("batch_state.json");
    let mut state = if resume {
        let s = BatchState::load(&state_path);
        if !s.processed.is_empty() {
            info!("Resuming: {} files already processed", s.processed.len());
        }
        s
    } else {
        BatchState::default()
    };

    // ── Collect files ─────────────────────────────────────────────────────────
    let image_files = collect_image_files(&input_path, pattern, recursive)?;

    if image_files.is_empty() {
        warn!("No image files found matching pattern: {}", pattern);
        return Ok(());
    }

    info!("Found {} images to process", image_files.len());
    state.total = image_files.len();

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
    // Per-image timing for ETA (sum of durations, count)
    let timing: Arc<Mutex<(f64, usize)>> = Arc::new(Mutex::new((0.0, 0)));

    // Wrap state in a mutex so parallel workers can mark completed files.
    let state_arc: Arc<Mutex<BatchState>> = Arc::new(Mutex::new(state));

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
                &state_path,
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
                &state_path,
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

    // ── Remove state file on clean completion ─────────────────────────────────
    if error_list.is_empty() {
        let _ = fs::remove_file(&state_path);
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

/// Read `batch_state.json` from `output_dir` and print current progress.
pub fn batch_status(app_m: &ArgMatches) -> Result<()> {
    let dir = app_m.value_of("DIR").unwrap_or(".");
    let state_path = Path::new(dir).join("batch_state.json");

    if !state_path.exists() {
        println!("No active batch job found in '{}'", dir);
        println!("(Expected: {})", state_path.display());
        return Ok(());
    }

    let state = BatchState::load(&state_path);
    let processed = state.processed.len();
    let total = state.total;
    let remaining = total.saturating_sub(processed);
    let pct = if total > 0 {
        processed as f32 / total as f32 * 100.0
    } else {
        0.0
    };

    println!("Batch status: {}/{} processed ({:.1}%)", processed, total, pct);
    println!("Remaining:    {}", remaining);

    if state.avg_secs_per_image > 0.0 && remaining > 0 {
        let eta_secs = state.avg_secs_per_image * remaining as f64;
        let eta_mins = eta_secs / 60.0;
        if eta_mins < 2.0 {
            println!("ETA:          {:.0}s", eta_secs);
        } else {
            println!("ETA:          {:.1} min", eta_mins);
        }
    }

    Ok(())
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
    state_arc: &Arc<Mutex<BatchState>>,
    state_path: &Path,
) {
    let relative = image_path.strip_prefix(input_base).unwrap_or(image_path);
    let output_path = output_base.join(relative).with_extension("png");
    let output_key = output_path.to_string_lossy().to_string();

    // Resume: skip if in state
    if let Ok(s) = state_arc.lock() {
        if s.processed.contains(&output_key) {
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
            mark_processed(state_arc, state_path, output_key, elapsed);
            progress.inc(1);
            progress.set_message(format!("Processed: {}", relative.display()));
        }
        Err(e) => {
            push_error(errors, image_path, &e.to_string());
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
    state_arc: &Arc<Mutex<BatchState>>,
    state_path: &Path,
) {
    let relative = image_path.strip_prefix(input_base).unwrap_or(image_path);
    let output_path = output_base.join(relative).with_extension("png");
    let output_key = output_path.to_string_lossy().to_string();

    if let Ok(s) = state_arc.lock() {
        if s.processed.contains(&output_key) {
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
            mark_processed(state_arc, state_path, output_key, elapsed);
            progress.inc(1);
            progress.set_message(format!("Processed: {}", relative.display()));
        }
        Err(e) => {
            push_error(errors, image_path, &e.to_string());
            progress.inc(1);
            progress.set_message(format!("Failed: {}", relative.display()));
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn process_image(input_path: &Path, output_path: &Path, network: &ThreadSafeNetwork) -> Result<()> {
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

fn mark_processed(
    state_arc: &Arc<Mutex<BatchState>>,
    state_path: &Path,
    key: String,
    elapsed: f64,
) {
    if let Ok(mut s) = state_arc.lock() {
        s.processed.insert(key);
        let n = s.processed.len() as f64;
        // Exponential moving average of time-per-image.
        s.avg_secs_per_image = s.avg_secs_per_image * (n - 1.0) / n + elapsed / n;
        // Persist every 10 completions to limit I/O.
        if s.processed.len() % 10 == 0 {
            s.save(state_path);
        }
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
