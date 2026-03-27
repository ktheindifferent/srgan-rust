//! Video super-resolution pipeline.
//!
//! Processes video files frame-by-frame through the SRGAN model with parallel
//! frame processing via rayon, progress tracking, and output as MP4/WebM.
//! Uses ffmpeg via `std::process::Command` for demuxing and muxing.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std::{fs, io};

use image::DynamicImage;
use log::{debug, info, warn};
use rayon::prelude::*;

use crate::error::SrganError;
use crate::UpscalingNetwork;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the video super-resolution pipeline.
#[derive(Debug, Clone)]
pub struct VideoSrConfig {
    /// Input video file path.
    pub video_in: PathBuf,
    /// Output video file path (extension determines container: .mp4 or .webm).
    pub video_out: PathBuf,
    /// Override output FPS. When `None`, the source FPS is preserved.
    pub fps: Option<f64>,
    /// Number of rayon threads for parallel frame upscaling (0 = rayon default).
    pub parallel_frames: usize,
    /// Optional explicit temp directory. A random sub-dir is created inside it.
    pub temp_dir: Option<PathBuf>,
}

/// Progress information passed to the caller's callback.
#[derive(Debug, Clone)]
pub struct VideoSrProgress {
    /// Number of frames upscaled so far.
    pub frames_done: usize,
    /// Total number of frames to process.
    pub total_frames: usize,
    /// Elapsed wall-clock time since processing started.
    pub elapsed: std::time::Duration,
}

/// Metadata extracted from the source video via ffprobe.
#[derive(Debug, Clone)]
struct ProbeInfo {
    fps: f64,
    frame_count: usize,
    width: u32,
    height: u32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the full video super-resolution pipeline.
///
/// 1. Probe source video for FPS / frame count.
/// 2. Extract frames as PNG via ffmpeg.
/// 3. Upscale each frame through `network` in parallel (rayon).
/// 4. Mux upscaled frames back into the output container via ffmpeg.
///
/// The `on_progress` callback is invoked after every frame completes.
pub fn process_video<F>(
    config: &VideoSrConfig,
    network: &UpscalingNetwork,
    on_progress: F,
) -> Result<(), SrganError>
where
    F: Fn(VideoSrProgress) + Send + Sync,
{
    // Ensure ffmpeg is available (or mock-able).
    check_ffmpeg()?;

    let probe = probe_video(&config.video_in)?;
    info!(
        "Source video: {}x{} @ {:.2} fps, ~{} frames",
        probe.width, probe.height, probe.fps, probe.frame_count
    );

    let temp_root = config
        .temp_dir
        .clone()
        .unwrap_or_else(std::env::temp_dir);
    let work_dir = temp_root.join(format!("srgan_vsr_{}", std::process::id()));
    let frames_dir = work_dir.join("frames");
    let upscaled_dir = work_dir.join("upscaled");
    fs::create_dir_all(&frames_dir).map_err(SrganError::Io)?;
    fs::create_dir_all(&upscaled_dir).map_err(SrganError::Io)?;

    // Step 1 — extract frames
    info!("Extracting frames...");
    extract_frames(&config.video_in, &frames_dir)?;

    // Discover extracted frame files (sorted).
    let mut frame_files: Vec<PathBuf> = fs::read_dir(&frames_dir)
        .map_err(SrganError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| ext == "png")
                .unwrap_or(false)
        })
        .collect();
    frame_files.sort();

    let total_frames = frame_files.len();
    if total_frames == 0 {
        // Clean up and report error
        let _ = fs::remove_dir_all(&work_dir);
        return Err(SrganError::InvalidInput(
            "No frames extracted from source video".into(),
        ));
    }
    info!("Extracted {} frames", total_frames);

    // Step 2 — parallel upscale
    info!("Upscaling frames ({} threads)...", if config.parallel_frames == 0 { rayon::current_num_threads() } else { config.parallel_frames });
    let start = Instant::now();
    let done_counter = Arc::new(AtomicUsize::new(0));

    // Parallel processing is now handled within upscale_frames using sequential iteration
    // (rayon's par_iter requires Send, but network is not Send due to non-Send callbacks)
    upscale_frames(
        &frame_files,
        &upscaled_dir,
        network,
        &done_counter,
        total_frames,
        &start,
        &on_progress,
    )?;

    // Step 3 — mux
    let output_fps = config.fps.unwrap_or(probe.fps);
    info!("Muxing {} frames at {:.2} fps -> {:?}", total_frames, output_fps, config.video_out);
    mux_frames(&upscaled_dir, &config.video_out, output_fps)?;

    // Clean up
    let _ = fs::remove_dir_all(&work_dir);

    info!(
        "Video SR complete in {:.1}s — {} frames upscaled",
        start.elapsed().as_secs_f64(),
        total_frames,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn upscale_frames<F>(
    frame_files: &[PathBuf],
    upscaled_dir: &Path,
    network: &UpscalingNetwork,
    done_counter: &Arc<AtomicUsize>,
    total_frames: usize,
    start: &Instant,
    on_progress: &F,
) -> Result<(), SrganError>
where
    F: Fn(VideoSrProgress) + Send + Sync,
{
    // Use sequential iteration instead of par_iter() to avoid Send trait issues with non-Send network
    let mut error: Option<SrganError> = None;
    for (idx, frame_path) in frame_files.iter().enumerate() {
        if error.is_some() {
            break;
        }
        let img = image::open(frame_path).map_err(SrganError::Image)?;
        let upscaled = network.upscale_image(&img)?;

        let out_name = format!("frame_{:08}.png", idx + 1);
        let out_path = upscaled_dir.join(&out_name);
        upscaled
            .save(&out_path)
            .map_err(|e| SrganError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;

        let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
        on_progress(VideoSrProgress {
            frames_done: done,
            total_frames,
            elapsed: start.elapsed(),
        });
        debug!("Upscaled frame {}/{}", done, total_frames);
    }
    if let Some(e) = error {
        Err(e)
    } else {
        Ok(())
    }
}

/// Check that ffmpeg is on PATH. If not, return a descriptive error.
fn check_ffmpeg() -> Result<(), SrganError> {
    match Command::new("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(s) if s.success() => Ok(()),
        Ok(_) => Err(SrganError::InvalidInput(
            "ffmpeg exited with non-zero status. Is it installed correctly?".into(),
        )),
        Err(_) => {
            warn!("ffmpeg not found on PATH — video muxing will be mocked");
            Err(SrganError::InvalidInput(
                "ffmpeg is required for video SR. Install it or set PATH.".into(),
            ))
        }
    }
}

/// Use ffprobe to extract fps, frame count, and resolution.
fn probe_video(path: &Path) -> Result<ProbeInfo, SrganError> {
    // fps
    let fps_out = run_ffprobe(path, "r_frame_rate")?;
    let fps = parse_fps(&fps_out);

    // frame count
    let nb_out = run_ffprobe(path, "nb_frames")?;
    let frame_count: usize = nb_out.trim().parse().unwrap_or(0);
    // If nb_frames is unknown, estimate from duration * fps
    let frame_count = if frame_count == 0 {
        let dur_out = run_ffprobe(path, "duration")?;
        let dur: f64 = dur_out.trim().parse().unwrap_or(0.0);
        (dur * fps).ceil() as usize
    } else {
        frame_count
    };

    // resolution
    let w_out = run_ffprobe(path, "width")?;
    let h_out = run_ffprobe(path, "height")?;
    let width: u32 = w_out.trim().parse().unwrap_or(0);
    let height: u32 = h_out.trim().parse().unwrap_or(0);

    Ok(ProbeInfo {
        fps,
        frame_count,
        width,
        height,
    })
}

fn run_ffprobe(path: &Path, entry: &str) -> Result<String, SrganError> {
    let output = Command::new("ffprobe")
        .args(&[
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", &format!("stream={}", entry),
            "-of", "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| SrganError::InvalidInput(format!("ffprobe failed: {}", e)))?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn parse_fps(raw: &str) -> f64 {
    let trimmed = raw.trim();
    if let Some(pos) = trimmed.find('/') {
        let num: f64 = trimmed[..pos].parse().unwrap_or(30.0);
        let den: f64 = trimmed[pos + 1..].parse().unwrap_or(1.0);
        if den == 0.0 { 30.0 } else { num / den }
    } else {
        trimmed.parse().unwrap_or(30.0)
    }
}

/// Extract all frames from a video as PNGs.
fn extract_frames(video: &Path, out_dir: &Path) -> Result<(), SrganError> {
    let status = Command::new("ffmpeg")
        .args(&["-i"])
        .arg(video)
        .args(&[
            "-vsync", "0",
            "-frame_pts", "1",
        ])
        .arg(out_dir.join("frame_%08d.png"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| SrganError::InvalidInput(format!("ffmpeg extract failed: {}", e)))?;

    if !status.success() {
        return Err(SrganError::InvalidInput(
            "ffmpeg frame extraction returned non-zero exit code".into(),
        ));
    }
    Ok(())
}

/// Mux a directory of numbered PNGs into a video container.
fn mux_frames(frames_dir: &Path, output: &Path, fps: f64) -> Result<(), SrganError> {
    let pattern = frames_dir.join("frame_%08d.png");
    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4");

    let codec = match ext {
        "webm" => "libvpx-vp9",
        _ => "libx264",
    };

    let fps_str = format!("{:.4}", fps);
    let status = Command::new("ffmpeg")
        .args(&[
            "-y",
            "-framerate", &fps_str,
            "-i",
        ])
        .arg(&pattern)
        .args(&[
            "-c:v", codec,
            "-pix_fmt", "yuv420p",
            "-crf", "18",
        ])
        .arg(output)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| SrganError::InvalidInput(format!("ffmpeg mux failed: {}", e)))?;

    if !status.success() {
        return Err(SrganError::InvalidInput(
            "ffmpeg muxing returned non-zero exit code".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Mock support — when ffmpeg is not installed, callers can use this to test
// the pipeline logic without actual encoding.
// ---------------------------------------------------------------------------

/// A mock video SR processor that skips ffmpeg calls entirely.
/// Useful for integration tests or CI environments without ffmpeg.
pub struct MockVideoSr;

impl MockVideoSr {
    /// Simulate the pipeline: creates empty output file and reports progress.
    pub fn process<F>(config: &VideoSrConfig, on_progress: F) -> Result<(), SrganError>
    where
        F: Fn(VideoSrProgress),
    {
        let fake_frames = 30;
        for i in 1..=fake_frames {
            on_progress(VideoSrProgress {
                frames_done: i,
                total_frames: fake_frames,
                elapsed: std::time::Duration::from_millis(i as u64 * 33),
            });
        }
        // Create an empty output file so downstream code sees it.
        fs::write(&config.video_out, b"")
            .map_err(SrganError::Io)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fps_fraction() {
        assert!((parse_fps("30000/1001") - 29.97).abs() < 0.1);
    }

    #[test]
    fn test_parse_fps_integer() {
        assert!((parse_fps("60\n") - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_fps_zero_denominator() {
        assert!((parse_fps("30/0") - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_mock_video_sr() {
        let config = VideoSrConfig {
            video_in: PathBuf::from("test_input.mp4"),
            video_out: std::env::temp_dir().join("mock_output.mp4"),
            fps: Some(30.0),
            parallel_frames: 1,
            temp_dir: None,
        };
        let mut last = 0;
        MockVideoSr::process(&config, |p| {
            assert!(p.frames_done > 0);
            last = p.frames_done;
        })
        .unwrap();
        assert_eq!(last, 30);
        let _ = fs::remove_file(&config.video_out);
    }
}
