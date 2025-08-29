use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use image::DynamicImage;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, debug};
use crate::error::SrganError;
use crate::UpscalingNetwork;

/// Video processing configuration
#[derive(Debug, Clone)]
pub struct VideoConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub model_path: Option<PathBuf>,
    pub fps: Option<f32>,
    pub quality: VideoQuality,
    pub codec: VideoCodec,
    pub preserve_audio: bool,
    pub parallel_frames: usize,
    pub temp_dir: Option<PathBuf>,
    pub start_time: Option<String>,
    pub duration: Option<String>,
}

/// Video quality presets
#[derive(Debug, Clone, Copy)]
pub enum VideoQuality {
    Low,
    Medium,
    High,
    Lossless,
    Custom(u8),  // CRF value
}

/// Video codec options
#[derive(Debug, Clone, Copy)]
pub enum VideoCodec {
    H264,
    H265,
    VP9,
    AV1,
    ProRes,
}

impl VideoCodec {
    fn to_ffmpeg_codec(&self) -> &'static str {
        match self {
            VideoCodec::H264 => "libx264",
            VideoCodec::H265 => "libx265",
            VideoCodec::VP9 => "libvpx-vp9",
            VideoCodec::AV1 => "libaom-av1",
            VideoCodec::ProRes => "prores_ks",
        }
    }
}

impl VideoQuality {
    fn to_crf(&self) -> u8 {
        match self {
            VideoQuality::Low => 28,
            VideoQuality::Medium => 23,
            VideoQuality::High => 18,
            VideoQuality::Lossless => 0,
            VideoQuality::Custom(crf) => *crf,
        }
    }
}

/// Video processor for upscaling videos frame by frame
pub struct VideoProcessor {
    config: VideoConfig,
    network: Option<UpscalingNetwork>,
    frame_count: Option<usize>,
    allowed_dirs: Vec<PathBuf>,
    command_timeout: Duration,
}

impl VideoProcessor {
    /// Create a new video processor
    pub fn new(config: VideoConfig) -> Result<Self, SrganError> {
        // Validate input and output paths
        Self::validate_path(&config.input_path)?;
        Self::validate_path(&config.output_path)?;
        
        if let Some(ref model_path) = config.model_path {
            Self::validate_path(model_path)?;
        }
        
        // Check if ffmpeg is available
        if !Self::check_ffmpeg()? {
            return Err(SrganError::InvalidInput(
                "FFmpeg is required for video processing. Please install ffmpeg.".into()
            ));
        }
        
        // Setup allowed directories (current working dir and temp dir)
        let mut allowed_dirs = vec![
            std::env::current_dir().map_err(|e| SrganError::Io(e))?,
            std::env::temp_dir(),
        ];
        
        // Add user home directory if available
        if let Ok(home) = std::env::var("HOME") {
            allowed_dirs.push(PathBuf::from(home));
        }
        
        Ok(Self {
            config,
            network: None,
            frame_count: None,
            allowed_dirs,
            command_timeout: Duration::from_secs(300), // 5 minute timeout
        })
    }
    
    /// Load the upscaling network
    pub fn load_network(&mut self, network: UpscalingNetwork) {
        self.network = Some(network);
    }
    
    /// Process the video
    pub fn process(&mut self) -> Result<(), SrganError> {
        let network = self.network.as_ref()
            .ok_or_else(|| SrganError::InvalidInput("No network loaded".into()))?;
        
        info!("Processing video: {:?}", self.config.input_path);
        
        // Create temporary directory for frames
        let temp_dir = self.create_temp_dir()?;
        
        // Extract video information
        let video_info = self.get_video_info()?;
        info!("Video info: {} frames @ {} fps", video_info.frame_count, video_info.fps);
        self.frame_count = Some(video_info.frame_count);
        
        // Extract frames
        info!("Extracting frames...");
        self.extract_frames(&temp_dir, &video_info)?;
        
        // Process frames
        info!("Upscaling frames...");
        let processed_dir = temp_dir.join("processed");
        fs::create_dir_all(&processed_dir)
            .map_err(|e| SrganError::Io(e))?;
        
        self.process_frames(&temp_dir.join("frames"), &processed_dir, network)?;
        
        // Reassemble video
        info!("Reassembling video...");
        self.reassemble_video(&processed_dir, &video_info)?;
        
        // Clean up temporary files
        if self.config.temp_dir.is_none() {
            fs::remove_dir_all(&temp_dir)
                .map_err(|e| SrganError::Io(e))?;
        }
        
        info!("✓ Video processing complete: {:?}", self.config.output_path);
        Ok(())
    }
    
    /// Validate a path for security
    fn validate_path(path: &Path) -> Result<(), SrganError> {
        // Convert to string and check for shell metacharacters
        let path_str = path.to_str()
            .ok_or_else(|| SrganError::InvalidInput("Invalid path encoding".to_string()))?;
        
        // Check for dangerous characters that could lead to command injection
        const FORBIDDEN_CHARS: &[char] = &[
            ';', '|', '&', '`', '$', '(', ')', '{', '}', '<', '>', '\n', '\r', '\0',
            '"', '\'', '\\', '*', '?', '[', ']', '!', '~', '#'
        ];
        
        if path_str.chars().any(|c| FORBIDDEN_CHARS.contains(&c)) {
            return Err(SrganError::InvalidInput(
                format!("Path contains forbidden characters: {}", path_str)
            ));
        }
        
        // Check for directory traversal attempts (both Unix and Windows style)
        if path_str.contains("..") || path_str.contains("..\\") {
            return Err(SrganError::InvalidInput(
                "Path contains directory traversal attempt".to_string()
            ));
        }
        
        // Ensure path doesn't start with a dash (could be interpreted as command flag)
        if path_str.starts_with('-') {
            return Err(SrganError::InvalidInput(
                "Path cannot start with a dash".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Canonicalize and validate path is within allowed directories
    fn validate_and_canonicalize_path(&self, path: &Path) -> Result<PathBuf, SrganError> {
        // First validate the path syntax
        Self::validate_path(path)?;
        
        // Canonicalize to resolve symlinks and get absolute path
        let canonical = path.canonicalize()
            .or_else(|_| {
                // If file doesn't exist yet, canonicalize the parent and append filename
                if let Some(parent) = path.parent() {
                    if let Some(file_name) = path.file_name() {
                        parent.canonicalize()
                            .map(|p| p.join(file_name))
                    } else {
                        Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Invalid path"))
                    }
                } else {
                    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Invalid path"))
                }
            })
            .map_err(|e| SrganError::InvalidInput(format!("Path validation failed: {}", e)))?;
        
        // Check if path is within allowed directories
        let is_allowed = self.allowed_dirs.iter().any(|allowed| {
            canonical.starts_with(allowed)
        });
        
        if !is_allowed {
            return Err(SrganError::InvalidInput(
                format!("Path is outside allowed directories: {:?}", canonical)
            ));
        }
        
        debug!("Validated path: {:?}", canonical);
        Ok(canonical)
    }
    
    /// Check if ffmpeg is installed
    fn check_ffmpeg() -> Result<bool, SrganError> {
        Self::log_command_execution("ffmpeg", &["-version"], None);
        
        Command::new("ffmpeg")
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .map_err(|_| SrganError::InvalidInput(
                "Failed to run ffmpeg. Please ensure ffmpeg is installed.".into()
            ))
    }
    
    /// Log command execution for security auditing
    fn log_command_execution(command: &str, args: &[&str], input_file: Option<&Path>) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let args_str = args.join(" ");
        
        if let Some(input) = input_file {
            info!(
                "[AUDIT] Command execution at {}: {} {} | Input: {:?}",
                timestamp, command, args_str, input
            );
        } else {
            info!(
                "[AUDIT] Command execution at {}: {} {}",
                timestamp, command, args_str
            );
        }
        
        // Log to a dedicated security audit file if configured
        // #[cfg(feature = "audit-log")]
        // Audit logging disabled - feature not configured
        if false {
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/var/log/srgan_audit.log")
            {
                let _ = writeln!(
                    file,
                    "{} | {} {} | Input: {:?}",
                    timestamp, command, args_str, input_file
                );
            }
        }
    }
    
    /// Create temporary directory for processing
    fn create_temp_dir(&self) -> Result<PathBuf, SrganError> {
        let temp_dir = self.config.temp_dir.clone()
            .unwrap_or_else(|| {
                let mut dir = std::env::temp_dir();
                dir.push(format!("srgan_video_{}", 
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or_else(|_| {
                            use std::sync::atomic::{AtomicU64, Ordering};
                            static COUNTER: AtomicU64 = AtomicU64::new(0);
                            COUNTER.fetch_add(1, Ordering::SeqCst)
                        })));
                dir
            });
        
        fs::create_dir_all(&temp_dir)
            .map_err(|e| SrganError::Io(e))?;
        
        fs::create_dir_all(temp_dir.join("frames"))
            .map_err(|e| SrganError::Io(e))?;
        
        Ok(temp_dir)
    }
    
    /// Get video information using ffprobe
    fn get_video_info(&self) -> Result<VideoInfo, SrganError> {
        // Validate input path again
        let safe_input_path = self.validate_and_canonicalize_path(&self.config.input_path)?;
        
        debug!("Getting video info for: {:?}", safe_input_path);
        
        // Log the command execution for audit
        Self::log_command_execution(
            "ffprobe",
            &["-v", "error", "-select_streams", "v:0", "-count_packets",
              "-show_entries", "stream=r_frame_rate,nb_read_packets,width,height",
              "-of", "csv=p=0"],
            Some(&safe_input_path)
        );
        
        let output = Command::new("ffprobe")
            .arg("-v")
            .arg("error")
            .arg("-select_streams")
            .arg("v:0")
            .arg("-count_packets")
            .arg("-show_entries")
            .arg("stream=r_frame_rate,nb_read_packets,width,height")
            .arg("-of")
            .arg("csv=p=0")
            .arg(&safe_input_path)  // Pass as separate argument, not formatted into string
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| SrganError::InvalidInput(
                format!("Failed to get video info: {}", e)
            ))?;
        
        let info_str = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = info_str.trim().split(',').collect();
        
        if parts.len() < 4 {
            return Err(SrganError::InvalidInput("Failed to parse video info".into()));
        }
        
        // Parse frame rate (e.g., "30/1" or "30")
        let fps = if parts[0].contains('/') {
            let fps_parts: Vec<&str> = parts[0].split('/').collect();
            let numerator = fps_parts[0].parse::<f32>().unwrap_or(30.0);
            let denominator = fps_parts.get(1)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.0);
            numerator / denominator
        } else {
            parts[0].parse().unwrap_or(30.0)
        };
        
        Ok(VideoInfo {
            fps: self.config.fps.unwrap_or(fps),
            frame_count: parts[1].parse().unwrap_or(0),
            width: parts[2].parse().unwrap_or(0),
            height: parts[3].parse().unwrap_or(0),
        })
    }
    
    /// Extract frames from video
    fn extract_frames(&self, temp_dir: &Path, info: &VideoInfo) -> Result<(), SrganError> {
        let frames_dir = temp_dir.join("frames");
        let safe_input_path = self.validate_and_canonicalize_path(&self.config.input_path)?;
        let safe_output_pattern = self.validate_and_canonicalize_path(&frames_dir.join("frame_%06d.png"))?;
        
        info!("Extracting frames from: {:?}", safe_input_path);
        
        // Build args for logging
        let mut log_args = vec!["-i"];
        if self.config.start_time.is_some() {
            log_args.push("-ss");
        }
        if self.config.duration.is_some() {
            log_args.push("-t");
        }
        log_args.extend(&["-vf", "fps=X"]);
        
        Self::log_command_execution("ffmpeg", &log_args, Some(&safe_input_path));
        
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-i")
           .arg(&safe_input_path);
        
        // Add time range if specified (validate these strings)
        if let Some(ref start) = self.config.start_time {
            Self::validate_time_string(start)?;
            cmd.arg("-ss")
               .arg(start);
        }
        if let Some(ref duration) = self.config.duration {
            Self::validate_time_string(duration)?;
            cmd.arg("-t")
               .arg(duration);
        }
        
        // Use validated FPS value
        let safe_fps = Self::validate_fps(info.fps)?;
        cmd.arg("-vf")
           .arg(format!("fps={}", safe_fps))
           .arg(&safe_output_pattern);
        
        // Add security constraints
        cmd.stdin(Stdio::null())
           .stdout(Stdio::null())
           .stderr(Stdio::null());
        
        let status = cmd.status()
            .map_err(|e| SrganError::InvalidInput(
                format!("Failed to extract frames: {}", e)
            ))?;
        
        if !status.success() {
            return Err(SrganError::InvalidInput("Frame extraction failed".into()));
        }
        
        Ok(())
    }
    
    /// Process extracted frames
    fn process_frames(&self, input_dir: &Path, output_dir: &Path, network: &UpscalingNetwork) -> Result<(), SrganError> {
        let frame_files: Vec<_> = fs::read_dir(input_dir)
            .map_err(|e| SrganError::Io(e))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        
        let total_frames = frame_files.len();
        
        // Note: Parallel processing is temporarily disabled due to network not being Send + Sync
        // This is a known limitation that can be addressed in future refactoring
        {
            // Sequential processing
            let pb = ProgressBar::new(total_frames as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} frames ({eta})")
                    .unwrap_or_else(|_| ProgressStyle::default_bar())
                    .progress_chars("=>-")
            );
            
            for frame_path in frame_files {
                self.process_single_frame(&frame_path, output_dir, network)?;
                pb.inc(1);
            }
            
            pb.finish_with_message("All frames processed");
        }
        
        Ok(())
    }
    
    /// Process a single frame
    fn process_single_frame(&self, input_path: &Path, output_dir: &Path, network: &UpscalingNetwork) -> Result<(), SrganError> {
        let img = image::open(input_path)
            .map_err(|e| SrganError::Image(e))?;
        
        // Upscale the frame
        let upscaled = network.upscale_image(&img)?;
        
        // Save processed frame
        let file_name = input_path.file_name()
            .ok_or_else(|| SrganError::InvalidInput("Invalid input frame filename".to_string()))?;
        let output_path = output_dir.join(file_name);
        upscaled.save(&output_path)
            .map_err(|e| SrganError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        
        Ok(())
    }
    
    /// Reassemble frames into video
    fn reassemble_video(&self, frames_dir: &Path, info: &VideoInfo) -> Result<(), SrganError> {
        let safe_frames_pattern = self.validate_and_canonicalize_path(&frames_dir.join("frame_%06d.png"))?;
        let safe_output_path = self.validate_and_canonicalize_path(&self.config.output_path)?;
        
        info!("Reassembling video to: {:?}", safe_output_path);
        
        // Log the reassembly command
        let mut log_args = vec!["-framerate", "X", "-i", "frames"];
        if self.config.preserve_audio {
            log_args.extend(&["-i", "input", "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]);
        }
        log_args.extend(&["-c:v", "codec", "-crf", "X", "-pix_fmt", "yuv420p"]);
        
        Self::log_command_execution("ffmpeg", &log_args, Some(&safe_output_path));
        
        let mut cmd = Command::new("ffmpeg");
        
        // Input frames with validated FPS
        let safe_fps = Self::validate_fps(info.fps)?;
        cmd.arg("-framerate")
           .arg(safe_fps.to_string())
           .arg("-i")
           .arg(&safe_frames_pattern);
        
        // Add original audio if requested
        if self.config.preserve_audio {
            let safe_input_path = self.validate_and_canonicalize_path(&self.config.input_path)?;
            cmd.arg("-i")
               .arg(&safe_input_path)
               .arg("-map")
               .arg("0:v:0")
               .arg("-map")
               .arg("1:a?")
               .arg("-c:a")
               .arg("copy");
        }
        
        // Video codec settings (already safe as they come from enums)
        cmd.arg("-c:v")
           .arg(self.config.codec.to_ffmpeg_codec())
           .arg("-crf")
           .arg(self.config.quality.to_crf().to_string())
           .arg("-pix_fmt")
           .arg("yuv420p");
        
        // Output file
        cmd.arg(&safe_output_path);
        
        // Add security constraints
        cmd.stdin(Stdio::null())
           .stdout(Stdio::null())
           .stderr(Stdio::null());
        
        let status = cmd.status()
            .map_err(|e| SrganError::InvalidInput(
                format!("Failed to reassemble video: {}", e)
            ))?;
        
        if !status.success() {
            return Err(SrganError::InvalidInput("Video reassembly failed".into()));
        }
        
        Ok(())
    }
    
    /// Get processing statistics
    pub fn get_stats(&self) -> VideoStats {
        VideoStats {
            frames_processed: self.frame_count.unwrap_or(0),
            input_path: self.config.input_path.clone(),
            output_path: self.config.output_path.clone(),
            codec: self.config.codec,
            quality: self.config.quality,
        }
    }
    
    /// Validate time string format (HH:MM:SS or seconds)
    fn validate_time_string(time: &str) -> Result<(), SrganError> {
        // Allow formats: "HH:MM:SS", "MM:SS", "SS" or decimal seconds
        let valid_chars = "0123456789:.";
        if !time.chars().all(|c| valid_chars.contains(c)) {
            return Err(SrganError::InvalidInput(
                format!("Invalid time format: {}", time)
            ));
        }
        
        // Check for reasonable length
        if time.len() > 12 {
            return Err(SrganError::InvalidInput(
                "Time string too long".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Validate FPS value
    fn validate_fps(fps: f32) -> Result<f32, SrganError> {
        if fps <= 0.0 || fps > 240.0 || !fps.is_finite() {
            return Err(SrganError::InvalidInput(
                format!("Invalid FPS value: {}", fps)
            ));
        }
        Ok(fps)
    }
}

/// Video information
#[derive(Debug)]
struct VideoInfo {
    fps: f32,
    frame_count: usize,
    width: u32,
    height: u32,
}

/// Video processing statistics
#[derive(Debug)]
pub struct VideoStats {
    pub frames_processed: usize,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub codec: VideoCodec,
    pub quality: VideoQuality,
}

/// Extract a single frame from video for preview
pub fn extract_preview_frame(video_path: &Path, time: Option<&str>) -> Result<DynamicImage, SrganError> {
    // Validate input path
    VideoProcessor::validate_path(video_path)?;
    
    // Canonicalize the path
    let safe_video_path = video_path.canonicalize()
        .map_err(|e| SrganError::InvalidInput(format!("Invalid video path: {}", e)))?;
    
    let temp_file = std::env::temp_dir().join(format!("preview_frame_{}.png", 
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()));
    
    // Log the preview extraction command
    let mut log_args = vec!["-i"];
    if time.is_some() {
        log_args.extend(&["-ss", "time"]);
    }
    log_args.extend(&["-vframes", "1", "-y"]);
    
    VideoProcessor::log_command_execution("ffmpeg", &log_args, Some(&safe_video_path));
    
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-i")
       .arg(&safe_video_path);
    
    // Extract frame at specified time or first frame
    if let Some(t) = time {
        VideoProcessor::validate_time_string(t)?;
        cmd.arg("-ss")
           .arg(t);
    }
    
    cmd.arg("-vframes")
       .arg("1")
       .arg("-y")  // Overwrite
       .arg(&temp_file);
    
    // Add security constraints
    cmd.stdin(Stdio::null())
       .stdout(Stdio::null())
       .stderr(Stdio::null());
    
    let status = cmd.status()
        .map_err(|e| SrganError::InvalidInput(
            format!("Failed to extract preview frame: {}", e)
        ))?;
    
    if !status.success() {
        return Err(SrganError::InvalidInput("Preview extraction failed".into()));
    }
    
    let img = image::open(&temp_file)
        .map_err(|e| SrganError::Image(e))?;
    
    // Clean up
    let _ = fs::remove_file(&temp_file);
    
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_video_codec_conversion() {
        assert_eq!(VideoCodec::H264.to_ffmpeg_codec(), "libx264");
        assert_eq!(VideoCodec::H265.to_ffmpeg_codec(), "libx265");
        assert_eq!(VideoCodec::VP9.to_ffmpeg_codec(), "libvpx-vp9");
    }
    
    #[test]
    fn test_video_quality_crf() {
        assert_eq!(VideoQuality::Low.to_crf(), 28);
        assert_eq!(VideoQuality::Medium.to_crf(), 23);
        assert_eq!(VideoQuality::High.to_crf(), 18);
        assert_eq!(VideoQuality::Lossless.to_crf(), 0);
        assert_eq!(VideoQuality::Custom(15).to_crf(), 15);
    }
}