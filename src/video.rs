use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::fs;
use image::{DynamicImage, ImageFormat};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
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
}

impl VideoProcessor {
    /// Create a new video processor
    pub fn new(config: VideoConfig) -> Result<Self, SrganError> {
        // Check if ffmpeg is available
        if !Self::check_ffmpeg()? {
            return Err(SrganError::InvalidInput(
                "FFmpeg is required for video processing. Please install ffmpeg.".to_string()
            ));
        }
        
        Ok(Self {
            config,
            network: None,
            frame_count: None,
        })
    }
    
    /// Load the upscaling network
    pub fn load_network(&mut self, network: UpscalingNetwork) {
        self.network = Some(network);
    }
    
    /// Process the video
    pub fn process(&mut self) -> Result<(), SrganError> {
        let network = self.network.as_ref()
            .ok_or_else(|| SrganError::InvalidInput("No network loaded".to_string()))?;
        
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
    
    /// Check if ffmpeg is installed
    fn check_ffmpeg() -> Result<bool, SrganError> {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .map_err(|_| SrganError::InvalidInput(
                "Failed to run ffmpeg. Please ensure ffmpeg is installed.".to_string()
            ))
    }
    
    /// Create temporary directory for processing
    fn create_temp_dir(&self) -> Result<PathBuf, SrganError> {
        let temp_dir = self.config.temp_dir.clone()
            .unwrap_or_else(|| {
                let mut dir = std::env::temp_dir();
                dir.push(format!("srgan_video_{}", 
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()));
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
        let output = Command::new("ffprobe")
            .args(&[
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=r_frame_rate,nb_read_packets,width,height",
                "-of", "csv=p=0",
                self.config.input_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| SrganError::InvalidInput(
                format!("Failed to get video info: {}", e)
            ))?;
        
        let info_str = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = info_str.trim().split(',').collect();
        
        if parts.len() < 4 {
            return Err(SrganError::InvalidInput("Failed to parse video info".to_string()));
        }
        
        // Parse frame rate (e.g., "30/1" or "30")
        let fps = if parts[0].contains('/') {
            let fps_parts: Vec<&str> = parts[0].split('/').collect();
            fps_parts[0].parse::<f32>().unwrap_or(30.0) / 
                fps_parts.get(1).and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0)
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
        
        let mut cmd = Command::new("ffmpeg");
        cmd.args(&["-i", self.config.input_path.to_str().unwrap()]);
        
        // Add time range if specified
        if let Some(ref start) = self.config.start_time {
            cmd.args(&["-ss", start]);
        }
        if let Some(ref duration) = self.config.duration {
            cmd.args(&["-t", duration]);
        }
        
        cmd.args(&[
            "-vf", &format!("fps={}", info.fps),
            frames_dir.join("frame_%06d.png").to_str().unwrap(),
        ]);
        
        let status = cmd.status()
            .map_err(|e| SrganError::InvalidInput(
                format!("Failed to extract frames: {}", e)
            ))?;
        
        if !status.success() {
            return Err(SrganError::InvalidInput("Frame extraction failed".to_string()));
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
        
        if self.config.parallel_frames > 1 {
            // Parallel processing
            let multi_progress = MultiProgress::new();
            let pb = multi_progress.add(ProgressBar::new(total_frames as u64));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} frames ({eta})")
                    .progress_chars("=>-")
            );
            
            frame_files.par_iter()
                .try_for_each(|frame_path| {
                    let result = self.process_single_frame(frame_path, output_dir, network);
                    pb.inc(1);
                    result
                })?;
            
            pb.finish_with_message("All frames processed");
        } else {
            // Sequential processing
            let pb = ProgressBar::new(total_frames as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} frames ({eta})")
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
        let output_path = output_dir.join(input_path.file_name().unwrap());
        upscaled.save_with_format(&output_path, ImageFormat::Png)
            .map_err(|e| SrganError::Image(e))?;
        
        Ok(())
    }
    
    /// Reassemble frames into video
    fn reassemble_video(&self, frames_dir: &Path, info: &VideoInfo) -> Result<(), SrganError> {
        let mut cmd = Command::new("ffmpeg");
        
        // Input frames
        cmd.args(&[
            "-framerate", &info.fps.to_string(),
            "-i", frames_dir.join("frame_%06d.png").to_str().unwrap(),
        ]);
        
        // Add original audio if requested
        if self.config.preserve_audio {
            cmd.args(&[
                "-i", self.config.input_path.to_str().unwrap(),
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:a", "copy",
            ]);
        }
        
        // Video codec settings
        cmd.args(&[
            "-c:v", self.config.codec.to_ffmpeg_codec(),
            "-crf", &self.config.quality.to_crf().to_string(),
            "-pix_fmt", "yuv420p",
        ]);
        
        // Output file
        cmd.arg(&self.config.output_path);
        
        let status = cmd.status()
            .map_err(|e| SrganError::InvalidInput(
                format!("Failed to reassemble video: {}", e)
            ))?;
        
        if !status.success() {
            return Err(SrganError::InvalidInput("Video reassembly failed".to_string()));
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
    let temp_file = std::env::temp_dir().join("preview_frame.png");
    
    let mut cmd = Command::new("ffmpeg");
    cmd.args(&["-i", video_path.to_str().unwrap()]);
    
    // Extract frame at specified time or first frame
    if let Some(t) = time {
        cmd.args(&["-ss", t]);
    }
    
    cmd.args(&[
        "-vframes", "1",
        "-y",  // Overwrite
        temp_file.to_str().unwrap(),
    ]);
    
    let status = cmd.status()
        .map_err(|e| SrganError::InvalidInput(
            format!("Failed to extract preview frame: {}", e)
        ))?;
    
    if !status.success() {
        return Err(SrganError::InvalidInput("Preview extraction failed".to_string()));
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