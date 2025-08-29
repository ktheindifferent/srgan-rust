use std::path::{Path, PathBuf};
use clap::ArgMatches;
use log::{info, warn};
use crate::error::SrganError;
use crate::UpscalingNetwork;
use crate::video::{VideoProcessor, VideoConfig, VideoCodec, VideoQuality, extract_preview_frame};

/// Upscale a video file frame by frame
pub fn upscale_video(matches: &ArgMatches) -> Result<(), SrganError> {
    let input_path = PathBuf::from(matches.value_of("input")
        .ok_or_else(|| SrganError::InvalidParameter("Input path is required".to_string()))?);
    let output_path = PathBuf::from(matches.value_of("output")
        .ok_or_else(|| SrganError::InvalidParameter("Output path is required".to_string()))?);
    
    // Check input file exists
    if !input_path.exists() {
        return Err(SrganError::FileNotFound(input_path));
    }
    
    // Load the upscaling network
    let network = load_network_from_matches(matches)?;
    
    // Parse video configuration
    let config = VideoConfig {
        input_path: input_path.clone(),
        output_path: output_path.clone(),
        model_path: matches.value_of("custom").map(PathBuf::from),
        fps: matches.value_of("fps").and_then(|f| f.parse().ok()),
        quality: parse_quality(matches.value_of("quality").unwrap_or("medium"))?,
        codec: parse_codec(matches.value_of("codec").unwrap_or("h264"))?,
        preserve_audio: !matches.is_present("no-audio"),
        parallel_frames: matches.value_of("parallel")
            .and_then(|p| p.parse().ok())
            .unwrap_or(4),
        temp_dir: matches.value_of("temp-dir").map(PathBuf::from),
        start_time: matches.value_of("start").map(String::from),
        duration: matches.value_of("duration").map(String::from),
    };
    
    info!("Video upscaling configuration:");
    info!("  Input: {:?}", config.input_path);
    info!("  Output: {:?}", config.output_path);
    info!("  Codec: {:?}", config.codec);
    info!("  Quality: {:?}", config.quality);
    info!("  Preserve audio: {}", config.preserve_audio);
    info!("  Parallel frames: {}", config.parallel_frames);
    
    // Show preview if requested
    if matches.is_present("preview") {
        info!("Generating preview...");
        let preview_time = matches.value_of("preview-time");
        let preview_frame = extract_preview_frame(&input_path, preview_time)?;
        
        // Upscale preview frame
        let upscaled_preview = network.upscale_image(&preview_frame)?;
        
        // Save preview
        let preview_path = output_path.with_extension("preview.png");
        upscaled_preview.save(&preview_path)
            .map_err(|e| SrganError::Io(e.into()))?;
        
        info!("✓ Preview saved to {:?}", preview_path);
        
        if matches.is_present("preview-only") {
            info!("Preview-only mode - skipping full video processing");
            return Ok(());
        }
    }
    
    // Create video processor
    let mut processor = VideoProcessor::new(config)?;
    processor.load_network(network);
    
    // Process the video
    info!("Starting video processing...");
    processor.process()?;
    
    // Display statistics
    let stats = processor.get_stats();
    info!("Video processing complete:");
    info!("  Frames processed: {}", stats.frames_processed);
    info!("  Output: {:?}", stats.output_path);
    
    Ok(())
}

/// Parse video quality setting
fn parse_quality(quality: &str) -> Result<VideoQuality, SrganError> {
    match quality.to_lowercase().as_str() {
        "low" => Ok(VideoQuality::Low),
        "medium" => Ok(VideoQuality::Medium),
        "high" => Ok(VideoQuality::High),
        "lossless" => Ok(VideoQuality::Lossless),
        crf => {
            // Try to parse as CRF value
            crf.parse::<u8>()
                .map(VideoQuality::Custom)
                .map_err(|_| SrganError::InvalidInput(
                    format!("Invalid quality setting: {}", quality)
                ))
        }
    }
}

/// Parse video codec
fn parse_codec(codec: &str) -> Result<VideoCodec, SrganError> {
    match codec.to_lowercase().as_str() {
        "h264" | "x264" => Ok(VideoCodec::H264),
        "h265" | "x265" | "hevc" => Ok(VideoCodec::H265),
        "vp9" => Ok(VideoCodec::VP9),
        "av1" => Ok(VideoCodec::AV1),
        "prores" => Ok(VideoCodec::ProRes),
        _ => Err(SrganError::InvalidInput(
            format!("Unknown codec: {}", codec)
        ))
    }
}

/// Load network from command line arguments
fn load_network_from_matches(matches: &ArgMatches) -> Result<UpscalingNetwork, SrganError> {
    if let Some(custom_path) = matches.value_of("custom") {
        // Load custom model
        info!("Loading custom model from {}", custom_path);
        UpscalingNetwork::load_from_file(Path::new(custom_path))
    } else {
        // Use built-in model
        let model_type = matches.value_of("parameters").unwrap_or("natural");
        info!("Using built-in {} model", model_type);
        
        match model_type {
            "natural" => UpscalingNetwork::load_builtin_natural(),
            "anime" => UpscalingNetwork::load_builtin_anime(),
            _ => Err(SrganError::InvalidInput(
                format!("Unknown model type: {}", model_type)
            ))
        }
    }
}

/// Batch process multiple videos
pub fn batch_video(matches: &ArgMatches) -> Result<(), SrganError> {
    let input_dir = Path::new(matches.value_of("input-dir")
        .ok_or_else(|| SrganError::InvalidParameter("Input directory is required".to_string()))?);
    let output_dir = Path::new(matches.value_of("output-dir")
        .ok_or_else(|| SrganError::InvalidParameter("Output directory is required".to_string()))?);
    
    if !input_dir.is_dir() {
        return Err(SrganError::InvalidInput(
            "Input must be a directory for batch processing".to_string()
        ));
    }
    
    // Create output directory
    std::fs::create_dir_all(output_dir)
        .map_err(|e| SrganError::Io(e))?;
    
    // Find all video files
    let video_extensions = vec!["mp4", "avi", "mkv", "mov", "webm", "flv"];
    let mut video_files = Vec::new();
    
    for entry in std::fs::read_dir(input_dir).map_err(|e| SrganError::Io(e))? {
        let entry = entry.map_err(|e| SrganError::Io(e))?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if video_extensions.contains(&ext.to_lowercase().as_str()) {
                    video_files.push(path);
                }
            }
        }
    }
    
    if video_files.is_empty() {
        return Err(SrganError::InvalidInput(
            "No video files found in input directory".to_string()
        ));
    }
    
    info!("Found {} video files to process", video_files.len());
    
    // Load network once
    let network = load_network_from_matches(matches)?;
    
    // Process each video
    let mut success_count = 0;
    let mut failed_files = Vec::new();
    
    for (idx, video_path) in video_files.iter().enumerate() {
        let filename = video_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        
        info!("[{}/{}] Processing: {}", idx + 1, video_files.len(), filename);
        
        // Generate output path
        let output_filename = format!("upscaled_{}", filename);
        let output_path = output_dir.join(output_filename);
        
        // Skip if already exists
        if output_path.exists() && !matches.is_present("overwrite") {
            info!("  Skipping (already exists): {:?}", output_path);
            continue;
        }
        
        // Create video config
        let config = VideoConfig {
            input_path: video_path.clone(),
            output_path,
            model_path: matches.value_of("custom").map(PathBuf::from),
            fps: matches.value_of("fps").and_then(|f| f.parse().ok()),
            quality: parse_quality(matches.value_of("quality").unwrap_or("medium"))?,
            codec: parse_codec(matches.value_of("codec").unwrap_or("h264"))?,
            preserve_audio: !matches.is_present("no-audio"),
            parallel_frames: matches.value_of("parallel")
                .and_then(|p| p.parse().ok())
                .unwrap_or(4),
            temp_dir: matches.value_of("temp-dir").map(PathBuf::from),
            start_time: None,
            duration: None,
        };
        
        // Process video
        let mut processor = VideoProcessor::new(config)?;
        processor.load_network(network.clone());
        
        match processor.process() {
            Ok(_) => {
                success_count += 1;
                info!("  ✓ Success");
            }
            Err(e) => {
                failed_files.push(filename.to_string());
                warn!("  ✗ Failed: {}", e);
            }
        }
    }
    
    // Summary
    info!("");
    info!("Batch processing complete:");
    info!("  Total: {}", video_files.len());
    info!("  Successful: {}", success_count);
    info!("  Failed: {}", failed_files.len());
    
    if !failed_files.is_empty() {
        warn!("Failed files:");
        for file in &failed_files {
            warn!("  - {}", file);
        }
    }
    
    Ok(())
}