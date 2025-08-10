use srgan_rust::video::{VideoCodec, VideoQuality, VideoConfig};
use std::path::PathBuf;

#[test]
fn test_video_codec_creation() {
    // Test that we can create different codec types
    let _ = VideoCodec::H264;
    let _ = VideoCodec::H265;
    let _ = VideoCodec::VP9;
    let _ = VideoCodec::AV1;
    let _ = VideoCodec::ProRes;
}

#[test]
fn test_video_quality_creation() {
    // Test that we can create different quality levels
    let _ = VideoQuality::Low;
    let _ = VideoQuality::Medium;
    let _ = VideoQuality::High;
    let _ = VideoQuality::Lossless;
    let _ = VideoQuality::Custom(15);
    let _ = VideoQuality::Custom(51);
}

#[test]
fn test_video_config_creation() {
    let config = VideoConfig {
        input_path: PathBuf::from("input.mp4"),
        output_path: PathBuf::from("output.mp4"),
        model_path: None,
        fps: Some(30.0),
        quality: VideoQuality::High,
        codec: VideoCodec::H264,
        preserve_audio: true,
        parallel_frames: 4,
        temp_dir: None,
        start_time: None,
        duration: None,
    };
    
    assert_eq!(config.input_path, PathBuf::from("input.mp4"));
    assert_eq!(config.output_path, PathBuf::from("output.mp4"));
    assert_eq!(config.fps, Some(30.0));
    assert!(config.preserve_audio);
    assert_eq!(config.parallel_frames, 4);
}

#[test]
fn test_video_config_with_time_range() {
    let config = VideoConfig {
        input_path: PathBuf::from("input.mp4"),
        output_path: PathBuf::from("output.mp4"),
        model_path: None,
        fps: None,
        quality: VideoQuality::Medium,
        codec: VideoCodec::H265,
        preserve_audio: false,
        parallel_frames: 1,
        temp_dir: Some(PathBuf::from("/tmp/srgan")),
        start_time: Some("00:01:00".to_string()),
        duration: Some("00:00:30".to_string()),
    };
    
    assert_eq!(config.start_time, Some("00:01:00".to_string()));
    assert_eq!(config.duration, Some("00:00:30".to_string()));
    assert!(!config.preserve_audio);
    assert_eq!(config.parallel_frames, 1);
}