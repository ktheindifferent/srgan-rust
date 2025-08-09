use srgan_rust::video::{VideoCodec, VideoQuality, VideoConfig};
use std::path::PathBuf;

#[test]
fn test_video_codec_conversion() {
    // Test codec to ffmpeg string conversion
    assert_eq!(VideoCodec::H264.to_ffmpeg_codec(), "libx264");
    assert_eq!(VideoCodec::H265.to_ffmpeg_codec(), "libx265");
    assert_eq!(VideoCodec::VP9.to_ffmpeg_codec(), "libvpx-vp9");
    assert_eq!(VideoCodec::AV1.to_ffmpeg_codec(), "libaom-av1");
    assert_eq!(VideoCodec::ProRes.to_ffmpeg_codec(), "prores_ks");
}

#[test]
fn test_video_quality_crf() {
    // Test quality to CRF value conversion
    assert_eq!(VideoQuality::Low.to_crf(), 28);
    assert_eq!(VideoQuality::Medium.to_crf(), 23);
    assert_eq!(VideoQuality::High.to_crf(), 18);
    assert_eq!(VideoQuality::Lossless.to_crf(), 0);
    assert_eq!(VideoQuality::Custom(15).to_crf(), 15);
    assert_eq!(VideoQuality::Custom(51).to_crf(), 51);
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