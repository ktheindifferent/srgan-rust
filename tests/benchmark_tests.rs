use srgan_rust::commands::benchmark::{BenchmarkConfig, BenchmarkType, BenchmarkResult};
use std::path::PathBuf;

#[test]
fn test_benchmark_config_creation() {
    let config = BenchmarkConfig {
        benchmark_type: BenchmarkType::Upscale,
        iterations: 10,
        warmup_iterations: 2,
        model_path: Some(PathBuf::from("model.rsr")),
        input_path: PathBuf::from("input.jpg"),
        output_path: Some(PathBuf::from("output.csv")),
        batch_size: 4,
        patch_size: 96,
        track_memory: true,
    };
    
    assert_eq!(config.iterations, 10);
    assert_eq!(config.warmup_iterations, 2);
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.patch_size, 96);
    assert!(config.track_memory);
}

#[test]
fn test_benchmark_type_variants() {
    let _ = BenchmarkType::Upscale;
    let _ = BenchmarkType::Train;
    let _ = BenchmarkType::Batch;
    let _ = BenchmarkType::All;
}

#[test]
fn test_benchmark_result_creation() {
    let result = BenchmarkResult {
        benchmark_type: "upscale".to_string(),
        iterations: 5,
        mean_time_ms: 100.5,
        std_dev_ms: 10.2,
        min_time_ms: 85.0,
        max_time_ms: 120.0,
        throughput: Some(10.0),
        memory_usage_mb: Some(512.0),
    };
    
    assert_eq!(result.benchmark_type, "upscale");
    assert_eq!(result.iterations, 5);
    assert_eq!(result.mean_time_ms, 100.5);
    assert_eq!(result.std_dev_ms, 10.2);
    assert_eq!(result.min_time_ms, 85.0);
    assert_eq!(result.max_time_ms, 120.0);
    assert_eq!(result.throughput, Some(10.0));
    assert_eq!(result.memory_usage_mb, Some(512.0));
}