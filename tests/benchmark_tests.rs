use srgan_rust::commands::benchmark::BenchmarkResult;
use std::time::Duration;

#[test]
fn test_benchmark_result_creation() {
    let result = BenchmarkResult {
        model_name: "test_model".to_string(),
        input_size: (256, 256),
        output_size: (1024, 1024),
        factor: 4,
        iterations: 5,
        warmup_iterations: 1,
        total_time: Duration::from_millis(500),
        avg_time: Duration::from_millis(100),
        min_time: Duration::from_millis(85),
        max_time: Duration::from_millis(120),
        throughput_mpx_per_sec: 10.0,
        images_per_sec: 10.0,
        memory_usage_mb: 512.0,
        mb_per_sec: 50.0,
    };

    assert_eq!(result.model_name, "test_model");
    assert_eq!(result.iterations, 5);
    assert_eq!(result.factor, 4);
    assert_eq!(result.input_size, (256, 256));
    assert_eq!(result.output_size, (1024, 1024));
    assert_eq!(result.min_time, Duration::from_millis(85));
    assert_eq!(result.max_time, Duration::from_millis(120));
    assert_eq!(result.memory_usage_mb, 512.0);
}

#[test]
fn test_benchmark_result_csv_row() {
    let result = BenchmarkResult {
        model_name: "csv_test".to_string(),
        input_size: (512, 512),
        output_size: (2048, 2048),
        factor: 4,
        iterations: 10,
        warmup_iterations: 2,
        total_time: Duration::from_secs(1),
        avg_time: Duration::from_millis(100),
        min_time: Duration::from_millis(90),
        max_time: Duration::from_millis(110),
        throughput_mpx_per_sec: 5.0,
        images_per_sec: 10.0,
        memory_usage_mb: 256.0,
        mb_per_sec: 25.0,
    };

    let csv = result.to_csv_row();
    assert!(csv.contains("csv_test"));
    assert!(csv.contains("512x512"));
}
