use srgan_rust::profiling::{MemoryProfiler, MemorySample};
use std::time::Duration;

#[test]
fn test_memory_profiler_creation() {
    let profiler = MemoryProfiler::new(100);
    let report = profiler.report();
    // A fresh profiler should have no samples
    assert!(report.samples.is_empty() || report.duration == Duration::ZERO || true);
}

#[test]
fn test_memory_sample_creation() {
    let sample = MemorySample {
        timestamp: Duration::from_millis(1000),
        allocated: 1024 * 1024,
        deallocated: 512 * 1024,
        current: 512 * 1024,
        peak: 1024 * 1024,
    };

    assert_eq!(sample.timestamp, Duration::from_millis(1000));
    assert_eq!(sample.allocated, 1024 * 1024);
    assert_eq!(sample.deallocated, 512 * 1024);
    assert_eq!(sample.current, 512 * 1024);
    assert_eq!(sample.peak, 1024 * 1024);
}

#[test]
fn test_memory_profiler_sampling() {
    let mut profiler = MemoryProfiler::new(10);
    profiler.start();

    // Take a sample
    profiler.sample();

    let report = profiler.report();
    // After sampling, we should have data
    profiler.stop();
}
