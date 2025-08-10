use srgan_rust::profiling::{MemoryProfiler, MemorySnapshot, MemorySample};
use std::time::SystemTime;

#[test]
fn test_memory_profiler_creation() {
    let profiler = MemoryProfiler::new(100);
    assert_eq!(profiler.get_sample_count(), 0);
}

#[test]
fn test_memory_sample_creation() {
    let sample = MemorySample {
        timestamp_ms: 1000,
        allocated_bytes: 1024 * 1024,
        resident_bytes: 2048 * 1024,
        scope: Some("test_scope".to_string()),
    };
    
    assert_eq!(sample.timestamp_ms, 1000);
    assert_eq!(sample.allocated_bytes, 1024 * 1024);
    assert_eq!(sample.resident_bytes, 2048 * 1024);
    assert_eq!(sample.scope, Some("test_scope".to_string()));
}

#[test]
fn test_memory_snapshot_creation() {
    let snapshot = MemorySnapshot {
        timestamp: SystemTime::now(),
        allocated_mb: 256.0,
        resident_mb: 512.0,
        peak_allocated_mb: 1024.0,
        peak_resident_mb: 2048.0,
        samples: vec![],
    };
    
    assert_eq!(snapshot.allocated_mb, 256.0);
    assert_eq!(snapshot.resident_mb, 512.0);
    assert_eq!(snapshot.peak_allocated_mb, 1024.0);
    assert_eq!(snapshot.peak_resident_mb, 2048.0);
    assert_eq!(snapshot.samples.len(), 0);
}

#[test]
fn test_memory_profiler_scope() {
    let mut profiler = MemoryProfiler::new(100);
    profiler.enter_scope("test");
    assert!(profiler.get_current_scope().is_some());
    profiler.exit_scope();
    assert!(profiler.get_current_scope().is_none());
}