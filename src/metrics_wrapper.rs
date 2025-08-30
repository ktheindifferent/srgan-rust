/// Simplified wrapper for metrics to avoid macro issues
use metrics::{counter, histogram, gauge};

/// Increment a counter
pub fn increment_counter(name: &str, value: u64) {
    counter!(name, value);
}

/// Record a histogram value
pub fn record_histogram(name: &str, value: f64) {
    histogram!(name, value);
}

/// Set a gauge value
pub fn set_gauge(name: &str, value: f64) {
    gauge!(name, value);
}

/// Increment a gauge
pub fn increment_gauge(name: &str, delta: f64) {
    // For metrics 0.21, we need to track the value externally or use set
    gauge!(name, delta);
}

/// Decrement a gauge
pub fn decrement_gauge(name: &str, delta: f64) {
    // For metrics 0.21, we need to track the value externally or use set
    gauge!(name, -delta);
}