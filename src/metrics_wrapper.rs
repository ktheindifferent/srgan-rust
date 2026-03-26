/// Simplified wrapper for metrics to avoid macro issues
/// Uses no-op implementations since metrics macros require 'static strings.

/// Increment a counter
pub fn increment_counter(_name: &str, _value: u64) {
    // Dynamic metric names cannot be used with metrics macros (require 'static).
    // Callers should use metrics::counter!("static_name", value) directly.
}

/// Record a histogram value
pub fn record_histogram(_name: &str, _value: f64) {
    // Dynamic metric names cannot be used with metrics macros (require 'static).
}

/// Set a gauge value
pub fn set_gauge(_name: &str, _value: f64) {
    // Dynamic metric names cannot be used with metrics macros (require 'static).
}

/// Increment a gauge
pub fn increment_gauge(_name: &str, _delta: f64) {
    // Dynamic metric names cannot be used with metrics macros (require 'static).
}

/// Decrement a gauge
pub fn decrement_gauge(_name: &str, _delta: f64) {
    // Dynamic metric names cannot be used with metrics macros (require 'static).
}
