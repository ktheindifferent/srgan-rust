use crate::logging::{LogConfig, LogFormat, LogOutput, init_logging, TelemetryCollector};
use crate::error::Result;
use std::time::Duration;
use tracing::{info, warn, error};
use metrics::{describe_counter, describe_histogram, describe_gauge};

/// Initialize the application with logging, metrics, and telemetry
pub async fn initialize_app(
    log_level: Option<tracing::Level>,
    enable_telemetry: bool,
    log_to_file: bool,
) -> Result<()> {
    // Configure logging
    let log_config = LogConfig {
        level: log_level.unwrap_or(tracing::Level::INFO),
        format: if std::env::var("LOG_FORMAT").unwrap_or_default() == "json" {
            LogFormat::Json
        } else {
            LogFormat::Pretty
        },
        output: LogOutput::Stdout,
        enable_telemetry,
        enable_file_logging: log_to_file,
        log_directory: Some("logs".to_string()),
        json_output: std::env::var("JSON_LOGS").unwrap_or_default() == "true",
    };
    
    // Initialize logging system
    init_logging(log_config)
        .map_err(|e| crate::error::SrganError::InvalidInput(format!("Failed to initialize logging: {}", e)))?;
    
    info!("SRGAN application initializing...");
    info!("Logging system initialized");
    
    // Initialize metrics if telemetry is enabled
    if enable_telemetry {
        initialize_metrics()?;
        info!("Metrics and telemetry initialized");
        
        // Start telemetry collector
        let collector = TelemetryCollector::new(Duration::from_secs(30));
        collector.start().await;
        info!("Telemetry collector started");
    }
    
    // Set up panic handler with better error reporting
    setup_panic_handler();
    
    // Set up signal handlers for graceful shutdown
    setup_signal_handlers()?;
    
    info!("Application initialization complete");
    Ok(())
}

/// Initialize metrics definitions
fn initialize_metrics() -> Result<()> {
    // Error tracking metrics
    describe_counter!("srgan_errors_total", "Total number of errors encountered");
    describe_counter!("srgan_errors_by_category", "Errors categorized by type");
    describe_counter!("srgan_retries_total", "Total number of retry attempts");
    describe_counter!("srgan_circuit_breaker_trips", "Number of circuit breaker trips");
    
    // Performance metrics
    describe_histogram!("srgan_processing_duration_seconds", "Image processing duration");
    describe_histogram!("srgan_batch_duration_seconds", "Batch processing duration");
    describe_histogram!("srgan_request_duration_seconds", "HTTP request duration");
    describe_histogram!("srgan_queue_wait_seconds", "Time spent waiting in queue");
    
    // Resource metrics
    describe_gauge!("srgan_active_jobs", "Number of currently active jobs");
    describe_gauge!("srgan_queue_depth", "Current queue depth");
    describe_gauge!("srgan_memory_usage_bytes", "Current memory usage in bytes");
    describe_gauge!("srgan_cache_size", "Number of items in cache");
    describe_gauge!("srgan_cache_hit_rate", "Cache hit rate");
    
    // Business metrics
    describe_counter!("srgan_images_processed", "Total images successfully processed");
    describe_counter!("srgan_images_failed", "Total images that failed processing");
    describe_counter!("srgan_cache_hits", "Number of cache hits");
    describe_counter!("srgan_cache_misses", "Number of cache misses");
    describe_counter!("srgan_rate_limit_exceeded", "Number of rate limit violations");
    
    // Health metrics
    describe_gauge!("srgan_health_score", "Overall health score (0-100)");
    describe_gauge!("srgan_health_error_rate", "Current error rate");
    describe_gauge!("srgan_cpu_load_1m", "1-minute CPU load average");
    describe_gauge!("srgan_cpu_load_5m", "5-minute CPU load average");
    describe_gauge!("srgan_cpu_load_15m", "15-minute CPU load average");
    
    Ok(())
}

/// Set up custom panic handler with logging
fn setup_panic_handler() {
    let default_panic = std::panic::take_hook();
    
    std::panic::set_hook(Box::new(move |panic_info| {
        let location = panic_info.location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown location".to_string());
        
        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic payload".to_string()
        };
        
        error!(
            location = %location,
            message = %message,
            "Application panicked"
        );
        
        // Log backtrace if available
        if let Ok(backtrace) = std::env::var("RUST_BACKTRACE") {
            if backtrace == "1" || backtrace == "full" {
                let bt = backtrace::Backtrace::new();
                error!("Backtrace:\n{:?}", bt);
            }
        }
        
        // Increment panic counter
        metrics::increment_counter!("srgan_panics_total", 1);
        
        // Call the default panic handler
        default_panic(panic_info);
    }));
}

/// Set up signal handlers for graceful shutdown
fn setup_signal_handlers() -> Result<()> {
    ctrlc::set_handler(move || {
        info!("Received interrupt signal, initiating graceful shutdown...");
        
        // Log final metrics
        info!("Flushing metrics and logs...");
        
        // Give time for final logs to be written
        std::thread::sleep(Duration::from_millis(100));
        
        std::process::exit(0);
    }).map_err(|e| crate::error::SrganError::InvalidInput(format!("Failed to set signal handler: {}", e)))?;
    
    Ok(())
}

/// Shutdown handler for cleanup
pub async fn shutdown() {
    info!("Starting application shutdown...");
    
    // Flush any pending metrics
    if let Err(e) = flush_metrics().await {
        warn!("Failed to flush metrics during shutdown: {}", e);
    }
    
    // Wait for background tasks to complete
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    info!("Application shutdown complete");
}

/// Flush pending metrics
async fn flush_metrics() -> Result<()> {
    // In a real implementation, this would flush metrics to the backend
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_initialize_app() {
        // Test basic initialization
        let result = initialize_app(
            Some(tracing::Level::DEBUG),
            false,  // Disable telemetry for test
            false,  // Disable file logging for test
        ).await;
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_metrics_initialization() {
        let result = initialize_metrics();
        assert!(result.is_ok());
    }
}