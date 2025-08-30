use std::path::Path;
use tracing::{Level, Subscriber};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
    Layer,
};
use tracing_appender::{non_blocking, rolling};
use std::sync::Arc;
use dashmap::DashMap;
use std::time::{Duration, Instant};
use metrics::{counter, histogram, gauge, describe_counter, describe_histogram, describe_gauge};

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LogConfig {
    pub level: Level,
    pub format: LogFormat,
    pub output: LogOutput,
    pub enable_telemetry: bool,
    pub enable_file_logging: bool,
    pub log_directory: Option<String>,
    pub json_output: bool,
}

#[derive(Debug, Clone)]
pub enum LogFormat {
    Compact,
    Pretty,
    Json,
}

#[derive(Debug, Clone)]
pub enum LogOutput {
    Stdout,
    Stderr,
    File(String),
    Both,
}

impl Default for LogConfig {
    fn default() -> Self {
        LogConfig {
            level: Level::INFO,
            format: LogFormat::Pretty,
            output: LogOutput::Stdout,
            enable_telemetry: true,
            enable_file_logging: false,
            log_directory: Some("logs".to_string()),
            json_output: false,
        }
    }
}

/// Initialize the logging system
pub fn init_logging(config: LogConfig) -> anyhow::Result<()> {
    let env_filter = EnvFilter::from_default_env()
        .add_directive(config.level.into());
    
    let fmt_layer = if config.json_output {
        fmt::layer()
            .json()
            .with_span_events(FmtSpan::FULL)
            .boxed()
    } else {
        match config.format {
            LogFormat::Compact => fmt::layer()
                .compact()
                .with_span_events(FmtSpan::CLOSE)
                .boxed(),
            LogFormat::Pretty => fmt::layer()
                .pretty()
                .with_span_events(FmtSpan::FULL)
                .boxed(),
            LogFormat::Json => fmt::layer()
                .json()
                .with_span_events(FmtSpan::FULL)
                .boxed(),
        }
    };
    
    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer);
    
    // Add file logging if enabled
    if config.enable_file_logging {
        let log_dir = config.log_directory.unwrap_or_else(|| "logs".to_string());
        std::fs::create_dir_all(&log_dir)?;
        
        let file_appender = rolling::daily(&log_dir, "srgan.log");
        let (non_blocking, _guard) = non_blocking(file_appender);
        
        let file_layer = fmt::layer()
            .json()
            .with_writer(non_blocking)
            .with_span_events(FmtSpan::FULL);
        
        subscriber.with(file_layer).init();
    } else {
        subscriber.init();
    }
    
    // Initialize metrics if telemetry is enabled
    if config.enable_telemetry {
        init_metrics()?;
    }
    
    Ok(())
}

/// Initialize metrics collection
fn init_metrics() -> anyhow::Result<()> {
    // Describe metrics
    describe_counter!("srgan_errors_total", "Total number of errors");
    describe_counter!("srgan_images_processed", "Total number of images processed");
    describe_counter!("srgan_retries_total", "Total number of retry attempts");
    describe_histogram!("srgan_processing_duration_seconds", "Image processing duration");
    describe_histogram!("srgan_batch_size", "Batch processing size");
    describe_gauge!("srgan_active_jobs", "Number of active processing jobs");
    describe_gauge!("srgan_memory_usage_bytes", "Current memory usage");
    describe_gauge!("srgan_cache_size", "Current cache size");
    
    // Set up Prometheus exporter
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    builder.install()?;
    
    Ok(())
}

/// Operation logger for tracking individual operations
pub struct OperationLogger {
    operation_id: String,
    start_time: Instant,
    metadata: DashMap<String, String>,
}

impl OperationLogger {
    pub fn new(operation_id: impl Into<String>) -> Self {
        let operation_id = operation_id.into();
        tracing::info!(operation_id = %operation_id, "Operation started");
        
        OperationLogger {
            operation_id,
            start_time: Instant::now(),
            metadata: DashMap::new(),
        }
    }
    
    pub fn add_metadata(&self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
    
    pub fn log_progress(&self, message: impl AsRef<str>) {
        let elapsed = self.start_time.elapsed();
        tracing::info!(
            operation_id = %self.operation_id,
            elapsed_ms = elapsed.as_millis() as u64,
            message = %message.as_ref(),
            "Operation progress"
        );
    }
    
    pub fn log_error(&self, error: &dyn std::error::Error) {
        let elapsed = self.start_time.elapsed();
        tracing::error!(
            operation_id = %self.operation_id,
            elapsed_ms = elapsed.as_millis() as u64,
            error = %error,
            "Operation failed"
        );
        
        metrics::increment_counter!("srgan_errors_total", 1);
    }
    
    pub fn complete(self) {
        let elapsed = self.start_time.elapsed();
        let metadata: std::collections::HashMap<String, String> = self.metadata
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        
        tracing::info!(
            operation_id = %self.operation_id,
            elapsed_ms = elapsed.as_millis() as u64,
            metadata = ?metadata,
            "Operation completed"
        );
        
        metrics::histogram!("srgan_processing_duration_seconds", elapsed.as_secs_f64());
    }
}

/// Structured logging macros with context
#[macro_export]
macro_rules! log_with_context {
    ($level:expr, $context:expr, $($arg:tt)*) => {{
        use tracing::{event, Level};
        
        let ctx = &$context;
        event!(
            $level,
            operation = %ctx.operation,
            file_path = ?ctx.file_path,
            attempt = ctx.attempt,
            max_attempts = ctx.max_attempts,
            metadata = ?ctx.metadata,
            $($arg)*
        );
    }};
}

#[macro_export]
macro_rules! info_with_context {
    ($context:expr, $($arg:tt)*) => {
        $crate::log_with_context!(Level::INFO, $context, $($arg)*)
    };
}

#[macro_export]
macro_rules! warn_with_context {
    ($context:expr, $($arg:tt)*) => {
        $crate::log_with_context!(Level::WARN, $context, $($arg)*)
    };
}

#[macro_export]
macro_rules! error_with_context {
    ($context:expr, $($arg:tt)*) => {
        $crate::log_with_context!(Level::ERROR, $context, $($arg)*)
    };
}

/// Performance tracking
pub struct PerformanceTracker {
    operations: Arc<DashMap<String, OperationStats>>,
}

#[derive(Debug, Clone)]
struct OperationStats {
    count: u64,
    total_duration: Duration,
    min_duration: Duration,
    max_duration: Duration,
    failures: u64,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        PerformanceTracker {
            operations: Arc::new(DashMap::new()),
        }
    }
    
    pub fn track<F, T>(&self, operation: impl Into<String>, f: F) -> Result<T, Box<dyn std::error::Error>>
    where
        F: FnOnce() -> Result<T, Box<dyn std::error::Error>>,
    {
        let operation = operation.into();
        let start = Instant::now();
        
        let result = f();
        let duration = start.elapsed();
        
        self.operations
            .entry(operation.clone())
            .and_modify(|stats| {
                stats.count += 1;
                stats.total_duration += duration;
                stats.min_duration = stats.min_duration.min(duration);
                stats.max_duration = stats.max_duration.max(duration);
                if result.is_err() {
                    stats.failures += 1;
                }
            })
            .or_insert(OperationStats {
                count: 1,
                total_duration: duration,
                min_duration: duration,
                max_duration: duration,
                failures: if result.is_err() { 1 } else { 0 },
            });
        
        // Update metrics
        metrics::histogram!(format!("srgan_{}_duration_seconds", operation).as_str(), duration.as_secs_f64());
        if result.is_err() {
            metrics::increment_counter!(format!("srgan_{}_failures", operation).as_str(), 1);
        }
        
        result
    }
    
    pub fn get_stats(&self) -> Vec<(String, OperationStats)> {
        self.operations
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
    
    pub fn report(&self) {
        tracing::info!("Performance Report:");
        for (operation, stats) in self.get_stats() {
            let avg_duration = if stats.count > 0 {
                stats.total_duration / stats.count as u32
            } else {
                Duration::ZERO
            };
            
            tracing::info!(
                "  {}: count={}, avg={:?}, min={:?}, max={:?}, failures={}",
                operation,
                stats.count,
                avg_duration,
                stats.min_duration,
                stats.max_duration,
                stats.failures
            );
        }
    }
}

/// Telemetry collector for system metrics
pub struct TelemetryCollector {
    interval: Duration,
}

impl TelemetryCollector {
    pub fn new(interval: Duration) -> Self {
        TelemetryCollector { interval }
    }
    
    pub async fn start(self) {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.interval);
            
            loop {
                interval.tick().await;
                self.collect_metrics();
            }
        });
    }
    
    fn collect_metrics(&self) {
        // Collect memory usage
        if let Ok(mem_info) = sys_info::mem_info() {
            let used_bytes = (mem_info.total - mem_info.free) * 1024;
            metrics::gauge!("srgan_memory_usage_bytes", used_bytes as f64);
        }
        
        // Collect CPU usage
        if let Ok(loadavg) = sys_info::loadavg() {
            metrics::gauge!("srgan_cpu_load_1m", loadavg.one);
            metrics::gauge!("srgan_cpu_load_5m", loadavg.five);
            metrics::gauge!("srgan_cpu_load_15m", loadavg.fifteen);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_operation_logger() {
        let logger = OperationLogger::new("test_op");
        logger.add_metadata("key1", "value1");
        logger.log_progress("50% complete");
        logger.complete();
    }
    
    #[test]
    fn test_performance_tracker() {
        let tracker = PerformanceTracker::new();
        
        let _ = tracker.track("test_operation", || {
            std::thread::sleep(Duration::from_millis(10));
            Ok::<_, Box<dyn std::error::Error>>(42)
        });
        
        let stats = tracker.get_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].0, "test_operation");
        assert_eq!(stats[0].1.count, 1);
        assert_eq!(stats[0].1.failures, 0);
    }
}