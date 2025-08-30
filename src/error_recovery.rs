use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use backtrace::Backtrace;
use thiserror::Error;
use tracing::{error, warn, info, debug};

/// Enhanced error type with backtrace support and categorization
#[derive(Debug, Error)]
pub enum EnhancedError {
    #[error("IO error: {message}")]
    Io {
        message: String,
        #[source]
        source: io::Error,
        backtrace: Backtrace,
    },
    
    #[error("Image processing error: {message}")]
    Image {
        message: String,
        #[source]
        source: image::ImageError,
        backtrace: Backtrace,
    },
    
    #[error("Network error: {message}")]
    Network {
        message: String,
        backtrace: Backtrace,
        retryable: bool,
    },
    
    #[error("Validation error: {message}")]
    Validation {
        message: String,
        field: Option<String>,
        backtrace: Backtrace,
    },
    
    #[error("Transient error (retryable): {message}")]
    Transient {
        message: String,
        retry_after: Option<Duration>,
        backtrace: Backtrace,
    },
    
    #[error("Permanent error: {message}")]
    Permanent {
        message: String,
        backtrace: Backtrace,
    },
    
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        message: String,
        retry_after: Duration,
        backtrace: Backtrace,
    },
    
    #[error("Circuit breaker open: {message}")]
    CircuitBreakerOpen {
        message: String,
        reset_after: Duration,
        backtrace: Backtrace,
    },
}

impl EnhancedError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            EnhancedError::Transient { .. }
                | EnhancedError::Network { retryable: true, .. }
                | EnhancedError::RateLimit { .. }
        )
    }
    
    /// Get suggested retry delay
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            EnhancedError::Transient { retry_after, .. } => *retry_after,
            EnhancedError::RateLimit { retry_after, .. } => Some(*retry_after),
            EnhancedError::CircuitBreakerOpen { reset_after, .. } => Some(*reset_after),
            _ => None,
        }
    }
    
    /// Create a transient error with retry hint
    pub fn transient<S: Into<String>>(message: S, retry_after: Option<Duration>) -> Self {
        EnhancedError::Transient {
            message: message.into(),
            retry_after,
            backtrace: Backtrace::new(),
        }
    }
    
    /// Create a permanent error
    pub fn permanent<S: Into<String>>(message: S) -> Self {
        EnhancedError::Permanent {
            message: message.into(),
            backtrace: Backtrace::new(),
        }
    }
}

/// Error context for tracking error occurrences
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub file_path: Option<PathBuf>,
    pub attempt: u32,
    pub max_attempts: u32,
    pub metadata: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>) -> Self {
        ErrorContext {
            operation: operation.into(),
            file_path: None,
            attempt: 1,
            max_attempts: 3,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_file(mut self, path: PathBuf) -> Self {
        self.file_path = Some(path);
        self
    }
    
    pub fn with_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }
    
    pub fn add_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            exponential_base: 2.0,
            jitter: true,
        }
    }
}

/// Retry logic implementation
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    pub fn new(config: RetryConfig) -> Self {
        RetryExecutor { config }
    }
    
    pub fn with_default() -> Self {
        RetryExecutor::new(RetryConfig::default())
    }
    
    /// Execute a function with retry logic
    pub async fn execute<F, T, E>(&self, mut operation: F, context: &mut ErrorContext) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
        E: std::fmt::Display + IsRetryable,
    {
        let mut delay = self.config.initial_delay;
        
        for attempt in 1..=self.config.max_attempts {
            context.attempt = attempt;
            
            match operation() {
                Ok(result) => {
                    if attempt > 1 {
                        info!(
                            "Operation '{}' succeeded on attempt {}",
                            context.operation, attempt
                        );
                    }
                    return Ok(result);
                }
                Err(err) => {
                    if !err.is_retryable() || attempt == self.config.max_attempts {
                        error!(
                            "Operation '{}' failed permanently after {} attempts: {}",
                            context.operation, attempt, err
                        );
                        return Err(err);
                    }
                    
                    warn!(
                        "Operation '{}' failed on attempt {}/{}: {}. Retrying after {:?}",
                        context.operation, attempt, self.config.max_attempts, err, delay
                    );
                    
                    tokio::time::sleep(delay).await;
                    
                    // Calculate next delay with exponential backoff
                    delay = self.calculate_next_delay(delay);
                }
            }
        }
        
        unreachable!()
    }
    
    fn calculate_next_delay(&self, current: Duration) -> Duration {
        let mut next = Duration::from_secs_f64(
            current.as_secs_f64() * self.config.exponential_base
        );
        
        if next > self.config.max_delay {
            next = self.config.max_delay;
        }
        
        if self.config.jitter {
            use rand::Rng;
            let jitter = rand::thread_rng().gen_range(0.8..1.2);
            next = Duration::from_secs_f64(next.as_secs_f64() * jitter);
        }
        
        next
    }
}

/// Trait for determining if an error is retryable
pub trait IsRetryable {
    fn is_retryable(&self) -> bool;
}

impl IsRetryable for EnhancedError {
    fn is_retryable(&self) -> bool {
        self.is_retryable()
    }
}

/// Circuit breaker implementation for protecting against cascading failures
pub struct CircuitBreaker {
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
    state: Arc<tokio::sync::RwLock<CircuitBreakerState>>,
}

#[derive(Debug, Clone)]
enum CircuitBreakerState {
    Closed {
        failure_count: u32,
    },
    Open {
        opened_at: std::time::Instant,
    },
    HalfOpen {
        success_count: u32,
    },
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, success_threshold: u32, timeout: Duration) -> Self {
        CircuitBreaker {
            failure_threshold,
            success_threshold,
            timeout,
            state: Arc::new(tokio::sync::RwLock::new(CircuitBreakerState::Closed {
                failure_count: 0,
            })),
        }
    }
    
    pub async fn call<F, T, E>(&self, operation: F) -> Result<T, EnhancedError>
    where
        F: FnOnce() -> Result<T, E>,
        E: Into<EnhancedError>,
    {
        let mut state = self.state.write().await;
        
        match &*state {
            CircuitBreakerState::Open { opened_at } => {
                if opened_at.elapsed() >= self.timeout {
                    info!("Circuit breaker transitioning to half-open");
                    *state = CircuitBreakerState::HalfOpen { success_count: 0 };
                } else {
                    let remaining = self.timeout - opened_at.elapsed();
                    return Err(EnhancedError::CircuitBreakerOpen {
                        message: "Circuit breaker is open".to_string(),
                        reset_after: remaining,
                        backtrace: Backtrace::new(),
                    });
                }
            }
            _ => {}
        }
        
        drop(state); // Release lock before operation
        
        match operation() {
            Ok(result) => {
                let mut state = self.state.write().await;
                match &mut *state {
                    CircuitBreakerState::HalfOpen { success_count } => {
                        *success_count += 1;
                        if *success_count >= self.success_threshold {
                            info!("Circuit breaker closing after successful recovery");
                            *state = CircuitBreakerState::Closed { failure_count: 0 };
                        }
                    }
                    CircuitBreakerState::Closed { failure_count } => {
                        *failure_count = 0;
                    }
                    _ => {}
                }
                Ok(result)
            }
            Err(err) => {
                let enhanced_err = err.into();
                let mut state = self.state.write().await;
                
                match &mut *state {
                    CircuitBreakerState::Closed { failure_count } => {
                        *failure_count += 1;
                        if *failure_count >= self.failure_threshold {
                            warn!("Circuit breaker opening after {} failures", failure_count);
                            *state = CircuitBreakerState::Open {
                                opened_at: std::time::Instant::now(),
                            };
                        }
                    }
                    CircuitBreakerState::HalfOpen { .. } => {
                        warn!("Circuit breaker reopening after failure in half-open state");
                        *state = CircuitBreakerState::Open {
                            opened_at: std::time::Instant::now(),
                        };
                    }
                    _ => {}
                }
                
                Err(enhanced_err)
            }
        }
    }
}

/// Error recovery strategies
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    Retry(RetryConfig),
    /// Use fallback value
    Fallback(Box<dyn Fn() -> Box<dyn std::any::Any + Send>>),
    /// Skip and continue
    Skip,
    /// Fail fast
    FailFast,
    /// Use circuit breaker
    CircuitBreaker(CircuitBreaker),
}

/// Error aggregator for batch operations
#[derive(Debug, Default)]
pub struct ErrorAggregator {
    errors: dashmap::DashMap<String, Vec<(PathBuf, String, Backtrace)>>,
    error_counts: dashmap::DashMap<String, usize>,
    total_errors: std::sync::atomic::AtomicUsize,
}

impl ErrorAggregator {
    pub fn new() -> Self {
        ErrorAggregator::default()
    }
    
    pub fn record_error(&self, category: impl Into<String>, path: PathBuf, message: impl Into<String>) {
        let category = category.into();
        let message = message.into();
        let backtrace = Backtrace::new();
        
        self.errors
            .entry(category.clone())
            .or_insert_with(Vec::new)
            .push((path, message, backtrace));
        
        self.error_counts
            .entry(category)
            .and_modify(|c| *c += 1)
            .or_insert(1);
        
        self.total_errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn get_summary(&self) -> ErrorSummary {
        let mut categories = Vec::new();
        
        for entry in self.error_counts.iter() {
            let (category, count) = entry.pair();
            let errors = self.errors.get(category)
                .map(|e| e.clone())
                .unwrap_or_default();
            
            categories.push(ErrorCategory {
                name: category.clone(),
                count: *count,
                sample_errors: errors.into_iter().take(5).collect(),
            });
        }
        
        ErrorSummary {
            total_errors: self.total_errors.load(std::sync::atomic::Ordering::Relaxed),
            categories,
        }
    }
    
    pub fn clear(&self) {
        self.errors.clear();
        self.error_counts.clear();
        self.total_errors.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct ErrorSummary {
    pub total_errors: usize,
    pub categories: Vec<ErrorCategory>,
}

#[derive(Debug)]
pub struct ErrorCategory {
    pub name: String,
    pub count: usize,
    pub sample_errors: Vec<(PathBuf, String, Backtrace)>,
}

impl fmt::Display for ErrorSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Error Summary - Total: {}", self.total_errors)?;
        for category in &self.categories {
            writeln!(f, "\n  {} ({} errors):", category.name, category.count)?;
            for (path, message, _) in category.sample_errors.iter().take(3) {
                writeln!(f, "    - {}: {}", path.display(), message)?;
            }
            if category.count > 3 {
                writeln!(f, "    ... and {} more", category.count - 3)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_retry_executor() {
        let executor = RetryExecutor::with_default();
        let mut context = ErrorContext::new("test_operation");
        let mut attempt_count = 0;
        
        let result = executor.execute(
            || {
                attempt_count += 1;
                if attempt_count < 3 {
                    Err(EnhancedError::transient("Temporary failure", None))
                } else {
                    Ok(42)
                }
            },
            &mut context
        ).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count, 3);
    }
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(2, 2, Duration::from_millis(100));
        
        // First failure
        let _ = breaker.call(|| Err::<(), _>(EnhancedError::permanent("Error 1"))).await;
        
        // Second failure - should open circuit
        let _ = breaker.call(|| Err::<(), _>(EnhancedError::permanent("Error 2"))).await;
        
        // Third call should fail immediately with circuit open
        let result = breaker.call(|| Ok::<_, EnhancedError>(42)).await;
        assert!(matches!(result, Err(EnhancedError::CircuitBreakerOpen { .. })));
        
        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should transition to half-open and allow call
        let result = breaker.call(|| Ok::<_, EnhancedError>(42)).await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_error_aggregator() {
        let aggregator = ErrorAggregator::new();
        
        aggregator.record_error("IO", PathBuf::from("file1.txt"), "Read failed");
        aggregator.record_error("IO", PathBuf::from("file2.txt"), "Write failed");
        aggregator.record_error("Network", PathBuf::from("remote.jpg"), "Download failed");
        
        let summary = aggregator.get_summary();
        assert_eq!(summary.total_errors, 3);
        assert_eq!(summary.categories.len(), 2);
    }
}