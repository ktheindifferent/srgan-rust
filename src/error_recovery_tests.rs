#[cfg(test)]
mod error_recovery_tests {
    use super::*;
    use crate::error_recovery::*;
    use crate::error::SrganError;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio;
    
    #[tokio::test]
    async fn test_retry_executor_success_after_failures() {
        let executor = RetryExecutor::with_default();
        let mut context = ErrorContext::new("test_operation")
            .with_attempts(5);
        
        let attempt_counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&attempt_counter);
        
        let result = executor.execute(
            || {
                let count = counter_clone.fetch_add(1, Ordering::SeqCst);
                if count < 2 {
                    Err(TestError::Transient)
                } else {
                    Ok(42)
                }
            },
            &mut context
        ).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_counter.load(Ordering::SeqCst), 3);
    }
    
    #[tokio::test]
    async fn test_retry_executor_permanent_failure() {
        let executor = RetryExecutor::with_default();
        let mut context = ErrorContext::new("test_operation");
        
        let result = executor.execute(
            || Err::<i32, _>(TestError::Permanent),
            &mut context
        ).await;
        
        assert!(result.is_err());
        assert_eq!(context.attempt, 1); // Should not retry permanent errors
    }
    
    #[tokio::test]
    async fn test_retry_executor_max_attempts() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(1),
            exponential_base: 2.0,
            jitter: false,
        };
        
        let executor = RetryExecutor::new(config);
        let mut context = ErrorContext::new("test_operation");
        let attempt_counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&attempt_counter);
        
        let result = executor.execute(
            || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(TestError::Transient)
            },
            &mut context
        ).await;
        
        assert!(result.is_err());
        assert_eq!(attempt_counter.load(Ordering::SeqCst), 3);
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        let breaker = CircuitBreaker::new(2, 2, Duration::from_millis(100));
        
        // First two failures should open the circuit
        for _ in 0..2 {
            let _ = breaker.call(|| Err::<(), _>(EnhancedError::permanent("Error"))).await;
        }
        
        // Circuit should be open now
        let result = breaker.call(|| Ok::<_, EnhancedError>(42)).await;
        assert!(matches!(result, Err(EnhancedError::CircuitBreakerOpen { .. })));
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_half_open_recovery() {
        let breaker = CircuitBreaker::new(2, 2, Duration::from_millis(50));
        
        // Open the circuit
        for _ in 0..2 {
            let _ = breaker.call(|| Err::<(), _>(EnhancedError::permanent("Error"))).await;
        }
        
        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(60)).await;
        
        // Should be half-open, first success
        let result = breaker.call(|| Ok::<_, EnhancedError>(1)).await;
        assert!(result.is_ok());
        
        // Second success should close the circuit
        let result = breaker.call(|| Ok::<_, EnhancedError>(2)).await;
        assert!(result.is_ok());
        
        // Circuit should be closed, can handle failures again
        let _ = breaker.call(|| Err::<(), _>(EnhancedError::permanent("Error"))).await;
        // Should still work (one failure doesn't open)
        let result = breaker.call(|| Ok::<_, EnhancedError>(3)).await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_error_aggregator_categorization() {
        let aggregator = ErrorAggregator::new();
        use std::path::PathBuf;
        
        // Add various errors
        aggregator.record_error("IO", PathBuf::from("file1.txt"), "Read failed");
        aggregator.record_error("IO", PathBuf::from("file2.txt"), "Write failed");
        aggregator.record_error("Network", PathBuf::from("remote.jpg"), "Download failed");
        aggregator.record_error("Processing", PathBuf::from("image.png"), "Corruption detected");
        
        let summary = aggregator.get_summary();
        
        assert_eq!(summary.total_errors, 4);
        assert!(summary.categories.iter().any(|c| c.name == "IO" && c.count == 2));
        assert!(summary.categories.iter().any(|c| c.name == "Network" && c.count == 1));
        assert!(summary.categories.iter().any(|c| c.name == "Processing" && c.count == 1));
    }
    
    #[test]
    fn test_error_aggregator_clear() {
        let aggregator = ErrorAggregator::new();
        use std::path::PathBuf;
        
        aggregator.record_error("Test", PathBuf::from("test.txt"), "Test error");
        assert_eq!(aggregator.get_summary().total_errors, 1);
        
        aggregator.clear();
        assert_eq!(aggregator.get_summary().total_errors, 0);
    }
    
    #[test]
    fn test_enhanced_error_retryable() {
        let transient = EnhancedError::transient("Temporary failure", Some(Duration::from_secs(1)));
        assert!(transient.is_retryable());
        assert_eq!(transient.retry_after(), Some(Duration::from_secs(1)));
        
        let permanent = EnhancedError::permanent("Permanent failure");
        assert!(!permanent.is_retryable());
        assert_eq!(permanent.retry_after(), None);
        
        let rate_limit = EnhancedError::RateLimit {
            message: "Too many requests".to_string(),
            retry_after: Duration::from_secs(60),
            backtrace: backtrace::Backtrace::new(),
        };
        assert!(rate_limit.is_retryable());
        assert_eq!(rate_limit.retry_after(), Some(Duration::from_secs(60)));
    }
    
    #[test]
    fn test_error_context_builder() {
        use std::path::PathBuf;
        
        let context = ErrorContext::new("test_op")
            .with_file(PathBuf::from("test.txt"))
            .with_attempts(5)
            .add_metadata("key", "value");
        
        assert_eq!(context.operation, "test_op");
        assert_eq!(context.file_path, Some(PathBuf::from("test.txt")));
        assert_eq!(context.max_attempts, 5);
        assert_eq!(context.metadata.get("key"), Some(&"value".to_string()));
    }
    
    #[tokio::test]
    async fn test_retry_with_exponential_backoff() {
        let config = RetryConfig {
            max_attempts: 4,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            exponential_base: 2.0,
            jitter: false,
        };
        
        let executor = RetryExecutor::new(config);
        let mut context = ErrorContext::new("backoff_test");
        
        let start = std::time::Instant::now();
        let attempt_times = Arc::new(std::sync::Mutex::new(Vec::new()));
        let times_clone = Arc::clone(&attempt_times);
        
        let _ = executor.execute(
            || {
                times_clone.lock().unwrap().push(std::time::Instant::now());
                Err::<(), _>(TestError::Transient)
            },
            &mut context
        ).await;
        
        let times = attempt_times.lock().unwrap();
        assert_eq!(times.len(), 4);
        
        // Verify exponential backoff delays
        for i in 1..times.len() {
            let delay = times[i].duration_since(times[i-1]);
            let expected_min = Duration::from_millis(10 * 2_u64.pow(i as u32 - 1));
            // Allow some tolerance for timing
            assert!(delay >= expected_min.saturating_sub(Duration::from_millis(5)));
        }
    }
    
    // Test error types for testing
    #[derive(Debug)]
    enum TestError {
        Transient,
        Permanent,
    }
    
    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestError::Transient => write!(f, "Transient test error"),
                TestError::Permanent => write!(f, "Permanent test error"),
            }
        }
    }
    
    impl IsRetryable for TestError {
        fn is_retryable(&self) -> bool {
            matches!(self, TestError::Transient)
        }
    }
}