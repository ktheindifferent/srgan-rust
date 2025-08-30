use srgan_rust::error_recovery::*;
use srgan_rust::logging::{LogConfig, init_logging};
use srgan_rust::init::initialize_app;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use tempfile::TempDir;
use std::fs;
use std::path::PathBuf;

#[tokio::test]
async fn test_batch_processing_with_failures() {
    // Initialize logging for tests
    let _ = initialize_app(
        Some(tracing::Level::DEBUG),
        false,  // Disable telemetry for tests
        false,  // Disable file logging for tests
    ).await;
    
    // Create test directory structure
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(&output_dir).unwrap();
    
    // Create some test files (some valid, some invalid)
    create_test_image(&input_dir.join("valid1.png"));
    create_test_image(&input_dir.join("valid2.jpg"));
    create_corrupted_file(&input_dir.join("corrupted.png"));
    create_empty_file(&input_dir.join("empty.jpg"));
    
    // Set up error aggregator
    let error_aggregator = Arc::new(ErrorAggregator::new());
    
    // Process files with error recovery
    let processed = Arc::new(AtomicU32::new(0));
    let failed = Arc::new(AtomicU32::new(0));
    
    for entry in fs::read_dir(&input_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        
        let result = process_file_with_recovery(
            &path,
            &output_dir,
            &error_aggregator,
        ).await;
        
        match result {
            Ok(_) => processed.fetch_add(1, Ordering::SeqCst),
            Err(_) => failed.fetch_add(1, Ordering::SeqCst),
        };
    }
    
    // Verify results
    assert_eq!(processed.load(Ordering::SeqCst), 2); // 2 valid files
    assert_eq!(failed.load(Ordering::SeqCst), 2); // 2 invalid files
    
    // Check error aggregation
    let summary = error_aggregator.get_summary();
    assert_eq!(summary.total_errors, 2);
}

#[tokio::test]
async fn test_circuit_breaker_in_production_scenario() {
    // Simulate a service that fails intermittently
    let breaker = Arc::new(CircuitBreaker::new(3, 2, Duration::from_millis(100)));
    let failure_count = Arc::new(AtomicU32::new(0));
    
    // Simulate multiple concurrent requests
    let mut handles = vec![];
    
    for i in 0..10 {
        let breaker_clone = Arc::clone(&breaker);
        let failure_clone = Arc::clone(&failure_count);
        
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(i * 10)).await;
            
            let result = breaker_clone.call(|| {
                // Simulate 40% failure rate
                if i % 5 < 2 {
                    failure_clone.fetch_add(1, Ordering::SeqCst);
                    Err(EnhancedError::transient("Service error", None))
                } else {
                    Ok(format!("Success {}", i))
                }
            }).await;
            
            result
        });
        
        handles.push(handle);
    }
    
    // Wait for all requests
    let mut successes = 0;
    let mut circuit_open_errors = 0;
    
    for handle in handles {
        match handle.await.unwrap() {
            Ok(_) => successes += 1,
            Err(EnhancedError::CircuitBreakerOpen { .. }) => circuit_open_errors += 1,
            Err(_) => {}
        }
    }
    
    // Circuit should have opened at some point
    assert!(circuit_open_errors > 0);
    println!("Successes: {}, Circuit breaker trips: {}", successes, circuit_open_errors);
}

#[tokio::test]
async fn test_retry_with_jitter_and_backoff() {
    let config = RetryConfig {
        max_attempts: 5,
        initial_delay: Duration::from_millis(50),
        max_delay: Duration::from_secs(1),
        exponential_base: 2.0,
        jitter: true,
    };
    
    let executor = RetryExecutor::new(config);
    let mut context = ErrorContext::new("jitter_test");
    
    let attempt_times = Arc::new(std::sync::Mutex::new(Vec::new()));
    let times_clone = Arc::clone(&attempt_times);
    let attempt_count = Arc::new(AtomicU32::new(0));
    let count_clone = Arc::clone(&attempt_count);
    
    let start = std::time::Instant::now();
    
    let result = executor.execute(
        || {
            let count = count_clone.fetch_add(1, Ordering::SeqCst);
            times_clone.lock().unwrap().push(std::time::Instant::now());
            
            if count < 3 {
                Err(TransientError)
            } else {
                Ok("Success after retries")
            }
        },
        &mut context
    ).await;
    
    assert!(result.is_ok());
    assert_eq!(attempt_count.load(Ordering::SeqCst), 4); // Initial + 3 retries
    
    // Verify that delays increase with jitter
    let times = attempt_times.lock().unwrap();
    for i in 1..times.len() {
        let delay = times[i].duration_since(times[i-1]);
        println!("Delay {}: {:?}", i, delay);
        // With jitter, delays should vary but generally increase
        assert!(delay.as_millis() > 0);
    }
}

#[tokio::test]
async fn test_error_recovery_with_fallback() {
    // Test fallback mechanism when primary operation fails
    let primary_failures = Arc::new(AtomicU32::new(0));
    let fallback_calls = Arc::new(AtomicU32::new(0));
    
    let primary_clone = Arc::clone(&primary_failures);
    let fallback_clone = Arc::clone(&fallback_calls);
    
    let result = execute_with_fallback(
        || {
            primary_clone.fetch_add(1, Ordering::SeqCst);
            Err::<String, _>(EnhancedError::permanent("Primary failed"))
        },
        || {
            fallback_clone.fetch_add(1, Ordering::SeqCst);
            Ok("Fallback result".to_string())
        }
    ).await;
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Fallback result");
    assert_eq!(primary_failures.load(Ordering::SeqCst), 1);
    assert_eq!(fallback_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_graceful_degradation_under_load() {
    // Simulate high load scenario with graceful degradation
    let error_aggregator = Arc::new(ErrorAggregator::new());
    let breaker = Arc::new(CircuitBreaker::new(5, 3, Duration::from_millis(200)));
    
    let mut handles = vec![];
    
    // Simulate 50 concurrent requests
    for i in 0..50 {
        let aggregator_clone = Arc::clone(&error_aggregator);
        let breaker_clone = Arc::clone(&breaker);
        
        let handle = tokio::spawn(async move {
            // Add some randomness to simulate real-world timing
            tokio::time::sleep(Duration::from_millis(i % 10)).await;
            
            let result = breaker_clone.call(|| {
                // Simulate varying success rates
                if i % 10 < 3 {
                    // 30% failure rate
                    Err(EnhancedError::transient("High load error", Some(Duration::from_millis(100))))
                } else {
                    Ok(i)
                }
            }).await;
            
            if let Err(e) = &result {
                aggregator_clone.record_error(
                    "high_load",
                    PathBuf::from(format!("request_{}", i)),
                    format!("{}", e)
                );
            }
            
            result
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut success_count = 0;
    let mut circuit_breaker_count = 0;
    let mut error_count = 0;
    
    for handle in handles {
        match handle.await.unwrap() {
            Ok(_) => success_count += 1,
            Err(EnhancedError::CircuitBreakerOpen { .. }) => circuit_breaker_count += 1,
            Err(_) => error_count += 1,
        }
    }
    
    println!("Load test results:");
    println!("  Successes: {}", success_count);
    println!("  Circuit breaker trips: {}", circuit_breaker_count);
    println!("  Other errors: {}", error_count);
    
    let summary = error_aggregator.get_summary();
    println!("  Total errors recorded: {}", summary.total_errors);
    
    // Verify that the system handled the load
    assert!(success_count > 0);
    assert_eq!(success_count + circuit_breaker_count + error_count, 50);
}

// Helper functions

fn create_test_image(path: &PathBuf) {
    use image::{ImageBuffer, Rgb};
    
    let img = ImageBuffer::from_fn(100, 100, |x, y| {
        Rgb([
            ((x * 255) / 100) as u8,
            ((y * 255) / 100) as u8,
            128
        ])
    });
    
    img.save(path).unwrap();
}

fn create_corrupted_file(path: &PathBuf) {
    fs::write(path, b"This is not a valid image file").unwrap();
}

fn create_empty_file(path: &PathBuf) {
    fs::write(path, b"").unwrap();
}

async fn process_file_with_recovery(
    input_path: &PathBuf,
    output_dir: &PathBuf,
    error_aggregator: &Arc<ErrorAggregator>,
) -> Result<(), EnhancedError> {
    // Simulate file processing with error handling
    match fs::read(input_path) {
        Ok(data) if data.is_empty() => {
            error_aggregator.record_error(
                "empty_file",
                input_path.clone(),
                "File is empty"
            );
            Err(EnhancedError::permanent("Empty file"))
        }
        Ok(data) => {
            // Try to load as image
            match image::load_from_memory(&data) {
                Ok(_img) => {
                    // Simulate successful processing
                    let output_path = output_dir.join(input_path.file_name().unwrap());
                    fs::write(output_path, data).map_err(|e| {
                        EnhancedError::transient(format!("Write failed: {}", e), None)
                    })?;
                    Ok(())
                }
                Err(e) => {
                    error_aggregator.record_error(
                        "invalid_image",
                        input_path.clone(),
                        format!("Invalid image: {}", e)
                    );
                    Err(EnhancedError::permanent(format!("Invalid image: {}", e)))
                }
            }
        }
        Err(e) => {
            error_aggregator.record_error(
                "read_error",
                input_path.clone(),
                format!("Read error: {}", e)
            );
            Err(EnhancedError::transient(format!("Read error: {}", e), None))
        }
    }
}

async fn execute_with_fallback<T, E>(
    primary: impl FnOnce() -> Result<T, E>,
    fallback: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    primary().or_else(|_| fallback())
}

// Test error type
#[derive(Debug)]
struct TransientError;

impl std::fmt::Display for TransientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Transient error for testing")
    }
}

impl IsRetryable for TransientError {
    fn is_retryable(&self) -> bool {
        true
    }
}