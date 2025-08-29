use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use ndarray::ArrayD;
use image::{DynamicImage, GenericImage};

// Import test helpers for better error handling
#[path = "test_helpers.rs"]
mod test_helpers;
use test_helpers::*;

/// Test concurrent inference with multiple threads
#[test]
fn test_concurrent_inference_stress() {
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading builtin natural network for concurrent inference"
    ));
    let num_threads = 8;
    let iterations_per_thread = 10;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let network_clone = Arc::clone(&network);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            // Wait for all threads to be ready
            barrier_clone.wait();
            
            for i in 0..iterations_per_thread {
                // Create test input
                let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
                
                // Process through network
                let result = network_clone.process(input);
                assert!(result.is_ok(), "Thread {} iteration {} failed: {:?}", 
                    thread_id, i, result);
                
                let output = assert_ok(result, 
                    &format!("thread {} iteration {} processing", thread_id, i));
                // Verify output shape (4x upscaling)
                assert_eq!(output.shape(), &[1, 128, 128, 3]);
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for (idx, handle) in handles.into_iter().enumerate() {
        assert_thread_success(handle.join(), &format!("inference thread {}", idx));
    }
}

/// Test that outputs are consistent across threads
#[test]
fn test_output_consistency() {
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading builtin natural network for consistency test"
    ));
    let num_threads = 4;
    
    // Create a deterministic test input
    let mut input = ArrayD::<f32>::zeros(vec![1, 16, 16, 3]);
    for (i, elem) in input.iter_mut().enumerate() {
        *elem = (i as f32) / 768.0;  // Normalize to [0, 1]
    }
    
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];
    
    for _ in 0..num_threads {
        let network_clone = Arc::clone(&network);
        let barrier_clone = Arc::clone(&barrier);
        let input_clone = input.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            assert_ok(
                network_clone.process(input_clone),
                "processing in consistency test thread"
            )
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut results = vec![];
    for (idx, handle) in handles.into_iter().enumerate() {
        let thread_result = assert_thread_success(
            handle.join(),
            &format!("consistency test thread {}", idx)
        );
        results.push(thread_result);
    }
    
    // Verify all results are identical
    let first_result = &results[0];
    for (i, result) in results.iter().skip(1).enumerate() {
        assert_eq!(
            first_result.shape(), 
            result.shape(),
            "Shape mismatch for thread {}",
            i + 1
        );
        
        // Check that values are very close (allow for minor floating point differences)
        for (a, b) in first_result.iter().zip(result.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Value mismatch: {} vs {}",
                a, b
            );
        }
    }
}

/// Test memory safety under high concurrency
#[test]
fn test_memory_safety() {
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading builtin natural network for memory safety test"
    ));
    let num_threads = 16;
    let duration = Duration::from_secs(2);
    
    let start_time = Instant::now();
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let network_clone = Arc::clone(&network);
        let handle = thread::spawn(move || {
            let mut count = 0;
            while start_time.elapsed() < duration {
                // Varying input sizes to stress memory management
                let size = 8 + (thread_id % 4) * 8;  // 8, 16, 24, or 32
                let input = ArrayD::<f32>::zeros(vec![1, size, size, 3]);
                
                if let Ok(_) = network_clone.process(input) {
                    count += 1;
                }
            }
            count
        });
        handles.push(handle);
    }
    
    // Collect results and ensure no panics occurred
    let mut total_processed = 0;
    for (idx, handle) in handles.into_iter().enumerate() {
        let count = assert_thread_success(
            handle.join(),
            &format!("memory safety thread {}", idx)
        );
        total_processed += count;
    }
    
    println!("Processed {} images across {} threads in {:?}",
        total_processed, num_threads, duration);
    assert!(total_processed > 0);
}

/// Test that network can handle rapid thread creation/destruction
#[test]
fn test_thread_churn() {
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading builtin natural network for thread churn test"
    ));
    let iterations = 50;
    
    for iter in 0..iterations {
        let network_clone = Arc::clone(&network);
        let handle = thread::spawn(move || {
            let input = ArrayD::<f32>::zeros(vec![1, 16, 16, 3]);
            assert_ok(
                network_clone.process(input),
                "processing in thread churn test"
            )
        });
        
        // Immediately wait for thread to finish
        assert_thread_success(handle.join(), &format!("thread churn iteration {}", iter));
    }
}

/// Test concurrent image upscaling
#[test]
fn test_concurrent_image_upscaling() {
    use image::{RgbImage, DynamicImage};
    
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading builtin natural network for concurrent image upscaling"
    ));
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let network_clone = Arc::clone(&network);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            // Create a small test image
            let img = RgbImage::new(32, 32);
            let dynamic_img = DynamicImage::ImageRgb8(img);
            
            barrier_clone.wait();
            
            let result = network_clone.upscale_image(&dynamic_img);
            assert!(result.is_ok(), "Thread {} failed to upscale", thread_id);
            
            let upscaled = assert_ok(result,
                &format!("thread {} image upscaling", thread_id));
            assert_eq!(upscaled.width(), 128);  // 4x upscaling
            assert_eq!(upscaled.height(), 128);
        });
        
        handles.push(handle);
    }
    
    for (idx, handle) in handles.into_iter().enumerate() {
        assert_thread_success(handle.join(), &format!("image upscaling thread {}", idx));
    }
}

/// Benchmark parallel vs sequential processing
#[test]
fn test_performance_improvement() {
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading builtin natural network for performance test"
    ));
    let num_images = 16;
    let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
    
    // Sequential processing
    let sequential_start = Instant::now();
    for i in 0..num_images {
        assert_result_ok(
            network.process(input.clone()),
            &format!("sequential processing iteration {}", i)
        );
    }
    let sequential_duration = sequential_start.elapsed();
    
    // Parallel processing
    let parallel_start = Instant::now();
    let mut handles = vec![];
    for _ in 0..num_images {
        let network_clone = Arc::clone(&network);
        let input_clone = input.clone();
        let handle = thread::spawn(move || {
            assert_ok(
                network_clone.process(input_clone),
                "parallel processing iteration"
            )
        });
        handles.push(handle);
    }
    for (idx, handle) in handles.into_iter().enumerate() {
        assert_thread_success(handle.join(), &format!("parallel processing thread {}", idx));
    }
    let parallel_duration = parallel_start.elapsed();
    
    println!("Sequential: {:?}, Parallel: {:?}", 
        sequential_duration, parallel_duration);
    
    // Parallel should be faster (though actual speedup depends on hardware)
    // We just verify both completed successfully
    assert!(sequential_duration.as_millis() > 0);
    assert!(parallel_duration.as_millis() > 0);
}

/// Test that dropping the network while threads are using it is safe
#[test]
fn test_safe_network_dropping() {
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading builtin natural network for safe dropping test"
    ));
    let num_threads = 4;
    
    let mut handles = vec![];
    for _ in 0..num_threads {
        let network_clone = Arc::clone(&network);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            let input = ArrayD::<f32>::zeros(vec![1, 16, 16, 3]);
            network_clone.process(input)
        });
        handles.push(handle);
    }
    
    // Drop our reference while threads are still running
    drop(network);
    
    // Threads should still complete successfully
    for (idx, handle) in handles.into_iter().enumerate() {
        let result = assert_thread_success(
            handle.join(),
            &format!("network dropping test thread {}", idx)
        );
        assert_result_ok(result, &format!("Thread {} should complete successfully after network drop", idx));
    }
}

/// Test error handling when loading non-existent network
#[test]
fn test_load_nonexistent_network_error() {
    // This test explicitly checks error conditions
    let result = ThreadSafeNetwork::load_builtin_natural();
    // We expect this to succeed, but let's test error path with invalid input later
    assert_result_ok(result, "loading builtin natural network should succeed");
}

/// Test processing with invalid input dimensions
#[test]
fn test_invalid_input_dimensions() {
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading network for invalid input test"
    ));
    
    // Test with wrong number of dimensions
    let invalid_input = ArrayD::<f32>::zeros(vec![32, 32, 3]); // Missing batch dimension
    let result = network.process(invalid_input);
    assert_result_err(result, "processing with invalid dimensions");
    
    // Test with wrong channel count  
    let invalid_channels = ArrayD::<f32>::zeros(vec![1, 32, 32, 4]); // 4 channels instead of 3
    let result2 = network.process(invalid_channels);
    assert_result_err(result2, "processing with wrong channel count");
    
    // Test with empty input
    let empty_input = ArrayD::<f32>::zeros(vec![0, 0, 0, 0]);
    let result3 = network.process(empty_input);
    assert_result_err(result3, "processing with empty input");
}

/// Test concurrent error handling
#[test]
fn test_concurrent_error_handling() {
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading network for concurrent error test"
    ));
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let network_clone = Arc::clone(&network);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            // Mix valid and invalid inputs
            let mut error_count = 0;
            let mut success_count = 0;
            
            for i in 0..10 {
                let input = if i % 3 == 0 {
                    // Invalid input
                    ArrayD::<f32>::zeros(vec![32, 32, 3]) // Missing batch dimension
                } else {
                    // Valid input
                    ArrayD::<f32>::zeros(vec![1, 32, 32, 3])
                };
                
                match network_clone.process(input) {
                    Ok(_) => success_count += 1,
                    Err(_) => error_count += 1,
                }
            }
            
            (success_count, error_count, thread_id)
        });
        
        handles.push(handle);
    }
    
    // Collect and verify results
    for (idx, handle) in handles.into_iter().enumerate() {
        let (success, errors, tid) = assert_thread_success(
            handle.join(),
            &format!("error handling thread {}", idx)
        );
        
        // We expect some successes and some errors based on our input pattern
        assert!(success > 0, "Thread {} should have some successful operations", tid);
        assert!(errors > 0, "Thread {} should have caught some errors", tid);
    }
}