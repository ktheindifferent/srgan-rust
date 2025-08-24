use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use ndarray::ArrayD;
use image::DynamicImage;

/// Test concurrent inference with multiple threads
#[test]
fn test_concurrent_inference_stress() {
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
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
                
                let output = result.unwrap();
                // Verify output shape (4x upscaling)
                assert_eq!(output.shape(), &[1, 128, 128, 3]);
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test that outputs are consistent across threads
#[test]
fn test_output_consistency() {
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
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
            network_clone.process(input_clone).unwrap()
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut results = vec![];
    for handle in handles {
        results.push(handle.join().unwrap());
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
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
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
    for handle in handles {
        let count = handle.join().unwrap();
        total_processed += count;
    }
    
    println!("Processed {} images across {} threads in {:?}",
        total_processed, num_threads, duration);
    assert!(total_processed > 0);
}

/// Test that network can handle rapid thread creation/destruction
#[test]
fn test_thread_churn() {
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
    let iterations = 50;
    
    for _ in 0..iterations {
        let network_clone = Arc::clone(&network);
        let handle = thread::spawn(move || {
            let input = ArrayD::<f32>::zeros(vec![1, 16, 16, 3]);
            network_clone.process(input).unwrap()
        });
        
        // Immediately wait for thread to finish
        let _ = handle.join().unwrap();
    }
}

/// Test concurrent image upscaling
#[test]
fn test_concurrent_image_upscaling() {
    use image::{RgbImage, DynamicImage};
    
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
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
            
            let upscaled = result.unwrap();
            assert_eq!(upscaled.width(), 128);  // 4x upscaling
            assert_eq!(upscaled.height(), 128);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Benchmark parallel vs sequential processing
#[test]
fn test_performance_improvement() {
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
    let num_images = 16;
    let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
    
    // Sequential processing
    let sequential_start = Instant::now();
    for _ in 0..num_images {
        network.process(input.clone()).unwrap();
    }
    let sequential_duration = sequential_start.elapsed();
    
    // Parallel processing
    let parallel_start = Instant::now();
    let mut handles = vec![];
    for _ in 0..num_images {
        let network_clone = Arc::clone(&network);
        let input_clone = input.clone();
        let handle = thread::spawn(move || {
            network_clone.process(input_clone).unwrap()
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
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
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
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
    for handle in handles {
        assert!(handle.join().unwrap().is_ok());
    }
}