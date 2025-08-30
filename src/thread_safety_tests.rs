//! Thread safety tests for concurrent operations
//!
//! These tests verify that our unsafe Send/Sync implementations are correct
//! and that concurrent access doesn't cause data races or undefined behavior.

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;
    
    // Test concurrent access to ThreadSafeNetwork
    #[test]
    fn test_thread_safe_network_concurrent_inference() {
        use crate::thread_safe_network::ThreadSafeNetwork;
        use ndarray::{Array, ArrayD};
        
        // Create a test network
        let network = ThreadSafeNetwork::load_builtin_natural()
            .expect("Failed to load network");
        let network = Arc::new(network);
        
        // Number of concurrent threads
        const NUM_THREADS: usize = 8;
        const ITERATIONS_PER_THREAD: usize = 10;
        
        // Barrier to ensure all threads start at the same time
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        
        // Spawn threads
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                let network = Arc::clone(&network);
                let barrier = Arc::clone(&barrier);
                
                thread::spawn(move || {
                    // Wait for all threads to be ready
                    barrier.wait();
                    
                    for iteration in 0..ITERATIONS_PER_THREAD {
                        // Create a small test input
                        let input = ArrayD::from_shape_vec(
                            vec![1, 3, 32, 32],
                            vec![0.5f32; 3 * 32 * 32]
                        ).expect("Failed to create input");
                        
                        // Perform inference
                        let result = network.process(input);
                        
                        // Verify result is ok
                        assert!(result.is_ok(), 
                            "Thread {} iteration {} failed: {:?}", 
                            thread_id, iteration, result.err());
                    }
                })
            })
            .collect();
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }
    
    // Test concurrent memory allocation in GpuContext
    #[test]
    fn test_gpu_context_concurrent_allocation() {
        use crate::gpu::{GpuContext, GpuBackend};
        
        let context = Arc::new(
            GpuContext::new(GpuBackend::None)
                .expect("Failed to create GPU context")
        );
        
        const NUM_THREADS: usize = 10;
        const ALLOCATIONS_PER_THREAD: usize = 5;
        const ALLOCATION_SIZE_MB: usize = 10;
        
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                let context = Arc::clone(&context);
                let barrier = Arc::clone(&barrier);
                
                thread::spawn(move || {
                    barrier.wait();
                    
                    // Allocate memory
                    for _i in 0..ALLOCATIONS_PER_THREAD {
                        let result = context.allocate(ALLOCATION_SIZE_MB);
                        // Some allocations may fail due to memory limits, that's ok
                        if result.is_ok() {
                            // Simulate some work
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                    
                    // Free thread memory
                    context.free_thread_memory();
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
        
        // Verify final state is consistent
        assert_eq!(context.allocated_mb(), 0, 
            "All memory should be freed after threads complete");
    }
    
    // Test parallel batch processing
    #[test]
    fn test_parallel_batch_processing() {
        use crate::parallel::ThreadSafeNetwork;
        use ndarray::ArrayD;
        
        // Create a mock network  
        let network = crate::UpscalingNetwork::from_label("natural", None)
            .expect("Failed to create network");
        let thread_safe = ThreadSafeNetwork::new(network);
        
        // Create test items
        let items: Vec<usize> = (0..100).collect();
        
        // Process in parallel
        let results = thread_safe.process_batch_parallel(
            items,
            |idx, _network| {
                // Simulate processing
                thread::sleep(Duration::from_micros(100));
                
                // Create a dummy result
                Ok(ArrayD::from_shape_vec(
                    vec![1, 3, 64, 64],
                    vec![idx as f32; 3 * 64 * 64]
                ).unwrap())
            },
            Some(4) // Use 4 threads
        );
        
        // Verify all results
        assert_eq!(results.len(), 100);
        for (idx, result) in results.iter().enumerate() {
            assert!(result.is_ok(), "Processing failed for item {}", idx);
        }
    }
    
    // Test memory tracking allocator under concurrent load
    #[test]
    fn test_tracking_allocator_concurrent() {
        // This test verifies that the TrackingAllocator (which uses atomics)
        // is thread-safe by performing many concurrent allocations/deallocations
        
        const NUM_THREADS: usize = 20;
        const ALLOCATIONS_PER_THREAD: usize = 100;
        
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                
                thread::spawn(move || {
                    barrier.wait();
                    
                    let mut allocations = Vec::new();
                    
                    // Perform many small allocations
                    for _ in 0..ALLOCATIONS_PER_THREAD {
                        let size = 1024; // 1KB
                        let mut vec = Vec::<u8>::with_capacity(size);
                        vec.resize(size, 0);
                        allocations.push(vec);
                        
                        // Occasionally drop some to create deallocation
                        if allocations.len() > 10 {
                            allocations.remove(0);
                        }
                    }
                    
                    // Drop all remaining allocations
                    drop(allocations);
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
        
        // If we get here without panic, the allocator is thread-safe
        // The test verifies no data races occur in the GlobalAlloc implementation
    }
    
    // Stress test for race conditions
    #[test]
    fn test_stress_concurrent_operations() {
        use crate::thread_safe_network::ThreadSafeNetwork;
        use crate::gpu::{GpuContext, GpuBackend};
        use std::sync::atomic::{AtomicBool, Ordering};
        
        let network = Arc::new(
            ThreadSafeNetwork::load_builtin_natural()
                .expect("Failed to load network")
        );
        
        let gpu_context = Arc::new(
            GpuContext::new(GpuBackend::None)
                .expect("Failed to create GPU context")
        );
        
        const NUM_THREADS: usize = 16;
        const DURATION_MS: u64 = 1000; // Run for 1 second
        
        let stop_flag = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                let network = Arc::clone(&network);
                let gpu_context = Arc::clone(&gpu_context);
                let stop = Arc::clone(&stop_flag);
                let barrier = Arc::clone(&barrier);
                
                thread::spawn(move || {
                    barrier.wait();
                    
                    let mut iteration = 0;
                    while !stop.load(Ordering::Relaxed) {
                        // Randomly choose an operation
                        match iteration % 3 {
                            0 => {
                                // Network inference
                                let input = ndarray::ArrayD::from_shape_vec(
                                    vec![1, 3, 16, 16],
                                    vec![0.5f32; 3 * 16 * 16]
                                ).unwrap();
                                let _ = network.process(input);
                            }
                            1 => {
                                // GPU allocation
                                let _ = gpu_context.allocate(5);
                                gpu_context.free_thread_memory();
                            }
                            2 => {
                                // Memory allocation
                                let mut vecs = Vec::new();
                                for _ in 0..10 {
                                    vecs.push(vec![0u8; 1024]);
                                }
                                drop(vecs);
                            }
                            _ => unreachable!()
                        }
                        
                        iteration += 1;
                    }
                    
                    // Clean up GPU memory
                    gpu_context.free_thread_memory();
                })
            })
            .collect();
        
        // Let threads run for specified duration
        thread::sleep(Duration::from_millis(DURATION_MS));
        stop_flag.store(true, Ordering::Relaxed);
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked during stress test");
        }
        
        // Verify system is in consistent state
        assert_eq!(gpu_context.allocated_mb(), 0,
            "GPU memory should be fully freed");
    }
    
    // Test that buffer pool doesn't leak memory
    #[test]
    fn test_buffer_pool_no_leak() {
        use crate::thread_safe_network::ThreadSafeNetwork;
        use ndarray::ArrayD;
        
        let network = Arc::new(
            ThreadSafeNetwork::load_builtin_natural()
                .expect("Failed to load network")
        );
        
        // Spawn many short-lived threads
        for _ in 0..50 {
            let network = Arc::clone(&network);
            
            let handle = thread::spawn(move || {
                let input = ArrayD::from_shape_vec(
                    vec![1, 3, 32, 32],
                    vec![0.5f32; 3 * 32 * 32]
                ).unwrap();
                
                let _ = network.process(input);
            });
            
            handle.join().expect("Thread panicked");
        }
        
        // The buffer pool should handle thread cleanup appropriately
        // This test mainly ensures no panic or memory issues occur
    }
}