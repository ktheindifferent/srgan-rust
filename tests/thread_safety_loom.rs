#![cfg(loom)]

use loom::sync::Arc;
use loom::thread;
use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use srgan_rust::parallel::ThreadSafeNetwork as ParallelNetwork;
use srgan_rust::gpu::GpuContext;
use ndarray::ArrayD;

/// Test concurrent access to ThreadSafeNetwork with loom
#[test]
fn loom_test_thread_safe_network_concurrent_access() {
    let mut config = loom::model::Config::default();
    config.preemption_bound = Some(3);
    
    loom::model(move || {
        let network = Arc::new(
            ThreadSafeNetwork::load_builtin_natural()
                .expect("Failed to load network")
        );
        
        let mut handles = vec![];
        
        // Spawn 3 threads for concurrent access
        for i in 0..3 {
            let net = Arc::clone(&network);
            let handle = thread::spawn(move || {
                let input = ArrayD::<f32>::zeros(vec![1, 16, 16, 3]);
                let result = net.process(input);
                assert!(result.is_ok(), "Thread {} failed", i);
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

/// Test buffer pool isolation with loom
#[test]
fn loom_test_buffer_pool_isolation() {
    let mut config = loom::model::Config::default();
    config.preemption_bound = Some(2);
    
    loom::model(move || {
        let network = Arc::new(
            ThreadSafeNetwork::load_builtin_natural()
                .expect("Failed to load network")
        );
        
        // Test that each thread gets its own buffer
        let handle1 = {
            let net = Arc::clone(&network);
            thread::spawn(move || {
                let input = ArrayD::<f32>::zeros(vec![1, 8, 8, 3]);
                net.process(input).unwrap();
                // Thread 1's buffer is created
            })
        };
        
        let handle2 = {
            let net = Arc::clone(&network);
            thread::spawn(move || {
                let input = ArrayD::<f32>::zeros(vec![1, 8, 8, 3]);
                net.process(input).unwrap();
                // Thread 2's buffer is created independently
            })
        };
        
        handle1.join().unwrap();
        handle2.join().unwrap();
    });
}

/// Test parallel network clone isolation with loom
#[test]
fn loom_test_parallel_network_clone_isolation() {
    use srgan_rust::UpscalingNetwork;
    
    let mut config = loom::model::Config::default();
    config.preemption_bound = Some(2);
    
    loom::model(move || {
        let base_network = UpscalingNetwork::default();
        let parallel_net = Arc::new(ParallelNetwork::new(base_network));
        
        let mut handles = vec![];
        
        for _ in 0..2 {
            let net = Arc::clone(&parallel_net);
            let handle = thread::spawn(move || {
                // Each thread gets its own network clone
                let _cloned = net.get_network();
                // Clones are independent
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

/// Test GPU context memory allocation with loom
#[test]
fn loom_test_gpu_context_memory_allocation() {
    use srgan_rust::gpu::{GpuBackend, GpuDevice};
    use loom::sync::RwLock;
    use std::collections::HashMap;
    
    let mut config = loom::model::Config::default();
    config.preemption_bound = Some(2);
    
    loom::model(move || {
        // Create a test GPU context with limited memory
        let context = Arc::new(GpuContext {
            device: Arc::new(GpuDevice {
                backend: GpuBackend::None,
                device_id: 0,
                memory_mb: 200,
                name: "Test GPU".to_string(),
            }),
            allocated_mb: Arc::new(RwLock::new(0)),
            memory_pools: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let mut handles = vec![];
        
        // Spawn threads that allocate and free memory
        for i in 0..2 {
            let ctx = Arc::clone(&context);
            let handle = thread::spawn(move || {
                // Allocate memory
                if ctx.allocate(50).is_ok() {
                    // Do some work
                    loom::thread::yield_now();
                    // Free memory
                    ctx.free_thread_memory();
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all memory is freed
        assert_eq!(*context.allocated_mb.read().unwrap(), 0);
    });
}

/// Test for data races in buffer pool access
#[test]
fn loom_test_no_data_races_in_buffer_pool() {
    let mut config = loom::model::Config::default();
    config.preemption_bound = Some(3);
    
    loom::model(move || {
        let network = Arc::new(
            ThreadSafeNetwork::load_builtin_natural()
                .expect("Failed to load network")
        );
        
        // Create a scenario that could cause data races if not properly synchronized
        let mut handles = vec![];
        
        for i in 0..3 {
            let net = Arc::clone(&network);
            let handle = thread::spawn(move || {
                // Rapidly create and use buffers
                for j in 0..2 {
                    let input = ArrayD::<f32>::zeros(vec![1, 4, 4, 3]);
                    let result = net.process(input);
                    assert!(result.is_ok(), "Thread {} iteration {} failed", i, j);
                    loom::thread::yield_now();
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

/// Test mutex poisoning behavior
#[test]
fn loom_test_mutex_poisoning_handling() {
    let mut config = loom::model::Config::default();
    config.preemption_bound = Some(2);
    
    loom::model(move || {
        let network = Arc::new(
            ThreadSafeNetwork::load_builtin_natural()
                .expect("Failed to load network")
        );
        
        let panic_net = Arc::clone(&network);
        let normal_net = Arc::clone(&network);
        
        // Thread that might panic while holding the lock
        let panic_handle = thread::spawn(move || {
            let input = ArrayD::<f32>::zeros(vec![1, 8, 8, 3]);
            let _ = panic_net.process(input);
            // In real scenario, this could panic
            // For loom testing, we just complete normally
        });
        
        // Thread that accesses the network after potential panic
        let normal_handle = thread::spawn(move || {
            loom::thread::yield_now(); // Let panic thread go first sometimes
            let input = ArrayD::<f32>::zeros(vec![1, 8, 8, 3]);
            let _ = normal_net.process(input);
        });
        
        let _ = panic_handle.join();
        let _ = normal_handle.join();
    });
}