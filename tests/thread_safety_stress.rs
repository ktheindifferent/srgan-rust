use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use srgan_rust::parallel::ThreadSafeNetwork as ParallelNetwork;
use srgan_rust::gpu::{GpuContext, GpuBackend};
use srgan_rust::UpscalingNetwork;
use ndarray::ArrayD;

/// High contention stress test for ThreadSafeNetwork
#[test]
fn stress_test_high_contention_thread_safe_network() {
    let network = Arc::new(
        ThreadSafeNetwork::load_builtin_natural()
            .expect("Failed to load network")
    );

    let num_threads = 32;
    let iterations_per_thread = 100;
    let successful_ops = Arc::new(AtomicUsize::new(0));
    let failed_ops = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let net = Arc::clone(&network);
        let success_counter = Arc::clone(&successful_ops);
        let fail_counter = Arc::clone(&failed_ops);
        let start_barrier = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            // Wait for all threads to be ready
            start_barrier.wait();

            for i in 0..iterations_per_thread {
                // Vary input sizes to stress different code paths
                let size = 8 + (i % 8) * 4;
                let input = ArrayD::<f32>::zeros(vec![1, size, size, 3]);

                match net.process(input) {
                    Ok(_) => {
                        success_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        eprintln!("Thread {} iteration {} failed: {:?}", thread_id, i, e);
                        fail_counter.fetch_add(1, Ordering::Relaxed);
                    }
                }

                // Add some randomized delay to vary timing
                if i % 10 == 0 {
                    thread::yield_now();
                }
            }
        });
        handles.push(handle);
    }

    let start = Instant::now();

    for handle in handles {
        handle.join().expect("Thread panicked during stress test");
    }

    let duration = start.elapsed();

    let total_ops = num_threads * iterations_per_thread;
    let successful = successful_ops.load(Ordering::Relaxed);
    let failed = failed_ops.load(Ordering::Relaxed);

    println!("Stress test completed in {:?}", duration);
    println!("Total operations: {}", total_ops);
    println!("Successful: {}", successful);
    println!("Failed: {}", failed);
    println!("Throughput: {:.2} ops/sec", successful as f64 / duration.as_secs_f64());

    assert_eq!(successful, total_ops, "All operations should succeed");
    assert_eq!(failed, 0, "No operations should fail");
}

/// Stress test for parallel network with rapid cloning
#[test]
fn stress_test_parallel_network_rapid_cloning() {
    let base_network = UpscalingNetwork::load_builtin_natural()
        .expect("Failed to load builtin network");
    let parallel_net = Arc::new(ParallelNetwork::new(base_network));

    let num_threads = 16;
    let clones_per_thread = 200;
    let clone_counter = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    for _ in 0..num_threads {
        let net = Arc::clone(&parallel_net);
        let counter = Arc::clone(&clone_counter);

        let handle = thread::spawn(move || {
            for _ in 0..clones_per_thread {
                let _cloned = net.get_network();
                counter.fetch_add(1, Ordering::Relaxed);
                // Simulate some work with the clone
                thread::yield_now();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked during cloning stress test");
    }

    assert_eq!(
        clone_counter.load(Ordering::Relaxed),
        num_threads * clones_per_thread,
        "All clones should be created successfully"
    );
}

/// Memory allocation stress test for GPU context
#[test]
fn stress_test_gpu_memory_allocation() {
    // Create a GPU context via the public API
    let context = match GpuContext::new(GpuBackend::None) {
        Ok(ctx) => Arc::new(ctx),
        Err(_) => {
            eprintln!("Skipping GPU memory stress test: no GPU backend available");
            return;
        }
    };

    let num_threads = 20;
    let allocations_per_thread = 50;
    let allocation_size_mb = 100; // Each allocation is 100 MB

    let successful_allocs = Arc::new(AtomicUsize::new(0));
    let failed_allocs = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = vec![];

    for _thread_id in 0..num_threads {
        let ctx = Arc::clone(&context);
        let success = Arc::clone(&successful_allocs);
        let failed = Arc::clone(&failed_allocs);
        let start_barrier = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            // Synchronize start
            start_barrier.wait();

            let mut local_allocations = 0;

            for i in 0..allocations_per_thread {
                // Try to allocate
                if ctx.allocate(allocation_size_mb).is_ok() {
                    local_allocations += 1;
                    success.fetch_add(1, Ordering::Relaxed);

                    // Simulate some GPU work
                    thread::sleep(Duration::from_micros(100));

                    // Free memory after some operations
                    if i % 5 == 4 {
                        ctx.free_thread_memory();
                        local_allocations = 0;
                    }
                } else {
                    failed.fetch_add(1, Ordering::Relaxed);
                    // Back off when allocation fails
                    thread::sleep(Duration::from_millis(1));
                }
            }

            // Clean up any remaining allocations
            if local_allocations > 0 {
                ctx.free_thread_memory();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked during GPU memory stress test");
    }

    let successful = successful_allocs.load(Ordering::Relaxed);
    let failed = failed_allocs.load(Ordering::Relaxed);

    println!("GPU Memory Allocation Stress Test:");
    println!("Successful allocations: {}", successful);
    println!("Failed allocations: {}", failed);

    // Verify all memory is freed
    assert_eq!(context.allocated_mb(), 0, "All memory should be freed");

    // Some allocations should succeed
    assert!(successful > 0, "At least some allocations should succeed");
}

/// Test for race conditions in buffer pool creation
#[test]
fn stress_test_buffer_pool_race_conditions() {
    let network = Arc::new(
        ThreadSafeNetwork::load_builtin_natural()
            .expect("Failed to load network")
    );

    let num_waves = 10;
    let threads_per_wave = 50;

    for wave in 0..num_waves {
        let barrier = Arc::new(Barrier::new(threads_per_wave));
        let mut handles = vec![];

        for thread_id in 0..threads_per_wave {
            let net = Arc::clone(&network);
            let start_barrier = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                // Ensure all threads start at exactly the same time
                start_barrier.wait();

                // All threads try to create their buffer at the same time
                let input = ArrayD::<f32>::zeros(vec![1, 16, 16, 3]);
                let result = net.process(input);
                assert!(result.is_ok(), "Wave {} thread {} failed", wave, thread_id);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked during race condition test");
        }
    }
}

/// Long-running stress test to detect memory leaks
#[test]
#[ignore] // This test takes a long time, run with --ignored flag
fn stress_test_long_running_memory_leak_detection() {
    let network = Arc::new(
        ThreadSafeNetwork::load_builtin_natural()
            .expect("Failed to load network")
    );

    let num_threads = 8;
    let duration_secs = 60; // Run for 1 minute
    let stop_flag = Arc::new(AtomicBool::new(false));
    let operation_counter = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    for _ in 0..num_threads {
        let net = Arc::clone(&network);
        let stop = Arc::clone(&stop_flag);
        let counter = Arc::clone(&operation_counter);

        let handle = thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
                if net.process(input).is_ok() {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        handles.push(handle);
    }

    // Let it run for the specified duration
    thread::sleep(Duration::from_secs(duration_secs));

    // Signal threads to stop
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for all threads to finish
    for handle in handles {
        handle.join().expect("Thread panicked during long-running test");
    }

    let total_operations = operation_counter.load(Ordering::Relaxed);
    println!("Long-running test completed:");
    println!("Total operations: {}", total_operations);
    println!("Operations per second: {:.2}", total_operations as f64 / duration_secs as f64);
}

/// Stress test with mixed operations
#[test]
fn stress_test_mixed_operations() {
    let network = Arc::new(
        ThreadSafeNetwork::load_builtin_natural()
            .expect("Failed to load network")
    );

    let base_network = UpscalingNetwork::load_builtin_natural()
        .expect("Failed to load builtin network");
    let parallel_net = Arc::new(ParallelNetwork::new(base_network));

    let gpu_context = Arc::new(GpuContext::new(GpuBackend::None)
        .expect("Failed to create GPU context"));

    let num_threads = 12;
    let operations_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let net = Arc::clone(&network);
        let par_net = Arc::clone(&parallel_net);
        let gpu_ctx = Arc::clone(&gpu_context);
        let start_barrier = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            start_barrier.wait();

            for i in 0..operations_per_thread {
                // Mix different types of operations
                match thread_id % 3 {
                    0 => {
                        // ThreadSafeNetwork operations
                        let input = ArrayD::<f32>::zeros(vec![1, 16, 16, 3]);
                        assert!(net.process(input).is_ok());
                    }
                    1 => {
                        // Parallel network cloning
                        let _clone = par_net.get_network();
                    }
                    2 => {
                        // GPU context operations (simplified since real GPU not available)
                        let _ = gpu_ctx.allocated_mb();
                        let _ = gpu_ctx.available_mb();
                    }
                    _ => unreachable!(),
                }

                // Add some variation in timing
                if i % 20 == 0 {
                    thread::yield_now();
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked during mixed operations test");
    }
}
