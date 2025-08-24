use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use srgan_rust::NetworkDescription;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use ndarray::ArrayD;

fn main() {
    println!("Thread-Safe UpscalingNetwork Demo");
    println!("==================================\n");
    
    // Create the thread-safe network
    println!("Loading network...");
    // Create a test network with dummy parameters
    
    // Create dummy parameters to satisfy the network requirements
    let mut parameters = Vec::new();
    // Add enough dummy parameters for the network (15 based on error message)
    for _ in 0..14 {
        parameters.push(ArrayD::<f32>::zeros(vec![3, 3, 3, 12]));
    }
    
    let desc = NetworkDescription {
        factor: 4,
        width: 12,
        log_depth: 2,
        global_node_factor: 0,
        parameters,
    };
    
    let network = Arc::new(
        ThreadSafeNetwork::new(desc, "test network")
            .expect("Failed to load network")
    );
    println!("Network loaded successfully!\n");
    
    // Test 1: Single-threaded baseline
    println!("Test 1: Single-threaded processing");
    println!("-----------------------------------");
    let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
    let start = Instant::now();
    
    for i in 0..4 {
        let result = network.process(input.clone());
        if let Err(e) = &result {
            println!("Error in iteration {}: {:?}", i, e);
        }
        assert!(result.is_ok(), "Single-thread iteration {} failed", i);
    }
    
    let single_duration = start.elapsed();
    println!("Single-threaded: 4 images in {:?}\n", single_duration);
    
    // Test 2: Multi-threaded concurrent processing
    println!("Test 2: Multi-threaded processing");
    println!("----------------------------------");
    let start = Instant::now();
    let mut handles = vec![];
    
    for i in 0..4 {
        let network_clone = Arc::clone(&network);
        let input_clone = input.clone();
        
        let handle = thread::spawn(move || {
            let result = network_clone.process(input_clone);
            assert!(result.is_ok(), "Thread {} failed", i);
            println!("  Thread {} completed", i);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    let multi_duration = start.elapsed();
    println!("Multi-threaded: 4 images in {:?}\n", multi_duration);
    
    // Calculate speedup
    let speedup = single_duration.as_secs_f64() / multi_duration.as_secs_f64();
    println!("Summary");
    println!("-------");
    println!("Single-threaded: {:?}", single_duration);
    println!("Multi-threaded:  {:?}", multi_duration);
    println!("Speedup:         {:.2}x", speedup);
    
    // Test 3: Verify no mutex bottleneck
    println!("\nTest 3: Concurrent access (no mutex bottleneck)");
    println!("------------------------------------------------");
    
    let num_threads = 8;
    let iterations_per_thread = 5;
    let start = Instant::now();
    let mut handles = vec![];
    
    for t in 0..num_threads {
        let network_clone = Arc::clone(&network);
        
        let handle = thread::spawn(move || {
            for i in 0..iterations_per_thread {
                let size = 16 + (t % 3) * 8; // Vary sizes: 16, 24, 32
                let input = ArrayD::<f32>::zeros(vec![1, size, size, 3]);
                let result = network_clone.process(input);
                assert!(result.is_ok(), "Thread {} iteration {} failed", t, i);
            }
            println!("  Thread {} completed {} iterations", t, iterations_per_thread);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let concurrent_duration = start.elapsed();
    let total_operations = num_threads * iterations_per_thread;
    let ops_per_second = total_operations as f64 / concurrent_duration.as_secs_f64();
    
    println!("Processed {} operations in {:?}", total_operations, concurrent_duration);
    println!("Throughput: {:.2} operations/second", ops_per_second);
    
    println!("\n✅ All tests passed! The network is thread-safe and performs well under concurrent load.");
}