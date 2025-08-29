use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use std::sync::Arc;
use std::thread;
use ndarray::ArrayD;

// Import test helpers for better error handling
#[path = "test_helpers.rs"]
mod test_helpers;
use test_helpers::*;

#[test]
fn test_basic_thread_safety() {
    // Create network
    let network = Arc::new(assert_ok(
        ThreadSafeNetwork::load_builtin_natural(),
        "loading builtin natural network for basic thread safety test"
    ));
    
    // Test with 4 threads
    let mut handles = vec![];
    
    for i in 0..4 {
        let network_clone = Arc::clone(&network);
        let handle = thread::spawn(move || {
            println!("Thread {} starting", i);
            
            // Create small test input
            let input = ArrayD::<f32>::zeros(vec![1, 16, 16, 3]);
            
            // Process
            let result = network_clone.process(input);
            
            println!("Thread {} completed: {:?}", i, result.is_ok());
            assert_result_ok(result, &format!("thread {} processing", i));
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for (idx, handle) in handles.into_iter().enumerate() {
        assert_thread_success(handle.join(), &format!("basic safety thread {}", idx));
    }
    
    println!("All threads completed successfully!");
}