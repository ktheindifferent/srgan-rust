use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use std::sync::Arc;
use std::thread;
use ndarray::ArrayD;

#[test]
fn test_basic_thread_safety() {
    // Create network
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
    
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
            assert!(result.is_ok());
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("All threads completed successfully!");
}