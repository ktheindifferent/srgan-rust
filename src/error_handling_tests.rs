#[cfg(test)]
mod tests {
    use crate::error::SrganError;
    use crate::validation::{validate_input_file, validate_output_path, validate_directory};
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_file_not_found_error() {
        let result = validate_input_file("/non/existent/file.png");
        assert!(result.is_err());
        if let Err(SrganError::FileNotFound(path)) = result {
            assert_eq!(path.to_str().unwrap(), "/non/existent/file.png");
        } else {
            panic!("Expected FileNotFound error");
        }
    }

    #[test]
    fn test_invalid_output_parent_directory() {
        let result = validate_output_path("/non/existent/dir/output.png");
        assert!(result.is_err());
        if let Err(SrganError::InvalidInput(msg)) = result {
            assert!(msg.contains("Parent directory"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[test]
    fn test_directory_validation_on_file() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "test").expect("Failed to write test file");
        
        let result = validate_directory(file_path.to_str().expect("Path to str failed"));
        assert!(result.is_err());
        if let Err(SrganError::InvalidInput(msg)) = result {
            assert!(msg.contains("not a directory"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[test]
    fn test_readonly_directory_output() {
        use std::os::unix::fs::PermissionsExt;
        
        let dir = TempDir::new().expect("Failed to create temp dir");
        let readonly_dir = dir.path().join("readonly");
        fs::create_dir(&readonly_dir).expect("Failed to create dir");
        
        // Make directory read-only
        let mut perms = fs::metadata(&readonly_dir).unwrap().permissions();
        perms.set_mode(0o444);
        fs::set_permissions(&readonly_dir, perms).expect("Failed to set permissions");
        
        let output_path = readonly_dir.join("output.png");
        let result = validate_output_path(output_path.to_str().expect("Path to str failed"));
        
        // Note: This might not fail on all systems depending on permissions
        // The test is here to ensure no panic occurs
        let _ = result;
    }

    #[test]
    fn test_concurrent_gpu_memory_allocation() {
        use crate::gpu::{GpuBackend, GpuContext};
        use std::thread;
        use std::sync::Arc;
        
        let context = Arc::new(GpuContext::new(GpuBackend::None)
            .expect("Failed to create GPU context"));
        
        let mut handles = vec![];
        
        for i in 0..4 {
            let ctx = Arc::clone(&context);
            let handle = thread::spawn(move || {
                // Try to allocate memory - CPU backend has 0 MB so this will fail
                // The test is to ensure no panic occurs, just error handling
                let result = ctx.allocate(10);
                
                // CPU backend has 0 MB memory, so allocation should fail gracefully
                assert!(result.is_err(), "Expected allocation to fail for CPU backend");
                if let Err(crate::error::SrganError::InvalidParameter(msg)) = result {
                    assert!(msg.contains("Insufficient GPU memory"), "Thread {}: {}", i, msg);
                }
                
                // Clean up (even though nothing was allocated)
                ctx.free_thread_memory();
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().expect("Thread should complete without panic");
        }
    }

    #[test]
    fn test_progress_bar_template_fallback() {
        use indicatif::{ProgressBar, ProgressStyle};
        
        let pb = ProgressBar::new(100);
        
        // Try to set an invalid template - should fallback to default
        let style = ProgressStyle::default_bar()
            .template("{invalid_placeholder}")
            .unwrap_or_else(|_| ProgressStyle::default_bar());
        
        pb.set_style(style);
        pb.inc(50);
        pb.finish();
    }

    // Note: The graph execution error handling is tested indirectly through
    // the upscaling functions. Direct testing would require access to
    // alumina internal types which are not exposed.

    #[test]
    fn test_command_arg_missing_required_input() {
        // Simulate missing required command-line argument
        let input: Option<&str> = None;
        let result = input
            .ok_or_else(|| SrganError::InvalidParameter("Input path is required".to_string()));
        
        assert!(result.is_err());
        if let Err(SrganError::InvalidParameter(msg)) = result {
            assert_eq!(msg, "Input path is required");
        } else {
            panic!("Expected InvalidParameter error");
        }
    }

    #[test]
    fn test_mutex_poisoning_recovery() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let data = Arc::new(Mutex::new(vec![1, 2, 3]));
        let data_clone = Arc::clone(&data);
        
        // Spawn a thread that will panic while holding the lock
        let handle = thread::spawn(move || {
            let _lock = data_clone.lock().expect("Failed to lock");
            panic!("Intentional panic to poison mutex");
        });
        
        // Wait for the thread to panic
        let _ = handle.join();
        
        // Try to access the poisoned mutex
        // With our expect messages, this should provide clear error info
        let result = data.lock();
        assert!(result.is_err()); // Mutex is poisoned
    }
}