use srgan_rust::parallel::ThreadSafeNetwork;
use srgan_rust::UpscalingNetwork;
use image::{DynamicImage, ImageBuffer, Rgb};

// Import test helpers for better error handling
#[path = "test_helpers.rs"]
mod test_helpers;
use test_helpers::*;

#[test]
fn test_thread_safe_network_creation() {
    // Create a simple bilinear network for testing
    let network = assert_ok(
        UpscalingNetwork::from_label("bilinear", Some(2)),
        "creating bilinear network for thread-safe test"
    );
    let thread_safe = ThreadSafeNetwork::new(network);
    
    // Verify we can get a network clone
    let _cloned = thread_safe.get_network();
}

#[test]
fn test_parallel_processing() {
    // Create test network
    let network = assert_ok(
        UpscalingNetwork::from_label("bilinear", Some(2)),
        "creating bilinear network for parallel processing test"
    );
    let thread_safe = ThreadSafeNetwork::new(network);
    
    // Create test images
    let images: Vec<DynamicImage> = (0..4)
        .map(|i| {
            let img = ImageBuffer::from_fn(32, 32, |x, y| {
                let val = ((x + y + i) % 256) as u8;
                Rgb([val, val, val])
            });
            DynamicImage::ImageRgb8(img)
        })
        .collect();
    
    // Process in parallel
    let results = thread_safe.process_batch_parallel(
        images,
        |img, net| {
            use alumina::data::image_folder::image_to_data;
            let tensor = image_to_data(&img);
            let shape = tensor.shape().to_vec();
            let input = tensor.into_shape(ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]]))
                .map_err(|_| srgan_rust::error::SrganError::ShapeError("Shape error".to_string()))?;
            srgan_rust::upscale(input, net)
                .map_err(|e| srgan_rust::error::SrganError::GraphExecution(e.to_string()))
        },
        Some(2),
    );
    
    // Verify all images were processed
    assert_eq!(results.len(), 4);
    for (idx, result) in results.iter().enumerate() {
        assert_result_ok(result.as_ref(), &format!("parallel processing result {}", idx));
    }
}

#[test]
fn test_parallel_vs_sequential_consistency() {
    use alumina::data::image_folder::image_to_data;
    
    // Create test network
    let network = assert_ok(
        UpscalingNetwork::from_label("bilinear", Some(2)),
        "creating bilinear network for consistency test"
    );
    
    // Create a test image
    let img = ImageBuffer::from_fn(16, 16, |x, y| {
        let val = ((x * y) % 256) as u8;
        Rgb([val, val, val])
    });
    let test_image = DynamicImage::ImageRgb8(img);
    
    // Process sequentially
    let tensor = image_to_data(&test_image);
    let shape = tensor.shape().to_vec();
    let input = assert_ok(
        tensor.into_shape(ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]])),
        "reshaping tensor for sequential processing"
    );
    let sequential_result = assert_ok(
        srgan_rust::upscale(input.clone(), &network),
        "sequential upscaling"
    );
    
    // Process in parallel (single image)
    let thread_safe = ThreadSafeNetwork::new(network);
    let parallel_results = thread_safe.process_batch_parallel(
        vec![test_image],
        |img, net| {
            let tensor = image_to_data(&img);
            let shape = tensor.shape().to_vec();
            let input = tensor.into_shape(ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]]))
                .map_err(|_| srgan_rust::error::SrganError::ShapeError("Shape error".to_string()))?;
            srgan_rust::upscale(input, net)
                .map_err(|e| srgan_rust::error::SrganError::GraphExecution(e.to_string()))
        },
        Some(1),
    );
    
    // Verify results are consistent
    assert_eq!(parallel_results.len(), 1);
    let parallel_result = assert_ok(
        parallel_results[0].as_ref(),
        "parallel processing single image"
    );
    
    // Compare shapes
    assert_eq!(sequential_result.shape(), parallel_result.shape());
}

#[test]
fn test_parallel_error_handling() {
    // Create test network
    let network = assert_ok(
        UpscalingNetwork::from_label("bilinear", Some(2)),
        "creating bilinear network for error handling test"
    );
    let thread_safe = ThreadSafeNetwork::new(network);
    
    // Create a mix of valid and invalid images
    let images: Vec<DynamicImage> = vec![
        // Valid image
        DynamicImage::ImageRgb8(ImageBuffer::from_fn(32, 32, |x, y| {
            Rgb([((x + y) % 256) as u8; 3])
        })),
        // Empty image (will cause error)
        DynamicImage::ImageRgb8(ImageBuffer::new(0, 0)),
        // Valid image
        DynamicImage::ImageRgb8(ImageBuffer::from_fn(16, 16, |x, y| {
            Rgb([((x * y) % 256) as u8; 3])
        })),
    ];
    
    // Process with error handling
    let results = thread_safe.process_batch_parallel(
        images,
        |img, net| {
            use alumina::data::image_folder::image_to_data;
            
            // Check for empty image
            if img.width() == 0 || img.height() == 0 {
                return Err(srgan_rust::error::SrganError::ShapeError(
                    "Empty image not supported".to_string()
                ));
            }
            
            let tensor = image_to_data(&img);
            let shape = tensor.shape().to_vec();
            let input = tensor.into_shape(ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]]))
                .map_err(|_| srgan_rust::error::SrganError::ShapeError("Shape error".to_string()))?;
            srgan_rust::upscale(input, net)
                .map_err(|e| srgan_rust::error::SrganError::GraphExecution(e.to_string()))
        },
        Some(2),
    );
    
    // Verify error handling
    assert_eq!(results.len(), 3);
    assert_result_ok(results[0].as_ref(), "first image processing");
    assert_result_err(results[1].as_ref(), "empty image processing");
    assert_result_ok(results[2].as_ref(), "third image processing");
}

#[test]
fn test_concurrent_network_cloning() {
    use std::thread;
    use std::sync::Arc;
    
    let network = assert_ok(
        UpscalingNetwork::from_label("bilinear", Some(2)),
        "creating bilinear network for cloning test"
    );
    let thread_safe = Arc::new(ThreadSafeNetwork::new(network));
    
    let mut handles = vec![];
    
    for i in 0..4 {
        let thread_safe_clone = Arc::clone(&thread_safe);
        let handle = thread::spawn(move || {
            // Each thread gets its own network clone
            let network_clone = thread_safe_clone.get_network();
            
            // Verify network is valid
            assert_eq!(network_clone.factor(), 2);
            
            println!("Thread {} successfully cloned network", i);
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for (idx, handle) in handles.into_iter().enumerate() {
        assert_thread_success(handle.join(), &format!("network cloning thread {}", idx));
    }
}