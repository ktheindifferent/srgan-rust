use srgan_rust::parallel::ThreadSafeNetwork;
use srgan_rust::UpscalingNetwork;
use image::{DynamicImage, ImageBuffer, Rgb};

#[test]
fn test_thread_safe_network_creation() {
    // Create a simple bilinear network for testing
    let network = UpscalingNetwork::from_label("bilinear", Some(2)).unwrap();
    let thread_safe = ThreadSafeNetwork::new(network);
    
    // Verify we can get a network clone
    let _cloned = thread_safe.get_network();
}

#[test]
fn test_parallel_processing() {
    // Create test network
    let network = UpscalingNetwork::from_label("bilinear", Some(2)).unwrap();
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
    for result in results {
        assert!(result.is_ok());
    }
}

#[test]
fn test_parallel_vs_sequential_consistency() {
    use alumina::data::image_folder::image_to_data;
    
    // Create test network
    let network = UpscalingNetwork::from_label("bilinear", Some(2)).unwrap();
    
    // Create a test image
    let img = ImageBuffer::from_fn(16, 16, |x, y| {
        let val = ((x * y) % 256) as u8;
        Rgb([val, val, val])
    });
    let test_image = DynamicImage::ImageRgb8(img);
    
    // Process sequentially
    let tensor = image_to_data(&test_image);
    let shape = tensor.shape().to_vec();
    let input = tensor.into_shape(ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]])).unwrap();
    let sequential_result = srgan_rust::upscale(input.clone(), &network).unwrap();
    
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
    let parallel_result = parallel_results[0].as_ref().unwrap();
    
    // Compare shapes
    assert_eq!(sequential_result.shape(), parallel_result.shape());
}