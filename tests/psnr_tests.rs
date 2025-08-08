use ndarray::{Array3, ArrayD, IxDyn};
use srgan_rust::psnr::psnr_calculation;

#[test]
fn test_psnr_identical_images() {
    let image = Array3::<f32>::ones((10, 10, 3));
    let (rgb_err, luma_err, pixel_count) = psnr_calculation(image.view().into_dyn(), image.view().into_dyn());
    
    assert_eq!(rgb_err, 0.0);
    assert_eq!(luma_err, 0.0);
    assert_eq!(pixel_count, 100.0);
}

#[test]
fn test_psnr_different_images() {
    let image1 = Array3::<f32>::ones((10, 10, 3));
    let image2 = Array3::<f32>::zeros((10, 10, 3));
    let (rgb_err, luma_err, pixel_count) = psnr_calculation(image1.view().into_dyn(), image2.view().into_dyn());
    
    assert!(rgb_err > 0.0);
    assert!(luma_err > 0.0);
    assert_eq!(pixel_count, 100.0);
}

#[test]
fn test_psnr_different_sizes() {
    let image1 = Array3::<f32>::ones((10, 10, 3));
    let image2 = Array3::<f32>::ones((8, 12, 3));
    let (rgb_err, luma_err, pixel_count) = psnr_calculation(image1.view().into_dyn(), image2.view().into_dyn());
    
    assert_eq!(rgb_err, 0.0);
    assert_eq!(luma_err, 0.0);
    assert_eq!(pixel_count, 80.0); // min(10, 8) * min(10, 12) = 8 * 10
}

#[test]
fn test_psnr_with_4d_input() {
    let shape = IxDyn(&[1, 10, 10, 3]);
    let image1 = ArrayD::<f32>::ones(shape.clone());
    let image2 = ArrayD::<f32>::ones(shape);
    let (rgb_err, luma_err, pixel_count) = psnr_calculation(image1.view(), image2.view());
    
    assert_eq!(rgb_err, 0.0);
    assert_eq!(luma_err, 0.0);
    assert_eq!(pixel_count, 100.0);
}

#[test]
fn test_psnr_rgb_error_calculation() {
    let mut image1 = Array3::<f32>::zeros((1, 1, 3));
    let mut image2 = Array3::<f32>::zeros((1, 1, 3));
    
    // Set different RGB values
    image1[[0, 0, 0]] = 0.5; // R
    image1[[0, 0, 1]] = 0.5; // G
    image1[[0, 0, 2]] = 0.5; // B
    
    image2[[0, 0, 0]] = 0.6; // R
    image2[[0, 0, 1]] = 0.6; // G
    image2[[0, 0, 2]] = 0.6; // B
    
    let (rgb_err, _, pixel_count) = psnr_calculation(image1.view().into_dyn(), image2.view().into_dyn());
    
    // RGB error should be average of squared differences
    let expected_rgb_err = 3.0 * 0.01 / 3.0; // 3 channels * (0.1)^2 / 3
    assert!((rgb_err - expected_rgb_err).abs() < 1e-6);
    assert_eq!(pixel_count, 1.0);
}

#[test]
fn test_psnr_luma_error_calculation() {
    let mut image1 = Array3::<f32>::zeros((1, 1, 3));
    let mut image2 = Array3::<f32>::zeros((1, 1, 3));
    
    // Set different RGB values
    image1[[0, 0, 0]] = 1.0; // R
    image1[[0, 0, 1]] = 1.0; // G
    image1[[0, 0, 2]] = 1.0; // B
    
    image2[[0, 0, 0]] = 0.0; // R
    image2[[0, 0, 1]] = 0.0; // G
    image2[[0, 0, 2]] = 0.0; // B
    
    let (_, luma_err, pixel_count) = psnr_calculation(image1.view().into_dyn(), image2.view().into_dyn());
    
    // Luma difference using BT.601 coefficients: 0.299 + 0.587 + 0.114 = 1.0
    let expected_luma_err = 1.0;
    assert!((luma_err - expected_luma_err).abs() < 1e-6);
    assert_eq!(pixel_count, 1.0);
}