use std::cmp;

use ndarray::{ArrayViewD, Axis, Zip};

/// Computes the Structural Similarity Index (SSIM) between two image tensors.
///
/// Takes two tensors of shape [H, W, 3] or [1, H, W, 3].
/// Returns a value in [-1, 1] where 1 indicates perfect similarity.
///
/// Uses luma (BT.601) for the comparison and global image statistics.
pub fn ssim_calculation(image1: ArrayViewD<f32>, image2: ArrayViewD<f32>) -> f32 {
	let image1 = if image1.ndim() == 4 {
		image1.subview(Axis(0), 0)
	} else {
		image1.view()
	};

	let image2 = if image2.ndim() == 4 {
		image2.subview(Axis(0), 0)
	} else {
		image2.view()
	};

	let min_height = cmp::min(image1.shape()[0], image2.shape()[0]);
	let min_width = cmp::min(image1.shape()[1], image2.shape()[1]);

	let image1 = image1.slice(s![0..min_height, 0..min_width, 0..3]);
	let image2 = image2.slice(s![0..min_height, 0..min_width, 0..3]);

	let mut sum1 = 0.0f32;
	let mut sum2 = 0.0f32;
	let mut pixel_count = 0.0f32;

	// Pass 1: compute luma means
	Zip::from(image1.genrows())
		.and(image2.genrows())
		.apply(|row1, row2| {
			let luma1 = clamp_pixel(row1[0]) * 0.299
				+ clamp_pixel(row1[1]) * 0.587
				+ clamp_pixel(row1[2]) * 0.114;
			let luma2 = clamp_pixel(row2[0]) * 0.299
				+ clamp_pixel(row2[1]) * 0.587
				+ clamp_pixel(row2[2]) * 0.114;
			sum1 += luma1;
			sum2 += luma2;
			pixel_count += 1.0;
		});

	if pixel_count == 0.0 {
		return 0.0;
	}

	let mean1 = sum1 / pixel_count;
	let mean2 = sum2 / pixel_count;

	let mut var1 = 0.0f32;
	let mut var2 = 0.0f32;
	let mut covar = 0.0f32;

	// Pass 2: compute variance and covariance
	Zip::from(image1.genrows())
		.and(image2.genrows())
		.apply(|row1, row2| {
			let luma1 = clamp_pixel(row1[0]) * 0.299
				+ clamp_pixel(row1[1]) * 0.587
				+ clamp_pixel(row1[2]) * 0.114;
			let luma2 = clamp_pixel(row2[0]) * 0.299
				+ clamp_pixel(row2[1]) * 0.587
				+ clamp_pixel(row2[2]) * 0.114;
			let d1 = luma1 - mean1;
			let d2 = luma2 - mean2;
			var1 += d1 * d1;
			var2 += d2 * d2;
			covar += d1 * d2;
		});

	var1 /= pixel_count;
	var2 /= pixel_count;
	covar /= pixel_count;

	// Stability constants from the SSIM paper (L = 1.0 for normalized float images)
	let c1 = 0.0001f32; // (0.01 * L)^2
	let c2 = 0.0009f32; // (0.03 * L)^2

	(2.0 * mean1 * mean2 + c1) * (2.0 * covar + c2)
		/ ((mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2))
}

fn clamp_pixel(value: f32) -> f32 {
	value.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
	use super::*;
	use ndarray::ArrayD;

	#[test]
	fn test_ssim_identical_images() {
		let img = ArrayD::<f32>::from_elem(vec![1, 8, 8, 3], 0.5);
		let ssim = ssim_calculation(img.view(), img.view());
		assert!((ssim - 1.0).abs() < 1e-4, "SSIM of identical images should be ~1.0, got {}", ssim);
	}

	#[test]
	fn test_ssim_different_images() {
		let img1 = ArrayD::<f32>::zeros(vec![1, 8, 8, 3]);
		let img2 = ArrayD::<f32>::from_elem(vec![1, 8, 8, 3], 1.0);
		let ssim = ssim_calculation(img1.view(), img2.view());
		assert!(ssim < 0.5, "SSIM of very different images should be low, got {}", ssim);
	}

	#[test]
	fn test_ssim_range() {
		let img1 = ArrayD::<f32>::zeros(vec![1, 4, 4, 3]);
		let img2 = ArrayD::<f32>::from_elem(vec![1, 4, 4, 3], 0.5);
		let ssim = ssim_calculation(img1.view(), img2.view());
		assert!(ssim >= -1.0 && ssim <= 1.0, "SSIM should be in [-1, 1], got {}", ssim);
	}
}
