use std::cmp;

use ndarray::{ArrayViewD, Axis, Zip};

/// Takes two tensors of shape [H, W, 3] and
/// returns the err, y_err and pixel count of a pair of images.
///
/// If a 4th dimension is present a subview at index 0 will be used.
pub fn psnr_calculation(image1: ArrayViewD<f32>, image2: ArrayViewD<f32>) -> (f32, f32, f32) {
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

	let mut rgb_error = 0.0;
	let mut luma_error = 0.0;
	let mut pixel_count = 0.0f32;

	Zip::from(image1.genrows())
		.and(image2.genrows())
		.apply(|output_row, input_row| {
			let r_diff = clamp_pixel(output_row[0]) - clamp_pixel(input_row[0]);
			let g_diff = clamp_pixel(output_row[1]) - clamp_pixel(input_row[1]);
			let b_diff = clamp_pixel(output_row[2]) - clamp_pixel(input_row[2]);
			
			// BT.601 luma coefficients
			let luma_diff = r_diff * 0.299 + g_diff * 0.587 + b_diff * 0.114;
			
			luma_error += luma_diff * luma_diff;
			rgb_error += (r_diff * r_diff + g_diff * g_diff + b_diff * b_diff) / 3.0;
			pixel_count += 1.0;
		});

	(rgb_error, luma_error, pixel_count)
}

fn clamp_pixel(value: f32) -> f32 {
	value.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
	use super::*;
	use ndarray::{ArrayD, IxDyn};

	fn to_psnr_db(sum_mse: f32, pix: f32) -> f32 {
		-10.0 * (sum_mse / pix).log10()
	}

	#[test]
	fn test_psnr_identical_images() {
		let img = ArrayD::<f32>::from_elem(vec![1, 8, 8, 3], 0.5);
		let (err, y_err, pix) = psnr_calculation(img.view(), img.view());
		assert_eq!(err, 0.0, "MSE should be 0 for identical images");
		assert_eq!(y_err, 0.0, "Luma MSE should be 0 for identical images");
		assert_eq!(pix, 64.0, "Pixel count should be 8*8=64");
	}

	#[test]
	fn test_psnr_different_images() {
		let img1 = ArrayD::<f32>::zeros(vec![8, 8, 3]);
		let img2 = ArrayD::<f32>::from_elem(vec![8, 8, 3], 1.0);
		let (err, y_err, pix) = psnr_calculation(img1.view(), img2.view());
		assert!(err > 0.0, "MSE should be positive for different images");
		assert!(y_err > 0.0, "Luma MSE should be positive for different images");
		assert_eq!(pix, 64.0);
	}

	#[test]
	fn test_psnr_known_value() {
		// All pixels differ by 1/255 ≈ 0.00392 (1 LSB in 8-bit).
		// Per-channel MSE ≈ (1/255)^2 ≈ 1.537e-5
		// PSNR = 10 * log10(1 / MSE) ≈ 48.13 dB
		let delta = 1.0f32 / 255.0;
		let img1 = ArrayD::<f32>::zeros(vec![4, 4, 3]);
		let img2 = ArrayD::<f32>::from_elem(vec![4, 4, 3], delta);
		let (err, _y_err, pix) = psnr_calculation(img1.view(), img2.view());
		let psnr = to_psnr_db(err, pix);
		assert!(
			(psnr - 48.13).abs() < 0.5,
			"Expected PSNR ~48.13 dB for 1-LSB difference, got {:.2}",
			psnr
		);
	}

	#[test]
	fn test_psnr_pixel_count_mismatched_sizes() {
		let img1 = ArrayD::<f32>::zeros(vec![4, 4, 3]);
		let img2 = ArrayD::<f32>::zeros(vec![8, 8, 3]);
		let (_err, _y_err, pix) = psnr_calculation(img1.view(), img2.view());
		assert_eq!(pix, 16.0, "Pixel count should be min(4,8)*min(4,8)=16");
	}

	#[test]
	fn test_psnr_luma_weights() {
		// Pure-red image vs black: luma_mse should equal (0.299)^2, rgb_mse = 1/3.
		let mut data = vec![0.0f32; 4 * 4 * 3];
		for i in 0..(4 * 4) {
			data[i * 3] = 1.0; // R channel
		}
		let img1 = ArrayD::from_shape_vec(IxDyn(&[4, 4, 3]), data).unwrap();
		let img2 = ArrayD::<f32>::zeros(vec![4, 4, 3]);
		let (err, y_err, pix) = psnr_calculation(img1.view(), img2.view());

		let expected_rgb_mse = 1.0f32 / 3.0;
		let expected_luma_mse = 0.299f32 * 0.299;
		assert!((err / pix - expected_rgb_mse).abs() < 1e-4,
			"RGB MSE expected {:.4}, got {:.4}", expected_rgb_mse, err / pix);
		assert!((y_err / pix - expected_luma_mse).abs() < 1e-4,
			"Luma MSE expected {:.4}, got {:.4}", expected_luma_mse, y_err / pix);
	}
}