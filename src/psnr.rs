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