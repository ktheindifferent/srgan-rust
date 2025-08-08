use crate::constants::psnr as psnr_constants;
use crate::error::Result;
use alumina::data::image_folder::image_to_data;
use clap::ArgMatches;
use std::path::Path;

pub fn psnr(app_m: &ArgMatches) -> Result<()> {
	let image1_path = app_m
		.value_of("IMAGE1")
		.ok_or_else(|| crate::error::SrganError::InvalidParameter("No IMAGE1 file given".to_string()))?;
	let image2_path = app_m
		.value_of("IMAGE2")
		.ok_or_else(|| crate::error::SrganError::InvalidParameter("No IMAGE2 file given".to_string()))?;

	let image1 = image::open(Path::new(image1_path))?;
	let image2 = image::open(Path::new(image2_path))?;

	let image1_data = image_to_data(&image1);
	let image2_data = image_to_data(&image2);

	if image1_data.shape() != image2_data.shape() {
		println!("Image shapes will be cropped to the top left areas which overlap");
	}

	let (err, y_err, pix) = crate::psnr::psnr_calculation(image1_data.view(), image2_data.view());

	println!(
		"sRGB PSNR: {}\tLuma PSNR:{}",
		psnr_constants::LOG10_MULTIPLIER * (err / pix).log10(),
		psnr_constants::LOG10_MULTIPLIER * (y_err / pix).log10()
	);

	Ok(())
}
