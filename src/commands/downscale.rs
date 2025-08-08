use crate::error::{Result, SrganError};
use crate::validation;
use clap::ArgMatches;
use std::fs::File;
use std::io::{stdout, Write};

pub fn downscale(app_m: &ArgMatches) -> Result<()> {
	let factor = parse_factor(app_m)?;
	let srgb = parse_colourspace(app_m);

	let input_path = app_m
		.value_of("INPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No input file given".to_string()))?;
	let output_path = app_m
		.value_of("OUTPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No output file given".to_string()))?;
	
	// Validate paths and factor
	let input_path_buf = validation::validate_input_file(input_path)?;
	validation::validate_image_extension(&input_path_buf)?;
	let output_path_buf = validation::validate_output_path(output_path)?;
	validation::validate_factor(factor)?;

	let mut input_file = File::open(&input_path_buf)?;
	let input = crate::read(&mut input_file)?;
	let output = crate::downscale(input, factor, srgb)?;

	print!(" Writing file...");
	stdout().flush().ok();

	let mut output_file = File::create(&output_path_buf)?;
	crate::save(output, &mut output_file)?;

	println!(" Done");
	Ok(())
}

fn parse_factor(app_m: &ArgMatches) -> Result<usize> {
	app_m
		.value_of("FACTOR")
		.ok_or_else(|| SrganError::InvalidParameter("Factor argument is required".to_string()))
		.and_then(|s| {
			s.parse::<usize>()
				.map_err(|_| SrganError::InvalidParameter("Factor must be an integer".to_string()))
		})
}

fn parse_colourspace(app_m: &ArgMatches) -> bool {
	match app_m.value_of("COLOURSPACE") {
		Some("RGB") => false,
		Some("sRGB") | None => true,
		_ => true,
	}
}
