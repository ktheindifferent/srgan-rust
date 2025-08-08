use crate::constants::network;
use crate::error::{Result, SrganError};
use crate::logging;
use crate::validation;
use crate::UpscalingNetwork;
use clap::ArgMatches;
use log::info;
use std::fs::File;
use std::io::Read;

pub fn upscale(app_m: &ArgMatches) -> Result<()> {
	let factor = parse_factor(app_m);
	let network = load_network(app_m, factor)?;

	info!("Upsampling using {}...", network);
	let spinner = logging::create_spinner("Processing image...");

	let input_path = app_m
		.value_of("INPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No input file given".to_string()))?;
	let output_path = app_m
		.value_of("OUTPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No output file given".to_string()))?;
	
	// Validate input and output paths
	let input_path_buf = validation::validate_input_file(input_path)?;
	validation::validate_image_extension(&input_path_buf)?;
	let output_path_buf = validation::validate_output_path(output_path)?;
	validation::validate_factor(factor)?;

	let mut input_file = File::open(&input_path_buf)?;
	let input = crate::read(&mut input_file)?;
	spinner.set_message("Running neural network...");
	let output = crate::upscale(input, &network)?;

	spinner.set_message("Writing output file...");
	let mut output_file = File::create(&output_path_buf)?;
	crate::save(output, &mut output_file)?;

	spinner.finish_with_message("✓ Upscaling complete");
	info!("Output saved to: {}", output_path_buf.display());
	Ok(())
}

fn parse_factor(app_m: &ArgMatches) -> usize {
	app_m
		.value_of("BILINEAR_FACTOR")
		.and_then(|s| s.parse::<usize>().ok())
		.unwrap_or(network::DEFAULT_FACTOR)
}

fn load_network(app_m: &ArgMatches, factor: usize) -> Result<UpscalingNetwork> {
	if let Some(file_str) = app_m.value_of("CUSTOM") {
		let param_path = validation::validate_input_file(file_str)?;
		let mut param_file = File::open(&param_path)?;
		let mut data = Vec::new();
		param_file.read_to_end(&mut data)?;
		let network_desc = crate::network_from_bytes(&data)?;
		UpscalingNetwork::new(network_desc, "custom trained neural net").map_err(|e| SrganError::Network(e))
	} else {
		let param_type = app_m.value_of("PARAMETERS").unwrap_or("natural");
		UpscalingNetwork::from_label(param_type, Some(factor)).map_err(|e| SrganError::Network(e))
	}
}
