use crate::constants::network;
use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;
use clap::ArgMatches;
use std::fs::File;
use std::io::{stdout, Read, Write};
use std::path::Path;

pub fn upscale(app_m: &ArgMatches) -> Result<()> {
	let factor = parse_factor(app_m);
	let network = load_network(app_m, factor)?;

	println!("Upsampling using {}...", network);

	let input_path = app_m
		.value_of("INPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No input file given".to_string()))?;
	let output_path = app_m
		.value_of("OUTPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No output file given".to_string()))?;

	let mut input_file = File::open(Path::new(input_path))?;
	let input = crate::read(&mut input_file)?;
	let output = crate::upscale(input, &network)?;

	print!(" Writing file...");
	stdout().flush().ok();

	let mut output_file = File::create(Path::new(output_path))?;
	crate::save(output, &mut output_file)?;

	println!(" Done");
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
		let mut param_file = File::open(Path::new(file_str))?;
		let mut data = Vec::new();
		param_file.read_to_end(&mut data)?;
		let network_desc = crate::network_from_bytes(&data)?;
		UpscalingNetwork::new(network_desc, "custom trained neural net").map_err(|e| SrganError::Network(e))
	} else {
		let param_type = app_m.value_of("PARAMETERS").unwrap_or("natural");
		UpscalingNetwork::from_label(param_type, Some(factor)).map_err(|e| SrganError::Network(e))
	}
}
