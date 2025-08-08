use crate::error::{Result, SrganError};
use crate::NetworkDescription;
use clap::ArgMatches;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub fn set_width(app_m: &ArgMatches) -> Result<()> {
	let width = parse_width(app_m)?;

	let input_path = app_m
		.value_of("INPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No input file given".to_string()))?;
	let output_path = app_m
		.value_of("OUTPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No output file given".to_string()))?;

	let mut input_file = File::open(Path::new(input_path))?;
	let mut input_data = Vec::new();
	input_file.read_to_end(&mut input_data)?;
	let input_network = crate::network_from_bytes(&input_data)?;

	let output_network = NetworkDescription {
		factor: input_network.factor,
		width,
		log_depth: input_network.log_depth,
		global_node_factor: input_network.global_node_factor,
		parameters: input_network.parameters,
	};

	let mut output_file = File::create(Path::new(output_path))?;
	let output_data = crate::network_to_bytes(output_network, false)?;
	output_file.write_all(&output_data)?;

	Ok(())
}

fn parse_width(app_m: &ArgMatches) -> Result<u32> {
	app_m
		.value_of("WIDTH")
		.ok_or_else(|| SrganError::InvalidParameter("Width argument is required".to_string()))
		.and_then(|s| {
			s.parse::<u32>()
				.map_err(|_| SrganError::InvalidParameter("Width must be an integer".to_string()))
		})
}
