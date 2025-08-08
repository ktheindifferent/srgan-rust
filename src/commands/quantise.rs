use crate::error::{Result, SrganError};
use clap::ArgMatches;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub fn quantise(app_m: &ArgMatches) -> Result<()> {
	let input_file_path = app_m
		.value_of("INPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No input file given".to_string()))?;
	let output_file_path = app_m
		.value_of("OUTPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No output file given".to_string()))?;

	let mut input_file = File::open(Path::new(input_file_path))?;
	let mut input_data = Vec::new();
	input_file.read_to_end(&mut input_data)?;

	let input_network = crate::network_from_bytes(&input_data)?;

	let mut output_file = File::create(Path::new(output_file_path))?;
	let output_data = crate::network_to_bytes(input_network, true)?;
	output_file.write_all(&output_data)?;

	Ok(())
}
