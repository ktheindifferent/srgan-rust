use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use crate::error::{SrganError, Result};
use crate::utils::error_helpers::IoErrorMapper;

pub fn read_file_bytes<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_io_err()?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).map_io_err()?;
    Ok(data)
}

pub fn read_file_string<P: AsRef<Path>>(path: P) -> Result<String> {
    fs::read_to_string(path).map_io_err()
}

pub fn write_file_bytes<P: AsRef<Path>>(path: P, data: &[u8]) -> Result<()> {
    let mut file = File::create(path).map_io_err()?;
    file.write_all(data).map_io_err()?;
    Ok(())
}

pub fn create_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
    fs::create_dir_all(path).map_io_err()
}

pub fn validate_and_read_network<P: AsRef<Path>>(param_path: P) -> Result<crate::NetworkDescription> {
    let data = read_file_bytes(param_path)?;
    crate::network_from_bytes(&data).map_err(|e| SrganError::Parse(e))
}

pub struct PathValidator;

impl PathValidator {
    pub fn validate_input_output(
        app_m: &clap::ArgMatches,
        input_key: &str,
        output_key: &str,
    ) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
        let input_path = app_m
            .value_of(input_key)
            .ok_or_else(|| SrganError::InvalidParameter(format!("No {} given", input_key)))?;
        let output_path = app_m
            .value_of(output_key)
            .ok_or_else(|| SrganError::InvalidParameter(format!("No {} given", output_key)))?;

        let input_path_buf = crate::validation::validate_input_file(input_path)?;
        crate::validation::validate_image_extension(&input_path_buf)?;
        let output_path_buf = crate::validation::validate_output_path(output_path)?;
        
        Ok((input_path_buf, output_path_buf))
    }
}