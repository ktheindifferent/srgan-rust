use crate::error::Result;
use crate::NetworkDescription;
use ndarray::ArrayD;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct CheckpointManager {
	output_path: PathBuf,
	interval: usize,
	quantise: bool,
}

impl CheckpointManager {
	pub fn new(output_path: impl AsRef<Path>, interval: usize, quantise: bool) -> Self {
		Self {
			output_path: output_path.as_ref().to_path_buf(),
			interval,
			quantise,
		}
	}

	pub fn should_checkpoint(&self, step: usize) -> bool {
		(step + 1) % self.interval == 0
	}

	pub fn save_checkpoint(
		&self,
		params: &[ArrayD<f32>],
		factor: u32,
		width: u32,
		log_depth: u32,
		global_node_factor: u32,
	) -> Result<()> {
		let network_desc = NetworkDescription {
			factor,
			width,
			log_depth,
			global_node_factor,
			parameters: params.to_vec(),
		};

		let mut file = File::create(&self.output_path)?;
		let bytes = crate::network_to_bytes(network_desc, self.quantise)?;
		file.write_all(&bytes)?;

		Ok(())
	}
}
