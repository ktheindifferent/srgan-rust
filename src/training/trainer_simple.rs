use crate::config::{NetworkConfig, TrainingConfig};
use crate::constants::training as training_constants;
use crate::error::Result;
use crate::NetworkDescription;
use alumina::data::DataStream;
use alumina::graph::GraphDef;
use alumina::opt::{adam::Adam, CallbackSignal, Opt, UnboxedCallbacks};
use ndarray::ArrayD;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub fn train_network(
	graph: GraphDef,
	network_config: NetworkConfig,
	training_config: TrainingConfig,
	checkpoint_path: &str,
	training_stream: &mut dyn DataStream,
	initial_params: Option<Vec<ArrayD<f32>>>,
) -> Result<()> {
	let mut solver = Adam::new(&graph)
		.map_err(|e| crate::error::SrganError::Training(e.to_string()))?
		.rate(training_config.learning_rate)
		.beta1(training_constants::ADAM_BETA1)
		.beta2(training_constants::ADAM_BETA2)
		.bias_correct(false);

	let params = initial_params.unwrap_or_else(|| {
		graph
			.initialise_nodes(solver.parameters())
			.map_err(|e| crate::error::SrganError::Training(format!("Could not initialise parameters: {}", e)))
			.unwrap()
	});

	let checkpoint_path_owned = checkpoint_path.to_string();
	let network_config_clone = network_config.clone();
	let quantise = training_config.quantise;

	solver.add_callback(move |data| {
		if (data.step + 1) % training_constants::CHECKPOINT_INTERVAL == 0 {
			save_checkpoint(
				&checkpoint_path_owned,
				data.params,
				network_config_clone.factor as u32,
				network_config_clone.width,
				network_config_clone.log_depth,
				network_config_clone.global_node_factor as u32,
				quantise,
			)
			.map_err(|e| {
				eprintln!("Warning: Could not save checkpoint: {}", e);
				e
			})
			.ok();
		}

		println!("step {}\terr:{}\tchange:{}", data.step, data.err, data.change_norm);
		CallbackSignal::Continue
	});

	println!("Beginning Training");
	solver
		.optimise_from(training_stream, params)
		.map_err(|e| crate::error::SrganError::Training(e.to_string()))?;
	println!("Done");

	Ok(())
}

fn save_checkpoint(
	output_path: &str,
	params: &[ArrayD<f32>],
	factor: u32,
	width: u32,
	log_depth: u32,
	global_node_factor: u32,
	quantise: bool,
) -> Result<()> {
	let network_desc = NetworkDescription {
		factor,
		width,
		log_depth,
		global_node_factor,
		parameters: params.to_vec(),
	};

	let mut file = File::create(Path::new(output_path))?;
	let bytes = crate::network_to_bytes(network_desc, quantise)?;
	file.write_all(&bytes)?;

	Ok(())
}
