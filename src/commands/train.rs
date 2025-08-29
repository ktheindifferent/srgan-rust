use crate::config::{LossType, NetworkConfig, TrainingConfig, ValidationConfig};
use crate::error::{Result, SrganError};
use crate::network::training_sr_net;
use crate::training::{train_network, DataLoader};
use crate::validation;
use crate::NetworkDescription;
use clap::ArgMatches;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn train(app_m: &ArgMatches) -> Result<()> {
	println!("Training with:");

	let network_config = parse_network_config(app_m)?;
	let training_config = parse_training_config(app_m)?;
	let _validation_config = parse_validation_config(app_m);

	network_config.validate()?;
	training_config.validate()?;

	print_training_info(&network_config, &training_config);

	let initial_params = load_initial_parameters(app_m, &network_config)?;

	let (power, scale) = training_config.get_loss_params();
	let graph = training_sr_net(
		network_config.factor,
		network_config.width,
		network_config.log_depth,
		2, // Use default global_node_factor for compatibility with inference
		crate::constants::training::REGULARIZATION_EPSILON,
		power,
		scale,
		training_config.srgb_downscale,
	)
	.map_err(|e| SrganError::GraphConstruction(format!("{}", e)))?;

	let training_folder = app_m
		.value_of("TRAINING_FOLDER")
		.ok_or_else(|| SrganError::InvalidParameter("No training folder specified".into()))?;
	let param_file_path = app_m
		.value_of("PARAMETER_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No parameter file specified".into()))?;
	
	// Validate training folder and output path
	validation::validate_directory(training_folder)?;
	validation::validate_output_path(param_file_path)?;

	let mut training_stream = DataLoader::create_training_stream(
		training_folder,
		training_config.recurse,
		training_config.patch_size,
		network_config.factor,
		training_config.batch_size,
	);

	train_network(
		graph,
		network_config,
		training_config,
		param_file_path,
		&mut *training_stream,
		initial_params,
	)
}

fn parse_network_config(app_m: &ArgMatches) -> Result<NetworkConfig> {
	let mut builder = NetworkConfig::builder();

	if let Some(factor) = app_m.value_of("FACTOR") {
		builder = builder.factor(
			factor
				.parse()
				.map_err(|_| SrganError::InvalidParameter("Factor must be an integer".into()))?,
		);
	}

	if let Some(width) = app_m.value_of("WIDTH") {
		builder = builder.width(
			width
				.parse()
				.map_err(|_| SrganError::InvalidParameter("Width must be an integer".into()))?,
		);
	}

	if let Some(log_depth) = app_m.value_of("LOG_DEPTH") {
		builder = builder.log_depth(
			log_depth
				.parse()
				.map_err(|_| SrganError::InvalidParameter("Log depth must be an integer".into()))?,
		);
	}

	Ok(builder.build())
}

fn parse_training_config(app_m: &ArgMatches) -> Result<TrainingConfig> {
	let mut builder = TrainingConfig::builder();

	if let Some(lr) = app_m.value_of("LEARNING_RATE") {
		builder = builder.learning_rate(
			lr.parse()
				.map_err(|_| SrganError::InvalidParameter("Learning rate must be a number".into()))?,
		);
	}

	if let Some(patch_size) = app_m.value_of("PATCH_SIZE") {
		builder = builder.patch_size(
			patch_size
				.parse()
				.map_err(|_| SrganError::InvalidParameter("Patch size must be an integer".into()))?,
		);
	}

	if let Some(batch_size) = app_m.value_of("BATCH_SIZE") {
		builder = builder.batch_size(
			batch_size
				.parse()
				.map_err(|_| SrganError::InvalidParameter("Batch size must be an integer".into()))?,
		);
	}

	if let Some(loss) = app_m.value_of("TRAINING_LOSS") {
		builder = builder.loss_type(LossType::from_str(loss)?);
	}

	let srgb = match app_m.value_of("DOWNSCALE_COLOURSPACE") {
		Some("RGB") => false,
		_ => true,
	};
	builder = builder.srgb_downscale(srgb);

	builder = builder
		.quantise(app_m.is_present("QUANTISE"))
		.recurse(app_m.is_present("RECURSE_SUBFOLDERS"));

	Ok(builder.build())
}

fn parse_validation_config(app_m: &ArgMatches) -> ValidationConfig {
	let folder = app_m.value_of("VALIDATION_FOLDER").map(String::from);
	let max_images = app_m.value_of("VAL_MAX").and_then(|s| s.parse::<usize>().ok());
	let recurse = app_m.is_present("RECURSE_SUBFOLDERS");

	ValidationConfig::new(folder, max_images, recurse)
}

fn load_initial_parameters(
	app_m: &ArgMatches,
	network_config: &NetworkConfig,
) -> Result<Option<Vec<ndarray::ArrayD<f32>>>> {
	if let Some(param_str) = app_m.value_of("START_PARAMETERS") {
		println!(" initialising with parameters from: {}", param_str);
		let mut param_file = File::open(Path::new(param_str))?;
		let mut data = Vec::new();
		param_file.read_to_end(&mut data)?;
		let network_desc = crate::network_from_bytes(&data)?;

		validate_loaded_params(&network_desc, network_config);

		Ok(Some(network_desc.parameters))
	} else {
		Ok(None)
	}
}

fn validate_loaded_params(loaded: &NetworkDescription, config: &NetworkConfig) {
	if loaded.factor as usize != config.factor {
		eprintln!(
			"Using factor from parameter file ({}) rather than factor from argument ({})",
			loaded.factor, config.factor
		);
	}
	if loaded.width != config.width {
		eprintln!(
			"Using width from parameter file ({}) rather than width from argument ({})",
			loaded.width, config.width
		);
	}
	if loaded.log_depth != config.log_depth {
		eprintln!(
			"Using log_depth from parameter file ({}) rather than log_depth from argument ({})",
			loaded.log_depth, config.log_depth
		);
	}
}

fn print_training_info(network_config: &NetworkConfig, training_config: &TrainingConfig) {
	if training_config.srgb_downscale {
		println!(" sRGB downscaling");
	} else {
		println!(" RGB downscaling");
	}

	match training_config.loss_type {
		LossType::L1 => println!(" L1 loss"),
		LossType::L2 => println!(" L2 loss"),
	}

	println!(" learning rate: {}", training_config.learning_rate);
	println!(" patch_size: {}", training_config.patch_size);
	println!(" batch_size: {}", training_config.batch_size);
	println!(" factor: {}", network_config.factor);
	println!(" width: {}", network_config.width);
	println!(" log_depth: {}", network_config.log_depth);
}
