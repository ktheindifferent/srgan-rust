use crate::config::{LossType, NetworkConfig, TrainingConfig, ValidationConfig};
use crate::error::{Result, SrganError};
use crate::network::training_prescale_sr_net;
use crate::training::{train_network, DataLoader};
use crate::NetworkDescription;
use clap::ArgMatches;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub fn train_prescaled(app_m: &ArgMatches) -> Result<()> {
	println!("Training prescaled with:");

	let network_config = parse_network_config(app_m)?;
	let training_config = parse_training_config(app_m)?;
	let _validation_config = parse_validation_config(app_m);

	network_config.validate()?;
	training_config.validate()?;

	print_training_info(&network_config, &training_config);

	let initial_params = load_initial_parameters(app_m, &network_config)?;

	let (power, scale) = training_config.get_loss_params();
	let graph = training_prescale_sr_net(
		network_config.factor,
		network_config.width,
		network_config.log_depth,
		network_config.global_node_factor,
		crate::constants::training::REGULARIZATION_EPSILON,
		power,
		scale,
	)
	.map_err(|e| SrganError::GraphConstruction(format!("{}", e)))?;

	let input_folders: Vec<&str> = app_m
		.values_of("TRAINING_INPUT_FOLDER")
		.ok_or_else(|| SrganError::InvalidParameter("No training input folder specified".into()))?
		.collect();

	let param_file_path = app_m
		.value_of("PARAMETER_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No parameter file specified".into()))?;

	let mut training_stream = DataLoader::create_prescaled_training_stream(
		input_folders,
		training_config.recurse,
		training_config.patch_size,
		network_config.factor,
		training_config.batch_size,
	)?;

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

	if let Some(global_size) = app_m.value_of("GLOBAL_SIZE") {
		builder = builder.global_node_factor(
			global_size
				.parse()
				.map_err(|_| SrganError::InvalidParameter("Global node factor must be an integer".into()))?,
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

	builder = builder
		.quantise(app_m.is_present("QUANTISE"))
		.recurse(app_m.is_present("RECURSE_SUBFOLDERS"));

	Ok(builder.build())
}

fn parse_validation_config(app_m: &ArgMatches) -> ValidationConfig {
	let max_images = app_m.value_of("VAL_MAX").and_then(|s| s.parse::<usize>().ok());
	let recurse = app_m.is_present("RECURSE_SUBFOLDERS");

	ValidationConfig::new(None, max_images, recurse)
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
	match training_config.loss_type {
		LossType::L1 => println!(" L1 loss"),
		LossType::L2 => println!(" L2 loss"),
	}

	println!(" learning rate: {}", training_config.learning_rate);
	println!(" patch_size: {}", training_config.patch_size);
	println!(" batch_size: {}", training_config.batch_size);
	println!(" global_node_factor: {}", network_config.global_node_factor);
	println!(" factor: {}", network_config.factor);
	println!(" width: {}", network_config.width);
	println!(" log_depth: {}", network_config.log_depth);
}

fn create_prescaled_validation_stream(input_folders: Vec<&str>, recurse: bool) -> Result<Box<dyn alumina::data::DataStream>> {
	use alumina::data::{image_folder::ImageFolder, DataSet, DataStream};

	let mut folders_iter = input_folders.into_iter();
	let first_folder = folders_iter.next()
		.ok_or_else(|| SrganError::MissingFolder("At least one validation folder required".into()))?;

	let input_folder = Path::new(first_folder);
	let mut target_folder = input_folder
		.parent()
		.ok_or_else(|| SrganError::InvalidInput("Don't use root as a validation folder".into()))?
		.to_path_buf();
	target_folder.push("Base");

	let initial_set = ImageFolder::new(input_folder, recurse)
		.concat_components(ImageFolder::new(target_folder, recurse))
		.boxed();

	let set = folders_iter.fold(initial_set, |set, input_folder| {
		let input_path = Path::new(input_folder);
		let mut target_folder = input_path
			.parent()
			.map(|p| p.to_path_buf())
			.unwrap_or_else(|| PathBuf::from("."));
		target_folder.push("Base");

		ImageFolder::new(input_path, recurse)
			.concat_components(ImageFolder::new(target_folder, recurse))
			.concat_elements(set)
			.boxed()
	});

	Ok(Box::new(
		set.shuffle_random()
			.batch(1)
			.buffered(crate::constants::io::VALIDATION_BUFFER_THREADS),
	))
}

fn calculate_prescaled_epoch_size(input_folders: clap::Values, recurse: bool) -> usize {
	use alumina::data::{image_folder::ImageFolder, DataSet};

	let mut total_size = 0;
	for folder in input_folders {
		let folder_size = ImageFolder::new(folder, recurse).length();
		total_size += folder_size;
	}
	total_size
}
