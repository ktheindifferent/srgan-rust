extern crate alumina;
extern crate bincode;
extern crate byteorder;
extern crate clap;
extern crate image;
extern crate ndarray;
extern crate rand;
extern crate serde;
extern crate xz2;

extern crate rusty_sr;

use std::{
	cmp,
	fs::*,
	io::{stdout, Read, Write},
	iter,
	path::Path,
};

use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use ndarray::ArrayD;

use alumina::opt::adam::Adam;
// use alumina::opt::emwa2::Emwa2;
use alumina::{
	data::{
		image_folder::{image_to_data, ImageFolder},
		Cropping, DataSet, DataStream,
	},
	graph::{GraphDef, Result},
	opt::{CallbackSignal, Opt, UnboxedCallbacks},
};

use rusty_sr::{aligned_crop::*, network::*, psnr, NetworkDescription, UpscalingNetwork};

fn main() {
	let app_m = build_cli().get_matches();
	run_command(&app_m).unwrap_or_else(|err| println!("{:?}", err));
}

fn build_cli() -> App<'static, 'static> {
	App::new("Rusty SR")
		.version("v0.2.0")
		.author("J. Millard <millard.jn@gmail.com>")
		.about("A convolutional neural network trained to upscale images")
		.settings(&[AppSettings::SubcommandsNegateReqs, AppSettings::VersionlessSubcommands])
		.arg(build_input_file_arg())
		.arg(build_output_file_arg())
		.arg(build_parameters_arg())
		.arg(build_custom_arg())
		.arg(build_bilinear_factor_arg())
		.subcommand(build_train_subcommand())
		.subcommand(build_train_prescaled_subcommand())
		.subcommand(build_downscale_subcommand())
		.subcommand(build_quantise_subcommand())
		.subcommand(build_psnr_subcommand())
		.subcommand(build_set_width_subcommand())
}

fn build_input_file_arg() -> Arg<'static, 'static> {
	Arg::with_name("INPUT_FILE")
		.help("Sets the input image to upscale")
		.required(true)
		.index(1)
}

fn build_output_file_arg() -> Arg<'static, 'static> {
	Arg::with_name("OUTPUT_FILE")
		.help("Sets the output file to write/overwrite (.png recommended)")
		.required(true)
		.index(2)
}

fn build_parameters_arg() -> Arg<'static, 'static> {
	Arg::with_name("PARAMETERS")
		.help("Sets which built-in parameters to use with the neural net. Default: natural")
		.short("p")
		.long("parameters")
		.value_name("PARAMETERS")
		.possible_values(&["natural", "anime", "bilinear"])
		.empty_values(false)
}

fn build_custom_arg() -> Arg<'static, 'static> {
	Arg::with_name("CUSTOM")
		.conflicts_with("PARAMETERS")
		.short("c")
		.long("custom")
		.value_name("PARAMETER_FILE")
		.help("Sets a custom parameter file (.rsr) to use with the neural net")
		.empty_values(false)
}

fn build_bilinear_factor_arg() -> Arg<'static, 'static> {
	Arg::with_name("BILINEAR_FACTOR")
		.short("f")
		.long("factor")
		.help("The integer upscaling factor used if bilinear upscaling is performed. Default 4")
		.empty_values(false)
}

fn build_train_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("train")
		.about("Train a new set of neural parameters on your own dataset")
		.arg(Arg::with_name("TRAINING_FOLDER")
			.required(true)
			.index(1)
			.help("Images from this folder(or sub-folders) will be used for training"))
		.arg(Arg::with_name("PARAMETER_FILE")
			.required(true)
			.index(2)
			.help("Learned network parameters will be (over)written to this parameter file (.rsr)"))
		.arg(build_learning_rate_arg())
		.arg(build_quantise_arg())
		.arg(build_training_loss_arg())
		.arg(Arg::with_name("DOWNSCALE_COLOURSPACE")
			.help("Colourspace in which to perform downsampling. Default: sRGB")
			.short("c")
			.long("colourspace")
			.value_name("COLOURSPACE")
			.possible_values(&["sRGB", "RGB"])
			.empty_values(false))
		.arg(build_recurse_arg())
		.arg(build_start_parameters_arg())
		.arg(build_factor_arg())
		.arg(build_width_arg())
		.arg(build_log_depth_arg())
		.arg(build_patch_size_arg())
		.arg(build_batch_size_arg())
		.arg(Arg::with_name("VALIDATION_FOLDER")
			.short("v")
			.long("val_folder")
			.help("Images from this folder(or sub-folders) will be used to evaluate training progress")
			.empty_values(false))
		.arg(build_val_max_arg())
}

fn build_train_prescaled_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("train_prescaled")
		.about("Train a new set of neural parameters on your own dataset")
		.arg(Arg::with_name("PARAMETER_FILE")
			.required(true)
			.index(1)
			.help("Learned network parameters will be (over)written to this parameter file (.rsr)"))
		.arg(Arg::with_name("TRAINING_INPUT_FOLDER")
			.required(true)
			.short("i")
			.long("input")
			.help("Images from this folder(or sub-folders) will be used as the target for training, comparing against target images in neighbour folder 'Base'")
			.empty_values(false)
			.multiple(true)
			.number_of_values(1))
		.arg(build_learning_rate_arg())
		.arg(build_quantise_arg())
		.arg(build_training_loss_arg())
		.arg(build_recurse_arg())
		.arg(build_start_parameters_arg())
		.arg(build_factor_arg())
		.arg(build_width_arg())
		.arg(build_log_depth_arg())
		.arg(build_patch_size_arg())
		.arg(build_batch_size_arg())
		.arg(Arg::with_name("GLOBAL_SIZE")
			.short("g")
			.long("global_node_factor")
			.help("The relative size of the global (non-spatial) nodes in each layer. Default: 2")
			.empty_values(false))
		.arg(Arg::with_name("VALIDATION_INPUT_FOLDER")
			.short("x")
			.long("val_folder")
			.help("Images from this folder(or sub-folders) will be used to evaluate training progress, comparing against target images in neighbour folder 'Base'")
			.empty_values(false)
			.multiple(true)
			.number_of_values(1))
		.arg(build_val_max_arg())
}

fn build_downscale_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("downscale")
		.about("Downscale images")
		.arg(Arg::with_name("FACTOR")
			.help("The integer downscaling factor")
			.required(true)
			.index(1))
		.arg(Arg::with_name("INPUT_FILE")
			.help("Sets the input image to downscale")
			.required(true)
			.index(2))
		.arg(Arg::with_name("OUTPUT_FILE")
			.help("Sets the output file to write/overwrite (.png recommended)")
			.required(true)
			.index(3))
		.arg(Arg::with_name("COLOURSPACE")
			.help("colourspace in which to perform downsampling. Default: sRGB")
			.short("c")
			.long("colourspace")
			.value_name("COLOURSPACE")
			.possible_values(&["sRGB", "RGB"])
			.empty_values(false))
}

fn build_quantise_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("quantise")
		.about("Quantise the weights of a network, reducing file size")
		.arg(Arg::with_name("INPUT_FILE")
			.help("The input network to be quantised")
			.required(true)
			.index(1))
		.arg(Arg::with_name("OUTPUT_FILE")
			.help("The location at which the quantised network will be saved")
			.required(true)
			.index(2))
}

fn build_psnr_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("psnr")
		.about("Print the PSNR value from the differences between the two images")
		.arg(Arg::with_name("IMAGE1")
			.required(true)
			.index(1)
			.help("PSNR is calculated using the difference between this image and IMAGE2"))
		.arg(Arg::with_name("IMAGE2")
			.required(true)
			.index(2)
			.help("PSNR is calculated using the difference between this image and IMAGE1"))
}

fn build_set_width_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("set_width")
		.about("Set the width of a network")
		.arg(Arg::with_name("INPUT_FILE")
			.help("The input network to be updated with a width")
			.required(true)
			.index(1))
		.arg(Arg::with_name("WIDTH")
			.help("The width to set")
			.required(true)
			.index(2))
		.arg(Arg::with_name("OUTPUT_FILE")
			.help("The location at which the new network will be saved")
			.required(true)
			.index(3))
}

fn build_learning_rate_arg() -> Arg<'static, 'static> {
	Arg::with_name("LEARNING_RATE")
		.short("R")
		.long("rate")
		.help("The learning rate used by the Adam optimiser. Default: 3e-3")
		.empty_values(false)
}

fn build_quantise_arg() -> Arg<'static, 'static> {
	Arg::with_name("QUANTISE")
		.short("q")
		.long("quantise")
		.help("Quantise the weights by zeroing the smallest 12 bits of each f32. Reduces parameter file size")
		.takes_value(false)
}

fn build_training_loss_arg() -> Arg<'static, 'static> {
	Arg::with_name("TRAINING_LOSS")
		.help("Selects whether the neural net learns to minimise the L1 or L2 loss. Default: L1")
		.short("l")
		.long("loss")
		.value_name("LOSS")
		.possible_values(&["L1", "L2"])
		.empty_values(false)
}

fn build_recurse_arg() -> Arg<'static, 'static> {
	Arg::with_name("RECURSE_SUBFOLDERS")
		.short("r")
		.long("recurse")
		.help("Recurse into subfolders of training and validation folders looking for files")
		.takes_value(false)
}

fn build_start_parameters_arg() -> Arg<'static, 'static> {
	Arg::with_name("START_PARAMETERS")
		.short("s")
		.long("start")
		.help("Start training from known parameters loaded from this .rsr file rather than random initialisation")
		.empty_values(false)
}

fn build_factor_arg() -> Arg<'static, 'static> {
	Arg::with_name("FACTOR")
		.short("f")
		.long("factor")
		.help("The integer upscaling factor of the network the be trained. Default: 4")
		.empty_values(false)
}

fn build_width_arg() -> Arg<'static, 'static> {
	Arg::with_name("WIDTH")
		.short("w")
		.long("width")
		.help("The minimum number of channels in the hidden layers in the network the be trained. Default: 16")
		.empty_values(false)
}

fn build_log_depth_arg() -> Arg<'static, 'static> {
	Arg::with_name("LOG_DEPTH")
		.short("d")
		.long("log_depth")
		.help("There will be 2^(log_depth)-1 hidden layers in the network the be trained. Default: 4")
		.empty_values(false)
}

fn build_patch_size_arg() -> Arg<'static, 'static> {
	Arg::with_name("PATCH_SIZE")
		.short("p")
		.long("patch_size")
		.help("The integer patch_size of the training input after downsampling. Default: 48")
		.empty_values(false)
}

fn build_batch_size_arg() -> Arg<'static, 'static> {
	Arg::with_name("BATCH_SIZE")
		.short("b")
		.long("batch_size")
		.help("The integer batch_size of the training input. Default: 4")
		.empty_values(false)
}

fn build_val_max_arg() -> Arg<'static, 'static> {
	Arg::with_name("VAL_MAX")
		.requires("VALIDATION_FOLDER")
		.short("m")
		.long("val_max")
		.value_name("N")
		.help("Set upper limit on number of images used for each validation pass")
		.empty_values(false)
}

fn run_command(app_m: &ArgMatches) -> ::std::result::Result<(), String> {
	match app_m.subcommand() {
		("train", Some(sub_m)) => train(sub_m).map_err(|e| format!("Train error: {}", e)),
		("train_prescaled", Some(sub_m)) => train_prescaled(sub_m).map_err(|e| format!("Train_prescaled error: {}", e)),
		("downscale", Some(sub_m)) => downscale(sub_m),
		("psnr", Some(sub_m)) => psnr(sub_m),
		("quantise", Some(sub_m)) => quantise(sub_m),
		("set_width", Some(sub_m)) => set_width(sub_m),
		_ => upscale(app_m),
	}
}

pub fn psnr(app_m: &ArgMatches) -> ::std::result::Result<(), String> {
	let image1 = image::open(Path::new(app_m.value_of("IMAGE1").expect("No input file given?")))
		.map_err(|e| format!("Error opening image1 file: {}", e))?;
	let image2 = image::open(Path::new(app_m.value_of("IMAGE2").expect("No input file given?")))
		.map_err(|e| format!("Error opening image2 file: {}", e))?;

	let image1 = image_to_data(&image1);
	let image2 = image_to_data(&image2);

	if image1.shape() != image2.shape() {
		println!("Image shapes will be cropped to the top left areas which overlap");
	}

	let (err, y_err, pix) = psnr::psnr_calculation(image1.view(), image2.view());

	println!(
		"sRGB PSNR: {}\tLuma PSNR:{}",
		-10.0 * (err / pix).log10(),
		-10.0 * (y_err / pix).log10()
	);
	Ok(())
}

fn quantise(app_m: &ArgMatches) -> ::std::result::Result<(), String> {
	let mut input_file = File::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?")))
		.map_err(|e| format!("Error opening input file: {}", e))?;
	let mut input_data = Vec::new();
	input_file
		.read_to_end(&mut input_data)
		.map_err(|e| format!("Error reading input file: {}", e))?;
	let input_network = rusty_sr::network_from_bytes(&input_data)?;

	let mut output_file = File::create(Path::new(app_m.value_of("OUTPUT_FILE").expect("No output file given?")))
		.map_err(|e| format!("Error creating output file: {}", e))?;
	let output_data = rusty_sr::network_to_bytes(input_network, true).map_err(|e| e.to_string())?;
	output_file
		.write_all(&output_data)
		.map_err(|e| format!("Error writing output file: {}", e))?;

	Ok(())
}

fn downscale(app_m: &ArgMatches) -> ::std::result::Result<(), String> {
	let factor = match app_m.value_of("FACTOR") {
		Some(string) => string.parse::<usize>().expect("Factor argument must be an integer"),
		_ => unreachable!(),
	};

	let srgb = match app_m.value_of("COLOURSPACE") {
		Some("sRGB") | None => true,
		Some("RGB") => false,
		_ => unreachable!(),
	};

	let input = rusty_sr::read(
		&mut File::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?")))
			.map_err(|e| format!("Error opening input file: {}", e))?,
	).map_err(|e| format!("Error reading input file: {}", e))?;

	let output = rusty_sr::downscale(input, factor, srgb).map_err(|e| format!("Error while downscaling: {}", e))?;

	print!(" Writing file...");
	stdout().flush().ok();
	rusty_sr::save(
		output,
		&mut File::create(Path::new(app_m.value_of("OUTPUT_FILE").expect("No output file given?")))
			.map_err(|e| format!("Error creating output file: {}", e))?,
	).map_err(|e| format!("Error writing output file: {}", e))?;

	println!(" Done");
	Ok(())
}

fn upscale(app_m: &ArgMatches) -> ::std::result::Result<(), String> {
	let factor = match app_m.value_of("BILINEAR_FACTOR") {
		Some(string) => string.parse::<usize>().expect("Factor argument must be an integer"),
		_ => 4,
	};

	//-- Sort out parameters and graph
	let network: UpscalingNetwork = if let Some(file_str) = app_m.value_of("CUSTOM") {
		let mut param_file = File::open(Path::new(file_str)).expect("Error opening parameter file");
		let mut data = Vec::new();
		param_file
			.read_to_end(&mut data)
			.expect("Reading parameter file failed");
		UpscalingNetwork::new(rusty_sr::network_from_bytes(&data)?, "custom trained neural net")?
	} else {
		UpscalingNetwork::from_label(app_m.value_of("PARAMETERS").unwrap_or("natural"), Some(factor))
			.map_err(|e| format!("Error parsing PARAMETERS: {}", e))?
	};
	println!("Upsampling using {}...", network);

	let input = rusty_sr::read(
		&mut File::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?")))
			.map_err(|e| format!("Error opening input file: {}", e))?,
	).map_err(|e| format!("Error reading input file: {}", e))?;

	let output = rusty_sr::upscale(input, &network).map_err(|e| format!("Error while upscaling: {}", e))?;

	print!(" Writing file...");
	stdout().flush().ok();
	rusty_sr::save(
		output,
		&mut File::create(Path::new(app_m.value_of("OUTPUT_FILE").expect("No output file given?")))
			.map_err(|e| format!("Error creating output file: {}", e))?,
	).map_err(|e| format!("Error writing output file: {}", e))?;

	println!(" Done");
	Ok(())
}

fn train(app_m: &ArgMatches) -> Result<()> {
	println!("Training with:");

	let srgb_downscale = match app_m.value_of("DOWNSCALE_COLOURSPACE") {
		Some("sRGB") | None => {
			println!(" sRGB downscaling");
			true
		},
		Some("RGB") => {
			println!(" RGB downscaling");
			false
		},
		_ => unreachable!(),
	};

	let (power, scale) = match app_m.value_of("TRAINING_LOSS") {
		Some("L1") | None => {
			println!(" L1 loss");
			(1.0, 1.0 / 255.0)
		},
		Some("L2") => {
			println!(" L2 loss");
			(2.0, 1.0 / 255.0)
		},
		_ => unreachable!(),
	};

	let lr = app_m
		.value_of("LEARNING_RATE")
		.map(|string| {
			string
				.parse::<f32>()
				.expect("Learning rate argument must be a numeric value")
		})
		.unwrap_or(3e-3);
	if lr <= 0.0 {
		eprintln!("Learning_rate ({}) probably should be greater than 0.", lr);
	}
	println!(" learning rate: {}", lr);

	let patch_size = app_m
		.value_of("PATCH_SIZE")
		.map(|string| string.parse::<usize>().expect("Patch_size argument must be an integer"))
		.unwrap_or(48);
	assert!(patch_size > 0, "Patch_size ({}) must be greater than 0.", patch_size);
	println!(" patch_size: {}", patch_size);

	let batch_size = app_m
		.value_of("BATCH_SIZE")
		.map(|string| string.parse::<usize>().expect("Batch_size argument must be an integer"))
		.unwrap_or(4);
	assert!(batch_size > 0, "Batch_size ({}) must be greater than 0.", batch_size);
	println!(" batch_size: {}", batch_size);

	let quantise = app_m.is_present("QUANTISE");

	let recurse = app_m.is_present("RECURSE_SUBFOLDERS");

	let training_folder = app_m.value_of("TRAINING_FOLDER").expect("No training folder?");

	let param_file_path = Path::new(app_m.value_of("PARAMETER_FILE").expect("No parameter file?")).to_path_buf();

	let mut factor_option = app_m
		.value_of("FACTOR")
		.map(|s| s.parse::<usize>().expect("Factor argument must be an integer"));

	let mut width_option = app_m
		.value_of("WIDTH")
		.map(|s| s.parse::<u32>().expect("Width argument must be an integer"));

	let mut log_depth_option = app_m
		.value_of("LOG_DEPTH")
		.map(|s| s.parse::<u32>().expect("Log_depth argument must be an integer"));

	let mut params_option = None;
	if let Some(param_str) = app_m.value_of("START_PARAMETERS") {
		println!(" initialising with parameters from: {}", param_str);
		let mut param_file = File::open(Path::new(param_str)).expect("Error opening start parameter file");
		let mut data = Vec::new();
		param_file
			.read_to_end(&mut data)
			.expect("Reading start parameter file failed");
		let network_desc = rusty_sr::network_from_bytes(&data)?;
		if let Some(factor) = factor_option {
			if factor != network_desc.factor as usize {
				eprintln!(
					"Using factor from parameter file ({}) rather than factor from argument ({})",
					network_desc.factor, factor
				);
			}
		}
		if let Some(width) = width_option {
			if width != network_desc.width {
				eprintln!(
					"Using width from parameter file ({}) rather than width from argument ({})",
					network_desc.width, width
				);
			}
		}
		if let Some(log_depth) = log_depth_option {
			if log_depth != network_desc.log_depth {
				eprintln!(
					"Using log_depth from parameter file ({}) rather than log_depth from argument ({})",
					network_desc.log_depth, log_depth
				);
			}
		}
		params_option = Some(network_desc.parameters);
		factor_option = Some(network_desc.factor as usize);
		width_option = Some(network_desc.width);
		log_depth_option = Some(network_desc.log_depth);
	}

	let factor = factor_option.unwrap_or(4);
	println!(" factor: {}", factor);

	let width = width_option.unwrap_or(16);
	println!(" width: {}", width);

	let log_depth = log_depth_option.unwrap_or(4);
	println!(" log_depth: {}", log_depth);

	let global_node_factor = 0; // no need for global nodes for one downsampling method

	let graph = training_sr_net(
		factor,
		width,
		log_depth,
		global_node_factor,
		1e-5,
		power,
		scale,
		srgb_downscale,
	)?;

	let mut training_stream = ImageFolder::new(training_folder, recurse)
		.crop(0, &[patch_size * factor, patch_size * factor, 3], Cropping::Random)
		.shuffle_random()
		.batch(batch_size)
		.buffered(2);

	let mut solver = Adam::new(&graph)?.rate(lr).beta1(0.95).beta2(0.995).bias_correct(false);

	let params = params_option.unwrap_or_else(|| {
		graph
			.initialise_nodes(solver.parameters())
			.expect("Could not initialise parameters")
	});

	solver.add_callback(move |data| {
		if (data.step + 1) % 1000 == 0 {
			let mut parameter_file = File::create(&param_file_path).expect("Could not make parameter file");
			let bytes = rusty_sr::network_to_bytes(
				NetworkDescription {
					factor: factor as u32,
					width: width,
					log_depth: log_depth,
					global_node_factor: global_node_factor as u32,
					parameters: data.params.to_vec(),
				},
				quantise,
			).unwrap();
			parameter_file
				.write_all(&bytes)
				.expect("Could not save to parameter file");
		}
		println!("step {}\terr:{}\tchange:{}", data.step, data.err, data.change_norm);
		CallbackSignal::Continue
	});

	let mut validation = validation(app_m, recurse, &mut solver, &graph)?;

	validation(&params);

	solver.add_boxed_callback(Box::new(move |data| {
		if (data.step + 1) % 1000 == 0 {
			validation(data.params)
		}
		CallbackSignal::Continue
	}));

	println!("Beginning Training");
	solver.optimise_from(&mut training_stream, params)?;
	println!("Done");
	Ok(())
}

/// Add occasional validation set evaluation as solver callback
fn validation(
	app_m: &ArgMatches,
	recurse: bool,
	solver: &mut Opt,
	graph: &GraphDef,
) -> Result<Box<FnMut(&[ArrayD<f32>])>> {
	if let Some(val_folder) = app_m.value_of("VALIDATION_FOLDER") {
		let training_input_id = graph.node_id("training_input").value_id();
		let input_ids: Vec<_> = iter::once(training_input_id.clone())
			.chain(solver.parameters().iter().map(|node_id| node_id.value_id()))
			.collect();
		let output_id = graph.node_id("output").value_id();
		let mut validation_subgraph = graph.subgraph(&input_ids, &[output_id.clone(), training_input_id.clone()])?;

		let validation_set = ImageFolder::new(val_folder, recurse);
		let epoch_size = validation_set.length();
		let mut validation_stream = validation_set.shuffle_random().batch(1).buffered(4);

		let n: usize = app_m
			.value_of("VAL_MAX")
			.map(|val_max| {
				cmp::min(
					epoch_size,
					val_max.parse::<usize>().expect("-val_max N must be a positive integer"),
				)
			})
			.unwrap_or(epoch_size);

		Ok(Box::new(move |params| {
			let mut err_sum = 0.0;
			let mut y_err_sum = 0.0;
			let mut pix_sum = 0.0f32;

			let mut psnr_sum = 0.0;
			let mut y_psnr_sum = 0.0;

			for _ in 0..n {
				let mut training_input = validation_stream.next();
				training_input.extend(params.to_vec());

				let result = validation_subgraph
					.execute(training_input)
					.expect("Could not execute upsampling graph");
				let output = result.get(&output_id).unwrap();
				let training_input = result.get(&training_input_id).unwrap();

				let (err, y_err, pix) = psnr::psnr_calculation(output, training_input);

				pix_sum += pix;
				err_sum += err;
				y_err_sum += y_err;

				psnr_sum += -10.0 * (err / pix).log10();
				y_psnr_sum += -10.0 * (y_err / pix).log10();
			}

			psnr_sum /= n as f32;
			y_psnr_sum /= n as f32;
			let psnr = -10.0 * (err_sum / pix_sum).log10();
			let y_psnr = -10.0 * (y_err_sum / pix_sum).log10();
			println!(
				"Validation PixAvgPSNR:\t{}\tPixAvgY_PSNR:\t{}\tImgAvgPSNR:\t{}\tImgAvgY_PSNR:\t{}",
				psnr, y_psnr, psnr_sum, y_psnr_sum
			);
		}))
	} else {
		Ok(Box::new(|_params| {}))
	}
}

fn train_prescaled(app_m: &ArgMatches) -> Result<()> {
	println!("Training prescaled with:");

	let (power, scale) = match app_m.value_of("TRAINING_LOSS") {
		Some("L1") | None => {
			println!(" L1 loss");
			(1.0, 1.0 / 255.0)
		},
		Some("L2") => {
			println!(" L2 loss");
			(2.0, 1.0 / 255.0)
		},
		_ => unreachable!(),
	};

	let lr = app_m
		.value_of("LEARNING_RATE")
		.map(|string| {
			string
				.parse::<f32>()
				.expect("Learning rate argument must be a numeric value")
		})
		.unwrap_or(3e-3);
	if lr <= 0.0 {
		eprintln!("Learning_rate ({}) probably should be greater than 0.", lr);
	}
	println!(" learning rate: {}", lr);

	let patch_size = app_m
		.value_of("PATCH_SIZE")
		.map(|string| string.parse::<usize>().expect("Patch_size argument must be an integer"))
		.unwrap_or(48);
	assert!(patch_size > 0, "Patch_size ({}) must be greater than 0.", patch_size);
	println!(" patch_size: {}", patch_size);

	let batch_size = app_m
		.value_of("BATCH_SIZE")
		.map(|string| string.parse::<usize>().expect("Batch_size argument must be an integer"))
		.unwrap_or(4);
	assert!(batch_size > 0, "Batch_size ({}) must be greater than 0.", batch_size);
	println!(" batch_size: {}", batch_size);

	let global_node_factor = app_m
		.value_of("GLOBAL_SIZE")
		.map(|string| {
			string
				.parse::<usize>()
				.expect("Global_node_factor argument must be an integer")
		})
		.unwrap_or(2);
	println!(" global_node_factor: {}", global_node_factor);

	let quantise = app_m.is_present("QUANTISE");

	let recurse = app_m.is_present("RECURSE_SUBFOLDERS");

	let mut input_folders = app_m
		.values_of("TRAINING_INPUT_FOLDER")
		.expect("No training input folder?");

	let param_file_path = Path::new(app_m.value_of("PARAMETER_FILE").expect("No parameter file?")).to_path_buf();

	let mut factor_option = app_m
		.value_of("FACTOR")
		.map(|s| s.parse::<usize>().expect("Factor argument must be an integer"));

	let mut width_option = app_m
		.value_of("WIDTH")
		.map(|s| s.parse::<u32>().expect("Width argument must be an integer"));

	let mut log_depth_option = app_m
		.value_of("LOG_DEPTH")
		.map(|s| s.parse::<u32>().expect("Log_depth argument must be an integer"));

	let mut params_option = None;
	if let Some(param_str) = app_m.value_of("START_PARAMETERS") {
		println!(" initialising with parameters from: {}", param_str);
		let mut param_file = File::open(Path::new(param_str)).expect("Error opening start parameter file");
		let mut data = Vec::new();
		param_file
			.read_to_end(&mut data)
			.expect("Reading start parameter file failed");
		let network_desc = rusty_sr::network_from_bytes(&data)?;
		if let Some(factor) = factor_option {
			if factor != network_desc.factor as usize {
				eprintln!(
					"Using factor from parameter file ({}) rather than factor from argument ({})",
					network_desc.factor, factor
				);
			}
		}
		if let Some(width) = width_option {
			if width != network_desc.width {
				eprintln!(
					"Using width from parameter file ({}) rather than width from argument ({})",
					network_desc.width, width
				);
			}
		}
		if let Some(log_depth) = log_depth_option {
			if log_depth != network_desc.log_depth {
				eprintln!(
					"Using log_depth from parameter file ({}) rather than log_depth from argument ({})",
					network_desc.log_depth, log_depth
				);
			}
		}
		params_option = Some(network_desc.parameters);
		factor_option = Some(network_desc.factor as usize);
		width_option = Some(network_desc.width);
		log_depth_option = Some(network_desc.log_depth);
	}

	let factor = factor_option.unwrap_or(4);
	println!(" factor: {}", factor);

	let width = width_option.unwrap_or(16);
	println!(" width: {}", width);

	let log_depth = log_depth_option.unwrap_or(4);
	println!(" log_depth: {}", log_depth);

	let graph = training_prescale_sr_net(
		factor as usize,
		width,
		log_depth,
		global_node_factor,
		1e-5,
		power,
		scale,
	)?;

	let input_folder = Path::new(input_folders.next().unwrap());
	let mut target_folder = input_folder
		.parent()
		.expect("Don't use root as a training folder.")
		.to_path_buf();
	target_folder.push("Base");
	let initial_set = ImageFolder::new(input_folder, recurse)
		.concat_components(ImageFolder::new(target_folder, recurse))
		.boxed();

	let set = input_folders.into_iter().fold(initial_set, |set, input_folder| {
		let mut target_folder = Path::new(input_folder)
			.parent()
			.expect("Don't use root as a training folder.")
			.to_path_buf();
		target_folder.push("Base");
		ImageFolder::new(input_folder, recurse)
			.concat_components(ImageFolder::new(target_folder, recurse))
			.concat_elements(set)
			.boxed()
	});

	let mut training_stream = set
		.aligned_crop(0, &[patch_size, patch_size, 3], Cropping::Random)
		.and_crop(1, &[factor, factor, 1])
		.shuffle_random()
		.batch(batch_size)
		.buffered(2);

	let mut solver = Adam::new(&graph)?.rate(lr).beta1(0.95).beta2(0.995).bias_correct(false);

	let params = params_option.unwrap_or_else(|| {
		graph
			.initialise_nodes(solver.parameters())
			.expect("Could not initialise parameters")
	});

	solver.add_callback(move |data| {
		if (data.step + 1) % 1000 == 0 {
			let mut parameter_file = File::create(&param_file_path).expect("Could not make parameter file");
			let bytes = rusty_sr::network_to_bytes(
				NetworkDescription {
					factor: factor as u32,
					width: width,
					log_depth: log_depth,
					global_node_factor: global_node_factor as u32,
					parameters: data.params.to_vec(),
				},
				quantise,
			).unwrap();
			parameter_file
				.write_all(&bytes)
				.expect("Could not save to parameter file");
		}
		println!("step {}\terr:{}\tchange:{}", data.step, data.err, data.change_norm);
		CallbackSignal::Continue
	});

	let mut validation = validation_prescaled(app_m, recurse, &mut solver, &graph)?;

	validation(&params);

	solver.add_boxed_callback(Box::new(move |data| {
		if (data.step + 1) % 1000 == 0 {
			validation(data.params)
		}
		CallbackSignal::Continue
	}));

	println!("Beginning Training");
	solver.optimise_from(&mut training_stream, params)?;
	println!("Done");
	Ok(())
}

/// Add occasional validation set evaluation as solver callback
fn validation_prescaled(
	app_m: &ArgMatches,
	recurse: bool,
	solver: &mut Opt,
	graph: &GraphDef,
) -> Result<Box<FnMut(&[ArrayD<f32>])>> {
	if let Some(mut input_folders) = app_m.values_of("VALIDATION_INPUT_FOLDER") {
		let input_id = graph.node_id("input").value_id();
		let input_ids: Vec<_> = iter::once(input_id.clone())
			.chain(solver.parameters().iter().map(|node_id| node_id.value_id()))
			.collect();
		let output_id = graph.node_id("output").value_id();
		let mut validation_subgraph = graph.subgraph(&input_ids, &[output_id.clone(), input_id.clone()])?;

		let input_folder = Path::new(input_folders.next().unwrap());
		let mut val_folder = input_folder
			.parent()
			.expect("Don't use root as a validation folder.")
			.to_path_buf();
		val_folder.push("Base");
		let initial_set = ImageFolder::new(input_folder, recurse)
			.concat_components(ImageFolder::new(val_folder, recurse))
			.boxed();

		let validation_set = input_folders.into_iter().fold(initial_set, |set, input_folder| {
			let mut val_folder = Path::new(input_folder)
				.parent()
				.expect("Don't use root as a validation folder.")
				.to_path_buf();
			val_folder.push("Base");
			ImageFolder::new(input_folder, recurse)
				.concat_components(ImageFolder::new(val_folder, recurse))
				.concat_elements(set)
				.boxed()
		});

		let epoch_size = validation_set.length();
		let mut validation_stream = validation_set.shuffle_random().batch(1).buffered(4);

		let n: usize = app_m
			.value_of("VAL_MAX")
			.map(|val_max| {
				cmp::min(
					epoch_size,
					val_max.parse::<usize>().expect("-val_max N must be a positive integer"),
				)
			})
			.unwrap_or(epoch_size);

		Ok(Box::new(move |params| {
			let mut err_sum = 0.0;
			let mut y_err_sum = 0.0;
			let mut pix_sum = 0.0f32;

			let mut psnr_sum = 0.0;
			let mut y_psnr_sum = 0.0;

			for _ in 0..n {
				let mut validation_input = validation_stream.next();
				let target = validation_input.remove(1);
				validation_input.extend(params.to_vec());

				let result = validation_subgraph
					.execute(validation_input)
					.expect("Could not execute upsampling graph");
				let output = result.get(&output_id).unwrap();

				let (err, y_err, pix) = psnr::psnr_calculation(output, target.view());

				pix_sum += pix;
				err_sum += err;
				y_err_sum += y_err;

				psnr_sum += -10.0 * (err / pix).log10();
				y_psnr_sum += -10.0 * (y_err / pix).log10();
			}

			psnr_sum /= n as f32;
			y_psnr_sum /= n as f32;
			let psnr = -10.0 * (err_sum / pix_sum).log10();
			let y_psnr = -10.0 * (y_err_sum / pix_sum).log10();
			println!(
				"Validation PixAvgPSNR:\t{}\tPixAvgY_PSNR:\t{}\tImgAvgPSNR:\t{}\tImgAvgY_PSNR:\t{}",
				psnr, y_psnr, psnr_sum, y_psnr_sum
			);
		}))
	} else {
		Ok(Box::new(|_params| {}))
	}
}

fn set_width(app_m: &ArgMatches) -> ::std::result::Result<(), String> {
	let width = app_m
		.value_of("WIDTH")
		.map(|s| s.parse::<u32>().expect("Width argument must be an integer"))
		.unwrap();

	let mut input_file = File::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?")))
		.map_err(|e| format!("Error opening input file: {}", e))?;
	let mut input_data = Vec::new();
	input_file
		.read_to_end(&mut input_data)
		.map_err(|e| format!("Error reading input file: {}", e))?;
	let input_network = rusty_sr::network_from_bytes(&input_data)?;

	let output_network = NetworkDescription {
		factor: input_network.factor,
		width: width,
		log_depth: input_network.log_depth,
		global_node_factor: input_network.global_node_factor,
		parameters: input_network.parameters,
	};

	let mut output_file = File::create(Path::new(app_m.value_of("OUTPUT_FILE").expect("No output file given?")))
		.map_err(|e| format!("Error creating output file: {}", e))?;
	let output_data = rusty_sr::network_to_bytes(output_network, false).map_err(|e| e.to_string())?;
	output_file
		.write_all(&output_data)
		.map_err(|e| format!("Error writing output file: {}", e))?;

	Ok(())
}
