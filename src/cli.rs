use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};

pub fn build_cli() -> ArgMatches<'static> {
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
		.get_matches()
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
		.arg(
			Arg::with_name("TRAINING_FOLDER")
				.required(true)
				.index(1)
				.help("Images from this folder(or sub-folders) will be used for training"),
		)
		.arg(
			Arg::with_name("PARAMETER_FILE")
				.required(true)
				.index(2)
				.help("Learned network parameters will be (over)written to this parameter file (.rsr)"),
		)
		.arg(build_learning_rate_arg())
		.arg(build_quantise_arg())
		.arg(build_training_loss_arg())
		.arg(build_downscale_colourspace_arg())
		.arg(build_recurse_arg())
		.arg(build_start_parameters_arg())
		.arg(build_factor_arg())
		.arg(build_width_arg())
		.arg(build_log_depth_arg())
		.arg(build_patch_size_arg())
		.arg(build_batch_size_arg())
		.arg(build_validation_folder_arg())
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
		.arg(
			Arg::with_name("FACTOR")
				.help("The integer downscaling factor")
				.required(true)
				.index(1),
		)
		.arg(
			Arg::with_name("INPUT_FILE")
				.help("Sets the input image to downscale")
				.required(true)
				.index(2),
		)
		.arg(
			Arg::with_name("OUTPUT_FILE")
				.help("Sets the output file to write/overwrite (.png recommended)")
				.required(true)
				.index(3),
		)
		.arg(
			Arg::with_name("COLOURSPACE")
				.help("colourspace in which to perform downsampling. Default: sRGB")
				.short("c")
				.long("colourspace")
				.value_name("COLOURSPACE")
				.possible_values(&["sRGB", "RGB"])
				.empty_values(false),
		)
}

fn build_quantise_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("quantise")
		.about("Quantise the weights of a network, reducing file size")
		.arg(
			Arg::with_name("INPUT_FILE")
				.help("The input network to be quantised")
				.required(true)
				.index(1),
		)
		.arg(
			Arg::with_name("OUTPUT_FILE")
				.help("The location at which the quantised network will be saved")
				.required(true)
				.index(2),
		)
}

fn build_psnr_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("psnr")
		.about("Print the PSNR value from the differences between the two images")
		.arg(
			Arg::with_name("IMAGE1")
				.required(true)
				.index(1)
				.help("PSNR is calculated using the difference between this image and IMAGE2"),
		)
		.arg(
			Arg::with_name("IMAGE2")
				.required(true)
				.index(2)
				.help("PSNR is calculated using the difference between this image and IMAGE1"),
		)
}

fn build_set_width_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("set_width")
		.about("Set the width of a network")
		.arg(
			Arg::with_name("INPUT_FILE")
				.help("The input network to be updated with a width")
				.required(true)
				.index(1),
		)
		.arg(Arg::with_name("WIDTH").help("The width to set").required(true).index(2))
		.arg(
			Arg::with_name("OUTPUT_FILE")
				.help("The location at which the new network will be saved")
				.required(true)
				.index(3),
		)
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

fn build_downscale_colourspace_arg() -> Arg<'static, 'static> {
	Arg::with_name("DOWNSCALE_COLOURSPACE")
		.help("Colourspace in which to perform downsampling. Default: sRGB")
		.short("c")
		.long("colourspace")
		.value_name("COLOURSPACE")
		.possible_values(&["sRGB", "RGB"])
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

fn build_validation_folder_arg() -> Arg<'static, 'static> {
	Arg::with_name("VALIDATION_FOLDER")
		.short("v")
		.long("val_folder")
		.help("Images from this folder(or sub-folders) will be used to evaluate training progress")
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
