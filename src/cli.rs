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
		.arg(build_auto_detect_arg())
		.arg(build_bilinear_factor_arg())
		.arg(build_gpu_flag_arg())
		.arg(build_format_arg())
		.arg(build_quality_arg())
		.arg(build_tile_size_arg())
		.arg(build_progressive_arg())
		.arg(build_auto_enhance_arg())
		.subcommand(build_train_subcommand())
		.subcommand(build_train_prescaled_subcommand())
		.subcommand(build_batch_subcommand())
		.subcommand(build_downscale_subcommand())
		.subcommand(build_quantise_subcommand())
		.subcommand(build_psnr_subcommand())
		.subcommand(build_set_width_subcommand())
		.subcommand(build_upscale_gpu_subcommand())
		.subcommand(build_list_gpus_subcommand())
		.subcommand(build_benchmark_subcommand())
		.subcommand(build_parallel_benchmark_subcommand())
		.subcommand(build_generate_config_subcommand())
		.subcommand(build_profile_memory_subcommand())
		.subcommand(build_analyze_memory_subcommand())
		.subcommand(build_server_subcommand())
		.subcommand(build_download_model_subcommand())
		.subcommand(build_download_models_subcommand())
		.subcommand(build_models_subcommand())
		.subcommand(build_compare_subcommand())
		.subcommand(build_completions_subcommand())
		.subcommand(build_classify_subcommand())
		.subcommand(build_batch_status_subcommand())
		.subcommand(build_video_subcommand())
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
		.help(
			"Sets which built-in parameters to use with the neural net. Default: natural\n\
             Values: natural, anime, bilinear, waifu2x\n\
             Waifu2x upscale: waifu2x-anime (anime/cartoon), waifu2x-photo (photo denoise+upscale)\n\
             Waifu2x enhance (no upscale): waifu2x-enhance, waifu2x-enhance-anime, waifu2x-enhance-photo\n\
             Waifu2x fine-tuned (noise-level + scale): waifu2x-noise{0..3}-scale{1..4}\n\
             Scales 3× and 4× use iterative 2× passes; NCNN-Vulkan backend used when available.\n\
             Real-ESRGAN variants: real-esrgan (×4 photos), real-esrgan-anime (×4 anime), real-esrgan-x2 (×2 photos)\n\
             Examples: waifu2x-anime, waifu2x-enhance, waifu2x-noise2-scale4, real-esrgan",
		)
		.short("p")
		.long("parameters")
		.value_name("PARAMETERS")
		.possible_values(&[
			"natural",
			"anime",
			"bilinear",
			"waifu2x",
			"waifu2x-anime",
			"waifu2x-photo",
			"waifu2x-enhance",
			"waifu2x-enhance-anime",
			"waifu2x-enhance-photo",
			"waifu2x-noise0-scale1",
			"waifu2x-noise0-scale2",
			"waifu2x-noise0-scale3",
			"waifu2x-noise0-scale4",
			"waifu2x-noise1-scale1",
			"waifu2x-noise1-scale2",
			"waifu2x-noise1-scale3",
			"waifu2x-noise1-scale4",
			"waifu2x-noise2-scale1",
			"waifu2x-noise2-scale2",
			"waifu2x-noise2-scale3",
			"waifu2x-noise2-scale4",
			"waifu2x-noise3-scale1",
			"waifu2x-noise3-scale2",
			"waifu2x-noise3-scale3",
			"waifu2x-noise3-scale4",
			"real-esrgan",
			"real-esrgan-anime",
			"real-esrgan-x2",
			"real-esrgan-x4",
		])
		.empty_values(false)
}

fn build_custom_arg() -> Arg<'static, 'static> {
	Arg::with_name("CUSTOM")
		.conflicts_with("PARAMETERS")
		.short("c")
		.long("custom")
		.value_name("PARAMETER_FILE")
		.help("Sets a custom parameter file to use with the neural net (.rsr, .onnx, or .pth)")
		.empty_values(false)
}

fn build_auto_detect_arg() -> Arg<'static, 'static> {
	Arg::with_name("AUTO_DETECT")
		.long("auto-detect")
		.help(
			"Auto-detect image type (photo/anime/illustration) and select the best model.\n\
             Ignored when --parameters or --custom is also specified.",
		)
		.takes_value(false)
}

fn build_bilinear_factor_arg() -> Arg<'static, 'static> {
	Arg::with_name("BILINEAR_FACTOR")
		.short("f")
		.long("factor")
		.help("The integer upscaling factor used if bilinear upscaling is performed. Default 4")
		.empty_values(false)
}

fn build_gpu_flag_arg() -> Arg<'static, 'static> {
	Arg::with_name("GPU")
		.long("gpu")
		.help("Enable GPU acceleration if available (Metal on macOS, CUDA on Linux/Windows)")
		.takes_value(false)
}

fn build_format_arg() -> Arg<'static, 'static> {
	Arg::with_name("FORMAT")
		.long("format")
		.help("Output image format. Auto-detected from extension if omitted.")
		.value_name("FORMAT")
		.possible_values(&["png", "jpeg", "webp"])
		.empty_values(false)
}

fn build_quality_arg() -> Arg<'static, 'static> {
	Arg::with_name("QUALITY")
		.long("quality")
		.help("JPEG output quality (1–100, default: 85)")
		.value_name("QUALITY")
		.empty_values(false)
}

fn build_tile_size_arg() -> Arg<'static, 'static> {
	Arg::with_name("TILE_SIZE")
		.long("tile-size")
		.help(
			"Tile size in pixels for tiled upscaling of large images (default: 512). \
             Images larger than 4 MP are automatically processed in tiles; this option \
             also controls the tile size when tiling is triggered.",
		)
		.value_name("TILE_SIZE")
		.empty_values(false)
}

fn build_progressive_arg() -> Arg<'static, 'static> {
	Arg::with_name("PROGRESSIVE")
		.long("progressive")
		.help(
			"Enable progressive multi-stage upscaling for very low-resolution sources \
             (under 128px on either dimension). Instead of a single 4× pass the image \
             is first pre-upscaled 2× with a high-quality Lanczos filter, processed by \
             the neural network (producing an 8× intermediate), then downscaled back to \
             the standard 4× output size. This two-stage approach often yields sharper \
             results on tiny thumbnails and pixel-art sources.",
		)
		.takes_value(false)
}

fn build_auto_enhance_arg() -> Arg<'static, 'static> {
	Arg::with_name("AUTO_ENHANCE")
		.long("auto-enhance")
		.help(
			"Content-aware upscaling: analyse the image for faces, text, edges, \
			 and flat-color regions, then apply the best model per region. \
			 Text/edge regions receive extra sharpening. Overrides --parameters \
			 and --auto-detect.",
		)
		.takes_value(false)
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

fn build_batch_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("batch")
		.about("Batch process images. Sub-commands: start, resume, status, list")
		.setting(AppSettings::SubcommandsNegateReqs)
		.arg(
			Arg::with_name("INPUT_DIR")
				.help("Input directory containing images to upscale (legacy positional usage)")
				.index(1),
		)
		.arg(
			Arg::with_name("OUTPUT_DIR")
				.help("Output directory for upscaled images (legacy positional usage)")
				.index(2),
		)
		.arg(
			Arg::with_name("PARAMETERS")
				.help("Sets which built-in parameters to use with the neural net")
				.short("p")
				.long("parameters")
				.value_name("PARAMETERS")
				.possible_values(&["natural", "anime", "bilinear"])
				.empty_values(false),
		)
		.arg(
			Arg::with_name("CUSTOM")
				.conflicts_with("PARAMETERS")
				.short("c")
				.long("custom")
				.value_name("PARAMETER_FILE")
				.help("Sets a custom parameter file to use with the neural net (.rsr, .onnx, or .pth)")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("FACTOR")
				.short("f")
				.long("factor")
				.help("The integer upscaling factor. Default: 4")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("RECURSIVE")
				.short("r")
				.long("recursive")
				.help("Process images in subdirectories recursively")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("PATTERN")
				.short("g")
				.long("glob")
				.help("Glob pattern for matching image files")
				.value_name("PATTERN")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("SEQUENTIAL")
				.short("s")
				.long("sequential")
				.help("Process images sequentially instead of in parallel")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("SKIP_EXISTING")
				.short("k")
				.long("skip-existing")
				.help("Skip images that already exist in the output directory")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("THREADS")
				.short("t")
				.long("threads")
				.help("Number of threads to use for parallel processing (default: all available)")
				.value_name("N")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("CHUNK_SIZE")
				.long("chunk-size")
				.help("Number of images to process per batch (default: 10)")
				.value_name("SIZE")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("BATCH_ID")
				.long("batch-id")
				.help("Assign a named ID to this batch run (auto-generates a UUID if omitted)")
				.value_name("ID")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("RESUME")
				.long("resume")
				.help("Resume an interrupted batch by its batch ID (reads the checkpoint from the output directory)")
				.value_name("BATCH_ID")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("ERRORS_LOG")
				.long("errors-log")
				.help("Path for the error log (default: <OUTPUT_DIR>/batch_errors.log)")
				.value_name("FILE")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("WEBHOOK")
				.long("webhook")
				.help("HTTP URL to POST a JSON completion notification to when the batch finishes")
				.value_name("URL")
				.empty_values(false),
		)
		.subcommand(build_batch_start_subcommand())
		.subcommand(build_batch_resume_subcommand())
		.subcommand(build_batch_status_id_subcommand())
		.subcommand(build_batch_list_subcommand())
}

fn build_batch_start_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("start")
		.about("Start a new batch job with checkpoint tracking (~/.srgan-rust/checkpoints/)")
		.arg(
			Arg::with_name("input-dir")
				.long("input-dir")
				.short("i")
				.required(true)
				.value_name("DIR")
				.help("Input directory containing images to upscale"),
		)
		.arg(
			Arg::with_name("output-dir")
				.long("output-dir")
				.short("o")
				.required(true)
				.value_name("DIR")
				.help("Output directory for upscaled images"),
		)
		.arg(
			Arg::with_name("model")
				.long("model")
				.short("m")
				.value_name("MODEL")
				.possible_values(&["natural", "anime", "bilinear"])
				.default_value("natural")
				.help("Model to use for upscaling"),
		)
		.arg(
			Arg::with_name("scale")
				.long("scale")
				.short("s")
				.value_name("N")
				.default_value("4")
				.help("Integer upscaling factor"),
		)
		.arg(
			Arg::with_name("recursive")
				.long("recursive")
				.short("r")
				.help("Process images in subdirectories recursively")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("sequential")
				.long("sequential")
				.help("Process images sequentially instead of in parallel")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("threads")
				.long("threads")
				.short("t")
				.value_name("N")
				.help("Number of parallel threads (default: all available)"),
		)
}

fn build_batch_resume_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("resume")
		.about("Resume an interrupted batch job by ID")
		.arg(
			Arg::with_name("BATCH_ID")
				.help("Batch job ID returned by 'batch start'")
				.required(true)
				.index(1),
		)
}

fn build_batch_status_id_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("status")
		.about("Show progress and ETA for a batch job by ID")
		.arg(
			Arg::with_name("BATCH_ID")
				.help("Batch job ID")
				.required(true)
				.index(1),
		)
}

fn build_batch_list_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("list")
		.about("List all batch job checkpoints in ~/.srgan-rust/checkpoints/")
}

fn build_batch_status_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("batch-status")
		.about("Show progress of a running or interrupted batch job")
		.arg(
			Arg::with_name("DIR")
				.help("Output directory containing the checkpoint file (default: current directory)")
				.index(1),
		)
		.arg(
			Arg::with_name("BATCH_ID")
				.long("batch-id")
				.help("Batch ID to look up (scans the directory for any checkpoint if omitted)")
				.value_name("ID")
				.empty_values(false),
		)
}

fn build_classify_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("classify")
		.about("Detect the type of an image and recommend the best upscaling model")
		.arg(
			Arg::with_name("IMAGE")
				.help("Image file to classify")
				.required(true)
				.index(1),
		)
		.arg(
			Arg::with_name("json")
				.long("json")
				.help("Output result as JSON")
				.takes_value(false),
		)
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

fn build_upscale_gpu_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("upscale-gpu")
		.about("Upscale images using GPU acceleration")
		.arg(
			Arg::with_name("input")
				.help("Input image to upscale")
				.required(true)
				.index(1),
		)
		.arg(
			Arg::with_name("output")
				.help("Output file (.png recommended)")
				.required(true)
				.index(2),
		)
		.arg(
			Arg::with_name("network")
				.help("Network to use (natural, anime, bilinear)")
				.short("n")
				.long("network")
				.default_value("natural")
				.possible_values(&["natural", "anime", "bilinear"]),
		)
		.arg(
			Arg::with_name("gpu")
				.help("GPU backend to use (auto, cuda, opencl, metal, vulkan, cpu)")
				.short("g")
				.long("gpu")
				.default_value("auto")
				.possible_values(&["auto", "cuda", "opencl", "metal", "vulkan", "cpu"]),
		)
}

fn build_list_gpus_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("list-gpus")
		.about("List available GPU devices and backends")
}

fn build_benchmark_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("benchmark")
		.about("Benchmark model performance, or compare quality across models")
		.long_about(
			"Performance mode (default): runs synthetic-image timing benchmarks.\n\n\
			 Quality-comparison mode (when --output-dir is given): upscales a real\n\
			 input image with each model, saves the outputs, computes PSNR / SSIM\n\
			 vs the bilinear baseline, and writes benchmark_report.json +\n\
			 benchmark_report.html (self-contained with comparison sliders).\n\n\
			 Examples:\n\
			   srgan-rust benchmark\n\
			   srgan-rust benchmark --models natural,anime --iterations 5\n\
			   srgan-rust benchmark --input photo.jpg --models natural,anime,bilinear \\\n\
			       --output-dir ./bench_out/ --scale 4",
		)
		.arg(
			Arg::with_name("input")
				.help("Optional real input image (performance mode only; use --input for quality mode)")
				.required(false)
				.index(1),
		)
		.arg(
			Arg::with_name("input-img")
				.help("Input image for quality-comparison mode (required with --output-dir)")
				.long("input")
				.value_name("IMAGE"),
		)
		.arg(
			Arg::with_name("output-dir")
				.help("Output directory for upscaled images + HTML/JSON report (enables quality mode)")
				.long("output-dir")
				.value_name("DIR"),
		)
		.arg(
			Arg::with_name("scale")
				.help("Upscaling factor for quality-comparison mode (2 or 4, default 4)")
				.long("scale")
				.short("s")
				.possible_values(&["2", "4"])
				.default_value("4"),
		)
		.arg(
			Arg::with_name("iterations")
				.help("Number of iterations (performance mode)")
				.short("i")
				.long("iterations")
				.default_value("10"),
		)
		.arg(
			Arg::with_name("models")
				.help("Comma-separated list of models to benchmark")
				.short("m")
				.long("models")
				.default_value("natural,anime,bilinear"),
		)
		.arg(
			Arg::with_name("warmup")
				.help("Number of warmup iterations (performance mode)")
				.short("w")
				.long("warmup")
				.default_value("2"),
		)
		.arg(
			Arg::with_name("output")
				.help("Output results file for performance mode (JSON)")
				.short("o")
				.long("output"),
		)
		.arg(
			Arg::with_name("compare")
				.help("Compare with previous performance results file")
				.short("c")
				.long("compare"),
		)
}

fn build_parallel_benchmark_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("parallel-benchmark")
		.about("Benchmark parallel processing performance")
		.arg(
			Arg::with_name("BATCH_SIZES")
				.help("Comma-separated list of batch sizes to test")
				.short("b")
				.long("batch-sizes")
				.default_value("10,50,100")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("THREAD_COUNTS")
				.help("Comma-separated list of thread counts to test")
				.short("t")
				.long("thread-counts")
				.default_value("1,2,4,8")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("IMAGE_SIZE")
				.help("Size of test images (e.g., 256 for 256x256)")
				.short("s")
				.long("image-size")
				.default_value("256")
				.empty_values(false),
		)
		.arg(
			Arg::with_name("MODEL")
				.help("Model to use for benchmarking")
				.short("m")
				.long("model")
				.possible_values(&["natural", "anime", "bilinear"])
				.default_value("natural")
				.empty_values(false),
		)
}

fn build_generate_config_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("generate-config")
		.about("Generate a training configuration file")
		.arg(
			Arg::with_name("output")
				.help("Output configuration file (.toml)")
				.required(true)
				.index(1),
		)
		.arg(
			Arg::with_name("preset")
				.help("Configuration preset")
				.short("p")
				.long("preset")
				.possible_values(&["basic", "advanced", "high-quality", "fast"])
				.default_value("basic"),
		)
}

fn build_profile_memory_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("profile-memory")
		.about("Profile memory usage during upscaling")
		.arg(
			Arg::with_name("input")
				.help("Input image to upscale")
				.required(true)
				.index(1),
		)
		.arg(
			Arg::with_name("model")
				.help("Model to use for upscaling")
				.short("m")
				.long("model")
				.possible_values(&["natural", "anime"])
				.default_value("natural"),
		)
		.arg(
			Arg::with_name("output")
				.help("Output image file")
				.short("o")
				.long("output"),
		)
		.arg(
			Arg::with_name("report")
				.help("Memory report output file")
				.short("r")
				.long("report")
				.default_value("memory_profile.txt"),
		)
		.arg(
			Arg::with_name("interval")
				.help("Sampling interval in milliseconds")
				.short("i")
				.long("interval")
				.default_value("100"),
		)
}

fn build_analyze_memory_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("analyze-memory")
		.about("Analyze memory usage of any command")
		.arg(
			Arg::with_name("command")
				.help("Command to analyze")
				.required(true)
				.index(1)
				.possible_values(&["upscale", "downscale", "batch", "train"]),
		)
		.arg(
			Arg::with_name("args")
				.help("Arguments for the command")
				.multiple(true)
				.required(true),
		)
		.arg(
			Arg::with_name("report")
				.help("Memory analysis report file")
				.short("r")
				.long("report")
				.default_value("memory_analysis.txt"),
		)
		.arg(
			Arg::with_name("interval")
				.help("Sampling interval in milliseconds")
				.short("i")
				.long("interval")
				.default_value("100"),
		)
}

fn build_download_models_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("download-models")
		.about("Fetch Real-ESRGAN ONNX weights and built-in models to disk")
		.long_about(
			"Download pre-trained model weights to the local models directory.\n\n\
			 By default, downloads all available models (built-in .rsr + ONNX).\n\
			 Use --name to download a specific model.\n\
			 Use --url to override the download URL (e.g. for a local mirror).\n\
			 If the URL is unreachable, synthetic test weights are generated.\n\n\
			 Examples:\n\
			 \x20 srgan download-models --list\n\
			 \x20 srgan download-models\n\
			 \x20 srgan download-models --name real-esrgan-x4\n\
			 \x20 srgan download-models --url http://localhost:8080/models/",
		)
		.arg(
			Arg::with_name("name")
				.help("Download only this model (omit to download all)")
				.short("n")
				.long("name")
				.value_name("MODEL"),
		)
		.arg(
			Arg::with_name("dir")
				.help("Directory to save models (default: ~/.srgan/models/)")
				.short("d")
				.long("dir")
				.value_name("DIR"),
		)
		.arg(
			Arg::with_name("url")
				.help("Override the download URL for ONNX models")
				.long("url")
				.value_name("URL"),
		)
		.arg(
			Arg::with_name("list")
				.help("List all available models and exit")
				.short("l")
				.long("list")
				.takes_value(false),
		)
}

fn build_download_model_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("download-model")
		.about("Extract a built-in SRGAN model to disk (or list available models)")
		.arg(
			Arg::with_name("name")
				.help("Model name to extract: natural (default), anime")
				.short("n")
				.long("name")
				.value_name("NAME")
				.default_value("natural"),
		)
		.arg(
			Arg::with_name("dir")
				.help("Directory to save the model (default: ~/.srgan-rust/models/)")
				.short("d")
				.long("dir")
				.value_name("DIR"),
		)
		.arg(
			Arg::with_name("list")
				.help("List available built-in models and exit")
				.short("l")
				.long("list")
				.takes_value(false),
		)
}

fn build_models_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("models")
		.about("Manage pre-trained SRGAN models")
		.subcommand(
			SubCommand::with_name("list")
				.about("List all available models (built-in + custom)"),
		)
		.subcommand(
			SubCommand::with_name("download")
				.about("Download a pre-trained model")
				.arg(
					Arg::with_name("name")
						.help("Model name: natural, anime, face, 2x")
						.required(true)
						.index(1),
				)
				.arg(
					Arg::with_name("dir")
						.help("Directory to save the model (default: ~/.srgan/models/)")
						.short("d")
						.long("dir")
						.value_name("DIR"),
				),
		)
		.subcommand(
			SubCommand::with_name("add")
				.about("Register a custom model in the plugin registry (~/.srgan/models/)")
				.arg(
					Arg::with_name("name")
						.help("Short identifier for the model (e.g. myphoto4x)")
						.short("n")
						.long("name")
						.value_name("NAME")
						.required(true),
				)
				.arg(
					Arg::with_name("display_name")
						.help("Human-readable display name (defaults to <name> if omitted)")
						.long("display-name")
						.value_name("DISPLAY_NAME"),
				)
				.arg(
					Arg::with_name("model_type")
						.help("Model architecture type: esrgan, waifu2x, custom")
						.short("t")
						.long("type")
						.value_name("TYPE")
						.default_value("custom"),
				)
				.arg(
					Arg::with_name("scale")
						.help("Scale factor(s), comma-separated (e.g. 4 or 2,4)")
						.short("s")
						.long("scale")
						.value_name("SCALE")
						.default_value("4"),
				)
				.arg(
					Arg::with_name("description")
						.help("Optional description")
						.long("desc")
						.value_name("DESCRIPTION")
						.default_value(""),
				)
				.arg(
					Arg::with_name("weights")
						.help("Absolute path to the model weights file")
						.short("w")
						.long("weights")
						.value_name("PATH")
						.required(true),
				),
		)
		.subcommand(
			SubCommand::with_name("remove")
				.about("Remove a custom model from the registry")
				.arg(
					Arg::with_name("name")
						.help("Name of the custom model to remove")
						.required(true)
						.index(1),
				),
		)
}

fn build_compare_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("compare")
		.about("Compare an original image with an upscaled/processed image (PSNR, SSIM, histogram)")
		.long_about(
			"Computes PSNR, SSIM, file-size ratio, and a pixel-difference histogram\n\
			 between ORIGINAL and UPSCALED.  Also saves a side-by-side centre-crop\n\
			 comparison image.\n\n\
			 Examples:\n  srgan-rust compare original.png upscaled.png\n  \
			 srgan-rust compare --original original.png --upscaled upscaled.png --format json\n  \
			 srgan-rust compare original.png upscaled.png --output comparison.jpg",
		)
		.arg(
			Arg::with_name("INPUT")
				.help("Original image (positional)")
				.index(1),
		)
		.arg(
			Arg::with_name("UPSCALED_POS")
				.help("Upscaled image (positional)")
				.index(2),
		)
		.arg(
			Arg::with_name("original")
				.help("Original image path")
				.long("original")
				.value_name("PATH"),
		)
		.arg(
			Arg::with_name("upscaled")
				.help("Upscaled image path")
				.long("upscaled")
				.value_name("PATH"),
		)
		.arg(
			Arg::with_name("OUTPUT")
				.help("Path for the side-by-side comparison image (default: comparison.jpg)")
				.short("o")
				.long("output")
				.value_name("FILE"),
		)
		.arg(
			Arg::with_name("format")
				.help("Output format: text (default) or json")
				.long("format")
				.value_name("FORMAT")
				.possible_values(&["text", "json"])
				.default_value("text"),
		)
}

fn build_completions_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("completions")
		.about("Generate shell completion scripts")
		.long_about(
			"Prints a shell completion script to stdout.\n\n\
			 Examples:\n  \
			 srgan-rust completions bash >> ~/.bash_completion\n  \
			 srgan-rust completions zsh  > ~/.zfunc/_srgan-rust\n  \
			 srgan-rust completions fish > ~/.config/fish/completions/srgan-rust.fish",
		)
		.arg(
			Arg::with_name("shell")
				.help("Target shell")
				.required(true)
				.index(1)
				.possible_values(&["bash", "zsh", "fish", "powershell"]),
		)
}

fn build_server_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("server")
		.about("Start the SRGAN web API server")
		.arg(
			Arg::with_name("host")
				.help("Host address to bind to")
				.long("host")
				.default_value("127.0.0.1"),
		)
		.arg(
			Arg::with_name("port")
				.help("Port to listen on")
				.short("p")
				.long("port")
				.default_value("8080"),
		)
		.arg(
			Arg::with_name("model")
				.help("Path to a custom model file (.rsr)")
				.short("m")
				.long("model"),
		)
		.arg(
			Arg::with_name("api-key")
				.help("API key for request authentication")
				.long("api-key"),
		)
		.arg(
			Arg::with_name("rate-limit")
				.help("Max requests per minute")
				.long("rate-limit")
				.default_value("60"),
		)
		.arg(
			Arg::with_name("max-size")
				.help("Maximum upload size (e.g. 50MB, 1GB)")
				.long("max-size")
				.default_value("50MB"),
		)
		.arg(
			Arg::with_name("no-cache")
				.help("Disable response caching")
				.long("no-cache")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("cache-ttl")
				.help("Cache TTL in seconds")
				.long("cache-ttl")
				.default_value("3600"),
		)
		.arg(
			Arg::with_name("no-cors")
				.help("Disable CORS headers")
				.long("no-cors")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("no-logging")
				.help("Disable request logging")
				.long("no-logging")
				.takes_value(false),
		)
}

fn build_video_subcommand() -> App<'static, 'static> {
	SubCommand::with_name("video")
		.about("Upscale a video file frame by frame")
		.arg(
			Arg::with_name("input")
				.help("Input video file path")
				.long("input")
				.short("i")
				.required(true)
				.takes_value(true),
		)
		.arg(
			Arg::with_name("output")
				.help("Output video file path")
				.long("output")
				.short("o")
				.required(true)
				.takes_value(true),
		)
		.arg(
			Arg::with_name("parameters")
				.help("The name of a parameter set to use")
				.short("p")
				.long("parameters")
				.takes_value(true)
				.default_value("natural")
				.possible_values(&["natural", "anime"]),
		)
		.arg(
			Arg::with_name("custom")
				.help("Path to a custom model file (.rsr)")
				.short("c")
				.long("custom")
				.takes_value(true),
		)
		.arg(
			Arg::with_name("scale")
				.help("Upscaling factor")
				.long("scale")
				.takes_value(true)
				.default_value("4")
				.possible_values(&["2", "4"]),
		)
		.arg(
			Arg::with_name("fps")
				.help("Override output frame rate")
				.long("fps")
				.takes_value(true),
		)
		.arg(
			Arg::with_name("quality")
				.help("Output quality preset or CRF value")
				.long("quality")
				.short("q")
				.takes_value(true)
				.default_value("medium"),
		)
		.arg(
			Arg::with_name("codec")
				.help("Output video codec")
				.long("codec")
				.takes_value(true)
				.default_value("h264")
				.possible_values(&["h264", "h265", "vp9", "av1", "prores"]),
		)
		.arg(
			Arg::with_name("preserve-audio")
				.help("Preserve the original audio track (default: true)")
				.long("preserve-audio")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("no-audio")
				.help("Strip audio from the output")
				.long("no-audio")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("parallel")
				.help("Number of frames to process in parallel")
				.long("parallel")
				.takes_value(true)
				.default_value("4"),
		)
		.arg(
			Arg::with_name("temp-dir")
				.help("Temporary directory for extracted frames")
				.long("temp-dir")
				.takes_value(true),
		)
		.arg(
			Arg::with_name("start")
				.help("Start time (e.g. 00:01:30 or 90)")
				.long("start")
				.takes_value(true),
		)
		.arg(
			Arg::with_name("duration")
				.help("Duration to process (e.g. 00:00:30 or 30)")
				.long("duration")
				.takes_value(true),
		)
		.arg(
			Arg::with_name("preview")
				.help("Generate a preview frame before processing")
				.long("preview")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("preview-only")
				.help("Only generate a preview frame, skip full processing")
				.long("preview-only")
				.takes_value(false),
		)
		.arg(
			Arg::with_name("preview-time")
				.help("Timestamp for preview frame extraction")
				.long("preview-time")
				.takes_value(true),
		)
		.arg(
			Arg::with_name("overwrite")
				.help("Overwrite existing output files")
				.long("overwrite")
				.takes_value(false),
		)
}
