extern crate srgan_rust;
#[macro_use]
extern crate log;

use srgan_rust::{cli, commands, logging};

fn main() {
	// Initialize simple logger for CLI output
	logging::init_simple_logger();
	
	let app_m = cli::build_cli();

	let result = match app_m.subcommand() {
		("train", Some(sub_m)) => commands::train(sub_m),
		("train_prescaled", Some(sub_m)) => commands::train_prescaled(sub_m),
		("batch", Some(sub_m)) => commands::batch_upscale(sub_m),
		("downscale", Some(sub_m)) => commands::downscale(sub_m),
		("psnr", Some(sub_m)) => commands::psnr(sub_m),
		("quantise", Some(sub_m)) => commands::quantise(sub_m),
		("set_width", Some(sub_m)) => commands::set_width(sub_m),
		("upscale-gpu", Some(sub_m)) => commands::upscale_gpu(sub_m),
		("list-gpus", Some(sub_m)) => commands::list_gpu_devices(sub_m),
		("benchmark", Some(sub_m)) => commands::benchmark(sub_m),
		("parallel-benchmark", Some(sub_m)) => commands::run_parallel_benchmark(sub_m),
		("generate-config", Some(sub_m)) => commands::generate_config(sub_m),
		("profile-memory", Some(sub_m)) => {
			match sub_m.value_of("input") {
				Some(input) => {
					let model = sub_m.value_of("model");
					let output = sub_m.value_of("output");
					let report = sub_m.value_of("report");
					let interval = sub_m.value_of("interval")
						.and_then(|s| s.parse().ok())
						.unwrap_or(100);
					commands::profile_memory_command(input, model, output, report, interval)
				},
				None => Err(srgan_rust::error::SrganError::InvalidParameter("Missing input parameter".to_string()))
			}
		},
		("analyze-memory", Some(sub_m)) => {
			match sub_m.value_of("command") {
				Some(command) => {
					let args: Vec<&str> = sub_m.values_of("args")
						.map(|v| v.collect())
						.unwrap_or_else(Vec::new);
					let report = sub_m.value_of("report");
					let interval = sub_m.value_of("interval")
						.and_then(|s| s.parse().ok())
						.unwrap_or(100);
					commands::analyze_memory_usage(command, args, report, interval)
				},
				None => Err(srgan_rust::error::SrganError::InvalidParameter("Missing command parameter".to_string()))
			}
		},
		_ => commands::upscale(&app_m),
	};

	if let Err(err) = result {
		error!("Error: {}", err);
		std::process::exit(1);
	}
}
