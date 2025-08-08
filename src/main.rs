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
		_ => commands::upscale(&app_m),
	};

	if let Err(err) = result {
		error!("Error: {}", err);
		std::process::exit(1);
	}
}
