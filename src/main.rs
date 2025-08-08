extern crate srgan_rust;

use srgan_rust::{cli, commands};

fn main() {
	let app_m = cli::build_cli();

	let result = match app_m.subcommand() {
		("train", Some(sub_m)) => commands::train(sub_m),
		("train_prescaled", Some(sub_m)) => commands::train_prescaled(sub_m),
		("downscale", Some(sub_m)) => commands::downscale(sub_m),
		("psnr", Some(sub_m)) => commands::psnr(sub_m),
		("quantise", Some(sub_m)) => commands::quantise(sub_m),
		("set_width", Some(sub_m)) => commands::set_width(sub_m),
		_ => commands::upscale(&app_m),
	};

	if let Err(err) = result {
		eprintln!("Error: {}", err);
		std::process::exit(1);
	}
}
