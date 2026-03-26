//! `srgan-rust models` subcommand — list and download pre-trained models.
//!
//! Usage:
//!   srgan-rust models list
//!   srgan-rust models download natural
//!   srgan-rust models download face --dir /custom/path

use clap::ArgMatches;
use log::info;

use crate::error::{Result, SrganError};
use crate::model_downloader;

/// Dispatch `models` subcommands.
pub fn models_command(app_m: &ArgMatches) -> Result<()> {
    match app_m.subcommand() {
        ("list", _) => models_list(),
        ("download", Some(sub)) => models_download(sub),
        _ => {
            // No subcommand: print help
            models_list()
        }
    }
}

/// `srgan-rust models list`
fn models_list() -> Result<()> {
    println!("Available models:");
    println!(
        "  {:<12}  {:<10}  {:<10}  {}",
        "NAME", "SCALE", "SOURCE", "DESCRIPTION"
    );
    println!("  {}", "-".repeat(70));
    for (name, desc, scale, source) in model_downloader::list_available_models() {
        println!("  {:<12}  {:<10}  {:<10}  {}", name, format!("{}×", scale), source, desc);
    }
    Ok(())
}

/// `srgan-rust models download <name> [--dir <dir>]`
fn models_download(sub_m: &ArgMatches) -> Result<()> {
    let name = sub_m
        .value_of("name")
        .ok_or_else(|| SrganError::InvalidParameter("Model name is required".to_string()))?;

    let dest_dir = if let Some(dir) = sub_m.value_of("dir") {
        std::path::PathBuf::from(dir)
    } else {
        model_downloader::default_models_dir()
    };

    info!("Downloading model '{}' to {}", name, dest_dir.display());
    println!(
        "Downloading model '{}' → {}",
        name,
        dest_dir.display()
    );

    let path = model_downloader::download_model(name, &dest_dir)?;

    println!("Saved to: {}", path.display());
    println!();
    println!("Use with:");
    println!(
        "  srgan-rust --custom {} input.png output.png",
        path.display()
    );

    Ok(())
}
