//! `srgan-rust models` subcommand — list, add, remove, and download models.
//!
//! Usage:
//!   srgan-rust models list
//!   srgan-rust models add --name mymodel --type esrgan --scale 4 --weights /path/to/weights.bin
//!   srgan-rust models remove mymodel
//!   srgan-rust models download natural

use clap::ArgMatches;
use log::info;

use crate::error::{Result, SrganError};
use crate::model_downloader;
use crate::model_registry::{ModelRegistry, ModelType};

/// Dispatch `models` subcommands.
pub fn models_command(app_m: &ArgMatches) -> Result<()> {
    match app_m.subcommand() {
        ("list", _) => models_list(),
        ("add", Some(sub)) => models_add(sub),
        ("remove", Some(sub)) => models_remove(sub),
        ("download", Some(sub)) => models_download(sub),
        _ => models_list(),
    }
}

/// `srgan-rust models list`
fn models_list() -> Result<()> {
    let registry = ModelRegistry::load()?;

    println!("Registered models:");
    println!(
        "  {:<20}  {:<28}  {:<8}  {:<10}  {}",
        "NAME", "DISPLAY NAME", "TYPE", "SCALE(S)", "SOURCE"
    );
    println!("  {}", "-".repeat(85));

    for entry in registry.list() {
        let scales = entry
            .scale_factors
            .iter()
            .map(|s| format!("{}×", s))
            .collect::<Vec<_>>()
            .join(", ");
        let source = if entry.builtin { "built-in" } else { "custom" };
        println!(
            "  {:<20}  {:<28}  {:<8}  {:<10}  {}",
            entry.name,
            entry.display_name,
            entry.model_type,
            scales,
            source,
        );
    }

    let custom_count = registry.custom_models().count();
    println!();
    println!(
        "  {} built-in model(s), {} custom model(s)",
        registry.list().len() - custom_count,
        custom_count
    );
    println!(
        "  Custom manifests stored in: {}",
        crate::model_registry::registry_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "~/.srgan/models/".to_string())
    );

    Ok(())
}

/// `srgan-rust models add --name ... --type ... --scale ... --weights ...`
fn models_add(sub_m: &ArgMatches) -> Result<()> {
    let name = sub_m
        .value_of("name")
        .ok_or_else(|| SrganError::InvalidParameter("--name is required".to_string()))?;

    let display_name = sub_m.value_of("display_name").unwrap_or(name);

    let model_type: ModelType = sub_m
        .value_of("model_type")
        .unwrap_or("custom")
        .parse()?;

    let scale_str = sub_m.value_of("scale").unwrap_or("4");
    let scale_factors: Vec<u32> = scale_str
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| {
            SrganError::InvalidParameter(format!(
                "Invalid scale value '{}'. Expected integer(s), e.g. 4 or 2,4",
                scale_str
            ))
        })?;

    let description = sub_m.value_of("description").unwrap_or("");

    let weight_path = sub_m
        .value_of("weights")
        .ok_or_else(|| SrganError::InvalidParameter("--weights is required".to_string()))?;

    let mut registry = ModelRegistry::load()?;
    let manifest_path = registry.add(
        name,
        display_name,
        model_type,
        scale_factors.clone(),
        description,
        weight_path,
    )?;

    info!("Registered model '{}' in registry", name);
    println!("Model '{}' registered successfully.", name);
    println!("  Manifest: {}", manifest_path.display());
    println!("  Weights:  {}", weight_path);
    println!(
        "  Scale(s): {}",
        scale_factors
            .iter()
            .map(|s| format!("{}×", s))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();
    println!("Use with:");
    println!("  srgan-rust --custom {} input.png output.png", weight_path);

    Ok(())
}

/// `srgan-rust models remove <name>`
fn models_remove(sub_m: &ArgMatches) -> Result<()> {
    let name = sub_m
        .value_of("name")
        .ok_or_else(|| SrganError::InvalidParameter("Model name is required".to_string()))?;

    let mut registry = ModelRegistry::load()?;
    registry.remove(name)?;

    info!("Removed model '{}' from registry", name);
    println!("Model '{}' removed from registry.", name);

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
