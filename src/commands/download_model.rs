use clap::ArgMatches;
use log::info;

use crate::error::{Result, SrganError};
use crate::model_downloader;

/// `srgan-rust download-model` command
pub fn download_model(app_m: &ArgMatches) -> Result<()> {
    // --list: just print available models
    if app_m.is_present("list") {
        println!("Available built-in models:");
        for (name, desc, _size, _checksum) in model_downloader::list_available_models() {
            println!("  {:12}  {}", name, desc);
        }
        return Ok(());
    }

    let name = app_m.value_of("name").unwrap_or("natural");

    let dest_dir = if let Some(dir) = app_m.value_of("dir") {
        std::path::PathBuf::from(dir)
    } else {
        model_downloader::default_models_dir()
    };

    info!("Extracting model '{}' to {}", name, dest_dir.display());
    println!("Extracting model '{}' → {}", name, dest_dir.display());

    let path = model_downloader::extract_model(name, &dest_dir)?;

    // Show checksum so users can compare
    {
        use std::io::Read;
        let mut data = Vec::new();
        if std::fs::File::open(&path)
            .and_then(|mut f| f.read_to_end(&mut data))
            .is_ok()
        {
            println!("Checksum (FNV-1a): {}", model_downloader::checksum_hex(&data));
        }
    }

    println!("Saved to: {}", path.display());
    println!();
    println!("Use with:");
    println!("  srgan-rust --custom {} input.png output.png", path.display());

    Ok(())
}
