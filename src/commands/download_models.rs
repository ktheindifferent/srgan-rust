use clap::ArgMatches;
use log::info;

use crate::error::Result;
use crate::model_downloader;

/// `srgan-rust download-models` command — fetch Real-ESRGAN ONNX weights
/// (and optionally all built-in models) to disk.
pub fn download_models(app_m: &ArgMatches) -> Result<()> {
    // --list: print all available models
    if app_m.is_present("list") {
        println!("Available models (built-in + ONNX):\n");
        println!("  {:<24} {:<6} {:<14} {}", "NAME", "SCALE", "SOURCE", "DESCRIPTION");
        println!("  {}", "-".repeat(80));
        for (name, desc, scale, source) in model_downloader::list_all_models() {
            println!("  {:<24} {:>4}×  {:<14} {}", name, scale, source, desc);
        }
        return Ok(());
    }

    let dest_dir = if let Some(dir) = app_m.value_of("dir") {
        std::path::PathBuf::from(dir)
    } else {
        model_downloader::default_models_dir()
    };

    let url_override = app_m.value_of("url");

    // If a specific model name is given, download only that one
    if let Some(name) = app_m.value_of("name") {
        info!("Downloading model '{}' to {}", name, dest_dir.display());
        println!("Downloading model '{}' → {}\n", name, dest_dir.display());

        // Try ONNX models first, then fall back to the standard downloader
        let path = match model_downloader::download_onnx_model(name, &dest_dir.join("onnx"), url_override) {
            Ok(p) => p,
            Err(_) => model_downloader::download_model(name, &dest_dir)?,
        };

        println!("\nSaved to: {}", path.display());
        let fmt = model_downloader::detect_model_format(&path);
        println!("Format:   {}", fmt);
        println!("\nUse with:");
        println!("  srgan-rust --custom {} input.png output.png", path.display());
        return Ok(());
    }

    // No name specified — download all models
    info!("Downloading all models to {}", dest_dir.display());
    println!("Downloading all models → {}\n", dest_dir.display());

    let paths = model_downloader::download_all_models(&dest_dir, url_override)?;

    println!("\n--- Summary ---");
    for path in &paths {
        let fmt = model_downloader::detect_model_format(path);
        println!("  [{}] {}", fmt, path.display());
    }
    println!("\n{} models downloaded.", paths.len());

    Ok(())
}
