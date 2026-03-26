//! Model downloader — extract built-in SRGAN models to disk for external use,
//! or download custom models from a URL.
//!
//! ## Usage
//! ```text
//! srgan-rust download-model --name natural
//! srgan-rust download-model --name anime --dir /custom/path
//! srgan-rust download-model --list
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};

use crate::error::{Result, SrganError};

// ── Known built-in models ─────────────────────────────────────────────────────

/// Metadata about a named built-in model.
pub struct BuiltinModelEntry {
    /// Short identifier used on the CLI (e.g. `"natural"`, `"anime"`).
    pub name: &'static str,
    /// Additional alias (e.g. `"default"` maps to `"natural"`).
    pub alias: Option<&'static str>,
    /// Human-readable description.
    pub description: &'static str,
    /// Filename to use when writing to disk.
    pub filename: &'static str,
    /// Upscaling factor this model was trained for.
    pub scale_factor: u32,
}

/// All built-in models that can be extracted.
pub const BUILTIN_MODELS: &[BuiltinModelEntry] = &[
    BuiltinModelEntry {
        name: "natural",
        alias: Some("default"),
        description: "Natural-image model trained on the UCID dataset with L1 loss (4× scale).",
        filename: "natural.rsr",
        scale_factor: 4,
    },
    BuiltinModelEntry {
        name: "anime",
        alias: None,
        description: "Animation / anime-optimised model trained with L1 loss (4× scale).",
        filename: "anime.rsr",
        scale_factor: 4,
    },
];

// ── Public helpers ────────────────────────────────────────────────────────────

/// Return the default directory for storing downloaded models:
/// `$HOME/.srgan-rust/models/`
pub fn default_models_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".srgan-rust").join("models")
}

/// List available built-in models as `(name, description)` pairs.
pub fn list_available_models() -> Vec<(&'static str, &'static str)> {
    BUILTIN_MODELS.iter().map(|m| (m.name, m.description)).collect()
}

/// Find a model entry by name or alias.
fn find_model(name: &str) -> Option<&'static BuiltinModelEntry> {
    BUILTIN_MODELS.iter().find(|m| {
        m.name == name || m.alias.map_or(false, |a| a == name)
    })
}

/// Extract a built-in model to `dest_dir` and return the path it was written to.
///
/// Shows a progress bar, then verifies the file is a valid SRGAN model.
pub fn extract_model(name: &str, dest_dir: &Path) -> Result<PathBuf> {
    let entry = find_model(name).ok_or_else(|| {
        SrganError::InvalidParameter(format!(
            "Unknown model '{}'. Run `download-model --list` to see available models.",
            name
        ))
    })?;

    // Pick the correct embedded bytes
    let data: &[u8] = match entry.name {
        "natural" => crate::L1_SRGB_NATURAL_PARAMS,
        "anime"   => crate::L1_SRGB_ANIME_PARAMS,
        other     => {
            return Err(SrganError::InvalidParameter(format!(
                "No embedded data for model '{}'", other
            )));
        }
    };

    fs::create_dir_all(dest_dir)
        .map_err(|e| SrganError::Io(e))?;

    let dest_path = dest_dir.join(entry.filename);

    // Write in chunks with a progress bar
    const CHUNK: usize = 64 * 1024; // 64 KiB
    let pb = ProgressBar::new(data.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Extracting {}", entry.filename));

    let mut file = fs::File::create(&dest_path)
        .map_err(|e| SrganError::Io(e))?;

    for chunk in data.chunks(CHUNK) {
        file.write_all(chunk).map_err(|e| SrganError::Io(e))?;
        pb.inc(chunk.len() as u64);
    }
    pb.finish_with_message("Done");

    // Verify the written file is a valid SRGAN model
    verify_model(&dest_path)?;

    Ok(dest_path)
}

/// Verify that the file at `path` is a parseable SRGAN model.
///
/// Returns `Ok(())` on success, or an error describing the problem.
pub fn verify_model(path: &Path) -> Result<()> {
    use std::io::Read;

    let mut data = Vec::new();
    fs::File::open(path)
        .map_err(|e| SrganError::Io(e))?
        .read_to_end(&mut data)
        .map_err(|e| SrganError::Io(e))?;

    if data.is_empty() {
        return Err(SrganError::Network(format!(
            "Model file is empty: {}",
            path.display()
        )));
    }

    crate::network_from_bytes(&data)
        .map_err(|e| SrganError::Network(format!("Model verification failed: {}", e)))?;

    Ok(())
}

/// Compute a simple 64-bit FNV-1a hash of `data`, returned as a hex string.
/// Used to give users a quick "fingerprint" they can compare.
pub fn checksum_hex(data: &[u8]) -> String {
    let mut hash: u64 = 14695981039346656037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_list_models() {
        let models = list_available_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|(n, _)| *n == "natural"));
        assert!(models.iter().any(|(n, _)| *n == "anime"));
    }

    #[test]
    fn test_find_model_by_alias() {
        assert!(find_model("default").is_some());
        assert!(find_model("natural").is_some());
        assert!(find_model("nonexistent").is_none());
    }

    #[test]
    fn test_extract_natural_model() {
        let dir = TempDir::new().unwrap();
        let path = extract_model("natural", dir.path()).unwrap();
        assert!(path.exists());
        assert!(path.metadata().unwrap().len() > 0);
    }

    #[test]
    fn test_extract_via_alias() {
        let dir = TempDir::new().unwrap();
        let path = extract_model("default", dir.path()).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = b"hello world";
        assert_eq!(checksum_hex(data), checksum_hex(data));
    }
}
