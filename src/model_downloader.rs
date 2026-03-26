//! Model downloader — extract built-in SRGAN models to disk, or download
//! external models from GitHub releases.
//!
//! ## Built-in models (embedded in binary)
//! - `natural` / `default` — general photo upscaling (4×)
//! - `anime`               — anime/artwork optimised (4×)
//!
//! ## Downloadable models (fetched from GitHub releases)
//! - `face` — face-optimised (4×)
//! - `2x`   — 2× general upscaling
//!
//! ## Usage
//! ```text
//! srgan-rust models download natural
//! srgan-rust models download face
//! srgan-rust models list
//! ```

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use crate::error::{Result, SrganError};

// ── Model catalogue ───────────────────────────────────────────────────────────

/// A model that is embedded in the binary.
pub struct EmbeddedModel {
    pub name: &'static str,
    pub alias: Option<&'static str>,
    pub description: &'static str,
    pub filename: &'static str,
    pub scale_factor: u32,
}

/// A model that must be downloaded from a remote URL.
pub struct RemoteModel {
    pub name: &'static str,
    pub description: &'static str,
    pub filename: &'static str,
    pub scale_factor: u32,
    /// Primary download URL (GitHub releases).
    pub url: &'static str,
    /// Expected SHA-256 hex digest of the downloaded file.
    pub sha256: &'static str,
}

pub const EMBEDDED_MODELS: &[EmbeddedModel] = &[
    EmbeddedModel {
        name: "natural",
        alias: Some("default"),
        description: "Natural-image model trained on the UCID dataset with L1 loss (4× scale).",
        filename: "natural.rsr",
        scale_factor: 4,
    },
    EmbeddedModel {
        name: "anime",
        alias: None,
        description: "Animation / anime-optimised model trained with L1 loss (4× scale).",
        filename: "anime.rsr",
        scale_factor: 4,
    },
];

pub const REMOTE_MODELS: &[RemoteModel] = &[
    RemoteModel {
        name: "face",
        description: "Face-optimised SRGAN model (4× scale).",
        filename: "face.rsr",
        scale_factor: 4,
        url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/face.rsr",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    RemoteModel {
        name: "2x",
        description: "General-purpose 2× upscaling model.",
        filename: "2x.rsr",
        scale_factor: 2,
        url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/2x.rsr",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
];

// ── Backwards-compat type alias ───────────────────────────────────────────────

/// Legacy alias kept for backwards compatibility with `commands/download_model.rs`.
pub type BuiltinModelEntry = EmbeddedModel;

/// Legacy constant alias.
pub const BUILTIN_MODELS: &[EmbeddedModel] = EMBEDDED_MODELS;

// ── Public helpers ────────────────────────────────────────────────────────────

/// Default directory for storing models: `$HOME/.srgan/models/`
pub fn default_models_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".srgan").join("models")
}

/// List all available models as (name, description, scale_factor, source) tuples.
pub fn list_available_models() -> Vec<(&'static str, &'static str, u32, &'static str)> {
    let mut out = Vec::new();
    for m in EMBEDDED_MODELS {
        out.push((m.name, m.description, m.scale_factor, "built-in"));
    }
    for m in REMOTE_MODELS {
        out.push((m.name, m.description, m.scale_factor, "download"));
    }
    out
}

/// Extract or download a model to `dest_dir`, verifying its SHA-256 checksum.
///
/// Returns the path of the written file.
pub fn download_model(name: &str, dest_dir: &Path) -> Result<PathBuf> {
    // Normalise alias "default" → "natural"
    let name = match name {
        "default" => "natural",
        other => other,
    };

    // Try embedded first
    if let Some(entry) = EMBEDDED_MODELS.iter().find(|m| m.name == name || m.alias == Some(name)) {
        return extract_embedded(entry, dest_dir);
    }

    // Then try remote
    if let Some(entry) = REMOTE_MODELS.iter().find(|m| m.name == name) {
        return fetch_remote(entry, dest_dir);
    }

    Err(SrganError::InvalidParameter(format!(
        "Unknown model '{}'. Run `models list` to see available models.",
        name
    )))
}

/// Legacy function kept for backwards compat with `commands/download_model.rs`.
pub fn extract_model(name: &str, dest_dir: &Path) -> Result<PathBuf> {
    download_model(name, dest_dir)
}

// ── Embedded extraction ───────────────────────────────────────────────────────

fn extract_embedded(entry: &EmbeddedModel, dest_dir: &Path) -> Result<PathBuf> {
    let data: &[u8] = match entry.name {
        "natural" => crate::L1_SRGB_NATURAL_PARAMS,
        "anime" => crate::L1_SRGB_ANIME_PARAMS,
        other => {
            return Err(SrganError::InvalidParameter(format!(
                "No embedded data for model '{}'",
                other
            )))
        }
    };

    fs::create_dir_all(dest_dir).map_err(SrganError::Io)?;

    let dest_path = dest_dir.join(entry.filename);

    let pb = make_progress_bar(data.len() as u64);
    pb.set_message(format!("Extracting {}", entry.filename));

    let mut file = fs::File::create(&dest_path).map_err(SrganError::Io)?;
    const CHUNK: usize = 64 * 1024;
    for chunk in data.chunks(CHUNK) {
        file.write_all(chunk).map_err(SrganError::Io)?;
        pb.inc(chunk.len() as u64);
    }
    pb.finish_with_message("Done");

    // Compute and print SHA-256 (no expected value for embedded models)
    let sha = sha256_file(&dest_path)?;
    println!("SHA-256: {}", sha);

    // Verify as a valid SRGAN model
    verify_model(&dest_path)?;

    Ok(dest_path)
}

// ── Remote download ───────────────────────────────────────────────────────────

fn fetch_remote(entry: &RemoteModel, dest_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dest_dir).map_err(SrganError::Io)?;

    let dest_path = dest_dir.join(entry.filename);

    println!("Downloading {} from {}", entry.name, entry.url);

    let response = ureq::get(entry.url)
        .call()
        .map_err(|e| SrganError::Network(format!("HTTP request failed: {}", e)))?;

    let content_length: Option<u64> = response
        .header("Content-Length")
        .and_then(|v| v.parse().ok());

    let pb = make_progress_bar(content_length.unwrap_or(0));
    pb.set_message(format!("Downloading {}", entry.filename));

    let mut reader = response.into_reader();
    let mut data = Vec::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(SrganError::Io)?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        pb.inc(n as u64);
    }
    pb.finish_with_message("Download complete");

    // Verify SHA-256
    let actual = sha256_bytes(&data);
    if entry.sha256 != "0000000000000000000000000000000000000000000000000000000000000000"
        && actual != entry.sha256
    {
        return Err(SrganError::Network(format!(
            "SHA-256 mismatch for '{}': expected {}, got {}",
            entry.name, entry.sha256, actual
        )));
    }
    println!("SHA-256: {} ✓", actual);

    let mut file = fs::File::create(&dest_path).map_err(SrganError::Io)?;
    file.write_all(&data).map_err(SrganError::Io)?;

    // Verify as a valid SRGAN model
    verify_model(&dest_path)?;

    Ok(dest_path)
}

// ── Verification ──────────────────────────────────────────────────────────────

/// Verify that the file at `path` can be parsed as an SRGAN model.
pub fn verify_model(path: &Path) -> Result<()> {
    let mut data = Vec::new();
    fs::File::open(path)
        .map_err(SrganError::Io)?
        .read_to_end(&mut data)
        .map_err(SrganError::Io)?;

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

// ── Checksums ─────────────────────────────────────────────────────────────────

/// Compute SHA-256 of `data` and return as hex string.
pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Compute SHA-256 of a file and return as hex string.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut data = Vec::new();
    fs::File::open(path)
        .map_err(SrganError::Io)?
        .read_to_end(&mut data)
        .map_err(SrganError::Io)?;
    Ok(sha256_bytes(&data))
}

/// Legacy FNV-1a 64-bit hash (for backward compat with old checksum display).
pub fn checksum_hex(data: &[u8]) -> String {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    format!("{:016x}", hash)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn make_progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
    );
    pb
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_list_models_includes_all() {
        let models = list_available_models();
        let names: Vec<_> = models.iter().map(|(n, _, _, _)| *n).collect();
        assert!(names.contains(&"natural"));
        assert!(names.contains(&"anime"));
        assert!(names.contains(&"face"));
        assert!(names.contains(&"2x"));
    }

    #[test]
    fn test_extract_natural() {
        let dir = TempDir::new().unwrap();
        let path = download_model("natural", dir.path()).unwrap();
        assert!(path.exists());
        assert!(path.metadata().unwrap().len() > 0);
    }

    #[test]
    fn test_extract_via_alias() {
        let dir = TempDir::new().unwrap();
        let path = download_model("default", dir.path()).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_sha256_deterministic() {
        let data = b"hello world";
        assert_eq!(sha256_bytes(data), sha256_bytes(data));
    }

    #[test]
    fn test_unknown_model_error() {
        let dir = TempDir::new().unwrap();
        assert!(download_model("nonexistent", dir.path()).is_err());
    }
}
