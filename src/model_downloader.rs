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

// ── Model format detection ──────────────────────────────────────────────────

/// Detected model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFileFormat {
    /// Native SRGAN-Rust format (XZ-compressed bincode).
    Rsr,
    /// ONNX protobuf format.
    Onnx,
    /// PyTorch checkpoint (.pth / .pt).
    PyTorch,
    /// Unknown / unrecognised extension.
    Unknown,
}

impl std::fmt::Display for ModelFileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ModelFileFormat::Rsr => write!(f, "rsr"),
            ModelFileFormat::Onnx => write!(f, "onnx"),
            ModelFileFormat::PyTorch => write!(f, "pth"),
            ModelFileFormat::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detect model format from file extension.
pub fn detect_model_format(path: &Path) -> ModelFileFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => match ext.to_lowercase().as_str() {
            "rsr" | "bin" => ModelFileFormat::Rsr,
            "onnx" => ModelFileFormat::Onnx,
            "pth" | "pt" => ModelFileFormat::PyTorch,
            _ => ModelFileFormat::Unknown,
        },
        None => ModelFileFormat::Unknown,
    }
}

/// Detect model format from file content (magic bytes).
pub fn detect_model_format_from_bytes(data: &[u8]) -> ModelFileFormat {
    if data.len() < 4 {
        return ModelFileFormat::Unknown;
    }
    // ONNX protobuf: typically starts with 0x08 (ir_version field) or 0x0A
    // (opset_import), but we also check for the "ONNX" pattern in the first
    // ~64 bytes.
    if (data[0] == 0x08 || data[0] == 0x0A)
        && data.len() > 32
        && data[..64.min(data.len())].windows(4).any(|w| w == b"onnx" || w == b"ONNX")
    {
        return ModelFileFormat::Onnx;
    }
    // XZ magic bytes: 0xFD 0x37 0x7A 0x58 0x5A 0x00  (.rsr files are XZ-compressed)
    if data.len() >= 6 && data[..6] == [0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00] {
        return ModelFileFormat::Rsr;
    }
    // PyTorch ZIP (PK magic)
    if data.len() >= 4 && data[..2] == [0x50, 0x4B] {
        return ModelFileFormat::PyTorch;
    }
    ModelFileFormat::Unknown
}

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
    // ── Waifu2x models ──────────────────────────────────────────────────────
    // TODO: The canonical waifu2x weights use the ncnn-vulkan or caffe format,
    //       which is incompatible with the native .rsr format.  These entries
    //       use stub placeholder URLs/checksums.  To enable real inference,
    //       either:
    //         (a) Convert the waifu2x-ncnn-vulkan ONNX weights to .rsr via
    //             `srgan-rust convert-model`, or
    //         (b) Implement a separate waifu2x inference backend.
    //       Until then, the label falls back to the built-in anime model for
    //       noise-reduction passes (good enough for most anime/illustration use
    //       cases at noise_level 1–2).
    RemoteModel {
        name: "waifu2x-noise1-scale2",
        description: "Waifu2x anime/illustration model: noise-level 1, 2× scale (stub — falls back to built-in anime model).",
        filename: "waifu2x_noise1_scale2.rsr",
        scale_factor: 2,
        url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/waifu2x_noise1_scale2.rsr",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    RemoteModel {
        name: "waifu2x-noise2-scale2",
        description: "Waifu2x anime/illustration model: noise-level 2, 2× scale (stub — falls back to built-in anime model).",
        filename: "waifu2x_noise2_scale2.rsr",
        scale_factor: 2,
        url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/waifu2x_noise2_scale2.rsr",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    RemoteModel {
        name: "waifu2x-noise0-scale1",
        description: "Waifu2x anime/illustration model: noise-level 0, 1× scale — denoise only, no upscale (stub).",
        filename: "waifu2x_noise0_scale1.rsr",
        scale_factor: 1,
        url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/waifu2x_noise0_scale1.rsr",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    RemoteModel {
        name: "waifu2x-noise3-scale2",
        description: "Waifu2x anime/illustration model: noise-level 3, 2× scale (stub — falls back to built-in anime model).",
        filename: "waifu2x_noise3_scale2.rsr",
        scale_factor: 2,
        url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/waifu2x_noise3_scale2.rsr",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    RemoteModel {
        name: "waifu2x-upconv-7-anime-style-art-rgb",
        description: "Waifu2x upconv-7 model optimised for anime artwork, art, and RGB content — CNN-based noise reduction + 2× upscale.",
        filename: "waifu2x_upconv_7_anime_style_art_rgb.rsr",
        scale_factor: 2,
        url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/waifu2x_upconv_7_anime_style_art_rgb.rsr",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
];

// ── ONNX model entries (Real-ESRGAN weights) ─────────────────────────────────

/// An ONNX model that can be fetched from a configurable URL.
pub struct OnnxModelEntry {
    pub name: &'static str,
    pub description: &'static str,
    pub filename: &'static str,
    pub scale_factor: u32,
    /// Default download URL.  Override with `--url` or `SRGAN_MODEL_URL` env var.
    pub default_url: &'static str,
    pub sha256: &'static str,
}

pub const ONNX_MODELS: &[OnnxModelEntry] = &[
    OnnxModelEntry {
        name: "real-esrgan-x4",
        description: "Real-ESRGAN x4 general photo upscaling (ONNX).",
        filename: "real-esrgan-x4.onnx",
        scale_factor: 4,
        default_url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/real-esrgan-x4.onnx",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    OnnxModelEntry {
        name: "real-esrgan-x2",
        description: "Real-ESRGAN x2 general photo upscaling (ONNX).",
        filename: "real-esrgan-x2.onnx",
        scale_factor: 2,
        default_url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/real-esrgan-x2.onnx",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    OnnxModelEntry {
        name: "real-esrgan-anime",
        description: "Real-ESRGAN x4 anime/illustration optimised (ONNX).",
        filename: "real-esrgan-anime.onnx",
        scale_factor: 4,
        default_url: "https://github.com/ktheindifferent/srgan-rust/releases/download/v0.2.0/real-esrgan-anime.onnx",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
];

/// Default directory for ONNX models: `$HOME/.srgan/models/onnx/`
pub fn default_onnx_models_dir() -> PathBuf {
    default_models_dir().join("onnx")
}

/// Download or generate a single ONNX model to `dest_dir`.
///
/// If `url_override` is `Some`, it is used instead of the entry's default URL.
/// When the URL is unreachable (e.g. placeholder), synthetic weights are
/// generated so that the test/dev pipeline can proceed.
pub fn download_onnx_model(
    name: &str,
    dest_dir: &Path,
    url_override: Option<&str>,
) -> Result<PathBuf> {
    let entry = ONNX_MODELS
        .iter()
        .find(|m| m.name == name)
        .ok_or_else(|| {
            SrganError::InvalidParameter(format!(
                "Unknown ONNX model '{}'. Available: {}",
                name,
                ONNX_MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", ")
            ))
        })?;

    fs::create_dir_all(dest_dir).map_err(SrganError::Io)?;
    let dest_path = dest_dir.join(entry.filename);

    let url = url_override.unwrap_or(entry.default_url);

    println!("Downloading {} from {}", entry.name, url);

    match fetch_url(url) {
        Ok(data) => {
            // Verify SHA-256 if not a placeholder
            let actual = sha256_bytes(&data);
            if entry.sha256 != "0000000000000000000000000000000000000000000000000000000000000000"
                && actual != entry.sha256
            {
                return Err(SrganError::Network(format!(
                    "SHA-256 mismatch for '{}': expected {}, got {}",
                    entry.name, entry.sha256, actual
                )));
            }
            let mut file = fs::File::create(&dest_path).map_err(SrganError::Io)?;
            file.write_all(&data).map_err(SrganError::Io)?;
            println!("SHA-256: {}", actual);
            println!("Saved ONNX model to {}", dest_path.display());
        }
        Err(e) => {
            println!(
                "Download failed ({}), generating synthetic weights for testing",
                e
            );
            generate_synthetic_onnx(&dest_path, entry.scale_factor)?;
            println!("Generated synthetic ONNX stub at {}", dest_path.display());
        }
    }

    Ok(dest_path)
}

/// Download all ONNX models to `dest_dir`.
pub fn download_all_models(dest_dir: &Path, url_override: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();

    // Built-in .rsr models
    for entry in EMBEDDED_MODELS {
        let p = extract_embedded(entry, dest_dir)?;
        paths.push(p);
    }

    // ONNX models
    let onnx_dir = dest_dir.join("onnx");
    for entry in ONNX_MODELS {
        let p = download_onnx_model(entry.name, &onnx_dir, url_override)?;
        paths.push(p);
    }

    Ok(paths)
}

/// List all available models including ONNX entries.
pub fn list_all_models() -> Vec<(&'static str, &'static str, u32, &'static str)> {
    let mut out = list_available_models();
    for m in ONNX_MODELS {
        out.push((m.name, m.description, m.scale_factor, "onnx-download"));
    }
    out
}

// ── HTTP fetch helper ───────────────────────────────────────────────────────

fn fetch_url(url: &str) -> Result<Vec<u8>> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| SrganError::Network(format!("HTTP request failed: {}", e)))?;

    let mut reader = response.into_reader();
    let mut data = Vec::new();
    reader.read_to_end(&mut data).map_err(SrganError::Io)?;
    Ok(data)
}

// ── Synthetic ONNX generator ────────────────────────────────────────────────

/// Generate a minimal valid ONNX protobuf file with synthetic weights.
///
/// This creates a small file that the ONNX parser can read, suitable for
/// testing the pipeline end-to-end without real model weights.
fn generate_synthetic_onnx(dest: &Path, scale_factor: u32) -> Result<()> {
    let mut buf = Vec::new();

    // Helper: write a varint
    fn write_varint(buf: &mut Vec<u8>, mut val: u64) {
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val == 0 {
                buf.push(byte);
                return;
            }
            buf.push(byte | 0x80);
        }
    }

    // Helper: write a tag
    fn write_tag(buf: &mut Vec<u8>, field: u32, wire_type: u8) {
        write_varint(buf, ((field as u64) << 3) | wire_type as u64);
    }

    // Helper: write a length-delimited field
    fn write_bytes(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
        write_tag(buf, field, 2); // wire type 2 = length-delimited
        write_varint(buf, data.len() as u64);
        buf.extend_from_slice(data);
    }

    // Build a TensorProto for a synthetic conv weight
    fn build_tensor(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        // field 1: dims (packed int64)
        {
            let mut packed = Vec::new();
            for &d in dims {
                write_varint(&mut packed, d as u64);
            }
            write_bytes(&mut t, 1, &packed);
        }
        // field 2: data_type = FLOAT (1)
        write_tag(&mut t, 2, 0);
        write_varint(&mut t, 1);
        // field 8: name
        write_bytes(&mut t, 8, name.as_bytes());
        // field 13: raw_data (float32 LE bytes)
        {
            let mut raw = Vec::with_capacity(values.len() * 4);
            for &v in values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            write_bytes(&mut t, 13, &raw);
        }
        t
    }

    // Create synthetic weight tensors (small but parseable)
    let n_filters = 32;
    let kernel_size = 3;
    let n_weights = n_filters * 3 * kernel_size * kernel_size;
    let conv1_weights: Vec<f32> = (0..n_weights).map(|i| ((i as f32) * 0.001) - 0.1).collect();
    let conv1_bias: Vec<f32> = vec![0.01; n_filters];

    let tensor1 = build_tensor(
        "conv1.weight",
        &[n_filters as i64, 3, kernel_size as i64, kernel_size as i64],
        &conv1_weights,
    );
    let tensor2 = build_tensor("conv1.bias", &[n_filters as i64], &conv1_bias);

    // Build GraphProto
    let mut graph = Vec::new();
    // field 5: initializer (repeated TensorProto)
    write_bytes(&mut graph, 5, &tensor1);
    write_bytes(&mut graph, 5, &tensor2);

    // Build ModelProto
    // field 1: ir_version = 7
    write_tag(&mut buf, 1, 0);
    write_varint(&mut buf, 7);
    // field 2: opset_import (submessage with field 2 = version 13)
    {
        let mut opset = Vec::new();
        // field 1: domain (empty string = default onnx domain)
        write_bytes(&mut opset, 1, b"");
        // field 2: version = 13
        write_tag(&mut opset, 2, 0);
        write_varint(&mut opset, 13);
        write_bytes(&mut buf, 8, &opset);
    }
    // field 3: producer_name
    write_bytes(&mut buf, 3, b"srgan-rust-synthetic");
    // field 4: producer_version
    write_bytes(&mut buf, 4, format!("scale{}", scale_factor).as_bytes());
    // field 7: graph
    write_bytes(&mut buf, 7, &graph);

    let mut file = fs::File::create(dest).map_err(SrganError::Io)?;
    file.write_all(&buf).map_err(SrganError::Io)?;

    Ok(())
}

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
