//! Model manager — scans a directory for SRGAN model files and their sidecar
//! JSON metadata, then exposes the list via the API.
//!
//! ## File layout expected on disk
//! ```text
//! models/
//!   default.rsr          ← model weights
//!   default.json         ← sidecar metadata (ModelInfo fields)
//!   anime.bin
//!   anime.json
//! ```
//!
//! A model file without a sidecar `.json` is still registered with inferred
//! metadata so it remains accessible.

use crate::error::{Result, SrganError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

// ── ModelArchitecture ─────────────────────────────────────────────────────────

/// The neural network architecture family for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelArchitecture {
    /// Original SRGAN (residual blocks with perceptual loss).
    Srgan,
    /// Enhanced SRGAN — residual-in-residual dense blocks, higher fidelity detail.
    Esrgan,
    /// Real-ESRGAN — trained on real-world degradation (noise, JPEG artifacts, blur).
    RealEsrgan,
    /// Waifu2x — optimised for anime/illustration upscaling and noise reduction.
    Waifu2x,
    /// Plain bilinear interpolation (no neural network).
    Bilinear,
}

impl ModelArchitecture {
    /// Return the built-in label that best approximates this architecture.
    /// Used as fallback when no custom weight file exists.
    pub fn fallback_label(&self) -> &'static str {
        match self {
            ModelArchitecture::Srgan => "natural",
            ModelArchitecture::Esrgan => "natural",
            ModelArchitecture::RealEsrgan => "natural",
            ModelArchitecture::Waifu2x => "anime",
            ModelArchitecture::Bilinear => "bilinear",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ModelArchitecture::Srgan => "SRGAN",
            ModelArchitecture::Esrgan => "ESRGAN",
            ModelArchitecture::RealEsrgan => "Real-ESRGAN",
            ModelArchitecture::Waifu2x => "Waifu2x",
            ModelArchitecture::Bilinear => "Bilinear",
        }
    }
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        ModelArchitecture::Srgan
    }
}

// ── ModelInfo ─────────────────────────────────────────────────────────────────

/// Public metadata for one model, returned by `GET /api/v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Short identifier used in API requests (e.g. `"default"`, `"anime"`).
    pub name: String,
    /// Human-readable description of the model.
    pub description: String,
    /// Network architecture family.
    #[serde(default)]
    pub architecture: ModelArchitecture,
    /// Upscaling factor this model was trained for (typically 4).
    pub scale_factor: u32,
    /// Semantic version string (e.g. `"1.0.0"`).
    pub version: String,
    /// Reported PSNR on the validation set (dB).
    pub psnr: f64,
    /// Tags describing ideal input material (e.g. `["photos", "general"]`).
    pub recommended_for: Vec<String>,
    /// Image type slugs for which this model should be auto-selected
    /// (e.g. `["photograph", "face-heavy"]`).
    #[serde(default)]
    pub auto_select_for: Vec<String>,
    /// Absolute path to the model weights file on disk (not exposed via API).
    #[serde(skip)]
    pub path: PathBuf,
    /// True if the weights file actually exists on disk (false = metadata-only).
    #[serde(skip)]
    pub weights_available: bool,
}

impl ModelInfo {
    /// Build a minimal `ModelInfo` when no sidecar JSON is present.
    fn inferred(name: &str, path: PathBuf) -> Self {
        Self {
            name: name.to_string(),
            description: format!(
                "SRGAN model ({})",
                path.file_name().unwrap_or_default().to_string_lossy()
            ),
            architecture: ModelArchitecture::Srgan,
            scale_factor: 4,
            version: "unknown".to_string(),
            psnr: 0.0,
            recommended_for: vec!["general".to_string()],
            auto_select_for: vec![],
            path,
            weights_available: true,
        }
    }
}

// ── ModelManager ──────────────────────────────────────────────────────────────

/// Holds all models discovered at startup.
pub struct ModelManager {
    models: Vec<ModelInfo>,
}

impl ModelManager {
    /// Scan `models_dir` for `.bin` and `.rsr` model files, plus any
    /// metadata-only `.json` files (without a matching weight file).
    ///
    /// Returns an error only if the directory cannot be read at all.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let mut models: Vec<ModelInfo> = Vec::new();

        let entries = fs::read_dir(models_dir).map_err(|e| {
            SrganError::MissingFolder(format!(
                "Cannot open models directory '{}': {}",
                models_dir.display(),
                e
            ))
        })?;

        // First pass: weight files → register with optional sidecar JSON.
        let mut seen_names = std::collections::HashSet::new();
        let mut all_entries: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect();
        all_entries.sort();

        for path in &all_entries {
            if !path.is_file() {
                continue;
            }
            let ext = match path.extension().and_then(|e| e.to_str()) {
                Some(e) => e,
                None => continue,
            };
            if ext != "bin" && ext != "rsr" {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let json_path = path.with_extension("json");
            let mut info = if json_path.exists() {
                load_json_metadata(&json_path, &stem, path.clone())
            } else {
                ModelInfo::inferred(&stem, path.clone())
            };
            info.weights_available = true;
            seen_names.insert(info.name.clone());
            models.push(info);
        }

        // Second pass: JSON-only files (metadata for built-in / future models).
        for path in &all_entries {
            if !path.is_file() {
                continue;
            }
            let ext = match path.extension().and_then(|e| e.to_str()) {
                Some(e) => e,
                None => continue,
            };
            if ext != "json" {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            // Skip if a weight file already registered this name.
            if seen_names.contains(&stem) {
                continue;
            }
            let mut info = load_json_metadata(path, &stem, PathBuf::new());
            info.weights_available = false;
            seen_names.insert(info.name.clone());
            models.push(info);
        }

        models.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { models })
    }

    /// Return metadata for all discovered models.
    pub fn list(&self) -> &[ModelInfo] {
        &self.models
    }

    /// Look up a model by its `name` field.
    pub fn get(&self, name: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Auto-select the best model for a given image-type slug
    /// (e.g. `"anime"`, `"photograph"`).  Returns the first model whose
    /// `auto_select_for` list contains the slug, or falls back to `"natural"`.
    pub fn select_for_image_type<'a>(&'a self, image_type_slug: &str) -> &'a ModelInfo {
        // Prefer models whose auto_select_for list explicitly matches.
        if let Some(m) = self
            .models
            .iter()
            .find(|m| m.auto_select_for.iter().any(|s| s == image_type_slug))
        {
            return m;
        }
        // Fall back to first model whose recommended_for includes the slug.
        if let Some(m) = self
            .models
            .iter()
            .find(|m| m.recommended_for.iter().any(|s| s == image_type_slug))
        {
            return m;
        }
        // Last resort: first model in the list.
        &self.models[0]
    }

    /// Number of loaded models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// `true` if no model files were found in the directory.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

fn load_json_metadata(json_path: &Path, stem: &str, weight_path: PathBuf) -> ModelInfo {
    match fs::read_to_string(json_path) {
        Ok(json_str) => match serde_json::from_str::<ModelInfo>(&json_str) {
            Ok(mut info) => {
                info.path = weight_path;
                info
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to parse model metadata '{}': {} — using inferred metadata",
                    json_path.display(),
                    e
                );
                ModelInfo::inferred(stem, weight_path)
            }
        },
        Err(e) => {
            eprintln!("Warning: cannot read '{}': {}", json_path.display(), e);
            ModelInfo::inferred(stem, weight_path)
        }
    }
}
