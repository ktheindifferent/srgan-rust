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

// ── ModelInfo ─────────────────────────────────────────────────────────────────

/// Public metadata for one SRGAN model, returned by `GET /api/v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Short identifier used in API requests (e.g. `"default"`, `"anime"`).
    pub name: String,
    /// Human-readable description of the model.
    pub description: String,
    /// Upscaling factor this model was trained for (typically 4).
    pub scale_factor: u32,
    /// Semantic version string (e.g. `"1.0.0"`).
    pub version: String,
    /// Reported PSNR on the validation set (dB).
    pub psnr: f64,
    /// Tags describing ideal input material (e.g. `["photos", "general"]`).
    pub recommended_for: Vec<String>,
    /// Absolute path to the model weights file on disk (not exposed via API).
    #[serde(skip)]
    pub path: PathBuf,
}

impl ModelInfo {
    /// Build a minimal `ModelInfo` when no sidecar JSON is present.
    fn inferred(name: &str, path: PathBuf) -> Self {
        Self {
            name: name.to_string(),
            description: format!("SRGAN model ({})", path.file_name().unwrap_or_default().to_string_lossy()),
            scale_factor: 4,
            version: "unknown".to_string(),
            psnr: 0.0,
            recommended_for: vec!["general".to_string()],
            path,
        }
    }
}

// ── ModelManager ──────────────────────────────────────────────────────────────

/// Holds all models discovered at startup.
pub struct ModelManager {
    models: Vec<ModelInfo>,
}

impl ModelManager {
    /// Scan `models_dir` for `.bin` and `.rsr` model files.
    ///
    /// For each weight file `<stem>.(bin|rsr)` the manager also looks for a
    /// sidecar `<stem>.json` in the same directory.  If the JSON is absent the
    /// model is registered with inferred metadata.
    ///
    /// Returns an error only if the directory cannot be read at all.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let mut models = Vec::new();

        let entries = fs::read_dir(models_dir).map_err(|e| {
            SrganError::MissingFolder(format!("Cannot open models directory '{}': {}", models_dir.display(), e))
        })?;

        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
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

            let info = if json_path.exists() {
                match fs::read_to_string(&json_path) {
                    Ok(json_str) => match serde_json::from_str::<ModelInfo>(&json_str) {
                        Ok(mut info) => {
                            info.path = path.clone();
                            info
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: failed to parse model metadata '{}': {} — using inferred metadata",
                                json_path.display(),
                                e
                            );
                            ModelInfo::inferred(&stem, path)
                        }
                    },
                    Err(e) => {
                        eprintln!("Warning: cannot read '{}': {}", json_path.display(), e);
                        ModelInfo::inferred(&stem, path)
                    }
                }
            } else {
                ModelInfo::inferred(&stem, path)
            };

            models.push(info);
        }

        // Stable ordering for consistent API responses.
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

    /// Number of loaded models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// `true` if no model files were found in the directory.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}
