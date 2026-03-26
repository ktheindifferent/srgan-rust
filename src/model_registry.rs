//! Plugin model registry — loads custom model manifests from `~/.srgan/models/`
//! and exposes them alongside built-in models.
//!
//! ## Manifest format (`~/.srgan/models/<name>.json`)
//! ```json
//! {
//!   "name": "mymodel",
//!   "display_name": "My Custom Model",
//!   "type": "esrgan",
//!   "scale_factors": [4],
//!   "description": "Custom upscaling model",
//!   "weight_path": "/path/to/weights.bin"
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Result, SrganError};

// ── ModelType ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Esrgan,
    Waifu2x,
    Custom,
}

impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelType::Esrgan => "esrgan",
            ModelType::Waifu2x => "waifu2x",
            ModelType::Custom => "custom",
        }
    }
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::Custom
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ModelType {
    type Err = SrganError;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "esrgan" => Ok(ModelType::Esrgan),
            "waifu2x" => Ok(ModelType::Waifu2x),
            "custom" => Ok(ModelType::Custom),
            other => Err(SrganError::InvalidParameter(format!(
                "Unknown model type '{}'. Valid types: esrgan, waifu2x, custom",
                other
            ))),
        }
    }
}

// ── RegistryEntry ─────────────────────────────────────────────────────────────

/// A single model manifest entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Short identifier, e.g. `"mymodel"`.
    pub name: String,
    /// Human-readable name, e.g. `"My Custom Model"`.
    pub display_name: String,
    /// Network architecture family.
    #[serde(rename = "type", default)]
    pub model_type: ModelType,
    /// Scale factors this model supports (e.g. `[2, 4]`).
    #[serde(default = "default_scale_factors")]
    pub scale_factors: Vec<u32>,
    /// Free-text description.
    #[serde(default)]
    pub description: String,
    /// Absolute path to the weights file on disk.
    #[serde(default)]
    pub weight_path: String,
    /// True for built-in models, false for user-added custom models.
    #[serde(skip, default)]
    pub builtin: bool,
}

fn default_scale_factors() -> Vec<u32> {
    vec![4]
}

// ── Built-in definitions ──────────────────────────────────────────────────────

fn builtin_entries() -> Vec<RegistryEntry> {
    vec![
        RegistryEntry {
            name: "natural".to_string(),
            display_name: "Natural (SRGAN)".to_string(),
            model_type: ModelType::Custom,
            scale_factors: vec![4],
            description: "Neural net trained on natural photographs with L1 loss (built-in)"
                .to_string(),
            weight_path: String::new(),
            builtin: true,
        },
        RegistryEntry {
            name: "anime".to_string(),
            display_name: "Anime (SRGAN)".to_string(),
            model_type: ModelType::Custom,
            scale_factors: vec![4],
            description: "Neural net trained on animation images with L1 loss (built-in)"
                .to_string(),
            weight_path: String::new(),
            builtin: true,
        },
        RegistryEntry {
            name: "waifu2x".to_string(),
            display_name: "Waifu2x".to_string(),
            model_type: ModelType::Waifu2x,
            scale_factors: vec![1, 2],
            description: "Waifu2x-style model for anime/illustration upscaling (built-in)"
                .to_string(),
            weight_path: String::new(),
            builtin: true,
        },
        RegistryEntry {
            name: "real-esrgan".to_string(),
            display_name: "Real-ESRGAN ×4".to_string(),
            model_type: ModelType::Esrgan,
            scale_factors: vec![4],
            description: "Real-ESRGAN ×4 for general photos (built-in)".to_string(),
            weight_path: String::new(),
            builtin: true,
        },
        RegistryEntry {
            name: "real-esrgan-anime".to_string(),
            display_name: "Real-ESRGAN Anime ×4".to_string(),
            model_type: ModelType::Esrgan,
            scale_factors: vec![4],
            description: "Real-ESRGAN ×4 optimised for anime content (built-in)".to_string(),
            weight_path: String::new(),
            builtin: true,
        },
        RegistryEntry {
            name: "real-esrgan-x2".to_string(),
            display_name: "Real-ESRGAN ×2".to_string(),
            model_type: ModelType::Esrgan,
            scale_factors: vec![2],
            description: "Real-ESRGAN ×2 for general photos (built-in)".to_string(),
            weight_path: String::new(),
            builtin: true,
        },
        RegistryEntry {
            name: "bilinear".to_string(),
            display_name: "Bilinear".to_string(),
            model_type: ModelType::Custom,
            scale_factors: vec![2, 4],
            description: "Bilinear interpolation (no neural network, built-in)".to_string(),
            weight_path: String::new(),
            builtin: true,
        },
    ]
}

// ── ModelRegistry ─────────────────────────────────────────────────────────────

/// Scans `~/.srgan/models/` for JSON manifests and exposes them alongside
/// built-in models.
pub struct ModelRegistry {
    entries: Vec<RegistryEntry>,
    registry_dir: PathBuf,
}

impl ModelRegistry {
    /// Load the registry from `~/.srgan/models/`.
    /// Creates the directory if it doesn't exist.
    pub fn load() -> Result<Self> {
        let dir = registry_dir()?;
        Self::load_from(&dir)
    }

    /// Load the registry from an explicit directory (useful for tests).
    pub fn load_from(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir).map_err(|e| {
            SrganError::MissingFolder(format!(
                "Cannot create registry directory '{}': {}",
                dir.display(),
                e
            ))
        })?;

        let mut entries = builtin_entries();
        let builtin_names: std::collections::HashSet<String> =
            entries.iter().map(|e| e.name.clone()).collect();

        // Scan for JSON manifest files
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| {
                SrganError::MissingFolder(format!(
                    "Cannot read registry directory '{}': {}",
                    dir.display(),
                    e
                ))
            })?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
            .collect();
        paths.sort();

        for path in &paths {
            match fs::read_to_string(path) {
                Ok(json_str) => match serde_json::from_str::<RegistryEntry>(&json_str) {
                    Ok(mut entry) => {
                        if builtin_names.contains(&entry.name) {
                            eprintln!(
                                "Warning: custom model '{}' shadows a built-in name — skipping",
                                entry.name
                            );
                            continue;
                        }
                        entry.builtin = false;
                        entries.push(entry);
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: failed to parse manifest '{}': {} — skipping",
                            path.display(),
                            e
                        );
                    }
                },
                Err(e) => {
                    eprintln!("Warning: cannot read manifest '{}': {}", path.display(), e);
                }
            }
        }

        Ok(Self {
            entries,
            registry_dir: dir.to_path_buf(),
        })
    }

    /// All registered models (built-in + custom).
    pub fn list(&self) -> &[RegistryEntry] {
        &self.entries
    }

    /// Look up a model by name.
    pub fn get(&self, name: &str) -> Option<&RegistryEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Add a new custom model entry to the registry.
    /// Writes a JSON manifest to `~/.srgan/models/<name>.json`.
    pub fn add(
        &mut self,
        name: &str,
        display_name: &str,
        model_type: ModelType,
        scale_factors: Vec<u32>,
        description: &str,
        weight_path: &str,
    ) -> Result<PathBuf> {
        if self.entries.iter().any(|e| e.name == name && e.builtin) {
            return Err(SrganError::InvalidParameter(format!(
                "Cannot override built-in model '{}'",
                name
            )));
        }

        let entry = RegistryEntry {
            name: name.to_string(),
            display_name: display_name.to_string(),
            model_type,
            scale_factors,
            description: description.to_string(),
            weight_path: weight_path.to_string(),
            builtin: false,
        };

        let manifest_path = self.registry_dir.join(format!("{}.json", name));
        let json = serde_json::to_string_pretty(&entry).map_err(|e| {
            SrganError::InvalidParameter(format!("Failed to serialize manifest: {}", e))
        })?;
        fs::write(&manifest_path, &json).map_err(|e| {
            SrganError::InvalidParameter(format!(
                "Failed to write manifest '{}': {}",
                manifest_path.display(),
                e
            ))
        })?;

        if let Some(existing) = self.entries.iter_mut().find(|e| e.name == name) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }

        Ok(manifest_path)
    }

    /// Remove a custom model by name.
    /// Deletes the JSON manifest from disk.
    pub fn remove(&mut self, name: &str) -> Result<()> {
        match self.entries.iter().find(|e| e.name == name) {
            Some(entry) if entry.builtin => {
                return Err(SrganError::InvalidParameter(format!(
                    "Cannot remove built-in model '{}'",
                    name
                )));
            }
            None => {
                return Err(SrganError::InvalidParameter(format!(
                    "Model '{}' not found in registry",
                    name
                )));
            }
            _ => {}
        }

        let manifest_path = self.registry_dir.join(format!("{}.json", name));
        if manifest_path.exists() {
            fs::remove_file(&manifest_path).map_err(|e| {
                SrganError::InvalidParameter(format!(
                    "Failed to remove manifest '{}': {}",
                    manifest_path.display(),
                    e
                ))
            })?;
        }

        self.entries.retain(|e| e.name != name);
        Ok(())
    }

    /// Returns only user-added custom models.
    pub fn custom_models(&self) -> impl Iterator<Item = &RegistryEntry> {
        self.entries.iter().filter(|e| !e.builtin)
    }
}

/// Returns `~/.srgan/models/` as a `PathBuf`.
pub fn registry_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| {
            SrganError::InvalidParameter("Cannot determine home directory".to_string())
        })?;
    Ok(PathBuf::from(home).join(".srgan").join("models"))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_registry() -> (TempDir, ModelRegistry) {
        let dir = TempDir::new().unwrap();
        let registry = ModelRegistry::load_from(dir.path()).unwrap();
        (dir, registry)
    }

    #[test]
    fn builtins_are_present() {
        let (_dir, reg) = temp_registry();
        assert!(reg.get("natural").is_some());
        assert!(reg.get("anime").is_some());
        assert!(reg.get("real-esrgan").is_some());
        assert!(reg.get("bilinear").is_some());
        assert_eq!(reg.custom_models().count(), 0);
    }

    #[test]
    fn add_and_remove_custom_model() {
        let (_dir, mut reg) = temp_registry();

        let path = reg
            .add(
                "testmodel",
                "Test Model",
                ModelType::Esrgan,
                vec![4],
                "A test model",
                "/tmp/test.bin",
            )
            .unwrap();

        assert!(path.exists());
        assert!(reg.get("testmodel").is_some());
        assert!(!reg.get("testmodel").unwrap().builtin);
        assert_eq!(reg.custom_models().count(), 1);

        reg.remove("testmodel").unwrap();
        assert!(reg.get("testmodel").is_none());
        assert!(!path.exists());
    }

    #[test]
    fn cannot_override_builtin() {
        let (_dir, mut reg) = temp_registry();
        let result = reg.add("natural", "Override", ModelType::Custom, vec![4], "", "");
        assert!(result.is_err());
    }

    #[test]
    fn cannot_remove_builtin() {
        let (_dir, mut reg) = temp_registry();
        let result = reg.remove("natural");
        assert!(result.is_err());
    }

    #[test]
    fn manifests_persist_across_loads() {
        let dir = TempDir::new().unwrap();
        {
            let mut reg = ModelRegistry::load_from(dir.path()).unwrap();
            reg.add(
                "persistent",
                "Persistent Model",
                ModelType::Waifu2x,
                vec![2],
                "Persists",
                "/tmp/p.bin",
            )
            .unwrap();
        }
        // Reload
        let reg2 = ModelRegistry::load_from(dir.path()).unwrap();
        let entry = reg2.get("persistent").unwrap();
        assert_eq!(entry.display_name, "Persistent Model");
        assert_eq!(entry.model_type, ModelType::Waifu2x);
        assert_eq!(entry.scale_factors, vec![2]);
    }
}
