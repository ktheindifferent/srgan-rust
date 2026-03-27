//! Model versioning — maps version tags (v1, v2, v3) to scale factors and
//! model directories for multi-version API support.
//!
//! Version layout on disk:
//! ```text
//! models/
//!   v1/   ← 2× models
//!   v2/   ← 4× models (default)
//!   v3/   ← 8× models
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// A single model version definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Version tag (e.g. "v1", "v2", "v3").
    pub tag: String,
    /// Upscaling factor for this version.
    pub scale_factor: u32,
    /// Human-readable description.
    pub description: String,
    /// Whether model weights are available on disk for this version.
    pub available: bool,
    /// Directory path for this version's weights.
    #[serde(skip)]
    pub directory: PathBuf,
}

/// Registry of all supported model versions.
pub struct ModelVersionRegistry {
    versions: HashMap<String, ModelVersion>,
    default_version: String,
    base_dir: PathBuf,
}

impl ModelVersionRegistry {
    /// Create the registry, scanning `base_dir` for version subdirectories.
    /// If `base_dir` does not exist, all versions are marked unavailable
    /// (built-in fallback models will be used).
    pub fn new(base_dir: &Path) -> Self {
        let mut versions = HashMap::new();

        let defs: Vec<(&str, u32, &str)> = vec![
            ("v1", 2, "2x upscaling — lightweight, fast inference"),
            ("v2", 4, "4x upscaling — default, balanced quality/speed"),
            ("v3", 8, "8x upscaling — maximum resolution, slower"),
        ];

        for (tag, factor, desc) in defs {
            let dir = base_dir.join(tag);
            let available = dir.is_dir() && has_weight_files(&dir);
            versions.insert(
                tag.to_string(),
                ModelVersion {
                    tag: tag.to_string(),
                    scale_factor: factor,
                    description: desc.to_string(),
                    available,
                    directory: dir,
                },
            );
        }

        Self {
            versions,
            default_version: "v2".to_string(),
            base_dir: base_dir.to_path_buf(),
        }
    }

    /// Look up a version by tag. Returns `None` for unknown tags.
    pub fn get(&self, tag: &str) -> Option<&ModelVersion> {
        self.versions.get(tag)
    }

    /// The default version tag.
    pub fn default_tag(&self) -> &str {
        &self.default_version
    }

    /// Resolve the scale factor for a version tag, falling back to the
    /// default version's factor for unknown tags.
    pub fn scale_factor_for(&self, tag: Option<&str>) -> u32 {
        match tag {
            Some(t) => self.versions.get(t).map(|v| v.scale_factor).unwrap_or(4),
            None => self
                .versions
                .get(&self.default_version)
                .map(|v| v.scale_factor)
                .unwrap_or(4),
        }
    }

    /// Find the first weight file in the version directory (if any).
    pub fn weight_path_for(&self, tag: &str) -> Option<PathBuf> {
        let version = self.versions.get(tag)?;
        if !version.available {
            return None;
        }
        find_first_weight(&version.directory)
    }

    /// List all versions sorted by scale factor.
    pub fn list(&self) -> Vec<&ModelVersion> {
        let mut vs: Vec<&ModelVersion> = self.versions.values().collect();
        vs.sort_by_key(|v| v.scale_factor);
        vs
    }

    /// Base directory for versioned models.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

fn has_weight_files(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                let ext = e.path().extension().and_then(|x| x.to_str()).unwrap_or("").to_string();
                ext == "rsr" || ext == "bin" || ext == "onnx" || ext == "pth" || ext == "pt"
            })
        })
        .unwrap_or(false)
}

fn find_first_weight(dir: &Path) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
            ext == "rsr" || ext == "bin" || ext == "onnx" || ext == "pth" || ext == "pt"
        })
        .collect();
    entries.sort();
    entries.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_registry_no_dir() {
        let reg = ModelVersionRegistry::new(Path::new("/tmp/srgan_nonexistent_test_dir_xyz"));
        assert_eq!(reg.list().len(), 3);
        assert_eq!(reg.default_tag(), "v2");
        assert_eq!(reg.scale_factor_for(Some("v1")), 2);
        assert_eq!(reg.scale_factor_for(Some("v2")), 4);
        assert_eq!(reg.scale_factor_for(Some("v3")), 8);
        assert_eq!(reg.scale_factor_for(None), 4);
        assert_eq!(reg.scale_factor_for(Some("v99")), 4);
        // All versions unavailable when dir doesn't exist
        for v in reg.list() {
            assert!(!v.available);
        }
    }

    #[test]
    fn test_registry_with_dir() {
        let tmp = std::env::temp_dir().join("srgan_versioning_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("v1")).unwrap();
        fs::create_dir_all(tmp.join("v2")).unwrap();
        // v1 has a weight file, v2 doesn't, v3 dir doesn't exist
        fs::write(tmp.join("v1/model.rsr"), b"fake").unwrap();

        let reg = ModelVersionRegistry::new(&tmp);
        let v1 = reg.get("v1").unwrap();
        assert!(v1.available);
        assert_eq!(v1.scale_factor, 2);

        let v2 = reg.get("v2").unwrap();
        assert!(!v2.available); // dir exists but no weight files

        let v3 = reg.get("v3").unwrap();
        assert!(!v3.available); // dir doesn't exist

        assert!(reg.weight_path_for("v1").is_some());
        assert!(reg.weight_path_for("v2").is_none());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_list_sorted() {
        let reg = ModelVersionRegistry::new(Path::new("/tmp/nonexistent"));
        let list = reg.list();
        assert_eq!(list[0].tag, "v1");
        assert_eq!(list[1].tag, "v2");
        assert_eq!(list[2].tag, "v3");
    }
}
