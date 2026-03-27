//! Model version registry — CRUD for named model definitions.
//!
//! Each `ModelVersion` carries an id, human-readable name, semantic version
//! string, upscale factor, supported media types, and creation timestamp.
//!
//! The registry ships with three default models (srgan-v1, real-esrgan-v1,
//! waifu2x-v1) and supports runtime additions/deletions.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Supported media types for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaType {
    Image,
    Video,
}

/// A registered model version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub id: String,
    pub name: String,
    pub version: String,
    pub upscale_factor: u32,
    pub supported_types: Vec<MediaType>,
    pub created_at: String,
}

/// Request body for creating a new model version.
#[derive(Debug, Deserialize)]
pub struct CreateModelVersionRequest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub upscale_factor: u32,
    pub supported_types: Vec<MediaType>,
}

/// Thread-safe in-memory model version store with CRUD operations.
pub struct ModelVersionStore {
    models: Mutex<HashMap<String, ModelVersion>>,
}

impl ModelVersionStore {
    /// Create the store pre-populated with default models.
    pub fn new() -> Self {
        let mut map = HashMap::new();

        let defaults = vec![
            ModelVersion {
                id: "srgan-v1".into(),
                name: "SRGAN".into(),
                version: "1.0.0".into(),
                upscale_factor: 4,
                supported_types: vec![MediaType::Image, MediaType::Video],
                created_at: "2024-01-01T00:00:00Z".into(),
            },
            ModelVersion {
                id: "real-esrgan-v1".into(),
                name: "Real-ESRGAN".into(),
                version: "1.0.0".into(),
                upscale_factor: 4,
                supported_types: vec![MediaType::Image, MediaType::Video],
                created_at: "2024-01-15T00:00:00Z".into(),
            },
            ModelVersion {
                id: "waifu2x-v1".into(),
                name: "Waifu2x".into(),
                version: "1.0.0".into(),
                upscale_factor: 2,
                supported_types: vec![MediaType::Image],
                created_at: "2024-02-01T00:00:00Z".into(),
            },
        ];

        for m in defaults {
            map.insert(m.id.clone(), m);
        }

        Self {
            models: Mutex::new(map),
        }
    }

    /// List all model versions sorted by id.
    pub fn list(&self) -> Vec<ModelVersion> {
        let map = self.models.lock().unwrap();
        let mut models: Vec<ModelVersion> = map.values().cloned().collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    /// Get a single model version by id.
    pub fn get(&self, id: &str) -> Option<ModelVersion> {
        let map = self.models.lock().unwrap();
        map.get(id).cloned()
    }

    /// Insert a new model version. Returns `Err` if the id already exists.
    pub fn create(&self, req: CreateModelVersionRequest) -> Result<ModelVersion, String> {
        let mut map = self.models.lock().unwrap();
        if map.contains_key(&req.id) {
            return Err(format!("Model '{}' already exists", req.id));
        }
        let model = ModelVersion {
            id: req.id.clone(),
            name: req.name,
            version: req.version,
            upscale_factor: req.upscale_factor,
            supported_types: req.supported_types,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        map.insert(req.id, model.clone());
        Ok(model)
    }

    /// Delete a model version by id. Returns `true` if it existed.
    pub fn delete(&self, id: &str) -> bool {
        let mut map = self.models.lock().unwrap();
        map.remove(id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let store = ModelVersionStore::new();
        let list = store.list();
        assert_eq!(list.len(), 3);
        assert!(store.get("srgan-v1").is_some());
        assert!(store.get("real-esrgan-v1").is_some());
        assert!(store.get("waifu2x-v1").is_some());
    }

    #[test]
    fn test_crud() {
        let store = ModelVersionStore::new();

        let req = CreateModelVersionRequest {
            id: "test-model".into(),
            name: "Test".into(),
            version: "0.1.0".into(),
            upscale_factor: 8,
            supported_types: vec![MediaType::Image],
        };
        let model = store.create(req).unwrap();
        assert_eq!(model.upscale_factor, 8);
        assert_eq!(store.list().len(), 4);

        // Duplicate should fail
        let req2 = CreateModelVersionRequest {
            id: "test-model".into(),
            name: "Test".into(),
            version: "0.1.0".into(),
            upscale_factor: 8,
            supported_types: vec![MediaType::Image],
        };
        assert!(store.create(req2).is_err());

        assert!(store.delete("test-model"));
        assert_eq!(store.list().len(), 3);
        assert!(!store.delete("nonexistent"));
    }
}
