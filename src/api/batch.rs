//! Batch job API — `POST /api/v1/batch` accepts an array of image URLs +
//! upscale config, creates a batch job, returns a `batch_id`.
//!
//! - `GET  /api/v1/batch/:id`    — per-item status + result URLs
//! - `DELETE /api/v1/batch/:id`  — cancel a running batch
//!
//! Batch jobs share the caller's per-org rate-limit quota.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Request / response types ─────────────────────────────────────────────────

/// A single image in a batch request.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchItem {
    /// URL to fetch the source image from.
    pub url: String,
    /// Model override for this item (optional; falls back to batch-level default).
    #[serde(default)]
    pub model: Option<String>,
}

/// `POST /api/v1/batch` request body.
#[derive(Debug, Deserialize)]
pub struct CreateBatchRequest {
    /// List of images to upscale.
    pub images: Vec<BatchItem>,
    /// Default model for all items (e.g. "natural", "anime").
    #[serde(default = "default_model")]
    pub model: String,
    /// Scale factor (default 4).
    #[serde(default = "default_scale")]
    pub scale: u32,
    /// Output format: "png" or "jpeg" (default "png").
    #[serde(default = "default_format")]
    pub format: String,
    /// Optional webhook URL to POST when the entire batch completes.
    #[serde(default)]
    pub webhook_url: Option<String>,
    /// Optional webhook secret for HMAC signing.
    #[serde(default)]
    pub webhook_secret: Option<String>,
}

fn default_model() -> String { "natural".to_string() }
fn default_scale() -> u32 { 4 }
fn default_format() -> String { "png".to_string() }

/// Status of a single item within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchItemStatus {
    Pending,
    Processing,
    Completed,
    Failed { error: String },
    Cancelled,
}

/// Per-item result in a batch job.
#[derive(Debug, Clone, Serialize)]
pub struct BatchItemResult {
    pub index: usize,
    pub url: String,
    pub model: String,
    pub status: BatchItemStatus,
    /// URL where the upscaled result can be downloaded (set on completion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_url: Option<String>,
    /// Processing time in milliseconds (set on completion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<u64>,
}

/// Overall batch job status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UrlBatchJobStatus {
    Processing,
    Completed,
    PartiallyCompleted,
    Failed,
    Cancelled,
}

/// A batch job record stored in the in-memory job store.
#[derive(Debug, Clone, Serialize)]
pub struct UrlBatchJob {
    pub batch_id: String,
    pub status: UrlBatchJobStatus,
    pub model: String,
    pub scale: u32,
    pub format: String,
    pub items: Vec<BatchItemResult>,
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub api_key: String,
    pub created_at: u64,
    pub updated_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_url: Option<String>,
}

/// Response to `POST /api/v1/batch`.
#[derive(Debug, Serialize)]
pub struct CreateBatchResponse {
    pub batch_id: String,
    pub status: UrlBatchJobStatus,
    pub total: usize,
    pub poll_url: String,
}

// ── Batch job store ──────────────────────────────────────────────────────────

/// Thread-safe in-memory store for URL-based batch jobs.
pub struct BatchJobStore {
    jobs: Mutex<HashMap<String, UrlBatchJob>>,
}

impl BatchJobStore {
    pub fn new() -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
        }
    }

    /// Create a new batch job and return the batch_id.
    pub fn create_batch(
        &self,
        req: &CreateBatchRequest,
        api_key: &str,
    ) -> String {
        let batch_id = format!("batch_{}", Uuid::new_v4());
        let now = unix_now();

        let items: Vec<BatchItemResult> = req
            .images
            .iter()
            .enumerate()
            .map(|(i, item)| BatchItemResult {
                index: i,
                url: item.url.clone(),
                model: item.model.clone().unwrap_or_else(|| req.model.clone()),
                status: BatchItemStatus::Pending,
                result_url: None,
                processing_time_ms: None,
            })
            .collect();

        let total = items.len();
        let job = UrlBatchJob {
            batch_id: batch_id.clone(),
            status: UrlBatchJobStatus::Processing,
            model: req.model.clone(),
            scale: req.scale,
            format: req.format.clone(),
            items,
            total,
            completed: 0,
            failed: 0,
            api_key: api_key.to_string(),
            created_at: now,
            updated_at: now,
            webhook_url: req.webhook_url.clone(),
        };

        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.insert(batch_id.clone(), job);
        }

        batch_id
    }

    /// Get a batch job by ID.
    pub fn get_batch(&self, batch_id: &str) -> Option<UrlBatchJob> {
        self.jobs.lock().ok()?.get(batch_id).cloned()
    }

    /// Update item status within a batch.
    pub fn update_item(
        &self,
        batch_id: &str,
        index: usize,
        status: BatchItemStatus,
        result_url: Option<String>,
        processing_time_ms: Option<u64>,
    ) {
        if let Ok(mut jobs) = self.jobs.lock() {
            if let Some(job) = jobs.get_mut(batch_id) {
                if let Some(item) = job.items.get_mut(index) {
                    item.status = status.clone();
                    item.result_url = result_url;
                    item.processing_time_ms = processing_time_ms;
                }

                // Recompute aggregates
                let mut completed = 0usize;
                let mut failed = 0usize;
                for item in &job.items {
                    match &item.status {
                        BatchItemStatus::Completed => completed += 1,
                        BatchItemStatus::Failed { .. } => failed += 1,
                        _ => {}
                    }
                }
                job.completed = completed;
                job.failed = failed;
                job.updated_at = unix_now();

                if completed + failed == job.total {
                    job.status = if failed == 0 {
                        UrlBatchJobStatus::Completed
                    } else if completed > 0 {
                        UrlBatchJobStatus::PartiallyCompleted
                    } else {
                        UrlBatchJobStatus::Failed
                    };
                }
            }
        }
    }

    /// Cancel a batch job.
    pub fn cancel_batch(&self, batch_id: &str) -> bool {
        if let Ok(mut jobs) = self.jobs.lock() {
            if let Some(job) = jobs.get_mut(batch_id) {
                if job.status == UrlBatchJobStatus::Processing {
                    job.status = UrlBatchJobStatus::Cancelled;
                    for item in &mut job.items {
                        if matches!(item.status, BatchItemStatus::Pending | BatchItemStatus::Processing) {
                            item.status = BatchItemStatus::Cancelled;
                        }
                    }
                    job.updated_at = unix_now();
                    return true;
                }
            }
        }
        false
    }

    /// Check if a batch is cancelled (used by worker threads).
    pub fn is_cancelled(&self, batch_id: &str) -> bool {
        self.jobs
            .lock()
            .ok()
            .and_then(|jobs| jobs.get(batch_id).map(|j| j.status == UrlBatchJobStatus::Cancelled))
            .unwrap_or(false)
    }

    /// Clean up old completed/failed/cancelled batches (older than 1 hour).
    pub fn cleanup_old_batches(&self) {
        let cutoff = unix_now().saturating_sub(3600);
        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.retain(|_, job| {
                job.status == UrlBatchJobStatus::Processing || job.updated_at > cutoff
            });
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_batch() {
        let store = BatchJobStore::new();
        let req = CreateBatchRequest {
            images: vec![
                BatchItem { url: "https://example.com/a.jpg".into(), model: None },
                BatchItem { url: "https://example.com/b.jpg".into(), model: Some("anime".into()) },
            ],
            model: "natural".into(),
            scale: 4,
            format: "png".into(),
            webhook_url: None,
            webhook_secret: None,
        };

        let id = store.create_batch(&req, "test-key");
        let job = store.get_batch(&id).unwrap();
        assert_eq!(job.total, 2);
        assert_eq!(job.items[0].model, "natural");
        assert_eq!(job.items[1].model, "anime");
        assert_eq!(job.status, UrlBatchJobStatus::Processing);
    }

    #[test]
    fn test_cancel_batch() {
        let store = BatchJobStore::new();
        let req = CreateBatchRequest {
            images: vec![BatchItem { url: "https://example.com/a.jpg".into(), model: None }],
            model: "natural".into(),
            scale: 4,
            format: "png".into(),
            webhook_url: None,
            webhook_secret: None,
        };

        let id = store.create_batch(&req, "test-key");
        assert!(store.cancel_batch(&id));
        let job = store.get_batch(&id).unwrap();
        assert_eq!(job.status, UrlBatchJobStatus::Cancelled);
    }

    #[test]
    fn test_update_item_completion() {
        let store = BatchJobStore::new();
        let req = CreateBatchRequest {
            images: vec![
                BatchItem { url: "https://example.com/a.jpg".into(), model: None },
            ],
            model: "natural".into(),
            scale: 4,
            format: "png".into(),
            webhook_url: None,
            webhook_secret: None,
        };

        let id = store.create_batch(&req, "test-key");
        store.update_item(&id, 0, BatchItemStatus::Completed, Some("/result/abc".into()), Some(150));
        let job = store.get_batch(&id).unwrap();
        assert_eq!(job.status, UrlBatchJobStatus::Completed);
        assert_eq!(job.completed, 1);
    }
}
