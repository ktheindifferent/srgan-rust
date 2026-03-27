//! Admin endpoint — GET /api/v1/admin/stats
//!
//! Requires the `ADMIN_KEY` environment variable to be set and passed as the
//! `X-Admin-Key` header (or `Authorization: Bearer <key>`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::api::auth::KeyStore;

// ── System stats ─────────────────────────────────────────────────────────────

/// Snapshot of host-level resource usage.
#[derive(Debug, Serialize)]
pub struct SystemStats {
    /// CPU count.
    pub cpu_count: usize,
    /// Total RAM in MB.
    pub total_ram_mb: u64,
    /// Estimated used RAM in MB (from tracking allocator).
    pub used_ram_mb: u64,
    /// Current queue depth (pending + processing jobs).
    pub queue_depth: usize,
    /// Server uptime in seconds.
    pub uptime_secs: u64,
}

impl SystemStats {
    pub fn collect(queue_depth: usize, uptime: Duration) -> Self {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let (total_ram_mb, used_ram_mb) = if let Ok(info) = sys_info::mem_info() {
            (info.total / 1024, (info.total - info.free) / 1024)
        } else {
            (0, 0)
        };

        Self {
            cpu_count,
            total_ram_mb,
            used_ram_mb,
            queue_depth,
            uptime_secs: uptime.as_secs(),
        }
    }
}

// ── Model performance metrics ────────────────────────────────────────────────

/// Per-model inference performance tracker.
#[derive(Debug, Clone, Serialize)]
pub struct ModelMetrics {
    pub model_name: String,
    pub total_inferences: u64,
    pub avg_inference_ms: f64,
    pub min_inference_ms: u64,
    pub max_inference_ms: u64,
    /// Throughput: inferences per minute (rolling window).
    pub throughput_per_min: f64,
}

/// Tracks inference timings per model label.
pub struct ModelMetricsTracker {
    entries: std::sync::Mutex<HashMap<String, Vec<(Instant, u64)>>>,
}

impl ModelMetricsTracker {
    pub fn new() -> Self {
        Self {
            entries: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Record an inference for the given model.
    pub fn record(&self, model: &str, duration_ms: u64) {
        if let Ok(mut map) = self.entries.lock() {
            map.entry(model.to_string())
                .or_insert_with(Vec::new)
                .push((Instant::now(), duration_ms));
        }
    }

    /// Snapshot all model metrics.
    pub fn snapshot(&self) -> Vec<ModelMetrics> {
        let map = match self.entries.lock() {
            Ok(m) => m,
            Err(_) => return vec![],
        };

        let now = Instant::now();
        let one_min_ago = now - Duration::from_secs(60);

        map.iter()
            .map(|(name, timings)| {
                let total = timings.len() as u64;
                let sum: u64 = timings.iter().map(|(_, ms)| *ms).sum();
                let min = timings.iter().map(|(_, ms)| *ms).min().unwrap_or(0);
                let max = timings.iter().map(|(_, ms)| *ms).max().unwrap_or(0);
                let avg = if total > 0 { sum as f64 / total as f64 } else { 0.0 };
                let recent = timings.iter().filter(|(t, _)| *t >= one_min_ago).count();

                ModelMetrics {
                    model_name: name.clone(),
                    total_inferences: total,
                    avg_inference_ms: avg,
                    min_inference_ms: min,
                    max_inference_ms: max,
                    throughput_per_min: recent as f64,
                }
            })
            .collect()
    }
}

// ── Job history ──────────────────────────────────────────────────────────────

/// A single entry in the recent job history table.
#[derive(Debug, Clone, Serialize)]
pub struct JobHistoryEntry {
    pub job_id: String,
    pub model: String,
    pub status: String,
    pub created_at: u64,
    pub processing_time_ms: Option<u64>,
    pub api_key_prefix: String,
    pub input_size: Option<String>,
    pub output_size: Option<String>,
}

// ── API key management ───────────────────────────────────────────────────────

/// Summary of an API key for the admin UI.
#[derive(Debug, Clone, Serialize)]
pub struct ApiKeyInfo {
    pub key_id: String,
    pub key_prefix: String,
    pub tier: String,
    pub created_at: String,
    pub requests_today: u64,
    pub is_active: bool,
}

// ── Full admin response ──────────────────────────────────────────────────────

/// Statistics returned by GET /api/v1/admin/stats
#[derive(Debug, Serialize)]
pub struct AdminStats {
    pub total_keys: usize,
    pub keys_by_tier: HashMap<String, usize>,
    pub usage_today_by_tier: HashMap<String, u64>,
    pub total_requests_today: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_metrics: Option<Vec<ModelMetrics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recent_jobs: Option<Vec<JobHistoryEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_keys: Option<Vec<ApiKeyInfo>>,
}

/// Verify that `provided` matches the `ADMIN_KEY` env var.
///
/// Returns `true` if the admin key is not set (open mode) OR if `provided` matches.
pub fn check_admin_auth(provided: &str) -> bool {
    match std::env::var("ADMIN_KEY") {
        Ok(expected) => provided == expected,
        // No ADMIN_KEY configured — deny all admin access
        Err(_) => false,
    }
}

/// Build an `AdminStats` snapshot.
pub fn get_stats(key_store: &Arc<KeyStore>) -> AdminStats {
    let all_keys = key_store.all_keys().unwrap_or_default();

    let total_keys = all_keys.len();

    let mut keys_by_tier: HashMap<String, usize> = HashMap::new();
    for k in &all_keys {
        *keys_by_tier.entry(k.tier.as_str().to_string()).or_insert(0) += 1;
    }

    let usage_today_by_tier = key_store.usage_today_by_tier();
    let total_requests_today: u64 = usage_today_by_tier.values().sum();

    // Build API key summaries
    let api_keys: Vec<ApiKeyInfo> = all_keys
        .iter()
        .map(|k| {
            let prefix = if k.key.len() >= 8 {
                format!("{}...", &k.key[..8])
            } else {
                k.key.clone()
            };
            ApiKeyInfo {
                key_id: k.key.clone(),
                key_prefix: prefix,
                tier: k.tier.as_str().to_string(),
                created_at: k.created_at.to_string(),
                requests_today: 0, // per-key usage requires rate limiter lookup
                is_active: true,
            }
        })
        .collect();

    AdminStats {
        total_keys,
        keys_by_tier,
        usage_today_by_tier,
        total_requests_today,
        system: None,
        model_metrics: None,
        recent_jobs: None,
        api_keys: Some(api_keys),
    }
}

/// Build full stats with system info and model metrics.
pub fn get_full_stats(
    key_store: &Arc<KeyStore>,
    queue_depth: usize,
    uptime: Duration,
    model_tracker: Option<&ModelMetricsTracker>,
    recent_jobs: Option<Vec<JobHistoryEntry>>,
) -> AdminStats {
    let mut stats = get_stats(key_store);
    stats.system = Some(SystemStats::collect(queue_depth, uptime));
    stats.model_metrics = model_tracker.map(|t| t.snapshot());
    stats.recent_jobs = recent_jobs;
    stats
}
