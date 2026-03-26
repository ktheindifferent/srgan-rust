//! Priority job queue for async upscaling jobs.
//!
//! - Enterprise jobs are processed before Pro, which are processed before Free.
//! - Jobs running longer than 5 minutes are cancelled with `TimedOut`.
//! - Completed / failed / timed-out jobs are purged after 1 hour.
//! - On completion, an optional `WebhookConfig` triggers a signed POST with retry logic.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::auth::KeyTier;

// ── Constants ─────────────────────────────────────────────────────────────────

const JOB_TIMEOUT: Duration = Duration::from_secs(5 * 60);      // 5 minutes
const JOB_RETAIN: Duration = Duration::from_secs(60 * 60);       // 1 hour

// ── Webhook types ─────────────────────────────────────────────────────────────

fn default_max_retries() -> u32 { 3 }
fn default_retry_delay_secs() -> u64 { 5 }

/// Configuration for webhook delivery on job completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// URL to POST the result payload to.
    pub url: String,
    /// Optional secret used to compute the HMAC-SHA256 `X-SRGAN-Signature` header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,
    /// Maximum number of retry attempts after the first failure (default 3).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Base delay in seconds between retries; doubles on each attempt (default 5).
    #[serde(default = "default_retry_delay_secs")]
    pub retry_delay_secs: u64,
}

/// Per-job webhook delivery state stored alongside the job record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebhookDeliveryState {
    /// Total number of delivery attempts made so far.
    pub attempts: u32,
    /// Unix timestamp of the most recent attempt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_attempt_at: Option<u64>,
    /// HTTP status code returned by the last attempt (None for transport errors).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_status_code: Option<u16>,
    /// True once a 2xx response is received.
    pub delivered: bool,
}

// ── Job types ─────────────────────────────────────────────────────────────────

/// Processing priority (higher = sooner).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JobPriority {
    Free = 0,
    Pro = 1,
    Enterprise = 2,
}

impl From<KeyTier> for JobPriority {
    fn from(tier: KeyTier) -> Self {
        match tier {
            KeyTier::Free => JobPriority::Free,
            KeyTier::Pro => JobPriority::Pro,
            KeyTier::Enterprise => JobPriority::Enterprise,
        }
    }
}

/// Current state of a job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    TimedOut,
}

impl JobStatus {
    pub fn as_str(&self) -> &str {
        match self {
            JobStatus::Pending => "pending",
            JobStatus::Processing => "processing",
            JobStatus::Completed => "completed",
            JobStatus::Failed(_) => "failed",
            JobStatus::TimedOut => "timed_out",
        }
    }
}

/// A single upscaling job record.
#[derive(Debug, Clone, Serialize)]
pub struct JobRecord {
    pub id: String,
    pub priority: JobPriority,
    pub status: JobStatus,
    /// Base64-encoded input image.
    pub input_data: String,
    /// Base64-encoded output image (set when Completed).
    pub result_data: Option<String>,
    pub api_key: String,
    pub model: String,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
    /// Full webhook configuration (supersedes the legacy `webhook_url` field).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_config: Option<WebhookConfig>,
    /// Delivery state updated on each attempt.
    pub webhook_delivery: WebhookDeliveryState,
}

impl JobRecord {
    pub fn new(
        input_data: String,
        api_key: String,
        model: String,
        priority: JobPriority,
        webhook_config: Option<WebhookConfig>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            priority,
            status: JobStatus::Pending,
            input_data,
            result_data: None,
            api_key,
            model,
            created_at: unix_now(),
            started_at: None,
            completed_at: None,
            webhook_config,
            webhook_delivery: WebhookDeliveryState::default(),
        }
    }
}

/// Request body for POST /api/v1/upscale/async
#[derive(Debug, Deserialize)]
pub struct AsyncUpscaleRequest {
    pub image_data: String,
    pub model: Option<String>,
    pub format: Option<String>,
    /// Full webhook configuration (url, secret, retries, delay).
    pub webhook_config: Option<WebhookConfig>,
}

/// Response for POST /api/v1/upscale/async
#[derive(Debug, Serialize)]
pub struct AsyncUpscaleResponse {
    pub job_id: String,
    pub status: String,
    pub priority: String,
}

// ── Priority job queue ────────────────────────────────────────────────────────

/// Thread-safe priority queue of upscaling jobs.
pub struct PriorityJobQueue {
    /// Wrapped in Arc so the webhook delivery thread can update delivery state.
    jobs: Arc<Mutex<HashMap<String, JobRecord>>>,
    /// Monotonic clock for timeout tracking (job_id → start instant)
    timers: Mutex<HashMap<String, Instant>>,
}

impl PriorityJobQueue {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            timers: Mutex::new(HashMap::new()),
        })
    }

    /// Enqueue a new job and return its ID.
    pub fn enqueue(&self, job: JobRecord) -> String {
        let id = job.id.clone();
        self.jobs.lock().unwrap().insert(id.clone(), job);
        id
    }

    /// Pop the highest-priority pending job (Enterprise > Pro > Free).
    pub fn pop_next(&self) -> Option<JobRecord> {
        let mut jobs = self.jobs.lock().unwrap();
        let id = jobs
            .values()
            .filter(|j| j.status == JobStatus::Pending)
            .max_by_key(|j| j.priority as u8)
            .map(|j| j.id.clone())?;

        if let Some(job) = jobs.get_mut(&id) {
            job.status = JobStatus::Processing;
            job.started_at = Some(unix_now());
            let job = job.clone();
            // Record start instant for timeout tracking
            self.timers.lock().unwrap().insert(id, Instant::now());
            return Some(job);
        }
        None
    }

    /// Mark a job as completed (with result data) and fire the webhook if set.
    pub fn complete(&self, id: &str, result_data: String) {
        let webhook_config = {
            let mut jobs = self.jobs.lock().unwrap();
            if let Some(job) = jobs.get_mut(id) {
                job.status = JobStatus::Completed;
                job.result_data = Some(result_data);
                job.completed_at = Some(unix_now());
                job.webhook_config.clone()
            } else {
                None
            }
        };
        self.timers.lock().unwrap().remove(id);
        if let Some(config) = webhook_config {
            self.fire_webhook_with_retry(id, "completed", config);
        }
    }

    /// Mark a job as failed and fire the webhook if set.
    pub fn fail(&self, id: &str, reason: String) {
        let webhook_config = {
            let mut jobs = self.jobs.lock().unwrap();
            if let Some(job) = jobs.get_mut(id) {
                job.status = JobStatus::Failed(reason);
                job.completed_at = Some(unix_now());
                job.webhook_config.clone()
            } else {
                None
            }
        };
        self.timers.lock().unwrap().remove(id);
        if let Some(config) = webhook_config {
            self.fire_webhook_with_retry(id, "failed", config);
        }
    }

    fn fire_webhook_with_retry(&self, id: &str, status: &str, config: WebhookConfig) {
        let payload = serde_json::json!({
            "job_id": id,
            "status": status,
            "timestamp": unix_now(),
        })
        .to_string();
        let jobs_arc = Arc::clone(&self.jobs);
        let job_id = id.to_string();
        deliver_webhook(
            config.url,
            config.secret,
            payload,
            config.max_retries,
            config.retry_delay_secs,
            move |attempts, last_status_code, delivered| {
                if let Ok(mut jobs) = jobs_arc.lock() {
                    if let Some(job) = jobs.get_mut(&job_id) {
                        job.webhook_delivery.attempts = attempts;
                        job.webhook_delivery.last_attempt_at = Some(unix_now());
                        job.webhook_delivery.last_status_code = last_status_code;
                        job.webhook_delivery.delivered = delivered;
                    }
                }
            },
        );
    }

    /// Get a snapshot of a job by ID.
    pub fn get(&self, id: &str) -> Option<JobRecord> {
        self.jobs.lock().unwrap().get(id).cloned()
    }

    /// Cancel jobs that have been processing longer than `JOB_TIMEOUT`.
    pub fn timeout_stale_jobs(&self) {
        let stale: Vec<String> = {
            let timers = self.timers.lock().unwrap();
            timers
                .iter()
                .filter(|(_, start)| start.elapsed() > JOB_TIMEOUT)
                .map(|(id, _)| id.clone())
                .collect()
        };

        for id in stale {
            let mut jobs = self.jobs.lock().unwrap();
            if let Some(job) = jobs.get_mut(&id) {
                job.status = JobStatus::TimedOut;
                job.completed_at = Some(unix_now());
            }
            self.timers.lock().unwrap().remove(&id);
        }
    }

    /// Delete completed / failed / timed-out jobs older than `JOB_RETAIN`.
    pub fn cleanup_old_jobs(&self) {
        let cutoff = unix_now().saturating_sub(JOB_RETAIN.as_secs());
        let mut jobs = self.jobs.lock().unwrap();
        jobs.retain(|_, job| {
            let is_terminal = matches!(
                job.status,
                JobStatus::Completed | JobStatus::Failed(_) | JobStatus::TimedOut
            );
            if is_terminal {
                job.completed_at.map_or(true, |t| t > cutoff)
            } else {
                true
            }
        });
    }

    /// List all jobs (used by GET /api/v1/jobs).
    pub fn all_jobs(&self) -> Vec<JobRecord> {
        self.jobs.lock().unwrap().values().cloned().collect()
    }
}

// ── Webhook ───────────────────────────────────────────────────────────────────

/// Compute HMAC-SHA256 of `body` using `secret` and return the lowercase hex digest.
fn hmac_sha256(secret: &str, body: &str) -> String {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC accepts keys of any length");
    mac.update(body.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Deliver a webhook payload to `url` with optional HMAC signing and exponential-backoff
/// retry.  Spawns a background thread so callers are never blocked.
///
/// `on_update(attempts, last_status_code, delivered)` is called after every attempt so
/// callers can persist delivery state.
pub fn deliver_webhook<F>(
    url: String,
    secret: Option<String>,
    payload: String,
    max_retries: u32,
    retry_delay_secs: u64,
    on_update: F,
) where
    F: Fn(u32, Option<u16>, bool) + Send + 'static,
{
    std::thread::spawn(move || {
        let total = max_retries + 1;
        for attempt in 1..=total {
            let sig = secret.as_deref().map(|s| hmac_sha256(s, &payload));

            let req = ureq::post(&url).set("Content-Type", "application/json");
            let req = match sig {
                Some(ref s) => req.set("X-SRGAN-Signature", &format!("sha256={}", s)),
                None => req,
            };

            let (status_code, delivered) = match req.send_string(&payload) {
                Ok(resp) => {
                    let code = resp.status();
                    (Some(code), true)
                }
                Err(ureq::Error::Status(code, _)) => (Some(code), false),
                Err(ref e) => {
                    log::warn!("Webhook attempt {}/{} transport error: {}", attempt, total, e);
                    (None, false)
                }
            };

            log::info!(
                "Webhook attempt {}/{} → {} status={:?} delivered={}",
                attempt, total, url, status_code, delivered
            );
            on_update(attempt, status_code, delivered);

            if delivered {
                return;
            }

            if attempt < total {
                // Exponential backoff: 5s, 10s, 20s, … (capped at 64× base)
                let delay = retry_delay_secs * (1u64 << (attempt - 1).min(6));
                std::thread::sleep(Duration::from_secs(delay));
            }
        }
    });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
