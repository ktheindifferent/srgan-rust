//! Priority job queue for async upscaling jobs.
//!
//! - Enterprise jobs are processed before Pro, which are processed before Free.
//! - Jobs running longer than 5 minutes are cancelled with `TimedOut`.
//! - Completed / failed / timed-out jobs are purged after 1 hour.
//! - On completion, an optional webhook URL receives a POST notification.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::auth::KeyTier;

// ── Constants ─────────────────────────────────────────────────────────────────

const JOB_TIMEOUT: Duration = Duration::from_secs(5 * 60);      // 5 minutes
const JOB_RETAIN: Duration = Duration::from_secs(60 * 60);       // 1 hour

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
    /// Optional URL to POST on completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_url: Option<String>,
}

impl JobRecord {
    pub fn new(
        input_data: String,
        api_key: String,
        model: String,
        priority: JobPriority,
        webhook_url: Option<String>,
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
            webhook_url,
        }
    }
}

/// Request body for POST /api/v1/upscale/async
#[derive(Debug, Deserialize)]
pub struct AsyncUpscaleRequest {
    pub image_data: String,
    pub model: Option<String>,
    pub format: Option<String>,
    pub webhook_url: Option<String>,
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
    jobs: Mutex<HashMap<String, JobRecord>>,
    /// Monotonic clock for timeout tracking (job_id → start instant)
    timers: Mutex<HashMap<String, Instant>>,
}

impl PriorityJobQueue {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            jobs: Mutex::new(HashMap::new()),
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
        let webhook = {
            let mut jobs = self.jobs.lock().unwrap();
            if let Some(job) = jobs.get_mut(id) {
                job.status = JobStatus::Completed;
                job.result_data = Some(result_data);
                job.completed_at = Some(unix_now());
                job.webhook_url.clone()
            } else {
                None
            }
        };
        self.timers.lock().unwrap().remove(id);
        if let Some(url) = webhook {
            fire_webhook(&url, id, "completed");
        }
    }

    /// Mark a job as failed and fire the webhook if set.
    pub fn fail(&self, id: &str, reason: String) {
        let webhook = {
            let mut jobs = self.jobs.lock().unwrap();
            if let Some(job) = jobs.get_mut(id) {
                job.status = JobStatus::Failed(reason);
                job.completed_at = Some(unix_now());
                job.webhook_url.clone()
            } else {
                None
            }
        };
        self.timers.lock().unwrap().remove(id);
        if let Some(url) = webhook {
            fire_webhook(&url, id, "failed");
        }
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

fn fire_webhook(url: &str, job_id: &str, status: &str) {
    let url = url.to_string();
    let job_id = job_id.to_string();
    let status = status.to_string();
    std::thread::spawn(move || {
        let body = serde_json::json!({
            "job_id": job_id,
            "status": status,
            "timestamp": unix_now(),
        });
        let _ = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body.to_string());
    });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
