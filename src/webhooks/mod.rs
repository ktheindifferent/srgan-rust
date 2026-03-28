//! Webhook delivery system with enterprise-grade reliability.
//!
//! Users register webhook URLs via `POST /api/v1/webhooks` and receive signed
//! JSON payloads for events: `job.completed`, `job.failed`, `batch.completed`.
//!
//! - HMAC-SHA256 signature in `X-SRGAN-Signature` header
//! - Exponential backoff retry (5 attempts: 1m → 5m → 15m → 1h → 4h)
//! - Dead letter queue for permanently failed deliveries
//! - `GET /api/v1/webhooks` — list registered webhooks
//! - `DELETE /api/v1/webhooks/:id` — remove a webhook
//! - `POST /api/v1/webhooks/test` — fire a test ping

pub mod stream;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Retry delays in seconds: 1m, 5m, 15m, 1h, 4h.
const RETRY_DELAYS_SECS: [u64; 5] = [60, 300, 900, 3600, 14400];
/// Maximum delivery attempts before dead-lettering.
const MAX_DELIVERY_ATTEMPTS: u32 = 5;

// ── Types ────────────────────────────────────────────────────────────────────

/// Events that can trigger webhook delivery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WebhookEvent {
    #[serde(rename = "job.completed")]
    JobCompleted,
    #[serde(rename = "job.failed")]
    JobFailed,
    #[serde(rename = "job.progress")]
    JobProgress,
    #[serde(rename = "batch.completed")]
    BatchCompleted,
    #[serde(rename = "org.quota_warning")]
    OrgQuotaWarning,
}

impl std::fmt::Display for WebhookEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobCompleted => write!(f, "job.completed"),
            Self::JobFailed => write!(f, "job.failed"),
            Self::JobProgress => write!(f, "job.progress"),
            Self::BatchCompleted => write!(f, "batch.completed"),
            Self::OrgQuotaWarning => write!(f, "org.quota_warning"),
        }
    }
}

/// Request to register a webhook.
#[derive(Debug, Deserialize)]
pub struct RegisterWebhookRequest {
    /// URL to POST event payloads to.
    pub url: String,
    /// Secret used for HMAC-SHA256 signing (optional but recommended).
    #[serde(default)]
    pub secret: Option<String>,
    /// Events to subscribe to (default: all).
    #[serde(default = "default_events")]
    pub events: Vec<WebhookEvent>,
}

fn default_events() -> Vec<WebhookEvent> {
    vec![
        WebhookEvent::JobCompleted,
        WebhookEvent::JobFailed,
        WebhookEvent::JobProgress,
        WebhookEvent::BatchCompleted,
        WebhookEvent::OrgQuotaWarning,
    ]
}

/// A registered webhook endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct WebhookEndpoint {
    pub id: String,
    pub url: String,
    #[serde(skip_serializing)]
    pub secret: Option<String>,
    pub events: Vec<WebhookEvent>,
    pub api_key: String,
    pub created_at: u64,
    /// Number of successful deliveries.
    pub deliveries_success: u64,
    /// Number of failed deliveries.
    pub deliveries_failed: u64,
}

/// Webhook delivery attempt record.
#[derive(Debug, Clone, Serialize)]
pub struct DeliveryAttempt {
    pub webhook_id: String,
    pub event: String,
    pub attempt: u32,
    pub status_code: Option<u16>,
    pub success: bool,
    pub timestamp: u64,
    pub error_message: Option<String>,
}

/// A failed delivery that has exhausted all retries.
#[derive(Debug, Clone, Serialize)]
pub struct DeadLetterEntry {
    pub id: String,
    pub webhook_id: String,
    pub endpoint_url: String,
    pub event: String,
    pub payload: String,
    pub attempts: Vec<DeliveryAttempt>,
    pub created_at: u64,
    pub dead_lettered_at: u64,
}

/// Payload sent to a webhook endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct WebhookPayload {
    pub event: String,
    pub timestamp: u64,
    pub data: serde_json::Value,
}

/// Response to `POST /api/v1/webhooks`.
#[derive(Debug, Serialize)]
pub struct RegisterWebhookResponse {
    pub id: String,
    pub url: String,
    pub events: Vec<WebhookEvent>,
    pub created_at: u64,
}

// ── Webhook store ────────────────────────────────────────────────────────────

/// Thread-safe in-memory webhook registry with dead letter queue.
pub struct WebhookStore {
    endpoints: Mutex<HashMap<String, WebhookEndpoint>>,
    dead_letter_queue: Mutex<Vec<DeadLetterEntry>>,
}

impl WebhookStore {
    pub fn new() -> Self {
        Self {
            endpoints: Mutex::new(HashMap::new()),
            dead_letter_queue: Mutex::new(Vec::new()),
        }
    }

    /// Register a new webhook endpoint.
    pub fn register(
        &self,
        req: &RegisterWebhookRequest,
        api_key: &str,
    ) -> RegisterWebhookResponse {
        let id = format!("wh_{}", Uuid::new_v4());
        let now = unix_now();

        let endpoint = WebhookEndpoint {
            id: id.clone(),
            url: req.url.clone(),
            secret: req.secret.clone(),
            events: req.events.clone(),
            api_key: api_key.to_string(),
            created_at: now,
            deliveries_success: 0,
            deliveries_failed: 0,
        };

        if let Ok(mut eps) = self.endpoints.lock() {
            eps.insert(id.clone(), endpoint);
        }

        RegisterWebhookResponse {
            id,
            url: req.url.clone(),
            events: req.events.clone(),
            created_at: now,
        }
    }

    /// List all webhooks for a given API key.
    pub fn list_for_key(&self, api_key: &str) -> Vec<WebhookEndpoint> {
        self.endpoints
            .lock()
            .ok()
            .map(|eps| {
                eps.values()
                    .filter(|ep| ep.api_key == api_key)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove a webhook by ID (only if owned by the given API key).
    pub fn remove(&self, webhook_id: &str, api_key: &str) -> bool {
        if let Ok(mut eps) = self.endpoints.lock() {
            if let Some(ep) = eps.get(webhook_id) {
                if ep.api_key == api_key {
                    eps.remove(webhook_id);
                    return true;
                }
            }
        }
        false
    }

    /// Get all endpoints subscribed to a given event for a given API key.
    pub fn endpoints_for_event(
        &self,
        api_key: &str,
        event: &WebhookEvent,
    ) -> Vec<WebhookEndpoint> {
        self.endpoints
            .lock()
            .ok()
            .map(|eps| {
                eps.values()
                    .filter(|ep| ep.api_key == api_key && ep.events.contains(event))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Record a delivery result.
    pub fn record_delivery(&self, webhook_id: &str, success: bool) {
        if let Ok(mut eps) = self.endpoints.lock() {
            if let Some(ep) = eps.get_mut(webhook_id) {
                if success {
                    ep.deliveries_success += 1;
                } else {
                    ep.deliveries_failed += 1;
                }
            }
        }
    }

    /// Add a failed delivery to the dead letter queue.
    pub fn dead_letter(
        &self,
        webhook_id: &str,
        endpoint_url: &str,
        event: &str,
        payload: &str,
        attempts: Vec<DeliveryAttempt>,
    ) {
        let entry = DeadLetterEntry {
            id: format!("dlq_{}", Uuid::new_v4()),
            webhook_id: webhook_id.to_string(),
            endpoint_url: endpoint_url.to_string(),
            event: event.to_string(),
            payload: payload.to_string(),
            created_at: attempts.first().map(|a| a.timestamp).unwrap_or(0),
            dead_lettered_at: unix_now(),
            attempts,
        };
        if let Ok(mut dlq) = self.dead_letter_queue.lock() {
            dlq.push(entry);
        }
    }

    /// List dead letter queue entries for a given API key's webhooks.
    pub fn list_dead_letters(&self, api_key: &str) -> Vec<DeadLetterEntry> {
        let webhook_ids: Vec<String> = self.endpoints
            .lock()
            .ok()
            .map(|eps| {
                eps.values()
                    .filter(|ep| ep.api_key == api_key)
                    .map(|ep| ep.id.clone())
                    .collect()
            })
            .unwrap_or_default();

        self.dead_letter_queue
            .lock()
            .ok()
            .map(|dlq| {
                dlq.iter()
                    .filter(|entry| webhook_ids.contains(&entry.webhook_id))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Count of entries in the dead letter queue.
    pub fn dead_letter_count(&self) -> usize {
        self.dead_letter_queue.lock().ok().map(|dlq| dlq.len()).unwrap_or(0)
    }
}

// ── Delivery logic ───────────────────────────────────────────────────────────

/// Compute HMAC-SHA256 signature for a payload.
pub fn compute_signature(secret: &str, payload: &str) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(payload.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Deliver a webhook payload to an endpoint with exponential backoff retry.
///
/// Retry schedule: 1m → 5m → 15m → 1h → 4h (5 attempts total).
/// Failed deliveries after all retries are moved to the dead letter queue.
///
/// Spawns a background thread; does not block the caller.
pub fn deliver_webhook_event(
    endpoint: WebhookEndpoint,
    event: WebhookEvent,
    data: serde_json::Value,
    store: std::sync::Arc<WebhookStore>,
) {
    std::thread::spawn(move || {
        let payload = WebhookPayload {
            event: event.to_string(),
            timestamp: unix_now(),
            data,
        };

        let body = match serde_json::to_string(&payload) {
            Ok(b) => b,
            Err(_) => return,
        };

        let mut attempts: Vec<DeliveryAttempt> = Vec::new();

        for attempt_idx in 0..MAX_DELIVERY_ATTEMPTS {
            let mut req = ureq::post(&endpoint.url)
                .set("Content-Type", "application/json")
                .set("User-Agent", "srgan-rust-webhooks/0.2.0");

            // Sign if secret is provided
            if let Some(ref secret) = endpoint.secret {
                let sig = compute_signature(secret, &body);
                req = req.set("X-SRGAN-Signature", &format!("sha256={}", sig));
            }

            match req.send_string(&body) {
                Ok(resp) => {
                    let status = resp.status();
                    let attempt = DeliveryAttempt {
                        webhook_id: endpoint.id.clone(),
                        event: event.to_string(),
                        attempt: attempt_idx + 1,
                        status_code: Some(status),
                        success: status >= 200 && status < 300,
                        timestamp: unix_now(),
                        error_message: None,
                    };
                    let success = attempt.success;
                    attempts.push(attempt);

                    if success {
                        store.record_delivery(&endpoint.id, true);
                        return;
                    }
                    // Non-2xx — retry
                }
                Err(e) => {
                    attempts.push(DeliveryAttempt {
                        webhook_id: endpoint.id.clone(),
                        event: event.to_string(),
                        attempt: attempt_idx + 1,
                        status_code: None,
                        success: false,
                        timestamp: unix_now(),
                        error_message: Some(e.to_string()),
                    });
                }
            }

            // Wait before next retry (if not the last attempt)
            if (attempt_idx as usize) < RETRY_DELAYS_SECS.len() {
                let delay = RETRY_DELAYS_SECS[attempt_idx as usize];
                std::thread::sleep(std::time::Duration::from_secs(delay));
            }
        }

        // All attempts failed — dead letter the payload
        store.record_delivery(&endpoint.id, false);
        store.dead_letter(
            &endpoint.id,
            &endpoint.url,
            &event.to_string(),
            &body,
            attempts,
        );
    });
}

/// Fire webhooks for a given event + API key combination.
pub fn fire_webhooks(
    store: &std::sync::Arc<WebhookStore>,
    api_key: &str,
    event: WebhookEvent,
    data: serde_json::Value,
) {
    let endpoints = store.endpoints_for_event(api_key, &event);
    for ep in endpoints {
        deliver_webhook_event(ep, event.clone(), data.clone(), std::sync::Arc::clone(store));
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── SQLite delivery log ──────────────────────────────────────────────────

/// Persistent delivery log backed by SQLite.
///
/// Stores every delivery attempt so operators can audit webhook reliability.
/// Table: `webhook_delivery_log` with columns:
///   id, webhook_id, event, attempt, status_code, success, error_message, timestamp
pub struct DeliveryLog {
    conn: Mutex<rusqlite::Connection>,
}

impl DeliveryLog {
    /// Open (or create) the delivery log database at the given path.
    pub fn open(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = rusqlite::Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS webhook_delivery_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                webhook_id    TEXT NOT NULL,
                event         TEXT NOT NULL,
                attempt       INTEGER NOT NULL,
                status_code   INTEGER,
                success       INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                payload       TEXT,
                timestamp     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_wdl_webhook_id ON webhook_delivery_log(webhook_id);
            CREATE INDEX IF NOT EXISTS idx_wdl_timestamp ON webhook_delivery_log(timestamp);",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory delivery log (useful for tests).
    pub fn open_in_memory() -> Result<Self, rusqlite::Error> {
        Self::open(":memory:")
    }

    /// Record a delivery attempt.
    pub fn record(&self, attempt: &DeliveryAttempt, payload: Option<&str>) {
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "INSERT INTO webhook_delivery_log (webhook_id, event, attempt, status_code, success, error_message, payload, timestamp)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    attempt.webhook_id,
                    attempt.event,
                    attempt.attempt,
                    attempt.status_code,
                    attempt.success as i32,
                    attempt.error_message,
                    payload,
                    attempt.timestamp,
                ],
            );
        }
    }

    /// Query recent delivery attempts for a given webhook.
    pub fn list_for_webhook(&self, webhook_id: &str, limit: usize) -> Vec<DeliveryAttempt> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return vec![],
        };
        let mut stmt = match conn.prepare(
            "SELECT webhook_id, event, attempt, status_code, success, error_message, timestamp
             FROM webhook_delivery_log
             WHERE webhook_id = ?1
             ORDER BY timestamp DESC
             LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        stmt.query_map(rusqlite::params![webhook_id, limit as i64], |row| {
            Ok(DeliveryAttempt {
                webhook_id: row.get(0)?,
                event: row.get(1)?,
                attempt: row.get(2)?,
                status_code: row.get(3)?,
                success: row.get::<_, i32>(4)? != 0,
                error_message: row.get(5)?,
                timestamp: row.get(6)?,
            })
        })
        .ok()
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Count total delivery attempts.
    pub fn count(&self) -> usize {
        self.conn
            .lock()
            .ok()
            .and_then(|c| {
                c.query_row("SELECT COUNT(*) FROM webhook_delivery_log", [], |row| {
                    row.get::<_, i64>(0)
                })
                .ok()
            })
            .unwrap_or(0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_list() {
        let store = WebhookStore::new();
        let req = RegisterWebhookRequest {
            url: "https://example.com/hook".into(),
            secret: Some("s3cret".into()),
            events: vec![WebhookEvent::JobCompleted],
        };

        let resp = store.register(&req, "key-1");
        assert!(resp.id.starts_with("wh_"));
        assert_eq!(resp.events.len(), 1);

        let list = store.list_for_key("key-1");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].url, "https://example.com/hook");
    }

    #[test]
    fn test_remove_webhook() {
        let store = WebhookStore::new();
        let req = RegisterWebhookRequest {
            url: "https://example.com/hook".into(),
            secret: None,
            events: default_events(),
        };

        let resp = store.register(&req, "key-1");
        assert!(store.remove(&resp.id, "key-1"));
        assert!(store.list_for_key("key-1").is_empty());
    }

    #[test]
    fn test_remove_wrong_owner() {
        let store = WebhookStore::new();
        let req = RegisterWebhookRequest {
            url: "https://example.com/hook".into(),
            secret: None,
            events: default_events(),
        };

        let resp = store.register(&req, "key-1");
        assert!(!store.remove(&resp.id, "key-2")); // wrong owner
        assert_eq!(store.list_for_key("key-1").len(), 1);
    }

    #[test]
    fn test_compute_signature() {
        let sig = compute_signature("secret", "hello");
        assert!(!sig.is_empty());
        // Deterministic
        assert_eq!(sig, compute_signature("secret", "hello"));
    }

    #[test]
    fn test_endpoints_for_event_filter() {
        let store = WebhookStore::new();
        store.register(
            &RegisterWebhookRequest {
                url: "https://a.com/hook".into(),
                secret: None,
                events: vec![WebhookEvent::JobCompleted],
            },
            "key-1",
        );
        store.register(
            &RegisterWebhookRequest {
                url: "https://b.com/hook".into(),
                secret: None,
                events: vec![WebhookEvent::BatchCompleted],
            },
            "key-1",
        );

        let eps = store.endpoints_for_event("key-1", &WebhookEvent::JobCompleted);
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].url, "https://a.com/hook");
    }

    #[test]
    fn test_dead_letter_queue() {
        let store = WebhookStore::new();
        let req = RegisterWebhookRequest {
            url: "https://example.com/hook".into(),
            secret: None,
            events: vec![WebhookEvent::JobCompleted],
        };
        let resp = store.register(&req, "key-1");

        assert_eq!(store.dead_letter_count(), 0);
        store.dead_letter(
            &resp.id,
            "https://example.com/hook",
            "job.completed",
            r#"{"event":"job.completed"}"#,
            vec![DeliveryAttempt {
                webhook_id: resp.id.clone(),
                event: "job.completed".into(),
                attempt: 1,
                status_code: Some(500),
                success: false,
                timestamp: unix_now(),
                error_message: None,
            }],
        );
        assert_eq!(store.dead_letter_count(), 1);

        let dlq = store.list_dead_letters("key-1");
        assert_eq!(dlq.len(), 1);
        assert_eq!(dlq[0].webhook_id, resp.id);
    }

    #[test]
    fn test_retry_delays_config() {
        assert_eq!(RETRY_DELAYS_SECS.len(), 5);
        assert_eq!(RETRY_DELAYS_SECS[0], 60);   // 1 minute
        assert_eq!(RETRY_DELAYS_SECS[1], 300);  // 5 minutes
        assert_eq!(RETRY_DELAYS_SECS[2], 900);  // 15 minutes
        assert_eq!(RETRY_DELAYS_SECS[3], 3600); // 1 hour
        assert_eq!(RETRY_DELAYS_SECS[4], 14400); // 4 hours
        assert_eq!(MAX_DELIVERY_ATTEMPTS, 5);
    }

    #[test]
    fn test_delivery_log_record_and_query() {
        let log = DeliveryLog::open_in_memory().unwrap();
        assert_eq!(log.count(), 0);

        let attempt = DeliveryAttempt {
            webhook_id: "wh_test".into(),
            event: "job.completed".into(),
            attempt: 1,
            status_code: Some(200),
            success: true,
            timestamp: unix_now(),
            error_message: None,
        };
        log.record(&attempt, Some(r#"{"event":"job.completed"}"#));
        assert_eq!(log.count(), 1);

        let rows = log.list_for_webhook("wh_test", 10);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].webhook_id, "wh_test");
        assert!(rows[0].success);
    }

    #[test]
    fn test_delivery_log_failed_attempt() {
        let log = DeliveryLog::open_in_memory().unwrap();
        let attempt = DeliveryAttempt {
            webhook_id: "wh_fail".into(),
            event: "job.failed".into(),
            attempt: 3,
            status_code: Some(500),
            success: false,
            timestamp: unix_now(),
            error_message: Some("Internal Server Error".into()),
        };
        log.record(&attempt, None);
        let rows = log.list_for_webhook("wh_fail", 10);
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].success);
        assert_eq!(rows[0].error_message.as_deref(), Some("Internal Server Error"));
    }
}
