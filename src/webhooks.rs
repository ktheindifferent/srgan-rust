//! Webhook delivery system.
//!
//! Users register webhook URLs via `POST /api/v1/webhooks` and receive signed
//! JSON payloads for events: `job.completed`, `job.failed`, `batch.completed`.
//!
//! - HMAC-SHA256 signature in `X-SRGAN-Signature` header
//! - Exponential backoff retry (3 attempts by default)
//! - `GET /api/v1/webhooks` — list registered webhooks
//! - `DELETE /api/v1/webhooks/:id` — remove a webhook

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Types ────────────────────────────────────────────────────────────────────

/// Events that can trigger webhook delivery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WebhookEvent {
    #[serde(rename = "job.completed")]
    JobCompleted,
    #[serde(rename = "job.failed")]
    JobFailed,
    #[serde(rename = "batch.completed")]
    BatchCompleted,
}

impl std::fmt::Display for WebhookEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobCompleted => write!(f, "job.completed"),
            Self::JobFailed => write!(f, "job.failed"),
            Self::BatchCompleted => write!(f, "batch.completed"),
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
        WebhookEvent::BatchCompleted,
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

/// Thread-safe in-memory webhook registry.
pub struct WebhookStore {
    endpoints: Mutex<HashMap<String, WebhookEndpoint>>,
}

impl WebhookStore {
    pub fn new() -> Self {
        Self {
            endpoints: Mutex::new(HashMap::new()),
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
}

// ── Delivery logic ───────────────────────────────────────────────────────────

/// Compute HMAC-SHA256 signature for a payload.
pub fn compute_signature(secret: &str, payload: &str) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(payload.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Deliver a webhook payload to an endpoint with retry logic.
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

        let max_attempts: u32 = 3;
        let mut delay_secs: u64 = 5;

        for attempt in 0..max_attempts {
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
                    if status >= 200 && status < 300 {
                        store.record_delivery(&endpoint.id, true);
                        return;
                    }
                    // Non-2xx — retry
                }
                Err(_) => {
                    // Transport error — retry
                }
            }

            if attempt + 1 < max_attempts {
                std::thread::sleep(std::time::Duration::from_secs(delay_secs));
                delay_secs *= 2; // exponential backoff
            }
        }

        // All attempts failed
        store.record_delivery(&endpoint.id, false);
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
}
