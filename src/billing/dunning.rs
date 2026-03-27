//! Stripe dunning engine — automated payment retry with exponential back-off.

use std::collections::HashMap;
use std::env;

/// Configures retry cadence and grace period.
#[derive(Debug, Clone)]
pub struct DunningConfig {
    /// Days after each failed attempt before retrying (e.g. [1, 3, 7]).
    pub retry_intervals_days: Vec<u32>,
    /// Maximum number of retry attempts before giving up.
    pub max_retries: u8,
    /// Days of grace before the account is suspended after exhaustion.
    pub grace_period_days: u32,
}

impl Default for DunningConfig {
    fn default() -> Self {
        Self {
            retry_intervals_days: vec![1, 3, 7],
            max_retries: 3,
            grace_period_days: 3,
        }
    }
}

/// Tracks a single invoice through the dunning lifecycle.
#[derive(Debug, Clone)]
pub struct DunningRecord {
    pub customer_id: String,
    pub invoice_id: String,
    pub attempt: u8,
    /// Unix timestamp of next scheduled retry.
    pub next_retry_at: u64,
    pub last_error: Option<String>,
    pub status: DunningStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DunningStatus {
    Active,
    Resolved,
    Exhausted,
    GracePeriod,
}

/// Core dunning engine: tracks failed invoices and retries payments via Stripe.
pub struct DunningEngine {
    pub records: HashMap<String, DunningRecord>,
    pub config: DunningConfig,
}

impl DunningEngine {
    pub fn new(config: DunningConfig) -> Self {
        Self {
            records: HashMap::new(),
            config,
        }
    }

    /// Called when a Stripe `invoice.payment_failed` event arrives.
    pub fn on_payment_failed(&mut self, customer_id: &str, invoice_id: &str, error: &str) {
        let interval_secs = self
            .config
            .retry_intervals_days
            .first()
            .copied()
            .unwrap_or(1) as u64
            * 86400;

        let now = unix_now();
        let record = DunningRecord {
            customer_id: customer_id.to_string(),
            invoice_id: invoice_id.to_string(),
            attempt: 1,
            next_retry_at: now + interval_secs,
            last_error: Some(error.to_string()),
            status: DunningStatus::Active,
        };
        self.records.insert(customer_id.to_string(), record);
    }

    /// Called when payment succeeds — clears the dunning record.
    pub fn on_payment_succeeded(&mut self, customer_id: &str, _invoice_id: &str) {
        if let Some(record) = self.records.get_mut(customer_id) {
            record.status = DunningStatus::Resolved;
        }
    }

    /// Returns all records whose retry is due (next_retry_at <= now and still Active).
    pub fn get_due_retries(&self, now: u64) -> Vec<DunningRecord> {
        self.records
            .values()
            .filter(|r| r.status == DunningStatus::Active && r.next_retry_at <= now)
            .cloned()
            .collect()
    }

    /// POST to Stripe to retry an invoice payment.
    pub fn retry_payment(&self, record: &DunningRecord) -> Result<(), String> {
        let secret = env::var("STRIPE_SECRET_KEY")
            .map_err(|_| "STRIPE_SECRET_KEY not set".to_string())?;

        let url = format!(
            "https://api.stripe.com/v1/invoices/{}/pay",
            record.invoice_id
        );

        let client = reqwest::blocking::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", secret))
            .send()
            .map_err(|e| format!("Stripe request failed: {}", e))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let body: serde_json::Value = response
                .json()
                .unwrap_or_else(|_| serde_json::json!({"error": {"message": "unknown"}}));
            let msg = body["error"]["message"]
                .as_str()
                .unwrap_or("Unknown Stripe error");
            Err(format!("Stripe error: {}", msg))
        }
    }

    /// Increment the attempt counter; if max retries exceeded, mark Exhausted.
    pub fn advance_or_exhaust(&mut self, customer_id: &str) {
        let max = self.config.max_retries;
        let intervals = &self.config.retry_intervals_days;
        let now = unix_now();

        if let Some(record) = self.records.get_mut(customer_id) {
            record.attempt += 1;
            if record.attempt >= max {
                record.status = DunningStatus::Exhausted;
            } else {
                let idx = (record.attempt as usize).min(intervals.len() - 1);
                let interval_secs = intervals[idx] as u64 * 86400;
                record.next_retry_at = now + interval_secs;
            }
        }
    }
}

fn unix_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
