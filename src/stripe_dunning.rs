//! Stripe dunning integration — webhook handlers for payment lifecycle events.
//!
//! Connects Stripe webhook events to the rate-limit dashboard and billing DB:
//!
//! - `invoice.payment_failed` → downgrade to free tier, mark `payment_failed`, send dunning email
//! - `invoice.paid` / `customer.subscription.updated` → restore limits, clear `payment_failed`
//!
//! ## Dunning Schedule
//!
//! - **Day 1**: Warning email, reduce to free-tier limits
//! - **Day 7**: Soft limit — further reduce rate limits
//! - **Day 14**: Suspend API key entirely

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── Dunning record ──────────────────────────────────────────────────────────

/// Tracks the dunning state for a single customer/API key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StripeDunningRecord {
    pub api_key: String,
    pub customer_id: String,
    pub invoice_id: String,
    pub original_plan: String,
    pub state: DunningPhase,
    pub started_at: u64,
    pub last_notified_at: u64,
    pub emails_sent: Vec<String>,
}

/// Dunning lifecycle phases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DunningPhase {
    /// Day 1: payment failed, warning sent, downgraded to free limits.
    Warning,
    /// Day 7: soft limit — further reduced rate limits.
    SoftLimit,
    /// Day 14: API key suspended entirely.
    Suspended,
    /// Payment recovered — account restored.
    Resolved,
}

impl DunningPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            DunningPhase::Warning => "warning",
            DunningPhase::SoftLimit => "soft_limit",
            DunningPhase::Suspended => "suspended",
            DunningPhase::Resolved => "resolved",
        }
    }
}

// ── Dunning actions ─────────────────────────────────────────────────────────

/// Actions emitted when processing dunning events.
#[derive(Debug, Clone)]
pub enum StripeDunningAction {
    /// Key downgraded to free tier, warning email sent.
    Downgraded {
        api_key: String,
        from_plan: String,
    },
    /// Rate limits further reduced (day 7).
    SoftLimited {
        api_key: String,
    },
    /// API key suspended (day 14).
    KeySuspended {
        api_key: String,
    },
    /// Payment recovered, limits restored.
    Restored {
        api_key: String,
        to_plan: String,
    },
    /// Email notification sent (or logged).
    EmailSent {
        api_key: String,
        subject: String,
    },
}

// ── Dunning manager ─────────────────────────────────────────────────────────

/// Manages the Stripe dunning lifecycle for API keys with failed payments.
pub struct StripeDunningManager {
    pub records: HashMap<String, StripeDunningRecord>,
}

impl StripeDunningManager {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Handle `invoice.payment_failed` webhook event.
    ///
    /// - Downgrades the key to free-tier limits
    /// - Marks key as `payment_failed`
    /// - Logs a dunning email
    ///
    /// Returns a list of actions taken.
    pub fn on_payment_failed(
        &mut self,
        api_key: &str,
        customer_id: &str,
        invoice_id: &str,
        current_plan: &str,
    ) -> Vec<StripeDunningAction> {
        let now = unix_now();
        let mut actions = Vec::new();

        // If already tracking, just update the record
        if let Some(record) = self.records.get_mut(api_key) {
            record.invoice_id = invoice_id.to_string();
            return actions;
        }

        let subject = format!(
            "[SRGAN] Payment failed — your {} plan has been downgraded",
            current_plan
        );
        log::warn!(
            "[STRIPE DUNNING] Payment failed for key={} customer={} invoice={}",
            api_key,
            customer_id,
            invoice_id
        );
        log::info!("[DUNNING EMAIL] To: {} — Subject: {}", customer_id, subject);

        let record = StripeDunningRecord {
            api_key: api_key.to_string(),
            customer_id: customer_id.to_string(),
            invoice_id: invoice_id.to_string(),
            original_plan: current_plan.to_string(),
            state: DunningPhase::Warning,
            started_at: now,
            last_notified_at: now,
            emails_sent: vec![subject.clone()],
        };

        self.records.insert(api_key.to_string(), record);

        actions.push(StripeDunningAction::Downgraded {
            api_key: api_key.to_string(),
            from_plan: current_plan.to_string(),
        });
        actions.push(StripeDunningAction::EmailSent {
            api_key: api_key.to_string(),
            subject,
        });

        actions
    }

    /// Handle `invoice.paid` or `customer.subscription.updated` webhook event.
    ///
    /// - Restores the key to its original plan limits
    /// - Clears the `payment_failed` flag
    /// - Logs a recovery email
    pub fn on_payment_recovered(
        &mut self,
        api_key: &str,
    ) -> Vec<StripeDunningAction> {
        let mut actions = Vec::new();

        let original_plan = if let Some(record) = self.records.get_mut(api_key) {
            record.state = DunningPhase::Resolved;
            let plan = record.original_plan.clone();

            let subject = "[SRGAN] Payment received — your account is restored".to_string();
            log::info!(
                "[DUNNING EMAIL] To: {} — Subject: {}",
                record.customer_id,
                subject
            );
            record.emails_sent.push(subject.clone());

            actions.push(StripeDunningAction::EmailSent {
                api_key: api_key.to_string(),
                subject,
            });

            plan
        } else {
            // Not in dunning — nothing to do
            return actions;
        };

        actions.push(StripeDunningAction::Restored {
            api_key: api_key.to_string(),
            to_plan: original_plan,
        });

        // Remove the resolved record
        self.records.remove(api_key);

        actions
    }

    /// Process the dunning schedule — escalate records that have aged past
    /// day 7 or day 14 thresholds.
    pub fn process_schedule(&mut self) -> Vec<StripeDunningAction> {
        let now = unix_now();
        let mut actions = Vec::new();
        let keys: Vec<String> = self.records.keys().cloned().collect();

        for key in keys {
            let record = match self.records.get_mut(&key) {
                Some(r) => r,
                None => continue,
            };

            if record.state == DunningPhase::Resolved {
                continue;
            }

            let days_elapsed = (now.saturating_sub(record.started_at)) / 86400;

            // Day 14+: suspend
            if days_elapsed >= 14 && record.state != DunningPhase::Suspended {
                record.state = DunningPhase::Suspended;
                let subject =
                    "[SRGAN] Your account has been suspended due to non-payment".to_string();
                log::info!(
                    "[DUNNING EMAIL] To: {} — Subject: {}",
                    record.customer_id,
                    subject
                );
                record.emails_sent.push(subject.clone());
                record.last_notified_at = now;

                actions.push(StripeDunningAction::KeySuspended {
                    api_key: key.clone(),
                });
                actions.push(StripeDunningAction::EmailSent {
                    api_key: key.clone(),
                    subject,
                });
            }
            // Day 7+: soft limit
            else if days_elapsed >= 7 && record.state == DunningPhase::Warning {
                record.state = DunningPhase::SoftLimit;
                let subject =
                    "[SRGAN] Urgent: your account will be suspended in 7 days".to_string();
                log::info!(
                    "[DUNNING EMAIL] To: {} — Subject: {}",
                    record.customer_id,
                    subject
                );
                record.emails_sent.push(subject.clone());
                record.last_notified_at = now;

                actions.push(StripeDunningAction::SoftLimited {
                    api_key: key.clone(),
                });
                actions.push(StripeDunningAction::EmailSent {
                    api_key: key.clone(),
                    subject,
                });
            }
        }

        actions
    }

    /// Check if a key is in active dunning (not resolved).
    pub fn is_in_dunning(&self, api_key: &str) -> bool {
        self.records
            .get(api_key)
            .map(|r| r.state != DunningPhase::Resolved)
            .unwrap_or(false)
    }

    /// Check if a key is suspended via dunning.
    pub fn is_suspended(&self, api_key: &str) -> bool {
        self.records
            .get(api_key)
            .map(|r| r.state == DunningPhase::Suspended)
            .unwrap_or(false)
    }

    /// Get dunning state for a key.
    pub fn get_state(&self, api_key: &str) -> Option<&DunningPhase> {
        self.records.get(api_key).map(|r| &r.state)
    }

    /// Get all active dunning records.
    pub fn active_records(&self) -> Vec<&StripeDunningRecord> {
        self.records
            .values()
            .filter(|r| r.state != DunningPhase::Resolved)
            .collect()
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payment_failed_creates_record() {
        let mut mgr = StripeDunningManager::new();
        let actions =
            mgr.on_payment_failed("key_abc", "cust_123", "inv_456", "pro");

        assert!(mgr.is_in_dunning("key_abc"));
        assert!(!mgr.is_suspended("key_abc"));
        assert_eq!(mgr.get_state("key_abc"), Some(&DunningPhase::Warning));
        assert!(actions.len() >= 2); // Downgraded + EmailSent
    }

    #[test]
    fn test_payment_recovered_clears_dunning() {
        let mut mgr = StripeDunningManager::new();
        mgr.on_payment_failed("key_abc", "cust_123", "inv_456", "pro");
        let actions = mgr.on_payment_recovered("key_abc");

        assert!(!mgr.is_in_dunning("key_abc"));
        assert!(actions.iter().any(|a| matches!(a, StripeDunningAction::Restored { .. })));
    }

    #[test]
    fn test_duplicate_payment_failed_no_double_record() {
        let mut mgr = StripeDunningManager::new();
        mgr.on_payment_failed("key_abc", "cust_123", "inv_456", "pro");
        let actions = mgr.on_payment_failed("key_abc", "cust_123", "inv_789", "pro");
        assert!(actions.is_empty()); // No new actions on duplicate
        assert_eq!(mgr.records.len(), 1);
    }

    #[test]
    fn test_recovery_without_dunning_is_noop() {
        let mut mgr = StripeDunningManager::new();
        let actions = mgr.on_payment_recovered("nonexistent");
        assert!(actions.is_empty());
    }

    #[test]
    fn test_dunning_phase_as_str() {
        assert_eq!(DunningPhase::Warning.as_str(), "warning");
        assert_eq!(DunningPhase::SoftLimit.as_str(), "soft_limit");
        assert_eq!(DunningPhase::Suspended.as_str(), "suspended");
        assert_eq!(DunningPhase::Resolved.as_str(), "resolved");
    }

    #[test]
    fn test_is_suspended() {
        let mut mgr = StripeDunningManager::new();
        mgr.on_payment_failed("key_s", "cust_1", "inv_1", "pro");
        assert!(!mgr.is_suspended("key_s"));

        // Manually set to suspended for testing
        mgr.records.get_mut("key_s").unwrap().state = DunningPhase::Suspended;
        assert!(mgr.is_suspended("key_s"));
    }

    #[test]
    fn test_active_records() {
        let mut mgr = StripeDunningManager::new();
        mgr.on_payment_failed("key_1", "c1", "i1", "pro");
        mgr.on_payment_failed("key_2", "c2", "i2", "pro");
        mgr.on_payment_recovered("key_1");

        let active = mgr.active_records();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].api_key, "key_2");
    }
}
