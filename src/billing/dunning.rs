//! Stripe dunning engine — automated payment retry with scheduled escalation.
//!
//! Handles `charge.failed` and `invoice.payment_failed` webhook events with a
//! configurable dunning schedule:
//!
//! - **Day 0**: Notify customer of failed payment
//! - **Day 3**: Retry payment, send reminder
//! - **Day 7**: Retry payment + escalate (warning email)
//! - **Day 14**: Suspend account, send final notice
//!
//! ## DunningState lifecycle
//!
//! ```text
//! Active → PastDue(0) → PastDue(3) → PastDue(7) → PastDue(14) → Suspended → Cancelled
//!                    ↘ (payment succeeds) → Active
//! ```

use std::collections::HashMap;
use std::env;

/// The current state of an account in the dunning lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub enum DunningState {
    /// Account is in good standing.
    Active,
    /// Payment failed; `days` since first failure.
    PastDue(u32),
    /// Account suspended after exhausting all retries.
    Suspended,
    /// Account permanently cancelled (manual action required to reactivate).
    Cancelled,
}

impl DunningState {
    pub fn as_str(&self) -> &str {
        match self {
            DunningState::Active => "active",
            DunningState::PastDue(_) => "past_due",
            DunningState::Suspended => "suspended",
            DunningState::Cancelled => "cancelled",
        }
    }

    pub fn is_suspended_or_cancelled(&self) -> bool {
        matches!(self, DunningState::Suspended | DunningState::Cancelled)
    }
}

/// Dunning schedule step: what to do at each interval.
#[derive(Debug, Clone)]
pub struct DunningStep {
    /// Days after initial failure.
    pub day: u32,
    /// Whether to retry payment via Stripe.
    pub retry_payment: bool,
    /// Whether to escalate (e.g. send warning email).
    pub escalate: bool,
    /// Whether to suspend the account.
    pub suspend: bool,
    /// Email template to send.
    pub email_template: EmailTemplate,
}

/// Email template identifiers for each dunning stage.
#[derive(Debug, Clone, PartialEq)]
pub enum EmailTemplate {
    /// Day 0: "Your payment failed, please update your card"
    PaymentFailedNotice,
    /// Day 3: "Reminder: payment still outstanding"
    PaymentRetryReminder,
    /// Day 7: "Urgent: your account will be suspended soon"
    PaymentEscalationWarning,
    /// Day 14: "Your account has been suspended"
    AccountSuspendedNotice,
    /// Account reactivated after payment succeeds
    PaymentRecoveredNotice,
}

impl EmailTemplate {
    /// Return a stub email subject line.
    pub fn subject(&self) -> &'static str {
        match self {
            EmailTemplate::PaymentFailedNotice => "[SRGAN] Your payment failed — please update your card",
            EmailTemplate::PaymentRetryReminder => "[SRGAN] Reminder: payment still outstanding",
            EmailTemplate::PaymentEscalationWarning => "[SRGAN] Urgent: your account will be suspended in 7 days",
            EmailTemplate::AccountSuspendedNotice => "[SRGAN] Your account has been suspended",
            EmailTemplate::PaymentRecoveredNotice => "[SRGAN] Payment received — your account is active again",
        }
    }

    /// Return a stub email body.
    pub fn body(&self, customer_id: &str) -> String {
        match self {
            EmailTemplate::PaymentFailedNotice => format!(
                "Hi {},\n\nWe were unable to process your payment. \
                 Please update your payment method at your account settings.\n\n\
                 If you need help, reply to this email.\n\n— The SRGAN Team",
                customer_id
            ),
            EmailTemplate::PaymentRetryReminder => format!(
                "Hi {},\n\nThis is a reminder that your payment is still outstanding. \
                 We'll retry the charge automatically, but please update your card \
                 to avoid service interruption.\n\n— The SRGAN Team",
                customer_id
            ),
            EmailTemplate::PaymentEscalationWarning => format!(
                "Hi {},\n\nYour payment has been outstanding for 7 days. \
                 If we don't receive payment within 7 more days, your account \
                 will be suspended and API access will be revoked.\n\n\
                 Please update your payment method immediately.\n\n— The SRGAN Team",
                customer_id
            ),
            EmailTemplate::AccountSuspendedNotice => format!(
                "Hi {},\n\nYour account has been suspended due to non-payment. \
                 Your API keys are no longer active.\n\n\
                 To reactivate, please update your payment method and settle \
                 the outstanding balance.\n\n— The SRGAN Team",
                customer_id
            ),
            EmailTemplate::PaymentRecoveredNotice => format!(
                "Hi {},\n\nGreat news! Your payment has been processed successfully. \
                 Your account is now active and your API keys are working again.\n\n\
                 — The SRGAN Team",
                customer_id
            ),
        }
    }
}

/// Default dunning schedule.
pub fn default_dunning_schedule() -> Vec<DunningStep> {
    vec![
        DunningStep {
            day: 0,
            retry_payment: false,
            escalate: false,
            suspend: false,
            email_template: EmailTemplate::PaymentFailedNotice,
        },
        DunningStep {
            day: 3,
            retry_payment: true,
            escalate: false,
            suspend: false,
            email_template: EmailTemplate::PaymentRetryReminder,
        },
        DunningStep {
            day: 7,
            retry_payment: true,
            escalate: true,
            suspend: false,
            email_template: EmailTemplate::PaymentEscalationWarning,
        },
        DunningStep {
            day: 14,
            retry_payment: false,
            escalate: true,
            suspend: true,
            email_template: EmailTemplate::AccountSuspendedNotice,
        },
    ]
}

/// Configures retry cadence and grace period.
#[derive(Debug, Clone)]
pub struct DunningConfig {
    /// Days after each failed attempt before retrying (e.g. [1, 3, 7]).
    pub retry_intervals_days: Vec<u32>,
    /// Maximum number of retry attempts before giving up.
    pub max_retries: u8,
    /// Days of grace before the account is suspended after exhaustion.
    pub grace_period_days: u32,
    /// The dunning schedule steps.
    pub schedule: Vec<DunningStep>,
}

impl Default for DunningConfig {
    fn default() -> Self {
        Self {
            retry_intervals_days: vec![3, 7, 14],
            max_retries: 3,
            grace_period_days: 3,
            schedule: default_dunning_schedule(),
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
    /// Current dunning state (richer than status).
    pub state: DunningState,
    /// Unix timestamp when the dunning process started.
    pub started_at: u64,
    /// Whether the account_suspended flag has been set in the DB.
    pub account_suspended: bool,
    /// Email notifications sent (template name → unix timestamp).
    pub emails_sent: Vec<(EmailTemplate, u64)>,
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

    /// Called when a Stripe `invoice.payment_failed` or `charge.failed` event arrives.
    pub fn on_payment_failed(&mut self, customer_id: &str, invoice_id: &str, error: &str) {
        let now = unix_now();

        // If already tracking this customer, update error and leave state
        if let Some(record) = self.records.get_mut(customer_id) {
            record.last_error = Some(error.to_string());
            if record.invoice_id != invoice_id {
                record.invoice_id = invoice_id.to_string();
            }
            return;
        }

        let interval_secs = self
            .config
            .retry_intervals_days
            .first()
            .copied()
            .unwrap_or(3) as u64
            * 86400;

        let record = DunningRecord {
            customer_id: customer_id.to_string(),
            invoice_id: invoice_id.to_string(),
            attempt: 0,
            next_retry_at: now + interval_secs,
            last_error: Some(error.to_string()),
            status: DunningStatus::Active,
            state: DunningState::PastDue(0),
            started_at: now,
            account_suspended: false,
            emails_sent: vec![(EmailTemplate::PaymentFailedNotice, now)],
        };

        // Log email stub
        log::info!(
            "[DUNNING EMAIL] To: {} — Subject: {}",
            customer_id,
            EmailTemplate::PaymentFailedNotice.subject()
        );

        self.records.insert(customer_id.to_string(), record);
    }

    /// Called when a Stripe `charge.failed` event arrives.
    /// Same behavior as payment_failed but may carry different metadata.
    pub fn on_charge_failed(&mut self, customer_id: &str, charge_id: &str, error: &str) {
        self.on_payment_failed(customer_id, charge_id, error);
    }

    /// Called when payment succeeds — clears the dunning record and reactivates.
    pub fn on_payment_succeeded(&mut self, customer_id: &str, _invoice_id: &str) {
        if let Some(record) = self.records.get_mut(customer_id) {
            record.status = DunningStatus::Resolved;
            record.state = DunningState::Active;
            record.account_suspended = false;

            let now = unix_now();
            record.emails_sent.push((EmailTemplate::PaymentRecoveredNotice, now));

            log::info!(
                "[DUNNING EMAIL] To: {} — Subject: {}",
                customer_id,
                EmailTemplate::PaymentRecoveredNotice.subject()
            );
        }
    }

    /// Process the dunning schedule: check all records against the schedule
    /// and execute retries, escalations, and suspensions as needed.
    pub fn process_schedule(&mut self) -> Vec<DunningAction> {
        let now = unix_now();
        let mut actions = Vec::new();

        let customer_ids: Vec<String> = self.records.keys().cloned().collect();

        for customer_id in customer_ids {
            let record = match self.records.get(&customer_id) {
                Some(r) => r.clone(),
                None => continue,
            };

            if record.status != DunningStatus::Active {
                continue;
            }

            let days_elapsed = ((now - record.started_at) / 86400) as u32;

            // Find the applicable schedule step
            let applicable_step = self
                .config
                .schedule
                .iter()
                .rev()
                .find(|step| days_elapsed >= step.day);

            if let Some(step) = applicable_step {
                // Only act if we haven't already sent this step's email
                let already_sent = record
                    .emails_sent
                    .iter()
                    .any(|(tmpl, _)| *tmpl == step.email_template);

                if !already_sent {
                    // Send email
                    log::info!(
                        "[DUNNING EMAIL] To: {} — Subject: {}",
                        customer_id,
                        step.email_template.subject()
                    );

                    if let Some(r) = self.records.get_mut(&customer_id) {
                        r.emails_sent.push((step.email_template.clone(), now));
                        r.state = if step.suspend {
                            DunningState::Suspended
                        } else {
                            DunningState::PastDue(days_elapsed)
                        };
                    }

                    // Retry payment if scheduled
                    if step.retry_payment && record.next_retry_at <= now {
                        match self.retry_payment(&record) {
                            Ok(()) => {
                                actions.push(DunningAction::PaymentRetried {
                                    customer_id: customer_id.clone(),
                                    success: true,
                                });
                                // Payment succeeded
                                self.on_payment_succeeded(&customer_id, &record.invoice_id);
                                continue;
                            }
                            Err(err) => {
                                actions.push(DunningAction::PaymentRetried {
                                    customer_id: customer_id.clone(),
                                    success: false,
                                });
                                if let Some(r) = self.records.get_mut(&customer_id) {
                                    r.attempt += 1;
                                    r.last_error = Some(err);
                                    // Set next retry
                                    let idx = (r.attempt as usize)
                                        .min(self.config.retry_intervals_days.len() - 1);
                                    let interval =
                                        self.config.retry_intervals_days[idx] as u64 * 86400;
                                    r.next_retry_at = now + interval;
                                }
                            }
                        }
                    }

                    // Suspend account if scheduled
                    if step.suspend {
                        actions.push(DunningAction::AccountSuspended {
                            customer_id: customer_id.clone(),
                        });
                        if let Some(r) = self.records.get_mut(&customer_id) {
                            r.account_suspended = true;
                            r.status = DunningStatus::Exhausted;
                            r.state = DunningState::Suspended;
                        }
                    }

                    actions.push(DunningAction::EmailSent {
                        customer_id: customer_id.clone(),
                        template: step.email_template.clone(),
                    });
                }
            }
        }

        actions
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
                record.state = DunningState::Suspended;
            } else {
                let idx = (record.attempt as usize).min(intervals.len() - 1);
                let interval_secs = intervals[idx] as u64 * 86400;
                record.next_retry_at = now + interval_secs;
                let days = ((now - record.started_at) / 86400) as u32;
                record.state = DunningState::PastDue(days);
            }
        }
    }

    /// Get the dunning state for a customer. Returns Active if not in dunning.
    pub fn get_state(&self, customer_id: &str) -> DunningState {
        self.records
            .get(customer_id)
            .map(|r| r.state.clone())
            .unwrap_or(DunningState::Active)
    }

    /// Check if an account is suspended via dunning.
    pub fn is_account_suspended(&self, customer_id: &str) -> bool {
        self.records
            .get(customer_id)
            .map(|r| r.account_suspended)
            .unwrap_or(false)
    }
}

/// Actions emitted by the dunning engine during schedule processing.
#[derive(Debug, Clone)]
pub enum DunningAction {
    EmailSent {
        customer_id: String,
        template: EmailTemplate,
    },
    PaymentRetried {
        customer_id: String,
        success: bool,
    },
    AccountSuspended {
        customer_id: String,
    },
}

fn unix_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dunning_state_lifecycle() {
        let mut engine = DunningEngine::new(DunningConfig::default());

        // Payment fails
        engine.on_payment_failed("cust_123", "inv_456", "card_declined");
        assert_eq!(engine.get_state("cust_123"), DunningState::PastDue(0));

        // Payment succeeds
        engine.on_payment_succeeded("cust_123", "inv_456");
        assert_eq!(engine.get_state("cust_123"), DunningState::Active);
        assert!(!engine.is_account_suspended("cust_123"));
    }

    #[test]
    fn test_charge_failed_handler() {
        let mut engine = DunningEngine::new(DunningConfig::default());
        engine.on_charge_failed("cust_789", "ch_abc", "insufficient_funds");
        assert_eq!(engine.get_state("cust_789"), DunningState::PastDue(0));
    }

    #[test]
    fn test_advance_or_exhaust() {
        let mut engine = DunningEngine::new(DunningConfig::default());
        engine.on_payment_failed("cust_123", "inv_456", "declined");

        // Exhaust retries
        for _ in 0..3 {
            engine.advance_or_exhaust("cust_123");
        }

        assert_eq!(engine.get_state("cust_123"), DunningState::Suspended);
    }

    #[test]
    fn test_email_templates() {
        let templates = vec![
            EmailTemplate::PaymentFailedNotice,
            EmailTemplate::PaymentRetryReminder,
            EmailTemplate::PaymentEscalationWarning,
            EmailTemplate::AccountSuspendedNotice,
            EmailTemplate::PaymentRecoveredNotice,
        ];

        for tmpl in templates {
            assert!(!tmpl.subject().is_empty());
            assert!(!tmpl.body("test_customer").is_empty());
        }
    }

    #[test]
    fn test_default_schedule() {
        let schedule = default_dunning_schedule();
        assert_eq!(schedule.len(), 4);
        assert_eq!(schedule[0].day, 0);
        assert_eq!(schedule[1].day, 3);
        assert_eq!(schedule[2].day, 7);
        assert_eq!(schedule[3].day, 14);
        assert!(schedule[3].suspend);
    }

    #[test]
    fn test_unknown_customer_state() {
        let engine = DunningEngine::new(DunningConfig::default());
        assert_eq!(engine.get_state("nonexistent"), DunningState::Active);
        assert!(!engine.is_account_suspended("nonexistent"));
    }
}
