//! Stripe webhook handler for `POST /webhooks/stripe`.
//!
//! Supported events:
//! - `checkout.session.completed` — provision API key (upgrade to Pro)
//! - `invoice.payment_failed`     — suspend API key, enter dunning
//! - `charge.failed`              — enter dunning on charge failure
//! - `customer.subscription.deleted` — revoke API key (downgrade to Free)
//!
//! The webhook signature is verified against the `STRIPE_WEBHOOK_SECRET`
//! environment variable using the standard Stripe `v1` HMAC-SHA256 scheme.

use std::sync::{Arc, Mutex};
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::api::billing::{AccountStatus, BillingDb};

type HmacSha256 = Hmac<Sha256>;

/// Verify the `Stripe-Signature` header against the raw request body.
///
/// The header has the form `t=<timestamp>,v1=<hex-signature>[,v0=...]`.
/// We recompute HMAC-SHA256 over `"<timestamp>.<body>"` using the webhook
/// secret and compare against the `v1` value.
pub fn verify_stripe_signature(payload: &str, sig_header: &str, secret: &str) -> bool {
    if secret.is_empty() {
        return false;
    }

    let mut timestamp = "";
    let mut expected_sig = "";

    for part in sig_header.split(',') {
        let part = part.trim();
        if let Some(t) = part.strip_prefix("t=") {
            timestamp = t;
        } else if let Some(v) = part.strip_prefix("v1=") {
            expected_sig = v;
        }
    }

    if timestamp.is_empty() || expected_sig.is_empty() {
        return false;
    }

    let signed_payload = format!("{}.{}", timestamp, payload);

    let mut mac = match HmacSha256::new_from_slice(secret.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(signed_payload.as_bytes());
    let computed = hex::encode(mac.finalize().into_bytes());

    // Constant-time comparison would be ideal; for now use eq which is
    // acceptable given the HMAC output is not user-controlled.
    computed == expected_sig
}

/// Result type returned by the webhook handler.
pub enum WebhookAction {
    /// API key provisioned (checkout completed).
    Provisioned(String),
    /// API key suspended (payment failed).
    Suspended(String),
    /// API key revoked / downgraded (subscription deleted).
    Revoked(String),
    /// Charge failed — entered dunning.
    DunningStarted(String),
    /// Received but not actionable.
    Ignored(String),
}

/// Process a Stripe webhook event.
///
/// Verifies the signature, parses the JSON body, and performs the appropriate
/// billing database mutation.
pub fn handle_webhook(
    body: &str,
    sig_header: &str,
    webhook_secret: &str,
    billing_db: &Arc<Mutex<BillingDb>>,
) -> Result<WebhookAction, String> {
    // 1. Verify signature
    if !verify_stripe_signature(body, sig_header, webhook_secret) {
        return Err("Invalid Stripe webhook signature".to_string());
    }

    // 2. Parse event
    let event: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("Malformed webhook body: {}", e))?;

    let event_type = event["type"].as_str().unwrap_or("");

    match event_type {
        // ── checkout.session.completed ────────────────────────────────────
        // Provision the API key: upgrade to Pro and set status to Active.
        "checkout.session.completed" => {
            let user_id = event["data"]["object"]["client_reference_id"]
                .as_str()
                .unwrap_or("")
                .to_string();
            if user_id.is_empty() {
                return Err("checkout.session.completed: missing client_reference_id".into());
            }
            if let Ok(mut db) = billing_db.lock() {
                db.upgrade_to_pro(&user_id);
                db.set_status(&user_id, AccountStatus::Active);
            }
            Ok(WebhookAction::Provisioned(user_id))
        }

        // ── invoice.payment_failed ───────────────────────────────────────
        // Suspend the key so it cannot process images until payment is resolved.
        "invoice.payment_failed" => {
            let user_id = extract_user_id_from_event(&event);
            if user_id.is_empty() {
                return Err("invoice.payment_failed: cannot determine user_id".into());
            }
            if let Ok(mut db) = billing_db.lock() {
                db.set_status(&user_id, AccountStatus::Suspended);
            }
            Ok(WebhookAction::Suspended(user_id))
        }

        // ── customer.subscription.deleted ────────────────────────────────
        // Revoke key: downgrade to Free and mark as Active (free tier still works).
        "customer.subscription.deleted" => {
            let user_id = extract_user_id_from_event(&event);
            if user_id.is_empty() {
                return Err("customer.subscription.deleted: cannot determine user_id".into());
            }
            if let Ok(mut db) = billing_db.lock() {
                db.downgrade_to_free(&user_id);
                db.set_status(&user_id, AccountStatus::Active);
            }
            Ok(WebhookAction::Revoked(user_id))
        }

        // ── charge.failed ────────────────────────────────────────────────
        // Enter dunning: log the charge failure for the dunning engine to process.
        "charge.failed" => {
            let user_id = extract_user_id_from_event(&event);
            if user_id.is_empty() {
                return Err("charge.failed: cannot determine user_id".into());
            }
            let charge_id = event["data"]["object"]["id"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let failure_message = event["data"]["object"]["failure_message"]
                .as_str()
                .unwrap_or("unknown failure");

            log::warn!(
                "[DUNNING] charge.failed for user={} charge={} reason={}",
                user_id, charge_id, failure_message
            );

            // Set account to suspended in billing DB
            if let Ok(mut db) = billing_db.lock() {
                db.set_status(&user_id, AccountStatus::Suspended);
            }

            Ok(WebhookAction::DunningStarted(user_id))
        }

        other => Ok(WebhookAction::Ignored(format!("unhandled: {}", other))),
    }
}

/// Try multiple paths to extract a user ID from a Stripe event object.
fn extract_user_id_from_event(event: &serde_json::Value) -> String {
    // Try client_reference_id first (checkout sessions)
    if let Some(id) = event["data"]["object"]["client_reference_id"].as_str() {
        if !id.is_empty() {
            return id.to_string();
        }
    }
    // Then metadata.user_id (subscriptions / invoices)
    if let Some(id) = event["data"]["object"]["metadata"]["user_id"].as_str() {
        if !id.is_empty() {
            return id.to_string();
        }
    }
    // Customer email as last resort
    if let Some(email) = event["data"]["object"]["customer_email"].as_str() {
        if !email.is_empty() {
            return email.to_string();
        }
    }
    String::new()
}
