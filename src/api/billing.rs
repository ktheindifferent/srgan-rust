use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Subscription tier for a user account
#[derive(Debug, Clone, PartialEq)]
pub enum SubscriptionTier {
    Free,
    Pro,
    Enterprise,
}

impl SubscriptionTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            SubscriptionTier::Free => "free",
            SubscriptionTier::Pro => "pro",
            SubscriptionTier::Enterprise => "enterprise",
        }
    }
}

/// User account stored in the database
#[derive(Debug, Clone)]
pub struct UserAccount {
    pub user_id: String,
    pub tier: SubscriptionTier,
    pub credits_remaining: u32,
    pub credits_reset_at: u64,
    pub credits_issued_today: u32,
    pub credits_consumed_today: u32,
}

impl UserAccount {
    pub fn new_free(user_id: String) -> Self {
        Self {
            user_id,
            tier: SubscriptionTier::Free,
            credits_remaining: 10,
            credits_reset_at: next_hour_epoch(),
            credits_issued_today: 10,
            credits_consumed_today: 0,
        }
    }
}

/// In-memory billing database (production: replace with rusqlite)
pub struct BillingDb {
    // api_key -> UserAccount
    users: HashMap<String, UserAccount>,
}

impl BillingDb {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }

    pub fn get_user(&self, api_key: &str) -> Option<&UserAccount> {
        self.users.get(api_key)
    }

    pub fn get_or_create_free(&mut self, api_key: &str) -> &UserAccount {
        if !self.users.contains_key(api_key) {
            let account = UserAccount::new_free(api_key.to_string());
            self.users.insert(api_key.to_string(), account);
        }
        self.users.get(api_key).unwrap()
    }

    pub fn upgrade_to_pro(&mut self, user_id: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Some(user) = self.users.get_mut(user_id) {
            user.tier = SubscriptionTier::Pro;
            user.credits_remaining = 1000;
            user.credits_reset_at = now + 30 * 24 * 3600;
            user.credits_issued_today += 1000;
        } else {
            let mut account = UserAccount::new_free(user_id.to_string());
            account.tier = SubscriptionTier::Pro;
            account.credits_remaining = 1000;
            account.credits_reset_at = now + 30 * 24 * 3600;
            account.credits_issued_today = 1000;
            self.users.insert(user_id.to_string(), account);
        }
    }

    pub fn downgrade_to_free(&mut self, user_id: &str) {
        if let Some(user) = self.users.get_mut(user_id) {
            user.tier = SubscriptionTier::Free;
            user.credits_remaining = 10;
        }
    }

    /// Deduct one credit. Returns false if no credits remain (Enterprise never fails).
    pub fn consume_credit(&mut self, user_id: &str) -> bool {
        if let Some(user) = self.users.get_mut(user_id) {
            if matches!(user.tier, SubscriptionTier::Enterprise) {
                user.credits_consumed_today += 1;
                return true;
            }
            if user.credits_remaining > 0 {
                user.credits_remaining -= 1;
                user.credits_consumed_today += 1;
                return true;
            }
            return false;
        }
        // Unknown user: treat as free with no credits
        false
    }

    /// Snapshot of all user accounts (for admin panel)
    pub fn all_users_snapshot(&self) -> Vec<serde_json::Value> {
        self.users.values().map(|u| serde_json::json!({
            "user_id": u.user_id,
            "tier": u.tier.as_str(),
            "credits_remaining": u.credits_remaining,
            "credits_reset_at": u.credits_reset_at,
            "credits_issued_today": u.credits_issued_today,
            "credits_consumed_today": u.credits_consumed_today,
        })).collect()
    }

    /// Totals across all users (for dashboard)
    pub fn totals_today(&self) -> (u32, u32) {
        let issued = self.users.values().map(|u| u.credits_issued_today).sum();
        let consumed = self.users.values().map(|u| u.credits_consumed_today).sum();
        (issued, consumed)
    }

    /// Extended snapshot including total jobs (credits consumed) and last active.
    pub fn all_users_extended(&self) -> Vec<serde_json::Value> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.users.values().map(|u| serde_json::json!({
            "user_id": u.user_id,
            "tier": u.tier.as_str(),
            "credits_remaining": u.credits_remaining,
            "credits_reset_at": u.credits_reset_at,
            "credits_issued_today": u.credits_issued_today,
            "credits_consumed_today": u.credits_consumed_today,
            "total_jobs": u.credits_consumed_today,
            "last_active": u.credits_reset_at.min(now),
        })).collect()
    }

    /// Credit usage grouped by tier.
    pub fn credits_by_tier(&self) -> serde_json::Value {
        let mut free_issued = 0u64;
        let mut free_consumed = 0u64;
        let mut pro_issued = 0u64;
        let mut pro_consumed = 0u64;
        let mut enterprise_issued = 0u64;
        let mut enterprise_consumed = 0u64;

        for u in self.users.values() {
            match u.tier {
                SubscriptionTier::Free => {
                    free_issued += u.credits_issued_today as u64;
                    free_consumed += u.credits_consumed_today as u64;
                }
                SubscriptionTier::Pro => {
                    pro_issued += u.credits_issued_today as u64;
                    pro_consumed += u.credits_consumed_today as u64;
                }
                SubscriptionTier::Enterprise => {
                    enterprise_issued += u.credits_issued_today as u64;
                    enterprise_consumed += u.credits_consumed_today as u64;
                }
            }
        }

        serde_json::json!({
            "free": { "issued": free_issued, "consumed": free_consumed },
            "pro": { "issued": pro_issued, "consumed": pro_consumed },
            "enterprise": { "issued": enterprise_issued, "consumed": enterprise_consumed },
        })
    }

    /// Top N users by jobs (credits consumed today), descending.
    pub fn top_users_by_jobs(&self, limit: usize) -> Vec<serde_json::Value> {
        let mut sorted: Vec<&UserAccount> = self.users.values().collect();
        sorted.sort_by(|a, b| b.credits_consumed_today.cmp(&a.credits_consumed_today));
        sorted.truncate(limit);
        sorted.iter().map(|u| serde_json::json!({
            "user_id": u.user_id,
            "tier": u.tier.as_str(),
            "job_count": u.credits_consumed_today,
        })).collect()
    }
}

fn next_hour_epoch() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let hour_start = now - (now % 3600);
    hour_start + 3600
}

// ── API response types ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct BillingStatus {
    pub tier: String,
    pub credits_remaining: u32,
    pub credits_reset_at: u64,
}

#[derive(Debug, Serialize)]
pub struct CheckoutSession {
    pub session_id: String,
    pub checkout_url: String,
}

#[derive(Debug, Deserialize)]
pub struct CheckoutRequest {
    pub user_id: String,
    pub success_url: String,
    pub cancel_url: String,
}

// ── Stripe integration ────────────────────────────────────────────────────────

/// Create a Stripe checkout session for a Pro upgrade.
pub fn create_checkout_session(
    req: &CheckoutRequest,
    stripe_secret_key: &str,
) -> Result<CheckoutSession, String> {
    let client = reqwest::blocking::Client::new();

    let mut params = HashMap::new();
    params.insert("mode", "subscription");
    params.insert("success_url", req.success_url.as_str());
    params.insert("cancel_url", req.cancel_url.as_str());
    params.insert("line_items[0][price]", "price_pro_monthly");
    params.insert("line_items[0][quantity]", "1");
    params.insert("client_reference_id", req.user_id.as_str());

    let response = client
        .post("https://api.stripe.com/v1/checkout/sessions")
        .header("Authorization", format!("Bearer {}", stripe_secret_key))
        .form(&params)
        .send()
        .map_err(|e| format!("Stripe request failed: {}", e))?;

    let status = response.status();
    let body: serde_json::Value = response
        .json()
        .map_err(|e| format!("Failed to parse Stripe response: {}", e))?;

    if !status.is_success() {
        let msg = body["error"]["message"].as_str().unwrap_or("Unknown Stripe error");
        return Err(format!("Stripe error: {}", msg));
    }

    let session_id = body["id"]
        .as_str()
        .ok_or_else(|| "Missing session id in Stripe response".to_string())?
        .to_string();
    let checkout_url = body["url"]
        .as_str()
        .ok_or_else(|| "Missing url in Stripe response".to_string())?
        .to_string();

    Ok(CheckoutSession {
        session_id,
        checkout_url,
    })
}

/// Process a Stripe webhook event body and update the billing database.
pub fn handle_stripe_webhook(
    body: &str,
    stripe_webhook_secret: &str,
    signature: &str,
    db: &Arc<Mutex<BillingDb>>,
) -> Result<String, String> {
    if !verify_stripe_signature(body, signature, stripe_webhook_secret) {
        return Err("Invalid webhook signature".to_string());
    }

    let event: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("Invalid webhook body: {}", e))?;

    let event_type = event["type"].as_str().unwrap_or("");

    match event_type {
        "checkout.session.completed" | "invoice.payment_succeeded" => {
            let user_id = event["data"]["object"]["client_reference_id"]
                .as_str()
                .unwrap_or("")
                .to_string();
            if !user_id.is_empty() {
                if let Ok(mut db_guard) = db.lock() {
                    db_guard.upgrade_to_pro(&user_id);
                }
            }
            Ok("payment_processed".to_string())
        }
        "customer.subscription.deleted" => {
            let user_id = event["data"]["object"]["metadata"]["user_id"]
                .as_str()
                .unwrap_or("")
                .to_string();
            if !user_id.is_empty() {
                if let Ok(mut db_guard) = db.lock() {
                    db_guard.downgrade_to_free(&user_id);
                }
            }
            Ok("subscription_cancelled".to_string())
        }
        other => Ok(format!("unhandled_event:{}", other)),
    }
}

/// Verify a Stripe webhook signature header using HMAC-SHA256.
fn verify_stripe_signature(payload: &str, signature: &str, secret: &str) -> bool {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    type HmacSha256 = Hmac<Sha256>;

    let mut timestamp = "";
    let mut expected_sig = "";

    for part in signature.split(',') {
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

    // Constant-time compare would be preferable in production
    computed == expected_sig
}
