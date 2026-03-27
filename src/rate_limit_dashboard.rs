//! Per-API-key rate-limit dashboard with bandwidth tracking, monthly limits,
//! and an admin HTML page with color-coded usage indicators.
//!
//! ## Endpoints
//!
//! - `GET  /admin/api-keys`              — list all keys with usage stats (JSON)
//! - `GET  /admin/api-keys/{key_id}/usage` — detailed usage for one key (JSON)
//! - `GET  /admin/rate-limits`           — HTML admin page with color-coded table
//!
//! ## Enforcement
//!
//! If a key exceeds its daily or monthly limit, the middleware returns HTTP 429
//! with a `Retry-After` header.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── Per-key usage record ────────────────────────────────────────────────────

/// Detailed usage record for one API key, stored in-memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyUsageRecord {
    pub api_key: String,
    pub plan: String,
    pub requests_today: u64,
    pub requests_this_month: u64,
    pub daily_limit: u64,
    pub monthly_limit: u64,
    pub bandwidth_bytes: u64,
    pub last_used: u64,
    pub payment_failed: bool,
    /// Internal: day boundary (UTC midnight epoch)
    #[serde(skip)]
    pub day_epoch: u64,
    /// Internal: month as YYYYMM
    #[serde(skip)]
    pub month_key: u32,
}

impl KeyUsageRecord {
    pub fn new(api_key: &str, plan: &str, daily_limit: u64, monthly_limit: u64) -> Self {
        Self {
            api_key: api_key.to_string(),
            plan: plan.to_string(),
            requests_today: 0,
            requests_this_month: 0,
            daily_limit,
            monthly_limit,
            bandwidth_bytes: 0,
            last_used: 0,
            payment_failed: false,
            day_epoch: today_epoch(),
            month_key: current_month(),
        }
    }

    /// Percentage of monthly limit used (0.0 if unlimited).
    pub fn pct_used(&self) -> f64 {
        if self.monthly_limit == 0 || self.monthly_limit == u64::MAX {
            0.0
        } else {
            (self.requests_this_month as f64 / self.monthly_limit as f64) * 100.0
        }
    }

    /// Color tier: "green" <50%, "yellow" 50-80%, "red" >80%.
    pub fn color_tier(&self) -> &'static str {
        let pct = self.pct_used();
        if pct > 80.0 {
            "red"
        } else if pct > 50.0 {
            "yellow"
        } else {
            "green"
        }
    }

    /// Mask the API key for display: show first 8 chars + "...".
    pub fn masked_key(&self) -> String {
        if self.api_key.len() > 8 {
            format!("{}...", &self.api_key[..8])
        } else {
            self.api_key.clone()
        }
    }
}

// ── Dashboard store ─────────────────────────────────────────────────────────

/// In-memory dashboard that tracks per-key usage, bandwidth, and limits.
pub struct KeyUsageDashboard {
    pub records: HashMap<String, KeyUsageRecord>,
}

impl KeyUsageDashboard {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Record an API request for a key, rolling over day/month counters as needed.
    pub fn record_request(&mut self, api_key: &str, plan: &str, bandwidth_bytes: u64) {
        let now = unix_now();
        let today = today_epoch();
        let month = current_month();

        let (daily_limit, monthly_limit) = limits_for_plan(plan);

        let record = self
            .records
            .entry(api_key.to_string())
            .or_insert_with(|| KeyUsageRecord::new(api_key, plan, daily_limit, monthly_limit));

        // Roll over day
        if record.day_epoch != today {
            record.requests_today = 0;
            record.day_epoch = today;
        }

        // Roll over month
        if record.month_key != month {
            record.requests_this_month = 0;
            record.bandwidth_bytes = 0;
            record.month_key = month;
        }

        record.requests_today += 1;
        record.requests_this_month += 1;
        record.bandwidth_bytes += bandwidth_bytes;
        record.last_used = now;
        record.plan = plan.to_string();
        record.daily_limit = daily_limit;
        record.monthly_limit = monthly_limit;
    }

    /// Check whether a key has exceeded its daily or monthly limit.
    /// Returns Ok(()) if within limits, or Err(retry_after_secs) if exceeded.
    pub fn check_limits(&self, api_key: &str) -> Result<(), u64> {
        let record = match self.records.get(api_key) {
            Some(r) => r,
            None => return Ok(()),
        };

        // Daily limit check
        if record.daily_limit > 0
            && record.daily_limit != u64::MAX
            && record.requests_today >= record.daily_limit
        {
            let now = unix_now();
            let next_day = record.day_epoch + 86400;
            let retry_after = if next_day > now { next_day - now } else { 60 };
            return Err(retry_after);
        }

        // Monthly limit check
        if record.monthly_limit > 0
            && record.monthly_limit != u64::MAX
            && record.requests_this_month >= record.monthly_limit
        {
            // Retry after roughly the start of next month (simplistic)
            return Err(86400);
        }

        Ok(())
    }

    /// Mark a key as having a failed payment.
    pub fn set_payment_failed(&mut self, api_key: &str, failed: bool) {
        if let Some(record) = self.records.get_mut(api_key) {
            record.payment_failed = failed;
        }
    }

    /// Get usage for a single key.
    pub fn get_key_usage(&self, api_key: &str) -> Option<&KeyUsageRecord> {
        self.records.get(api_key)
    }

    /// Get all keys with usage stats (for admin listing).
    pub fn get_all_keys(&self) -> Vec<&KeyUsageRecord> {
        let mut keys: Vec<_> = self.records.values().collect();
        keys.sort_by(|a, b| b.last_used.cmp(&a.last_used));
        keys
    }
}

// ── JSON response builders ──────────────────────────────────────────────────

/// Build JSON response for GET /admin/api-keys
pub fn admin_api_keys_json(dashboard: &KeyUsageDashboard) -> serde_json::Value {
    let keys: Vec<serde_json::Value> = dashboard
        .get_all_keys()
        .iter()
        .map(|r| {
            serde_json::json!({
                "api_key": r.masked_key(),
                "plan": r.plan,
                "requests_today": r.requests_today,
                "requests_this_month": r.requests_this_month,
                "daily_limit": r.daily_limit,
                "monthly_limit": r.monthly_limit,
                "bandwidth_bytes": r.bandwidth_bytes,
                "pct_used": (r.pct_used() * 100.0).round() / 100.0,
                "last_used": r.last_used,
                "payment_failed": r.payment_failed,
                "color": r.color_tier(),
            })
        })
        .collect();
    serde_json::json!({ "keys": keys })
}

/// Build JSON response for GET /admin/api-keys/{key_id}/usage
pub fn admin_key_usage_json(
    dashboard: &KeyUsageDashboard,
    key_id: &str,
) -> Option<serde_json::Value> {
    // Try exact match first, then prefix match (since callers may use masked keys)
    let record = dashboard.get_key_usage(key_id).or_else(|| {
        dashboard
            .records
            .values()
            .find(|r| r.api_key.starts_with(key_id))
    })?;

    Some(serde_json::json!({
        "api_key": record.masked_key(),
        "plan": record.plan,
        "requests_today": record.requests_today,
        "requests_this_month": record.requests_this_month,
        "daily_limit": record.daily_limit,
        "monthly_limit": record.monthly_limit,
        "bandwidth_bytes": record.bandwidth_bytes,
        "bandwidth_mb": (record.bandwidth_bytes as f64 / 1_048_576.0 * 100.0).round() / 100.0,
        "pct_used": (record.pct_used() * 100.0).round() / 100.0,
        "last_used": record.last_used,
        "payment_failed": record.payment_failed,
        "color": record.color_tier(),
    }))
}

// ── Admin HTML page ─────────────────────────────────────────────────────────

/// Render the admin HTML page for /admin/rate-limits with color-coded table.
pub fn admin_rate_limits_html(dashboard: &KeyUsageDashboard) -> String {
    let keys = dashboard.get_all_keys();

    let mut rows = String::new();
    for r in &keys {
        let pct = r.pct_used();
        let color = match r.color_tier() {
            "red" => "#ef4444",
            "yellow" => "#eab308",
            _ => "#22c55e",
        };
        let bg = match r.color_tier() {
            "red" => "#fef2f2",
            "yellow" => "#fefce8",
            _ => "#f0fdf4",
        };
        let last_used = if r.last_used > 0 {
            format_timestamp(r.last_used)
        } else {
            "Never".to_string()
        };
        let payment_badge = if r.payment_failed {
            r#"<span style="color:#ef4444;font-weight:600"> PAYMENT FAILED</span>"#
        } else {
            ""
        };

        rows.push_str(&format!(
            r#"<tr style="background:{}">
  <td><code>{}</code>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td style="color:{};font-weight:600">{:.1}%</td>
  <td>{}</td>
</tr>
"#,
            bg,
            r.masked_key(),
            payment_badge,
            r.plan,
            r.requests_today,
            r.monthly_limit,
            color,
            pct,
            last_used,
        ));
    }

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rate Limit Dashboard — SRGAN Admin</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 2rem; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #1e293b; color: #fff; text-align: left; padding: 0.75rem 1rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; }}
  code {{ background: #f1f5f9; padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.82rem; }}
  .empty {{ text-align: center; padding: 3rem; color: #94a3b8; }}
</style>
</head>
<body>
<h1>Per-API-Key Rate Limits</h1>
<table>
<thead>
<tr>
  <th>API Key</th>
  <th>Plan</th>
  <th>Requests Today</th>
  <th>Monthly Limit</th>
  <th>% Used</th>
  <th>Last Used</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
{empty}
</body>
</html>"##,
        rows = rows,
        empty = if keys.is_empty() {
            r#"<p class="empty">No API keys with usage data yet.</p>"#
        } else {
            ""
        },
    )
}

// ── Plan limits ─────────────────────────────────────────────────────────────

/// Return (daily_limit, monthly_limit) for a billing plan name.
pub fn limits_for_plan(plan: &str) -> (u64, u64) {
    match plan {
        "pro" => (1000, 30_000),
        "enterprise" => (u64::MAX, u64::MAX),
        _ => (10, 300), // free
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn today_epoch() -> u64 {
    let now = unix_now();
    now - (now % 86400)
}

fn current_month() -> u32 {
    let now = unix_now();
    let days = now / 86400;
    let years = days / 365;
    let remaining_days = days % 365;
    let month = remaining_days / 30;
    ((1970 + years) * 100 + month + 1) as u32
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn format_timestamp(ts: u64) -> String {
    // Simple UTC date/time formatting without pulling in chrono
    let secs_per_day: u64 = 86400;
    let days = ts / secs_per_day;
    let time_of_day = ts % secs_per_day;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;

    // Approximate date from epoch days
    let mut y: u64 = 1970;
    let mut remaining = days;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }
    let month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let mut m: u64 = 0;
    for (i, &md) in month_days.iter().enumerate() {
        let md = if i == 1 && leap { 29 } else { md };
        if remaining < md {
            break;
        }
        remaining -= md;
        m += 1;
    }
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02} UTC",
        y,
        m + 1,
        remaining + 1,
        hours,
        minutes
    )
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_check_limits() {
        let mut dash = KeyUsageDashboard::new();
        for _ in 0..10 {
            dash.record_request("key_free", "free", 1024);
        }
        // 10 requests = daily limit for free tier
        assert!(dash.check_limits("key_free").is_err());
    }

    #[test]
    fn test_pro_within_limits() {
        let mut dash = KeyUsageDashboard::new();
        for _ in 0..100 {
            dash.record_request("key_pro", "pro", 2048);
        }
        assert!(dash.check_limits("key_pro").is_ok());
    }

    #[test]
    fn test_color_tiers() {
        let mut r = KeyUsageRecord::new("k", "free", 10, 300);
        r.requests_this_month = 100;
        assert_eq!(r.color_tier(), "green"); // 33%

        r.requests_this_month = 200;
        assert_eq!(r.color_tier(), "yellow"); // 67%

        r.requests_this_month = 280;
        assert_eq!(r.color_tier(), "red"); // 93%
    }

    #[test]
    fn test_masked_key() {
        let r = KeyUsageRecord::new("sk_live_1234567890abcdef", "pro", 1000, 30000);
        assert_eq!(r.masked_key(), "sk_live_...");
    }

    #[test]
    fn test_payment_failed_flag() {
        let mut dash = KeyUsageDashboard::new();
        dash.record_request("key_a", "pro", 0);
        dash.set_payment_failed("key_a", true);
        assert!(dash.get_key_usage("key_a").unwrap().payment_failed);
        dash.set_payment_failed("key_a", false);
        assert!(!dash.get_key_usage("key_a").unwrap().payment_failed);
    }

    #[test]
    fn test_admin_json_output() {
        let mut dash = KeyUsageDashboard::new();
        dash.record_request("key_test", "free", 512);
        let json = admin_api_keys_json(&dash);
        assert!(json["keys"].is_array());
        assert_eq!(json["keys"][0]["plan"], "free");
    }

    #[test]
    fn test_key_usage_json() {
        let mut dash = KeyUsageDashboard::new();
        dash.record_request("key_detail", "pro", 4096);
        let json = admin_key_usage_json(&dash, "key_detail").unwrap();
        assert_eq!(json["plan"], "pro");
        assert_eq!(json["bandwidth_bytes"], 4096);
    }

    #[test]
    fn test_html_output() {
        let mut dash = KeyUsageDashboard::new();
        dash.record_request("key_html", "free", 0);
        let html = admin_rate_limits_html(&dash);
        assert!(html.contains("Rate Limit"));
        assert!(html.contains("key_html"));
    }

    #[test]
    fn test_unknown_key_passes_limits() {
        let dash = KeyUsageDashboard::new();
        assert!(dash.check_limits("nonexistent").is_ok());
    }

    #[test]
    fn test_enterprise_unlimited() {
        let mut dash = KeyUsageDashboard::new();
        for _ in 0..10000 {
            dash.record_request("key_ent", "enterprise", 1_000_000);
        }
        assert!(dash.check_limits("key_ent").is_ok());
    }
}
