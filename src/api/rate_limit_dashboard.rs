//! Per-API-key rate-limit dashboard — daily usage stats and throttle counts.

use std::collections::HashMap;

use crate::api::billing::BillingDb;
use crate::billing::plans;

/// Aggregated rate-limit statistics for a single API key.
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    pub api_key: String,
    pub tier: String,
    pub requests_today: u32,
    pub quota_daily: u32,
    pub pct_daily: f64,
    pub throttled_today: u32,
}

/// Per-day usage bucket for a single API key.
#[derive(Debug, Clone, Default)]
pub struct DayBucket {
    pub count: u32,
    pub throttled: u32,
    pub day: u64,
}

/// In-memory dashboard tracking request counts per API key per day.
pub struct RateLimitDashboard {
    pub buckets: HashMap<String, DayBucket>,
}

impl RateLimitDashboard {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Record a request (and whether it was throttled) for an API key.
    pub fn record_request(&mut self, api_key: &str, throttled: bool) {
        let today = today_epoch();
        let bucket = self
            .buckets
            .entry(api_key.to_string())
            .or_insert_with(|| DayBucket {
                count: 0,
                throttled: 0,
                day: today,
            });

        // Reset if the day has rolled over.
        if bucket.day != today {
            bucket.count = 0;
            bucket.throttled = 0;
            bucket.day = today;
        }

        bucket.count += 1;
        if throttled {
            bucket.throttled += 1;
        }
    }

    /// Get stats for a single API key, enriched with billing tier info.
    pub fn get_stats(&self, api_key: &str, billing: &BillingDb) -> Option<RateLimitStats> {
        let bucket = self.buckets.get(api_key)?;
        let user = billing.get_user(api_key);
        let tier_name = user
            .map(|u| u.tier.as_str())
            .unwrap_or("free");
        let plan = plans::plan_for(tier_name);
        let quota = plan.daily_quota;
        let pct = if quota == u32::MAX || quota == 0 {
            0.0
        } else {
            (bucket.count as f64 / quota as f64) * 100.0
        };

        Some(RateLimitStats {
            api_key: api_key.to_string(),
            tier: tier_name.to_string(),
            requests_today: bucket.count,
            quota_daily: quota,
            pct_daily: pct,
            throttled_today: bucket.throttled,
        })
    }

    /// Get stats for all tracked API keys.
    pub fn get_all_stats(&self, billing: &BillingDb) -> Vec<RateLimitStats> {
        self.buckets
            .keys()
            .filter_map(|key| self.get_stats(key, billing))
            .collect()
    }
}

/// Start-of-day as a Unix timestamp (UTC midnight).
fn today_epoch() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    now - (now % 86400)
}
