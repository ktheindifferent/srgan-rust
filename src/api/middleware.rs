use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use dashmap::DashMap;
use crate::api::billing::SubscriptionTier;

/// Per-tier rate limit policy
#[derive(Debug, Clone, Copy)]
pub struct RateLimitPolicy {
    /// Maximum requests per hour (u32::MAX = unlimited)
    pub requests_per_hour: u32,
    /// Maximum output megapixels per request (u32::MAX = unlimited)
    pub max_output_megapixels: u32,
}

impl RateLimitPolicy {
    pub fn for_tier(tier: &SubscriptionTier) -> Self {
        // Allow overriding the default per-key rate via DEFAULT_KEY_RATE env.
        // Format: "N/hour" (e.g. "100/hour"). Applied to Free and Pro tiers.
        let env_rate: Option<u32> = std::env::var("DEFAULT_KEY_RATE")
            .ok()
            .and_then(|v| {
                let v = v.trim().to_lowercase();
                v.split('/').next().and_then(|n| n.trim().parse().ok())
            });

        match tier {
            SubscriptionTier::Free => Self {
                requests_per_hour: env_rate.unwrap_or(10),
                max_output_megapixels: 2,
            },
            SubscriptionTier::Pro => Self {
                requests_per_hour: env_rate.map(|r| r * 10).unwrap_or(1000),
                max_output_megapixels: 10,
            },
            SubscriptionTier::Enterprise => Self {
                requests_per_hour: u32::MAX,
                max_output_megapixels: u32::MAX,
            },
        }
    }
}

/// Result of a rate limit check, including headers to return to the client
#[derive(Debug)]
pub struct RateLimitResult {
    pub allowed: bool,
    /// X-RateLimit-Limit
    pub limit: u32,
    /// X-RateLimit-Remaining
    pub remaining: u32,
    /// X-RateLimit-Reset (Unix timestamp when the window resets)
    pub reset_at: u64,
    /// Retry-After seconds (only set when allowed == false)
    pub retry_after: Option<u64>,
}

impl RateLimitResult {
    /// Build the rate-limit header string to inject into an HTTP response.
    pub fn headers(&self) -> String {
        let mut h = format!(
            "X-RateLimit-Limit: {}\r\nX-RateLimit-Remaining: {}\r\nX-RateLimit-Reset: {}\r\n",
            self.limit, self.remaining, self.reset_at
        );
        if let Some(retry) = self.retry_after {
            h.push_str(&format!("Retry-After: {}\r\n", retry));
        }
        h
    }
}

/// Sliding-hourly-window entry stored per API key
struct WindowEntry {
    count: u32,
    window_start: u64,
}

/// Per-API-key, per-tier rate limiter backed by DashMap for lock-free concurrent access.
pub struct TierRateLimiter {
    windows: Arc<DashMap<String, WindowEntry>>,
}

impl TierRateLimiter {
    pub fn new() -> Self {
        Self {
            windows: Arc::new(DashMap::new()),
        }
    }

    /// Check and record a request for `api_key` at the given `tier`.
    /// Increments the counter on success.
    pub fn check(&self, api_key: &str, tier: &SubscriptionTier) -> RateLimitResult {
        let policy = RateLimitPolicy::for_tier(tier);

        // Enterprise is unlimited — skip tracking overhead
        if matches!(tier, SubscriptionTier::Enterprise) {
            return RateLimitResult {
                allowed: true,
                limit: u32::MAX,
                remaining: u32::MAX,
                reset_at: 0,
                retry_after: None,
            };
        }

        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let window_start = now_secs - (now_secs % 3600);
        let reset_at = window_start + 3600;

        let mut entry = self
            .windows
            .entry(api_key.to_string())
            .or_insert(WindowEntry {
                count: 0,
                window_start,
            });

        // Roll over to a new hour window if needed
        if entry.window_start < window_start {
            entry.count = 0;
            entry.window_start = window_start;
        }

        if entry.count >= policy.requests_per_hour {
            let retry_after = reset_at.saturating_sub(now_secs);
            return RateLimitResult {
                allowed: false,
                limit: policy.requests_per_hour,
                remaining: 0,
                reset_at,
                retry_after: Some(retry_after),
            };
        }

        entry.count += 1;
        let remaining = policy.requests_per_hour.saturating_sub(entry.count);

        RateLimitResult {
            allowed: true,
            limit: policy.requests_per_hour,
            remaining,
            reset_at,
            retry_after: None,
        }
    }
}
