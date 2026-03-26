//! Rate-limiting middleware: check whether an API key is within its daily quota.

use std::sync::Arc;

use crate::api::auth::{KeyStore, KeyTier};
use crate::error::{Result, SrganError};

/// Result of a rate-limit check.
#[derive(Debug)]
pub enum RateLimitResult {
    /// Request is allowed; `used` is the updated count after this request.
    Allowed { used: u64, limit: Option<u64> },
    /// Request is denied: the key has exceeded its daily quota.
    RateLimited { used: u64, limit: u64 },
}

/// HTTP 429 response body.
#[derive(Debug, serde::Serialize)]
pub struct RateLimitError {
    pub error: &'static str,
    pub used: u64,
    pub limit: u64,
    pub resets_in: &'static str,
}

/// Shared rate limiter backed by the key store.
pub struct ApiRateLimiter {
    store: Arc<KeyStore>,
}

impl ApiRateLimiter {
    pub fn new(store: Arc<KeyStore>) -> Self {
        Self { store }
    }

    /// Validate `key` exists, check its quota, and — if allowed — record usage.
    ///
    /// Returns `Err` if the key is unknown.
    pub fn check(&self, key: &str) -> Result<RateLimitResult> {
        let api_key = self
            .store
            .get_key(key)?
            .ok_or_else(|| SrganError::InvalidParameter("Unknown API key".to_string()))?;

        let limit = api_key.tier.daily_limit();
        let current_usage = self.store.today_usage(key);

        if let Some(max) = limit {
            if current_usage >= max {
                return Ok(RateLimitResult::RateLimited {
                    used: current_usage,
                    limit: max,
                });
            }
        }

        // Record usage and return updated count
        let used = self.store.record_usage(key)?;
        Ok(RateLimitResult::Allowed { used, limit })
    }

    /// The tier for an API key (used to set job priority).
    pub fn tier_for_key(&self, key: &str) -> Option<KeyTier> {
        self.store.get_key(key).ok().flatten().map(|k| k.tier)
    }
}
