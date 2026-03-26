//! Admin endpoint — GET /api/v1/admin/stats
//!
//! Requires the `ADMIN_KEY` environment variable to be set and passed as the
//! `X-Admin-Key` header (or `Authorization: Bearer <key>`).

use std::sync::Arc;

use serde::Serialize;

use crate::api::auth::KeyStore;

/// Statistics returned by GET /api/v1/admin/stats
#[derive(Debug, Serialize)]
pub struct AdminStats {
    pub total_keys: usize,
    pub keys_by_tier: std::collections::HashMap<String, usize>,
    pub usage_today_by_tier: std::collections::HashMap<String, u64>,
    pub total_requests_today: u64,
}

/// Verify that `provided` matches the `ADMIN_KEY` env var.
///
/// Returns `true` if the admin key is not set (open mode) OR if `provided` matches.
pub fn check_admin_auth(provided: &str) -> bool {
    match std::env::var("ADMIN_KEY") {
        Ok(expected) => provided == expected,
        // No ADMIN_KEY configured — deny all admin access
        Err(_) => false,
    }
}

/// Build an `AdminStats` snapshot.
pub fn get_stats(key_store: &Arc<KeyStore>) -> AdminStats {
    let all_keys = key_store.all_keys().unwrap_or_default();

    let total_keys = all_keys.len();

    let mut keys_by_tier: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for k in &all_keys {
        *keys_by_tier.entry(k.tier.as_str().to_string()).or_insert(0) += 1;
    }

    let usage_today_by_tier = key_store.usage_today_by_tier();
    let total_requests_today: u64 = usage_today_by_tier.values().sum();

    AdminStats {
        total_keys,
        keys_by_tier,
        usage_today_by_tier,
        total_requests_today,
    }
}
