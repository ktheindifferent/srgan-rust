//! Billing plan definitions with daily image quotas.
//!
//! - **Free**: 10 images/day, no cost
//! - **Pro**: 1 000 images/day, $19/month
//! - **Enterprise**: unlimited images/day, custom pricing

use serde::{Deserialize, Serialize};

/// A billing plan with its associated daily quota.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub name: &'static str,
    pub display_name: &'static str,
    /// Monthly price in USD cents (0 = free).
    pub price_cents: u32,
    /// Maximum images per day (`u32::MAX` = unlimited).
    pub daily_quota: u32,
    /// Stripe Price ID used for checkout (None for free/enterprise).
    pub stripe_price_id: Option<&'static str>,
}

pub const PLAN_FREE: Plan = Plan {
    name: "free",
    display_name: "Free",
    price_cents: 0,
    daily_quota: 10,
    stripe_price_id: None,
};

pub const PLAN_PRO: Plan = Plan {
    name: "pro",
    display_name: "Pro",
    price_cents: 1900, // $19/mo
    daily_quota: 1000,
    stripe_price_id: Some("price_pro_monthly"),
};

pub const PLAN_ENTERPRISE: Plan = Plan {
    name: "enterprise",
    display_name: "Enterprise",
    price_cents: 0, // custom pricing
    daily_quota: u32::MAX,
    stripe_price_id: None,
};

/// Look up plan details by name.
pub fn plan_for(name: &str) -> &'static Plan {
    match name {
        "pro" => &PLAN_PRO,
        "enterprise" => &PLAN_ENTERPRISE,
        _ => &PLAN_FREE,
    }
}

/// Return all plans (useful for listing on a pricing page).
pub fn all_plans() -> Vec<&'static Plan> {
    vec![&PLAN_FREE, &PLAN_PRO, &PLAN_ENTERPRISE]
}
