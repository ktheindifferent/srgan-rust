use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Serialize, Deserialize};

use crate::error::{Result, SrganError};

/// A referral/affiliate code that can be shared to earn credits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferralCode {
    pub code: String,
    pub owner_user_id: String,
    pub created_at: u64,
    pub uses: u64,
    pub max_uses: Option<u64>,
    pub discount_pct: u8,
    pub commission_pct: u8,
    pub active: bool,
}

/// Records a single referral signup event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferralSignup {
    pub id: String,
    pub referral_code: String,
    pub referrer_user_id: String,
    pub referred_user_id: String,
    pub signed_up_at: u64,
    pub credited: bool,
}

/// Per-user referral credit balance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferralCredits {
    pub user_id: String,
    pub total_earned: f64,
    pub total_redeemed: f64,
    pub pending: f64,
}

/// Aggregate referral statistics for a user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferralStats {
    pub total_codes: usize,
    pub total_signups: u64,
    pub total_earned: f64,
    pub total_redeemed: f64,
    pub available_credits: f64,
    pub active_codes: usize,
}

/// In-memory referral store (mirrors the HashMap pattern used elsewhere in the
/// codebase; can be swapped for SQLite later).
pub struct ReferralStore {
    referrals: Arc<RwLock<HashMap<String, ReferralCode>>>,
    signups: Arc<RwLock<Vec<ReferralSignup>>>,
    credits: Arc<RwLock<HashMap<String, ReferralCredits>>>,
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn generate_code_string() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = now.as_nanos();

    let mut hasher = DefaultHasher::new();
    nanos.hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let hash = hasher.finish();

    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut code = String::with_capacity(8);
    let mut val = hash;
    for _ in 0..8 {
        let idx = (val % CHARSET.len() as u64) as usize;
        code.push(CHARSET[idx] as char);
        val /= CHARSET.len() as u64;
    }
    code
}

impl ReferralStore {
    /// Create a new empty referral store.
    pub fn new() -> Self {
        Self {
            referrals: Arc::new(RwLock::new(HashMap::new())),
            signups: Arc::new(RwLock::new(Vec::new())),
            credits: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate a new referral code owned by `user_id`.
    pub fn generate_code(
        &self,
        user_id: &str,
        discount_pct: u8,
        commission_pct: u8,
    ) -> Result<ReferralCode> {
        let mut referrals = self.referrals.write().map_err(|e| {
            SrganError::InvalidInput(format!("lock poisoned: {}", e))
        })?;

        // Ensure uniqueness (retry up to 10 times on collision)
        let mut code_str = generate_code_string();
        for _ in 0..10 {
            if !referrals.contains_key(&code_str) {
                break;
            }
            code_str = generate_code_string();
        }
        if referrals.contains_key(&code_str) {
            return Err(SrganError::InvalidInput(
                "Failed to generate a unique referral code".into(),
            ));
        }

        let code = ReferralCode {
            code: code_str.clone(),
            owner_user_id: user_id.to_string(),
            created_at: unix_now(),
            uses: 0,
            max_uses: None,
            discount_pct,
            commission_pct,
            active: true,
        };

        referrals.insert(code_str, code.clone());
        Ok(code)
    }

    /// Validate a referral code. Returns a clone if active and within usage limits.
    pub fn validate_code(&self, code: &str) -> Option<ReferralCode> {
        let referrals = self.referrals.read().ok()?;
        let rc = referrals.get(code)?;
        if !rc.active {
            return None;
        }
        if let Some(max) = rc.max_uses {
            if rc.uses >= max {
                return None;
            }
        }
        Some(rc.clone())
    }

    /// Record a signup attributed to the given referral code.
    pub fn record_signup(
        &self,
        code: &str,
        referred_user_id: &str,
    ) -> Result<ReferralSignup> {
        // Validate code first
        let referral = self.validate_code(code).ok_or_else(|| {
            SrganError::InvalidInput(format!("Referral code '{}' is invalid or inactive", code))
        })?;

        // Increment usage count
        {
            let mut referrals = self.referrals.write().map_err(|e| {
                SrganError::InvalidInput(format!("lock poisoned: {}", e))
            })?;
            if let Some(rc) = referrals.get_mut(code) {
                rc.uses += 1;
            }
        }

        let signup = ReferralSignup {
            id: format!("signup_{}", unix_now()),
            referral_code: code.to_string(),
            referrer_user_id: referral.owner_user_id.clone(),
            referred_user_id: referred_user_id.to_string(),
            signed_up_at: unix_now(),
            credited: false,
        };

        let mut signups = self.signups.write().map_err(|e| {
            SrganError::InvalidInput(format!("lock poisoned: {}", e))
        })?;
        signups.push(signup.clone());

        Ok(signup)
    }

    /// Credit the referrer for a completed signup.
    pub fn credit_referrer(&self, signup_id: &str) -> Result<()> {
        let mut signups = self.signups.write().map_err(|e| {
            SrganError::InvalidInput(format!("lock poisoned: {}", e))
        })?;

        let signup = signups
            .iter_mut()
            .find(|s| s.id == signup_id)
            .ok_or_else(|| {
                SrganError::InvalidInput(format!("Signup '{}' not found", signup_id))
            })?;

        if signup.credited {
            return Err(SrganError::InvalidInput(
                "Signup already credited".into(),
            ));
        }

        let referrer_id = signup.referrer_user_id.clone();
        let code_str = signup.referral_code.clone();
        signup.credited = true;

        // Look up commission percentage from the referral code
        let commission_pct = {
            let referrals = self.referrals.read().map_err(|e| {
                SrganError::InvalidInput(format!("lock poisoned: {}", e))
            })?;
            referrals
                .get(&code_str)
                .map(|rc| rc.commission_pct)
                .unwrap_or(10)
        };

        let credit_amount = commission_pct as f64;

        let mut credits = self.credits.write().map_err(|e| {
            SrganError::InvalidInput(format!("lock poisoned: {}", e))
        })?;
        let entry = credits
            .entry(referrer_id.clone())
            .or_insert_with(|| ReferralCredits {
                user_id: referrer_id,
                total_earned: 0.0,
                total_redeemed: 0.0,
                pending: 0.0,
            });
        entry.total_earned += credit_amount;
        entry.pending += credit_amount;

        Ok(())
    }

    /// Get all signups attributed to a given user (as referrer).
    pub fn get_user_referrals(&self, user_id: &str) -> Vec<ReferralSignup> {
        let signups = match self.signups.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        signups
            .iter()
            .filter(|s| s.referrer_user_id == user_id)
            .cloned()
            .collect()
    }

    /// Get the credit balance for a user.
    pub fn get_user_credits(&self, user_id: &str) -> ReferralCredits {
        let credits = match self.credits.read() {
            Ok(c) => c,
            Err(_) => {
                return ReferralCredits {
                    user_id: user_id.to_string(),
                    total_earned: 0.0,
                    total_redeemed: 0.0,
                    pending: 0.0,
                }
            }
        };
        credits
            .get(user_id)
            .cloned()
            .unwrap_or_else(|| ReferralCredits {
                user_id: user_id.to_string(),
                total_earned: 0.0,
                total_redeemed: 0.0,
                pending: 0.0,
            })
    }

    /// Redeem credits. Returns the new available balance after redemption.
    pub fn redeem_credits(&self, user_id: &str, amount: f64) -> Result<f64> {
        if amount <= 0.0 {
            return Err(SrganError::InvalidInput(
                "Redemption amount must be positive".into(),
            ));
        }

        let mut credits = self.credits.write().map_err(|e| {
            SrganError::InvalidInput(format!("lock poisoned: {}", e))
        })?;

        let entry = credits.get_mut(user_id).ok_or_else(|| {
            SrganError::InvalidInput(format!("No credits found for user '{}'", user_id))
        })?;

        let available = entry.total_earned - entry.total_redeemed;
        if amount > available {
            return Err(SrganError::InvalidInput(format!(
                "Insufficient credits: requested {:.2}, available {:.2}",
                amount, available
            )));
        }

        entry.total_redeemed += amount;
        entry.pending = (entry.pending - amount).max(0.0);

        Ok(entry.total_earned - entry.total_redeemed)
    }

    /// Deactivate a referral code so it can no longer be used.
    pub fn deactivate_code(&self, code: &str) -> Result<()> {
        let mut referrals = self.referrals.write().map_err(|e| {
            SrganError::InvalidInput(format!("lock poisoned: {}", e))
        })?;
        let rc = referrals.get_mut(code).ok_or_else(|| {
            SrganError::InvalidInput(format!("Referral code '{}' not found", code))
        })?;
        rc.active = false;
        Ok(())
    }

    /// Get aggregate referral statistics for a user.
    pub fn get_referral_stats(&self, user_id: &str) -> ReferralStats {
        let referrals = self.referrals.read().unwrap_or_else(|e| e.into_inner());
        let user_codes: Vec<&ReferralCode> = referrals
            .values()
            .filter(|rc| rc.owner_user_id == user_id)
            .collect();

        let total_codes = user_codes.len();
        let active_codes = user_codes.iter().filter(|rc| rc.active).count();
        let total_signups: u64 = user_codes.iter().map(|rc| rc.uses).sum();

        let creds = self.get_user_credits(user_id);

        ReferralStats {
            total_codes,
            total_signups,
            total_earned: creds.total_earned,
            total_redeemed: creds.total_redeemed,
            available_credits: creds.total_earned - creds.total_redeemed,
            active_codes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_validate_code() {
        let store = ReferralStore::new();
        let code = store.generate_code("user1", 20, 10).unwrap();
        assert_eq!(code.code.len(), 8);
        assert!(code.active);
        assert_eq!(code.discount_pct, 20);
        assert_eq!(code.commission_pct, 10);

        let validated = store.validate_code(&code.code);
        assert!(validated.is_some());
    }

    #[test]
    fn test_record_signup_and_credit() {
        let store = ReferralStore::new();
        let code = store.generate_code("referrer1", 15, 10).unwrap();

        let signup = store.record_signup(&code.code, "newuser1").unwrap();
        assert_eq!(signup.referrer_user_id, "referrer1");
        assert!(!signup.credited);

        store.credit_referrer(&signup.id).unwrap();

        let credits = store.get_user_credits("referrer1");
        assert_eq!(credits.total_earned, 10.0);
    }

    #[test]
    fn test_redeem_credits() {
        let store = ReferralStore::new();
        let code = store.generate_code("referrer2", 20, 25).unwrap();

        let signup = store.record_signup(&code.code, "newuser2").unwrap();
        store.credit_referrer(&signup.id).unwrap();

        let balance = store.redeem_credits("referrer2", 10.0).unwrap();
        assert_eq!(balance, 15.0);

        // Over-redeem should fail
        assert!(store.redeem_credits("referrer2", 20.0).is_err());
    }

    #[test]
    fn test_deactivate_code() {
        let store = ReferralStore::new();
        let code = store.generate_code("user3", 10, 5).unwrap();

        store.deactivate_code(&code.code).unwrap();
        assert!(store.validate_code(&code.code).is_none());
    }

    #[test]
    fn test_get_referral_stats() {
        let store = ReferralStore::new();
        let code = store.generate_code("user4", 20, 10).unwrap();
        store.record_signup(&code.code, "ref1").unwrap();
        store.record_signup(&code.code, "ref2").unwrap();

        let stats = store.get_referral_stats("user4");
        assert_eq!(stats.total_codes, 1);
        assert_eq!(stats.total_signups, 2);
        assert_eq!(stats.active_codes, 1);
    }

    #[test]
    fn test_invalid_code_signup() {
        let store = ReferralStore::new();
        assert!(store.record_signup("NONEXIST", "user5").is_err());
    }
}
