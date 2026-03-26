//! Multi-tenant organization system.
//!
//! Organizations have a shared credit pool. When a member submits a job,
//! credits are deducted from the org pool instead of their personal balance.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Models ───────────────────────────────────────────────────────────────────

/// An organization with a shared credit pool.
#[derive(Debug, Clone, Serialize)]
pub struct Organization {
    pub id: String,
    pub name: String,
    pub owner_user_id: String,
    pub created_at: u64,
    pub credit_pool: u32,
}

/// Tracks per-member credit consumption within an org for the current month.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MemberUsage {
    pub user_id: String,
    pub credits_consumed: u32,
    /// YYYY-MM of the usage period.
    pub month: String,
}

// ── Request / Response types ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateOrgRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct AddMemberRequest {
    pub user_id: String,
}

#[derive(Debug, Serialize)]
pub struct OrgUsageResponse {
    pub org_id: String,
    pub month: String,
    pub credit_pool_remaining: u32,
    pub members: Vec<MemberUsage>,
    pub total_consumed: u32,
}

// ── In-memory org database ───────────────────────────────────────────────────

pub struct OrgDb {
    /// org_id → Organization
    orgs: HashMap<String, Organization>,
    /// org_id → set of member user_ids (owner is always a member)
    members: HashMap<String, Vec<String>>,
    /// user_id → org_id (a user can belong to at most one org)
    user_org: HashMap<String, String>,
    /// (org_id, user_id, "YYYY-MM") → credits consumed
    usage: HashMap<(String, String, String), u32>,
}

impl OrgDb {
    pub fn new() -> Self {
        Self {
            orgs: HashMap::new(),
            members: HashMap::new(),
            user_org: HashMap::new(),
            usage: HashMap::new(),
        }
    }

    /// Create a new organization. The caller becomes the owner and first member.
    pub fn create_org(&mut self, name: String, owner_user_id: String) -> Result<Organization, String> {
        // Owner can only own/belong to one org
        if self.user_org.contains_key(&owner_user_id) {
            return Err("User already belongs to an organization".to_string());
        }

        let org = Organization {
            id: Uuid::new_v4().to_string(),
            name,
            owner_user_id: owner_user_id.clone(),
            created_at: unix_now(),
            credit_pool: 0,
        };

        let id = org.id.clone();
        self.orgs.insert(id.clone(), org.clone());
        self.members.insert(id.clone(), vec![owner_user_id.clone()]);
        self.user_org.insert(owner_user_id, id);

        Ok(org)
    }

    /// Look up an organization by ID.
    pub fn get_org(&self, org_id: &str) -> Option<&Organization> {
        self.orgs.get(org_id)
    }

    /// Add a member to an organization. Only the owner may call this.
    pub fn add_member(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        new_member_id: String,
    ) -> Result<(), String> {
        let org = self.orgs.get(org_id).ok_or("Organization not found")?;
        if org.owner_user_id != requester_user_id {
            return Err("Only the org owner can add members".to_string());
        }
        if self.user_org.contains_key(&new_member_id) {
            return Err("User already belongs to an organization".to_string());
        }

        self.members
            .entry(org_id.to_string())
            .or_default()
            .push(new_member_id.clone());
        self.user_org.insert(new_member_id, org_id.to_string());
        Ok(())
    }

    /// Remove a member from an organization. Owner cannot remove themselves.
    pub fn remove_member(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        target_user_id: &str,
    ) -> Result<(), String> {
        let org = self.orgs.get(org_id).ok_or("Organization not found")?;
        if org.owner_user_id != requester_user_id {
            return Err("Only the org owner can remove members".to_string());
        }
        if target_user_id == org.owner_user_id {
            return Err("Cannot remove the org owner".to_string());
        }

        if let Some(list) = self.members.get_mut(org_id) {
            list.retain(|id| id != target_user_id);
        }
        self.user_org.remove(target_user_id);
        Ok(())
    }

    /// Returns the org_id a user belongs to, if any.
    pub fn user_org_id(&self, user_id: &str) -> Option<&String> {
        self.user_org.get(user_id)
    }

    /// Check membership.
    pub fn is_member(&self, org_id: &str, user_id: &str) -> bool {
        self.members
            .get(org_id)
            .map_or(false, |list| list.contains(&user_id.to_string()))
    }

    /// Consume one credit from the org pool. Returns false if empty.
    pub fn consume_org_credit(&mut self, org_id: &str, user_id: &str) -> bool {
        if let Some(org) = self.orgs.get_mut(org_id) {
            if org.credit_pool > 0 {
                org.credit_pool -= 1;
                let month = current_month();
                let key = (org_id.to_string(), user_id.to_string(), month);
                *self.usage.entry(key).or_insert(0) += 1;
                return true;
            }
        }
        false
    }

    /// Add credits to an org's pool (e.g. via admin or Stripe).
    pub fn add_credits(&mut self, org_id: &str, amount: u32) -> Result<(), String> {
        let org = self.orgs.get_mut(org_id).ok_or("Organization not found")?;
        org.credit_pool += amount;
        Ok(())
    }

    /// Per-member usage breakdown for the current month.
    pub fn usage_this_month(&self, org_id: &str) -> Option<OrgUsageResponse> {
        let org = self.orgs.get(org_id)?;
        let month = current_month();
        let member_list = self.members.get(org_id)?;

        let mut members = Vec::new();
        let mut total = 0u32;
        for uid in member_list {
            let key = (org_id.to_string(), uid.clone(), month.clone());
            let consumed = self.usage.get(&key).copied().unwrap_or(0);
            total += consumed;
            members.push(MemberUsage {
                user_id: uid.clone(),
                credits_consumed: consumed,
                month: month.clone(),
            });
        }

        Some(OrgUsageResponse {
            org_id: org_id.to_string(),
            month,
            credit_pool_remaining: org.credit_pool,
            members,
            total_consumed: total,
        })
    }

    /// List member user_ids for an org.
    pub fn list_members(&self, org_id: &str) -> Vec<String> {
        self.members.get(org_id).cloned().unwrap_or_default()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn current_month() -> String {
    let secs = unix_now();
    // Approximate: good enough for billing buckets
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let month = remaining_days / 30 + 1;
    format!("{}-{:02}", years, month.min(12))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_org_and_add_member() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        assert!(!org.id.is_empty());
        assert!(db.is_member(&org.id, "owner-1"));

        db.add_member(&org.id, "owner-1", "user-2".into()).unwrap();
        assert!(db.is_member(&org.id, "user-2"));
    }

    #[test]
    fn test_remove_member() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.add_member(&org.id, "owner-1", "user-2".into()).unwrap();
        db.remove_member(&org.id, "owner-1", "user-2").unwrap();
        assert!(!db.is_member(&org.id, "user-2"));
    }

    #[test]
    fn test_cannot_remove_owner() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        assert!(db.remove_member(&org.id, "owner-1", "owner-1").is_err());
    }

    #[test]
    fn test_org_credit_consumption() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.add_credits(&org.id, 5).unwrap();
        assert!(db.consume_org_credit(&org.id, "owner-1"));
        assert_eq!(db.get_org(&org.id).unwrap().credit_pool, 4);
    }

    #[test]
    fn test_user_cannot_join_two_orgs() {
        let mut db = OrgDb::new();
        let org1 = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.add_member(&org1.id, "owner-1", "user-2".into()).unwrap();

        let org2 = db.create_org("Beta".into(), "owner-3".into()).unwrap();
        assert!(db.add_member(&org2.id, "owner-3", "user-2".into()).is_err());
    }

    #[test]
    fn test_usage_this_month() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.add_credits(&org.id, 100).unwrap();
        db.add_member(&org.id, "owner-1", "user-2".into()).unwrap();

        db.consume_org_credit(&org.id, "owner-1");
        db.consume_org_credit(&org.id, "owner-1");
        db.consume_org_credit(&org.id, "user-2");

        let usage = db.usage_this_month(&org.id).unwrap();
        assert_eq!(usage.total_consumed, 3);
        assert_eq!(usage.credit_pool_remaining, 97);
    }
}
