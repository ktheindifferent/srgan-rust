//! Multi-tenant organization system with role-based access control.
//!
//! Organizations have a shared credit pool, per-org API key pools, usage
//! quotas, and member roles (Admin / Editor / Viewer).  When a member submits
//! a job, credits are deducted from the org pool instead of their personal
//! balance.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Models ───────────────────────────────────────────────────────────────────

/// Role-based access within an organization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemberRole {
    Admin,
    Editor,
    Viewer,
}

impl MemberRole {
    /// Whether this role can manage members and keys.
    pub fn can_manage(&self) -> bool {
        matches!(self, MemberRole::Admin)
    }

    /// Whether this role can submit upscaling jobs.
    pub fn can_submit_jobs(&self) -> bool {
        matches!(self, MemberRole::Admin | MemberRole::Editor)
    }
}

impl std::fmt::Display for MemberRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemberRole::Admin => write!(f, "admin"),
            MemberRole::Editor => write!(f, "editor"),
            MemberRole::Viewer => write!(f, "viewer"),
        }
    }
}

/// Pending invitation to join an organization.
#[derive(Debug, Clone, Serialize)]
pub struct OrgInvite {
    pub id: String,
    pub org_id: String,
    pub email: String,
    pub role: MemberRole,
    pub invited_by: String,
    pub created_at: u64,
    pub accepted: bool,
}

/// An API key belonging to an organization's shared pool.
#[derive(Debug, Clone, Serialize)]
pub struct OrgApiKey {
    pub id: String,
    pub org_id: String,
    pub name: String,
    pub key_hash: String,
    pub created_by: String,
    pub created_at: u64,
    pub active: bool,
}

/// An organization with a shared credit pool.
#[derive(Debug, Clone, Serialize)]
pub struct Organization {
    pub id: String,
    pub name: String,
    pub owner_user_id: String,
    pub created_at: u64,
    pub credit_pool: u32,
    /// Monthly usage quota (0 = unlimited).
    pub monthly_quota: u32,
    /// Billing plan identifier (e.g. "free", "pro", "enterprise").
    pub billing_plan: String,
}

/// Tracks per-member credit consumption within an org for the current month.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MemberUsage {
    pub user_id: String,
    pub credits_consumed: u32,
    /// YYYY-MM of the usage period.
    pub month: String,
}

/// Membership record including role.
#[derive(Debug, Clone, Serialize)]
pub struct OrgMember {
    pub user_id: String,
    pub role: MemberRole,
    pub joined_at: u64,
}

// ── Request / Response types ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateOrgRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct AddMemberRequest {
    pub user_id: String,
    #[serde(default = "default_editor_role")]
    pub role: MemberRole,
}

fn default_editor_role() -> MemberRole {
    MemberRole::Editor
}

#[derive(Debug, Deserialize)]
pub struct InviteMemberRequest {
    pub email: String,
    #[serde(default = "default_editor_role")]
    pub role: MemberRole,
}

#[derive(Debug, Deserialize)]
pub struct UpdateMemberRoleRequest {
    pub role: MemberRole,
}

#[derive(Debug, Deserialize)]
pub struct CreateOrgApiKeyRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct SetOrgQuotaRequest {
    pub monthly_quota: u32,
}

#[derive(Debug, Deserialize)]
pub struct SetOrgBillingPlanRequest {
    pub plan: String,
}

#[derive(Debug, Serialize)]
pub struct OrgUsageResponse {
    pub org_id: String,
    pub month: String,
    pub credit_pool_remaining: u32,
    pub monthly_quota: u32,
    pub billing_plan: String,
    pub members: Vec<MemberUsage>,
    pub total_consumed: u32,
    pub quota_remaining: Option<u32>,
}

// ── In-memory org database ───────────────────────────────────────────────────

pub struct OrgDb {
    /// org_id → Organization
    orgs: HashMap<String, Organization>,
    /// org_id → list of OrgMember
    members: HashMap<String, Vec<OrgMember>>,
    /// user_id → org_id (a user can belong to at most one org)
    user_org: HashMap<String, String>,
    /// (org_id, user_id, "YYYY-MM") → credits consumed
    usage: HashMap<(String, String, String), u32>,
    /// invite_id → OrgInvite
    invites: HashMap<String, OrgInvite>,
    /// org_id → list of OrgApiKey
    api_keys: HashMap<String, Vec<OrgApiKey>>,
}

impl OrgDb {
    pub fn new() -> Self {
        Self {
            orgs: HashMap::new(),
            members: HashMap::new(),
            user_org: HashMap::new(),
            usage: HashMap::new(),
            invites: HashMap::new(),
            api_keys: HashMap::new(),
        }
    }

    /// Create a new organization. The caller becomes the owner (Admin) and first member.
    pub fn create_org(&mut self, name: String, owner_user_id: String) -> Result<Organization, String> {
        if self.user_org.contains_key(&owner_user_id) {
            return Err("User already belongs to an organization".to_string());
        }

        let org = Organization {
            id: Uuid::new_v4().to_string(),
            name,
            owner_user_id: owner_user_id.clone(),
            created_at: unix_now(),
            credit_pool: 0,
            monthly_quota: 0,
            billing_plan: "free".to_string(),
        };

        let id = org.id.clone();
        let member = OrgMember {
            user_id: owner_user_id.clone(),
            role: MemberRole::Admin,
            joined_at: unix_now(),
        };
        self.orgs.insert(id.clone(), org.clone());
        self.members.insert(id.clone(), vec![member]);
        self.user_org.insert(owner_user_id, id);

        Ok(org)
    }

    /// Look up an organization by ID.
    pub fn get_org(&self, org_id: &str) -> Option<&Organization> {
        self.orgs.get(org_id)
    }

    /// Get a member's role within an org.
    pub fn member_role(&self, org_id: &str, user_id: &str) -> Option<MemberRole> {
        self.members.get(org_id).and_then(|list| {
            list.iter().find(|m| m.user_id == user_id).map(|m| m.role)
        })
    }

    /// Add a member to an organization. Requires Admin role.
    pub fn add_member(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        new_member_id: String,
    ) -> Result<(), String> {
        self.add_member_with_role(org_id, requester_user_id, new_member_id, MemberRole::Editor)
    }

    /// Add a member with a specific role. Requires Admin role.
    pub fn add_member_with_role(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        new_member_id: String,
        role: MemberRole,
    ) -> Result<(), String> {
        let _org = self.orgs.get(org_id).ok_or("Organization not found")?;
        let requester_role = self.member_role(org_id, requester_user_id)
            .ok_or("Requester is not a member of this organization")?;
        if !requester_role.can_manage() {
            return Err("Only admins can add members".to_string());
        }
        if self.user_org.contains_key(&new_member_id) {
            return Err("User already belongs to an organization".to_string());
        }

        let member = OrgMember {
            user_id: new_member_id.clone(),
            role,
            joined_at: unix_now(),
        };
        self.members
            .entry(org_id.to_string())
            .or_default()
            .push(member);
        self.user_org.insert(new_member_id, org_id.to_string());
        Ok(())
    }

    /// Invite a user by email. Creates a pending invite record.
    pub fn invite_member(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        email: String,
        role: MemberRole,
    ) -> Result<OrgInvite, String> {
        let _org = self.orgs.get(org_id).ok_or("Organization not found")?;
        let requester_role = self.member_role(org_id, requester_user_id)
            .ok_or("Requester is not a member of this organization")?;
        if !requester_role.can_manage() {
            return Err("Only admins can invite members".to_string());
        }

        // Check for duplicate pending invite
        for inv in self.invites.values() {
            if inv.org_id == org_id && inv.email == email && !inv.accepted {
                return Err("An invite for this email is already pending".to_string());
            }
        }

        let invite = OrgInvite {
            id: format!("inv_{}", Uuid::new_v4()),
            org_id: org_id.to_string(),
            email,
            role,
            invited_by: requester_user_id.to_string(),
            created_at: unix_now(),
            accepted: false,
        };

        self.invites.insert(invite.id.clone(), invite.clone());
        Ok(invite)
    }

    /// Accept a pending invite by invite ID, binding user_id to the org.
    pub fn accept_invite(
        &mut self,
        invite_id: &str,
        user_id: String,
    ) -> Result<(), String> {
        let invite = self.invites.get_mut(invite_id)
            .ok_or("Invite not found")?;
        if invite.accepted {
            return Err("Invite already accepted".to_string());
        }
        if self.user_org.contains_key(&user_id) {
            return Err("User already belongs to an organization".to_string());
        }
        let org_id = invite.org_id.clone();
        let role = invite.role;
        invite.accepted = true;

        let member = OrgMember {
            user_id: user_id.clone(),
            role,
            joined_at: unix_now(),
        };
        self.members
            .entry(org_id.clone())
            .or_default()
            .push(member);
        self.user_org.insert(user_id, org_id);
        Ok(())
    }

    /// List pending invites for an org.
    pub fn list_invites(&self, org_id: &str) -> Vec<&OrgInvite> {
        self.invites.values()
            .filter(|inv| inv.org_id == org_id && !inv.accepted)
            .collect()
    }

    /// Update a member's role. Requires Admin. Cannot demote the owner.
    pub fn update_member_role(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        target_user_id: &str,
        new_role: MemberRole,
    ) -> Result<(), String> {
        let org = self.orgs.get(org_id).ok_or("Organization not found")?;
        let requester_role = self.member_role(org_id, requester_user_id)
            .ok_or("Requester is not a member")?;
        if !requester_role.can_manage() {
            return Err("Only admins can update roles".to_string());
        }
        if target_user_id == org.owner_user_id && new_role != MemberRole::Admin {
            return Err("Cannot demote the org owner".to_string());
        }

        if let Some(list) = self.members.get_mut(org_id) {
            if let Some(m) = list.iter_mut().find(|m| m.user_id == target_user_id) {
                m.role = new_role;
                return Ok(());
            }
        }
        Err("Member not found".to_string())
    }

    /// Remove a member from an organization. Owner cannot remove themselves.
    pub fn remove_member(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        target_user_id: &str,
    ) -> Result<(), String> {
        let org = self.orgs.get(org_id).ok_or("Organization not found")?;
        let requester_role = self.member_role(org_id, requester_user_id)
            .ok_or("Requester is not a member")?;
        if !requester_role.can_manage() {
            return Err("Only admins can remove members".to_string());
        }
        if target_user_id == org.owner_user_id {
            return Err("Cannot remove the org owner".to_string());
        }

        if let Some(list) = self.members.get_mut(org_id) {
            list.retain(|m| m.user_id != target_user_id);
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
            .map_or(false, |list| list.iter().any(|m| m.user_id == user_id))
    }

    /// Consume one credit from the org pool. Checks quota. Returns false if exhausted.
    pub fn consume_org_credit(&mut self, org_id: &str, user_id: &str) -> bool {
        // Read immutable state first to avoid overlapping borrows
        let (credit_pool, monthly_quota) = match self.orgs.get(org_id) {
            Some(org) => (org.credit_pool, org.monthly_quota),
            None => return false,
        };
        if credit_pool == 0 {
            return false;
        }
        // Enforce monthly quota if set
        if monthly_quota > 0 {
            let month = current_month();
            let total_this_month: u32 = self.members.get(org_id)
                .map(|members| {
                    members.iter().map(|m| {
                        let key = (org_id.to_string(), m.user_id.clone(), month.clone());
                        self.usage.get(&key).copied().unwrap_or(0)
                    }).sum()
                })
                .unwrap_or(0);
            if total_this_month >= monthly_quota {
                return false;
            }
        }
        // Now take the mutable borrow — we already checked existence above via credit_pool access
        let org = match self.orgs.get_mut(org_id) {
            Some(o) => o,
            None => return false,
        };
        org.credit_pool -= 1;
        let month = current_month();
        let key = (org_id.to_string(), user_id.to_string(), month);
        *self.usage.entry(key).or_insert(0) += 1;
        true
    }

    /// Add credits to an org's pool (e.g. via admin or Stripe).
    pub fn add_credits(&mut self, org_id: &str, amount: u32) -> Result<(), String> {
        let org = self.orgs.get_mut(org_id).ok_or("Organization not found")?;
        org.credit_pool += amount;
        Ok(())
    }

    /// Set the monthly usage quota for an org.
    pub fn set_monthly_quota(&mut self, org_id: &str, quota: u32) -> Result<(), String> {
        let org = self.orgs.get_mut(org_id).ok_or("Organization not found")?;
        org.monthly_quota = quota;
        Ok(())
    }

    /// Set the billing plan for an org.
    pub fn set_billing_plan(&mut self, org_id: &str, plan: String) -> Result<(), String> {
        let org = self.orgs.get_mut(org_id).ok_or("Organization not found")?;
        org.billing_plan = plan;
        Ok(())
    }

    // ── Per-org API key pool ─────────────────────────────────────────────────

    /// Create a new API key for the org. Requires Admin role.
    pub fn create_org_api_key(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        name: String,
    ) -> Result<OrgApiKey, String> {
        let _org = self.orgs.get(org_id).ok_or("Organization not found")?;
        let role = self.member_role(org_id, requester_user_id)
            .ok_or("Not a member")?;
        if !role.can_manage() {
            return Err("Only admins can create org API keys".to_string());
        }

        let key = OrgApiKey {
            id: format!("orgkey_{}", Uuid::new_v4()),
            org_id: org_id.to_string(),
            name,
            key_hash: format!("osk_{}", Uuid::new_v4()),
            created_by: requester_user_id.to_string(),
            created_at: unix_now(),
            active: true,
        };
        self.api_keys
            .entry(org_id.to_string())
            .or_default()
            .push(key.clone());
        Ok(key)
    }

    /// List active API keys for an org.
    pub fn list_org_api_keys(&self, org_id: &str) -> Vec<&OrgApiKey> {
        self.api_keys.get(org_id)
            .map(|keys| keys.iter().filter(|k| k.active).collect())
            .unwrap_or_default()
    }

    /// Revoke an org API key.
    pub fn revoke_org_api_key(
        &mut self,
        org_id: &str,
        requester_user_id: &str,
        key_id: &str,
    ) -> Result<(), String> {
        let role = self.member_role(org_id, requester_user_id)
            .ok_or("Not a member")?;
        if !role.can_manage() {
            return Err("Only admins can revoke org API keys".to_string());
        }
        if let Some(keys) = self.api_keys.get_mut(org_id) {
            if let Some(key) = keys.iter_mut().find(|k| k.id == key_id) {
                key.active = false;
                return Ok(());
            }
        }
        Err("API key not found".to_string())
    }

    /// Per-member usage breakdown for the current month.
    pub fn usage_this_month(&self, org_id: &str) -> Option<OrgUsageResponse> {
        let org = self.orgs.get(org_id)?;
        let month = current_month();
        let member_list = self.members.get(org_id)?;

        let mut members = Vec::new();
        let mut total = 0u32;
        for m in member_list {
            let key = (org_id.to_string(), m.user_id.clone(), month.clone());
            let consumed = self.usage.get(&key).copied().unwrap_or(0);
            total += consumed;
            members.push(MemberUsage {
                user_id: m.user_id.clone(),
                credits_consumed: consumed,
                month: month.clone(),
            });
        }

        let quota_remaining = if org.monthly_quota > 0 {
            Some(org.monthly_quota.saturating_sub(total))
        } else {
            None
        };

        Some(OrgUsageResponse {
            org_id: org_id.to_string(),
            month,
            credit_pool_remaining: org.credit_pool,
            monthly_quota: org.monthly_quota,
            billing_plan: org.billing_plan.clone(),
            members,
            total_consumed: total,
            quota_remaining,
        })
    }

    /// List member user_ids for an org.
    pub fn list_members(&self, org_id: &str) -> Vec<String> {
        self.members.get(org_id)
            .map(|list| list.iter().map(|m| m.user_id.clone()).collect())
            .unwrap_or_default()
    }

    /// List members with role info.
    pub fn list_members_with_roles(&self, org_id: &str) -> Vec<&OrgMember> {
        self.members.get(org_id)
            .map(|list| list.iter().collect())
            .unwrap_or_default()
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
        assert_eq!(db.member_role(&org.id, "owner-1"), Some(MemberRole::Admin));

        db.add_member(&org.id, "owner-1", "user-2".into()).unwrap();
        assert!(db.is_member(&org.id, "user-2"));
        assert_eq!(db.member_role(&org.id, "user-2"), Some(MemberRole::Editor));
    }

    #[test]
    fn test_add_member_with_role() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.add_member_with_role(&org.id, "owner-1", "viewer-1".into(), MemberRole::Viewer).unwrap();
        assert_eq!(db.member_role(&org.id, "viewer-1"), Some(MemberRole::Viewer));
    }

    #[test]
    fn test_non_admin_cannot_add_member() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.add_member_with_role(&org.id, "owner-1", "editor-1".into(), MemberRole::Editor).unwrap();
        assert!(db.add_member(&org.id, "editor-1", "user-3".into()).is_err());
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
    fn test_monthly_quota_enforcement() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.add_credits(&org.id, 100).unwrap();
        db.set_monthly_quota(&org.id, 2).unwrap();

        assert!(db.consume_org_credit(&org.id, "owner-1")); // 1 of 2
        assert!(db.consume_org_credit(&org.id, "owner-1")); // 2 of 2
        assert!(!db.consume_org_credit(&org.id, "owner-1")); // over quota
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
    fn test_invite_and_accept() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();

        let invite = db.invite_member(&org.id, "owner-1", "bob@example.com".into(), MemberRole::Editor).unwrap();
        assert!(!invite.accepted);

        db.accept_invite(&invite.id, "user-bob".into()).unwrap();
        assert!(db.is_member(&org.id, "user-bob"));
        assert_eq!(db.member_role(&org.id, "user-bob"), Some(MemberRole::Editor));
    }

    #[test]
    fn test_duplicate_invite_rejected() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.invite_member(&org.id, "owner-1", "bob@example.com".into(), MemberRole::Editor).unwrap();
        assert!(db.invite_member(&org.id, "owner-1", "bob@example.com".into(), MemberRole::Viewer).is_err());
    }

    #[test]
    fn test_org_api_key_pool() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        let key = db.create_org_api_key(&org.id, "owner-1", "production".into()).unwrap();
        assert!(key.active);
        assert_eq!(db.list_org_api_keys(&org.id).len(), 1);

        db.revoke_org_api_key(&org.id, "owner-1", &key.id).unwrap();
        assert_eq!(db.list_org_api_keys(&org.id).len(), 0);
    }

    #[test]
    fn test_update_member_role() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        db.add_member(&org.id, "owner-1", "user-2".into()).unwrap();
        assert_eq!(db.member_role(&org.id, "user-2"), Some(MemberRole::Editor));

        db.update_member_role(&org.id, "owner-1", "user-2", MemberRole::Viewer).unwrap();
        assert_eq!(db.member_role(&org.id, "user-2"), Some(MemberRole::Viewer));
    }

    #[test]
    fn test_cannot_demote_owner() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        assert!(db.update_member_role(&org.id, "owner-1", "owner-1", MemberRole::Viewer).is_err());
    }

    #[test]
    fn test_billing_plan() {
        let mut db = OrgDb::new();
        let org = db.create_org("Acme".into(), "owner-1".into()).unwrap();
        assert_eq!(org.billing_plan, "free");
        db.set_billing_plan(&org.id, "enterprise".into()).unwrap();
        assert_eq!(db.get_org(&org.id).unwrap().billing_plan, "enterprise");
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
