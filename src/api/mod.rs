//! REST API support modules.
//!
//! - `auth`       — API key management (SQLite)
//! - `rate_limit` — per-key daily quota enforcement
//! - `admin`      — admin stats endpoint (requires ADMIN_KEY env var)
//! - `upscale`    — priority job queue with timeout, cleanup, and webhooks
//! - `jobs`       — job listing with filtering and pagination
//! - `billing`    — subscription/billing types
//! - `middleware` — HTTP middleware utilities

pub mod admin;
pub mod auth;
pub mod batch;
pub mod billing;
pub mod jobs;
pub mod middleware;
pub mod org;
pub mod rate_limit;
pub mod rate_limit_dashboard;
pub mod upscale;

pub use auth::{
    ApiKey, Claims, CreateKeyRequest, CreateKeyResponse, KeyListEntry, KeyStore, KeyTier,
    RegisterRequest, RegisterResponse, RotateKeyResponse,
    issue_jwt, verify_jwt,
};
pub use rate_limit::{ApiRateLimiter, RateLimitResult};
pub use admin::{
    AdminStats, ApiKeyInfo, JobHistoryEntry, ModelMetrics, ModelMetricsTracker,
    SystemStats, check_admin_auth, get_full_stats, get_stats,
};
pub use upscale::{
    deliver_webhook, unix_now,
    AsyncUpscaleRequest, AsyncUpscaleResponse,
    JobPriority, JobRecord, JobStatus, PriorityJobQueue,
    WebhookConfig, WebhookDeliveryState,
};
pub use jobs::{JobFilter, JobListResponse, list_jobs};
pub use org::{OrgDb, Organization, OrgUsageResponse, CreateOrgRequest, AddMemberRequest};
