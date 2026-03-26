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
pub mod billing;
pub mod jobs;
pub mod middleware;
pub mod rate_limit;
pub mod upscale;

pub use auth::{ApiKey, CreateKeyRequest, CreateKeyResponse, KeyStore, KeyTier};
pub use rate_limit::{ApiRateLimiter, RateLimitResult};
pub use admin::{AdminStats, check_admin_auth, get_stats};
pub use upscale::{AsyncUpscaleRequest, AsyncUpscaleResponse, JobPriority, JobRecord, JobStatus, PriorityJobQueue};
pub use jobs::{JobFilter, JobListResponse, list_jobs};
