//! GET /api/v1/jobs — list jobs with optional filtering and pagination.

use serde::{Deserialize, Serialize};

use crate::api::upscale::{JobRecord, PriorityJobQueue};

/// Query parameters for GET /api/v1/jobs
#[derive(Debug, Default, Deserialize)]
pub struct JobFilter {
    /// Filter by status: pending, processing, completed, failed, timed_out
    pub status: Option<String>,
    /// Zero-based page index (default: 0)
    pub page: Option<usize>,
    /// Results per page (default: 20, max: 100)
    pub per_page: Option<usize>,
    /// Only return jobs for this API key
    pub api_key: Option<String>,
}

/// Paginated response for GET /api/v1/jobs
#[derive(Debug, Serialize)]
pub struct JobListResponse {
    pub jobs: Vec<JobRecord>,
    pub total: usize,
    pub page: usize,
    pub per_page: usize,
    pub total_pages: usize,
}

/// List jobs matching `filter`, sorted newest-first.
pub fn list_jobs(queue: &PriorityJobQueue, filter: &JobFilter) -> JobListResponse {
    let per_page = filter.per_page.unwrap_or(20).min(100).max(1);
    let page = filter.page.unwrap_or(0);

    let mut all: Vec<JobRecord> = queue
        .all_jobs()
        .into_iter()
        .filter(|job| {
            // Status filter
            if let Some(ref s) = filter.status {
                if job.status.as_str() != s.as_str() {
                    return false;
                }
            }
            // API key filter
            if let Some(ref key) = filter.api_key {
                if &job.api_key != key {
                    return false;
                }
            }
            true
        })
        .collect();

    // Sort newest first
    all.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    let total = all.len();
    let total_pages = (total + per_page - 1) / per_page;

    let jobs: Vec<JobRecord> = all
        .into_iter()
        .skip(page * per_page)
        .take(per_page)
        .collect();

    JobListResponse {
        jobs,
        total,
        page,
        per_page,
        total_pages,
    }
}
