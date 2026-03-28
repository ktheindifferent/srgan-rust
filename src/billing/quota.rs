//! Per-org monthly usage quota enforcement.
//!
//! Each organisation has a monthly pixel budget (total pixels processed).
//! On every upscale request the server calls [`QuotaDb::check_and_increment`]
//! which atomically checks the current-month counter and returns whether the
//! request is allowed.
//!
//! State is stored in SQLite with a `(org_id, month)` composite key so that
//! history is retained and old months roll over automatically.
//!
//! Admin endpoints:
//! - `GET  /api/v1/admin/quotas?org_id=X` — view quota + usage
//! - `PUT  /api/v1/admin/quotas`          — update an org's monthly limit

use std::sync::Mutex;
use rusqlite::params;
use serde::{Deserialize, Serialize};

/// Current UTC month as `"YYYY-MM"`.
fn current_month() -> String {
    let now = chrono::Utc::now();
    format!("{}", now.format("%Y-%m"))
}

// ── Types ────────────────────────────────────────────────────────────────────

/// Quota status returned to callers.
#[derive(Debug, Clone, Serialize)]
pub struct QuotaStatus {
    pub org_id: String,
    pub month: String,
    pub pixels_used: u64,
    pub pixels_limit: u64,
    pub remaining: u64,
    pub exceeded: bool,
}

/// Request body for `PUT /api/v1/admin/quotas`.
#[derive(Debug, Deserialize)]
pub struct UpdateQuotaRequest {
    pub org_id: String,
    /// Monthly pixel limit.  Set to 0 for unlimited.
    pub pixels_limit: u64,
}

// ── Database ─────────────────────────────────────────────────────────────────

/// SQLite-backed per-org monthly quota store.
pub struct QuotaDb {
    conn: Mutex<rusqlite::Connection>,
}

impl QuotaDb {
    /// Open (or create) the quota database at the given path.
    pub fn open(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = rusqlite::Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS org_quotas (
                org_id       TEXT NOT NULL,
                pixels_limit INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (org_id)
            );
            CREATE TABLE IF NOT EXISTS org_usage (
                org_id       TEXT NOT NULL,
                month        TEXT NOT NULL,
                pixels_used  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (org_id, month)
            );",
        )?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Open an in-memory database (useful for tests and default server startup).
    pub fn open_in_memory() -> Result<Self, rusqlite::Error> {
        Self::open(":memory:")
    }

    /// Set the monthly pixel limit for an org. Creates the row if absent.
    pub fn set_limit(&self, org_id: &str, pixels_limit: u64) {
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "INSERT INTO org_quotas (org_id, pixels_limit) VALUES (?1, ?2)
                 ON CONFLICT(org_id) DO UPDATE SET pixels_limit = excluded.pixels_limit",
                params![org_id, pixels_limit as i64],
            );
        }
    }

    /// Get the configured limit for an org (0 = unlimited / not configured).
    pub fn get_limit(&self, org_id: &str) -> u64 {
        self.conn
            .lock()
            .ok()
            .and_then(|c| {
                c.query_row(
                    "SELECT pixels_limit FROM org_quotas WHERE org_id = ?1",
                    params![org_id],
                    |row| row.get::<_, i64>(0),
                )
                .ok()
            })
            .unwrap_or(0) as u64
    }

    /// Get current month usage for an org.
    fn get_usage(&self, conn: &rusqlite::Connection, org_id: &str, month: &str) -> u64 {
        conn.query_row(
            "SELECT pixels_used FROM org_usage WHERE org_id = ?1 AND month = ?2",
            params![org_id, month],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) as u64
    }

    /// Check whether the org can process `pixels` more pixels this month.
    /// If yes, atomically increments the counter and returns `Ok(QuotaStatus)`.
    /// If no, returns `Ok(QuotaStatus)` with `exceeded = true`.
    pub fn check_and_increment(&self, org_id: &str, pixels: u64) -> QuotaStatus {
        let month = current_month();
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => {
                return QuotaStatus {
                    org_id: org_id.to_string(),
                    month,
                    pixels_used: 0,
                    pixels_limit: 0,
                    remaining: 0,
                    exceeded: true,
                };
            }
        };

        let limit = conn
            .query_row(
                "SELECT pixels_limit FROM org_quotas WHERE org_id = ?1",
                params![org_id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0) as u64;

        let used = self.get_usage(&conn, org_id, &month);

        // 0 means unlimited
        if limit > 0 && used + pixels > limit {
            return QuotaStatus {
                org_id: org_id.to_string(),
                month,
                pixels_used: used,
                pixels_limit: limit,
                remaining: limit.saturating_sub(used),
                exceeded: true,
            };
        }

        // Increment
        let _ = conn.execute(
            "INSERT INTO org_usage (org_id, month, pixels_used) VALUES (?1, ?2, ?3)
             ON CONFLICT(org_id, month) DO UPDATE SET pixels_used = pixels_used + ?3",
            params![org_id, &month, pixels as i64],
        );

        let new_used = used + pixels;
        QuotaStatus {
            org_id: org_id.to_string(),
            month,
            pixels_used: new_used,
            pixels_limit: limit,
            remaining: if limit > 0 { limit.saturating_sub(new_used) } else { u64::MAX },
            exceeded: false,
        }
    }

    /// View quota status without incrementing (for admin/reporting).
    pub fn status(&self, org_id: &str) -> QuotaStatus {
        let month = current_month();
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => {
                return QuotaStatus {
                    org_id: org_id.to_string(),
                    month,
                    pixels_used: 0,
                    pixels_limit: 0,
                    remaining: 0,
                    exceeded: false,
                };
            }
        };

        let limit = conn
            .query_row(
                "SELECT pixels_limit FROM org_quotas WHERE org_id = ?1",
                params![org_id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0) as u64;

        let used = self.get_usage(&conn, org_id, &month);

        QuotaStatus {
            org_id: org_id.to_string(),
            month,
            pixels_used: used,
            pixels_limit: limit,
            remaining: if limit > 0 { limit.saturating_sub(used) } else { u64::MAX },
            exceeded: limit > 0 && used >= limit,
        }
    }

    /// Return top N orgs by pixel usage in the current month.
    pub fn top_orgs(&self, n: usize) -> Vec<QuotaStatus> {
        let month = current_month();
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return vec![],
        };
        let mut stmt = match conn.prepare(
            "SELECT u.org_id, u.pixels_used, COALESCE(q.pixels_limit, 0)
             FROM org_usage u
             LEFT JOIN org_quotas q ON q.org_id = u.org_id
             WHERE u.month = ?1
             ORDER BY u.pixels_used DESC
             LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        stmt.query_map(params![&month, n as i64], |row| {
            let org_id: String = row.get(0)?;
            let used: i64 = row.get(1)?;
            let limit: i64 = row.get(2)?;
            let used = used as u64;
            let limit = limit as u64;
            Ok(QuotaStatus {
                org_id,
                month: month.clone(),
                pixels_used: used,
                pixels_limit: limit,
                remaining: if limit > 0 { limit.saturating_sub(used) } else { u64::MAX },
                exceeded: limit > 0 && used >= limit,
            })
        })
        .ok()
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quota_set_and_check() {
        let db = QuotaDb::open_in_memory().unwrap();
        db.set_limit("org-1", 1_000_000);

        let status = db.check_and_increment("org-1", 500_000);
        assert!(!status.exceeded);
        assert_eq!(status.pixels_used, 500_000);
        assert_eq!(status.remaining, 500_000);

        let status = db.check_and_increment("org-1", 600_000);
        assert!(status.exceeded);
        // Usage should not have increased
        assert_eq!(status.pixels_used, 500_000);
    }

    #[test]
    fn test_unlimited_quota() {
        let db = QuotaDb::open_in_memory().unwrap();
        // No limit set (default 0 = unlimited)
        let status = db.check_and_increment("org-free", 10_000_000);
        assert!(!status.exceeded);
    }

    #[test]
    fn test_status_view() {
        let db = QuotaDb::open_in_memory().unwrap();
        db.set_limit("org-2", 2_000_000);
        db.check_and_increment("org-2", 100_000);

        let s = db.status("org-2");
        assert_eq!(s.pixels_used, 100_000);
        assert_eq!(s.pixels_limit, 2_000_000);
        assert!(!s.exceeded);
    }

    #[test]
    fn test_top_orgs() {
        let db = QuotaDb::open_in_memory().unwrap();
        db.set_limit("org-a", 5_000_000);
        db.set_limit("org-b", 5_000_000);
        db.check_and_increment("org-a", 3_000_000);
        db.check_and_increment("org-b", 1_000_000);

        let top = db.top_orgs(10);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].org_id, "org-a");
    }
}
