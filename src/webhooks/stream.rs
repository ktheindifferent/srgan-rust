//! Webhook event stream — persists all webhook events in SQLite and
//! exposes a replay endpoint for debugging:
//!
//! `GET /api/v1/webhooks/events?org_id=X&limit=100&since=<unix_timestamp>`

use std::sync::Mutex;
use rusqlite::params;
use serde::Serialize;

// ── Types ────────────────────────────────────────────────────────────────────

/// A persisted webhook event record.
#[derive(Debug, Clone, Serialize)]
pub struct WebhookEventRecord {
    pub id: i64,
    pub org_id: String,
    pub event_type: String,
    pub payload: String,
    pub webhook_id: Option<String>,
    pub status_code: Option<u16>,
    pub success: bool,
    pub error_message: Option<String>,
    pub created_at: u64,
}

/// Query parameters for the events replay endpoint.
#[derive(Debug)]
pub struct EventQuery {
    pub org_id: String,
    pub limit: usize,
    pub since: Option<u64>,
}

impl EventQuery {
    /// Parse from a query string like `org_id=X&limit=100&since=123456`.
    pub fn from_query_string(qs: &str) -> Option<Self> {
        let mut org_id = None;
        let mut limit = 100usize;
        let mut since = None;

        for pair in qs.split('&') {
            let mut kv = pair.splitn(2, '=');
            let key = kv.next().unwrap_or("");
            let val = kv.next().unwrap_or("");
            match key {
                "org_id" => org_id = Some(val.to_string()),
                "limit" => limit = val.parse().unwrap_or(100).min(1000),
                "since" => since = val.parse().ok(),
                _ => {}
            }
        }

        Some(Self {
            org_id: org_id?,
            limit,
            since,
        })
    }
}

// ── Database ─────────────────────────────────────────────────────────────────

/// SQLite-backed webhook event store for replay and debugging.
pub struct WebhookEventStore {
    conn: Mutex<rusqlite::Connection>,
}

impl WebhookEventStore {
    /// Open (or create) the event store at the given path.
    pub fn open(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = rusqlite::Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS webhook_events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id        TEXT NOT NULL,
                event_type    TEXT NOT NULL,
                payload       TEXT NOT NULL,
                webhook_id    TEXT,
                status_code   INTEGER,
                success       INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at    INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_we_org_id ON webhook_events(org_id);
            CREATE INDEX IF NOT EXISTS idx_we_created_at ON webhook_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_we_org_created ON webhook_events(org_id, created_at);",
        )?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Open an in-memory store (useful for tests and default startup).
    pub fn open_in_memory() -> Result<Self, rusqlite::Error> {
        Self::open(":memory:")
    }

    /// Record a webhook event.
    pub fn record(
        &self,
        org_id: &str,
        event_type: &str,
        payload: &str,
        webhook_id: Option<&str>,
        status_code: Option<u16>,
        success: bool,
        error_message: Option<&str>,
    ) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "INSERT INTO webhook_events
                    (org_id, event_type, payload, webhook_id, status_code, success, error_message, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    org_id,
                    event_type,
                    payload,
                    webhook_id,
                    status_code.map(|c| c as i64),
                    success as i32,
                    error_message,
                    now as i64,
                ],
            );
        }
    }

    /// Query events for an org with optional `since` timestamp filter.
    pub fn query(&self, q: &EventQuery) -> Vec<WebhookEventRecord> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return vec![],
        };

        let (sql, since_val) = match q.since {
            Some(ts) => (
                "SELECT id, org_id, event_type, payload, webhook_id, status_code, success, error_message, created_at
                 FROM webhook_events
                 WHERE org_id = ?1 AND created_at >= ?2
                 ORDER BY created_at DESC
                 LIMIT ?3",
                ts as i64,
            ),
            None => (
                "SELECT id, org_id, event_type, payload, webhook_id, status_code, success, error_message, created_at
                 FROM webhook_events
                 WHERE org_id = ?1 AND created_at >= ?2
                 ORDER BY created_at DESC
                 LIMIT ?3",
                0i64,
            ),
        };

        let mut stmt = match conn.prepare(sql) {
            Ok(s) => s,
            Err(_) => return vec![],
        };

        stmt.query_map(params![&q.org_id, since_val, q.limit as i64], |row| {
            Ok(WebhookEventRecord {
                id: row.get(0)?,
                org_id: row.get(1)?,
                event_type: row.get(2)?,
                payload: row.get(3)?,
                webhook_id: row.get(4)?,
                status_code: row.get::<_, Option<i64>>(5)?.map(|c| c as u16),
                success: row.get::<_, i32>(6)? != 0,
                error_message: row.get(7)?,
                created_at: row.get::<_, i64>(8)? as u64,
            })
        })
        .ok()
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Total number of events stored.
    pub fn count(&self) -> usize {
        self.conn
            .lock()
            .ok()
            .and_then(|c| {
                c.query_row("SELECT COUNT(*) FROM webhook_events", [], |row| {
                    row.get::<_, i64>(0)
                })
                .ok()
            })
            .unwrap_or(0) as usize
    }

    /// Delivery success rate (0.0–1.0) across all stored events.
    pub fn success_rate(&self) -> f64 {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return 0.0,
        };
        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM webhook_events", [], |row| row.get(0))
            .unwrap_or(0);
        if total == 0 {
            return 1.0;
        }
        let success: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM webhook_events WHERE success = 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        success as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_query() {
        let store = WebhookEventStore::open_in_memory().unwrap();
        store.record("org-1", "job.completed", r#"{"job_id":"j1"}"#, Some("wh_1"), Some(200), true, None);
        store.record("org-1", "job.failed", r#"{"job_id":"j2"}"#, Some("wh_1"), Some(500), false, Some("timeout"));
        store.record("org-2", "job.completed", r#"{"job_id":"j3"}"#, None, Some(200), true, None);

        let q = EventQuery { org_id: "org-1".into(), limit: 100, since: None };
        let events = store.query(&q);
        assert_eq!(events.len(), 2);

        let q2 = EventQuery { org_id: "org-2".into(), limit: 100, since: None };
        let events2 = store.query(&q2);
        assert_eq!(events2.len(), 1);
    }

    #[test]
    fn test_success_rate() {
        let store = WebhookEventStore::open_in_memory().unwrap();
        store.record("org-1", "job.completed", "{}", None, Some(200), true, None);
        store.record("org-1", "job.completed", "{}", None, Some(200), true, None);
        store.record("org-1", "job.failed", "{}", None, Some(500), false, None);

        let rate = store.success_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_query_string() {
        let q = EventQuery::from_query_string("org_id=org-1&limit=50&since=1700000000").unwrap();
        assert_eq!(q.org_id, "org-1");
        assert_eq!(q.limit, 50);
        assert_eq!(q.since, Some(1700000000));
    }
}
