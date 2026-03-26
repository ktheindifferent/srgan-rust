//! API key management backed by SQLite.
//!
//! ## Endpoints (handled by the web server)
//!   POST /api/v1/auth/keys — create a new key
//!   GET  /api/v1/auth/keys/:key — look up a key (admin only)

use std::path::Path;
use std::sync::Mutex;

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, SrganError};

// ── Tier ──────────────────────────────────────────────────────────────────────

/// Key tier — determines daily request quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyTier {
    Free,
    Pro,
    Enterprise,
}

impl KeyTier {
    /// Daily limit (`None` = unlimited).
    pub fn daily_limit(self) -> Option<u64> {
        match self {
            KeyTier::Free => Some(10),
            KeyTier::Pro => Some(1_000),
            KeyTier::Enterprise => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            KeyTier::Free => "free",
            KeyTier::Pro => "pro",
            KeyTier::Enterprise => "enterprise",
        }
    }

    /// Job processing priority (higher = processed sooner).
    pub fn priority(self) -> u8 {
        match self {
            KeyTier::Free => 0,
            KeyTier::Pro => 1,
            KeyTier::Enterprise => 2,
        }
    }
}

impl std::str::FromStr for KeyTier {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "free" => Ok(KeyTier::Free),
            "pro" => Ok(KeyTier::Pro),
            "enterprise" => Ok(KeyTier::Enterprise),
            other => Err(format!("unknown tier: {}", other)),
        }
    }
}

// ── ApiKey ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ApiKey {
    pub key: String,
    pub tier: KeyTier,
    pub created_at: i64,
    pub label: Option<String>,
}

// ── HTTP types ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateKeyRequest {
    pub tier: Option<String>,
    pub label: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateKeyResponse {
    pub key: String,
    pub tier: String,
    pub daily_limit: Option<u64>,
    pub label: Option<String>,
}

// ── KeyStore ──────────────────────────────────────────────────────────────────

/// Thread-safe SQLite-backed key + usage store.
pub struct KeyStore {
    conn: Mutex<Connection>,
}

impl KeyStore {
    /// Open (or create) a persistent store at `db_path`.
    pub fn open(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path).map_err(|e| {
            SrganError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("SQLite open: {}", e),
            ))
        })?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Open an in-memory store (for testing or single-run servers).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| {
            SrganError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("SQLite in-memory: {}", e),
            ))
        })?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS api_keys (
                key        TEXT PRIMARY KEY,
                tier       TEXT NOT NULL DEFAULT 'free',
                label      TEXT,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS key_usage (
                key      TEXT NOT NULL,
                date_utc TEXT NOT NULL,
                count    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key, date_utc)
            );",
        )
        .map_err(|e| {
            SrganError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("SQLite schema: {}", e),
            ))
        })
    }

    /// Generate and persist a new API key.
    pub fn create_key(&self, tier: KeyTier, label: Option<String>) -> Result<ApiKey> {
        let key = format!("sk-{}", Uuid::new_v4().to_string().replace('-', ""));
        let now = Utc::now().timestamp();

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO api_keys (key, tier, label, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![key, tier.as_str(), label, now],
        )
        .map_err(|e| {
            SrganError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("SQLite insert: {}", e),
            ))
        })?;

        Ok(ApiKey {
            key,
            tier,
            created_at: now,
            label,
        })
    }

    /// Look up a key; returns `None` if not found.
    pub fn get_key(&self, key: &str) -> Result<Option<ApiKey>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT key, tier, label, created_at FROM api_keys WHERE key = ?1",
            )
            .map_err(|e| SrganError::Network(format!("SQLite prepare: {}", e)))?;

        let result = stmt
            .query_row(params![key], |row| {
                let tier_str: String = row.get(1)?;
                let tier = tier_str.parse::<KeyTier>().unwrap_or(KeyTier::Free);
                Ok(ApiKey {
                    key: row.get(0)?,
                    tier,
                    label: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })
            .optional()
            .map_err(|e| SrganError::Network(format!("SQLite query: {}", e)))?;

        Ok(result)
    }

    /// Increment usage counter for today and return the new count.
    pub fn record_usage(&self, key: &str) -> Result<u64> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO key_usage (key, date_utc, count) VALUES (?1, ?2, 1)
             ON CONFLICT(key, date_utc) DO UPDATE SET count = count + 1",
            params![key, today],
        )
        .map_err(|e| {
            SrganError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("SQLite usage: {}", e),
            ))
        })?;

        let count: u64 = conn
            .query_row(
                "SELECT count FROM key_usage WHERE key = ?1 AND date_utc = ?2",
                params![key, today],
                |row| row.get(0),
            )
            .unwrap_or(0);

        Ok(count)
    }

    /// Today's usage for `key`.
    pub fn today_usage(&self, key: &str) -> u64 {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT count FROM key_usage WHERE key = ?1 AND date_utc = ?2",
            params![key, today],
            |row| row.get(0),
        )
        .unwrap_or(0)
    }

    /// All registered keys (for admin endpoint).
    pub fn all_keys(&self) -> Result<Vec<ApiKey>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT key, tier, label, created_at FROM api_keys ORDER BY created_at DESC",
            )
            .map_err(|e| SrganError::Network(format!("SQLite prepare: {}", e)))?;

        let keys = stmt
            .query_map([], |row| {
                let tier_str: String = row.get(1)?;
                let tier = tier_str.parse::<KeyTier>().unwrap_or(KeyTier::Free);
                Ok(ApiKey {
                    key: row.get(0)?,
                    tier,
                    label: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })
            .map_err(|e| SrganError::Network(format!("SQLite query: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(keys)
    }

    /// Usage today, grouped by tier.
    pub fn usage_today_by_tier(&self) -> std::collections::HashMap<String, u64> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT k.tier, COALESCE(SUM(u.count), 0)
             FROM api_keys k
             LEFT JOIN key_usage u ON k.key = u.key AND u.date_utc = ?1
             GROUP BY k.tier",
        ) {
            Ok(s) => s,
            Err(_) => return std::collections::HashMap::new(),
        };

        stmt.query_map(params![today], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_retrieve_key() {
        let store = KeyStore::open_in_memory().unwrap();
        let key = store.create_key(KeyTier::Pro, Some("test".into())).unwrap();
        assert!(key.key.starts_with("sk-"));

        let retrieved = store.get_key(&key.key).unwrap().unwrap();
        assert_eq!(retrieved.tier, KeyTier::Pro);
    }

    #[test]
    fn test_usage_tracking() {
        let store = KeyStore::open_in_memory().unwrap();
        let key = store.create_key(KeyTier::Free, None).unwrap();
        assert_eq!(store.record_usage(&key.key).unwrap(), 1);
        assert_eq!(store.record_usage(&key.key).unwrap(), 2);
        assert_eq!(store.today_usage(&key.key), 2);
    }

    #[test]
    fn test_tier_limits() {
        assert_eq!(KeyTier::Free.daily_limit(), Some(10));
        assert_eq!(KeyTier::Pro.daily_limit(), Some(1_000));
        assert_eq!(KeyTier::Enterprise.daily_limit(), None);
    }
}
