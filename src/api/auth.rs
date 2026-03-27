//! API key management and user authentication backed by SQLite.
//!
//! ## Endpoints (handled by the web server)
//!   POST /api/register              — sign up (email + password → JWT)
//!   GET  /api/keys                  — list caller's active API keys
//!   POST /api/keys                  — create a named API key
//!   DELETE /api/keys/:id            — revoke a key
//!   POST /api/keys/:id/rotate       — rotate key (24 h grace period on old key)
//!   POST /api/v1/auth/keys          — legacy: create a new key (admin)
//!   GET  /api/v1/auth/keys/:key     — legacy: look up a key (admin only)

use std::path::Path;
use std::sync::Mutex;

use chrono::Utc;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, SrganError};

// ── JWT ──────────────────────────────────────────────────────────────────────

/// JWT claims embedded in every token.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    /// Subject — the user_id.
    pub sub: String,
    /// Email address.
    pub email: String,
    /// Issued-at (Unix seconds).
    pub iat: i64,
    /// Expiration (Unix seconds).
    pub exp: i64,
}

fn jwt_secret() -> String {
    std::env::var("JWT_SECRET").unwrap_or_else(|_| "srgan-jwt-dev-secret".to_string())
}

/// Create a signed JWT for `user_id` / `email`, valid for 24 hours.
pub fn issue_jwt(user_id: &str, email: &str) -> Result<String> {
    let now = Utc::now().timestamp();
    let claims = Claims {
        sub: user_id.to_string(),
        email: email.to_string(),
        iat: now,
        exp: now + 86400, // 24 h
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(jwt_secret().as_bytes()),
    )
    .map_err(|e| SrganError::Network(format!("JWT encode: {}", e)))
}

/// Validate a JWT and return its claims.
pub fn verify_jwt(token: &str) -> Result<Claims> {
    let data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(jwt_secret().as_bytes()),
        &Validation::default(),
    )
    .map_err(|e| SrganError::Network(format!("JWT verify: {}", e)))?;
    Ok(data.claims)
}

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
    /// Owner user_id (None for legacy keys created before user accounts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Unix timestamp of last usage (0 = never used).
    pub last_used_at: i64,
    /// Whether this key has been revoked.
    #[serde(skip_serializing)]
    pub revoked: bool,
    /// If this key was rotated, the old key remains valid until this timestamp.
    #[serde(skip_serializing)]
    pub grace_expires_at: Option<i64>,
    /// The key id (short identifier for management endpoints).
    pub id: String,
}

// ── HTTP types ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub user_id: String,
    pub email: String,
    pub token: String,
}

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

#[derive(Debug, Serialize)]
pub struct KeyListEntry {
    pub id: String,
    pub key_prefix: String,
    pub label: Option<String>,
    pub tier: String,
    pub created_at: i64,
    pub last_used_at: i64,
}

#[derive(Debug, Serialize)]
pub struct RotateKeyResponse {
    pub new_key: String,
    pub old_key_valid_until: i64,
    pub id: String,
}

// ── KeyStore ──────────────────────────────────────────────────────────────────

/// Thread-safe SQLite-backed key + usage + user store.
pub struct KeyStore {
    conn: Mutex<Connection>,
}

fn sqlite_err(msg: String) -> SrganError {
    SrganError::Io(std::io::Error::new(std::io::ErrorKind::Other, msg))
}

impl KeyStore {
    /// Open (or create) a persistent store at `db_path`.
    pub fn open(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)
            .map_err(|e| sqlite_err(format!("SQLite open: {}", e)))?;
        let store = Self { conn: Mutex::new(conn) };
        store.init_schema()?;
        Ok(store)
    }

    /// Open an in-memory store (for testing or single-run servers).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| sqlite_err(format!("SQLite in-memory: {}", e)))?;
        let store = Self { conn: Mutex::new(conn) };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS users (
                user_id       TEXT PRIMARY KEY,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                key              TEXT PRIMARY KEY,
                id               TEXT NOT NULL UNIQUE,
                tier             TEXT NOT NULL DEFAULT 'free',
                label            TEXT,
                user_id          TEXT,
                created_at       INTEGER NOT NULL,
                last_used_at     INTEGER NOT NULL DEFAULT 0,
                revoked          INTEGER NOT NULL DEFAULT 0,
                grace_expires_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS key_usage (
                key      TEXT NOT NULL,
                date_utc TEXT NOT NULL,
                count    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key, date_utc)
            );",
        )
        .map_err(|e| sqlite_err(format!("SQLite schema: {}", e)))
    }

    // ── User registration ────────────────────────────────────────────────────

    /// Register a new user. Returns user_id.
    pub fn register_user(&self, email: &str, password: &str) -> Result<String> {
        let hash = bcrypt::hash(password, bcrypt::DEFAULT_COST)
            .map_err(|e| sqlite_err(format!("bcrypt: {}", e)))?;
        let user_id = format!("usr_{}", Uuid::new_v4().to_string().replace('-', ""));
        let now = Utc::now().timestamp();

        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        conn.execute(
            "INSERT INTO users (user_id, email, password_hash, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![user_id, email, hash, now],
        )
        .map_err(|e| {
            if e.to_string().contains("UNIQUE") {
                SrganError::InvalidInput("Email already registered".into())
            } else {
                sqlite_err(format!("SQLite insert user: {}", e))
            }
        })?;

        log::info!("[WELCOME EMAIL] To: {} — Welcome to SRGAN! Your account is ready.", email);

        Ok(user_id)
    }

    /// Authenticate a user by email + password. Returns (user_id, email).
    pub fn authenticate_user(&self, email: &str, password: &str) -> Result<(String, String)> {
        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT user_id, password_hash FROM users WHERE email = ?1",
                params![email],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| sqlite_err(format!("SQLite query user: {}", e)))?;

        match row {
            None => Err(SrganError::InvalidInput("Invalid email or password".into())),
            Some((user_id, hash)) => {
                let valid = bcrypt::verify(password, &hash)
                    .map_err(|e| sqlite_err(format!("bcrypt verify: {}", e)))?;
                if valid {
                    Ok((user_id, email.to_string()))
                } else {
                    Err(SrganError::InvalidInput("Invalid email or password".into()))
                }
            }
        }
    }

    /// Look up the user_id that owns a given API key (active, non-revoked).
    pub fn user_id_for_key(&self, key: &str) -> Result<Option<String>> {
        let now = Utc::now().timestamp();
        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        conn.query_row(
            "SELECT user_id FROM api_keys
             WHERE key = ?1 AND (revoked = 0 OR (grace_expires_at IS NOT NULL AND grace_expires_at > ?2))",
            params![key, now],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| SrganError::Network(format!("SQLite query: {}", e)))
    }

    // ── Key CRUD ─────────────────────────────────────────────────────────────

    /// Generate and persist a new API key.
    pub fn create_key(&self, tier: KeyTier, label: Option<String>) -> Result<ApiKey> {
        self.create_key_for_user(tier, label, None)
    }

    /// Generate a key owned by a specific user.
    pub fn create_key_for_user(
        &self,
        tier: KeyTier,
        label: Option<String>,
        user_id: Option<&str>,
    ) -> Result<ApiKey> {
        let key = format!("sk_{}", Uuid::new_v4().to_string().replace('-', ""));
        let id = format!("key_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
        let now = Utc::now().timestamp();

        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        conn.execute(
            "INSERT INTO api_keys (key, id, tier, label, user_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![key, id, tier.as_str(), label, user_id, now],
        )
        .map_err(|e| sqlite_err(format!("SQLite insert: {}", e)))?;

        Ok(ApiKey {
            key,
            id,
            tier,
            created_at: now,
            label,
            user_id: user_id.map(|s| s.to_string()),
            last_used_at: 0,
            revoked: false,
            grace_expires_at: None,
        })
    }

    /// Look up a key; returns `None` if not found or revoked (past grace period).
    pub fn get_key(&self, key: &str) -> Result<Option<ApiKey>> {
        let now = Utc::now().timestamp();
        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        let mut stmt = conn
            .prepare(
                "SELECT key, id, tier, label, user_id, created_at, last_used_at, revoked, grace_expires_at
                 FROM api_keys WHERE key = ?1",
            )
            .map_err(|e| SrganError::Network(format!("SQLite prepare: {}", e)))?;

        let result = stmt
            .query_row(params![key], |row| {
                let tier_str: String = row.get(2)?;
                let tier = tier_str.parse::<KeyTier>().unwrap_or(KeyTier::Free);
                let revoked: bool = row.get(7)?;
                let grace: Option<i64> = row.get(8)?;
                Ok(ApiKey {
                    key: row.get(0)?,
                    id: row.get(1)?,
                    tier,
                    label: row.get(3)?,
                    user_id: row.get(4)?,
                    created_at: row.get(5)?,
                    last_used_at: row.get(6)?,
                    revoked,
                    grace_expires_at: grace,
                })
            })
            .optional()
            .map_err(|e| SrganError::Network(format!("SQLite query: {}", e)))?;

        // Filter out fully-revoked keys (past grace period)
        match result {
            Some(ref k) if k.revoked && k.grace_expires_at.map_or(true, |g| g <= now) => Ok(None),
            other => Ok(other),
        }
    }

    /// Record a "last used" timestamp for a key.
    pub fn touch_key(&self, key: &str) {
        let now = Utc::now().timestamp();
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "UPDATE api_keys SET last_used_at = ?1 WHERE key = ?2",
                params![now, key],
            );
        }
    }

    /// List all active (non-revoked) keys for a user.
    pub fn list_keys_for_user(&self, user_id: &str) -> Result<Vec<ApiKey>> {
        let now = Utc::now().timestamp();
        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        let mut stmt = conn
            .prepare(
                "SELECT key, id, tier, label, user_id, created_at, last_used_at, revoked, grace_expires_at
                 FROM api_keys
                 WHERE user_id = ?1
                   AND (revoked = 0 OR (grace_expires_at IS NOT NULL AND grace_expires_at > ?2))
                 ORDER BY created_at DESC",
            )
            .map_err(|e| SrganError::Network(format!("SQLite prepare: {}", e)))?;

        let keys = stmt
            .query_map(params![user_id, now], |row| {
                let tier_str: String = row.get(2)?;
                let tier = tier_str.parse::<KeyTier>().unwrap_or(KeyTier::Free);
                Ok(ApiKey {
                    key: row.get(0)?,
                    id: row.get(1)?,
                    tier,
                    label: row.get(3)?,
                    user_id: row.get(4)?,
                    created_at: row.get(5)?,
                    last_used_at: row.get(6)?,
                    revoked: row.get(7)?,
                    grace_expires_at: row.get(8)?,
                })
            })
            .map_err(|e| SrganError::Network(format!("SQLite query: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(keys)
    }

    /// Revoke a key by its short id, scoped to a user. Returns true if found.
    pub fn revoke_key(&self, key_id: &str, user_id: &str) -> Result<bool> {
        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        let rows = conn
            .execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ?1 AND user_id = ?2",
                params![key_id, user_id],
            )
            .map_err(|e| sqlite_err(format!("SQLite revoke: {}", e)))?;
        Ok(rows > 0)
    }

    /// Rotate a key: mark old key revoked with 24 h grace, create new key with same label/tier.
    /// Returns (new_key, grace_expires_at).
    pub fn rotate_key(&self, key_id: &str, user_id: &str) -> Result<Option<(ApiKey, i64)>> {
        let grace_until = Utc::now().timestamp() + 86400; // 24 h

        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;

        // Find the old key
        let old: Option<(String, String, Option<String>)> = conn
            .query_row(
                "SELECT key, tier, label FROM api_keys WHERE id = ?1 AND user_id = ?2 AND revoked = 0",
                params![key_id, user_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .map_err(|e| sqlite_err(format!("SQLite rotate lookup: {}", e)))?;

        let (old_key, tier_str, label) = match old {
            Some(o) => o,
            None => return Ok(None),
        };

        // Mark old key as revoked with grace period
        conn.execute(
            "UPDATE api_keys SET revoked = 1, grace_expires_at = ?1 WHERE key = ?2",
            params![grace_until, old_key],
        )
        .map_err(|e| sqlite_err(format!("SQLite rotate revoke: {}", e)))?;

        // Create new key
        let tier = tier_str.parse::<KeyTier>().unwrap_or(KeyTier::Free);
        let new_key_str = format!("sk_{}", Uuid::new_v4().to_string().replace('-', ""));
        let new_id = format!("key_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
        let now = Utc::now().timestamp();

        conn.execute(
            "INSERT INTO api_keys (key, id, tier, label, user_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![new_key_str, new_id, tier.as_str(), label, user_id, now],
        )
        .map_err(|e| sqlite_err(format!("SQLite rotate insert: {}", e)))?;

        let new_key = ApiKey {
            key: new_key_str,
            id: new_id,
            tier,
            created_at: now,
            label,
            user_id: Some(user_id.to_string()),
            last_used_at: 0,
            revoked: false,
            grace_expires_at: None,
        };

        Ok(Some((new_key, grace_until)))
    }

    /// Increment usage counter for today and return the new count.
    pub fn record_usage(&self, key: &str) -> Result<u64> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;

        conn.execute(
            "INSERT INTO key_usage (key, date_utc, count) VALUES (?1, ?2, 1)
             ON CONFLICT(key, date_utc) DO UPDATE SET count = count + 1",
            params![key, today],
        )
        .map_err(|e| sqlite_err(format!("SQLite usage: {}", e)))?;

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
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return 0,
        };
        conn.query_row(
            "SELECT count FROM key_usage WHERE key = ?1 AND date_utc = ?2",
            params![key, today],
            |row| row.get(0),
        )
        .unwrap_or(0)
    }

    /// All registered keys (for admin endpoint).
    pub fn all_keys(&self) -> Result<Vec<ApiKey>> {
        let conn = self.conn.lock()
            .map_err(|_| SrganError::Network("auth store mutex poisoned".into()))?;
        let mut stmt = conn
            .prepare(
                "SELECT key, id, tier, label, user_id, created_at, last_used_at, revoked, grace_expires_at
                 FROM api_keys ORDER BY created_at DESC",
            )
            .map_err(|e| SrganError::Network(format!("SQLite prepare: {}", e)))?;

        let keys = stmt
            .query_map([], |row| {
                let tier_str: String = row.get(2)?;
                let tier = tier_str.parse::<KeyTier>().unwrap_or(KeyTier::Free);
                Ok(ApiKey {
                    key: row.get(0)?,
                    id: row.get(1)?,
                    tier,
                    label: row.get(3)?,
                    user_id: row.get(4)?,
                    created_at: row.get(5)?,
                    last_used_at: row.get(6)?,
                    revoked: row.get(7)?,
                    grace_expires_at: row.get(8)?,
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
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return std::collections::HashMap::new(),
        };
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
        assert!(key.key.starts_with("sk_"));

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

    #[test]
    fn test_user_registration_and_auth() {
        let store = KeyStore::open_in_memory().unwrap();
        let uid = store.register_user("test@example.com", "secret123").unwrap();
        assert!(uid.starts_with("usr_"));

        let (uid2, email) = store.authenticate_user("test@example.com", "secret123").unwrap();
        assert_eq!(uid, uid2);
        assert_eq!(email, "test@example.com");

        // Wrong password
        assert!(store.authenticate_user("test@example.com", "wrong").is_err());
        // Duplicate email
        assert!(store.register_user("test@example.com", "other").is_err());
    }

    #[test]
    fn test_key_lifecycle() {
        let store = KeyStore::open_in_memory().unwrap();
        let uid = store.register_user("user@test.com", "pass").unwrap();

        // Create key for user
        let key = store
            .create_key_for_user(KeyTier::Free, Some("my key".into()), Some(&uid))
            .unwrap();

        // List
        let keys = store.list_keys_for_user(&uid).unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].id, key.id);

        // Touch
        store.touch_key(&key.key);
        let refreshed = store.get_key(&key.key).unwrap().unwrap();
        assert!(refreshed.last_used_at > 0);

        // Revoke
        assert!(store.revoke_key(&key.id, &uid).unwrap());
        let keys = store.list_keys_for_user(&uid).unwrap();
        assert_eq!(keys.len(), 0);
    }

    #[test]
    fn test_key_rotation() {
        let store = KeyStore::open_in_memory().unwrap();
        let uid = store.register_user("rot@test.com", "pass").unwrap();
        let key = store
            .create_key_for_user(KeyTier::Pro, Some("rotate me".into()), Some(&uid))
            .unwrap();

        let (new_key, grace_until) = store.rotate_key(&key.id, &uid).unwrap().unwrap();
        assert_ne!(new_key.key, key.key);
        assert!(grace_until > Utc::now().timestamp());

        // Old key still resolves during grace period
        let old = store.get_key(&key.key).unwrap();
        assert!(old.is_some());

        // New key works
        let new = store.get_key(&new_key.key).unwrap();
        assert!(new.is_some());
    }

    #[test]
    fn test_jwt_roundtrip() {
        let token = issue_jwt("usr_123", "a@b.com").unwrap();
        let claims = verify_jwt(&token).unwrap();
        assert_eq!(claims.sub, "usr_123");
        assert_eq!(claims.email, "a@b.com");
    }
}
