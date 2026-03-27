//! Usage analytics — per-API-key tracking of images upscaled, pixels processed,
//! average latency, model breakdown, and daily/weekly/monthly aggregates.
//!
//! Data is stored in SQLite for durability. Exposes:
//! - `GET /api/v1/analytics/usage?period=7d|30d|90d` — per-key time-series
//! - `GET /api/v1/admin/analytics` — aggregate across all keys (admin only)

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── Types ────────────────────────────────────────────────────────────────────

/// A single analytics event recorded after each upscale job.
#[derive(Debug, Clone)]
pub struct AnalyticsEvent {
    pub api_key: String,
    pub model: String,
    pub input_pixels: u64,
    pub output_pixels: u64,
    pub latency_ms: u64,
    pub success: bool,
    pub timestamp: u64,
}

/// Daily aggregate for one API key.
#[derive(Debug, Clone, Serialize)]
pub struct DailyAggregate {
    pub date: String, // "YYYY-MM-DD"
    pub images_upscaled: u64,
    pub images_failed: u64,
    pub total_input_pixels: u64,
    pub total_output_pixels: u64,
    pub avg_latency_ms: f64,
    pub model_breakdown: std::collections::HashMap<String, u64>,
}

/// Time-series response for `GET /api/v1/analytics/usage`.
#[derive(Debug, Serialize)]
pub struct UsageResponse {
    pub api_key: String,
    pub period: String,
    pub total_images: u64,
    pub total_pixels_processed: u64,
    pub avg_latency_ms: f64,
    pub daily: Vec<DailyAggregate>,
}

/// Admin aggregate across all keys.
#[derive(Debug, Serialize)]
pub struct AdminAnalyticsResponse {
    pub period: String,
    pub total_images: u64,
    pub total_pixels_processed: u64,
    pub avg_latency_ms: f64,
    pub unique_keys: u64,
    pub top_models: Vec<ModelCount>,
    pub daily: Vec<DailyAggregate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCount {
    pub model: String,
    pub count: u64,
}

/// Query parameters for usage endpoint.
#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    /// Period: "7d", "30d", "90d" (default "30d")
    #[serde(default = "default_period")]
    pub period: String,
}

fn default_period() -> String { "30d".to_string() }

// ── Analytics store (SQLite-backed) ──────────────────────────────────────────

/// SQLite-backed analytics store.
pub struct AnalyticsStore {
    db: Mutex<rusqlite::Connection>,
}

impl AnalyticsStore {
    /// Open (or create) the analytics database at the given path.
    pub fn open(path: &str) -> Result<Self, String> {
        let conn = rusqlite::Connection::open(path)
            .map_err(|e| format!("Failed to open analytics db: {}", e))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key TEXT NOT NULL,
                model TEXT NOT NULL,
                input_pixels INTEGER NOT NULL,
                output_pixels INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                success INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                date TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_analytics_key_date ON analytics_events(api_key, date);
            CREATE INDEX IF NOT EXISTS idx_analytics_date ON analytics_events(date);",
        )
        .map_err(|e| format!("Failed to create analytics tables: {}", e))?;

        Ok(Self { db: Mutex::new(conn) })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self, String> {
        Self::open(":memory:")
    }

    /// Record an analytics event.
    pub fn record(&self, event: &AnalyticsEvent) {
        let date = timestamp_to_date(event.timestamp);
        if let Ok(db) = self.db.lock() {
            let _ = db.execute(
                "INSERT INTO analytics_events (api_key, model, input_pixels, output_pixels, latency_ms, success, timestamp, date)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    event.api_key,
                    event.model,
                    event.input_pixels,
                    event.output_pixels,
                    event.latency_ms,
                    event.success as i32,
                    event.timestamp,
                    date,
                ],
            );
        }
    }

    /// Query usage for a specific API key over a period.
    pub fn usage_for_key(&self, api_key: &str, period: &str) -> UsageResponse {
        let days = parse_period(period);
        let cutoff_date = date_n_days_ago(days);

        let mut daily = Vec::new();
        let mut total_images: u64 = 0;
        let mut total_pixels: u64 = 0;
        let mut total_latency: u64 = 0;
        let mut latency_count: u64 = 0;

        if let Ok(db) = self.db.lock() {
            // Daily aggregates
            let mut stmt = match db.prepare(
                "SELECT date,
                        SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as ok,
                        SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as fail,
                        SUM(input_pixels) as inp,
                        SUM(output_pixels) as outp,
                        AVG(latency_ms) as avg_lat
                 FROM analytics_events
                 WHERE api_key = ?1 AND date >= ?2
                 GROUP BY date
                 ORDER BY date",
            ) {
                Ok(s) => s,
                Err(_) => {
                    return UsageResponse {
                        api_key: api_key.to_string(),
                        period: period.to_string(),
                        total_images: 0,
                        total_pixels_processed: 0,
                        avg_latency_ms: 0.0,
                        daily: vec![],
                    };
                }
            };

            let rows = stmt.query_map(rusqlite::params![api_key, cutoff_date], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, u64>(2)?,
                    row.get::<_, u64>(3)?,
                    row.get::<_, u64>(4)?,
                    row.get::<_, f64>(5)?,
                ))
            });

            if let Ok(rows) = rows {
                for row in rows.flatten() {
                    let (date, ok, fail, inp, outp, avg_lat) = row;
                    total_images += ok + fail;
                    total_pixels += outp;
                    total_latency += (avg_lat * (ok + fail) as f64) as u64;
                    latency_count += ok + fail;
                    daily.push(DailyAggregate {
                        date,
                        images_upscaled: ok,
                        images_failed: fail,
                        total_input_pixels: inp,
                        total_output_pixels: outp,
                        avg_latency_ms: avg_lat,
                        model_breakdown: std::collections::HashMap::new(),
                    });
                }
            }

            // Model breakdown per day
            if let Ok(mut stmt2) = db.prepare(
                "SELECT date, model, COUNT(*) as cnt
                 FROM analytics_events
                 WHERE api_key = ?1 AND date >= ?2 AND success = 1
                 GROUP BY date, model",
            ) {
                if let Ok(rows) = stmt2.query_map(rusqlite::params![api_key, cutoff_date], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, u64>(2)?,
                    ))
                }) {
                    for row in rows.flatten() {
                        let (date, model, cnt) = row;
                        if let Some(d) = daily.iter_mut().find(|d| d.date == date) {
                            d.model_breakdown.insert(model, cnt);
                        }
                    }
                }
            }
        }

        let avg_latency = if latency_count > 0 {
            total_latency as f64 / latency_count as f64
        } else {
            0.0
        };

        UsageResponse {
            api_key: api_key.to_string(),
            period: period.to_string(),
            total_images,
            total_pixels_processed: total_pixels,
            avg_latency_ms: avg_latency,
            daily,
        }
    }

    /// Admin: aggregate analytics across all keys.
    pub fn admin_analytics(&self, period: &str) -> AdminAnalyticsResponse {
        let days = parse_period(period);
        let cutoff_date = date_n_days_ago(days);

        let mut daily = Vec::new();
        let mut total_images: u64 = 0;
        let mut total_pixels: u64 = 0;
        let mut total_latency: u64 = 0;
        let mut latency_count: u64 = 0;
        let mut unique_keys: u64 = 0;
        let mut top_models: Vec<ModelCount> = Vec::new();

        if let Ok(db) = self.db.lock() {
            // Unique keys
            if let Ok(mut stmt) = db.prepare(
                "SELECT COUNT(DISTINCT api_key) FROM analytics_events WHERE date >= ?1",
            ) {
                unique_keys = stmt
                    .query_row(rusqlite::params![cutoff_date], |row| row.get(0))
                    .unwrap_or(0);
            }

            // Daily aggregates
            if let Ok(mut stmt) = db.prepare(
                "SELECT date,
                        SUM(CASE WHEN success=1 THEN 1 ELSE 0 END),
                        SUM(CASE WHEN success=0 THEN 1 ELSE 0 END),
                        SUM(input_pixels),
                        SUM(output_pixels),
                        AVG(latency_ms)
                 FROM analytics_events
                 WHERE date >= ?1
                 GROUP BY date
                 ORDER BY date",
            ) {
                if let Ok(rows) = stmt.query_map(rusqlite::params![cutoff_date], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, u64>(1)?,
                        row.get::<_, u64>(2)?,
                        row.get::<_, u64>(3)?,
                        row.get::<_, u64>(4)?,
                        row.get::<_, f64>(5)?,
                    ))
                }) {
                    for row in rows.flatten() {
                        let (date, ok, fail, inp, outp, avg_lat) = row;
                        total_images += ok + fail;
                        total_pixels += outp;
                        total_latency += (avg_lat * (ok + fail) as f64) as u64;
                        latency_count += ok + fail;
                        daily.push(DailyAggregate {
                            date,
                            images_upscaled: ok,
                            images_failed: fail,
                            total_input_pixels: inp,
                            total_output_pixels: outp,
                            avg_latency_ms: avg_lat,
                            model_breakdown: std::collections::HashMap::new(),
                        });
                    }
                }
            }

            // Top models
            if let Ok(mut stmt) = db.prepare(
                "SELECT model, COUNT(*) as cnt
                 FROM analytics_events
                 WHERE date >= ?1 AND success = 1
                 GROUP BY model
                 ORDER BY cnt DESC
                 LIMIT 10",
            ) {
                if let Ok(rows) = stmt.query_map(rusqlite::params![cutoff_date], |row| {
                    Ok(ModelCount {
                        model: row.get(0)?,
                        count: row.get(1)?,
                    })
                }) {
                    top_models = rows.flatten().collect();
                }
            }
        }

        let avg_latency = if latency_count > 0 {
            total_latency as f64 / latency_count as f64
        } else {
            0.0
        };

        AdminAnalyticsResponse {
            period: period.to_string(),
            total_images,
            total_pixels_processed: total_pixels,
            avg_latency_ms: avg_latency,
            unique_keys,
            top_models,
            daily,
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn parse_period(period: &str) -> u64 {
    match period {
        "7d" => 7,
        "90d" => 90,
        _ => 30, // default 30d
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn timestamp_to_date(ts: u64) -> String {
    // Simple UTC date calculation without chrono
    let days_since_epoch = ts / 86400;
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

fn date_n_days_ago(days: u64) -> String {
    let ts = unix_now().saturating_sub(days * 86400);
    timestamp_to_date(ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_query() {
        let store = AnalyticsStore::open_in_memory().unwrap();
        let now = unix_now();

        store.record(&AnalyticsEvent {
            api_key: "key-1".into(),
            model: "natural".into(),
            input_pixels: 1_000_000,
            output_pixels: 16_000_000,
            latency_ms: 250,
            success: true,
            timestamp: now,
        });

        store.record(&AnalyticsEvent {
            api_key: "key-1".into(),
            model: "anime".into(),
            input_pixels: 500_000,
            output_pixels: 8_000_000,
            latency_ms: 150,
            success: true,
            timestamp: now,
        });

        let usage = store.usage_for_key("key-1", "7d");
        assert_eq!(usage.total_images, 2);
        assert_eq!(usage.total_pixels_processed, 24_000_000);
        assert!(usage.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_admin_analytics() {
        let store = AnalyticsStore::open_in_memory().unwrap();
        let now = unix_now();

        for i in 0..5 {
            store.record(&AnalyticsEvent {
                api_key: format!("key-{}", i % 3),
                model: "natural".into(),
                input_pixels: 1_000_000,
                output_pixels: 16_000_000,
                latency_ms: 200,
                success: true,
                timestamp: now,
            });
        }

        let admin = store.admin_analytics("30d");
        assert_eq!(admin.total_images, 5);
        assert_eq!(admin.unique_keys, 3);
    }

    #[test]
    fn test_timestamp_to_date() {
        // 2024-01-01 00:00:00 UTC = 1704067200
        assert_eq!(timestamp_to_date(1704067200), "2024-01-01");
    }

    #[test]
    fn test_parse_period() {
        assert_eq!(parse_period("7d"), 7);
        assert_eq!(parse_period("30d"), 30);
        assert_eq!(parse_period("90d"), 90);
        assert_eq!(parse_period("invalid"), 30);
    }
}
