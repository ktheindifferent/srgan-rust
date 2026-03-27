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

// ── Batch job analytics (per-job metrics) ────────────────────────────────────

/// A single batch-job analytics record with resolution, timing, and file sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub processing_time_ms: u64,
    pub model: String,
    pub scale_factor: u32,
    pub input_file_size: u64,
    pub output_file_size: u64,
    pub completed_at: u64,
}

/// Aggregate summary across all recorded batch jobs.
#[derive(Debug, Clone, Serialize)]
pub struct BatchAnalyticsSummary {
    pub total_jobs: u64,
    pub total_processing_time_ms: u64,
    pub avg_processing_time_ms: f64,
    pub total_input_bytes: u64,
    pub total_output_bytes: u64,
    pub avg_size_ratio: f64,
    pub jobs_by_model: Vec<ModelCount>,
    pub avg_scale_factor: f64,
}

/// Query parameters for `/v1/analytics/jobs`.
#[derive(Debug, Deserialize)]
pub struct JobsQuery {
    #[serde(default = "default_jobs_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
    pub model: Option<String>,
}

fn default_jobs_limit() -> u32 { 50 }

/// SQLite-backed batch-job analytics database.
pub struct BatchAnalyticsDb {
    conn: Mutex<rusqlite::Connection>,
}

impl BatchAnalyticsDb {
    pub fn open(path: &str) -> Result<Self, String> {
        let conn = if path == ":memory:" {
            rusqlite::Connection::open_in_memory()
        } else {
            rusqlite::Connection::open(path)
        }
        .map_err(|e| format!("batch analytics db open: {}", e))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS job_analytics (
                job_id            TEXT PRIMARY KEY,
                input_width       INTEGER NOT NULL,
                input_height      INTEGER NOT NULL,
                output_width      INTEGER NOT NULL,
                output_height     INTEGER NOT NULL,
                processing_time_ms INTEGER NOT NULL,
                model             TEXT NOT NULL,
                scale_factor      INTEGER NOT NULL,
                input_file_size   INTEGER NOT NULL,
                output_file_size  INTEGER NOT NULL,
                completed_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ja_model ON job_analytics(model);
            CREATE INDEX IF NOT EXISTS idx_ja_completed ON job_analytics(completed_at);",
        )
        .map_err(|e| format!("batch analytics db init: {}", e))?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn open_in_memory() -> Result<Self, String> {
        Self::open(":memory:")
    }

    /// Record a completed job.
    pub fn record(&self, rec: &JobRecord) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT OR REPLACE INTO job_analytics
             (job_id,input_width,input_height,output_width,output_height,
              processing_time_ms,model,scale_factor,input_file_size,
              output_file_size,completed_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            rusqlite::params![
                rec.job_id, rec.input_width, rec.input_height,
                rec.output_width, rec.output_height, rec.processing_time_ms,
                rec.model, rec.scale_factor, rec.input_file_size,
                rec.output_file_size, rec.completed_at,
            ],
        ).map_err(|e| format!("batch analytics insert: {}", e))?;
        Ok(())
    }

    /// Aggregate summary.
    pub fn summary(&self) -> Result<BatchAnalyticsSummary, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let (total_jobs, total_time, total_in, total_out, avg_scale): (u64, u64, u64, u64, f64) =
            conn.query_row(
                "SELECT COUNT(*), COALESCE(SUM(processing_time_ms),0),
                        COALESCE(SUM(input_file_size),0), COALESCE(SUM(output_file_size),0),
                        COALESCE(AVG(scale_factor),0.0)
                 FROM job_analytics", [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            ).map_err(|e| format!("batch analytics summary: {}", e))?;

        let avg_time = if total_jobs > 0 { total_time as f64 / total_jobs as f64 } else { 0.0 };
        let avg_ratio = if total_in > 0 { total_out as f64 / total_in as f64 } else { 0.0 };

        let mut stmt = conn.prepare(
            "SELECT model, COUNT(*) FROM job_analytics GROUP BY model ORDER BY COUNT(*) DESC",
        ).map_err(|e| format!("batch analytics models: {}", e))?;
        let models: Vec<ModelCount> = stmt.query_map([], |row| {
            Ok(ModelCount { model: row.get(0)?, count: row.get(1)? })
        }).map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect();

        Ok(BatchAnalyticsSummary {
            total_jobs, total_processing_time_ms: total_time,
            avg_processing_time_ms: avg_time, total_input_bytes: total_in,
            total_output_bytes: total_out, avg_size_ratio: avg_ratio,
            jobs_by_model: models, avg_scale_factor: avg_scale,
        })
    }

    /// List job records with pagination and optional model filter.
    pub fn list_jobs(&self, query: &JobsQuery) -> Result<Vec<JobRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let base = "SELECT job_id,input_width,input_height,output_width,output_height,\
                     processing_time_ms,model,scale_factor,input_file_size,\
                     output_file_size,completed_at FROM job_analytics";

        let records = if let Some(ref model) = query.model {
            let sql = format!("{} WHERE model=?1 ORDER BY completed_at DESC LIMIT ?2 OFFSET ?3", base);
            let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
            stmt.query_map(rusqlite::params![model, query.limit, query.offset], map_job_row)
                .map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect()
        } else {
            let sql = format!("{} ORDER BY completed_at DESC LIMIT ?1 OFFSET ?2", base);
            let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
            stmt.query_map(rusqlite::params![query.limit, query.offset], map_job_row)
                .map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect()
        };
        Ok(records)
    }
}

fn map_job_row(row: &rusqlite::Row) -> rusqlite::Result<JobRecord> {
    Ok(JobRecord {
        job_id: row.get(0)?, input_width: row.get(1)?, input_height: row.get(2)?,
        output_width: row.get(3)?, output_height: row.get(4)?,
        processing_time_ms: row.get(5)?, model: row.get(6)?, scale_factor: row.get(7)?,
        input_file_size: row.get(8)?, output_file_size: row.get(9)?, completed_at: row.get(10)?,
    })
}

/// Parse "WxH" dimension strings.
pub fn parse_dimensions(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() == 2 {
        Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
    } else {
        None
    }
}

/// Self-contained HTML analytics dashboard page.
pub fn analytics_html_page() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>SRGAN Analytics</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:20px;background:#0f172a;color:#e2e8f0}
h1{color:#38bdf8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.card{background:#1e293b;border-radius:12px;padding:20px}.card h2{margin-top:0;color:#94a3b8;font-size:14px;text-transform:uppercase}
.card .value{font-size:32px;font-weight:700;color:#f1f5f9}canvas{width:100%;height:200px;margin-top:16px}
table{width:100%;border-collapse:collapse;margin-top:16px}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #334155}
th{color:#94a3b8;font-size:12px;text-transform:uppercase}td{color:#cbd5e1;font-size:14px}#error{color:#f87171;display:none}
</style></head><body>
<h1>SRGAN Job Analytics</h1><p id="error"></p>
<div class="grid" id="cards"></div>
<div class="card" style="margin-top:16px"><h2>Jobs by Model</h2><canvas id="mc"></canvas></div>
<div class="card" style="margin-top:16px"><h2>Recent Jobs</h2>
<table><thead><tr><th>Job</th><th>Model</th><th>Input</th><th>Output</th><th>Time</th><th>Size Change</th></tr></thead>
<tbody id="jb"></tbody></table></div>
<script>
async function load(){try{
const[s,j]=await Promise.all([fetch('/v1/analytics/summary').then(r=>r.json()),fetch('/v1/analytics/jobs?limit=20').then(r=>r.json())]);
const c=document.getElementById('cards');
c.innerHTML=[['Total Jobs',s.total_jobs],['Avg Time',s.avg_processing_time_ms.toFixed(0)+' ms'],
['Input',fb(s.total_input_bytes)],['Output',fb(s.total_output_bytes)],
['Size Ratio',s.avg_size_ratio.toFixed(2)+'x'],['Scale',s.avg_scale_factor.toFixed(1)+'x']]
.map(([t,v])=>`<div class="card"><h2>${t}</h2><div class="value">${v}</div></div>`).join('');
if(s.jobs_by_model&&s.jobs_by_model.length)drawBar(document.getElementById('mc'),s.jobs_by_model);
document.getElementById('jb').innerHTML=(j.jobs||[]).map(r=>{
const ch=r.input_file_size>0?((r.output_file_size/r.input_file_size-1)*100).toFixed(1)+'%':'N/A';
return`<tr><td>${r.job_id.substring(0,8)}</td><td>${r.model}</td><td>${r.input_width}x${r.input_height}</td><td>${r.output_width}x${r.output_height}</td><td>${r.processing_time_ms}ms</td><td>${ch}</td></tr>`}).join('');
}catch(e){const el=document.getElementById('error');el.style.display='block';el.textContent='Error: '+e.message}}
function fb(b){return b>=1e9?(b/1e9).toFixed(1)+' GB':b>=1e6?(b/1e6).toFixed(1)+' MB':b>=1e3?(b/1e3).toFixed(1)+' KB':b+' B'}
function drawBar(cv,d){const x=cv.getContext('2d'),dp=devicePixelRatio||1,r=cv.getBoundingClientRect();
cv.width=r.width*dp;cv.height=r.height*dp;x.scale(dp,dp);const mx=Math.max(...d.map(i=>i.count),1),
bw=Math.min(60,(r.width-40)/d.length-10),ch=r.height-40,cl=['#38bdf8','#a78bfa','#34d399','#fb923c','#f87171'];
d.forEach((i,n)=>{const px=20+n*(bw+10),h=i.count/mx*ch,py=ch-h+10;x.fillStyle=cl[n%cl.length];
x.fillRect(px,py,bw,h);x.fillStyle='#94a3b8';x.font='11px sans-serif';x.textAlign='center';
x.fillText(i.model,px+bw/2,r.height-4);x.fillStyle='#e2e8f0';x.fillText(i.count,px+bw/2,py-4)})}
load();
</script></body></html>"##.to_string()
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

    #[test]
    fn test_batch_analytics_record_and_summary() {
        let db = BatchAnalyticsDb::open_in_memory().unwrap();
        db.record(&JobRecord {
            job_id: "j1".into(), input_width: 640, input_height: 480,
            output_width: 2560, output_height: 1920, processing_time_ms: 1200,
            model: "natural".into(), scale_factor: 4,
            input_file_size: 50_000, output_file_size: 200_000, completed_at: 1700000000,
        }).unwrap();
        db.record(&JobRecord {
            job_id: "j2".into(), input_width: 320, input_height: 240,
            output_width: 1280, output_height: 960, processing_time_ms: 800,
            model: "anime".into(), scale_factor: 4,
            input_file_size: 30_000, output_file_size: 120_000, completed_at: 1700000010,
        }).unwrap();
        let s = db.summary().unwrap();
        assert_eq!(s.total_jobs, 2);
        assert_eq!(s.total_processing_time_ms, 2000);
        assert_eq!(s.jobs_by_model.len(), 2);
    }

    #[test]
    fn test_batch_analytics_list_jobs() {
        let db = BatchAnalyticsDb::open_in_memory().unwrap();
        for i in 0..5u64 {
            db.record(&JobRecord {
                job_id: format!("j{}", i), input_width: 100, input_height: 100,
                output_width: 400, output_height: 400, processing_time_ms: 100,
                model: "natural".into(), scale_factor: 4,
                input_file_size: 1000, output_file_size: 4000, completed_at: 1700000000 + i,
            }).unwrap();
        }
        let jobs = db.list_jobs(&JobsQuery { limit: 2, offset: 0, model: None }).unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].job_id, "j4");
    }

    #[test]
    fn test_parse_dimensions() {
        assert_eq!(parse_dimensions("640x480"), Some((640, 480)));
        assert_eq!(parse_dimensions("bad"), None);
    }
}
