//! Per-API-key rate-limit dashboard — daily/monthly usage stats, throttle counts,
//! and admin endpoints for viewing and updating rate limits.
//!
//! ## Endpoints
//!
//! - `GET  /admin/rate-limits`      — JSON of all API keys with usage stats
//! - `PUT  /admin/rate-limits/:id`  — update rate limits for a specific key
//!
//! ## Token Bucket Rate Limiting
//!
//! Each API key has a configurable RPM (requests per minute) and daily limit.
//! The middleware enforces these using a token bucket algorithm.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::api::billing::BillingDb;
use crate::billing::plans;

// ── Rate limit stats ─────────────────────────────────────────────────────────

/// Aggregated rate-limit statistics for a single API key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStats {
    pub api_key: String,
    pub tier: String,
    pub requests_today: u32,
    pub requests_this_month: u32,
    pub quota_daily: u32,
    pub rate_limit_rpm: u32,
    pub rate_limit_daily: u32,
    pub pct_daily: f64,
    pub throttled_today: u32,
    pub overage_count: u32,
    pub last_seen: u64,
}

/// Per-day usage bucket for a single API key.
#[derive(Debug, Clone, Default)]
pub struct DayBucket {
    pub count: u32,
    pub throttled: u32,
    pub day: u64,
    pub month_count: u32,
    pub month: u32,
    pub last_seen: u64,
    pub overage_count: u32,
}

/// Per-key rate limit overrides (from admin PUT endpoint).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitOverride {
    pub rate_limit_rpm: Option<u32>,
    pub rate_limit_daily: Option<u32>,
}

/// Request body for PUT /admin/rate-limits/:key_id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRateLimitRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit_rpm: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit_daily: Option<u32>,
}

// ── Token bucket ─────────────────────────────────────────────────────────────

/// Token bucket for per-key RPM enforcement.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum tokens (= RPM limit).
    pub capacity: u32,
    /// Current tokens available.
    pub tokens: f64,
    /// Last refill timestamp (fractional seconds).
    pub last_refill: f64,
    /// Refill rate: tokens per second (capacity / 60).
    pub rate: f64,
}

impl TokenBucket {
    pub fn new(rpm: u32) -> Self {
        Self {
            capacity: rpm,
            tokens: rpm as f64,
            last_refill: now_secs_f64(),
            rate: rpm as f64 / 60.0,
        }
    }

    /// Try to consume one token. Returns true if allowed.
    pub fn try_consume(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = now_secs_f64();
        let elapsed = now - self.last_refill;
        if elapsed > 0.0 {
            self.tokens = (self.tokens + elapsed * self.rate).min(self.capacity as f64);
            self.last_refill = now;
        }
    }
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ── Dashboard ────────────────────────────────────────────────────────────────

/// In-memory dashboard tracking request counts per API key per day,
/// with token bucket RPM enforcement and admin-configurable overrides.
pub struct RateLimitDashboard {
    pub buckets: HashMap<String, DayBucket>,
    /// Per-key token buckets for RPM enforcement.
    pub token_buckets: HashMap<String, TokenBucket>,
    /// Admin-configured rate limit overrides per key.
    pub overrides: HashMap<String, RateLimitOverride>,
}

impl RateLimitDashboard {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            token_buckets: HashMap::new(),
            overrides: HashMap::new(),
        }
    }

    /// Check RPM rate limit for an API key using token bucket.
    /// Returns true if the request is allowed.
    pub fn check_rpm(&mut self, api_key: &str, default_rpm: u32) -> bool {
        let rpm = self
            .overrides
            .get(api_key)
            .and_then(|o| o.rate_limit_rpm)
            .unwrap_or(default_rpm);

        let bucket = self
            .token_buckets
            .entry(api_key.to_string())
            .or_insert_with(|| TokenBucket::new(rpm));

        // Update capacity if override changed
        if bucket.capacity != rpm {
            bucket.capacity = rpm;
            bucket.rate = rpm as f64 / 60.0;
        }

        bucket.try_consume()
    }

    /// Check daily rate limit for an API key.
    /// Returns true if within daily limit.
    pub fn check_daily(&self, api_key: &str, default_daily: u32) -> bool {
        let daily_limit = self
            .overrides
            .get(api_key)
            .and_then(|o| o.rate_limit_daily)
            .unwrap_or(default_daily);

        if daily_limit == 0 || daily_limit == u32::MAX {
            return true; // unlimited
        }

        let today = today_epoch();
        self.buckets
            .get(api_key)
            .map(|b| {
                if b.day == today {
                    b.count < daily_limit
                } else {
                    true // new day, reset
                }
            })
            .unwrap_or(true)
    }

    /// Record a request (and whether it was throttled) for an API key.
    pub fn record_request(&mut self, api_key: &str, throttled: bool) {
        let today = today_epoch();
        let current_month = current_month();
        let now = unix_now();

        let bucket = self
            .buckets
            .entry(api_key.to_string())
            .or_insert_with(|| DayBucket {
                count: 0,
                throttled: 0,
                day: today,
                month_count: 0,
                month: current_month,
                last_seen: now,
                overage_count: 0,
            });

        // Reset if the day has rolled over.
        if bucket.day != today {
            bucket.count = 0;
            bucket.throttled = 0;
            bucket.day = today;
            bucket.overage_count = 0;
        }

        // Reset monthly if month rolled over.
        if bucket.month != current_month {
            bucket.month_count = 0;
            bucket.month = current_month;
        }

        bucket.count += 1;
        bucket.month_count += 1;
        bucket.last_seen = now;

        if throttled {
            bucket.throttled += 1;
            bucket.overage_count += 1;
        }
    }

    /// Update rate limit overrides for a specific API key.
    pub fn update_rate_limits(&mut self, api_key: &str, update: &UpdateRateLimitRequest) {
        let entry = self
            .overrides
            .entry(api_key.to_string())
            .or_insert_with(|| RateLimitOverride {
                rate_limit_rpm: None,
                rate_limit_daily: None,
            });

        if let Some(rpm) = update.rate_limit_rpm {
            entry.rate_limit_rpm = Some(rpm);
        }
        if let Some(daily) = update.rate_limit_daily {
            entry.rate_limit_daily = Some(daily);
        }
    }

    /// Get stats for a single API key, enriched with billing tier info.
    pub fn get_stats(&self, api_key: &str, billing: &BillingDb) -> Option<RateLimitStats> {
        let bucket = self.buckets.get(api_key)?;
        let user = billing.get_user(api_key);
        let tier_name = user.map(|u| u.tier.as_str()).unwrap_or("free");
        let plan = plans::plan_for(tier_name);
        let quota = plan.daily_quota;
        let pct = if quota == u32::MAX || quota == 0 {
            0.0
        } else {
            (bucket.count as f64 / quota as f64) * 100.0
        };

        let overrides = self.overrides.get(api_key);
        let rpm = overrides
            .and_then(|o| o.rate_limit_rpm)
            .unwrap_or(60);
        let daily = overrides
            .and_then(|o| o.rate_limit_daily)
            .unwrap_or(quota);

        Some(RateLimitStats {
            api_key: api_key.to_string(),
            tier: tier_name.to_string(),
            requests_today: bucket.count,
            requests_this_month: bucket.month_count,
            quota_daily: quota,
            rate_limit_rpm: rpm,
            rate_limit_daily: daily,
            pct_daily: pct,
            throttled_today: bucket.throttled,
            overage_count: bucket.overage_count,
            last_seen: bucket.last_seen,
        })
    }

    /// Get stats for all tracked API keys.
    pub fn get_all_stats(&self, billing: &BillingDb) -> Vec<RateLimitStats> {
        self.buckets
            .keys()
            .filter_map(|key| self.get_stats(key, billing))
            .collect()
    }
}

// ── SQLite migration for rate limit columns ──────────────────────────────────

/// Run the SQLite migration to add rate limit columns to the api_keys table.
/// Safe to call multiple times — uses IF NOT EXISTS semantics.
pub fn migrate_rate_limit_columns(conn: &rusqlite::Connection) -> Result<(), String> {
    // SQLite doesn't have IF NOT EXISTS for ALTER TABLE ADD COLUMN,
    // so we check the table info first.
    let columns: Vec<String> = conn
        .prepare("PRAGMA table_info(api_keys)")
        .map_err(|e| format!("PRAGMA failed: {}", e))?
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| format!("query failed: {}", e))?
        .filter_map(|r| r.ok())
        .collect();

    let migrations = vec![
        ("rate_limit_rpm", "ALTER TABLE api_keys ADD COLUMN rate_limit_rpm INTEGER DEFAULT 60"),
        ("rate_limit_daily", "ALTER TABLE api_keys ADD COLUMN rate_limit_daily INTEGER DEFAULT 0"),
        ("requests_today", "ALTER TABLE api_keys ADD COLUMN requests_today INTEGER DEFAULT 0"),
        ("requests_this_month", "ALTER TABLE api_keys ADD COLUMN requests_this_month INTEGER DEFAULT 0"),
        ("overage_count", "ALTER TABLE api_keys ADD COLUMN overage_count INTEGER DEFAULT 0"),
        ("last_seen_at", "ALTER TABLE api_keys ADD COLUMN last_seen_at INTEGER DEFAULT 0"),
        ("account_suspended", "ALTER TABLE api_keys ADD COLUMN account_suspended INTEGER DEFAULT 0"),
    ];

    for (col_name, sql) in migrations {
        if !columns.contains(&col_name.to_string()) {
            conn.execute(sql, [])
                .map_err(|e| format!("Migration '{}' failed: {}", col_name, e))?;
        }
    }

    Ok(())
}

// ── Admin HTML tab content ───────────────────────────────────────────────────

/// Returns the inline JS/HTML for the "Rate Limits" tab in the admin dashboard.
/// This is injected into the admin panel SPA.
pub fn rate_limits_tab_html() -> &'static str {
    r##"
<div id="rateLimitsTab" style="display:none">
  <div class="sec-ttl">Per-API-Key Rate Limits</div>
  <div style="margin-bottom:.8rem">
    <input type="text" id="rlSearch" placeholder="Filter by key ID or tier..."
      style="background:var(--bg);border:1px solid var(--border);border-radius:var(--r);
      padding:.4rem .65rem;color:var(--text);font-family:var(--font);font-size:.82rem;width:260px">
  </div>
  <div class="tbl-wrap" id="rateLimitsTbl">
    <div style="padding:1.5rem;color:var(--muted);text-align:center">Loading...</div>
  </div>
</div>
<script>
function renderRateLimitsTab(keys){
  if(!keys||!keys.length){
    document.getElementById('rateLimitsTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No API keys with usage data</div>';
    return;
  }
  var html='<table><thead><tr><th>Key ID</th><th>Tier</th><th>Today</th><th>Month</th>'
    +'<th>RPM Limit</th><th>Daily Limit</th><th>Overages</th><th>Last Seen</th><th>Usage</th><th>Actions</th></tr></thead><tbody>';
  keys.forEach(function(k){
    var pct=k.pct_daily||0;
    var barColor=pct>90?'var(--red)':pct>70?'var(--yellow)':'var(--green)';
    var lastSeen=k.last_seen?new Date(k.last_seen*1000).toLocaleString():'Never';
    var keyDisp=k.api_key.length>16?k.api_key.slice(0,12)+'...':k.api_key;
    html+='<tr>'
      +'<td><code>'+keyDisp+'</code></td>'
      +'<td><span class="badge tier-'+k.tier+'">'+k.tier+'</span></td>'
      +'<td>'+k.requests_today+'</td>'
      +'<td>'+k.requests_this_month+'</td>'
      +'<td>'+k.rate_limit_rpm+'/min</td>'
      +'<td>'+k.rate_limit_daily+'/day</td>'
      +'<td style="color:'+(k.overage_count>0?'var(--red)':'var(--muted)')+'">'+k.overage_count+'</td>'
      +'<td style="font-size:.75rem">'+lastSeen+'</td>'
      +'<td><div style="background:var(--bg3);border-radius:3px;height:14px;width:80px;overflow:hidden">'
      +'<div style="background:'+barColor+';height:100%;width:'+Math.min(pct,100)+'%"></div></div>'
      +'<span style="font-size:.7rem;color:var(--muted)">'+pct.toFixed(1)+'%</span></td>'
      +'<td><button class="btn" style="width:auto;padding:.15rem .4rem;font-size:.72rem;'
      +'background:var(--bg3);color:var(--accent);border:1px solid var(--border)"'
      +' onclick="editRateLimit(\''+k.api_key+'\','+k.rate_limit_rpm+','+k.rate_limit_daily+')">Edit</button></td>'
      +'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('rateLimitsTbl').innerHTML=html;
}

function editRateLimit(keyId,currentRpm,currentDaily){
  var rpm=prompt('RPM limit for '+keyId.slice(0,12)+'...?',currentRpm);
  if(rpm===null)return;
  var daily=prompt('Daily limit?',currentDaily);
  if(daily===null)return;
  fetch('/admin/rate-limits/'+encodeURIComponent(keyId),{
    method:'PUT',
    headers:{Authorization:'Bearer '+tok,'Content-Type':'application/json'},
    body:JSON.stringify({rate_limit_rpm:parseInt(rpm)||60,rate_limit_daily:parseInt(daily)||0})
  }).then(function(r){return r.json();}).then(function(d){
    if(d.ok)refresh();
    else alert('Error: '+(d.error||'unknown'));
  }).catch(function(e){alert('Error: '+e.message);});
}

document.getElementById('rlSearch').addEventListener('input',function(e){
  var q=e.target.value.toLowerCase();
  var rows=document.querySelectorAll('#rateLimitsTbl tbody tr');
  rows.forEach(function(row){row.style.display=row.textContent.toLowerCase().includes(q)?'':'none';});
});
</script>
"##
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Start-of-day as a Unix timestamp (UTC midnight).
fn today_epoch() -> u64 {
    let now = unix_now();
    now - (now % 86400)
}

/// Current month as YYYYMM integer.
fn current_month() -> u32 {
    let now = unix_now();
    // Approximate: days since epoch / 30.44 gives rough month index
    // For exact: use chrono. But we just need a month boundary for resets.
    let days = now / 86400;
    let years = days / 365;
    let remaining_days = days % 365;
    let month = remaining_days / 30;
    ((1970 + years) * 100 + month + 1) as u32
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_allows_within_limit() {
        let mut bucket = TokenBucket::new(10);
        for _ in 0..10 {
            assert!(bucket.try_consume());
        }
    }

    #[test]
    fn test_token_bucket_denies_over_limit() {
        let mut bucket = TokenBucket::new(2);
        assert!(bucket.try_consume());
        assert!(bucket.try_consume());
        // Third request should be denied (no time to refill)
        assert!(!bucket.try_consume());
    }

    #[test]
    fn test_dashboard_record_request() {
        let mut dash = RateLimitDashboard::new();
        dash.record_request("key_abc", false);
        dash.record_request("key_abc", true);
        dash.record_request("key_abc", false);

        let bucket = dash.buckets.get("key_abc").unwrap();
        assert_eq!(bucket.count, 3);
        assert_eq!(bucket.throttled, 1);
        assert_eq!(bucket.overage_count, 1);
    }

    #[test]
    fn test_dashboard_update_overrides() {
        let mut dash = RateLimitDashboard::new();
        dash.update_rate_limits("key_123", &UpdateRateLimitRequest {
            rate_limit_rpm: Some(120),
            rate_limit_daily: Some(5000),
        });

        let o = dash.overrides.get("key_123").unwrap();
        assert_eq!(o.rate_limit_rpm, Some(120));
        assert_eq!(o.rate_limit_daily, Some(5000));
    }

    #[test]
    fn test_check_rpm_with_override() {
        let mut dash = RateLimitDashboard::new();
        dash.update_rate_limits("key_fast", &UpdateRateLimitRequest {
            rate_limit_rpm: Some(1000),
            rate_limit_daily: None,
        });
        // Should allow many requests with high RPM
        for _ in 0..100 {
            assert!(dash.check_rpm("key_fast", 60));
        }
    }

    #[test]
    fn test_rate_limits_tab_html_not_empty() {
        let html = rate_limits_tab_html();
        assert!(html.contains("rateLimitsTab"));
        assert!(html.contains("editRateLimit"));
    }

    #[test]
    fn test_migrate_columns_sql() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE api_keys (
                key TEXT PRIMARY KEY,
                id TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'free'
            )"
        ).unwrap();

        let result = migrate_rate_limit_columns(&conn);
        assert!(result.is_ok());

        // Verify columns were added
        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(api_keys)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert!(columns.contains(&"rate_limit_rpm".to_string()));
        assert!(columns.contains(&"rate_limit_daily".to_string()));
        assert!(columns.contains(&"account_suspended".to_string()));
    }
}
