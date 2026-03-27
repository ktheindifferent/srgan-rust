//! Batch job queue dashboard.
//!
//! Serves a real-time HTML dashboard at `/dashboard/queue` showing active,
//! queued, completed, and failed jobs with per-model stats.  Uses Server-Sent
//! Events (SSE) at `/api/v1/queue/events` for live updates.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

// ── Snapshot types ──────────────────────────────────────────────────────────

/// Per-model aggregate statistics.
#[derive(Debug, Clone, Serialize)]
pub struct ModelStats {
    pub model: String,
    pub jobs_completed: u64,
    pub total_processing_ms: u64,
    pub avg_processing_ms: u64,
}

/// Point-in-time snapshot of the queue for the dashboard.
#[derive(Debug, Clone, Serialize)]
pub struct QueueSnapshot {
    pub active_jobs: u64,
    pub queued_jobs: u64,
    pub completed_24h: u64,
    pub failed_24h: u64,
    pub throughput_per_hour: f64,
    pub model_stats: Vec<ModelStats>,
    pub timestamp: u64,
}

// ── Tracker ─────────────────────────────────────────────────────────────────

/// Lightweight record kept for dashboard aggregation.
#[derive(Debug, Clone)]
struct FinishedJob {
    model: String,
    processing_ms: u64,
    finished_at: u64,
    success: bool,
}

/// Thread-safe tracker fed by the job queue.
pub struct QueueTracker {
    active: Mutex<u64>,
    queued: Mutex<u64>,
    finished: Mutex<Vec<FinishedJob>>,
}

impl QueueTracker {
    pub fn new() -> Self {
        Self {
            active: Mutex::new(0),
            queued: Mutex::new(0),
            finished: Mutex::new(Vec::new()),
        }
    }

    pub fn job_enqueued(&self) {
        if let Ok(mut q) = self.queued.lock() {
            *q += 1;
        }
    }

    pub fn job_started(&self) {
        if let Ok(mut q) = self.queued.lock() {
            *q = q.saturating_sub(1);
        }
        if let Ok(mut a) = self.active.lock() {
            *a += 1;
        }
    }

    pub fn job_finished(&self, model: &str, processing_ms: u64, success: bool) {
        if let Ok(mut a) = self.active.lock() {
            *a = a.saturating_sub(1);
        }
        if let Ok(mut f) = self.finished.lock() {
            f.push(FinishedJob {
                model: model.to_string(),
                processing_ms,
                finished_at: unix_now(),
                success,
            });
        }
    }

    /// Build a snapshot for the dashboard, pruning jobs older than 24 h.
    pub fn snapshot(&self) -> QueueSnapshot {
        let now = unix_now();
        let cutoff = now.saturating_sub(86_400);

        let active = self.active.lock().map(|a| *a).unwrap_or(0);
        let queued = self.queued.lock().map(|q| *q).unwrap_or(0);

        let mut completed_24h: u64 = 0;
        let mut failed_24h: u64 = 0;
        let mut model_map: HashMap<String, (u64, u64)> = HashMap::new();

        if let Ok(mut fin) = self.finished.lock() {
            // Prune older than 24 h
            fin.retain(|j| j.finished_at >= cutoff);

            for j in fin.iter() {
                if j.success {
                    completed_24h += 1;
                } else {
                    failed_24h += 1;
                }
                let entry = model_map.entry(j.model.clone()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += j.processing_ms;
            }
        }

        let hours_window = 24.0_f64;
        let throughput_per_hour = completed_24h as f64 / hours_window;

        let model_stats: Vec<ModelStats> = model_map
            .into_iter()
            .map(|(model, (count, total_ms))| ModelStats {
                model,
                avg_processing_ms: if count > 0 { total_ms / count } else { 0 },
                jobs_completed: count,
                total_processing_ms: total_ms,
            })
            .collect();

        QueueSnapshot {
            active_jobs: active,
            queued_jobs: queued,
            completed_24h,
            failed_24h,
            throughput_per_hour,
            model_stats,
            timestamp: now,
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── SSE formatting ──────────────────────────────────────────────────────────

/// Format a `QueueSnapshot` as an SSE `data:` frame.
pub fn snapshot_to_sse(snap: &QueueSnapshot) -> String {
    let json = serde_json::to_string(snap).unwrap_or_default();
    format!("event: queue\ndata: {}\n\n", json)
}

// ── HTML page ───────────────────────────────────────────────────────────────

/// Return the full HTML for the queue dashboard page.
pub fn render_queue_dashboard() -> String {
    QUEUE_DASHBOARD_HTML.to_string()
}

const QUEUE_DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Job Queue Dashboard — SRGAN Rust</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,monospace;background:#0d1117;color:#c9d1d9;min-height:100vh}
header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:20px;font-weight:600;color:#58a6ff}
.status-dot{width:10px;height:10px;border-radius:50%;background:#3fb950;display:inline-block;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
main{max-width:1100px;margin:0 auto;padding:24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:32px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;text-align:center}
.card .value{font-size:36px;font-weight:700;color:#58a6ff;margin:8px 0 4px}
.card .label{font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:1px}
.card.active .value{color:#3fb950}
.card.queued .value{color:#d29922}
.card.failed .value{color:#f85149}
.card.throughput .value{font-size:28px;color:#bc8cff}
h2{font-size:16px;color:#8b949e;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;margin-bottom:32px}
th,td{padding:12px 16px;text-align:left;border-bottom:1px solid #21262d}
th{background:#21262d;color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:1px}
td{font-size:14px}
.updated{color:#484f58;font-size:12px;text-align:center;margin-top:16px}
</style>
</head>
<body>
<header>
  <span class="status-dot" id="statusDot"></span>
  <h1>Job Queue Dashboard</h1>
</header>
<main>
  <div class="cards">
    <div class="card active"><div class="label">Active</div><div class="value" id="active">-</div></div>
    <div class="card queued"><div class="label">Queued</div><div class="value" id="queued">-</div></div>
    <div class="card"><div class="label">Completed (24h)</div><div class="value" id="completed">-</div></div>
    <div class="card failed"><div class="label">Failed (24h)</div><div class="value" id="failed">-</div></div>
    <div class="card throughput"><div class="label">Images / hour</div><div class="value" id="throughput">-</div></div>
  </div>

  <h2>Per-Model Stats</h2>
  <table>
    <thead><tr><th>Model</th><th>Jobs</th><th>Avg Time</th><th>Total Time</th></tr></thead>
    <tbody id="modelBody"><tr><td colspan="4" style="color:#484f58">Waiting for data…</td></tr></tbody>
  </table>

  <p class="updated" id="updated"></p>
</main>
<script>
(function(){
  var src = new EventSource('/api/v1/queue/events');
  src.addEventListener('queue', function(e){
    var d = JSON.parse(e.data);
    document.getElementById('active').textContent = d.active_jobs;
    document.getElementById('queued').textContent = d.queued_jobs;
    document.getElementById('completed').textContent = d.completed_24h;
    document.getElementById('failed').textContent = d.failed_24h;
    document.getElementById('throughput').textContent = d.throughput_per_hour.toFixed(1);
    var tb = document.getElementById('modelBody');
    if(d.model_stats.length === 0){
      tb.innerHTML = '<tr><td colspan="4" style="color:#484f58">No jobs in last 24 h</td></tr>';
    } else {
      tb.innerHTML = d.model_stats.map(function(m){
        return '<tr><td>'+m.model+'</td><td>'+m.jobs_completed+'</td><td>'+m.avg_processing_ms+' ms</td><td>'+(m.total_processing_ms/1000).toFixed(1)+' s</td></tr>';
      }).join('');
    }
    document.getElementById('updated').textContent = 'Updated ' + new Date(d.timestamp * 1000).toLocaleTimeString();
  });
  src.onerror = function(){
    document.getElementById('statusDot').style.background = '#f85149';
  };
})();
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_lifecycle() {
        let t = QueueTracker::new();
        t.job_enqueued();
        t.job_enqueued();
        t.job_started();
        t.job_finished("anime", 1200, true);
        t.job_started();
        t.job_finished("natural", 800, false);

        let snap = t.snapshot();
        assert_eq!(snap.active_jobs, 0);
        assert_eq!(snap.queued_jobs, 0);
        assert_eq!(snap.completed_24h, 1);
        assert_eq!(snap.failed_24h, 1);
        assert_eq!(snap.model_stats.len(), 2);
    }

    #[test]
    fn test_snapshot_to_sse() {
        let t = QueueTracker::new();
        let snap = t.snapshot();
        let sse = snapshot_to_sse(&snap);
        assert!(sse.starts_with("event: queue\ndata: "));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_render_html() {
        let html = render_queue_dashboard();
        assert!(html.contains("Job Queue Dashboard"));
        assert!(html.contains("EventSource"));
    }
}
