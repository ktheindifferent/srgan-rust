//! Admin observability dashboard served at `/admin/observability`.
//!
//! Shows: API latency histogram, model inference times, cache hit ratio,
//! webhook delivery success rate, and top-10 org quota usage.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;

// ── Latency ring buffer ──────────────────────────────────────────────────────

/// Fixed-size ring buffer that stores the last N request latencies.
pub struct LatencyRing {
    buf: Vec<u64>,
    pos: usize,
    len: usize,
    cap: usize,
}

impl LatencyRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0; capacity],
            pos: 0,
            len: 0,
            cap: capacity,
        }
    }

    pub fn push(&mut self, latency_us: u64) {
        self.buf[self.pos] = latency_us;
        self.pos = (self.pos + 1) % self.cap;
        if self.len < self.cap {
            self.len += 1;
        }
    }

    /// Return histogram buckets (in ms): <1, <5, <10, <50, <100, <500, <1000, 1000+.
    pub fn histogram(&self) -> Vec<HistogramBucket> {
        let thresholds_ms: &[u64] = &[1, 5, 10, 50, 100, 500, 1000];
        let mut counts = vec![0u64; thresholds_ms.len() + 1];

        for i in 0..self.len {
            let ms = self.buf[i] / 1000; // us → ms
            let mut placed = false;
            for (j, &t) in thresholds_ms.iter().enumerate() {
                if ms < t {
                    counts[j] += 1;
                    placed = true;
                    break;
                }
            }
            if !placed {
                *counts.last_mut().unwrap() += 1;
            }
        }

        let labels = ["<1ms", "<5ms", "<10ms", "<50ms", "<100ms", "<500ms", "<1000ms", "1000ms+"];
        labels
            .iter()
            .zip(counts.iter())
            .map(|(label, &count)| HistogramBucket {
                label: label.to_string(),
                count,
            })
            .collect()
    }

    /// Percentile (0–100) in microseconds.
    pub fn percentile(&self, p: f64) -> u64 {
        if self.len == 0 {
            return 0;
        }
        let mut sorted: Vec<u64> = self.buf[..self.len].to_vec();
        sorted.sort_unstable();
        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HistogramBucket {
    pub label: String,
    pub count: u64,
}

// ── Cache stats tracker ──────────────────────────────────────────────────────

/// Atomic cache hit/miss counters.
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

impl CacheStats {
    pub fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn hit_ratio(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        let total = h + m;
        if total == 0 {
            return 0.0;
        }
        h as f64 / total as f64
    }

    pub fn snapshot(&self) -> CacheStatsSnapshot {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        CacheStatsSnapshot {
            hits: h,
            misses: m,
            hit_ratio: if h + m > 0 { h as f64 / (h + m) as f64 } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub hit_ratio: f64,
}

// ── Model inference tracker ──────────────────────────────────────────────────

/// Per-model inference time tracker.
pub struct InferenceTracker {
    models: Mutex<HashMap<String, Vec<u64>>>,
}

impl InferenceTracker {
    pub fn new() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
        }
    }

    pub fn record(&self, model: &str, duration: Duration) {
        if let Ok(mut m) = self.models.lock() {
            let times = m.entry(model.to_string()).or_insert_with(Vec::new);
            times.push(duration.as_micros() as u64);
            // Keep last 500 per model
            if times.len() > 500 {
                let drain = times.len() - 500;
                times.drain(..drain);
            }
        }
    }

    pub fn snapshot(&self) -> Vec<ModelInferenceStats> {
        let m = match self.models.lock() {
            Ok(m) => m,
            Err(_) => return vec![],
        };
        m.iter()
            .map(|(name, times)| {
                let mut sorted = times.clone();
                sorted.sort_unstable();
                let count = sorted.len() as u64;
                let avg_ms = if sorted.is_empty() {
                    0.0
                } else {
                    sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1000.0
                };
                let p50 = percentile_of(&sorted, 50.0);
                let p95 = percentile_of(&sorted, 95.0);
                let p99 = percentile_of(&sorted, 99.0);
                ModelInferenceStats {
                    model: name.clone(),
                    count,
                    avg_ms,
                    p50_ms: p50 as f64 / 1000.0,
                    p95_ms: p95 as f64 / 1000.0,
                    p99_ms: p99 as f64 / 1000.0,
                }
            })
            .collect()
    }
}

fn percentile_of(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInferenceStats {
    pub model: String,
    pub count: u64,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

// ── Dashboard snapshot ───────────────────────────────────────────────────────

/// Complete observability snapshot returned as JSON and rendered in the HTML dashboard.
#[derive(Debug, Clone, Serialize)]
pub struct ObservabilitySnapshot {
    pub latency_histogram: Vec<HistogramBucket>,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub model_inference: Vec<ModelInferenceStats>,
    pub cache: CacheStatsSnapshot,
    pub webhook_success_rate: f64,
    pub webhook_total_events: u64,
    pub top_orgs: Vec<OrgQuotaRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrgQuotaRow {
    pub org_id: String,
    pub pixels_used: u64,
    pub pixels_limit: u64,
    pub usage_pct: f64,
}

/// Render the observability dashboard as an HTML page.
pub fn render_dashboard_html(snap: &ObservabilitySnapshot) -> String {
    let histogram_rows: String = snap
        .latency_histogram
        .iter()
        .map(|b| {
            format!(
                "<tr><td>{}</td><td>{}</td><td><div class=\"bar\" style=\"width:{}px\"></div></td></tr>",
                b.label,
                b.count,
                (b.count as f64).min(400.0),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let inference_rows: String = snap
        .model_inference
        .iter()
        .map(|m| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td></tr>",
                m.model, m.count, m.avg_ms, m.p50_ms, m.p95_ms, m.p99_ms
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let org_rows: String = snap
        .top_orgs
        .iter()
        .map(|o| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td></tr>",
                o.org_id, o.pixels_used, o.pixels_limit, o.usage_pct
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<!DOCTYPE html>
<html><head>
<title>SRGAN Observability</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #8b949e; margin-top: 2rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 12px; text-align: left; }}
  th {{ background: #161b22; color: #58a6ff; }}
  .bar {{ height: 16px; background: #238636; border-radius: 3px; }}
  .stat {{ display: inline-block; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem 1.5rem; margin: 0.5rem; }}
  .stat .label {{ color: #8b949e; font-size: 0.85rem; }}
  .stat .value {{ color: #58a6ff; font-size: 1.4rem; font-weight: bold; }}
</style>
</head><body>
<h1>Observability Dashboard</h1>

<div>
  <div class="stat"><div class="label">P50 latency</div><div class="value">{:.1}ms</div></div>
  <div class="stat"><div class="label">P95 latency</div><div class="value">{:.1}ms</div></div>
  <div class="stat"><div class="label">P99 latency</div><div class="value">{:.1}ms</div></div>
  <div class="stat"><div class="label">Cache hit ratio</div><div class="value">{:.1}%</div></div>
  <div class="stat"><div class="label">Webhook success</div><div class="value">{:.1}%</div></div>
  <div class="stat"><div class="label">Webhook events</div><div class="value">{}</div></div>
</div>

<h2>API Latency Histogram (last 1000 requests)</h2>
<table>
<tr><th>Bucket</th><th>Count</th><th>Distribution</th></tr>
{}
</table>

<h2>Model Inference Times</h2>
<table>
<tr><th>Model</th><th>Count</th><th>Avg (ms)</th><th>P50</th><th>P95</th><th>P99</th></tr>
{}
</table>

<h2>Org Quota Usage (Top 10)</h2>
<table>
<tr><th>Org</th><th>Pixels Used</th><th>Limit</th><th>Usage</th></tr>
{}
</table>

</body></html>"#,
        snap.latency_p50_ms,
        snap.latency_p95_ms,
        snap.latency_p99_ms,
        snap.cache.hit_ratio * 100.0,
        snap.webhook_success_rate * 100.0,
        snap.webhook_total_events,
        histogram_rows,
        inference_rows,
        org_rows,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_ring() {
        let mut ring = LatencyRing::new(10);
        for i in 0..5 {
            ring.push(i * 1000); // 0ms, 1ms, 2ms, 3ms, 4ms
        }
        assert_eq!(ring.len, 5);
        let hist = ring.histogram();
        assert!(!hist.is_empty());
    }

    #[test]
    fn test_cache_stats() {
        let cs = CacheStats::new();
        cs.record_hit();
        cs.record_hit();
        cs.record_miss();
        assert!((cs.hit_ratio() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_inference_tracker() {
        let tracker = InferenceTracker::new();
        tracker.record("natural", Duration::from_millis(50));
        tracker.record("natural", Duration::from_millis(100));
        tracker.record("anime", Duration::from_millis(75));

        let snap = tracker.snapshot();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn test_render_dashboard_html() {
        let snap = ObservabilitySnapshot {
            latency_histogram: vec![
                HistogramBucket { label: "<1ms".into(), count: 10 },
                HistogramBucket { label: "<5ms".into(), count: 20 },
            ],
            latency_p50_ms: 2.5,
            latency_p95_ms: 15.0,
            latency_p99_ms: 45.0,
            model_inference: vec![],
            cache: CacheStatsSnapshot { hits: 100, misses: 20, hit_ratio: 0.833 },
            webhook_success_rate: 0.95,
            webhook_total_events: 500,
            top_orgs: vec![],
        };
        let html = render_dashboard_html(&snap);
        assert!(html.contains("Observability Dashboard"));
        assert!(html.contains("2.5ms"));
    }
}
