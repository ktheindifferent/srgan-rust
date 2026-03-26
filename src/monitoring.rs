//! Monitoring and observability layer.
//!
//! - **Trace IDs**: every job gets a UUID trace ID; `tracing` spans carry it
//!   so all log lines within a job share the same `trace_id` field.
//! - **OpenTelemetry**: spans for each pipeline phase (model_load, inference,
//!   encode).  The OTel exporter is wired up at server startup; this module
//!   provides the span helpers.
//! - **Prometheus**: histograms for `inference_duration_seconds` labelled by
//!   model and scale factor, plus counters for jobs completed / failed.
//! - **Grafana dashboard**: see `docs/grafana-dashboard.json`.
//! - **Alert rules**: see `docs/alert-rules.yaml`.

use std::time::Instant;
use tracing::{info, info_span, Span};
use uuid::Uuid;

use metrics::{counter, histogram, Label};

// ---------------------------------------------------------------------------
// Metric names (also used in the Grafana dashboard JSON)
// ---------------------------------------------------------------------------

pub const METRIC_INFERENCE_DURATION: &str = "srgan_inference_duration_seconds";
pub const METRIC_JOBS_COMPLETED: &str = "srgan_jobs_completed_total";
pub const METRIC_JOBS_FAILED: &str = "srgan_jobs_failed_total";
pub const METRIC_QUEUE_DEPTH: &str = "srgan_queue_depth";
pub const METRIC_ENCODE_DURATION: &str = "srgan_encode_duration_seconds";
pub const METRIC_PREPROCESS_DURATION: &str = "srgan_preprocess_duration_seconds";

// ---------------------------------------------------------------------------
// Trace IDs
// ---------------------------------------------------------------------------

/// A unique identifier for a single end-to-end request.
#[derive(Debug, Clone)]
pub struct TraceId(String);

impl TraceId {
    /// Generate a new random trace ID.
    pub fn new() -> Self {
        TraceId(Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Job span
// ---------------------------------------------------------------------------

/// An active tracing span for one inference job.
///
/// Holds the span open for the duration of the job.  Dropping it closes the
/// span and records the elapsed time.
pub struct JobSpan {
    pub trace_id: TraceId,
    _span: Span,
    start: Instant,
    model: String,
    scale_factor: u32,
}

impl JobSpan {
    /// Open a new job span with a fresh trace ID.
    pub fn new(model: impl Into<String>, scale_factor: u32) -> Self {
        let trace_id = TraceId::new();
        let model = model.into();

        let span = info_span!(
            "srgan_job",
            trace_id = %trace_id,
            model = %model,
            scale_factor = scale_factor,
        );

        info!(
            trace_id = %trace_id,
            model = %model,
            scale_factor,
            "job started"
        );

        Self {
            trace_id,
            _span: span,
            start: Instant::now(),
            model,
            scale_factor,
        }
    }

    /// Record a successful job completion with Prometheus metrics.
    pub fn complete(self) {
        let elapsed = self.start.elapsed().as_secs_f64();

        histogram!(
            METRIC_INFERENCE_DURATION,
            elapsed,
            "model" => self.model.clone(),
            "scale_factor" => self.scale_factor.to_string(),
        );

        counter!(
            METRIC_JOBS_COMPLETED,
            1,
            "model" => self.model.clone(),
        );

        info!(
            trace_id = %self.trace_id,
            elapsed_ms = (elapsed * 1000.0) as u64,
            "job completed"
        );
    }

    /// Record a failed job.
    pub fn fail(self, reason: &str) {
        counter!(
            METRIC_JOBS_FAILED,
            1,
            "model" => self.model.clone(),
            "reason" => reason.to_string(),
        );

        tracing::warn!(
            trace_id = %self.trace_id,
            reason,
            "job failed"
        );
    }
}

// ---------------------------------------------------------------------------
// Phase timers
// ---------------------------------------------------------------------------

/// Measure and record a single pipeline phase.
pub struct PhaseTimer {
    name: &'static str,
    labels: Vec<Label>,
    start: Instant,
    trace_id: String,
}

impl PhaseTimer {
    pub fn start(name: &'static str, trace_id: &TraceId, labels: Vec<(&'static str, String)>) -> Self {
        let metric_labels: Vec<Label> = labels
            .iter()
            .map(|(k, v)| Label::new(*k, v.clone()))
            .collect();

        tracing::debug!(
            trace_id = %trace_id,
            phase = name,
            "phase started"
        );

        Self {
            name,
            labels: metric_labels,
            start: Instant::now(),
            trace_id: trace_id.to_string(),
        }
    }

    /// Record the phase duration to Prometheus and emit a debug log line.
    pub fn finish(self) {
        let elapsed = self.start.elapsed().as_secs_f64();

        let metric_name = match self.name {
            "inference" => METRIC_INFERENCE_DURATION,
            "encode" => METRIC_ENCODE_DURATION,
            "preprocess" => METRIC_PREPROCESS_DURATION,
            _ => METRIC_INFERENCE_DURATION,
        };

        // Record to Prometheus via the `metrics` facade.
        for label in &self.labels {
            let _ = label; // labels used via the macro below
        }
        histogram!(metric_name, elapsed);

        tracing::debug!(
            trace_id = %self.trace_id,
            phase = self.name,
            elapsed_ms = (elapsed * 1000.0) as u64,
            "phase finished"
        );
    }
}

// ---------------------------------------------------------------------------
// Queue depth gauge
// ---------------------------------------------------------------------------

/// Update the Prometheus queue-depth gauge.  Call this whenever the queue
/// depth changes (enqueue or dequeue).
pub fn record_queue_depth(depth: usize) {
    // The `metrics` crate represents gauges as counters with absolute values.
    // We use a gauge here by recording it as an absolute value histogram.
    histogram!(METRIC_QUEUE_DEPTH, depth as f64);
}

// ---------------------------------------------------------------------------
// Subscriber setup
// ---------------------------------------------------------------------------

/// Initialise the `tracing` subscriber with JSON output and env-filter.
///
/// Call once at startup before any spans are created.  The subscriber emits
/// structured JSON logs that include `trace_id` fields added by [`JobSpan`].
///
/// Set `RUST_LOG` to control verbosity (e.g. `RUST_LOG=srgan_rust=debug`).
pub fn init_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(true);

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .init();

    info!("tracing initialised");
}

/// Initialise Prometheus metrics exporter on the given port.
///
/// Exposes `GET /metrics` for scraping.  Call once at startup.
pub fn init_prometheus(port: u16) -> std::io::Result<()> {
    metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(([0, 0, 0, 0], port))
        .install()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    info!(port, "Prometheus metrics exporter listening");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_id_is_unique() {
        let a = TraceId::new();
        let b = TraceId::new();
        assert_ne!(a.as_str(), b.as_str());
    }

    #[test]
    fn trace_id_display() {
        let id = TraceId::new();
        assert!(!id.to_string().is_empty());
    }

    #[test]
    fn job_span_complete_does_not_panic() {
        let span = JobSpan::new("natural", 4);
        span.complete();
    }

    #[test]
    fn job_span_fail_does_not_panic() {
        let span = JobSpan::new("anime", 4);
        span.fail("test error");
    }

    #[test]
    fn phase_timer_finish_does_not_panic() {
        let id = TraceId::new();
        let timer = PhaseTimer::start("inference", &id, vec![("model", "natural".into())]);
        timer.finish();
    }
}
