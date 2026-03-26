//! Worker pool for parallel inference.
//!
//! Maintains a pool of OS threads; each thread lazily loads model instances
//! in thread-local storage so they are never shared between threads.
//! Jobs enter via a `std::sync::mpsc` channel and results are returned via
//! per-job sync-channels.  Shutdown sends a `None` sentinel per worker and
//! waits for the queue to drain.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::SrganError;
use crate::thread_safe_network::ThreadSafeNetwork;

// ---------------------------------------------------------------------------
// Thread-local model cache
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-worker cache of loaded models, keyed by label.
    static MODEL_CACHE: RefCell<HashMap<String, ThreadSafeNetwork>> =
        RefCell::new(HashMap::new());
}

fn get_or_load(label: &str) -> Result<(), SrganError> {
    MODEL_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        if !map.contains_key(label) {
            let net = ThreadSafeNetwork::from_label(label, None)?;
            map.insert(label.to_string(), net);
        }
        Ok(())
    })
}

fn run_inference(image_data: &[u8], label: &str) -> Result<Vec<u8>, SrganError> {
    get_or_load(label)?;

    MODEL_CACHE.with(|cache| {
        let map = cache.borrow();
        let net = map
            .get(label)
            .ok_or_else(|| SrganError::Network("model not in cache after load".into()))?;

        let img = image::load_from_memory(image_data)
            .map_err(SrganError::Image)?;

        let upscaled = net.upscale_image(&img)?;

        let mut buf = Cursor::new(Vec::new());
        upscaled
            .write_to(&mut buf, image::ImageFormat::PNG)
            .map_err(SrganError::Image)?;

        Ok(buf.into_inner())
    })
}

// ---------------------------------------------------------------------------
// Job type
// ---------------------------------------------------------------------------

/// An inference job sent to the pool.
pub struct WorkerJob {
    /// Raw image bytes (any format the `image` crate can decode).
    pub image_data: Vec<u8>,
    /// Model label: `"natural"`, `"anime"`, or `"bilinear"`.
    pub model_label: String,
    /// Rendezvous channel for the result.
    pub result_tx: mpsc::SyncSender<Result<Vec<u8>, SrganError>>,
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Shared counters updated by every worker.
#[derive(Clone)]
pub struct WorkerMetrics {
    pub queue_depth: Arc<AtomicUsize>,
    pub workers_busy: Arc<AtomicUsize>,
    pub jobs_completed: Arc<AtomicU64>,
    pub jobs_failed: Arc<AtomicU64>,
    /// Cumulative inference time (ms) across all workers.
    pub total_inference_ms: Arc<AtomicU64>,
    start: Instant,
}

impl WorkerMetrics {
    fn new() -> Self {
        Self {
            queue_depth: Arc::new(AtomicUsize::new(0)),
            workers_busy: Arc::new(AtomicUsize::new(0)),
            jobs_completed: Arc::new(AtomicU64::new(0)),
            jobs_failed: Arc::new(AtomicU64::new(0)),
            total_inference_ms: Arc::new(AtomicU64::new(0)),
            start: Instant::now(),
        }
    }

    /// Completed jobs / elapsed seconds since pool creation.
    pub fn jobs_per_second(&self) -> f64 {
        let secs = self.start.elapsed().as_secs_f64();
        if secs < 0.001 {
            return 0.0;
        }
        self.jobs_completed.load(Ordering::Relaxed) as f64 / secs
    }
}

// ---------------------------------------------------------------------------
// Health snapshot
// ---------------------------------------------------------------------------

/// Point-in-time health report for the pool.
#[derive(Debug, Clone)]
pub struct WorkerHealth {
    pub worker_count: usize,
    pub workers_busy: usize,
    pub workers_idle: usize,
    pub queue_depth: usize,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub jobs_per_second: f64,
    /// Fraction of workers currently busy (0.0–1.0).
    pub utilisation: f64,
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

/// A fixed-size pool of inference worker threads.
pub struct WorkerPool {
    /// Sending end of the job queue.
    job_tx: mpsc::Sender<Option<WorkerJob>>,
    metrics: WorkerMetrics,
    worker_count: usize,
}

impl WorkerPool {
    /// Create a pool.  `worker_count = 0` → one worker per logical CPU.
    pub fn new(worker_count: usize) -> Self {
        let n = if worker_count == 0 {
            rayon::current_num_threads().max(1)
        } else {
            worker_count
        };

        let (job_tx, job_rx) = mpsc::channel::<Option<WorkerJob>>();
        // Wrap the receiver so multiple threads can pull from it.
        let shared_rx: Arc<Mutex<mpsc::Receiver<Option<WorkerJob>>>> =
            Arc::new(Mutex::new(job_rx));

        let metrics = WorkerMetrics::new();

        for id in 0..n {
            let rx = Arc::clone(&shared_rx);
            let m = metrics.clone();

            thread::Builder::new()
                .name(format!("srgan-worker-{}", id))
                .spawn(move || worker_loop(rx, m))
                .expect("failed to spawn worker thread");
        }

        Self { job_tx, metrics, worker_count: n }
    }

    /// Submit a job and block the calling thread until the result is ready.
    ///
    /// Times out after 5 minutes (matching the server's `JOB_TIMEOUT`).
    pub fn submit(
        &self,
        image_data: Vec<u8>,
        model_label: impl Into<String>,
    ) -> Result<Vec<u8>, SrganError> {
        let (result_tx, result_rx) = mpsc::sync_channel(1);

        self.metrics.queue_depth.fetch_add(1, Ordering::Relaxed);

        self.job_tx
            .send(Some(WorkerJob {
                image_data,
                model_label: model_label.into(),
                result_tx,
            }))
            .map_err(|_| SrganError::Network("worker pool shut down".into()))?;

        result_rx
            .recv_timeout(Duration::from_secs(300))
            .map_err(|_| SrganError::Network("worker job timed out after 300 s".into()))?
    }

    /// Return a live health snapshot.
    pub fn health(&self) -> WorkerHealth {
        let busy = self
            .metrics
            .workers_busy
            .load(Ordering::Relaxed)
            .min(self.worker_count);
        WorkerHealth {
            worker_count: self.worker_count,
            workers_busy: busy,
            workers_idle: self.worker_count - busy,
            queue_depth: self.metrics.queue_depth.load(Ordering::Relaxed),
            jobs_completed: self.metrics.jobs_completed.load(Ordering::Relaxed),
            jobs_failed: self.metrics.jobs_failed.load(Ordering::Relaxed),
            jobs_per_second: self.metrics.jobs_per_second(),
            utilisation: if self.worker_count == 0 {
                0.0
            } else {
                busy as f64 / self.worker_count as f64
            },
        }
    }

    /// Signal all workers to stop after draining in-flight jobs.
    pub fn shutdown(&self) {
        for _ in 0..self.worker_count {
            let _ = self.job_tx.send(None);
        }
    }

    /// Number of workers in this pool.
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

fn worker_loop(
    rx: Arc<Mutex<mpsc::Receiver<Option<WorkerJob>>>>,
    metrics: WorkerMetrics,
) {
    loop {
        // Hold the lock only long enough to dequeue one job.
        let job = {
            let guard = rx.lock().expect("worker receiver mutex poisoned");
            guard.recv().ok()
        };

        match job {
            // Channel closed or shutdown sentinel.
            None | Some(None) => break,

            Some(Some(job)) => {
                metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
                metrics.workers_busy.fetch_add(1, Ordering::Relaxed);

                let t0 = Instant::now();
                let result = run_inference(&job.image_data, &job.model_label);
                let elapsed_ms = t0.elapsed().as_millis() as u64;

                metrics.total_inference_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
                metrics.workers_busy.fetch_sub(1, Ordering::Relaxed);

                match &result {
                    Ok(_) => {
                        metrics.jobs_completed.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        metrics.jobs_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }

                // Best-effort send; caller may have already timed out.
                let _ = job.result_tx.send(result);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_returns_sane_defaults() {
        let pool = WorkerPool::new(2);
        let h = pool.health();
        assert_eq!(h.worker_count, 2);
        assert_eq!(h.workers_idle, 2);
        assert_eq!(h.queue_depth, 0);
        assert_eq!(h.jobs_completed, 0);
    }

    #[test]
    fn shutdown_is_idempotent() {
        let pool = WorkerPool::new(1);
        pool.shutdown();
        pool.shutdown(); // second call must not panic
    }
}
