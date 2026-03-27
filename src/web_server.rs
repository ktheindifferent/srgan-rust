use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::thread;
use std::net::SocketAddr;
use std::io::{BufRead, BufReader, Write};
use base64::{Engine as _, engine::general_purpose};
use image::{ImageFormat, GenericImage};
use log::{info, warn};
use crate::error::SrganError;
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::api::auth::{self, KeyStore, RegisterRequest};
use crate::api::billing::{BillingDb, BillingStatus, CheckoutRequest, SubscriptionTier};
use crate::api::middleware::TierRateLimiter;
use crate::api::rate_limit_dashboard::RateLimitDashboard;
use crate::api::upscale::{deliver_webhook, unix_now as api_unix_now, WebhookConfig, WebhookDeliveryState};
use crate::rate_limit_dashboard::KeyUsageDashboard;
use crate::stripe_dunning::StripeDunningManager;
use crate::storage::S3Config;

// ── Request metrics ──────────────────────────────────────────────────────────

/// Per-endpoint metrics accumulator.
struct EndpointMetrics {
    request_count: u64,
    error_count: u64,
    /// All recorded latencies in microseconds (kept in memory; bounded by job volume).
    latencies_us: Vec<u64>,
}

impl EndpointMetrics {
    fn new() -> Self {
        Self { request_count: 0, error_count: 0, latencies_us: Vec::new() }
    }

    fn record(&mut self, latency: Duration, is_error: bool) {
        self.request_count += 1;
        if is_error {
            self.error_count += 1;
        }
        self.latencies_us.push(latency.as_micros() as u64);
    }

    /// Compute a percentile (0–100) over recorded latencies. Returns microseconds.
    fn percentile(&self, p: f64) -> u64 {
        if self.latencies_us.is_empty() {
            return 0;
        }
        let mut sorted = self.latencies_us.clone();
        sorted.sort_unstable();
        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn error_rate(&self) -> f64 {
        if self.request_count == 0 { 0.0 } else { self.error_count as f64 / self.request_count as f64 }
    }
}

/// Global request metrics store, protected by a mutex.
struct RequestMetrics {
    endpoints: HashMap<String, EndpointMetrics>,
}

impl RequestMetrics {
    fn new() -> Self {
        Self { endpoints: HashMap::new() }
    }

    fn record(&mut self, endpoint: &str, latency: Duration, is_error: bool) {
        self.endpoints
            .entry(endpoint.to_string())
            .or_insert_with(EndpointMetrics::new)
            .record(latency, is_error);
    }

    fn snapshot(&self) -> Vec<serde_json::Value> {
        self.endpoints.iter().map(|(ep, m)| {
            serde_json::json!({
                "endpoint": ep,
                "request_count": m.request_count,
                "error_count": m.error_count,
                "error_rate": format!("{:.4}", m.error_rate()),
                "p50_ms": m.percentile(50.0) as f64 / 1000.0,
                "p95_ms": m.percentile(95.0) as f64 / 1000.0,
                "p99_ms": m.percentile(99.0) as f64 / 1000.0,
            })
        }).collect()
    }
}

fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    if days > 0 {
        format!("{}d {}h {}m", days, hours, mins)
    } else if hours > 0 {
        format!("{}h {}m", hours, mins)
    } else {
        format!("{}m {}s", mins, secs % 60)
    }
}

/// Web server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_file_size: usize,
    pub cache_enabled: bool,
    pub cache_ttl: Duration,
    pub cors_enabled: bool,
    pub api_key: Option<String>,
    pub rate_limit: Option<usize>,  // Requests per minute
    pub model_path: Option<PathBuf>,
    pub log_requests: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            max_file_size: 50 * 1024 * 1024,  // 50MB
            cache_enabled: true,
            cache_ttl: Duration::from_secs(3600),  // 1 hour
            cors_enabled: true,
            api_key: None,
            rate_limit: Some(60),
            model_path: None,
            log_requests: true,
        }
    }
}

fn default_auto_detect() -> bool {
    true
}

fn default_true() -> bool {
    true
}

/// API request for image upscaling
#[derive(Debug, Serialize, Deserialize)]
pub struct UpscaleRequest {
    pub image_data: String,  // Base64 encoded image
    pub scale_factor: Option<u32>,
    pub format: Option<String>,
    pub quality: Option<u8>,
    /// Model label: `"natural"`, `"anime"`, `"bilinear"`,
    /// `"waifu2x"`, or `"waifu2x-noise{0..3}-scale{1,2}"`.
    pub model: Option<String>,
    /// Waifu2x noise-reduction level (0–3).
    /// Only used when `model` is `"waifu2x"` (ignored otherwise).
    pub waifu2x_noise_level: Option<u8>,
    /// Waifu2x upscaling factor (1 or 2).
    /// Only used when `model` is `"waifu2x"` (ignored otherwise).
    pub waifu2x_scale: Option<u8>,
    /// Waifu2x content style: `"anime"` (default), `"photo"`, or `"artwork"`.
    /// Selects the weight set / sharpening profile best suited for the input.
    /// Only used when `model` is `"waifu2x"` (ignored otherwise).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waifu2x_style: Option<String>,
    /// When `true` (default) and `model` is `None`, auto-detect the image type
    /// and select the best model automatically.
    #[serde(default = "default_auto_detect")]
    pub auto_detect: bool,
    /// Tile size in pixels for tiled upscaling (default: 512).
    /// Images larger than 4 MP are processed in overlapping tiles to avoid
    /// running out of memory.  Set this to override the default tile size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tile_size: Option<usize>,
    /// Optional webhook to call when the job completes (async jobs only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_config: Option<WebhookConfig>,
    /// Optional preprocessing pipeline toggles.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preprocessing: Option<crate::image_pipeline::PipelineConfig>,
    /// Optional output format / quality / scale settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<crate::output_options::OutputConfig>,
    /// When `true`, enable content-aware region analysis and per-region model
    /// selection (faces → natural, flat color → anime, text/edges → sharpened
    /// natural).  Overrides `model` and `auto_detect` when set.
    #[serde(default)]
    pub auto_enhance: bool,
}

/// Request body for the `/api/v1/detect` endpoint.
#[derive(Debug, Deserialize)]
pub struct DetectRequest {
    pub image_data: String,  // Base64 encoded image
}

/// Response from the `/api/v1/detect` endpoint.
#[derive(Debug, Serialize)]
pub struct DetectResponse {
    pub detected_type: String,
    pub recommended_model: String,
}

/// API response for image upscaling
#[derive(Debug, Serialize)]
pub struct UpscaleResponse {
    pub success: bool,
    pub image_data: Option<String>,  // Base64 encoded result (None when s3_url is set)
    pub s3_url: Option<String>,      // Presigned S3 URL (set when S3 is configured)
    pub error: Option<String>,
    pub metadata: ResponseMetadata,
}

/// Response metadata
#[derive(Debug, Serialize)]
pub struct ResponseMetadata {
    pub original_size: (u32, u32),
    pub upscaled_size: (u32, u32),
    pub processing_time_ms: u64,
    pub format: String,
    pub model_used: String,
}

/// Job status for async processing
#[derive(Debug, Clone, Serialize)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
}

// ── Batch processing types ────────────────────────────────────────────────────

/// Request for batch image upscaling
#[derive(Debug, Deserialize)]
pub struct BatchUpscaleRequest {
    /// Base64-encoded images
    pub images: Vec<String>,
    pub format: Option<String>,
    pub model: Option<String>,
}

/// Result for one image in a batch
#[derive(Debug, Clone, Serialize)]
pub struct BatchImageResult {
    pub index: usize,
    pub success: bool,
    pub image_data: Option<String>,
    pub error: Option<String>,
    pub processing_time_ms: u64,
}

/// Request to start a directory-based batch job with checkpoint tracking.
#[derive(Debug, Deserialize)]
pub struct BatchDirRequest {
    /// Server-side input directory path.
    pub input_dir: String,
    /// Server-side output directory path.
    pub output_dir: String,
    pub model: Option<String>,
    pub scale: Option<u32>,
    pub recursive: Option<bool>,
}

/// Synchronous batch response (≤10 images)
#[derive(Debug, Serialize)]
pub struct BatchUpscaleResponse {
    pub success: bool,
    pub results: Vec<BatchImageResult>,
    pub total: usize,
    pub successful: usize,
    pub failed: usize,
    pub total_processing_time_ms: u64,
}

/// Status of an async batch job
#[derive(Debug, Clone, Serialize)]
pub enum BatchJobStatus {
    Processing { completed: usize, total: usize },
    Completed,
    Failed(String),
}

/// Async batch job tracking
#[derive(Debug, Clone, Serialize)]
pub struct BatchJobInfo {
    pub batch_id: String,
    pub status: BatchJobStatus,
    pub results: Vec<BatchImageResult>,
    pub total: usize,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Async job information
#[derive(Debug, Clone, Serialize)]
pub struct JobInfo {
    pub id: String,
    pub status: JobStatus,
    pub created_at: u64,
    pub updated_at: u64,
    pub result_url: Option<String>,
    pub result_data: Option<String>,  // base64-encoded result image
    pub error: Option<String>,
    /// Model used for this job (e.g. "anime", "natural").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Input image dimensions "WxH".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_size: Option<String>,
    /// Output image dimensions "WxH".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_size: Option<String>,
    /// Webhook delivery state (present when a webhook_config was supplied).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_delivery: Option<WebhookDeliveryState>,
    /// Organization ID if this job was submitted by an org member.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org_id: Option<String>,
    /// Input file size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_file_size: Option<u64>,
    /// Output file size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file_size: Option<u64>,
    /// Scale factor used for upscaling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_factor: Option<u32>,
    /// Base64-encoded input image data (kept for preview/compare).
    #[serde(skip_serializing)]
    pub input_data: Option<String>,
}

/// Request for POST /api/v1/compare
#[derive(Debug, Deserialize)]
pub struct CompareRequest {
    /// Base64-encoded input image used as the high-resolution reference.
    /// Each model will receive a downscaled (degraded) version of this image
    /// and its output is evaluated against the original.
    pub image_data: String,
    /// Model labels to compare (1–8 entries), e.g. `["natural","anime","bilinear"]`.
    pub models: Vec<String>,
    /// Output image format: `"png"` (default) or `"jpeg"`.
    pub format: Option<String>,
    /// Tile size in pixels for tiled upscaling of large images (default: 512).
    pub tile_size: Option<usize>,
    /// Include base64-encoded output images in the response (default: true).
    #[serde(default = "default_true")]
    pub include_images: bool,
}

/// Per-model result returned by /api/v1/compare
#[derive(Debug, Serialize)]
pub struct ModelCompareResult {
    /// Model label that was evaluated.
    pub model: String,
    pub success: bool,
    /// Peak Signal-to-Noise Ratio in dB vs. the HR reference (higher is better).
    pub psnr_db: Option<f64>,
    /// Structural Similarity Index vs. the HR reference, range [0, 1] (higher is better).
    pub ssim: Option<f64>,
    /// Base64-encoded upscaled output image (absent when `include_images` is false).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<String>,
    pub processing_time_ms: u64,
    /// Width × height of the upscaled output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upscaled_size: Option<(u32, u32)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response from POST /api/v1/compare
#[derive(Debug, Serialize)]
pub struct CompareResponse {
    pub success: bool,
    /// Dimensions of the original (HR reference) input image.
    pub original_size: (u32, u32),
    /// Dimensions of the degraded (LR) image fed to each model.
    pub degraded_size: (u32, u32),
    pub results: Vec<ModelCompareResult>,
    /// Model with the highest PSNR (None if all models failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_psnr_model: Option<String>,
    /// Model with the highest SSIM (None if all models failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_ssim_model: Option<String>,
    pub total_processing_time_ms: u64,
}

/// Web API server
pub struct WebServer {
    config: ServerConfig,
    network: Arc<ThreadSafeNetwork>,
    cache: Arc<Mutex<HashMap<String, CachedResult>>>,
    jobs: Arc<Mutex<HashMap<String, JobInfo>>>,
    batch_jobs: Arc<Mutex<HashMap<String, BatchJobInfo>>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    /// Per-tier rate limiter (task 3)
    tier_rate_limiter: Arc<TierRateLimiter>,
    /// Billing database: API key → user account (task 1)
    billing_db: Arc<Mutex<BillingDb>>,
    /// S3 storage config loaded from env vars (task 2)
    s3_config: Option<S3Config>,
    /// Server start time for uptime reporting (task 4)
    server_start_time: SystemTime,
    /// Total images successfully processed (task 4)
    images_processed: Arc<AtomicU64>,
    /// Per-endpoint request metrics (admin dashboard)
    request_metrics: Arc<Mutex<RequestMetrics>>,
    /// Multi-tenant organization database
    org_db: Arc<Mutex<crate::api::org::OrgDb>>,
    /// SQLite-backed user + API key store
    key_store: Arc<KeyStore>,
    /// Per-API-key rate-limit dashboard (daily usage + throttle counts)
    rate_limit_dashboard: Arc<Mutex<RateLimitDashboard>>,
    /// Per-API-key usage dashboard (bandwidth, monthly limits)
    key_usage_dashboard: Arc<Mutex<KeyUsageDashboard>>,
    /// Stripe dunning manager
    stripe_dunning: Arc<Mutex<StripeDunningManager>>,
}

/// Cached result
struct CachedResult {
    data: Vec<u8>,
    created_at: SystemTime,
    metadata: ResponseMetadata,
}

/// Simple rate limiter
struct RateLimiter {
    requests: HashMap<String, Vec<SystemTime>>,
    limit: usize,
}

impl RateLimiter {
    fn new(limit: usize) -> Self {
        Self {
            requests: HashMap::new(),
            limit,
        }
    }
    
    fn check_rate_limit(&mut self, client_id: &str) -> bool {
        let now = SystemTime::now();
        let one_minute_ago = now - Duration::from_secs(60);
        
        // Clean old requests
        if let Some(requests) = self.requests.get_mut(client_id) {
            requests.retain(|&t| t > one_minute_ago);
            
            if requests.len() >= self.limit {
                return false;
            }
            
            requests.push(now);
        } else {
            self.requests.insert(client_id.into(), vec![now]);
        }
        
        true
    }
}

impl WebServer {
    /// Create new web server
    pub fn new(config: ServerConfig) -> Result<Self, SrganError> {
        // Load network - no mutex needed!
        let network = if let Some(ref model_path) = config.model_path {
            ThreadSafeNetwork::load_from_file(model_path)?
        } else {
            ThreadSafeNetwork::load_builtin_natural()?
        };
        
        let rate_limit = config.rate_limit.unwrap_or(60);
        
        let s3_config = S3Config::from_env();

        let key_store = KeyStore::open_in_memory()
            .map_err(|e| SrganError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("KeyStore init: {}", e),
            )))?;

        Ok(Self {
            config,
            network: Arc::new(network),
            cache: Arc::new(Mutex::new(HashMap::new())),
            jobs: Arc::new(Mutex::new(HashMap::new())),
            batch_jobs: Arc::new(Mutex::new(HashMap::new())),
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(rate_limit))),
            tier_rate_limiter: Arc::new(TierRateLimiter::new()),
            billing_db: Arc::new(Mutex::new(BillingDb::new())),
            s3_config,
            server_start_time: SystemTime::now(),
            images_processed: Arc::new(AtomicU64::new(0)),
            request_metrics: Arc::new(Mutex::new(RequestMetrics::new())),
            org_db: Arc::new(Mutex::new(crate::api::org::OrgDb::new())),
            key_store: Arc::new(key_store),
            rate_limit_dashboard: Arc::new(Mutex::new(RateLimitDashboard::new())),
            key_usage_dashboard: Arc::new(Mutex::new(KeyUsageDashboard::new())),
            stripe_dunning: Arc::new(Mutex::new(StripeDunningManager::new())),
        })
    }
    
    /// Start the web server
    pub fn start(&self) -> Result<(), SrganError> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|_| SrganError::InvalidInput("Invalid server address".into()))?;
        
        info!("Starting web server at http://{}", addr);
        info!("API endpoints:");
        info!("  POST /api/v1/upscale         - Synchronous image upscaling");
        info!("  POST /api/v1/upscale/async   - Asynchronous image upscaling");
        info!("  POST /api/v1/detect          - Detect image type (photo/anime/illustration)");
        info!("  POST /api/v1/batch           - Batch upscaling (sync ≤10 images, else async)");
        info!("  GET  /api/v1/batch/{{id}}     - Poll async batch status");
        info!("  GET  /api/v1/batch/{{id}}/checkpoint - Query CLI batch checkpoint");
        info!("  GET  /api/v1/job/{{id}}       - Check single-image job status");
        info!("  GET  /api/v1/health          - Health check");
        info!("  GET  /api/models             - List available models");
        info!("  POST /api/v1/video/upscale    - Async video upscaling (returns job_id)");
        info!("  POST /api/v1/compare         - Multi-model PSNR/SSIM comparison");
        info!("  GET  /dashboard              - Live processing dashboard (HTML)");
        info!("  GET  /api/v1/dashboard/stream - SSE live stats stream");
        info!("  GET  /preview                - WASM browser preview (HTML)");
        info!("  GET  /admin                  - Admin dashboard (HTML)");
        info!("  GET  /api/admin/stats        - Admin analytics (JSON)");
        info!("  GET  /api/admin/users        - User listing (JSON)");
        info!("  GET  /api/v1/admin/keys      - Per-key rate limit dashboard (JSON)");
        info!("  POST /webhooks/stripe        - Stripe webhook endpoint");
        info!("  (Legacy /api/* routes also supported)");
        
        if let Some(ref _api_key) = self.config.api_key {
            info!("API key authentication enabled");
        }
        
        // Start cache cleanup thread
        if self.config.cache_enabled {
            self.start_cache_cleanup();
        }
        
        // In a real implementation, would use a web framework like actix-web or warp
        // For now, simplified HTTP handling
        self.handle_requests(addr)?;
        
        Ok(())
    }
    
    /// Handle incoming requests (simplified)
    fn handle_requests(&self, addr: SocketAddr) -> Result<(), SrganError> {
        // This is a simplified implementation
        // In production, use a proper web framework
        
        use std::net::TcpListener;
        use std::io::Read;
        
        let listener = TcpListener::bind(addr)
            .map_err(|e| SrganError::Io(e))?;
        
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    warn!("Connection error: {}", e);
                    continue;
                }
            };

            // Set a generous read timeout so large uploads don't block forever
            let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));

            // Read the full HTTP request: headers via BufReader on a cloned fd,
            // then body up to Content-Length (max_file_size cap).
            let request: String = match stream.try_clone() {
                Err(_) => continue,
                Ok(cloned) => {
                    let mut rdr = BufReader::new(cloned);
                    let mut head = String::with_capacity(4096);
                    loop {
                        let mut line = String::new();
                        match rdr.read_line(&mut line) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                let eoh = line == "\r\n";
                                head.push_str(&line);
                                if eoh { break; }
                            }
                        }
                    }
                    let cl: usize = head.lines()
                        .find(|l| l.to_lowercase().starts_with("content-length:"))
                        .and_then(|l| l.splitn(2, ':').nth(1))
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0)
                        .min(self.config.max_file_size);
                    if cl > 0 {
                        let mut body = vec![0u8; cl];
                        let _ = rdr.read_exact(&mut body);
                        format!("{}\r\n{}", head, String::from_utf8_lossy(&body))
                    } else {
                        head
                    }
                }
            };

            let lines: Vec<&str> = request.lines().collect();

            if lines.is_empty() {
                continue;
            }

            let parts: Vec<&str> = lines[0].split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let method = parts[0];
            // Strip query string from path for routing
            let path = parts[1].split('?').next().unwrap_or(parts[1]);
            
            // SSE stream endpoints: take ownership of `stream` and write events
            // directly, bypassing the normal single-string response path.
            if method == "GET"
                && (path.starts_with("/api/v1/job/") || path.starts_with("/api/job/"))
                && path.ends_with("/stream")
            {
                let path_owned = path.to_string();
                self.handle_job_stream(stream, &path_owned);
                continue;
            }

            // Dashboard SSE stream — pushes live queue stats every second.
            if method == "GET" && (path == "/api/v1/dashboard/stream" || path == "/api/dashboard/stream") {
                self.handle_dashboard_stream(stream);
                continue;
            }

            // Metrics: start timing
            let req_start = Instant::now();

            // Route request
            let response = match (method, path) {
                ("GET", "/") | ("GET", "/landing") => self.handle_landing_page(),
                ("GET", "/app") => self.handle_public_ui(),
                ("GET", "/docs") => self.handle_docs_page(),
                ("GET", "/docs/openapi.json") => self.handle_openapi_spec(),
                ("GET", "/docs/webhooks") => self.handle_webhook_docs_page(),
                ("GET", "/docs/sdk") => self.handle_sdk_docs_page(),
                ("GET", "/pricing") => self.handle_pricing_page(),
                ("GET", "/demo") => self.handle_demo_page(),
                ("GET", "/preview") => self.handle_wasm_preview_page(),
                ("GET", "/dashboard") => self.handle_root_dashboard(),
                ("GET", "/admin") => self.handle_admin_panel(),
                ("GET", "/api/me") => self.handle_api_me(&request),
                ("GET", "/api/admin/users") => self.handle_admin_users(&request),
                ("GET", "/api/admin/stats") => self.handle_admin_stats(&request),
                ("GET", "/api/health") | ("GET", "/api/v1/health") => self.handle_health_check(),
                ("GET", "/api/models") | ("GET", "/api/v1/models") => self.handle_list_models(),
                ("GET", "/api/v1/stats") => self.handle_stats(),
                ("GET", "/api/v1/jobs") | ("GET", "/api/jobs") => self.handle_jobs(&request),
                ("POST", "/api/upscale") | ("POST", "/api/v1/upscale") => self.handle_upscale_sync(&request),
                ("POST", "/api/upscale/async") | ("POST", "/api/v1/upscale/async") => self.handle_upscale_async(&request),
                ("POST", "/api/batch") | ("POST", "/api/v1/batch") => self.handle_batch(&request),
                ("POST", "/api/v1/batch/start") => self.handle_batch_dir_start(&request),
                ("POST", "/api/v1/detect") => self.handle_detect(&request),
                ("POST", "/api/v1/billing/checkout") => self.handle_billing_checkout(&request),
                ("POST", "/api/v1/billing/webhook") => self.handle_billing_webhook(&request),
                ("POST", "/webhooks/stripe") => self.handle_stripe_webhook_v2(&request),
                ("GET", "/api/v1/billing/status") => self.handle_billing_status(&request),
                ("GET", "/api/v1/admin/keys") => self.handle_admin_keys(&request),
                ("GET", "/admin/api-keys") => self.handle_admin_api_keys_list(&request),
                ("GET", "/admin/rate-limits") => self.handle_admin_rate_limits_page(&request),
                ("GET", "/api/v1/rate-limit") => self.handle_rate_limit_self(&request),
                ("POST", "/api/v1/video/upscale") | ("POST", "/api/video/upscale") => self.handle_video_upscale(&request),
                ("POST", "/api/v1/compare") => self.handle_compare(&request),
                ("POST", "/api/v1/webhooks/test") | ("POST", "/api/v1/webhook/test") => self.handle_webhook_test(&request),
                // Registration & API key management
                ("POST", "/api/register") | ("POST", "/api/v1/register") => self.handle_register(&request),
                ("GET", "/api/keys") | ("GET", "/api/v1/keys") => self.handle_list_keys(&request),
                ("POST", "/api/keys") | ("POST", "/api/v1/keys") => self.handle_create_key(&request),
                _ if method == "DELETE"
                    && (path.starts_with("/api/keys/") || path.starts_with("/api/v1/keys/"))
                    && !path.contains("/rotate") => self.handle_revoke_key(&request, path),
                _ if method == "POST"
                    && (path.starts_with("/api/keys/") || path.starts_with("/api/v1/keys/"))
                    && path.ends_with("/rotate") => self.handle_rotate_key(&request, path),
                // Organization endpoints
                ("POST", "/api/orgs") | ("POST", "/api/v1/orgs") => self.handle_create_org(&request),
                _ if method == "GET"
                    && (path.starts_with("/api/orgs/") || path.starts_with("/api/v1/orgs/"))
                    && path.ends_with("/usage") => self.handle_org_usage(&request, path),
                _ if method == "POST"
                    && (path.starts_with("/api/orgs/") || path.starts_with("/api/v1/orgs/"))
                    && path.ends_with("/members") => self.handle_org_add_member(&request, path),
                _ if method == "DELETE"
                    && (path.starts_with("/api/orgs/") || path.starts_with("/api/v1/orgs/"))
                    && path.contains("/members/") => self.handle_org_remove_member(&request, path),
                _ if method == "GET"
                    && (path.starts_with("/api/orgs/") || path.starts_with("/api/v1/orgs/")) => self.handle_get_org(&request, path),
                _ if method == "GET"
                    && (path.starts_with("/api/v1/job/") || path.starts_with("/api/job/"))
                    && path.ends_with("/webhook") => self.handle_job_webhook_status(path),
                _ if method == "GET"
                    && (path.starts_with("/api/preview/") || path.starts_with("/api/v1/preview/")) => self.handle_preview(path),
                _ if method == "GET"
                    && (path.starts_with("/api/jobs/") || path.starts_with("/api/v1/jobs/"))
                    && path.ends_with("/compare") => self.handle_job_compare(path),
                _ if method == "GET" && (path.starts_with("/api/job/") || path.starts_with("/api/v1/job/")) => self.handle_job_status(path),
                _ if method == "GET" && (path.starts_with("/api/result/") || path.starts_with("/api/v1/result/")) => self.handle_job_result(path),
                _ if method == "GET" && (path.starts_with("/api/v1/batch/") || path.starts_with("/api/batch/")) && path.ends_with("/checkpoint") => self.handle_batch_checkpoint(path),
                _ if method == "GET" && (path.starts_with("/api/batch/") || path.starts_with("/api/v1/batch/")) => self.handle_batch_status(path),
                _ if method == "GET" && path.starts_with("/preview/pkg/") => self.handle_wasm_pkg_file(path),
                _ if method == "GET" && path.starts_with("/admin/api-keys/") && path.ends_with("/usage") => self.handle_admin_key_usage(&request, path),
                _ if method == "PUT" && path.starts_with("/admin/rate-limits/") => self.handle_update_rate_limit(&request, path),
                _ => self.handle_not_found(),
            };

            // Metrics: record latency and error status
            let latency = req_start.elapsed();
            let is_error = response.starts_with("HTTP/1.1 4") || response.starts_with("HTTP/1.1 5");
            // Normalise dynamic path segments for grouping
            let metrics_key = Self::normalise_metrics_path(method, path);
            if let Ok(mut m) = self.request_metrics.lock() {
                m.record(&metrics_key, latency, is_error);
            }

            // Send response
            let _ = stream.write_all(response.as_bytes());
        }
        
        Ok(())
    }
    
    /// Handle health check
    fn handle_health_check(&self) -> String {
        let response = serde_json::json!({
            "status": "ok",
            "model_loaded": true,
            "model": self.network.display(),
            "model_factor": self.network.factor(),
            "version": "0.2.0",
            "uptime": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });
        
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            response
        )
    }
    
    /// Handle list models — includes built-in models plus any custom models
    /// registered in the plugin registry (~/.srgan/models/).
    fn handle_list_models(&self) -> String {
        // Built-in model list (static)
        let mut models = vec![
            serde_json::json!({
                "name": "natural",
                "display_name": "Natural (SRGAN)",
                "description": "Neural net trained on natural photographs with L1 loss",
                "architecture": "srgan",
                "scale_factors": [4],
                "recommended_for": ["photos", "general"],
                "source": "built-in"
            }),
            serde_json::json!({
                "name": "anime",
                "display_name": "Anime (SRGAN)",
                "description": "Neural net trained on animation images with L1 loss",
                "architecture": "srgan",
                "scale_factors": [4],
                "recommended_for": ["anime", "illustrations", "cartoons"],
                "source": "built-in"
            }),
            serde_json::json!({
                "name": "waifu2x",
                "display_name": "Waifu2x",
                "description": "Waifu2x VGG7 CNN for anime/illustration upscaling and noise reduction. Uses real neural network inference when weight files are available (models/waifu2x/*.rsr), otherwise falls back to Lanczos3 + unsharp-mask approximation. Supports noise levels 0–3, scale 1×/2×, and content styles (anime/photo/artwork).",
                "architecture": "waifu2x-vgg7",
                "scale_factors": [1, 2],
                "recommended_for": ["anime", "illustrations", "manga", "photos"],
                "parameters": {
                    "waifu2x_noise_level": "0–3 (0 = none, 3 = aggressive; default 1)",
                    "waifu2x_scale": "1 or 2 (default 2)",
                    "waifu2x_style": "'anime' (default), 'photo', or 'artwork' — selects weight set / sharpening profile"
                },
                "variants": crate::waifu2x::WAIFU2X_LABELS,
                "source": "built-in / external weights",
                "weight_format": "Convert waifu2x JSON weights with: convert-model --format waifu2x-json",
                "weight_search_paths": ["$SRGAN_MODEL_PATH/waifu2x/", "./models/waifu2x/"]
            }),
            serde_json::json!({
                "name": "real-esrgan",
                "display_name": "Real-ESRGAN ×4",
                "description": "Real-ESRGAN ×4 for general photos — trained on synthetic real-world degradations (JPEG artifacts, Gaussian noise, blur). Best for compressed or noisy source images.",
                "architecture": "real-esrgan",
                "scale_factors": [4],
                "recommended_for": ["photos", "compressed", "noisy", "real-world"],
                "source": "built-in"
            }),
            serde_json::json!({
                "name": "real-esrgan-anime",
                "display_name": "Real-ESRGAN Anime ×4",
                "description": "Real-ESRGAN ×4 optimised for anime and illustration content — uses the anime-specific degradation pipeline for sharper line art.",
                "architecture": "real-esrgan",
                "scale_factors": [4],
                "recommended_for": ["anime", "illustrations", "cartoons", "line-art"],
                "source": "built-in"
            }),
            serde_json::json!({
                "name": "real-esrgan-x2",
                "display_name": "Real-ESRGAN ×2",
                "description": "Real-ESRGAN ×2 for general photos — lower memory usage than the ×4 variant; ideal when only a moderate resolution boost is needed.",
                "architecture": "real-esrgan",
                "scale_factors": [2],
                "recommended_for": ["photos", "general", "low-memory"],
                "source": "built-in"
            }),
            serde_json::json!({
                "name": "bilinear",
                "display_name": "Bilinear",
                "description": "Bilinear interpolation (no neural network)",
                "architecture": "bilinear",
                "scale_factors": [2, 4],
                "recommended_for": ["general", "quick-preview"],
                "source": "built-in"
            }),
        ];

        // Append custom models from the plugin registry
        if let Ok(registry) = crate::model_registry::ModelRegistry::load() {
            for entry in registry.custom_models() {
                models.push(serde_json::json!({
                    "name": entry.name,
                    "display_name": entry.display_name,
                    "description": entry.description,
                    "architecture": entry.model_type.as_str(),
                    "scale_factors": entry.scale_factors,
                    "weight_path": entry.weight_path,
                    "source": "custom"
                }));
            }
        }

        let response = serde_json::json!({
            "models": models,
            "default": "natural",
        });

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            response
        )
    }
    
    /// Handle synchronous upscale
    fn handle_upscale_sync(&self, request: &str) -> String {
        let body = self.extract_body(request);

        match serde_json::from_str::<UpscaleRequest>(&body) {
            Ok(req) => {
                let (allowed, rl_headers) = self.tier_rate_limit_check(request);
                if !allowed {
                    return self.rate_limit_response(&rl_headers, 3600);
                }

                // Deduct credit (org pool or personal)
                let api_key = self
                    .extract_header(request, "x-api-key")
                    .unwrap_or_default();
                if !api_key.is_empty() && !self.consume_credit_for_user(&api_key) {
                    return self.error_response(402, "No credits remaining");
                }

                // Decode base64 up-front so we can check dimensions
                let image_data = match general_purpose::STANDARD.decode(&req.image_data) {
                    Ok(d) => d,
                    Err(e) => return self.error_response(400, &format!("Invalid base64: {}", e)),
                };

                // Load image to inspect size
                let img = match image::load_from_memory(&image_data) {
                    Ok(img) => img,
                    Err(e) => return self.error_response(400, &format!("Invalid image: {}", e)),
                };

                // Images > 2MP are queued as async jobs to avoid blocking
                let pixel_count = img.width() as u64 * img.height() as u64;
                if pixel_count > 2_000_000 {
                    return self.queue_async_job_from_data(req, image_data);
                }

                // Process synchronously
                match self.process_image_decoded(req, img) {
                    Ok(response) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                        serde_json::to_string(&response)
                            .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
                    ),
                    Err(e) => self.error_response(500, &format!("{}", e)),
                }
            }
            Err(e) => self.error_response(400, &format!("Invalid request: {}", e)),
        }
    }
    
    /// Handle asynchronous upscale
    fn handle_upscale_async(&self, request: &str) -> String {
        let body = self.extract_body(request);
        
        match serde_json::from_str::<UpscaleRequest>(&body) {
            Ok(req) => {
                // Deduct credit (org pool or personal)
                let api_key_for_credit = self
                    .extract_header(request, "x-api-key")
                    .unwrap_or_default();
                if !api_key_for_credit.is_empty() && !self.consume_credit_for_user(&api_key_for_credit) {
                    return self.error_response(402, "No credits remaining");
                }

                // Generate job ID
                let job_id = self.generate_job_id();

                // Extract webhook config before moving req into the thread
                let webhook_config = req.webhook_config.clone();

                // Decode image dimensions for job metadata (best-effort)
                let (job_model, job_input_size) = {
                    let m = req.model.clone().unwrap_or_else(|| "natural".to_string());
                    let sz = general_purpose::STANDARD.decode(&req.image_data).ok()
                        .and_then(|b| image::load_from_memory(&b).ok())
                        .map(|img| format!("{}x{}", img.width(), img.height()));
                    (m, sz)
                };
                let job_scale = req.scale_factor.unwrap_or(self.network.factor() as u32);
                let job_output_size = job_input_size.as_ref().and_then(|s| {
                    let parts: Vec<&str> = s.split('x').collect();
                    if parts.len() == 2 {
                        let w: u32 = parts[0].parse().ok()?;
                        let h: u32 = parts[1].parse().ok()?;
                        Some(format!("{}x{}", w * job_scale, h * job_scale))
                    } else { None }
                });

                // Create job entry
                let job = JobInfo {
                    id: job_id.clone(),
                    status: JobStatus::Pending,
                    created_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                    updated_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                    result_url: None,
                    result_data: None,
                    error: None,
                    model: Some(job_model),
                    input_size: job_input_size,
                    output_size: job_output_size,
                    webhook_delivery: webhook_config.as_ref().map(|_| WebhookDeliveryState::default()),
                    org_id: {
                        let api_key = self.extract_header(request, "x-api-key")
                            .unwrap_or_default();
                        if let Ok(db) = self.org_db.lock() {
                            db.user_org_id(&api_key).cloned()
                        } else {
                            None
                        }
                    },
                    input_file_size: Some(req.image_data.len() as u64),
                    output_file_size: None,
                    scale_factor: Some(job_scale),
                    input_data: Some(req.image_data.clone()),
                };

                if let Ok(mut jobs) = self.jobs.lock() {
                    jobs.insert(job_id.clone(), job.clone());
                } else {
                    return self.error_response(500, "Failed to acquire job lock");
                }

                // Start processing in background
                let jobs = Arc::clone(&self.jobs);
                let network = Arc::clone(&self.network);
                let job_id_clone = job_id.clone();
                let req_clone = req;

                thread::spawn(move || {
                    // Update status to processing
                    if let Ok(mut jobs_guard) = jobs.lock() {
                        if let Some(job) = jobs_guard.get_mut(&job_id_clone) {
                            job.status = JobStatus::Processing;
                            job.updated_at = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                        }
                    }

                    // Process image with thread-safe network, return base64-encoded result
                    let result: std::result::Result<String, SrganError> = (|| {
                        let image_data = general_purpose::STANDARD.decode(&req_clone.image_data)
                            .map_err(|e| SrganError::InvalidInput(format!("Invalid base64: {}", e)))?;
                        let img = image::load_from_memory(&image_data)
                            .map_err(|e| SrganError::Image(e))?;

                        let upscaled = network.upscale_image(&img)?;

                        let format = req_clone.format.as_deref().unwrap_or("png");
                        let img_format = match format {
                            "jpeg" | "jpg" => ImageFormat::JPEG,
                            _ => ImageFormat::PNG,
                        };
                        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
                        upscaled.write_to(&mut cursor, img_format)
                            .map_err(|e| SrganError::Image(e))?;

                        Ok(general_purpose::STANDARD.encode(cursor.into_inner()))
                    })();

                    // Capture outcome for webhook before consuming result
                    let webhook_status = match &result {
                        Ok(_) => "completed",
                        Err(_) => "failed",
                    };

                    // Update job with result
                    if let Ok(mut jobs_guard) = jobs.lock() {
                        if let Some(job) = jobs_guard.get_mut(&job_id_clone) {
                            match result {
                                Ok(encoded) => {
                                    job.output_file_size = general_purpose::STANDARD.decode(&encoded)
                                        .ok().map(|b| b.len() as u64);
                                    job.status = JobStatus::Completed;
                                    job.result_url = Some(format!("/api/result/{}", job_id_clone));
                                    job.result_data = Some(encoded);
                                }
                                Err(e) => {
                                    job.status = JobStatus::Failed(e.to_string());
                                    job.error = Some(e.to_string());
                                }
                            }
                            job.updated_at = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                        }
                    }

                    // Fire webhook with retry if configured
                    if let Some(config) = webhook_config {
                        if !config.url.is_empty() {
                            let payload = serde_json::json!({
                                "job_id": job_id_clone,
                                "status": webhook_status,
                                "timestamp": api_unix_now(),
                            })
                            .to_string();
                            let jobs_wh = Arc::clone(&jobs);
                            let id_wh = job_id_clone.clone();
                            deliver_webhook(
                                config.url,
                                config.secret,
                                payload,
                                config.max_retries,
                                config.retry_delay_secs,
                                move |attempts, last_status_code, delivered| {
                                    if let Ok(mut jg) = jobs_wh.lock() {
                                        if let Some(job) = jg.get_mut(&id_wh) {
                                            job.webhook_delivery = Some(WebhookDeliveryState {
                                                attempts,
                                                last_attempt_at: Some(api_unix_now()),
                                                last_status_code,
                                                delivered,
                                            });
                                        }
                                    }
                                },
                            );
                        }
                    }
                });
                
                // Return job info
                let response = serde_json::json!({
                    "job_id": job_id,
                    "status": "pending",
                    "check_url": format!("/api/job/{}", job_id),
                });
                
                format!(
                    "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\n\r\n{}",
                    response
                )
            }
            Err(e) => self.error_response(400, &format!("Invalid request: {}", e)),
        }
    }
    
    /// Handle async video upscale request.
    ///
    /// Accepts a JSON body with video metadata and returns a job ID.
    /// The actual processing happens in the background.
    fn handle_video_upscale(&self, request: &str) -> String {
        let body = self.extract_body(request);

        // Parse video upscale request
        #[derive(serde::Deserialize)]
        struct VideoUpscaleRequest {
            /// Base64-encoded video data
            video_data: String,
            /// Model to use (natural, anime)
            #[serde(default = "default_model")]
            model: String,
            /// Upscale factor (2 or 4)
            #[serde(default = "default_scale")]
            scale: u32,
            /// Output codec (h264, h265, vp9)
            #[serde(default = "default_codec")]
            codec: String,
            /// Quality preset (low, medium, high)
            #[serde(default = "default_quality")]
            quality: String,
            /// Output FPS override
            fps: Option<f32>,
            /// Whether to preserve audio
            #[serde(default = "default_true")]
            preserve_audio: bool,
        }
        fn default_model() -> String { "natural".into() }
        fn default_scale() -> u32 { 4 }
        fn default_codec() -> String { "h264".into() }
        fn default_quality() -> String { "medium".into() }
        fn default_true() -> bool { true }

        let req: VideoUpscaleRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid request: {}", e)),
        };

        // Validate codec
        let valid_codecs = ["h264", "h265", "vp9", "av1", "prores"];
        if !valid_codecs.contains(&req.codec.to_lowercase().as_str()) {
            return self.error_response(400, &format!("Unsupported codec: {}", req.codec));
        }

        // Validate quality
        let valid_qualities = ["low", "medium", "high", "lossless"];
        if !valid_qualities.contains(&req.quality.to_lowercase().as_str()) {
            // Allow numeric CRF values
            if req.quality.parse::<u8>().is_err() {
                return self.error_response(400, &format!("Invalid quality: {}", req.quality));
            }
        }

        // Validate scale
        if req.scale != 2 && req.scale != 4 {
            return self.error_response(400, "Scale must be 2 or 4");
        }

        // Deduct credit
        let api_key = self.extract_header(request, "x-api-key").unwrap_or_default();
        if !api_key.is_empty() && !self.consume_credit_for_user(&api_key) {
            return self.error_response(402, "No credits remaining");
        }

        // Generate job ID
        let job_id = self.generate_job_id();

        // Estimate input size from base64 length
        let input_file_size = (req.video_data.len() as u64) * 3 / 4;

        // Create job entry
        let job = JobInfo {
            id: job_id.clone(),
            status: JobStatus::Pending,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            updated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            result_url: None,
            result_data: None,
            error: None,
            model: Some(req.model.clone()),
            input_size: None,
            output_size: None,
            webhook_delivery: None,
            org_id: {
                if let Ok(db) = self.org_db.lock() {
                    db.user_org_id(&api_key).cloned()
                } else {
                    None
                }
            },
            input_file_size: Some(input_file_size),
            output_file_size: None,
            scale_factor: Some(req.scale),
            input_data: None,
        };

        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.insert(job_id.clone(), job);
        } else {
            return self.error_response(500, "Failed to acquire job lock");
        }

        // Mark as processing in background
        let jobs = Arc::clone(&self.jobs);
        let job_id_bg = job_id.clone();

        thread::spawn(move || {
            // Update status to processing
            if let Ok(mut jobs_guard) = jobs.lock() {
                if let Some(job) = jobs_guard.get_mut(&job_id_bg) {
                    job.status = JobStatus::Processing;
                    job.updated_at = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                }
            }

            // Video processing is long-running; write input to a temp file,
            // invoke the VideoProcessor, then base64-encode the output.
            let result: std::result::Result<String, SrganError> = (|| {
                use crate::video::{VideoProcessor, VideoConfig, VideoCodec, VideoQuality};
                use std::fs;

                let video_bytes = general_purpose::STANDARD.decode(&req.video_data)
                    .map_err(|e| SrganError::InvalidInput(format!("Invalid base64: {}", e)))?;

                let temp_dir = std::env::temp_dir().join(format!("srgan_video_{}", job_id_bg));
                fs::create_dir_all(&temp_dir).map_err(SrganError::Io)?;

                let input_path = temp_dir.join("input.mp4");
                let output_path = temp_dir.join("output.mp4");
                fs::write(&input_path, &video_bytes).map_err(SrganError::Io)?;

                let codec = match req.codec.to_lowercase().as_str() {
                    "h265" | "hevc" => VideoCodec::H265,
                    "vp9" => VideoCodec::VP9,
                    "av1" => VideoCodec::AV1,
                    "prores" => VideoCodec::ProRes,
                    _ => VideoCodec::H264,
                };

                let quality = match req.quality.to_lowercase().as_str() {
                    "low" => VideoQuality::Low,
                    "high" => VideoQuality::High,
                    "lossless" => VideoQuality::Lossless,
                    other => other.parse::<u8>()
                        .map(VideoQuality::Custom)
                        .unwrap_or(VideoQuality::Medium),
                };

                let config = VideoConfig {
                    input_path: input_path.clone(),
                    output_path: output_path.clone(),
                    model_path: None,
                    fps: req.fps,
                    quality,
                    codec,
                    preserve_audio: req.preserve_audio,
                    parallel_frames: 4,
                    temp_dir: Some(temp_dir.clone()),
                    start_time: None,
                    duration: None,
                };

                let mut processor = VideoProcessor::new(config)?;

                // Load the network
                let network = match req.model.as_str() {
                    "anime" => crate::UpscalingNetwork::load_builtin_anime()?,
                    _ => crate::UpscalingNetwork::load_builtin_natural()?,
                };
                processor.load_network(network);
                processor.process()?;

                // Read output and encode
                let output_bytes = fs::read(&output_path).map_err(SrganError::Io)?;
                let encoded = general_purpose::STANDARD.encode(&output_bytes);

                // Cleanup temp files (best effort)
                let _ = fs::remove_dir_all(&temp_dir);

                Ok(encoded)
            })();

            // Update job with result
            if let Ok(mut jobs_guard) = jobs.lock() {
                if let Some(job) = jobs_guard.get_mut(&job_id_bg) {
                    match result {
                        Ok(data) => {
                            job.status = JobStatus::Completed;
                            job.result_data = Some(data);
                        }
                        Err(e) => {
                            job.status = JobStatus::Failed(format!("{}", e));
                            job.error = Some(format!("{}", e));
                        }
                    }
                    job.updated_at = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                }
            }
        });

        // Return job info
        let response = serde_json::json!({
            "job_id": job_id,
            "status": "pending",
            "type": "video_upscale",
            "check_url": format!("/api/v1/job/{}", job_id),
        });

        format!(
            "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\n\r\n{}",
            response
        )
    }

    /// Handle job status check
    fn handle_job_status(&self, path: &str) -> String {
        let job_id = path.trim_start_matches("/api/job/");

        if let Ok(jobs) = self.jobs.lock() {
            if let Some(job) = jobs.get(job_id) {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                    serde_json::to_string(&job)
                        .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
                )
            } else {
                self.error_response(404, "Job not found")
            }
        } else {
            self.error_response(500, "Failed to acquire job lock")
        }
    }

    /// GET /api/v1/job/:id/webhook — webhook delivery status for a job.
    fn handle_job_webhook_status(&self, path: &str) -> String {
        // Strip prefix and /webhook suffix to extract job id
        let inner = path
            .trim_start_matches("/api/v1/job/")
            .trim_start_matches("/api/job/");
        let job_id = inner.trim_end_matches("/webhook");

        if let Ok(jobs) = self.jobs.lock() {
            if let Some(job) = jobs.get(job_id) {
                let delivery = job.webhook_delivery.as_ref();
                let response = serde_json::json!({
                    "job_id": job_id,
                    "webhook_configured": delivery.is_some(),
                    "delivery": delivery,
                });
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                    response
                )
            } else {
                self.error_response(404, "Job not found")
            }
        } else {
            self.error_response(500, "Failed to acquire job lock")
        }
    }

    /// POST /api/v1/webhook/test — fire a one-shot test ping to a provided URL.
    fn handle_webhook_test(&self, request: &str) -> String {
        #[derive(serde::Deserialize)]
        struct TestWebhookRequest {
            url: String,
            secret: Option<String>,
        }

        let body = self.extract_body(request);
        let req: TestWebhookRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid request: {}", e)),
        };

        if req.url.is_empty() {
            return self.error_response(400, "url is required");
        }

        let payload = serde_json::json!({
            "event": "ping",
            "timestamp": api_unix_now(),
        })
        .to_string();

        deliver_webhook(
            req.url.clone(),
            req.secret,
            payload,
            0, // no retries for a test ping
            0,
            |_attempts, _status, _delivered| {}, // fire-and-forget, no state to update
        );

        let response = serde_json::json!({
            "status": "queued",
            "url": req.url,
            "message": "Test ping dispatched",
        });
        format!(
            "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\n\r\n{}",
            response
        )
    }

    // ── Public UI endpoints ─────────────────────────────────────────────────

    /// GET / — public-facing image upscaling page
    fn handle_public_ui(&self) -> String {
        const HTML: &str = include_str!("static/index.html");
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            HTML.len(), HTML
        )
    }

    /// GET /demo — demo gallery with pre-baked before/after examples
    fn handle_demo_page(&self) -> String {
        const HTML: &str = include_str!("static/demo.html");
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            HTML.len(), HTML
        )
    }

    /// GET /docs — full API reference page
    fn handle_docs_page(&self) -> String {
        let html = crate::web::docs::render_docs_page();
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            html.len(), html
        )
    }

    /// GET /docs/openapi.json — OpenAPI 3.0 specification
    fn handle_openapi_spec(&self) -> String {
        let json = crate::docs::render_openapi_spec();
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            json.len(), json
        )
    }

    /// GET /docs/webhooks — webhook documentation and test UI
    fn handle_webhook_docs_page(&self) -> String {
        let html = crate::webhook_docs::render_webhook_docs_page();
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            html.len(), html
        )
    }

    /// GET /docs/sdk — SDK reference page
    fn handle_sdk_docs_page(&self) -> String {
        let html = crate::sdk_docs::render_sdk_docs_page();
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            html.len(), html
        )
    }

    /// GET /pricing — pricing plans page
    fn handle_pricing_page(&self) -> String {
        let html = crate::web::docs::render_pricing_page();
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            html.len(), html
        )
    }

    /// GET / or GET /landing — marketing landing page
    fn handle_landing_page(&self) -> String {
        let html = crate::web::docs::render_landing_page();
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            html.len(), html
        )
    }

    /// GET /preview — WASM browser preview page (bilinear upscaling in-browser)
    fn handle_wasm_preview_page(&self) -> String {
        const HTML: &str = include_str!("../wasm/index.html");
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            HTML.len(), HTML
        )
    }

    /// GET /preview/pkg/* — serve WASM package files (JS glue, .wasm binary)
    fn handle_wasm_pkg_file(&self, path: &str) -> String {
        let relative = path.trim_start_matches("/preview/pkg/");
        // Sanitise: reject path traversal
        if relative.contains("..") || relative.contains('/') {
            return self.handle_not_found();
        }

        let pkg_dir = std::path::Path::new("wasm/pkg");
        let file_path = pkg_dir.join(relative);

        match std::fs::read(&file_path) {
            Ok(contents) => {
                let content_type = if relative.ends_with(".js") {
                    "application/javascript"
                } else if relative.ends_with(".wasm") {
                    "application/wasm"
                } else if relative.ends_with(".d.ts") {
                    "application/typescript"
                } else {
                    "application/octet-stream"
                };

                // For .wasm files we need binary response; use base64-free approach
                // by returning raw bytes via the HTTP text (works since we write_all as bytes)
                if relative.ends_with(".wasm") {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: public, max-age=3600\r\n\r\n",
                        content_type, contents.len()
                    );
                    // We need to return String but .wasm is binary. Use Latin-1 mapping.
                    let mut result = header;
                    // Safety: each byte maps to a valid char in 0..=255
                    for &b in &contents {
                        result.push(b as char);
                    }
                    result
                } else {
                    let text = String::from_utf8_lossy(&contents);
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {}; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: public, max-age=3600\r\n\r\n{}",
                        content_type, contents.len(), text
                    )
                }
            }
            Err(_) => self.handle_not_found(),
        }
    }

    // ── Dashboard / admin endpoints ──────────────────────────────────────────

    /// GET /dashboard — internal dashboard (SPA; all data fetched via JS)
    fn handle_root_dashboard(&self) -> String {
        let html = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SRGAN-Rust — Live Dashboard</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--accent:#00d4ff;
  --green:#3fb950;--yellow:#d29922;--red:#f85149;
  --text:#c9d1d9;--muted:#8b949e;--r:8px;--font:ui-monospace,'Menlo',monospace}
html,body{height:100%;font-family:var(--font);background:var(--bg);color:var(--text);font-size:14px}
.top-bar{background:var(--bg2);border-bottom:1px solid var(--border);padding:.55rem 1.2rem;
  display:flex;align-items:center;gap:.7rem;font-size:.82rem}
.pulse{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
.pulse.off{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.wrap{max-width:1100px;margin:0 auto;padding:1.4rem 1.2rem}
h1{font-size:1.15rem;color:var(--accent);font-weight:700;margin-bottom:1rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.6rem;margin-bottom:1.3rem}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:.8rem}
.card-val{font-size:1.6rem;font-weight:700;color:var(--accent);line-height:1}
.card-lbl{color:var(--muted);font-size:.66rem;margin-top:.25rem;text-transform:uppercase;letter-spacing:.05em}
.card.g .card-val{color:var(--green)}
.card.y .card-val{color:var(--yellow)}
.card.r .card-val{color:var(--red)}
.util-bar{height:18px;background:var(--bg3);border-radius:4px;overflow:hidden;margin-bottom:1.3rem}
.util-fill{height:100%;background:var(--accent);transition:width .4s ease;border-radius:4px}
.section{font-size:.9rem;font-weight:600;margin-bottom:.6rem;color:var(--text)}
.tbl-wrap{border:1px solid var(--border);border-radius:var(--r);overflow:auto;margin-bottom:1.2rem}
table{width:100%;border-collapse:collapse;font-size:.8rem}
thead th{background:var(--bg3);color:var(--muted);text-align:left;padding:.45rem .7rem;
  border-bottom:1px solid var(--border);font-size:.66rem;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
tbody tr{border-bottom:1px solid var(--border)}
tbody tr:last-child{border-bottom:none}
td{padding:.4rem .7rem;vertical-align:middle;white-space:nowrap}
.badge{display:inline-block;padding:.1rem .35rem;border-radius:4px;font-size:.68rem;font-weight:600}
.bp{background:#21262d;color:var(--yellow);border:1px solid #9e6a03}
.bpr{background:#21262d;color:var(--accent);border:1px solid #0f6ab6}
.bc{background:#21262d;color:var(--green);border:1px solid #238636}
.bf{background:#21262d;color:var(--red);border:1px solid #8b2121}
.empty{padding:1.5rem;text-align:center;color:var(--muted)}
.footer{text-align:center;color:var(--muted);font-size:.72rem;margin-top:1rem}
</style>
</head>
<body>
<div class="top-bar">
  <span class="pulse" id="sseIndicator"></span>
  <span style="color:var(--text);font-weight:600">SRGAN-Rust</span>
  <span style="color:var(--border)">|</span>
  <span style="color:var(--muted)">Live Processing Dashboard</span>
  <span style="margin-left:auto;color:var(--muted);font-size:.74rem" id="lastUpdate">Connecting…</span>
</div>
<div class="wrap">
  <h1>Processing Dashboard</h1>

  <div class="cards">
    <div class="card"><div class="card-val" id="cPend">–</div><div class="card-lbl">Pending</div></div>
    <div class="card"><div class="card-val" id="cProc">–</div><div class="card-lbl">Processing</div></div>
    <div class="card g"><div class="card-val" id="cComp">–</div><div class="card-lbl">Completed</div></div>
    <div class="card r"><div class="card-val" id="cFail">–</div><div class="card-lbl">Failed</div></div>
    <div class="card"><div class="card-val" id="cTotal">–</div><div class="card-lbl">Total Jobs</div></div>
    <div class="card y"><div class="card-val" id="cUp">–</div><div class="card-lbl">Uptime</div></div>
    <div class="card"><div class="card-val" id="cImgs">–</div><div class="card-lbl">Processed</div></div>
  </div>

  <div class="section">Worker Utilization: <span id="utilPct">0</span>%</div>
  <div class="util-bar"><div class="util-fill" id="utilBar" style="width:0%"></div></div>

  <div class="section">Active Jobs</div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Job ID</th><th>Status</th><th>Model</th><th>Created</th></tr></thead>
      <tbody id="jobsBody"><tr><td colspan="4" class="empty">No active jobs</td></tr></tbody>
    </table>
  </div>

  <div class="footer">Powered by Server-Sent Events — updates every second</div>
</div>

<script>
(function(){
  var es;
  var retryDelay = 1000;

  function connect() {
    es = new EventSource('/api/v1/dashboard/stream');
    var indicator = document.getElementById('sseIndicator');
    var lastUpdate = document.getElementById('lastUpdate');

    es.addEventListener('stats', function(e) {
      var d = JSON.parse(e.data);
      set('cPend', d.pending);
      set('cProc', d.processing);
      set('cComp', d.completed);
      set('cFail', d.failed);
      set('cTotal', d.total);
      set('cImgs', d.images_processed);
      set('cUp', fmtUp(d.uptime_secs || 0));

      var pct = d.worker_utilization_pct || 0;
      set('utilPct', pct.toFixed(1));
      document.getElementById('utilBar').style.width = Math.min(pct, 100) + '%';

      indicator.classList.remove('off');
      lastUpdate.textContent = 'Live — ' + new Date().toLocaleTimeString();
      retryDelay = 1000;
    });

    es.addEventListener('jobs', function(e) {
      var jobs = JSON.parse(e.data);
      var tbody = document.getElementById('jobsBody');
      if (!jobs.length) {
        tbody.innerHTML = '<tr><td colspan="4" class="empty">No active jobs</td></tr>';
        return;
      }
      var html = '';
      for (var i = 0; i < jobs.length; i++) {
        var j = jobs[i];
        var badge = j.status === 'processing'
          ? '<span class="badge bpr">processing</span>'
          : '<span class="badge bp">pending</span>';
        var t = j.created_at ? new Date(j.created_at * 1000).toLocaleTimeString() : '–';
        html += '<tr>'
          + '<td><code style="font-size:.74rem">' + (j.id || '').substring(0, 8) + '…</code></td>'
          + '<td>' + badge + '</td>'
          + '<td>' + (j.model || '–') + '</td>'
          + '<td>' + t + '</td>'
          + '</tr>';
      }
      tbody.innerHTML = html;
    });

    es.onerror = function() {
      indicator.classList.add('off');
      lastUpdate.textContent = 'Disconnected — retrying…';
      es.close();
      setTimeout(connect, retryDelay);
      retryDelay = Math.min(retryDelay * 2, 10000);
    };
  }

  function set(id, v) {
    var e = document.getElementById(id);
    if (e) e.textContent = v;
  }

  function fmtUp(s) {
    if (s < 60) return s + 's';
    if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
    if (s < 86400) return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
    return Math.floor(s / 86400) + 'd ' + Math.floor((s % 86400) / 3600) + 'h';
  }

  connect();
})();
</script>
</body>
</html>"##;
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            html.len(), html
        )
    }

    /// GET /admin — admin panel SPA (protected by ADMIN_TOKEN bearer token)
    fn handle_admin_panel(&self) -> String {
        let html = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SRGAN Admin</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--accent:#00d4ff;
  --green:#3fb950;--yellow:#d29922;--red:#f85149;
  --text:#c9d1d9;--muted:#8b949e;--r:8px;--font:ui-monospace,'Menlo',monospace}
body{font-family:var(--font);background:var(--bg);color:var(--text);font-size:14px;min-height:100vh}
.hdr{background:var(--bg2);border-bottom:1px solid var(--border);padding:.7rem 1.4rem;
  display:flex;align-items:center;gap:.9rem}
.hdr h1{font-size:.95rem;color:var(--accent);font-weight:700}
a.back{color:var(--muted);text-decoration:none;font-size:.85rem}.a.back:hover{color:var(--text)}
.main{padding:1.4rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.65rem;margin-bottom:1.3rem}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:.85rem}
.card-val{font-size:1.5rem;font-weight:700;color:var(--accent);line-height:1}
.card-lbl{color:var(--muted);font-size:.68rem;margin-top:.28rem;text-transform:uppercase;letter-spacing:.05em}
.card.g .card-val{color:var(--green)}.card.y .card-val{color:var(--yellow)}
.tbl-wrap{border:1px solid var(--border);border-radius:var(--r);overflow:auto;margin-bottom:1.3rem}
table{width:100%;border-collapse:collapse;font-size:.81rem}
thead th{background:var(--bg3);color:var(--muted);text-align:left;padding:.5rem .75rem;
  border-bottom:1px solid var(--border);font-size:.68rem;text-transform:uppercase;white-space:nowrap}
tbody tr{border-bottom:1px solid var(--border)}tbody tr:last-child{border-bottom:none}
td{padding:.45rem .75rem;white-space:nowrap}
.badge{display:inline-block;padding:.1rem .38rem;border-radius:4px;font-size:.7rem;font-weight:600}
.tier-free{background:#21262d;color:var(--muted);border:1px solid var(--border)}
.tier-pro{background:#21262d;color:var(--accent);border:1px solid #0f6ab6}
.tier-enterprise{background:#21262d;color:var(--yellow);border:1px solid #9e6a03}
.bc{background:#21262d;color:var(--green);border:1px solid #238636}
.bp{background:#21262d;color:var(--yellow);border:1px solid #9e6a03}
.bpr{background:#21262d;color:var(--accent);border:1px solid #0f6ab6}
.bf{background:#21262d;color:var(--red);border:1px solid #8b2121}
.login-wrap{display:flex;align-items:center;justify-content:center;min-height:calc(100vh - 48px)}
.login-box{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:1.8rem;width:300px}
.login-box h2{color:var(--accent);font-size:1rem;margin-bottom:1.1rem}
input[type=password]{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:var(--r);
  padding:.45rem .65rem;color:var(--text);font-family:var(--font);margin-bottom:.65rem;font-size:.88rem}
.btn{display:inline-block;padding:.42rem .9rem;border-radius:var(--r);font-family:var(--font);
  font-size:.86rem;cursor:pointer;border:none;width:100%}
.btn-p{background:var(--accent);color:#000;font-weight:600}.btn-p:hover{opacity:.85}
#errMsg{color:var(--red);font-size:.8rem;margin-top:.45rem}
.sec-ttl{font-size:.95rem;font-weight:600;color:var(--text);margin:1.2rem 0 .65rem}
</style>
</head>
<body>
<div class="hdr">
  <a href="/" class="back">&#8592; Dashboard</a>
  <h1>&#128274; Admin Panel</h1>
  <span style="margin-left:auto;font-size:.78rem;color:var(--muted)" id="adminSt"></span>
  <button id="logoutBtn" class="btn" style="width:auto;padding:.25rem .65rem;font-size:.78rem;background:var(--bg3);color:var(--muted);border:1px solid var(--border);display:none" onclick="logout()">Logout</button>
</div>
<div id="loginView" class="login-wrap">
  <div class="login-box">
    <h2>Admin Login</h2>
    <input type="password" id="tokInp" placeholder="Admin secret (ADMIN_TOKEN)" onkeydown="if(event.key==='Enter')login()">
    <button class="btn btn-p" onclick="login()">Sign In</button>
    <div id="errMsg"></div>
  </div>
</div>
<div id="adminView" style="display:none">
  <div class="main">
    <div class="sec-ttl">System Metrics</div>
    <div class="cards">
      <div class="card"><div class="card-val" id="mUp">–</div><div class="card-lbl">Uptime</div></div>
      <div class="card"><div class="card-val" id="mMem">–</div><div class="card-lbl">RAM Used</div></div>
      <div class="card"><div class="card-val" id="mLoad">–</div><div class="card-lbl">Load</div></div>
      <div class="card g"><div class="card-val" id="mComp">–</div><div class="card-lbl">Completed</div></div>
      <div class="card"><div class="card-val" id="mTotal">–</div><div class="card-lbl">Total Jobs</div></div>
      <div class="card y"><div class="card-val" id="mUsers">–</div><div class="card-lbl">Users</div></div>
      <div class="card"><div class="card-val" id="mAvgMs">–</div><div class="card-lbl">Avg Process ms</div></div>
      <div class="card"><div class="card-val" id="mPending">–</div><div class="card-lbl">Pending</div></div>
      <div class="card" style="border-color:var(--red)"><div class="card-val" style="color:var(--red)" id="mFailed">–</div><div class="card-lbl">Failed</div></div>
    </div>
    <div class="sec-ttl">Credit Usage by Tier</div>
    <div class="cards" id="tierCards">
      <div class="card"><div class="card-val" id="cFree">–</div><div class="card-lbl">Free consumed</div></div>
      <div class="card"><div class="card-val" id="cPro">–</div><div class="card-lbl">Pro consumed</div></div>
      <div class="card"><div class="card-val" id="cEnt">–</div><div class="card-lbl">Enterprise consumed</div></div>
    </div>
    <div class="sec-ttl">API Key Rate Limits</div>
    <div style="margin-bottom:.6rem">
      <button class="btn" id="rlTabBtn" style="width:auto;padding:.25rem .65rem;font-size:.78rem;background:var(--accent);color:#000;font-weight:600" onclick="toggleRlTab()">Show Detailed Rate Limits</button>
    </div>
    <div id="rateLimitsPanel" style="display:none">
      <div style="margin-bottom:.8rem">
        <input type="text" id="rlSearch" placeholder="Filter by key ID or tier..."
          style="background:var(--bg);border:1px solid var(--border);border-radius:var(--r);
          padding:.4rem .65rem;color:var(--text);font-family:var(--font);font-size:.82rem;width:260px">
      </div>
      <div class="tbl-wrap" id="rlDetailTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    </div>
    <div class="tbl-wrap" id="keysTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    <div class="sec-ttl">Top Users by Job Count</div>
    <div class="tbl-wrap" id="topUsersTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    <div class="sec-ttl">Users</div>
    <div class="tbl-wrap" id="usersTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    <div class="sec-ttl">Model Performance</div>
    <div class="tbl-wrap" id="modelMetricsTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    <div class="sec-ttl">Endpoint Metrics</div>
    <div class="tbl-wrap" id="metricsTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    <div class="sec-ttl">Recent Jobs</div>
    <div class="tbl-wrap" id="jobsTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    <div class="sec-ttl">API Key Management</div>
    <div class="tbl-wrap" id="keyMgmtTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
  </div>
</div>
<script>
var tok=sessionStorage.getItem('srgan_admin')||'';
if(tok)init();

function login(){tok=document.getElementById('tokInp').value.trim();init();}

function logout(){
  tok='';sessionStorage.removeItem('srgan_admin');
  document.getElementById('adminView').style.display='none';
  document.getElementById('loginView').style.display='flex';
  document.getElementById('logoutBtn').style.display='none';
  document.getElementById('adminSt').textContent='';
}

function init(){
  document.getElementById('errMsg').textContent='';
  fetch('/api/admin/users',{headers:{Authorization:'Bearer '+tok}}).then(function(r){
    if(r.status===401){document.getElementById('errMsg').textContent='Invalid admin token';tok='';return;}
    if(!r.ok)throw new Error(r.statusText);
    return r.json();
  }).then(function(d){
    if(!d)return;
    sessionStorage.setItem('srgan_admin',tok);
    document.getElementById('loginView').style.display='none';
    document.getElementById('adminView').style.display='block';
    document.getElementById('logoutBtn').style.display='inline-block';
    document.getElementById('adminSt').textContent='Authenticated';
    renderUsers(d.users||[]);
    refresh();
    setInterval(refresh,5000);
  }).catch(function(e){document.getElementById('errMsg').textContent='Error: '+e.message;});
}

function refresh(){
  var h={Authorization:'Bearer '+tok};
  Promise.all([
    fetch('/api/admin/users',{headers:h}).then(function(r){return r.ok?r.json():null;}),
    fetch('/api/v1/stats',{headers:h}).then(function(r){return r.ok?r.json():null;}),
    fetch('/api/v1/jobs',{headers:h}).then(function(r){return r.ok?r.json():null;}),
    fetch('/api/admin/stats',{headers:h}).then(function(r){return r.ok?r.json():null;}),
    fetch('/api/v1/admin/keys',{headers:h}).then(function(r){return r.ok?r.json():null;})
  ]).then(function(res){
    if(res[0])renderUsers(res[0].users||[]);
    if(res[1])renderMetrics(res[1]);
    if(res[2])renderJobs(res[2].jobs||[]);
    if(res[3])renderAdminStats(res[3]);
    if(res[4])renderKeys(res[4].keys||[]);
    if(document.getElementById('rateLimitsPanel').style.display!=='none')fetchRateLimits();
  }).catch(function(){});
}

function set(id,v){var e=document.getElementById(id);if(e)e.textContent=v;}

function fmtUp(s){
  if(s<60)return s+'s';if(s<3600)return Math.floor(s/60)+'m';
  if(s<86400)return Math.floor(s/3600)+'h';return Math.floor(s/86400)+'d';
}

function renderMetrics(s){
  set('mUp',fmtUp(s.uptime_secs||0));
  set('mMem',(s.system&&s.system.mem_used_mb||0)+' MB');
  set('mLoad',((s.system&&s.system.load_avg_1m)||0).toFixed(2));
  set('mComp',s.jobs&&s.jobs.completed!=null?s.jobs.completed:'–');
  set('mTotal',s.jobs&&s.jobs.total!=null?s.jobs.total:'–');
}

function renderUsers(users){
  set('mUsers',users.length);
  if(!users.length){document.getElementById('usersTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No users yet</div>';return;}
  var html='<table><thead><tr><th>API Key</th><th>Tier</th><th>Credits Left</th><th>Total Jobs</th><th>Last Active</th></tr></thead><tbody>';
  users.forEach(function(u){
    var la=u.last_active?new Date(u.last_active*1000).toLocaleString():'–';
    html+='<tr>'
      +'<td><code>'+u.user_id.slice(0,24)+(u.user_id.length>24?'…':'')+'</code></td>'
      +'<td><span class="badge tier-'+u.tier+'">'+u.tier+'</span></td>'
      +'<td>'+u.credits_remaining+'</td>'
      +'<td>'+(u.total_jobs||u.credits_consumed_today||0)+'</td>'
      +'<td>'+la+'</td>'
      +'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('usersTbl').innerHTML=html;
}

function jobBadge(st){
  var s=st||'';
  if(s==='completed')return '<span class="badge bc">completed</span>';
  if(s==='processing')return '<span class="badge bpr">processing</span>';
  if(s.startsWith('failed'))return '<span class="badge bf">failed</span>';
  return '<span class="badge bp">'+s+'</span>';
}

function renderJobs(jobs){
  if(!jobs.length){document.getElementById('jobsTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No jobs yet</div>';return;}
  var html='<table><thead><tr><th>Job ID</th><th>Status</th><th>Model</th><th>Input</th><th>Output</th><th>Duration</th><th>Time</th></tr></thead><tbody>';
  jobs.forEach(function(j){
    var dur=j.duration_secs!=null?j.duration_secs.toFixed(1)+'s':'–';
    html+='<tr>'
      +'<td><code>'+j.id.slice(0,12)+'…</code></td>'
      +'<td>'+jobBadge(j.status)+'</td>'
      +'<td>'+(j.model||'–')+'</td>'
      +'<td>'+(j.input_size||'–')+'</td>'
      +'<td>'+(j.output_size||'–')+'</td>'
      +'<td>'+dur+'</td>'
      +'<td>'+new Date(j.created_at*1000).toLocaleTimeString()+'</td>'
      +'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('jobsTbl').innerHTML=html;
}

function renderAdminStats(s){
  set('mAvgMs',s.avg_processing_time_ms!=null?s.avg_processing_time_ms.toFixed(1):'–');
  set('mPending',s.jobs_by_status?s.jobs_by_status.pending:'–');
  set('mFailed',s.jobs_by_status?s.jobs_by_status.failed:'–');
  var c=s.credit_usage_by_tier||{};
  set('cFree',c.free?c.free.consumed:'0');
  set('cPro',c.pro?c.pro.consumed:'0');
  set('cEnt',c.enterprise?c.enterprise.consumed:'0');
  if(s.system){
    set('mMem',s.system.used_ram_mb+'/'+s.system.total_ram_mb+' MB');
    set('mLoad',s.system.cpu_count+' cores');
  }
  renderTopUsers(s.top_users||[]);
  renderEndpointMetrics(s.endpoint_metrics||[]);
  renderModelMetrics(s.model_metrics||[]);
  renderKeyMgmt(s.api_keys||[]);
}

function renderModelMetrics(metrics){
  if(!metrics.length){document.getElementById('modelMetricsTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No model data yet</div>';return;}
  var html='<table><thead><tr><th>Model</th><th>Total Inferences</th><th>Avg ms</th><th>Min ms</th><th>Max ms</th><th>Throughput/min</th></tr></thead><tbody>';
  metrics.forEach(function(m){
    html+='<tr>'
      +'<td><code>'+m.model_name+'</code></td>'
      +'<td>'+m.total_inferences+'</td>'
      +'<td>'+m.avg_inference_ms.toFixed(1)+'</td>'
      +'<td>'+m.min_inference_ms+'</td>'
      +'<td>'+m.max_inference_ms+'</td>'
      +'<td>'+m.throughput_per_min.toFixed(1)+'</td>'
      +'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('modelMetricsTbl').innerHTML=html;
}

function renderKeyMgmt(keys){
  if(!keys.length){document.getElementById('keyMgmtTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No API keys</div>';return;}
  var html='<table><thead><tr><th>Key</th><th>Tier</th><th>Requests Today</th><th>Created</th><th>Status</th></tr></thead><tbody>';
  keys.forEach(function(k){
    var stCls=k.is_active?'bc':'bf';
    html+='<tr>'
      +'<td><code>'+k.key_prefix+'</code></td>'
      +'<td><span class="badge tier-'+k.tier+'">'+k.tier+'</span></td>'
      +'<td>'+k.requests_today+'</td>'
      +'<td style="font-size:.75rem">'+k.created_at+'</td>'
      +'<td><span class="badge '+stCls+'">'+(k.is_active?'active':'revoked')+'</span></td>'
      +'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('keyMgmtTbl').innerHTML=html;
}

function renderTopUsers(users){
  if(!users.length){document.getElementById('topUsersTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No data</div>';return;}
  var html='<table><thead><tr><th>User</th><th>Tier</th><th>Jobs</th></tr></thead><tbody>';
  users.forEach(function(u){
    html+='<tr><td><code>'+u.user_id.slice(0,24)+(u.user_id.length>24?'…':'')+'</code></td>'
      +'<td><span class="badge tier-'+u.tier+'">'+u.tier+'</span></td>'
      +'<td>'+u.job_count+'</td></tr>';
  });
  html+='</tbody></table>';
  document.getElementById('topUsersTbl').innerHTML=html;
}

function renderEndpointMetrics(metrics){
  if(!metrics.length){document.getElementById('metricsTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No requests yet</div>';return;}
  var html='<table><thead><tr><th>Endpoint</th><th>Requests</th><th>Errors</th><th>Error Rate</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th></tr></thead><tbody>';
  metrics.sort(function(a,b){return b.request_count-a.request_count;});
  metrics.forEach(function(m){
    html+='<tr>'
      +'<td><code>'+m.endpoint+'</code></td>'
      +'<td>'+m.request_count+'</td>'
      +'<td>'+m.error_count+'</td>'
      +'<td>'+(parseFloat(m.error_rate)*100).toFixed(1)+'%</td>'
      +'<td>'+m.p50_ms.toFixed(1)+'</td>'
      +'<td>'+m.p95_ms.toFixed(1)+'</td>'
      +'<td>'+m.p99_ms.toFixed(1)+'</td>'
      +'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('metricsTbl').innerHTML=html;
}

function renderKeys(keys){
  if(!keys.length){document.getElementById('keysTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No API keys</div>';return;}
  var html='<table><thead><tr><th>Key ID</th><th>Plan</th><th>Usage Today</th><th>Daily Quota</th><th>Status</th></tr></thead><tbody>';
  keys.forEach(function(k){
    var stCls=k.status==='active'?'bc':'bf';
    html+='<tr>'
      +'<td><code>'+(k.key_id||k.user_id.slice(0,8)+'...')+'</code></td>'
      +'<td><span class="badge tier-'+k.tier+'">'+k.tier+'</span></td>'
      +'<td>'+(k.credits_consumed_today||0)+'</td>'
      +'<td>'+(k.daily_quota||'–')+'</td>'
      +'<td><span class="badge '+stCls+'">'+(k.status||'active')+'</span></td>'
      +'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('keysTbl').innerHTML=html;
}
function toggleRlTab(){
  var p=document.getElementById('rateLimitsPanel');
  var b=document.getElementById('rlTabBtn');
  if(p.style.display==='none'){p.style.display='block';b.textContent='Hide Detailed Rate Limits';fetchRateLimits();}
  else{p.style.display='none';b.textContent='Show Detailed Rate Limits';}
}
function fetchRateLimits(){
  fetch('/admin/rate-limits',{headers:{Authorization:'Bearer '+tok}}).then(function(r){return r.ok?r.json():null;}).then(function(d){
    if(d)renderRateLimitsDetail(d.keys||[]);
  }).catch(function(){});
}
function renderRateLimitsDetail(keys){
  if(!keys.length){document.getElementById('rlDetailTbl').innerHTML='<div style="padding:1.5rem;color:var(--muted);text-align:center">No rate limit data</div>';return;}
  var html='<table><thead><tr><th>Key</th><th>Tier</th><th>Today</th><th>Month</th><th>RPM</th><th>Daily Limit</th><th>Overages</th><th>Last Seen</th><th>Usage</th><th>Edit</th></tr></thead><tbody>';
  keys.forEach(function(k){
    var pct=k.pct_daily||0;var bc=pct>90?'var(--red)':pct>70?'var(--yellow)':'var(--green)';
    var ls=k.last_seen?new Date(k.last_seen*1000).toLocaleString():'–';
    var kd=k.api_key.length>16?k.api_key.slice(0,12)+'...':k.api_key;
    html+='<tr><td><code>'+kd+'</code></td>'
      +'<td><span class="badge tier-'+k.tier+'">'+k.tier+'</span></td>'
      +'<td>'+k.requests_today+'</td>'
      +'<td>'+k.requests_this_month+'</td>'
      +'<td>'+k.rate_limit_rpm+'/min</td>'
      +'<td>'+k.rate_limit_daily+'</td>'
      +'<td style="color:'+(k.overage_count>0?'var(--red)':'var(--muted)')+'">'+k.overage_count+'</td>'
      +'<td style="font-size:.75rem">'+ls+'</td>'
      +'<td><div style="background:var(--bg3);border-radius:3px;height:14px;width:70px;overflow:hidden">'
      +'<div style="background:'+bc+';height:100%;width:'+Math.min(pct,100)+'%"></div></div></td>'
      +'<td><button class="btn" style="width:auto;padding:.12rem .35rem;font-size:.7rem;background:var(--bg3);color:var(--accent);border:1px solid var(--border)" onclick="editRL(\''+k.api_key+'\','+k.rate_limit_rpm+','+k.rate_limit_daily+')">Edit</button></td></tr>';
  });
  html+='</tbody></table>';document.getElementById('rlDetailTbl').innerHTML=html;
}
function editRL(k,rpm,daily){
  var r=prompt('RPM limit for '+k.slice(0,12)+'...?',rpm);if(r===null)return;
  var d=prompt('Daily limit?',daily);if(d===null)return;
  fetch('/admin/rate-limits/'+encodeURIComponent(k),{method:'PUT',headers:{Authorization:'Bearer '+tok,'Content-Type':'application/json'},
    body:JSON.stringify({rate_limit_rpm:parseInt(r)||60,rate_limit_daily:parseInt(d)||0})
  }).then(function(r){return r.json();}).then(function(d){if(d.ok){fetchRateLimits();refresh();}else alert('Error: '+(d.error||'unknown'));}).catch(function(e){alert(e.message);});
}
if(document.getElementById('rlSearch')){document.getElementById('rlSearch').addEventListener('input',function(e){
  var q=e.target.value.toLowerCase();document.querySelectorAll('#rlDetailTbl tbody tr').forEach(function(r){r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';});
});}
</script>
</body>
</html>"##;
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            html.len(), html
        )
    }

    /// GET /api/me — current user profile and credit balance
    fn handle_api_me(&self, request: &str) -> String {
        let api_key = self.resolve_api_identity(request);

        let status = if let Ok(mut db) = self.billing_db.lock() {
            let account = db.get_or_create_free(&api_key);
            serde_json::json!({
                "user_id": account.user_id,
                "tier": account.tier.as_str(),
                "credits_remaining": account.credits_remaining,
                "credits_reset_at": account.credits_reset_at,
                "credits_issued_today": account.credits_issued_today,
                "credits_consumed_today": account.credits_consumed_today,
            })
        } else {
            return self.error_response(500, "Internal error");
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            status
        )
    }

    // ── Registration & API key management ────────────────────────────────────

    /// POST /api/register — create account, return JWT.
    fn handle_register(&self, request: &str) -> String {
        let body = self.extract_body(request);
        let req: RegisterRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid JSON: {}", e)),
        };
        if req.email.is_empty() || !req.email.contains('@') {
            return self.error_response(400, "Valid email required");
        }
        if req.password.len() < 8 {
            return self.error_response(400, "Password must be at least 8 characters");
        }

        let user_id = match self.key_store.register_user(&req.email, &req.password) {
            Ok(uid) => uid,
            Err(e) => return self.error_response(409, &format!("{}", e)),
        };

        let token = match auth::issue_jwt(&user_id, &req.email) {
            Ok(t) => t,
            Err(_) => return self.error_response(500, "Failed to issue token"),
        };

        let resp = serde_json::json!({
            "user_id": user_id,
            "email": req.email,
            "token": token,
        });
        format!(
            "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            resp
        )
    }

    /// POST /api/keys — create a named API key (requires auth).
    fn handle_create_key(&self, request: &str) -> String {
        let user_id = match self.authenticate_request(request) {
            Some(uid) => uid,
            None => return self.error_response(401, "Authentication required"),
        };

        let body = self.extract_body(request);
        let req: crate::api::auth::CreateKeyRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid JSON: {}", e)),
        };

        let tier = req
            .tier
            .as_deref()
            .and_then(|t| t.parse::<crate::api::auth::KeyTier>().ok())
            .unwrap_or(crate::api::auth::KeyTier::Free);

        match self.key_store.create_key_for_user(tier, req.label.clone(), Some(&user_id)) {
            Ok(key) => {
                let resp = serde_json::json!({
                    "id": key.id,
                    "key": key.key,
                    "label": key.label,
                    "tier": tier.as_str(),
                    "daily_limit": tier.daily_limit(),
                    "created_at": key.created_at,
                });
                format!(
                    "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                    resp
                )
            }
            Err(e) => self.error_response(500, &format!("Failed to create key: {}", e)),
        }
    }

    /// GET /api/keys — list caller's active API keys (masked).
    fn handle_list_keys(&self, request: &str) -> String {
        let user_id = match self.authenticate_request(request) {
            Some(uid) => uid,
            None => return self.error_response(401, "Authentication required"),
        };

        match self.key_store.list_keys_for_user(&user_id) {
            Ok(keys) => {
                let entries: Vec<serde_json::Value> = keys
                    .iter()
                    .map(|k| {
                        let prefix = if k.key.len() > 8 {
                            format!("{}...{}", &k.key[..7], &k.key[k.key.len() - 4..])
                        } else {
                            k.key.clone()
                        };
                        serde_json::json!({
                            "id": k.id,
                            "key_prefix": prefix,
                            "label": k.label,
                            "tier": k.tier.as_str(),
                            "created_at": k.created_at,
                            "last_used_at": k.last_used_at,
                        })
                    })
                    .collect();
                let resp = serde_json::json!({ "keys": entries, "count": entries.len() });
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                    resp
                )
            }
            Err(e) => self.error_response(500, &format!("Failed to list keys: {}", e)),
        }
    }

    /// DELETE /api/keys/:id — revoke a key.
    fn handle_revoke_key(&self, request: &str, path: &str) -> String {
        let user_id = match self.authenticate_request(request) {
            Some(uid) => uid,
            None => return self.error_response(401, "Authentication required"),
        };

        // Extract key id from path: /api/keys/{id} or /api/v1/keys/{id}
        let key_id = path
            .trim_start_matches("/api/v1/keys/")
            .trim_start_matches("/api/keys/");

        if key_id.is_empty() {
            return self.error_response(400, "Key ID required");
        }

        match self.key_store.revoke_key(key_id, &user_id) {
            Ok(true) => {
                let resp = serde_json::json!({ "revoked": true, "id": key_id });
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                    resp
                )
            }
            Ok(false) => self.error_response(404, "Key not found"),
            Err(e) => self.error_response(500, &format!("Failed to revoke key: {}", e)),
        }
    }

    /// POST /api/keys/:id/rotate — rotate key with 24 h grace period.
    fn handle_rotate_key(&self, request: &str, path: &str) -> String {
        let user_id = match self.authenticate_request(request) {
            Some(uid) => uid,
            None => return self.error_response(401, "Authentication required"),
        };

        // Extract key id from: /api/keys/{id}/rotate or /api/v1/keys/{id}/rotate
        let trimmed = path
            .trim_start_matches("/api/v1/keys/")
            .trim_start_matches("/api/keys/");
        let key_id = trimmed.trim_end_matches("/rotate");

        if key_id.is_empty() {
            return self.error_response(400, "Key ID required");
        }

        match self.key_store.rotate_key(key_id, &user_id) {
            Ok(Some((new_key, grace_until))) => {
                let resp = serde_json::json!({
                    "id": new_key.id,
                    "new_key": new_key.key,
                    "old_key_valid_until": grace_until,
                    "label": new_key.label,
                    "tier": new_key.tier.as_str(),
                });
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                    resp
                )
            }
            Ok(None) => self.error_response(404, "Key not found or already revoked"),
            Err(e) => self.error_response(500, &format!("Failed to rotate key: {}", e)),
        }
    }

    /// Authenticate a request via JWT Bearer token or X-API-Key header.
    /// Returns the user_id on success, or None if unauthenticated.
    fn authenticate_request(&self, request: &str) -> Option<String> {
        // Try JWT Bearer first
        if let Some(auth_header) = self.extract_header(request, "authorization") {
            if let Some(token) = auth_header.trim().strip_prefix("Bearer ") {
                // Skip if it looks like an admin token (not a JWT)
                if token.contains('.') {
                    if let Ok(claims) = auth::verify_jwt(token.trim()) {
                        return Some(claims.sub);
                    }
                }
            }
        }
        // Fall back to X-API-Key
        if let Some(api_key) = self.extract_header(request, "x-api-key") {
            let api_key = api_key.trim().to_string();
            if let Ok(Some(user_id)) = self.key_store.user_id_for_key(&api_key) {
                // Record last-used timestamp
                self.key_store.touch_key(&api_key);
                return Some(user_id);
            }
            // Legacy keys without user_id — allow through with synthetic id
            if let Ok(Some(_)) = self.key_store.get_key(&api_key) {
                self.key_store.touch_key(&api_key);
                return Some(api_key);
            }
        }
        None
    }

    /// Extract the effective API key from the request — resolves both X-API-Key
    /// and JWT (returning the user_id as the billing identity).
    fn resolve_api_identity(&self, request: &str) -> String {
        if let Some(api_key) = self.extract_header(request, "x-api-key") {
            return api_key.trim().to_string();
        }
        if let Some(auth_header) = self.extract_header(request, "authorization") {
            if let Some(token) = auth_header.trim().strip_prefix("Bearer ") {
                if token.contains('.') {
                    if let Ok(claims) = auth::verify_jwt(token.trim()) {
                        return claims.sub;
                    }
                }
            }
        }
        "anonymous".to_string()
    }

    /// Returns true when the request carries a valid admin bearer token.
    fn verify_admin_token(&self, request: &str) -> bool {
        let secret = std::env::var("ADMIN_TOKEN")
            .unwrap_or_else(|_| "srgan-admin".to_string());
        if let Some(auth) = self.extract_header(request, "authorization") {
            if let Some(token) = auth.trim().strip_prefix("Bearer ") {
                return token.trim() == secret;
            }
        }
        false
    }

    /// GET /api/admin/users — list all users with tier, credits, total jobs, last active
    fn handle_admin_users(&self, request: &str) -> String {
        if !self.verify_admin_token(request) {
            return format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n\r\n{}",
                serde_json::json!({"error": "Unauthorized"})
            );
        }

        let users: Vec<serde_json::Value> = if let Ok(db) = self.billing_db.lock() {
            db.all_users_extended()
        } else {
            vec![]
        };

        let response = serde_json::json!({ "users": users, "count": users.len() });
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            response
        )
    }

    /// GET /api/admin/stats — admin analytics (requires ADMIN_TOKEN bearer)
    fn handle_admin_stats(&self, request: &str) -> String {
        if !self.verify_admin_token(request) {
            return format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n\r\n{}",
                serde_json::json!({"error": "Unauthorized"})
            );
        }

        let uptime_secs = SystemTime::now()
            .duration_since(self.server_start_time)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Jobs by status
        let (pending, processing, completed, failed, total_processing_us, completed_count) =
            if let Ok(jobs) = self.jobs.lock() {
                let mut pe = 0u64; let mut pr = 0u64; let mut co = 0u64; let mut fa = 0u64;
                let mut total_us: u64 = 0;
                let mut cnt: u64 = 0;
                for job in jobs.values() {
                    match &job.status {
                        JobStatus::Pending    => pe += 1,
                        JobStatus::Processing => pr += 1,
                        JobStatus::Completed  => {
                            co += 1;
                            let dur = job.updated_at.saturating_sub(job.created_at);
                            total_us += dur * 1_000_000;
                            cnt += 1;
                        }
                        JobStatus::Failed(_)  => fa += 1,
                    }
                }
                (pe, pr, co, fa, total_us, cnt)
            } else {
                (0, 0, 0, 0, 0, 0)
            };

        let avg_processing_ms = if completed_count > 0 {
            (total_processing_us / completed_count) as f64 / 1000.0
        } else {
            0.0
        };

        // Credit usage by tier and top users
        let (credit_by_tier, top_users) = if let Ok(db) = self.billing_db.lock() {
            (db.credits_by_tier(), db.top_users_by_jobs(10))
        } else {
            (serde_json::json!({}), vec![])
        };

        // Request metrics
        let endpoint_metrics = if let Ok(m) = self.request_metrics.lock() {
            m.snapshot()
        } else {
            vec![]
        };

        let total_jobs = pending + processing + completed + failed;

        let response = serde_json::json!({
            "uptime_secs": uptime_secs,
            "total_jobs": total_jobs,
            "jobs_by_status": {
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
            },
            "avg_processing_time_ms": avg_processing_ms,
            "credit_usage_by_tier": credit_by_tier,
            "top_users": top_users,
            "endpoint_metrics": endpoint_metrics,
        });

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            response
        )
    }

    /// Normalise a request path for metrics grouping (collapse IDs to `{id}`).
    fn normalise_metrics_path(method: &str, path: &str) -> String {
        let normalised = if path.starts_with("/api/v1/job/") || path.starts_with("/api/job/") {
            if path.ends_with("/stream") {
                format!("{} /api/v1/job/{{id}}/stream", method)
            } else if path.ends_with("/webhook") {
                format!("{} /api/v1/job/{{id}}/webhook", method)
            } else {
                format!("{} /api/v1/job/{{id}}", method)
            }
        } else if path.starts_with("/api/v1/batch/") || path.starts_with("/api/batch/") {
            if path.ends_with("/checkpoint") {
                format!("{} /api/v1/batch/{{id}}/checkpoint", method)
            } else {
                format!("{} /api/v1/batch/{{id}}", method)
            }
        } else if path.starts_with("/api/v1/result/") || path.starts_with("/api/result/") {
            format!("{} /api/v1/result/{{id}}", method)
        } else {
            format!("{} {}", method, path)
        };
        normalised
    }

    // ── Organization endpoints ────────────────────────────────────────────────

    /// GET /api/orgs/:id — get org details. Requires membership.
    fn handle_get_org(&self, request: &str, path: &str) -> String {
        let api_key = self
            .extract_header(request, "x-api-key")
            .unwrap_or_default();
        let org_id = Self::extract_path_segment(path, "orgs");
        if org_id.is_empty() {
            return self.error_response(400, "Missing org ID");
        }

        match self.org_db.lock() {
            Ok(db) => {
                if !db.is_member(&org_id, &api_key) {
                    return self.error_response(403, "Not a member of this organization");
                }
                match db.get_org(&org_id) {
                    Some(org) => {
                        let members = db.list_members(&org_id);
                        let json = serde_json::json!({
                            "id": org.id,
                            "name": org.name,
                            "owner_user_id": org.owner_user_id,
                            "created_at": org.created_at,
                            "credit_pool": org.credit_pool,
                            "members": members,
                        });
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                            json
                        )
                    }
                    None => self.error_response(404, "Organization not found"),
                }
            }
            Err(_) => self.error_response(500, "Internal error"),
        }
    }

    /// POST /api/orgs/:id/members — add a member.
    fn handle_org_add_member(&self, request: &str, path: &str) -> String {
        let api_key = self
            .extract_header(request, "x-api-key")
            .unwrap_or_default();
        let org_id = Self::extract_path_segment(path, "orgs");
        if org_id.is_empty() {
            return self.error_response(400, "Missing org ID");
        }

        let body = self.extract_body(request);
        let req: crate::api::org::AddMemberRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid request: {}", e)),
        };

        match self.org_db.lock() {
            Ok(mut db) => match db.add_member(&org_id, &api_key, req.user_id) {
                Ok(()) => format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                    serde_json::json!({"ok": true})
                ),
                Err(e) => self.error_response(400, &e),
            },
            Err(_) => self.error_response(500, "Internal error"),
        }
    }

    /// DELETE /api/orgs/:id/members/:user_id — remove a member.
    fn handle_org_remove_member(&self, request: &str, path: &str) -> String {
        let api_key = self
            .extract_header(request, "x-api-key")
            .unwrap_or_default();

        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let (org_id, target_user_id) = if let Some(orgs_idx) = segments.iter().position(|&s| s == "orgs") {
            if orgs_idx + 3 < segments.len() && segments[orgs_idx + 2] == "members" {
                (segments[orgs_idx + 1].to_string(), segments[orgs_idx + 3].to_string())
            } else {
                return self.error_response(400, "Invalid path");
            }
        } else {
            return self.error_response(400, "Invalid path");
        };

        match self.org_db.lock() {
            Ok(mut db) => match db.remove_member(&org_id, &api_key, &target_user_id) {
                Ok(()) => format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                    serde_json::json!({"ok": true})
                ),
                Err(e) => self.error_response(400, &e),
            },
            Err(_) => self.error_response(500, "Internal error"),
        }
    }

    /// GET /api/orgs/:id/usage — credit usage breakdown by member this month.
    fn handle_org_usage(&self, request: &str, path: &str) -> String {
        let api_key = self
            .extract_header(request, "x-api-key")
            .unwrap_or_default();
        let base = path.trim_end_matches("/usage");
        let org_id = Self::extract_path_segment(base, "orgs");
        if org_id.is_empty() {
            return self.error_response(400, "Missing org ID");
        }

        match self.org_db.lock() {
            Ok(db) => {
                if !db.is_member(&org_id, &api_key) {
                    return self.error_response(403, "Not a member of this organization");
                }
                match db.usage_this_month(&org_id) {
                    Some(usage) => {
                        let json = serde_json::to_string(&usage).unwrap_or_default();
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                            json
                        )
                    }
                    None => self.error_response(404, "Organization not found"),
                }
            }
            Err(_) => self.error_response(500, "Internal error"),
        }
    }

    /// Extract the segment after a named path component.
    fn extract_path_segment(path: &str, key: &str) -> String {
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if let Some(idx) = segments.iter().position(|&s| s == key) {
            if idx + 1 < segments.len() {
                return segments[idx + 1].to_string();
            }
        }
        String::new()
    }

    fn handle_not_found(&self) -> String {
        self.error_response(404, "Not found")
    }

    /// Generate error response
    fn error_response(&self, status: u16, message: &str) -> String {
        let status_text = match status {
            400 => "Bad Request",
            401 => "Unauthorized",
            402 => "Payment Required",
            403 => "Forbidden",
            404 => "Not Found",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            _ => "Error",
        };
        
        let response = serde_json::json!({
            "error": message,
            "status": status,
        });
        
        format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\n\r\n{}",
            status, status_text, response
        )
    }
    
    /// Extract body from HTTP request
    fn extract_body(&self, request: &str) -> String {
        if let Some(idx) = request.find("\r\n\r\n") {
            request[idx + 4..].into()
        } else {
            String::new()
        }
    }
    
    /// Resolve the effective model label from an `UpscaleRequest`.
    ///
    /// When `model` is `"waifu2x"`, the `waifu2x_noise_level`,
    /// `waifu2x_scale`, and `waifu2x_style` fields are used to build the
    /// canonical label (e.g. `"waifu2x-noise1-scale2"`).  The style is
    /// not encoded in the label — it is passed separately to the
    /// `Waifu2xNetwork` via [`resolve_waifu2x_style`].
    fn resolve_model_label(request: &UpscaleRequest) -> String {
        match request.model.as_deref() {
            Some("waifu2x") => {
                let noise = request.waifu2x_noise_level.unwrap_or(1).min(3);
                let scale = match request.waifu2x_scale.unwrap_or(2) {
                    1 => 1u8,
                    _ => 2u8,
                };
                format!("waifu2x-noise{}-scale{}", noise, scale)
            }
            Some(label) => label.to_string(),
            None => "natural".to_string(),
        }
    }

    /// Resolve the waifu2x style from the request, defaulting to `Anime`.
    fn resolve_waifu2x_style(request: &UpscaleRequest) -> crate::config::Waifu2xStyle {
        request.waifu2x_style.as_deref()
            .and_then(|s| crate::config::Waifu2xStyle::from_str(s).ok())
            .unwrap_or_default()
    }

    /// POST /api/v1/detect — classify image type without upscaling.
    fn handle_detect(&self, request: &str) -> String {
        let body = self.extract_body(request);

        let req: DetectRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid request: {}", e)),
        };

        let image_data = match general_purpose::STANDARD.decode(&req.image_data) {
            Ok(d) => d,
            Err(e) => return self.error_response(400, &format!("Invalid base64: {}", e)),
        };

        let img = match image::load_from_memory(&image_data) {
            Ok(img) => img,
            Err(e) => return self.error_response(400, &format!("Invalid image: {}", e)),
        };

        let image_type = crate::detection::detect_image_type(&img);
        let response = DetectResponse {
            detected_type: image_type.as_str().to_string(),
            recommended_model: crate::detection::recommended_model_for(&image_type).to_string(),
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            serde_json::to_string(&response)
                .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
        )
    }

    /// Process image upscaling from a pre-decoded DynamicImage
    fn process_image_decoded(
        &self,
        request: UpscaleRequest,
        img: image::DynamicImage,
    ) -> std::result::Result<UpscaleResponse, SrganError> {
        let start_time = SystemTime::now();

        // ── Validate output options early ────────────────────────────────
        let output_cfg = request.output.clone().unwrap_or_default();
        output_cfg.validate()?;

        // ── Run optional preprocessing pipeline ─────────────────────────
        let pipeline_cfg = request.preprocessing.clone().unwrap_or_default();
        let img = crate::image_pipeline::run_pipeline(img, &pipeline_cfg)?;

        let original_size = (img.width(), img.height());
        let network_factor = self.network.factor();
        let requested_factor = request.scale_factor.unwrap_or(network_factor);

        // ── Content-aware auto-enhance shortcut ─────────────────────────
        if request.auto_enhance {
            log::info!("Auto-enhance enabled via API — running content-aware upscaling");
            let upscaled = crate::auto_enhance::auto_enhance_upscale(
                &img,
                network_factor as usize,
            )?;
            let upscaled = crate::output_options::apply_output_scale(
                upscaled,
                original_size.0,
                original_size.1,
                output_cfg.scale,
            );
            let upscaled_size = (upscaled.width(), upscaled.height());
            let format = if request.output.is_some() {
                output_cfg.effective_format().to_string()
            } else {
                request.format.as_deref().unwrap_or("png").to_string()
            };
            let quality = if request.output.is_some() {
                output_cfg.quality
            } else {
                request.quality.unwrap_or(85)
            };
            let img_format = match format.as_str() {
                "jpeg" | "jpg" => ImageFormat::JPEG,
                _ => ImageFormat::PNG,
            };
            let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
            if format == "jpeg" {
                let rgb = upscaled.to_rgb();
                let (w, h) = rgb.dimensions();
                image::jpeg::JPEGEncoder::new_with_quality(&mut cursor, quality)
                    .encode(rgb.as_ref(), w, h, image::ColorType::RGB(8))
                    .map_err(SrganError::Io)?;
            } else {
                upscaled.write_to(&mut cursor, img_format)
                    .map_err(|e| SrganError::Image(e))?;
            }
            let encoded = general_purpose::STANDARD.encode(cursor.into_inner());
            let processing_time = start_time.elapsed()
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            self.images_processed.fetch_add(1, Ordering::Relaxed);
            return Ok(UpscaleResponse {
                success: true,
                image_data: Some(encoded),
                s3_url: None,
                error: None,
                metadata: ResponseMetadata {
                    original_size,
                    upscaled_size,
                    processing_time_ms: processing_time,
                    format,
                    model_used: "auto-enhance".to_string(),
                },
            });
        }

        // When auto_detect is enabled and no model is explicitly specified,
        // detect the image type and pick the recommended model label.
        let effective_label = if request.auto_detect && request.model.is_none() {
            let image_type = crate::detection::detect_image_type(&img);
            let label = crate::detection::recommended_model_for(&image_type);
            log::info!("Auto-detected image type: {} → model: {}", image_type, label);
            label.to_string()
        } else {
            Self::resolve_model_label(&request)
        };

        if requested_factor != network_factor {
            warn!(
                "scale_factor={} requested but model supports {}x; using native factor",
                requested_factor, network_factor
            );
        }

        // Determine whether to use tiled processing.
        const LARGE_IMAGE_THRESHOLD: usize = 4_000_000;
        const DEFAULT_TILE_SIZE: usize = 512;
        let pixel_count = (img.width() as usize) * (img.height() as usize);
        let tile_size = request.tile_size.map(|s| s.max(64)).unwrap_or(DEFAULT_TILE_SIZE);
        let use_tiling = pixel_count > LARGE_IMAGE_THRESHOLD || request.tile_size.is_some();

        // Select the inference path for the effective label.
        //
        // Waifu2x labels use the Waifu2xNetwork which automatically selects
        // between VGG7 CNN inference (when weights are available) and the
        // compat software fallback (Lanczos3 + unsharp mask).
        // All other labels go through ThreadSafeNetwork / the default net.
        let upscaled = if effective_label == "waifu2x"
            || effective_label.starts_with("waifu2x-")
        {
            let style = Self::resolve_waifu2x_style(&request);
            let waifu = crate::waifu2x::Waifu2xNetwork::from_label_with_style(
                &effective_label, style,
            )?;
            log::info!("Using {}", waifu.description());
            waifu.upscale_image(&img)?
        } else if effective_label == "natural"
            || effective_label == "bilinear"
            || effective_label.is_empty()
        {
            if use_tiling {
                self.network.upscale_image_tiled(&img, tile_size)?
            } else {
                self.network.upscale_image(&img)?
            }
        } else {
            match crate::thread_safe_network::ThreadSafeNetwork::from_label(
                &effective_label,
                None,
            ) {
                Ok(net) => {
                    if use_tiling {
                        net.upscale_image_tiled(&img, tile_size)?
                    } else {
                        net.upscale_image(&img)?
                    }
                }
                Err(_) => {
                    if use_tiling {
                        self.network.upscale_image_tiled(&img, tile_size)?
                    } else {
                        self.network.upscale_image(&img)?
                    }
                }
            }
        };
        // ── Apply output scale (e.g. 2× or 3× from the 4× result) ──────
        let upscaled = crate::output_options::apply_output_scale(
            upscaled,
            original_size.0,
            original_size.1,
            output_cfg.scale,
        );
        let upscaled_size = (upscaled.width(), upscaled.height());

        // Encode result to the requested format (prefer output config, fall back to legacy field)
        let format = if request.output.is_some() {
            output_cfg.effective_format().to_string()
        } else {
            request.format.as_deref().unwrap_or("png").to_string()
        };
        let quality = if request.output.is_some() {
            output_cfg.quality
        } else {
            request.quality.unwrap_or(85)
        };
        let img_format = match format.as_str() {
            "jpeg" | "jpg" => ImageFormat::JPEG,
            _ => ImageFormat::PNG,
        };
        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
        if format == "jpeg" {
            let rgb = upscaled.to_rgb();
            let (w, h) = rgb.dimensions();
            image::jpeg::JPEGEncoder::new_with_quality(&mut cursor, quality)
                .encode(rgb.as_ref(), w, h, image::ColorType::RGB(8))
                .map_err(SrganError::Io)?;
        } else {
            upscaled.write_to(&mut cursor, img_format)
                .map_err(|e| SrganError::Image(e))?;
        }
        let encoded = general_purpose::STANDARD.encode(cursor.into_inner());

        let processing_time = start_time.elapsed()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.images_processed.fetch_add(1, Ordering::Relaxed);

        // Try to upload to S3 if configured; fall back to base64 on failure
        let (image_data_field, s3_url_field) = if let Some(ref cfg) = self.s3_config {
            let key = format!("results/{}.png", self.generate_job_id());
            let raw = general_purpose::STANDARD.decode(&encoded).unwrap_or_default();
            match crate::storage::upload_result(cfg, &key, &raw, "image/png") {
                Ok(presigned) => (None, Some(presigned)),
                Err(e) => {
                    warn!("S3 upload failed, falling back to base64: {}", e);
                    (Some(encoded), None)
                }
            }
        } else {
            (Some(encoded), None)
        };

        Ok(UpscaleResponse {
            success: true,
            image_data: image_data_field,
            s3_url: s3_url_field,
            error: None,
            metadata: ResponseMetadata {
                original_size,
                upscaled_size,
                processing_time_ms: processing_time,
                format: format.clone(),
                model_used: std::format!("{}_{}x", effective_label, network_factor),
            },
        })
    }

    /// Queue a large image as a background async job and return 202 with job_id
    fn queue_async_job_from_data(&self, req: UpscaleRequest, _image_data: Vec<u8>) -> String {
        // Delegate to the regular async handler by reconstructing the request body
        // (image_data is already embedded in req.image_data as base64)
        let body = match serde_json::to_string(&req) {
            Ok(b) => b,
            Err(e) => return self.error_response(500, &format!("Serialization error: {}", e)),
        };
        // Wrap in a minimal HTTP request fragment so handle_upscale_async can extract it
        let synthetic_request = format!("\r\n\r\n{}", body);
        self.handle_upscale_async(&synthetic_request)
    }

    /// Return the stored result data for a completed job
    fn handle_job_result(&self, path: &str) -> String {
        let job_id = path.trim_start_matches("/api/result/");

        if let Ok(jobs) = self.jobs.lock() {
            if let Some(job) = jobs.get(job_id) {
                match &job.result_data {
                    Some(data) => {
                        let response = serde_json::json!({
                            "success": true,
                            "job_id": job_id,
                            "image_data": data,
                        });
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                            response
                        )
                    }
                    None => {
                        // Job exists but result not yet available
                        self.error_response(404, "Result not yet available")
                    }
                }
            } else {
                self.error_response(404, "Job not found")
            }
        } else {
            self.error_response(500, "Failed to acquire job lock")
        }
    }
    
    /// GET /api/preview/:job_id — returns a downscaled PNG thumbnail (max 300px) of the
    /// upscaled result for the given job.
    fn handle_preview(&self, path: &str) -> String {
        let job_id = path
            .trim_start_matches("/api/v1/preview/")
            .trim_start_matches("/api/preview/");

        if let Ok(jobs) = self.jobs.lock() {
            if let Some(job) = jobs.get(job_id) {
                match &job.result_data {
                    Some(data) => {
                        let decoded = match general_purpose::STANDARD.decode(data) {
                            Ok(d) => d,
                            Err(e) => return self.error_response(500, &format!("Base64 decode error: {}", e)),
                        };
                        let img = match image::load_from_memory(&decoded) {
                            Ok(i) => i,
                            Err(e) => return self.error_response(500, &format!("Image decode error: {}", e)),
                        };

                        // Resize to fit within 300x300, preserving aspect ratio
                        let (w, h) = (img.width(), img.height());
                        let max_dim = 300u32;
                        let (tw, th) = if w >= h {
                            (max_dim, (max_dim as f64 * h as f64 / w as f64).round() as u32)
                        } else {
                            ((max_dim as f64 * w as f64 / h as f64).round() as u32, max_dim)
                        };
                        let thumbnail = img.resize_exact(tw.max(1), th.max(1), image::FilterType::Lanczos3);

                        let mut buf = std::io::Cursor::new(Vec::new());
                        if let Err(e) = thumbnail.write_to(&mut buf, ImageFormat::PNG) {
                            return self.error_response(500, &format!("PNG encode error: {}", e));
                        }
                        let png_bytes = buf.into_inner();
                        let b64 = general_purpose::STANDARD.encode(&png_bytes);
                        let response = serde_json::json!({
                            "success": true,
                            "job_id": job_id,
                            "preview_image": b64,
                            "width": tw,
                            "height": th,
                        });
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                            response
                        )
                    }
                    None => self.error_response(404, "Result not yet available"),
                }
            } else {
                self.error_response(404, "Job not found")
            }
        } else {
            self.error_response(500, "Failed to acquire job lock")
        }
    }

    /// GET /api/jobs/:job_id/compare — returns a side-by-side comparison image
    /// (original left, upscaled right, same height) as PNG.
    fn handle_job_compare(&self, path: &str) -> String {
        let inner = path
            .trim_start_matches("/api/v1/jobs/")
            .trim_start_matches("/api/jobs/");
        let job_id = inner.trim_end_matches("/compare");

        if let Ok(jobs) = self.jobs.lock() {
            if let Some(job) = jobs.get(job_id) {
                let result_b64 = match &job.result_data {
                    Some(d) => d.clone(),
                    None => return self.error_response(404, "Result not yet available"),
                };
                let input_b64 = match &job.input_data {
                    Some(d) => d.clone(),
                    None => return self.error_response(404, "Original input not available for this job"),
                };

                // Decode both images
                let input_bytes = match general_purpose::STANDARD.decode(&input_b64) {
                    Ok(b) => b,
                    Err(e) => return self.error_response(500, &format!("Input decode error: {}", e)),
                };
                let result_bytes = match general_purpose::STANDARD.decode(&result_b64) {
                    Ok(b) => b,
                    Err(e) => return self.error_response(500, &format!("Result decode error: {}", e)),
                };

                let input_img = match image::load_from_memory(&input_bytes) {
                    Ok(i) => i,
                    Err(e) => return self.error_response(500, &format!("Input image error: {}", e)),
                };
                let result_img = match image::load_from_memory(&result_bytes) {
                    Ok(i) => i,
                    Err(e) => return self.error_response(500, &format!("Result image error: {}", e)),
                };

                // Scale both to the same height (use the upscaled image's height)
                let target_h = result_img.height();
                let left = if input_img.height() != target_h {
                    let scale = target_h as f64 / input_img.height() as f64;
                    let new_w = (input_img.width() as f64 * scale).round() as u32;
                    input_img.resize_exact(new_w.max(1), target_h, image::FilterType::Lanczos3)
                } else {
                    input_img
                };

                // Create side-by-side canvas
                let total_w = left.width() + result_img.width();
                let mut canvas = image::DynamicImage::new_rgb8(total_w, target_h);

                // Copy left (original)
                for y in 0..target_h {
                    for x in 0..left.width() {
                        canvas.put_pixel(x, y, left.get_pixel(x, y));
                    }
                }
                // Copy right (upscaled)
                let offset_x = left.width();
                for y in 0..target_h {
                    for x in 0..result_img.width() {
                        canvas.put_pixel(offset_x + x, y, result_img.get_pixel(x, y));
                    }
                }

                let mut buf = std::io::Cursor::new(Vec::new());
                if let Err(e) = canvas.write_to(&mut buf, ImageFormat::PNG) {
                    return self.error_response(500, &format!("PNG encode error: {}", e));
                }
                let png_bytes = buf.into_inner();
                let b64 = general_purpose::STANDARD.encode(&png_bytes);
                let response = serde_json::json!({
                    "success": true,
                    "job_id": job_id,
                    "compare_image": b64,
                    "left": "original",
                    "right": "upscaled",
                    "width": total_w,
                    "height": target_h,
                });
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                    response
                )
            } else {
                self.error_response(404, "Job not found")
            }
        } else {
            self.error_response(500, "Failed to acquire job lock")
        }
    }

    /// Check rate limit
    fn check_rate_limit(&self, client_id: &str) -> bool {
        if self.config.rate_limit.is_none() {
            return true;
        }
        
        self.rate_limiter.lock()
            .map(|mut limiter| limiter.check_rate_limit(client_id))
            .unwrap_or(true)
    }
    
    /// Generate unique job ID
    fn generate_job_id(&self) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_else(|_| {
                use std::sync::atomic::{AtomicU64, Ordering};
                static COUNTER: AtomicU64 = AtomicU64::new(0);
                COUNTER.fetch_add(1, Ordering::SeqCst) as u128
            });
        
        format!("job_{}", timestamp)
    }
    
    // ── Batch endpoints ──────────────────────────────────────────────────────

    /// POST /api/v1/batch — sync (≤10 images) or async (>10)
    fn handle_batch(&self, request: &str) -> String {
        let body = self.extract_body(request);

        let req: BatchUpscaleRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid request: {}", e)),
        };

        if req.images.is_empty() {
            return self.error_response(400, "No images provided");
        }

        if !self.check_rate_limit("batch_client") {
            return self.error_response(429, "Rate limit exceeded");
        }

        if req.images.len() > crate::api::batch::MAX_BATCH_SIZE {
            return self.error_response(400, &format!(
                "Batch size {} exceeds maximum of {} images",
                req.images.len(),
                crate::api::batch::MAX_BATCH_SIZE,
            ));
        }

        if req.images.len() > 10 {
            return self.submit_async_batch(req);
        }

        // Synchronous path
        let start = SystemTime::now();
        let format = req.format.as_deref().unwrap_or("png");
        let img_format = match format {
            "jpeg" | "jpg" => ImageFormat::JPEG,
            _ => ImageFormat::PNG,
        };

        let mut results = Vec::new();
        let mut successful = 0usize;
        let mut failed = 0usize;

        for (i, img_b64) in req.images.iter().enumerate() {
            let r = self.process_one_image(i, img_b64, img_format);
            if r.success { successful += 1; } else { failed += 1; }
            results.push(r);
        }

        let total_time = start.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);

        let response = BatchUpscaleResponse {
            success: failed == 0,
            results,
            total: req.images.len(),
            successful,
            failed,
            total_processing_time_ms: total_time,
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            serde_json::to_string(&response).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
        )
    }

    /// Process one image; used by both sync and async batch paths
    fn process_one_image(&self, index: usize, img_b64: &str, img_format: ImageFormat) -> BatchImageResult {
        let start = SystemTime::now();

        let result: std::result::Result<String, String> = (|| {
            let image_data = general_purpose::STANDARD.decode(img_b64)
                .map_err(|e| format!("Invalid base64: {}", e))?;

            if image_data.len() > self.config.max_file_size {
                return Err("Image exceeds max file size".to_string());
            }

            let img = image::load_from_memory(&image_data)
                .map_err(|e| format!("Invalid image: {}", e))?;

            let upscaled = self.network.upscale_image(&img)
                .map_err(|e| format!("Upscaling failed: {}", e))?;

            let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
            upscaled.write_to(&mut cursor, img_format)
                .map_err(|e| format!("Encode failed: {}", e))?;

            Ok(general_purpose::STANDARD.encode(cursor.into_inner()))
        })();

        let ms = start.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
        match result {
            Ok(encoded) => BatchImageResult { index, success: true, image_data: Some(encoded), error: None, processing_time_ms: ms },
            Err(e)      => BatchImageResult { index, success: false, image_data: None, error: Some(e), processing_time_ms: ms },
        }
    }

    /// Queue a large batch for background processing; return 202 with batch_id
    fn submit_async_batch(&self, req: BatchUpscaleRequest) -> String {
        let batch_id = format!("batch_{}", SystemTime::now()
            .duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));

        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let total = req.images.len();

        let job = BatchJobInfo {
            batch_id: batch_id.clone(),
            status: BatchJobStatus::Processing { completed: 0, total },
            results: Vec::new(),
            total,
            created_at: now,
            updated_at: now,
        };

        if let Ok(mut jobs) = self.batch_jobs.lock() {
            jobs.insert(batch_id.clone(), job);
        } else {
            return self.error_response(500, "Failed to acquire batch lock");
        }

        let batch_jobs = Arc::clone(&self.batch_jobs);
        let network    = Arc::clone(&self.network);
        let max_size   = self.config.max_file_size;
        let bid        = batch_id.clone();

        thread::spawn(move || {
            use std::sync::atomic::{AtomicUsize, Ordering};

            let fmt = req.format.as_deref().unwrap_or("png");
            let img_format = match fmt {
                "jpeg" | "jpg" => ImageFormat::JPEG,
                _ => ImageFormat::PNG,
            };

            let total = req.images.len();
            let done_counter = Arc::new(AtomicUsize::new(0));
            let results_lock: Arc<Mutex<Vec<BatchImageResult>>> =
                Arc::new(Mutex::new(Vec::with_capacity(total)));

            // Process images in parallel using rayon
            let handles: Vec<_> = req.images.iter().enumerate().map(|(i, img_b64)| {
                let network = Arc::clone(&network);
                let batch_jobs = Arc::clone(&batch_jobs);
                let bid = bid.clone();
                let done_counter = Arc::clone(&done_counter);
                let results_lock = Arc::clone(&results_lock);
                let img_b64 = img_b64.clone();

                thread::spawn(move || {
                    let start = SystemTime::now();

                    let result: std::result::Result<String, String> = (|| {
                        let data = general_purpose::STANDARD.decode(&img_b64)
                            .map_err(|e| format!("Invalid base64: {}", e))?;
                        if data.len() > max_size {
                            return Err("Image exceeds max file size".to_string());
                        }
                        let img = image::load_from_memory(&data)
                            .map_err(|e| format!("Invalid image: {}", e))?;
                        let up = network.upscale_image(&img)
                            .map_err(|e| format!("Upscaling failed: {}", e))?;
                        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
                        up.write_to(&mut cursor, img_format)
                            .map_err(|e| format!("Encode failed: {}", e))?;
                        Ok(general_purpose::STANDARD.encode(cursor.into_inner()))
                    })();

                    let ms = start.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
                    let batch_result = match result {
                        Ok(enc) => BatchImageResult { index: i, success: true, image_data: Some(enc), error: None, processing_time_ms: ms },
                        Err(e)  => BatchImageResult { index: i, success: false, image_data: None, error: Some(e), processing_time_ms: ms },
                    };

                    if let Ok(mut r) = results_lock.lock() {
                        r.push(batch_result);
                    }

                    let completed_so_far = done_counter.fetch_add(1, Ordering::SeqCst) + 1;
                    let progress_pct = ((completed_so_far as f64 / total as f64) * 100.0).round() as usize;

                    // Update progress
                    if let Ok(mut jobs) = batch_jobs.lock() {
                        if let Some(job) = jobs.get_mut(&bid) {
                            job.status = BatchJobStatus::Processing { completed: completed_so_far, total };
                            if let Ok(r) = results_lock.lock() {
                                job.results = r.clone();
                            }
                            job.updated_at = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                        }
                    }

                    let _ = progress_pct; // used via batch_jobs above
                })
            }).collect();

            // Wait for all spawned threads
            for h in handles {
                let _ = h.join();
            }

            // Mark done
            if let Ok(mut jobs) = batch_jobs.lock() {
                if let Some(job) = jobs.get_mut(&bid) {
                    job.status = BatchJobStatus::Completed;
                    if let Ok(r) = results_lock.lock() {
                        let mut final_results = r.clone();
                        final_results.sort_by_key(|r| r.index);
                        job.results = final_results;
                    }
                    job.updated_at = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                }
            }
        });

        let response = serde_json::json!({
            "batch_id": batch_id,
            "status": "processing",
            "total": total,
            "poll_url": format!("/api/v1/batch/{}", batch_id),
        });

        format!(
            "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\n\r\n{}",
            response
        )
    }

    /// GET /api/v1/batch/{id}
    fn handle_batch_status(&self, path: &str) -> String {
        let batch_id = path
            .trim_start_matches("/api/v1/batch/")
            .trim_start_matches("/api/batch/");

        // Check in-memory jobs first (base64-batch async jobs).
        if let Ok(jobs) = self.batch_jobs.lock() {
            if let Some(job) = jobs.get(batch_id) {
                return format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                    serde_json::to_string(job).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
                );
            }
        }

        // Fall back to on-disk checkpoint (directory-based batch jobs).
        let cwd = std::env::current_dir().unwrap_or_default();
        match crate::checkpoint::load_checkpoint(batch_id, &cwd) {
            Ok(Some(cp)) => {
                let completed = cp.completed_images.len();
                let total = cp.total_images;
                let failed = cp.failed_images.len();
                let remaining = total.saturating_sub(completed + failed);
                let response = serde_json::json!({
                    "batch_id": cp.batch_id,
                    "total_images": total,
                    "completed": completed,
                    "failed": failed,
                    "remaining": remaining,
                    "input_dir": cp.input_dir,
                    "output_dir": cp.output_dir,
                    "started_at": cp.started_at,
                    "last_updated": cp.last_updated,
                });
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                    response
                )
            }
            Ok(None) => self.error_response(404, "Batch job not found"),
            Err(e) => self.error_response(500, &format!("Checkpoint read error: {}", e)),
        }
    }

    /// GET /api/v1/batch/{id}/checkpoint — query the persisted checkpoint for a CLI batch run.
    ///
    /// Looks for `.srgan_checkpoint_<id>.json` in the current working directory.
    fn handle_batch_checkpoint(&self, path: &str) -> String {
        // Strip prefix and the trailing "/checkpoint" suffix to get the batch ID.
        let without_prefix = path
            .trim_start_matches("/api/v1/batch/")
            .trim_start_matches("/api/batch/");
        let batch_id = without_prefix.trim_end_matches("/checkpoint");

        let cwd = match std::env::current_dir() {
            Ok(d) => d,
            Err(e) => return self.error_response(500, &format!("Cannot determine cwd: {}", e)),
        };

        match crate::checkpoint::load_checkpoint(batch_id, &cwd) {
            Ok(Some(cp)) => {
                match serde_json::to_string(&cp) {
                    Ok(json) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                        json
                    ),
                    Err(e) => self.error_response(500, &format!("Serialization error: {}", e)),
                }
            }
            Ok(None) => self.error_response(404, &format!("No checkpoint found for batch '{}'", batch_id)),
            Err(e) => self.error_response(500, &format!("Failed to load checkpoint: {}", e)),
        }
    }

    /// POST /api/v1/batch/start — start a directory-based batch job with checkpoint support.
    fn handle_batch_dir_start(&self, request: &str) -> String {
        let body = self.extract_body(request);
        let req: BatchDirRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid request: {}", e)),
        };

        // Validate and canonicalize directories.
        let input_path = match std::fs::canonicalize(&req.input_dir) {
            Ok(p) => p,
            Err(e) => return self.error_response(400, &format!("Invalid input_dir: {}", e)),
        };
        if let Err(e) = std::fs::create_dir_all(&req.output_dir) {
            return self.error_response(400, &format!("Cannot create output_dir: {}", e));
        }
        let output_path = match std::fs::canonicalize(&req.output_dir) {
            Ok(p) => p,
            Err(e) => return self.error_response(400, &format!("Invalid output_dir: {}", e)),
        };

        let recursive = req.recursive.unwrap_or(false);
        let batch_options = crate::checkpoint::BatchOptions {
            parameters: req.model.clone(),
            custom_model: None,
            factor: req.scale.unwrap_or(4) as usize,
            recursive,
            parallel: true,
            skip_existing: false,
        };

        let batch_id = uuid::Uuid::new_v4().to_string();
        let mut checkpoint = crate::checkpoint::BatchCheckpoint::new(
            batch_id.clone(),
            req.input_dir.clone(),
            req.output_dir.clone(),
            0,
            batch_options,
        );

        let image_files = match crate::commands::batch::collect_image_files(&input_path, "", recursive) {
            Ok(f) => f,
            Err(e) => return self.error_response(500, &format!("Failed to scan input_dir: {}", e)),
        };

        checkpoint.total_images = image_files.len();
        if let Err(e) = crate::checkpoint::save_checkpoint(&mut checkpoint, &output_path) {
            return self.error_response(500, &format!("Failed to save checkpoint: {}", e));
        }

        let total = image_files.len();
        let batch_id_ret = batch_id.clone();
        let network = Arc::clone(&self.network);

        thread::spawn(move || {
            let cp_arc = std::sync::Arc::new(std::sync::Mutex::new(checkpoint));
            for image_file in &image_files {
                let relative = image_file.strip_prefix(&input_path).unwrap_or(image_file);
                let output_file = output_path.join(relative).with_extension("png");
                let output_key = output_file.to_string_lossy().to_string();
                if let Some(parent) = output_file.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match crate::commands::batch::process_image(image_file, &output_file, &*network) {
                    Ok(_) => {
                        if let Ok(mut cp) = cp_arc.lock() {
                            cp.completed_images.push(output_key);
                            let _ = crate::checkpoint::save_checkpoint(&mut cp, &output_path);
                        }
                    }
                    Err(_) => {
                        if let Ok(mut cp) = cp_arc.lock() {
                            cp.failed_images.push(output_key);
                            let _ = crate::checkpoint::save_checkpoint(&mut cp, &output_path);
                        }
                    }
                }
            }
        });

        let response = serde_json::json!({
            "batch_id": batch_id_ret,
            "total_images": total,
            "status": "processing",
            "check_url": format!("/api/v1/batch/{}", batch_id_ret),
        });
        format!(
            "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\n\r\n{}",
            response
        )
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Extract the value of a named HTTP header from a raw request string.
    fn extract_header(&self, request: &str, header_name: &str) -> Option<String> {
        let prefix_lc = format!("{}: ", header_name.to_lowercase());
        for line in request.lines() {
            let line_lc = line.to_lowercase();
            if line_lc.starts_with(&prefix_lc) {
                return Some(line[prefix_lc.len()..].trim().to_string());
            }
        }
        None
    }

    /// Extract a query parameter from the request line (e.g. `GET /path?key=val HTTP/1.1`).
    fn extract_query_param(&self, request: &str, param: &str) -> Option<String> {
        let first_line = request.lines().next()?;
        let url = first_line.split_whitespace().nth(1)?;
        let query = url.split('?').nth(1)?;
        for pair in query.split('&') {
            if let Some((k, v)) = pair.split_once('=') {
                if k == param {
                    return Some(v.to_string());
                }
            }
        }
        None
    }

    /// Resolve the tier for an API key (creates a free account on first use).
    fn tier_for_key(&self, api_key: &str) -> SubscriptionTier {
        if let Ok(mut db) = self.billing_db.lock() {
            db.get_or_create_free(api_key).tier.clone()
        } else {
            SubscriptionTier::Free
        }
    }

    /// Consume a credit for a job. If the user is in an org, deducts from the
    /// org credit pool; otherwise deducts from personal credits. Returns true
    /// if a credit was available and consumed.
    fn consume_credit_for_user(&self, api_key: &str) -> bool {
        // Check if user belongs to an org
        if let Ok(mut org_db) = self.org_db.lock() {
            if let Some(org_id) = org_db.user_org_id(api_key).cloned() {
                return org_db.consume_org_credit(&org_id, api_key);
            }
        }
        // Fall through to personal credits
        if let Ok(mut db) = self.billing_db.lock() {
            return db.consume_credit(api_key);
        }
        false
    }

    /// Rate-limit check using the tier-aware limiter.
    /// Returns the HTTP headers string and whether the request is allowed.
    fn tier_rate_limit_check(&self, request: &str) -> (bool, String) {
        let api_key = self.resolve_api_identity(request);
        let tier = self.tier_for_key(&api_key);
        let result = self.tier_rate_limiter.check(&api_key, &tier);
        let headers = result.headers();
        // Record the request in the per-key rate-limit dashboard
        if let Ok(mut dashboard) = self.rate_limit_dashboard.lock() {
            dashboard.record_request(&api_key, !result.allowed);
        }
        (result.allowed, headers)
    }

    /// Build a 429 response with rate-limit headers.
    fn rate_limit_response(&self, rl_headers: &str, retry_after: u64) -> String {
        let body = serde_json::json!({
            "error": "Rate limit exceeded",
            "retry_after_seconds": retry_after,
        });
        format!(
            "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n{}Retry-After: {}\r\n\r\n{}",
            rl_headers, retry_after, body
        )
    }

    // ── Billing handlers ──────────────────────────────────────────────────────

    /// POST /api/v1/billing/checkout — create Stripe checkout session
    fn handle_billing_checkout(&self, request: &str) -> String {
        let body = self.extract_body(request);
        let checkout_req: CheckoutRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid request: {}", e)),
        };

        let stripe_key = match std::env::var("STRIPE_SECRET_KEY") {
            Ok(k) => k,
            Err(_) => return self.error_response(500, "Stripe not configured (STRIPE_SECRET_KEY missing)"),
        };

        match crate::api::billing::create_checkout_session(&checkout_req, &stripe_key) {
            Ok(session) => format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                serde_json::to_string(&session)
                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
            ),
            Err(e) => self.error_response(502, &format!("Stripe error: {}", e)),
        }
    }

    /// POST /api/v1/billing/webhook — handle Stripe webhook events
    fn handle_billing_webhook(&self, request: &str) -> String {
        let body = self.extract_body(request);
        let signature = self
            .extract_header(request, "stripe-signature")
            .unwrap_or_default();

        let webhook_secret = std::env::var("STRIPE_WEBHOOK_SECRET").unwrap_or_default();

        match crate::api::billing::handle_stripe_webhook(
            &body,
            &webhook_secret,
            &signature,
            &self.billing_db,
        ) {
            Ok(msg) => {
                let response = serde_json::json!({ "received": true, "result": msg });
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                    response
                )
            }
            Err(e) => self.error_response(400, &e),
        }
    }

    /// GET /api/v1/billing/status — subscription tier and credits remaining
    fn handle_billing_status(&self, request: &str) -> String {
        let api_key = self
            .extract_header(request, "x-api-key")
            .unwrap_or_else(|| "anonymous".to_string());

        let status = if let Ok(mut db) = self.billing_db.lock() {
            let account = db.get_or_create_free(&api_key);
            BillingStatus {
                tier: account.tier.as_str().to_string(),
                credits_remaining: account.credits_remaining,
                credits_reset_at: account.credits_reset_at,
            }
        } else {
            return self.error_response(500, "Failed to acquire billing lock");
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            serde_json::to_string(&status)
                .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
        )
    }

    /// POST /webhooks/stripe — Stripe webhook endpoint using billing::stripe module
    fn handle_stripe_webhook_v2(&self, request: &str) -> String {
        let body = self.extract_body(request);
        let signature = self
            .extract_header(request, "stripe-signature")
            .unwrap_or_default();

        let webhook_secret = std::env::var("STRIPE_WEBHOOK_SECRET").unwrap_or_default();
        if webhook_secret.is_empty() {
            return self.error_response(500, "STRIPE_WEBHOOK_SECRET not configured");
        }

        match crate::billing::stripe::handle_webhook(
            &body,
            &signature,
            &webhook_secret,
            &self.billing_db,
        ) {
            Ok(action) => {
                let msg = match &action {
                    crate::billing::stripe::WebhookAction::Provisioned(id) => {
                        // Payment recovered — clear dunning state
                        if let Ok(mut dunning) = self.stripe_dunning.lock() {
                            dunning.on_payment_recovered(id);
                        }
                        if let Ok(mut dash) = self.key_usage_dashboard.lock() {
                            dash.set_payment_failed(id, false);
                        }
                        format!("provisioned:{}", id)
                    }
                    crate::billing::stripe::WebhookAction::Suspended(id) => {
                        // Payment failed — enter dunning, downgrade, mark payment_failed
                        if let Ok(mut dunning) = self.stripe_dunning.lock() {
                            dunning.on_payment_failed(id, id, "", "pro");
                        }
                        if let Ok(mut dash) = self.key_usage_dashboard.lock() {
                            dash.set_payment_failed(id, true);
                        }
                        format!("suspended:{}", id)
                    }
                    crate::billing::stripe::WebhookAction::Revoked(id) => {
                        // Subscription deleted — clear dunning
                        if let Ok(mut dunning) = self.stripe_dunning.lock() {
                            dunning.on_payment_recovered(id);
                        }
                        if let Ok(mut dash) = self.key_usage_dashboard.lock() {
                            dash.set_payment_failed(id, false);
                        }
                        format!("revoked:{}", id)
                    }
                    crate::billing::stripe::WebhookAction::DunningStarted(id) => {
                        if let Ok(mut dunning) = self.stripe_dunning.lock() {
                            dunning.on_payment_failed(id, id, "", "pro");
                        }
                        if let Ok(mut dash) = self.key_usage_dashboard.lock() {
                            dash.set_payment_failed(id, true);
                        }
                        format!("dunning_started:{}", id)
                    }
                    crate::billing::stripe::WebhookAction::Ignored(reason) =>
                        format!("ignored:{}", reason),
                };
                let response = serde_json::json!({ "received": true, "result": msg });
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                    response
                )
            }
            Err(e) => self.error_response(400, &e),
        }
    }

    /// GET /api/v1/admin/keys — admin dashboard showing all API keys with plan, usage, quota, status
    fn handle_admin_keys(&self, request: &str) -> String {
        if !self.verify_admin_token(request) {
            return self.error_response(401, "Unauthorized");
        }

        let dashboard_stats = self.rate_limit_dashboard.lock().ok().and_then(|dash| {
            self.billing_db.lock().ok().map(|db| dash.get_all_stats(&db))
        });

        let keys = if let Ok(db) = self.billing_db.lock() {
            db.all_users_snapshot()
                .into_iter()
                .map(|mut entry| {
                    // Mask the key_id: show first 8 chars + "..."
                    if let Some(uid) = entry["user_id"].as_str().map(|s| s.to_string()) {
                        let masked = if uid.len() > 8 {
                            format!("{}...", &uid[..8])
                        } else {
                            uid.clone()
                        };
                        entry["key_id"] = serde_json::Value::String(masked);

                        // Merge rate-limit dashboard stats for this key
                        if let Some(ref stats) = dashboard_stats {
                            if let Some(rl) = stats.iter().find(|s| s.api_key == uid) {
                                entry["requests_today"] = serde_json::json!(rl.requests_today);
                                entry["quota_daily"] = serde_json::json!(rl.quota_daily);
                                entry["pct_quota_used"] = serde_json::json!((rl.pct_daily * 100.0).round() / 100.0);
                                entry["throttled_today"] = serde_json::json!(rl.throttled_today);
                            }
                        }
                    }
                    entry
                })
                .collect::<Vec<_>>()
        } else {
            return self.error_response(500, "Failed to acquire billing lock");
        };

        let body = serde_json::json!({ "keys": keys });
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            body
        )
    }

    /// GET /api/v1/rate-limit — self-service rate-limit stats for the caller's API key
    fn handle_rate_limit_self(&self, request: &str) -> String {
        let api_key = self.resolve_api_identity(request);

        let stats = self.rate_limit_dashboard.lock().ok().and_then(|dash| {
            self.billing_db.lock().ok().and_then(|db| dash.get_stats(&api_key, &db))
        });

        let body = match stats {
            Some(s) => serde_json::json!({
                "api_key": format!("{}...", &s.api_key[..s.api_key.len().min(8)]),
                "tier": s.tier,
                "requests_today": s.requests_today,
                "quota_daily": s.quota_daily,
                "pct_quota_used": (s.pct_daily * 100.0).round() / 100.0,
                "throttled_today": s.throttled_today,
            }),
            None => serde_json::json!({
                "api_key": format!("{}...", &api_key[..api_key.len().min(8)]),
                "requests_today": 0,
                "quota_daily": 0,
                "pct_quota_used": 0.0,
                "throttled_today": 0,
            }),
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            body
        )
    }

    /// GET /admin/rate-limits — JSON of all API keys with usage, limits, overages
    fn handle_admin_rate_limits(&self, request: &str) -> String {
        if !self.verify_admin_token(request) {
            return self.error_response(401, "Unauthorized");
        }

        let stats = self.rate_limit_dashboard.lock().ok().and_then(|dash| {
            self.billing_db.lock().ok().map(|db| dash.get_all_stats(&db))
        });

        let keys = stats.unwrap_or_default();
        let body = serde_json::json!({ "keys": keys });
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            body
        )
    }

    /// PUT /admin/rate-limits/:key_id — update rate limits for a specific key
    fn handle_update_rate_limit(&self, request: &str, path: &str) -> String {
        if !self.verify_admin_token(request) {
            return self.error_response(401, "Unauthorized");
        }

        let key_id = path.strip_prefix("/admin/rate-limits/").unwrap_or("");
        if key_id.is_empty() {
            return self.error_response(400, "Missing key_id");
        }

        let body = self.extract_body(request);
        let update: crate::api::rate_limit_dashboard::UpdateRateLimitRequest = match serde_json::from_str(&body) {
            Ok(u) => u,
            Err(e) => return self.error_response(400, &format!("Invalid JSON: {}", e)),
        };

        if let Ok(mut dashboard) = self.rate_limit_dashboard.lock() {
            dashboard.update_rate_limits(key_id, &update);
        }

        let resp = serde_json::json!({
            "ok": true,
            "key_id": key_id,
            "rate_limit_rpm": update.rate_limit_rpm,
            "rate_limit_daily": update.rate_limit_daily,
        });
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            resp
        )
    }

    // ── New rate-limit dashboard + dunning handlers ────────────────────────────

    /// GET /admin/api-keys — list all keys with usage stats (JSON)
    fn handle_admin_api_keys_list(&self, request: &str) -> String {
        if !self.verify_admin_token(request) {
            return self.error_response(401, "Unauthorized");
        }

        let body = if let Ok(dash) = self.key_usage_dashboard.lock() {
            crate::rate_limit_dashboard::admin_api_keys_json(&dash)
        } else {
            serde_json::json!({ "keys": [] })
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            body
        )
    }

    /// GET /admin/api-keys/{key_id}/usage — detailed usage for one key (JSON)
    fn handle_admin_key_usage(&self, request: &str, path: &str) -> String {
        if !self.verify_admin_token(request) {
            return self.error_response(401, "Unauthorized");
        }

        // Extract key_id from /admin/api-keys/{key_id}/usage
        let key_id = path
            .strip_prefix("/admin/api-keys/")
            .and_then(|rest| rest.strip_suffix("/usage"))
            .unwrap_or("");

        if key_id.is_empty() {
            return self.error_response(400, "Missing key_id");
        }

        let body = if let Ok(dash) = self.key_usage_dashboard.lock() {
            match crate::rate_limit_dashboard::admin_key_usage_json(&dash, key_id) {
                Some(json) => json,
                None => return self.error_response(404, "API key not found"),
            }
        } else {
            return self.error_response(500, "Failed to acquire dashboard lock");
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            body
        )
    }

    /// GET /admin/rate-limits — HTML admin page with color-coded table
    fn handle_admin_rate_limits_page(&self, request: &str) -> String {
        if !self.verify_admin_token(request) {
            return self.error_response(401, "Unauthorized");
        }

        let html = if let Ok(dash) = self.key_usage_dashboard.lock() {
            crate::rate_limit_dashboard::admin_rate_limits_html(&dash)
        } else {
            "<html><body><h1>Error loading dashboard</h1></body></html>".to_string()
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n{}",
            html
        )
    }

    // ── Dashboard handler ─────────────────────────────────────────────────────

    /// GET /dashboard — HTML health dashboard
    /// GET /api/v1/stats — server + queue statistics for the dashboard
    fn handle_stats(&self) -> String {
        let uptime_secs = SystemTime::now()
            .duration_since(self.server_start_time)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let images_processed = self.images_processed.load(Ordering::Relaxed);

        let (pending, processing, completed, failed) =
            if let Ok(jobs) = self.jobs.lock() {
                let mut pe = 0u64; let mut pr = 0u64; let mut co = 0u64; let mut fa = 0u64;
                for job in jobs.values() {
                    match &job.status {
                        JobStatus::Pending    => pe += 1,
                        JobStatus::Processing => pr += 1,
                        JobStatus::Completed  => co += 1,
                        JobStatus::Failed(_)  => fa += 1,
                    }
                }
                (pe, pr, co, fa)
            } else {
                (0, 0, 0, 0)
            };

        let (credits_issued, credits_consumed) =
            if let Ok(db) = self.billing_db.lock() {
                let (i, c) = db.totals_today();
                (i as u64, c as u64)
            } else {
                (0, 0)
            };

        let load_avg = sys_info::loadavg().map(|la| la.one).unwrap_or(0.0);
        let mem_total_mb = sys_info::mem_info().map(|m| m.total / 1024).unwrap_or(0);
        let mem_free_mb  = sys_info::mem_info().map(|m| m.free  / 1024).unwrap_or(0);
        let mem_used_mb  = mem_total_mb.saturating_sub(mem_free_mb);

        let response = serde_json::json!({
            "version": "0.2.0",
            "uptime_secs": uptime_secs,
            "model": self.network.display(),
            "model_factor": self.network.factor(),
            "s3_enabled": self.s3_config.is_some(),
            "images_processed": images_processed,
            "jobs": {
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "total": pending + processing + completed + failed,
            },
            "credits_today": {
                "issued": credits_issued,
                "consumed": credits_consumed,
            },
            "system": {
                "load_avg_1m": load_avg,
                "mem_used_mb": mem_used_mb,
                "mem_total_mb": mem_total_mb,
            }
        });

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            response
        )
    }

    /// GET /api/v1/jobs — list recent jobs (newest first, up to 100).
    /// Supports `?org_id=xxx` query parameter to filter by organization.
    fn handle_jobs(&self, request: &str) -> String {
        // Extract org_id query param from the request line
        let org_id_filter = self.extract_query_param(request, "org_id");

        let jobs_snapshot: Vec<serde_json::Value> =
            if let Ok(jobs) = self.jobs.lock() {
                let mut list: Vec<&JobInfo> = jobs.values()
                    .filter(|j| {
                        if let Some(ref filter_org) = org_id_filter {
                            j.org_id.as_deref() == Some(filter_org.as_str())
                        } else {
                            true
                        }
                    })
                    .collect();
                list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                list.truncate(100);
                list.iter().map(|j| {
                    let status_str = match &j.status {
                        JobStatus::Pending    => "pending".to_string(),
                        JobStatus::Processing => "processing".to_string(),
                        JobStatus::Completed  => "completed".to_string(),
                        JobStatus::Failed(e)  => format!("failed: {}", e),
                    };
                    let duration_secs = if matches!(&j.status, JobStatus::Completed | JobStatus::Failed(_)) {
                        Some(j.updated_at.saturating_sub(j.created_at))
                    } else {
                        None
                    };
                    serde_json::json!({
                        "id": j.id,
                        "status": status_str,
                        "model": j.model,
                        "input_size": j.input_size,
                        "output_size": j.output_size,
                        "duration_secs": duration_secs,
                        "created_at": j.created_at,
                        "updated_at": j.updated_at,
                        "result_url": j.result_url,
                        "error": j.error,
                        "webhook_delivery": j.webhook_delivery,
                        "org_id": j.org_id,
                    })
                }).collect()
            } else {
                vec![]
            };

        let response = serde_json::json!({ "jobs": jobs_snapshot, "count": jobs_snapshot.len() });

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            response
        )
    }

    fn _handle_dashboard_removed(&self) -> String {
        // Removed: old standalone dashboard replaced by handle_root_dashboard at /dashboard
        let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SRGAN-Rust &mdash; Dashboard</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#21262d;
  --border:#30363d;--accent:#00d4ff;--accent2:#7ec8e3;
  --green:#3fb950;--yellow:#d29922;--red:#f85149;--purple:#a371f7;
  --text:#c9d1d9;--text-muted:#8b949e;--text-dim:#484f58;
  --radius:8px;--font:ui-monospace,'Cascadia Code','Fira Mono','Menlo',monospace;
}
html{font-size:14px}
body{font-family:var(--font);background:var(--bg);color:var(--text);
     min-height:100vh;padding:0}

/* ── layout ── */
.shell{display:grid;grid-template-columns:220px 1fr;min-height:100vh}
.sidebar{background:var(--bg2);border-right:1px solid var(--border);
         padding:1.5rem 1rem;display:flex;flex-direction:column;gap:1rem;
         position:sticky;top:0;height:100vh;overflow-y:auto}
.main{padding:2rem;overflow-y:auto;min-width:0}

/* ── sidebar ── */
.logo{color:var(--accent);font-size:1.1rem;font-weight:700;
      letter-spacing:.04em;margin-bottom:.5rem}
.logo span{color:var(--text-muted);font-weight:400}
.nav-item{display:flex;align-items:center;gap:.6rem;padding:.5rem .75rem;
          border-radius:var(--radius);color:var(--text-muted);cursor:pointer;
          transition:background .15s,color .15s;font-size:.9rem}
.nav-item:hover,.nav-item.active{background:var(--bg3);color:var(--accent)}
.nav-icon{width:16px;text-align:center;flex-shrink:0}
.sidebar-section{margin-top:auto;border-top:1px solid var(--border);padding-top:1rem}
.pulse{display:inline-block;width:8px;height:8px;border-radius:50%;
       background:var(--green);margin-right:.4rem;
       animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ── section pages ── */
.page{display:none}.page.active{display:block}
.section-title{font-size:1.4rem;color:var(--accent);font-weight:700;
               margin-bottom:.25rem}
.section-sub{color:var(--text-muted);font-size:.85rem;margin-bottom:1.5rem}

/* ── stat cards ── */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
       gap:.85rem;margin-bottom:1.75rem}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);
      padding:1rem 1.1rem}
.card-value{font-size:1.75rem;font-weight:700;color:var(--accent);line-height:1}
.card-label{color:var(--text-muted);font-size:.72rem;margin-top:.35rem;
            text-transform:uppercase;letter-spacing:.06em}
.card.green .card-value{color:var(--green)}
.card.yellow .card-value{color:var(--yellow)}
.card.red .card-value{color:var(--red)}
.card.purple .card-value{color:var(--purple)}

/* ── status bar ── */
.statusbar{display:flex;align-items:center;gap:1.5rem;padding:.6rem 1rem;
           background:var(--bg2);border-bottom:1px solid var(--border);
           font-size:.78rem;color:var(--text-muted);flex-wrap:wrap}
.statusbar strong{color:var(--text)}
.tag{display:inline-block;padding:.1rem .45rem;border-radius:4px;font-size:.72rem;
     background:var(--bg3);border:1px solid var(--border)}
.tag.ok{color:var(--green);border-color:#238636}
.tag.warn{color:var(--yellow);border-color:#9e6a03}
.tag.err{color:var(--red);border-color:#8b2121}
.tag.info{color:var(--accent);border-color:#0f6ab6}

/* ── table ── */
.tbl-wrap{overflow-x:auto;border-radius:var(--radius);
          border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:.83rem}
thead th{background:var(--bg3);color:var(--text-muted);font-weight:600;
         text-align:left;padding:.6rem .9rem;border-bottom:1px solid var(--border);
         text-transform:uppercase;font-size:.7rem;letter-spacing:.05em;white-space:nowrap}
tbody tr{border-bottom:1px solid var(--border);cursor:pointer;
         transition:background .1s}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--bg3)}
td{padding:.55rem .9rem;vertical-align:middle;white-space:nowrap}
.id-cell{font-family:var(--font);color:var(--accent2);font-size:.78rem;
         max-width:140px;overflow:hidden;text-overflow:ellipsis}
.status-badge{display:inline-flex;align-items:center;gap:.35rem;
              padding:.15rem .55rem;border-radius:4px;font-size:.73rem;font-weight:600}
.s-pending{background:#1c2333;color:var(--text-muted);border:1px solid var(--border)}
.s-processing{background:#12213b;color:var(--accent);border:1px solid #0f3460}
.s-completed{background:#122116;color:var(--green);border:1px solid #238636}
.s-failed{background:#2d1113;color:var(--red);border:1px solid #8b2121}
.dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}
.s-processing .dot{animation:pulse 1.2s ease-in-out infinite}
.empty-row td{text-align:center;color:var(--text-dim);padding:2.5rem;font-size:.85rem}

/* ── models grid ── */
.models-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
             gap:.85rem}
.model-card{background:var(--bg2);border:1px solid var(--border);
            border-radius:var(--radius);padding:1rem 1.1rem}
.model-card:hover{border-color:var(--accent)}
.model-name{color:var(--accent);font-weight:700;font-size:.95rem;margin-bottom:.4rem}
.model-desc{color:var(--text-muted);font-size:.78rem;line-height:1.5}
.model-meta{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.6rem}
.chip{background:var(--bg3);border:1px solid var(--border);border-radius:4px;
      padding:.1rem .4rem;font-size:.68rem;color:var(--text-muted)}

/* ── submit form ── */
.form-card{background:var(--bg2);border:1px solid var(--border);
           border-radius:var(--radius);padding:1.5rem;max-width:540px}
.form-group{margin-bottom:1rem}
.form-label{display:block;color:var(--text-muted);font-size:.78rem;
            text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem}
.form-input,.form-select{width:100%;background:var(--bg3);border:1px solid var(--border);
  border-radius:6px;padding:.5rem .7rem;color:var(--text);font-family:var(--font);
  font-size:.88rem;outline:none;transition:border-color .15s}
.form-input:focus,.form-select:focus{border-color:var(--accent)}
.form-select option{background:var(--bg3)}
.file-drop{border:2px dashed var(--border);border-radius:var(--radius);
           padding:1.5rem;text-align:center;cursor:pointer;
           color:var(--text-muted);transition:border-color .15s,background .15s}
.file-drop:hover,.file-drop.dragover{border-color:var(--accent);
  background:rgba(0,212,255,.04)}
.file-drop input[type=file]{display:none}
.file-name{color:var(--accent);font-size:.82rem;margin-top:.4rem}
.btn{display:inline-flex;align-items:center;gap:.5rem;padding:.55rem 1.2rem;
     border-radius:6px;border:none;cursor:pointer;font-family:var(--font);
     font-size:.88rem;font-weight:600;transition:background .15s,opacity .15s}
.btn-primary{background:var(--accent);color:#0d1117}
.btn-primary:hover{background:#33dcff}
.btn-primary:disabled{opacity:.45;cursor:not-allowed}
.btn-ghost{background:var(--bg3);color:var(--text);border:1px solid var(--border)}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent)}
.form-msg{margin-top:.75rem;padding:.5rem .75rem;border-radius:6px;font-size:.82rem}
.form-msg.ok{background:#122116;color:var(--green);border:1px solid #238636}
.form-msg.err{background:#2d1113;color:var(--red);border:1px solid #8b2121}
.form-msg.info{background:#12213b;color:var(--accent);border:1px solid #0f3460}
.preview-img{max-width:100%;max-height:180px;border-radius:6px;margin-top:.75rem;
             border:1px solid var(--border);display:none}

/* ── modal ── */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);
               display:none;align-items:center;justify-content:center;z-index:100;
               padding:1rem}
.modal-overlay.open{display:flex}
.modal{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
       width:100%;max-width:560px;max-height:90vh;overflow-y:auto}
.modal-header{display:flex;align-items:center;justify-content:space-between;
              padding:1rem 1.25rem;border-bottom:1px solid var(--border)}
.modal-title{color:var(--accent);font-weight:700;font-size:.95rem}
.modal-close{background:none;border:none;color:var(--text-muted);cursor:pointer;
             font-size:1.2rem;line-height:1;padding:.2rem .4rem;border-radius:4px}
.modal-close:hover{background:var(--bg3);color:var(--text)}
.modal-body{padding:1.25rem}
.detail-row{display:grid;grid-template-columns:130px 1fr;gap:.5rem;
            padding:.45rem 0;border-bottom:1px solid var(--bg3);font-size:.83rem}
.detail-row:last-child{border-bottom:none}
.detail-key{color:var(--text-muted)}
.detail-val{color:var(--text);word-break:break-all}
.webhook-section{margin-top:1rem;padding:1rem;background:var(--bg3);
                 border-radius:var(--radius)}
.webhook-title{color:var(--accent2);font-size:.78rem;font-weight:600;
               text-transform:uppercase;letter-spacing:.06em;margin-bottom:.75rem}

/* ── progress bar ── */
.progress-wrap{margin-top:1rem;display:none}
.progress-wrap.visible{display:block}
.progress-label{color:var(--text-muted);font-size:.78rem;margin-bottom:.35rem;
                display:flex;justify-content:space-between}
.progress-track{height:8px;background:var(--bg3);border-radius:4px;
                border:1px solid var(--border);overflow:hidden}
.progress-bar{height:100%;width:0%;background:var(--accent);border-radius:4px;
              transition:width .18s ease}
.progress-bar.done{background:var(--green)}
.progress-bar.err{background:var(--red)}
.sse-status{margin-top:.5rem;font-size:.78rem;color:var(--text-muted)}

/* ── refresh indicator ── */
.refresh-indicator{display:flex;align-items:center;gap:.4rem;color:var(--text-dim);
                   font-size:.72rem;margin-left:auto}
.refresh-dot{width:6px;height:6px;border-radius:50%;background:var(--green)}
.refresh-dot.refreshing{animation:pulse .6s ease-in-out infinite}

/* ── top nav ── */
.topbar{display:flex;align-items:center;gap:.75rem;padding:.7rem 2rem;
        background:var(--bg2);border-bottom:1px solid var(--border);
        position:sticky;top:0;z-index:10}
.topbar-title{color:var(--text);font-weight:600;font-size:.9rem}
</style>
</head>
<body>

<div class="topbar">
  <span class="logo">SRGAN<span>-Rust</span></span>
  <span class="topbar-title">Admin Dashboard</span>
  <div class="refresh-indicator">
    <span class="refresh-dot" id="rdot"></span>
    <span id="refresh-ts">--</span>
  </div>
</div>

<div class="statusbar" id="statusbar">
  <span><strong id="sb-model">--</strong></span>
  <span>Uptime: <strong id="sb-uptime">--</strong></span>
  <span>Version: <span class="tag info" id="sb-version">--</span></span>
  <span>S3: <span class="tag" id="sb-s3">--</span></span>
  <span>Load: <strong id="sb-load">--</strong></span>
  <span>RAM: <strong id="sb-ram">--</strong></span>
</div>

<div class="shell">
  <div class="sidebar">
    <nav style="display:flex;flex-direction:column;gap:.25rem">
      <div class="nav-item active" onclick="showPage('overview')">
        <span class="nav-icon">&#9635;</span> Overview
      </div>
      <div class="nav-item" onclick="showPage('jobs')">
        <span class="nav-icon">&#8801;</span> Jobs
      </div>
      <div class="nav-item" onclick="showPage('submit')">
        <span class="nav-icon">&#8679;</span> Submit Job
      </div>
      <div class="nav-item" onclick="showPage('models')">
        <span class="nav-icon">&#9670;</span> Models
      </div>
    </nav>
    <div class="sidebar-section">
      <div style="color:var(--text-dim);font-size:.72rem">
        <span class="pulse"></span>Live &mdash; refreshes every 5s
      </div>
    </div>
  </div>

  <div class="main">

    <!-- ── Overview ── -->
    <div class="page active" id="page-overview">
      <div class="section-title">Overview</div>
      <div class="section-sub">Real-time server and job queue metrics</div>

      <div style="margin-bottom:.5rem;color:var(--text-muted);font-size:.75rem;
                  text-transform:uppercase;letter-spacing:.06em">Job Queue</div>
      <div class="cards" id="queue-cards">
        <div class="card"><div class="card-value" id="c-pending">--</div><div class="card-label">Pending</div></div>
        <div class="card s-processing" style="border-color:#0f3460"><div class="card-value" id="c-processing">--</div><div class="card-label">Processing</div></div>
        <div class="card green"><div class="card-value" id="c-completed">--</div><div class="card-label">Completed</div></div>
        <div class="card red"><div class="card-value" id="c-failed">--</div><div class="card-label">Failed</div></div>
        <div class="card purple"><div class="card-value" id="c-total">--</div><div class="card-label">All&#8209;Time Processed</div></div>
      </div>

      <div style="margin-bottom:.5rem;color:var(--text-muted);font-size:.75rem;
                  text-transform:uppercase;letter-spacing:.06em;margin-top:1.25rem">
        Credits Today</div>
      <div class="cards">
        <div class="card green"><div class="card-value" id="c-issued">--</div><div class="card-label">Issued</div></div>
        <div class="card yellow"><div class="card-value" id="c-consumed">--</div><div class="card-label">Consumed</div></div>
      </div>

      <div style="margin-top:1.5rem;margin-bottom:.5rem;color:var(--text-muted);
                  font-size:.75rem;text-transform:uppercase;letter-spacing:.06em">
        Recent Jobs</div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Status</th><th>Model</th>
              <th>Input</th><th>Output</th><th>Duration</th><th>Created</th>
            </tr>
          </thead>
          <tbody id="jobs-tbody-overview"></tbody>
        </table>
      </div>
    </div>

    <!-- ── Jobs ── -->
    <div class="page" id="page-jobs">
      <div class="section-title">Job Queue</div>
      <div class="section-sub">All jobs &mdash; newest first (up to 100)</div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Status</th><th>Model</th>
              <th>Input</th><th>Output</th><th>Duration</th><th>Created</th>
            </tr>
          </thead>
          <tbody id="jobs-tbody-full"></tbody>
        </table>
      </div>
    </div>

    <!-- ── Submit Job ── -->
    <div class="page" id="page-submit">
      <div class="section-title">Submit Job</div>
      <div class="section-sub">Upload an image to upscale asynchronously</div>
      <div class="form-card">
        <div class="form-group">
          <label class="form-label">Image File</label>
          <div class="file-drop" id="file-drop" onclick="document.getElementById('file-input').click()">
            <input type="file" id="file-input" accept="image/*">
            <div>&#8679; Click or drag &amp; drop an image</div>
            <div class="file-name" id="file-name"></div>
          </div>
          <img id="preview-img" class="preview-img" alt="preview">
        </div>
        <div class="form-group">
          <label class="form-label">Model</label>
          <select class="form-select" id="model-select">
            <option value="natural">natural &mdash; Photos &amp; general (&#215;4)</option>
            <option value="anime">anime &mdash; Anime / illustrations (&#215;4)</option>
            <option value="waifu2x">waifu2x &mdash; Anime + noise reduction (&#215;4)</option>
            <option value="waifu2x-noise0-scale2">waifu2x-noise0-scale2 &mdash; No noise reduction (&#215;2)</option>
            <option value="waifu2x-noise1-scale2">waifu2x-noise1-scale2 &mdash; Light denoising (&#215;2)</option>
            <option value="waifu2x-noise2-scale2">waifu2x-noise2-scale2 &mdash; Medium denoising (&#215;2)</option>
            <option value="waifu2x-noise3-scale2">waifu2x-noise3-scale2 &mdash; Aggressive denoising (&#215;2)</option>
            <option value="bilinear">bilinear &mdash; No neural network</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Scale Factor <span style="color:var(--text-dim)">(leave 0 for model default)</span></label>
          <input type="number" class="form-input" id="scale-input"
                 min="0" max="8" value="0" placeholder="0 = model default">
        </div>
        <div class="form-group">
          <label class="form-label">Output Format</label>
          <select class="form-select" id="format-select">
            <option value="png">PNG (lossless)</option>
            <option value="jpeg">JPEG (lossy, smaller)</option>
          </select>
        </div>
        <button class="btn btn-primary" id="submit-btn" disabled onclick="submitJob()">
          &#8679; Submit Job
        </button>
        <div id="submit-msg" style="display:none" class="form-msg"></div>

        <!-- SSE live progress -->
        <div class="progress-wrap" id="progress-wrap">
          <div class="progress-label">
            <span id="progress-event">Queued&hellip;</span>
            <span id="progress-pct">0%</span>
          </div>
          <div class="progress-track">
            <div class="progress-bar" id="progress-bar"></div>
          </div>
          <div class="sse-status" id="sse-status"></div>
        </div>
      </div>
    </div>

    <!-- ── Models ── -->
    <div class="page" id="page-models">
      <div class="section-title">Supported Models</div>
      <div class="section-sub">Available upscaling models and their capabilities</div>
      <div class="models-grid" id="models-grid"></div>
    </div>

  </div><!-- .main -->
</div><!-- .shell -->

<!-- ── Job Detail Modal ── -->
<div class="modal-overlay" id="modal" onclick="closeModal(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-header">
      <span class="modal-title">Job Details</span>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>

<script>
'use strict';
// ── navigation ─────────────────────────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  event.currentTarget.classList.add('active');
}

// ── helpers ─────────────────────────────────────────────────────────────────
function fmtUptime(s) {
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
  if (h < 24) return h + 'h ' + m + 'm';
  const d = Math.floor(h/24);
  return d + 'd ' + (h%24) + 'h';
}
function fmtTs(unix) {
  if (!unix) return '--';
  const d = new Date(unix * 1000);
  return d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
function fmtDur(secs) {
  if (secs == null) return '--';
  if (secs < 1) return '<1s';
  if (secs < 60) return secs + 's';
  return Math.floor(secs/60) + 'm ' + (secs%60) + 's';
}
function statusBadge(s) {
  const st = s.startsWith('failed') ? 'failed'
           : s === 'completed' ? 'completed'
           : s === 'processing' ? 'processing' : 'pending';
  const icons = {pending:'&#9675;',processing:'&#9679;',completed:'&#10003;',failed:'&#10005;'};
  return `<span class="status-badge s-${st}"><span class="dot"></span>${icons[st]} ${s}</span>`;
}
function el(id) { return document.getElementById(id); }
function setText(id, v) { const e = el(id); if(e) e.textContent = v; }

// ── fetch helpers ─────────────────────────────────────────────────────────
let lastStats = null, lastJobs = [];

async function fetchStats() {
  try {
    const r = await fetch('/api/v1/stats');
    if (!r.ok) return;
    lastStats = await r.json();
    applyStats(lastStats);
  } catch(e) { /* server may be starting */ }
}

async function fetchJobs() {
  try {
    const r = await fetch('/api/v1/jobs');
    if (!r.ok) return;
    const d = await r.json();
    lastJobs = d.jobs || [];
    renderJobTable('jobs-tbody-overview', lastJobs.slice(0, 10));
    renderJobTable('jobs-tbody-full', lastJobs);
  } catch(e) {}
}

function applyStats(s) {
  // status bar
  el('sb-model').textContent = s.model + ' \u00d7' + s.model_factor;
  el('sb-uptime').textContent = fmtUptime(s.uptime_secs);
  el('sb-version').textContent = 'v' + s.version;
  const s3el = el('sb-s3');
  s3el.textContent = s.s3_enabled ? 'S3 on' : 'S3 off';
  s3el.className = 'tag ' + (s.s3_enabled ? 'ok' : 'warn');
  el('sb-load').textContent = s.system.load_avg_1m.toFixed(2);
  el('sb-ram').textContent = s.system.mem_used_mb + ' / ' + s.system.mem_total_mb + ' MB';
  // cards
  const j = s.jobs;
  setText('c-pending',    j.pending);
  setText('c-processing', j.processing);
  setText('c-completed',  j.completed);
  setText('c-failed',     j.failed);
  setText('c-total',      s.images_processed);
  setText('c-issued',   s.credits_today.issued);
  setText('c-consumed', s.credits_today.consumed);
}

function renderJobTable(tbodyId, jobs) {
  const tb = el(tbodyId);
  if (!tb) return;
  if (!jobs || jobs.length === 0) {
    tb.innerHTML = '<tr class="empty-row"><td colspan="7">No jobs yet</td></tr>';
    return;
  }
  tb.innerHTML = jobs.map(j => `
    <tr onclick="openJobModal('${j.id}')">
      <td class="id-cell" title="${j.id}">${j.id.substring(0,16)}&hellip;</td>
      <td>${statusBadge(j.status)}</td>
      <td>${j.model || '<span style="color:var(--text-dim)">--</span>'}</td>
      <td>${j.input_size || '--'}</td>
      <td>${j.output_size || '--'}</td>
      <td>${fmtDur(j.duration_secs)}</td>
      <td>${fmtTs(j.created_at)}</td>
    </tr>`).join('');
}

// ── models ────────────────────────────────────────────────────────────────
const MODELS = [
  { name:'natural',   arch:'SRGAN',    scale:'&#215;4',
    desc:'Neural network trained on natural photographs using L1 loss. Best for photos and real-world images.',
    tags:['photos','general','landscapes'] },
  { name:'anime',     arch:'SRGAN',    scale:'&#215;4',
    desc:'Neural network trained on animation art using L1 loss. Sharp edges and vibrant colours.',
    tags:['anime','illustrations','cartoons'] },
  { name:'waifu2x',   arch:'Waifu2x',  scale:'&#215;2 / &#215;4',
    desc:'Waifu2x-style model with configurable noise reduction (noise_level 0&ndash;3, scale 1&times; or 2&times;).',
    tags:['anime','illustrations','denoising'] },
  { name:'real-esrgan',       arch:'Real-ESRGAN', scale:'&#215;4',
    desc:'Real-ESRGAN ×4 for general photos — handles JPEG artifacts, Gaussian noise, blur, and other real-world degradations.',
    tags:['photos','compressed','noisy','real-world'] },
  { name:'real-esrgan-anime', arch:'Real-ESRGAN', scale:'&#215;4',
    desc:'Real-ESRGAN ×4 optimised for anime and illustration content — sharper line art via the anime degradation pipeline.',
    tags:['anime','illustrations','line-art'] },
  { name:'real-esrgan-x2',    arch:'Real-ESRGAN', scale:'&#215;2',
    desc:'Real-ESRGAN ×2 for general photos — lower memory than the ×4 variant; ideal for moderate resolution boosts.',
    tags:['photos','general','low-memory'] },
  { name:'bilinear',  arch:'Bilinear', scale:'&#215;4',
    desc:'Classical bilinear interpolation. No neural network &mdash; fast preview or fallback.',
    tags:['general','quick-preview'] },
];

function renderModels() {
  const g = el('models-grid');
  if (!g) return;
  g.innerHTML = MODELS.map(m => `
    <div class="model-card">
      <div class="model-name">${m.name}</div>
      <div class="model-desc">${m.desc}</div>
      <div class="model-meta">
        <span class="chip">${m.arch}</span>
        <span class="chip">${m.scale}</span>
        ${m.tags.map(t => `<span class="chip">${t}</span>`).join('')}
      </div>
    </div>`).join('');
}

// ── job detail modal ──────────────────────────────────────────────────────
function openJobModal(id) {
  const job = lastJobs.find(j => j.id === id);
  if (!job) return;
  const wh = job.webhook_delivery;
  let whHtml = '';
  if (wh) {
    const delivered = wh.delivered
      ? '<span class="tag ok">delivered</span>'
      : '<span class="tag warn">not delivered</span>';
    whHtml = `
      <div class="webhook-section">
        <div class="webhook-title">Webhook Delivery</div>
        <div class="detail-row"><span class="detail-key">Status</span><span class="detail-val">${delivered}</span></div>
        <div class="detail-row"><span class="detail-key">Attempts</span><span class="detail-val">${wh.attempts}</span></div>
        ${wh.last_status_code != null ? `<div class="detail-row"><span class="detail-key">Last HTTP</span><span class="detail-val">${wh.last_status_code}</span></div>` : ''}
        ${wh.last_attempt_at ? `<div class="detail-row"><span class="detail-key">Last Attempt</span><span class="detail-val">${fmtTs(wh.last_attempt_at)}</span></div>` : ''}
      </div>`;
  }
  el('modal-body').innerHTML = `
    <div class="detail-row"><span class="detail-key">Job ID</span><span class="detail-val" style="font-size:.78rem;word-break:break-all">${job.id}</span></div>
    <div class="detail-row"><span class="detail-key">Status</span><span class="detail-val">${statusBadge(job.status)}</span></div>
    <div class="detail-row"><span class="detail-key">Model</span><span class="detail-val">${job.model || '--'}</span></div>
    <div class="detail-row"><span class="detail-key">Input Size</span><span class="detail-val">${job.input_size || '--'}</span></div>
    <div class="detail-row"><span class="detail-key">Output Size</span><span class="detail-val">${job.output_size || '--'}</span></div>
    <div class="detail-row"><span class="detail-key">Duration</span><span class="detail-val">${fmtDur(job.duration_secs)}</span></div>
    <div class="detail-row"><span class="detail-key">Created</span><span class="detail-val">${fmtTs(job.created_at)}</span></div>
    <div class="detail-row"><span class="detail-key">Updated</span><span class="detail-val">${fmtTs(job.updated_at)}</span></div>
    ${job.result_url ? `<div class="detail-row"><span class="detail-key">Result</span><span class="detail-val"><a href="${job.result_url}" style="color:var(--accent)" target="_blank">Download &#8599;</a></span></div>` : ''}
    ${job.error ? `<div class="detail-row"><span class="detail-key">Error</span><span class="detail-val" style="color:var(--red)">${job.error}</span></div>` : ''}
    ${whHtml}`;
  el('modal').classList.add('open');
}
function closeModal(e) {
  if (!e || e.target === el('modal') || e.currentTarget === el('modal')) {
    el('modal').classList.remove('open');
  }
}
document.addEventListener('keydown', e => { if(e.key==='Escape') el('modal').classList.remove('open'); });

// ── submit job ─────────────────────────────────────────────────────────────
let selectedFileB64 = null;

el('file-input').addEventListener('change', function() {
  const file = this.files[0];
  if (!file) return;
  el('file-name').textContent = file.name;
  const reader = new FileReader();
  reader.onload = function(ev) {
    const dataUrl = ev.target.result;
    // strip data:image/...;base64, prefix
    selectedFileB64 = dataUrl.split(',')[1];
    const img = el('preview-img');
    img.src = dataUrl;
    img.style.display = 'block';
    el('submit-btn').disabled = false;
  };
  reader.readAsDataURL(file);
});

// drag-and-drop
const dropZone = el('file-drop');
['dragenter','dragover'].forEach(evt => {
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('dragover'); });
});
['dragleave','drop'].forEach(evt => {
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('dragover'); });
});
dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (!file || !file.type.startsWith('image/')) return;
  el('file-name').textContent = file.name;
  const reader = new FileReader();
  reader.onload = ev => {
    const dataUrl = ev.target.result;
    selectedFileB64 = dataUrl.split(',')[1];
    const img = el('preview-img');
    img.src = dataUrl;
    img.style.display = 'block';
    el('submit-btn').disabled = false;
  };
  reader.readAsDataURL(file);
});

// ── SSE progress tracking ──────────────────────────────────────────────────
let activeSSE = null;

function startSSEProgress(jobId) {
  // Close any previous stream.
  if (activeSSE) { try { activeSSE.close(); } catch(_) {} activeSSE = null; }

  const wrap = el('progress-wrap');
  const bar  = el('progress-bar');
  const pct  = el('progress-pct');
  const evt  = el('progress-event');
  const sts  = el('sse-status');

  wrap.classList.add('visible');
  bar.style.width = '0%';
  bar.className = 'progress-bar';
  pct.textContent = '0%';
  evt.textContent = 'Connecting\u2026';
  sts.textContent = '';

  const es = new EventSource('/api/v1/job/' + encodeURIComponent(jobId) + '/stream');
  activeSSE = es;

  es.onmessage = function(e) {
    let data;
    try { data = JSON.parse(e.data); } catch(_) { return; }

    switch (data.event) {
      case 'queued':
        evt.textContent = 'Queued (position ' + (data.position || 0) + ')';
        pct.textContent = '0%';
        break;
      case 'started':
        evt.textContent = 'Processing\u2026';
        pct.textContent = '0%';
        break;
      case 'progress':
        const p = data.percent || 0;
        bar.style.width = p + '%';
        pct.textContent = p + '%';
        evt.textContent = 'Processing\u2026';
        break;
      case 'done':
        bar.style.width = '100%';
        bar.classList.add('done');
        pct.textContent = '100%';
        evt.textContent = 'Done \u2713';
        sts.innerHTML = data.output_url
          ? 'Result: <a href="' + data.output_url + '" style="color:var(--accent)" target="_blank">Download \u2197</a>'
          : '';
        es.close(); activeSSE = null;
        setTimeout(() => { showPageByName('jobs'); refreshAll(); }, 1200);
        break;
      case 'error':
        bar.classList.add('err');
        evt.textContent = 'Error';
        sts.textContent = data.message || 'Unknown error';
        es.close(); activeSSE = null;
        break;
    }
  };

  es.onerror = function() {
    sts.textContent = 'Stream disconnected.';
    es.close(); activeSSE = null;
  };
}

async function submitJob() {
  if (!selectedFileB64) return;
  const btn = el('submit-btn');
  const msgEl = el('submit-msg');
  const model = el('model-select').value;
  const scale = parseInt(el('scale-input').value, 10) || null;
  const format = el('format-select').value;

  btn.disabled = true;
  btn.textContent = 'Submitting\u2026';
  msgEl.style.display = 'none';
  // Hide any previous progress bar
  el('progress-wrap').classList.remove('visible');

  const payload = { image_data: selectedFileB64, model, format };
  if (scale && scale > 0) payload.scale_factor = scale;

  try {
    const r = await fetch('/api/v1/upscale/async', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (r.ok || r.status === 202) {
      msgEl.className = 'form-msg ok';
      msgEl.textContent = 'Job queued: ' + (d.job_id || '');
      msgEl.style.display = 'block';
      selectedFileB64 = null;
      el('file-name').textContent = '';
      el('preview-img').style.display = 'none';
      el('file-input').value = '';
      btn.textContent = '\u2191 Submit Job';
      // Open SSE stream for live progress.
      if (d.job_id) startSSEProgress(d.job_id);
    } else {
      throw new Error(d.error || ('HTTP ' + r.status));
    }
  } catch(e) {
    msgEl.className = 'form-msg err';
    msgEl.textContent = 'Error: ' + e.message;
    msgEl.style.display = 'block';
    btn.disabled = false;
    btn.textContent = '\u2191 Submit Job';
  }
}

function showPageByName(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  // activate matching nav item
  document.querySelectorAll('.nav-item').forEach(n => {
    if (n.getAttribute('onclick') && n.getAttribute('onclick').includes("'" + name + "'"))
      n.classList.add('active');
  });
}

// ── auto-refresh ─────────────────────────────────────────────────────────
async function refreshAll() {
  const dot = el('rdot');
  dot.classList.add('refreshing');
  await Promise.all([fetchStats(), fetchJobs()]);
  dot.classList.remove('refreshing');
  el('refresh-ts').textContent = new Date().toLocaleTimeString();
}

// initial render
renderModels();
refreshAll();
setInterval(refreshAll, 5000);
</script>
</body>
</html>"#;

        format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nCache-Control: no-cache\r\n\r\n{}", html)
    }

    /// POST /api/v1/compare — upscale one image with multiple models simultaneously
    /// and return side-by-side PSNR / SSIM quality metrics plus output images.
    ///
    /// The input image is treated as the high-resolution reference.  It is
    /// downscaled by the native model upscale factor (usually 4×) to produce a
    /// degraded LR image, which is then upscaled by each requested model.  Each
    /// output is compared against the original HR image to yield PSNR and SSIM.
    fn handle_compare(&self, request: &str) -> String {
        let body = self.extract_body(request);

        let req: CompareRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return self.error_response(400, &format!("Invalid request: {}", e)),
        };

        if req.models.is_empty() {
            return self.error_response(400, "models list must not be empty");
        }
        if req.models.len() > 8 {
            return self.error_response(400, "models list must not exceed 8 entries");
        }

        // Decode and load input image
        let image_bytes = match general_purpose::STANDARD.decode(&req.image_data) {
            Ok(d) => d,
            Err(e) => return self.error_response(400, &format!("Invalid base64: {}", e)),
        };
        let hr_img = match image::load_from_memory(&image_bytes) {
            Ok(img) => img,
            Err(e) => return self.error_response(400, &format!("Invalid image: {}", e)),
        };

        let original_size = (hr_img.width(), hr_img.height());
        let total_start = SystemTime::now();

        // Downscale HR → LR using the network's native upscale factor
        let upscale_factor = self.network.factor().max(1);
        let lr_w = (original_size.0 / upscale_factor).max(1);
        let lr_h = (original_size.1 / upscale_factor).max(1);
        let lr_img = hr_img.resize_exact(lr_w, lr_h, image::imageops::FilterType::Lanczos3);
        let degraded_size = (lr_img.width(), lr_img.height());

        let format = req.format.as_deref().unwrap_or("png");
        let img_format = match format {
            "jpeg" | "jpg" => ImageFormat::JPEG,
            _ => ImageFormat::PNG,
        };
        let tile_size = req.tile_size.map(|s| s.max(64)).unwrap_or(512);
        let use_tiling = (lr_w as usize * lr_h as usize) > 4_000_000 || req.tile_size.is_some();

        let mut results: Vec<ModelCompareResult> = Vec::with_capacity(req.models.len());

        for model_label in &req.models {
            let model_start = SystemTime::now();

            let upscale_result: std::result::Result<image::DynamicImage, SrganError> = (|| {
                if model_label == "natural" || model_label.is_empty() {
                    if use_tiling {
                        self.network.upscale_image_tiled(&lr_img, tile_size)
                    } else {
                        self.network.upscale_image(&lr_img)
                    }
                } else {
                    let net_result = crate::thread_safe_network::ThreadSafeNetwork::from_label(
                        model_label,
                        None,
                    );
                    let net = net_result.unwrap_or_else(|_| {
                        // Shadow-load the default network as fallback
                        crate::thread_safe_network::ThreadSafeNetwork::load_builtin_natural()
                            .expect("built-in natural model must always load")
                    });
                    if use_tiling {
                        net.upscale_image_tiled(&lr_img, tile_size)
                    } else {
                        net.upscale_image(&lr_img)
                    }
                }
            })();

            let processing_time_ms = model_start
                .elapsed()
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            match upscale_result {
                Ok(upscaled) => {
                    let upscaled_size = (upscaled.width(), upscaled.height());

                    // Compute quality metrics against the HR reference
                    let (psnr, ssim) = Self::compute_metrics(&hr_img, &upscaled);

                    // Encode to the requested output format
                    let image_data = if req.include_images {
                        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
                        match upscaled.write_to(&mut cursor, img_format) {
                            Ok(()) => Some(general_purpose::STANDARD.encode(cursor.into_inner())),
                            Err(_) => None,
                        }
                    } else {
                        None
                    };

                    results.push(ModelCompareResult {
                        model: model_label.clone(),
                        success: true,
                        psnr_db: Some(psnr),
                        ssim: Some(ssim),
                        image_data,
                        processing_time_ms,
                        upscaled_size: Some(upscaled_size),
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(ModelCompareResult {
                        model: model_label.clone(),
                        success: false,
                        psnr_db: None,
                        ssim: None,
                        image_data: None,
                        processing_time_ms,
                        upscaled_size: None,
                        error: Some(format!("{}", e)),
                    });
                }
            }
        }

        // Identify the top-performing model by each metric
        let best_psnr_model = results
            .iter()
            .filter(|r| r.psnr_db.is_some())
            .max_by(|a, b| {
                a.psnr_db
                    .unwrap()
                    .partial_cmp(&b.psnr_db.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.model.clone());

        let best_ssim_model = results
            .iter()
            .filter(|r| r.ssim.is_some())
            .max_by(|a, b| {
                a.ssim
                    .unwrap()
                    .partial_cmp(&b.ssim.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.model.clone());

        let total_processing_time_ms = total_start
            .elapsed()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let response = CompareResponse {
            success: true,
            original_size,
            degraded_size,
            results,
            best_psnr_model,
            best_ssim_model,
            total_processing_time_ms,
        };

        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            serde_json::to_string(&response)
                .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
        )
    }

    /// Compute PSNR (dB) and a global SSIM between two images.
    ///
    /// Both images are compared in RGB colour space with pixel values in [0, 255].
    /// If the images differ in size the reference is resized to match the upscaled
    /// output so that every pixel has a direct counterpart.
    ///
    /// Returns `(psnr_db, ssim)` where SSIM is in [0, 1].
    fn compute_metrics(
        reference: &image::DynamicImage,
        upscaled: &image::DynamicImage,
    ) -> (f64, f64) {
        let (w, h) = (upscaled.width(), upscaled.height());

        // Resize reference to match the upscaled output dimensions
        let ref_resized = if reference.width() != w || reference.height() != h {
            reference.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
        } else {
            reference.clone()
        };

        let ref_rgb = ref_resized.to_rgb();
        let up_rgb = upscaled.to_rgb();

        let n = (w as usize * h as usize * 3) as f64;
        let mut mse_acc = 0.0f64;
        let mut sum_r = 0.0f64;
        let mut sum_u = 0.0f64;
        let mut sum_r2 = 0.0f64;
        let mut sum_u2 = 0.0f64;
        let mut sum_ru = 0.0f64;

        for (rp, up) in ref_rgb.pixels().zip(up_rgb.pixels()) {
            for c in 0..3usize {
                let rv = rp[c] as f64;
                let uv = up[c] as f64;
                let d = rv - uv;
                mse_acc += d * d;
                sum_r += rv;
                sum_u += uv;
                sum_r2 += rv * rv;
                sum_u2 += uv * uv;
                sum_ru += rv * uv;
            }
        }

        // PSNR — clamped to 100 dB for identical images to avoid +∞
        let mse = mse_acc / n;
        let psnr = if mse < 1e-10 {
            100.0
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        };

        // Global SSIM using full-image statistics (simplified, no sliding window)
        let mu_r = sum_r / n;
        let mu_u = sum_u / n;
        let sigma_r2 = (sum_r2 / n) - mu_r * mu_r;
        let sigma_u2 = (sum_u2 / n) - mu_u * mu_u;
        let sigma_ru = (sum_ru / n) - mu_r * mu_u;

        // SSIM stability constants (L = 255)
        let c1 = (0.01 * 255.0f64).powi(2); // 6.5025
        let c2 = (0.03 * 255.0f64).powi(2); // 58.5225

        let ssim = ((2.0 * mu_r * mu_u + c1) * (2.0 * sigma_ru + c2))
            / ((mu_r * mu_r + mu_u * mu_u + c1) * (sigma_r2 + sigma_u2 + c2));

        (psnr, ssim.clamp(0.0, 1.0))
    }

    /// GET /api/v1/job/:id/stream — Server-Sent Events endpoint.
    ///
    /// Emits a sequence of progress events:
    ///   queued → started → progress (×10, 0 %→100 %) → done | error
    ///
    /// The progress is simulated over ~2 seconds (10 × 200 ms) so clients can
    /// exercise the SSE path without waiting for real job completion.
    /// GET /api/v1/dashboard/stream — SSE endpoint that pushes live queue stats,
    /// worker utilization, and per-job progress updates every second.
    fn handle_dashboard_stream(&self, mut stream: std::net::TcpStream) {
        let headers = concat!(
            "HTTP/1.1 200 OK\r\n",
            "Content-Type: text/event-stream\r\n",
            "Cache-Control: no-cache\r\n",
            "Connection: keep-alive\r\n",
            "Access-Control-Allow-Origin: *\r\n",
            "\r\n",
        );
        if stream.write_all(headers.as_bytes()).is_err() {
            return;
        }

        let write_sse = |stream: &mut std::net::TcpStream,
                         event: &str,
                         data: &serde_json::Value|
         -> bool {
            let msg = format!("event: {}\ndata: {}\n\n", event, data);
            stream.write_all(msg.as_bytes()).is_ok()
                && stream.flush().is_ok()
        };

        // Keep pushing until the client disconnects.
        loop {
            // ── Queue stats ──────────────────────────────────────────────
            let (pending, processing, completed, failed) =
                if let Ok(jobs) = self.jobs.lock() {
                    let mut pe = 0u64;
                    let mut pr = 0u64;
                    let mut co = 0u64;
                    let mut fa = 0u64;
                    for job in jobs.values() {
                        match &job.status {
                            JobStatus::Pending => pe += 1,
                            JobStatus::Processing => pr += 1,
                            JobStatus::Completed => co += 1,
                            JobStatus::Failed(_) => fa += 1,
                        }
                    }
                    (pe, pr, co, fa)
                } else {
                    (0, 0, 0, 0)
                };

            let total = pending + processing + completed + failed;
            let images_processed = self.images_processed.load(Ordering::Relaxed);

            // Worker utilization: ratio of processing jobs to total active
            // (pending + processing).  If no active jobs, utilization is 0 %.
            let active = pending + processing;
            let worker_utilization = if active > 0 {
                (processing as f64 / active as f64) * 100.0
            } else {
                0.0
            };

            let uptime_secs = SystemTime::now()
                .duration_since(self.server_start_time)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let stats = serde_json::json!({
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "total": total,
                "images_processed": images_processed,
                "worker_utilization_pct": (worker_utilization * 100.0).round() / 100.0,
                "uptime_secs": uptime_secs,
            });

            if !write_sse(&mut stream, "stats", &stats) {
                return;
            }

            // ── Per-job progress snapshot ────────────────────────────────
            // Emit an array of currently active (pending + processing) jobs
            // so the dashboard can show individual progress rows.
            let job_updates: Vec<serde_json::Value> =
                if let Ok(jobs) = self.jobs.lock() {
                    jobs.values()
                        .filter(|j| {
                            matches!(j.status, JobStatus::Pending | JobStatus::Processing)
                        })
                        .map(|j| {
                            serde_json::json!({
                                "id": j.id,
                                "status": match &j.status {
                                    JobStatus::Pending => "pending",
                                    JobStatus::Processing => "processing",
                                    _ => "unknown",
                                },
                                "model": j.model,
                                "created_at": j.created_at,
                            })
                        })
                        .collect()
                } else {
                    vec![]
                };

            if !write_sse(&mut stream, "jobs", &serde_json::json!(job_updates)) {
                return;
            }

            thread::sleep(Duration::from_secs(1));
        }
    }

    fn handle_job_stream(&self, mut stream: std::net::TcpStream, path: &str) {
        // Extract job_id: strip prefix + "/stream" suffix.
        let inner = path
            .trim_start_matches("/api/v1/job/")
            .trim_start_matches("/api/job/");
        let job_id = inner.trim_end_matches("/stream").to_string();

        // SSE response headers.
        let headers = concat!(
            "HTTP/1.1 200 OK\r\n",
            "Content-Type: text/event-stream\r\n",
            "Cache-Control: no-cache\r\n",
            "Connection: keep-alive\r\n",
            "Access-Control-Allow-Origin: *\r\n",
            "\r\n",
        );
        if stream.write_all(headers.as_bytes()).is_err() {
            return;
        }

        // Helper: write one SSE data line and flush.
        let write_event = |stream: &mut std::net::TcpStream, json: serde_json::Value| -> bool {
            let line = format!("data: {}\n\n", json);
            stream.write_all(line.as_bytes()).is_ok()
        };

        // Check whether the job exists and get its queue position.
        let (job_exists, queue_position) = {
            if let Ok(jobs) = self.jobs.lock() {
                let exists = jobs.contains_key(&job_id);
                let position = jobs
                    .values()
                    .filter(|j| matches!(j.status, JobStatus::Pending))
                    .count();
                (exists, position)
            } else {
                (false, 0)
            }
        };

        if !job_exists {
            write_event(
                &mut stream,
                serde_json::json!({
                    "event": "error",
                    "job_id": job_id,
                    "message": "Job not found"
                }),
            );
            return;
        }

        // --- queued ---
        if !write_event(
            &mut stream,
            serde_json::json!({
                "event": "queued",
                "job_id": job_id,
                "position": queue_position
            }),
        ) {
            return;
        }
        thread::sleep(Duration::from_millis(100));

        // --- started ---
        if !write_event(
            &mut stream,
            serde_json::json!({
                "event": "started",
                "job_id": job_id
            }),
        ) {
            return;
        }

        // --- progress: 10 events, 200 ms apart → ~2 s total ---
        for i in 1u8..=10 {
            thread::sleep(Duration::from_millis(200));
            let percent = i * 10;
            if !write_event(
                &mut stream,
                serde_json::json!({
                    "event": "progress",
                    "job_id": job_id,
                    "percent": percent
                }),
            ) {
                return;
            }
        }

        // --- done ---
        let output_url = format!("/api/v1/result/{}", job_id);
        write_event(
            &mut stream,
            serde_json::json!({
                "event": "done",
                "job_id": job_id,
                "output_url": output_url
            }),
        );
    }

    /// Start cache cleanup thread
    fn start_cache_cleanup(&self) {
        let cache = Arc::clone(&self.cache);
        let ttl = self.config.cache_ttl;
        
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(60));
                
                let now = SystemTime::now();
                if let Ok(mut cache_guard) = cache.lock() {
                    cache_guard.retain(|_, v| {
                        now.duration_since(v.created_at)
                            .map(|d| d < ttl)
                            .unwrap_or(false)
                    });
                }
            }
        });
    }

    // ── Multi-tenant org handlers ────────────────────────────────────────────

    /// POST /api/orgs — create a new organization.
    ///
    /// Body: `{ "name": "Acme Corp" }`
    /// Requires a valid API key (the caller becomes the org owner).
    fn handle_create_org(&self, request: &str) -> String {
        let api_key = match self.extract_header(request, "x-api-key") {
            Some(k) if !k.is_empty() => k,
            _ => return self.error_response(401, "Missing x-api-key header"),
        };

        let body = self.extract_body(request);
        let req: serde_json::Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(_) => return self.error_response(400, "Invalid JSON body"),
        };

        let name = match req.get("name").and_then(|v| v.as_str()) {
            Some(n) if !n.trim().is_empty() => n.trim().to_string(),
            _ => return self.error_response(400, "Missing required field: name"),
        };

        let mut db = match self.org_db.lock() {
            Ok(g) => g,
            Err(_) => return self.error_response(500, "Internal lock error"),
        };

        match db.create_org(name, api_key) {
            Ok(org) => {
                let body = serde_json::to_string(&org).unwrap_or_default();
                format!("HTTP/1.1 201 Created\r\nContent-Type: application/json\r\n\r\n{}", body)
            }
            Err(e) => self.error_response(400, &e),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
        assert!(config.cache_enabled);
    }
    
    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(2);
        assert!(limiter.check_rate_limit("client1"));
        assert!(limiter.check_rate_limit("client1"));
        assert!(!limiter.check_rate_limit("client1"));
        
        assert!(limiter.check_rate_limit("client2"));
    }
}