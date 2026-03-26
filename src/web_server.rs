use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::thread;
use std::net::SocketAddr;
use std::io::{BufRead, BufReader, Write};
use base64::{Engine as _, engine::general_purpose};
use image::{ImageFormat, GenericImage};
use log::{info, warn};
use crate::error::SrganError;
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::api::billing::{BillingDb, BillingStatus, CheckoutRequest, SubscriptionTier};
use crate::api::middleware::TierRateLimiter;
use crate::api::upscale::{deliver_webhook, unix_now as api_unix_now, WebhookConfig, WebhookDeliveryState};
use crate::storage::S3Config;

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
        info!("  POST /api/v1/compare         - Multi-model PSNR/SSIM comparison");
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
            
            // SSE stream endpoint: takes ownership of `stream` and writes events
            // directly, bypassing the normal single-string response path.
            if method == "GET"
                && (path.starts_with("/api/v1/job/") || path.starts_with("/api/job/"))
                && path.ends_with("/stream")
            {
                let path_owned = path.to_string();
                self.handle_job_stream(stream, &path_owned);
                continue;
            }

            // Route request
            let response = match (method, path) {
                ("GET", "/") => self.handle_root_dashboard(),
                ("GET", "/admin") => self.handle_admin_panel(),
                ("GET", "/api/me") => self.handle_api_me(&request),
                ("GET", "/api/admin/users") => self.handle_admin_users(&request),
                ("GET", "/api/health") | ("GET", "/api/v1/health") => self.handle_health_check(),
                ("GET", "/api/models") | ("GET", "/api/v1/models") => self.handle_list_models(),
                ("GET", "/api/v1/stats") => self.handle_stats(),
                ("GET", "/api/v1/jobs") | ("GET", "/api/jobs") => self.handle_jobs(),
                ("GET", "/dashboard") => self.handle_dashboard(),
                ("POST", "/api/upscale") | ("POST", "/api/v1/upscale") => self.handle_upscale_sync(&request),
                ("POST", "/api/upscale/async") | ("POST", "/api/v1/upscale/async") => self.handle_upscale_async(&request),
                ("POST", "/api/batch") | ("POST", "/api/v1/batch") => self.handle_batch(&request),
                ("POST", "/api/v1/batch/start") => self.handle_batch_dir_start(&request),
                ("POST", "/api/v1/detect") => self.handle_detect(&request),
                ("POST", "/api/v1/billing/checkout") => self.handle_billing_checkout(&request),
                ("POST", "/api/v1/billing/webhook") => self.handle_billing_webhook(&request),
                ("GET", "/api/v1/billing/status") => self.handle_billing_status(&request),
                ("POST", "/api/v1/compare") => self.handle_compare(&request),
                ("POST", "/api/v1/webhook/test") => self.handle_webhook_test(&request),
                _ if method == "GET"
                    && (path.starts_with("/api/v1/job/") || path.starts_with("/api/job/"))
                    && path.ends_with("/webhook") => self.handle_job_webhook_status(path),
                _ if method == "GET" && (path.starts_with("/api/job/") || path.starts_with("/api/v1/job/")) => self.handle_job_status(path),
                _ if method == "GET" && (path.starts_with("/api/result/") || path.starts_with("/api/v1/result/")) => self.handle_job_result(path),
                _ if method == "GET" && (path.starts_with("/api/v1/batch/") || path.starts_with("/api/batch/")) && path.ends_with("/checkpoint") => self.handle_batch_checkpoint(path),
                _ if method == "GET" && (path.starts_with("/api/batch/") || path.starts_with("/api/v1/batch/")) => self.handle_batch_status(path),
                _ => self.handle_not_found(),
            };
            
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
                "description": "Waifu2x-style model for anime/illustration upscaling with configurable noise reduction (noise_level 0–3, scale 1×/2×)",
                "architecture": "waifu2x",
                "scale_factors": [1, 2],
                "recommended_for": ["anime", "illustrations", "photos"],
                "parameters": {
                    "waifu2x_noise_level": "0–3 (0 = none, 3 = aggressive; default 1)",
                    "waifu2x_scale": "1 or 2 (default 2)"
                },
                "variants": crate::waifu2x::WAIFU2X_LABELS,
                "source": "built-in"
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

    // ── New dashboard / admin endpoints ──────────────────────────────────────

    /// GET / — main web dashboard (SPA; all data fetched via JS)
    fn handle_root_dashboard(&self) -> String {
        let html = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SRGAN-Rust</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--accent:#00d4ff;
  --green:#3fb950;--yellow:#d29922;--red:#f85149;
  --text:#c9d1d9;--muted:#8b949e;--r:8px;--font:ui-monospace,'Menlo',monospace}
html,body{height:100%;font-family:var(--font);background:var(--bg);color:var(--text);font-size:14px}
.shell{display:grid;grid-template-columns:200px 1fr;height:calc(100vh - 40px)}
.sidebar{background:var(--bg2);border-right:1px solid var(--border);padding:1.25rem .9rem;
  display:flex;flex-direction:column;gap:.35rem}
.main{padding:1.5rem;overflow-y:auto}
.logo{color:var(--accent);font-size:1rem;font-weight:700;margin-bottom:.9rem;letter-spacing:.03em}
.nav{display:flex;align-items:center;gap:.5rem;padding:.42rem .65rem;border-radius:var(--r);
  color:var(--muted);cursor:pointer;transition:all .15s;font-size:.86rem;user-select:none}
.nav:hover,.nav.active{background:var(--bg3);color:var(--accent)}
.page{display:none}.page.active{display:block}
.section-title{font-size:1.2rem;color:var(--accent);font-weight:700;margin-bottom:1.1rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.65rem;margin-bottom:1.4rem}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:.85rem}
.card-val{font-size:1.5rem;font-weight:700;color:var(--accent);line-height:1}
.card-lbl{color:var(--muted);font-size:.68rem;margin-top:.28rem;text-transform:uppercase;letter-spacing:.05em}
.card.g .card-val{color:var(--green)}.card.y .card-val{color:var(--yellow)}.card.r .card-val{color:var(--red)}
.tbl-wrap{border:1px solid var(--border);border-radius:var(--r);overflow:auto;margin-bottom:1.2rem}
table{width:100%;border-collapse:collapse;font-size:.81rem}
thead th{background:var(--bg3);color:var(--muted);text-align:left;padding:.5rem .75rem;
  border-bottom:1px solid var(--border);font-size:.68rem;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
tbody tr{border-bottom:1px solid var(--border);cursor:pointer;transition:background .1s}
tbody tr:hover{background:var(--bg3)}tbody tr:last-child{border-bottom:none}
td{padding:.45rem .75rem;vertical-align:middle;white-space:nowrap}
.badge{display:inline-block;padding:.1rem .38rem;border-radius:4px;font-size:.7rem;font-weight:600}
.bp{background:#21262d;color:var(--yellow);border:1px solid #9e6a03}
.bpr{background:#21262d;color:var(--accent);border:1px solid #0f6ab6}
.bc{background:#21262d;color:var(--green);border:1px solid #238636}
.bf{background:#21262d;color:var(--red);border:1px solid #8b2121}
.thumb{width:44px;height:44px;object-fit:cover;border-radius:4px;border:1px solid var(--border)}
.tp{width:44px;height:44px;background:var(--bg3);border-radius:4px;border:1px solid var(--border);
  display:inline-flex;align-items:center;justify-content:center;color:var(--muted);font-size:.6rem}
.form-group{margin-bottom:.9rem}
label{display:block;color:var(--muted);font-size:.78rem;margin-bottom:.35rem}
input[type=text],select{width:100%;background:var(--bg2);border:1px solid var(--border);
  border-radius:var(--r);padding:.45rem .65rem;color:var(--text);font-family:var(--font);font-size:.86rem}
input[type=text]:focus,select:focus{outline:none;border-color:var(--accent)}
.btn{display:inline-block;padding:.42rem .9rem;border-radius:var(--r);font-family:var(--font);
  font-size:.86rem;cursor:pointer;border:none;transition:opacity .15s}
.btn-p{background:var(--accent);color:#000;font-weight:600}.btn-p:hover{opacity:.85}
.btn-s{background:var(--bg3);color:var(--text);border:1px solid var(--border)}.btn-s:hover{border-color:var(--accent)}
.btn:disabled{opacity:.45;cursor:not-allowed}
.dropzone{border:2px dashed var(--border);border-radius:var(--r);padding:1.75rem;text-align:center;
  cursor:pointer;transition:all .2s;color:var(--muted)}
.dropzone.drag,.dropzone:hover{border-color:var(--accent);background:rgba(0,212,255,.04)}
.api-bar{background:var(--bg2);border-bottom:1px solid var(--border);
  padding:.5rem 1rem;display:flex;align-items:center;gap:.6rem;font-size:.8rem;height:40px}
.api-bar input{flex:none;width:220px;font-size:.78rem;padding:.25rem .5rem;background:var(--bg);
  border:1px solid var(--border);border-radius:4px;color:var(--text);font-family:var(--font)}
.pulse{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:100;align-items:center;justify-content:center}
.overlay.open{display:flex}
.modal{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);
  padding:1.4rem;max-width:620px;width:90vw;max-height:80vh;overflow-y:auto}
.modal-title{font-size:1rem;color:var(--accent);margin-bottom:.9rem;font-weight:700}
.close-x{float:right;cursor:pointer;color:var(--muted);font-size:1.1rem;line-height:1}.close-x:hover{color:var(--text)}
.sidebar-ft{margin-top:auto;padding-top:.9rem;border-top:1px solid var(--border)}
</style>
</head>
<body>
<div class="api-bar">
  <span class="pulse"></span>
  <span style="color:var(--text);font-weight:600">SRGAN-Rust</span>
  <span style="color:var(--border)">|</span>
  <label for="apiK" style="margin:0">API Key:</label>
  <input type="text" id="apiK" placeholder="sk-…">
  <button class="btn btn-s" onclick="saveKey()" style="padding:.2rem .6rem;font-size:.76rem">Save</button>
  <span id="keySt" style="color:var(--green);font-size:.76rem"></span>
  <span style="margin-left:auto;color:var(--muted);font-size:.76rem" id="refreshTime"></span>
</div>
<div class="shell">
  <nav class="sidebar">
    <div class="logo">&#9650; SRGAN</div>
    <div class="nav active" id="nav-ov" onclick="nav('ov')">&#128202; Overview</div>
    <div class="nav" id="nav-jobs" onclick="nav('jobs')">&#128736; Jobs</div>
    <div class="nav" id="nav-sub" onclick="nav('sub')">&#128228; Submit Job</div>
    <div class="sidebar-ft">
      <div class="nav" onclick="location='/admin'" style="color:var(--muted)">&#128274; Admin</div>
    </div>
  </nav>
  <main class="main">
    <div class="page active" id="pg-ov">
      <div class="section-title">Overview</div>
      <div class="cards">
        <div class="card"><div class="card-val" id="cPend">–</div><div class="card-lbl">Pending</div></div>
        <div class="card"><div class="card-val" id="cProc">–</div><div class="card-lbl">Processing</div></div>
        <div class="card g"><div class="card-val" id="cComp">–</div><div class="card-lbl">Completed</div></div>
        <div class="card r"><div class="card-val" id="cFail">–</div><div class="card-lbl">Failed</div></div>
        <div class="card"><div class="card-val" id="cCred">–</div><div class="card-lbl">Credits</div></div>
        <div class="card y"><div class="card-val" id="cUp">–</div><div class="card-lbl">Uptime</div></div>
        <div class="card"><div class="card-val" id="cMem">–</div><div class="card-lbl">RAM</div></div>
        <div class="card"><div class="card-val" id="cLoad">–</div><div class="card-lbl">Load</div></div>
      </div>
      <div style="font-weight:600;margin-bottom:.65rem;font-size:.88rem">Recent Jobs</div>
      <div class="tbl-wrap" id="recentTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    </div>
    <div class="page" id="pg-jobs">
      <div class="section-title">All Jobs</div>
      <div class="tbl-wrap" id="allTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    </div>
    <div class="page" id="pg-sub">
      <div class="section-title">Submit Job</div>
      <div style="max-width:500px">
        <div id="dz" class="dropzone" onclick="document.getElementById('fi').click()">
          <div style="font-size:1.8rem;margin-bottom:.4rem">&#128247;</div>
          <div>Drag &amp; drop an image or <strong style="color:var(--accent)">browse</strong></div>
          <div style="font-size:.75rem;margin-top:.3rem">PNG · JPG · WebP · max 50 MB</div>
        </div>
        <input type="file" id="fi" accept="image/*" style="display:none">
        <div id="prevWrap" style="margin:.65rem 0;display:none">
          <img id="prevImg" style="max-width:100%;max-height:180px;border-radius:var(--r);border:1px solid var(--border)">
          <div id="prevName" style="font-size:.76rem;color:var(--muted);margin-top:.25rem"></div>
        </div>
        <div class="form-group">
          <label>Model</label>
          <select id="modelSel">
            <option value="natural">Natural (photos)</option>
            <option value="anime">Anime / illustration</option>
            <option value="waifu2x">Waifu2x</option>
            <option value="real-esrgan">Real-ESRGAN</option>
          </select>
        </div>
        <button class="btn btn-p" id="subBtn" onclick="submitJob()" disabled>Submit Job</button>
        <div id="subSt" style="margin-top:.65rem;font-size:.84rem"></div>
      </div>
    </div>
  </main>
</div>
<div class="overlay" id="modal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal">
    <span class="close-x" onclick="document.getElementById('modal').classList.remove('open')">&#10005;</span>
    <div class="modal-title" id="modalTtl">Job Detail</div>
    <div id="modalBdy"></div>
  </div>
</div>
<script>
var apiKey=localStorage.getItem('srgan_k')||'';
var jobs=[];var thumbs={};var selFile=null;
document.getElementById('apiK').value=apiKey;
setupDz();
setInterval(refresh,2000);
refresh();

function saveKey(){apiKey=document.getElementById('apiK').value.trim();localStorage.setItem('srgan_k',apiKey);var s=document.getElementById('keySt');s.textContent='Saved';setTimeout(function(){s.textContent=''},1500);}

function hdrs(){var h={'Content-Type':'application/json'};if(apiKey)h['X-API-Key']=apiKey;return h;}

function nav(p){
  document.querySelectorAll('.page').forEach(function(e){e.classList.remove('active');});
  document.querySelectorAll('.nav').forEach(function(e){e.classList.remove('active');});
  document.getElementById('pg-'+p).classList.add('active');
  document.getElementById('nav-'+p).classList.add('active');
}

function refresh(){
  var h=hdrs();
  Promise.all([
    fetch('/api/v1/stats',{headers:h}).then(function(r){return r.ok?r.json():null;}),
    fetch('/api/v1/jobs',{headers:h}).then(function(r){return r.ok?r.json():null;}),
    fetch('/api/me',{headers:h}).then(function(r){return r.ok?r.json():null;}).catch(function(){return null;})
  ]).then(function(res){
    var stats=res[0];var jData=res[1];var me=res[2];
    if(stats){
      set('cPend',stats.jobs&&stats.jobs.pending!=null?stats.jobs.pending:'–');
      set('cProc',stats.jobs&&stats.jobs.processing!=null?stats.jobs.processing:'–');
      set('cComp',stats.jobs&&stats.jobs.completed!=null?stats.jobs.completed:'–');
      set('cFail',stats.jobs&&stats.jobs.failed!=null?stats.jobs.failed:'–');
      set('cUp',fmtUp(stats.uptime_secs||0));
      set('cMem',(stats.system&&stats.system.mem_used_mb||0)+' MB');
      set('cLoad',((stats.system&&stats.system.load_avg_1m)||0).toFixed(2));
    }
    if(me)set('cCred',me.credits_remaining!=null?me.credits_remaining:'–');
    if(jData)jobs=jData.jobs||[];
    renderTbls();
    fetchThumbs();
    document.getElementById('refreshTime').textContent='Updated '+new Date().toLocaleTimeString();
  }).catch(function(){});
}

function set(id,v){var e=document.getElementById(id);if(e)e.textContent=v;}

function fmtUp(s){
  if(s<60)return s+'s';
  if(s<3600)return Math.floor(s/60)+'m';
  if(s<86400)return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';
  return Math.floor(s/86400)+'d '+Math.floor((s%86400)/3600)+'h';
}

function badge(st){
  var s=st||'';
  if(s==='completed')return '<span class="badge bc">completed</span>';
  if(s==='processing')return '<span class="badge bpr">processing</span>';
  if(s.startsWith('failed'))return '<span class="badge bf">failed</span>';
  return '<span class="badge bp">'+s+'</span>';
}

function thumbCell(j){
  if(thumbs[j.id])return '<img class="thumb" src="'+thumbs[j.id]+'" alt="">';
  return '<div class="tp">'+(j.status==='completed'?'…':'')+'</div>';
}

function jobRow(j){
  var dur=j.duration_secs!=null?j.duration_secs.toFixed(1)+'s':'–';
  var t=new Date(j.created_at*1000).toLocaleTimeString();
  return '<tr onclick="detail(\''+j.id+'\')">'
    +'<td>'+thumbCell(j)+'</td>'
    +'<td><code style="font-size:.76rem">'+j.id.slice(0,8)+'…</code></td>'
    +'<td>'+badge(j.status)+'</td>'
    +'<td>'+(j.model||'–')+'</td>'
    +'<td>'+(j.input_size||'–')+'</td>'
    +'<td>'+(j.output_size||'–')+'</td>'
    +'<td>'+dur+'</td>'
    +'<td>'+t+'</td>'
    +'</tr>';
}

function tblHTML(list){
  if(!list.length)return '<div style="padding:1.75rem;text-align:center;color:var(--muted)">No jobs yet</div>';
  return '<table><thead><tr><th></th><th>Job ID</th><th>Status</th><th>Model</th><th>Input</th><th>Output</th><th>Dur.</th><th>Time</th></tr></thead>'
    +'<tbody>'+list.map(jobRow).join('')+'</tbody></table>';
}

function renderTbls(){
  var r=document.getElementById('recentTbl');if(r)r.innerHTML=tblHTML(jobs.slice(0,10));
  var a=document.getElementById('allTbl');if(a)a.innerHTML=tblHTML(jobs);
}

function fetchThumbs(){
  var need=jobs.filter(function(j){return j.status==='completed'&&!thumbs[j.id];}).slice(0,5);
  need.forEach(function(j){
    fetch('/api/v1/job/'+j.id,{headers:hdrs()}).then(function(r){return r.ok?r.json():null;}).then(function(d){
      if(d&&d.result_data){thumbs[j.id]='data:image/png;base64,'+d.result_data;renderTbls();}
    }).catch(function(){});
  });
}

function detail(id){
  document.getElementById('modalTtl').textContent='Job: '+id;
  document.getElementById('modalBdy').innerHTML='Loading…';
  document.getElementById('modal').classList.add('open');
  fetch('/api/v1/job/'+id,{headers:hdrs()}).then(function(r){return r.json();}).then(function(d){
    var st='–';
    if(d.status==='Completed'||d.status==='completed')st='completed';
    else if(d.status==='Pending'||d.status==='pending')st='pending';
    else if(d.status==='Processing'||d.status==='processing')st='processing';
    else if(typeof d.status==='object'&&d.status!==null){
      if(d.status.Completed!=null)st='completed';
      else if(d.status.Pending!=null)st='pending';
      else if(d.status.Processing!=null)st='processing';
      else if(d.status.Failed!=null)st='failed: '+d.status.Failed;
    }
    var html='<table style="width:100%;font-size:.82rem">'
      +'<tr><td style="color:var(--muted);padding:.28rem .45rem">Status</td><td>'+badge(st)+'</td></tr>'
      +'<tr><td style="color:var(--muted);padding:.28rem .45rem">Model</td><td>'+(d.model||'–')+'</td></tr>'
      +'<tr><td style="color:var(--muted);padding:.28rem .45rem">Input</td><td>'+(d.input_size||'–')+'</td></tr>'
      +'<tr><td style="color:var(--muted);padding:.28rem .45rem">Output</td><td>'+(d.output_size||'–')+'</td></tr>'
      +'<tr><td style="color:var(--muted);padding:.28rem .45rem">Created</td><td>'+new Date(d.created_at*1000).toLocaleString()+'</td></tr>'
      +'</table>';
    if(d.result_data){
      var src='data:image/png;base64,'+d.result_data;
      thumbs[id]=src;
      html+='<img src="'+src+'" style="max-width:100%;border-radius:var(--r);margin-top:.9rem;display:block">'
        +'<a href="'+src+'" download="result_'+id+'.png" class="btn btn-s" style="margin-top:.5rem;display:inline-block;text-decoration:none">&#8595; Download</a>';
    }
    if(d.error)html+='<div style="color:var(--red);margin-top:.65rem">'+d.error+'</div>';
    document.getElementById('modalBdy').innerHTML=html;
    renderTbls();
  }).catch(function(){document.getElementById('modalBdy').innerHTML='<span style="color:var(--red)">Error loading job</span>';});
}

function setupDz(){
  var dz=document.getElementById('dz');
  var fi=document.getElementById('fi');
  dz.addEventListener('dragover',function(e){e.preventDefault();dz.classList.add('drag');});
  dz.addEventListener('dragleave',function(){dz.classList.remove('drag');});
  dz.addEventListener('drop',function(e){e.preventDefault();dz.classList.remove('drag');if(e.dataTransfer.files[0])loadFile(e.dataTransfer.files[0]);});
  fi.addEventListener('change',function(){if(fi.files[0])loadFile(fi.files[0]);});
}

function loadFile(f){
  selFile=f;
  var r=new FileReader();
  r.onload=function(e){document.getElementById('prevImg').src=e.target.result;};
  r.readAsDataURL(f);
  document.getElementById('prevWrap').style.display='block';
  document.getElementById('prevName').textContent=f.name+' ('+(f.size/1024).toFixed(1)+' KB)';
  document.getElementById('subBtn').disabled=false;
}

function submitJob(){
  if(!selFile)return;
  var btn=document.getElementById('subBtn');
  var st=document.getElementById('subSt');
  btn.disabled=true;
  st.textContent='Reading file…';
  var r=new FileReader();
  r.onload=function(e){
    var b64=e.target.result;
    if(b64.indexOf(',')>-1)b64=b64.split(',')[1];
    var model=document.getElementById('modelSel').value;
    st.textContent='Submitting…';
    fetch('/api/v1/upscale/async',{
      method:'POST',
      headers:hdrs(),
      body:JSON.stringify({image_data:b64,model:model})
    }).then(function(res){
      return res.json().then(function(d){return{ok:res.ok,d:d};});
    }).then(function(r){
      if(!r.ok){st.innerHTML='<span style="color:var(--red)">Error: '+(r.d.error||'Unknown')+'</span>';btn.disabled=false;return;}
      st.innerHTML='<span style="color:var(--green)">Submitted! Job ID: <code>'+r.d.job_id+'</code></span>';
      btn.disabled=false;
      nav('jobs');
    }).catch(function(e){st.innerHTML='<span style="color:var(--red)">'+e.message+'</span>';btn.disabled=false;});
  };
  r.readAsDataURL(selFile);
}
</script>
</body>
</html>"##;
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            html.len(), html
        )
    }

    /// GET /admin — admin panel SPA (protected by SRGAN_ADMIN_SECRET bearer token)
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
    <input type="password" id="tokInp" placeholder="Admin secret (SRGAN_ADMIN_SECRET)" onkeydown="if(event.key==='Enter')login()">
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
    </div>
    <div class="sec-ttl">Users</div>
    <div class="tbl-wrap" id="usersTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
    <div class="sec-ttl">Recent Jobs</div>
    <div class="tbl-wrap" id="jobsTbl"><div style="padding:1.5rem;color:var(--muted);text-align:center">Loading…</div></div>
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
    fetch('/api/v1/jobs',{headers:h}).then(function(r){return r.ok?r.json():null;})
  ]).then(function(res){
    if(res[0])renderUsers(res[0].users||[]);
    if(res[1])renderMetrics(res[1]);
    if(res[2])renderJobs(res[2].jobs||[]);
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
  var html='<table><thead><tr><th>API Key</th><th>Tier</th><th>Credits Left</th><th>Issued Today</th><th>Consumed Today</th></tr></thead><tbody>';
  users.forEach(function(u){
    html+='<tr>'
      +'<td><code>'+u.user_id.slice(0,24)+(u.user_id.length>24?'…':'')+'</code></td>'
      +'<td><span class="badge tier-'+u.tier+'">'+u.tier+'</span></td>'
      +'<td>'+u.credits_remaining+'</td>'
      +'<td>'+u.credits_issued_today+'</td>'
      +'<td>'+u.credits_consumed_today+'</td>'
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
        let api_key = self
            .extract_header(request, "x-api-key")
            .unwrap_or_else(|| "anonymous".to_string());

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

    /// Returns true when the request carries a valid admin bearer token.
    fn verify_admin_token(&self, request: &str) -> bool {
        let secret = std::env::var("SRGAN_ADMIN_SECRET")
            .unwrap_or_else(|_| "srgan-admin".to_string());
        if let Some(auth) = self.extract_header(request, "authorization") {
            if let Some(token) = auth.trim().strip_prefix("Bearer ") {
                return token.trim() == secret;
            }
        }
        false
    }

    /// GET /api/admin/users — list all users (requires admin bearer token)
    fn handle_admin_users(&self, request: &str) -> String {
        if !self.verify_admin_token(request) {
            return format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n\r\n{}",
                serde_json::json!({"error": "Unauthorized"})
            );
        }

        let users = if let Ok(db) = self.billing_db.lock() {
            db.all_users_snapshot()
        } else {
            vec![]
        };

        let response = serde_json::json!({ "users": users, "count": users.len() });
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            response
        )
    }

    /// Handle not found
    fn handle_not_found(&self) -> String {
        self.error_response(404, "Not found")
    }
    
    /// Generate error response
    fn error_response(&self, status: u16, message: &str) -> String {
        let status_text = match status {
            400 => "Bad Request",
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
    /// When `model` is `"waifu2x"`, the `waifu2x_noise_level` and
    /// `waifu2x_scale` fields are used to build the canonical label
    /// (e.g. `"waifu2x-noise1-scale2"`).
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
        let original_size = (img.width(), img.height());
        let network_factor = self.network.factor();
        let requested_factor = request.scale_factor.unwrap_or(network_factor);

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

        // Select the network for the effective label.  For waifu2x labels the
        // built-in anime model is used as a fallback until native waifu2x weights
        // are bundled.  For all other labels we load the appropriate built-in
        // model; if the label is unknown we fall back to the default network.
        let upscaled = if effective_label == "natural"
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
        let upscaled_size = (upscaled.width(), upscaled.height());

        // Encode result to the requested format
        let format = request.format.as_deref().unwrap_or("png");
        let img_format = match format {
            "jpeg" | "jpg" => ImageFormat::JPEG,
            _ => ImageFormat::PNG,
        };
        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
        upscaled.write_to(&mut cursor, img_format)
            .map_err(|e| SrganError::Image(e))?;
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
                format: format.into(),
                model_used: format!("{}_{}x", effective_label, network_factor),
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
            let fmt = req.format.as_deref().unwrap_or("png");
            let img_format = match fmt {
                "jpeg" | "jpg" => ImageFormat::JPEG,
                _ => ImageFormat::PNG,
            };

            let mut results: Vec<BatchImageResult> = Vec::new();

            for (i, img_b64) in req.images.iter().enumerate() {
                let start = SystemTime::now();

                let result: std::result::Result<String, String> = (|| {
                    let data = general_purpose::STANDARD.decode(img_b64)
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
                results.push(match result {
                    Ok(enc) => BatchImageResult { index: i, success: true, image_data: Some(enc), error: None, processing_time_ms: ms },
                    Err(e)  => BatchImageResult { index: i, success: false, image_data: None, error: Some(e), processing_time_ms: ms },
                });

                // Update progress
                if let Ok(mut jobs) = batch_jobs.lock() {
                    if let Some(job) = jobs.get_mut(&bid) {
                        job.status = BatchJobStatus::Processing { completed: i + 1, total: req.images.len() };
                        job.results = results.clone();
                        job.updated_at = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                    }
                }
            }

            // Mark done
            if let Ok(mut jobs) = batch_jobs.lock() {
                if let Some(job) = jobs.get_mut(&bid) {
                    job.status = BatchJobStatus::Completed;
                    job.results = results;
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

    /// Resolve the tier for an API key (creates a free account on first use).
    fn tier_for_key(&self, api_key: &str) -> SubscriptionTier {
        if let Ok(mut db) = self.billing_db.lock() {
            db.get_or_create_free(api_key).tier.clone()
        } else {
            SubscriptionTier::Free
        }
    }

    /// Rate-limit check using the tier-aware limiter.
    /// Returns the HTTP headers string and whether the request is allowed.
    fn tier_rate_limit_check(&self, request: &str) -> (bool, String) {
        let api_key = self
            .extract_header(request, "x-api-key")
            .unwrap_or_else(|| "anonymous".to_string());
        let tier = self.tier_for_key(&api_key);
        let result = self.tier_rate_limiter.check(&api_key, &tier);
        let headers = result.headers();
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

    /// GET /api/v1/jobs — list recent jobs (newest first, up to 100)
    fn handle_jobs(&self) -> String {
        let jobs_snapshot: Vec<serde_json::Value> =
            if let Ok(jobs) = self.jobs.lock() {
                let mut list: Vec<&JobInfo> = jobs.values().collect();
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

    fn handle_dashboard(&self) -> String {
        // All live data is fetched client-side via /api/v1/stats and /api/v1/jobs.
        // Return a static self-contained SPA with no external dependencies.
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