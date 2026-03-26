use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::thread;
use std::net::SocketAddr;
use std::io::Write;
use base64::{Engine as _, engine::general_purpose};
use image::{ImageFormat, GenericImage};
use log::{info, warn};
use crate::error::SrganError;
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::api::billing::{BillingDb, BillingStatus, CheckoutRequest, SubscriptionTier};
use crate::api::middleware::TierRateLimiter;
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
        info!("  GET  /api/v1/job/{{id}}       - Check single-image job status");
        info!("  GET  /api/v1/health          - Health check");
        info!("  GET  /api/models             - List available models");
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
            
            // Read request (simplified)
            let mut buffer = [0; 1024];
            let _ = stream.read(&mut buffer);
            
            // Parse request line
            let request = String::from_utf8_lossy(&buffer);
            let lines: Vec<&str> = request.lines().collect();
            
            if lines.is_empty() {
                continue;
            }
            
            let parts: Vec<&str> = lines[0].split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            
            let method = parts[0];
            let path = parts[1];
            
            // Route request
            let response = match (method, path) {
                ("GET", "/api/health") | ("GET", "/api/v1/health") => self.handle_health_check(),
                ("GET", "/api/models") => self.handle_list_models(),
                ("GET", "/dashboard") => self.handle_dashboard(),
                ("POST", "/api/upscale") | ("POST", "/api/v1/upscale") => self.handle_upscale_sync(&request),
                ("POST", "/api/upscale/async") | ("POST", "/api/v1/upscale/async") => self.handle_upscale_async(&request),
                ("POST", "/api/batch") | ("POST", "/api/v1/batch") => self.handle_batch(&request),
                ("POST", "/api/v1/detect") => self.handle_detect(&request),
                ("POST", "/api/v1/billing/checkout") => self.handle_billing_checkout(&request),
                ("POST", "/api/v1/billing/webhook") => self.handle_billing_webhook(&request),
                ("GET", "/api/v1/billing/status") => self.handle_billing_status(&request),
                _ if method == "GET" && (path.starts_with("/api/job/") || path.starts_with("/api/v1/job/")) => self.handle_job_status(path),
                _ if method == "GET" && (path.starts_with("/api/result/") || path.starts_with("/api/v1/result/")) => self.handle_job_result(path),
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
    
    /// Handle list models
    fn handle_list_models(&self) -> String {
        let response = serde_json::json!({
            "models": ["natural", "anime", "custom"],
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

        // Run SRGAN inference
        let upscaled = self.network.upscale_image(&img)?;
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

        if let Ok(jobs) = self.batch_jobs.lock() {
            if let Some(job) = jobs.get(batch_id) {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                    serde_json::to_string(job).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
                )
            } else {
                self.error_response(404, "Batch job not found")
            }
        } else {
            self.error_response(500, "Failed to acquire batch lock")
        }
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
    fn handle_dashboard(&self) -> String {
        let uptime_secs = SystemTime::now()
            .duration_since(self.server_start_time)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let images_processed = self.images_processed.load(Ordering::Relaxed);

        // Count jobs by status
        let (queued, processing, completed, failed) =
            if let Ok(jobs) = self.jobs.lock() {
                let mut q = 0u64; let mut p = 0u64; let mut c = 0u64; let mut f = 0u64;
                for job in jobs.values() {
                    match &job.status {
                        JobStatus::Pending    => q += 1,
                        JobStatus::Processing => p += 1,
                        JobStatus::Completed  => c += 1,
                        JobStatus::Failed(_)  => f += 1,
                    }
                }
                (q, p, c, f)
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

        let model_name = self.network.display();
        let model_factor = self.network.factor();
        let s3_status = if self.s3_config.is_some() { "configured" } else { "disabled" };
        let uptime_str = format_uptime(uptime_secs);

        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="30">
  <title>SRGAN-Rust Dashboard</title>
  <style>
    body  {{ font-family: monospace; background:#1a1a2e; color:#e0e0e0; padding:2rem; margin:0; }}
    h1    {{ color:#00d4ff; margin-bottom:0.25rem; }}
    h2    {{ color:#7ec8e3; border-bottom:1px solid #333; padding-bottom:0.3rem; margin-top:2rem; }}
    .meta {{ color:#888; font-size:0.85rem; margin-bottom:1.5rem; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:1rem; margin:1rem 0; }}
    .card {{ background:#16213e; border:1px solid #0f3460; border-radius:8px; padding:1rem; }}
    .stat {{ font-size:1.8rem; color:#00d4ff; font-weight:bold; }}
    .label{{ color:#888; font-size:0.75rem; margin-top:0.25rem; }}
    .ok   {{ color:#4caf50; }} .warn {{ color:#ff9800; }}
    .badge{{ display:inline-block; padding:0.15rem 0.5rem; border-radius:4px;
             background:#0f3460; color:#00d4ff; font-size:0.8rem; }}
  </style>
</head>
<body>
  <h1>SRGAN-Rust Dashboard</h1>
  <p class="meta">
    Model: <span class="ok">{model_name} {model_factor}x</span> &nbsp;|&nbsp;
    Uptime: <strong>{uptime_str}</strong> &nbsp;|&nbsp;
    Version: <span class="badge">0.2.0</span> &nbsp;|&nbsp;
    S3: <span class="{s3_class}">{s3_status}</span>
  </p>

  <h2>Job Queue</h2>
  <div class="grid">
    <div class="card"><div class="stat">{queued}</div><div class="label">Queued</div></div>
    <div class="card"><div class="stat">{processing}</div><div class="label">Processing</div></div>
    <div class="card"><div class="stat">{completed}</div><div class="label">Completed</div></div>
    <div class="card"><div class="stat">{failed}</div><div class="label">Failed</div></div>
    <div class="card"><div class="stat">{images_processed}</div><div class="label">Total Processed</div></div>
  </div>

  <h2>Credits (Today)</h2>
  <div class="grid">
    <div class="card"><div class="stat">{credits_issued}</div><div class="label">Issued</div></div>
    <div class="card"><div class="stat">{credits_consumed}</div><div class="label">Consumed</div></div>
  </div>

  <h2>System</h2>
  <div class="grid">
    <div class="card"><div class="stat">{load_avg:.2}</div><div class="label">Load Avg (1m)</div></div>
    <div class="card"><div class="stat">{mem_used_mb} MB</div><div class="label">RAM Used</div></div>
    <div class="card"><div class="stat">{mem_total_mb} MB</div><div class="label">RAM Total</div></div>
  </div>
</body>
</html>"#,
            model_name = model_name,
            model_factor = model_factor,
            uptime_str = uptime_str,
            s3_class = if self.s3_config.is_some() { "ok" } else { "warn" },
            s3_status = s3_status,
            queued = queued,
            processing = processing,
            completed = completed,
            failed = failed,
            images_processed = images_processed,
            credits_issued = credits_issued,
            credits_consumed = credits_consumed,
            load_avg = load_avg,
            mem_used_mb = mem_used_mb,
            mem_total_mb = mem_total_mb,
        );

        format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{}", html)
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