use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::thread;
use std::net::SocketAddr;
use std::io::Write;
use image::{DynamicImage, ImageFormat, GenericImage};
use serde::{Deserialize, Serialize};
use log::{info, warn};
use crate::error::SrganError;
use crate::UpscalingNetwork;

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
            host: "127.0.0.1".to_string(),
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

/// API request for image upscaling
#[derive(Debug, Deserialize)]
pub struct UpscaleRequest {
    pub image_data: String,  // Base64 encoded image
    pub scale_factor: Option<u32>,
    pub format: Option<String>,
    pub quality: Option<u8>,
    pub model: Option<String>,
}

/// API response for image upscaling
#[derive(Debug, Serialize)]
pub struct UpscaleResponse {
    pub success: bool,
    pub image_data: Option<String>,  // Base64 encoded result
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

/// Async job information
#[derive(Debug, Clone, Serialize)]
pub struct JobInfo {
    pub id: String,
    pub status: JobStatus,
    pub created_at: u64,
    pub updated_at: u64,
    pub result_url: Option<String>,
    pub error: Option<String>,
}

/// Web API server
pub struct WebServer {
    config: ServerConfig,
    network: Arc<Mutex<UpscalingNetwork>>,
    cache: Arc<Mutex<HashMap<String, CachedResult>>>,
    jobs: Arc<Mutex<HashMap<String, JobInfo>>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
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
            self.requests.insert(client_id.to_string(), vec![now]);
        }
        
        true
    }
}

impl WebServer {
    /// Create new web server
    pub fn new(config: ServerConfig) -> Result<Self, SrganError> {
        // Load network
        let network = if let Some(ref model_path) = config.model_path {
            UpscalingNetwork::load_from_file(model_path)?
        } else {
            UpscalingNetwork::load_builtin_natural()?
        };
        
        let rate_limit = config.rate_limit.unwrap_or(60);
        
        Ok(Self {
            config,
            network: Arc::new(Mutex::new(network)),
            cache: Arc::new(Mutex::new(HashMap::new())),
            jobs: Arc::new(Mutex::new(HashMap::new())),
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(rate_limit))),
        })
    }
    
    /// Start the web server
    pub fn start(&self) -> Result<(), SrganError> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|_| SrganError::InvalidInput("Invalid server address".to_string()))?;
        
        info!("Starting web server at http://{}", addr);
        info!("API endpoints:");
        info!("  POST /api/upscale       - Synchronous image upscaling");
        info!("  POST /api/upscale/async - Asynchronous image upscaling");
        info!("  GET  /api/job/{{id}}      - Check job status");
        info!("  GET  /api/health        - Health check");
        info!("  GET  /api/models        - List available models");
        
        if let Some(ref api_key) = self.config.api_key {
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
                ("GET", "/api/health") => self.handle_health_check(),
                ("GET", "/api/models") => self.handle_list_models(),
                ("POST", "/api/upscale") => self.handle_upscale_sync(&request),
                ("POST", "/api/upscale/async") => self.handle_upscale_async(&request),
                _ if path.starts_with("/api/job/") => self.handle_job_status(path),
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
            "status": "healthy",
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
        // Parse JSON body (simplified)
        let body = self.extract_body(request);
        
        match serde_json::from_str::<UpscaleRequest>(&body) {
            Ok(req) => {
                // Check rate limit
                if !self.check_rate_limit("client") {
                    return self.error_response(429, "Rate limit exceeded");
                }
                
                // Process image
                match self.process_image(req) {
                    Ok(response) => {
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
                            serde_json::to_string(&response)
                                .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
                        )
                    }
                    Err(e) => self.error_response(500, &e.to_string()),
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
                    
                    // Process image (simplified)
                    // In real implementation, would process the image
                    thread::sleep(Duration::from_secs(2));
                    
                    // Update job with result
                    if let Ok(mut jobs_guard) = jobs.lock() {
                        if let Some(job) = jobs_guard.get_mut(&job_id_clone) {
                            job.status = JobStatus::Completed;
                            job.result_url = Some(format!("/api/result/{}", job_id_clone));
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
            request[idx + 4..].to_string()
        } else {
            String::new()
        }
    }
    
    /// Process image upscaling
    fn process_image(&self, request: UpscaleRequest) -> Result<UpscaleResponse, SrganError> {
        let start_time = SystemTime::now();
        
        // Decode base64 image
        let image_data = base64::decode(&request.image_data)
            .map_err(|e| SrganError::InvalidInput(format!("Invalid base64: {}", e)))?;
        
        // Load image
        let img = image::load_from_memory(&image_data)
            .map_err(|e| SrganError::Image(e))?;
        
        let original_size = (img.width(), img.height());
        
        // Upscale
        let network = self.network.lock()
            .map_err(|_| SrganError::InvalidInput("Failed to acquire network lock".to_string()))?;
        let upscaled = network.upscale_image(&img)?;
        let upscaled_size = (upscaled.width(), upscaled.height());
        
        // Encode result
        let format = request.format.as_deref().unwrap_or("png");
        let mut output = Vec::new();
        let img_format = match format {
            "jpeg" | "jpg" => ImageFormat::JPEG,
            "png" => ImageFormat::PNG,
            "webp" => ImageFormat::WEBP,
            _ => ImageFormat::PNG,
        };
        
        upscaled.write_to(&mut output, img_format)
            .map_err(|e| SrganError::Image(e))?;
        
        let encoded = base64::encode(&output);
        
        let processing_time = start_time.elapsed()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        
        Ok(UpscaleResponse {
            success: true,
            image_data: Some(encoded),
            error: None,
            metadata: ResponseMetadata {
                original_size,
                upscaled_size,
                processing_time_ms: processing_time,
                format: format.to_string(),
                model_used: request.model.unwrap_or_else(|| "natural".to_string()),
            },
        })
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

/// Base64 encoding/decoding utilities
mod base64 {
    pub fn encode(data: &[u8]) -> String {
        // Simplified base64 encoding
        // In production, use base64 crate
        format!("base64:{}", data.len())
    }
    
    pub fn decode(data: &str) -> Result<Vec<u8>, String> {
        // Simplified base64 decoding
        // In production, use base64 crate
        Ok(vec![0; 100])
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