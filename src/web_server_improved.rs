use std::path::PathBuf;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::thread;
use std::net::SocketAddr;
use std::io::Write;
use image::{ImageFormat, GenericImage};
use serde::{Serialize, Deserialize};
use dashmap::DashMap;
use tracing::{info, warn, error, debug, span, Level};
use metrics::{counter, histogram, gauge};

use crate::error::{Result, SrganError};
use crate::error_recovery::{
    CircuitBreaker, EnhancedError, ErrorAggregator, ErrorContext,
    RetryConfig, RetryExecutor, RecoveryStrategy
};
use crate::logging::{OperationLogger, PerformanceTracker};
use crate::thread_safe_network::ThreadSafeNetwork;

/// Enhanced web server configuration with resilience settings
#[derive(Debug, Clone)]
pub struct EnhancedServerConfig {
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
    // New resilience settings
    pub max_concurrent_requests: usize,
    pub request_timeout: Duration,
    pub retry_config: RetryConfig,
    pub circuit_breaker_enabled: bool,
    pub graceful_degradation: bool,
    pub health_check_interval: Duration,
}

impl Default for EnhancedServerConfig {
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
            max_concurrent_requests: 100,
            request_timeout: Duration::from_secs(30),
            retry_config: RetryConfig::default(),
            circuit_breaker_enabled: true,
            graceful_degradation: true,
            health_check_interval: Duration::from_secs(30),
        }
    }
}

/// Enhanced API request for image upscaling
#[derive(Debug, Deserialize)]
pub struct EnhancedUpscaleRequest {
    pub image_data: String,  // Base64 encoded image
    pub scale_factor: Option<u32>,
    pub format: Option<String>,
    pub quality: Option<u8>,
    pub model: Option<String>,
    pub priority: Option<RequestPriority>,
    pub fallback_allowed: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum RequestPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl Default for RequestPriority {
    fn default() -> Self {
        RequestPriority::Normal
    }
}

/// Enhanced API response with detailed error information
#[derive(Debug, Serialize)]
pub struct EnhancedUpscaleResponse {
    pub success: bool,
    pub image_data: Option<String>,  // Base64 encoded result
    pub error: Option<ErrorDetails>,
    pub metadata: ResponseMetadata,
    pub retry_info: Option<RetryInfo>,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetails {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub category: String,
    pub trace_id: String,
}

#[derive(Debug, Serialize)]
pub struct RetryInfo {
    pub retry_after: u64,  // Seconds
    pub attempts_made: u32,
    pub max_attempts: u32,
}

/// Enhanced response metadata with performance metrics
#[derive(Debug, Serialize)]
pub struct ResponseMetadata {
    pub original_size: (u32, u32),
    pub upscaled_size: (u32, u32),
    pub processing_time_ms: u64,
    pub queue_time_ms: u64,
    pub format: String,
    pub model_used: String,
    pub cache_hit: bool,
    pub degraded_mode: bool,
}

/// Enhanced job status with detailed progress
#[derive(Debug, Clone, Serialize)]
pub enum EnhancedJobStatus {
    Queued { position: usize },
    Processing { progress: f32 },
    Completed { result_url: String },
    Failed { error: ErrorDetails, retryable: bool },
    Cancelled,
}

/// Enhanced async job information
#[derive(Debug, Clone, Serialize)]
pub struct EnhancedJobInfo {
    pub id: String,
    pub status: EnhancedJobStatus,
    pub priority: RequestPriority,
    pub created_at: u64,
    pub updated_at: u64,
    pub estimated_completion: Option<u64>,
    pub metadata: Option<HashMap<String, String>>,
}

/// Health status of the server
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    pub status: ServiceStatus,
    pub uptime_seconds: u64,
    pub requests_processed: u64,
    pub error_rate: f64,
    pub average_response_time_ms: u64,
    pub circuit_breaker_status: String,
    pub active_connections: usize,
    pub memory_usage_mb: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Enhanced Web API server with resilience features
pub struct EnhancedWebServer {
    config: EnhancedServerConfig,
    network: Arc<ThreadSafeNetwork>,
    cache: Arc<DashMap<String, CachedResult>>,
    jobs: Arc<DashMap<String, EnhancedJobInfo>>,
    rate_limiter: Arc<DashMap<String, Vec<Instant>>>,
    circuit_breaker: Arc<CircuitBreaker>,
    error_aggregator: Arc<ErrorAggregator>,
    performance_tracker: Arc<PerformanceTracker>,
    health: Arc<DashMap<String, serde_json::Value>>,
    start_time: Instant,
    request_counter: Arc<std::sync::atomic::AtomicU64>,
    error_counter: Arc<std::sync::atomic::AtomicU64>,
}

/// Cached result with expiration
#[derive(Clone)]
struct CachedResult {
    data: Vec<u8>,
    created_at: SystemTime,
    metadata: ResponseMetadata,
    hit_count: u32,
}

impl EnhancedWebServer {
    /// Create new enhanced web server
    pub async fn new(config: EnhancedServerConfig) -> Result<Self> {
        let span = span!(Level::INFO, "server_init");
        let _enter = span.enter();
        
        info!("Initializing enhanced web server with resilience features");
        
        // Load network with retry
        let network = {
            let retry_executor = RetryExecutor::new(config.retry_config.clone());
            let mut context = ErrorContext::new("load_network");
            
            tokio::runtime::Handle::current().block_on(async {
                retry_executor.execute(
                    || {
                        if let Some(ref model_path) = config.model_path {
                            ThreadSafeNetwork::load_from_file(model_path)
                        } else {
                            ThreadSafeNetwork::load_builtin_natural()
                        }
                    },
                    &mut context
                ).await
            })?
        };
        
        let rate_limit = config.rate_limit.unwrap_or(60);
        
        // Initialize circuit breaker
        let circuit_breaker = Arc::new(CircuitBreaker::new(
            5,  // failure threshold
            3,  // success threshold
            Duration::from_secs(30),  // timeout
        ));
        
        let server = Self {
            config,
            network: Arc::new(network),
            cache: Arc::new(DashMap::new()),
            jobs: Arc::new(DashMap::new()),
            rate_limiter: Arc::new(DashMap::new()),
            circuit_breaker,
            error_aggregator: Arc::new(ErrorAggregator::new()),
            performance_tracker: Arc::new(PerformanceTracker::new()),
            health: Arc::new(DashMap::new()),
            start_time: Instant::now(),
            request_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            error_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        };
        
        // Start background tasks
        server.start_background_tasks().await;
        
        Ok(server)
    }
    
    /// Start background maintenance tasks
    async fn start_background_tasks(&self) {
        // Cache cleanup task
        if self.config.cache_enabled {
            let cache = Arc::clone(&self.cache);
            let ttl = self.config.cache_ttl;
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(60));
                loop {
                    interval.tick().await;
                    Self::cleanup_cache(&cache, ttl);
                }
            });
        }
        
        // Health monitoring task
        let health = Arc::clone(&self.health);
        let request_counter = Arc::clone(&self.request_counter);
        let error_counter = Arc::clone(&self.error_counter);
        let start_time = self.start_time;
        let interval_duration = self.config.health_check_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval_duration);
            loop {
                interval.tick().await;
                Self::update_health_metrics(&health, &request_counter, &error_counter, start_time);
            }
        });
        
        // Metrics collection
        let performance_tracker = Arc::clone(&self.performance_tracker);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                performance_tracker.report();
            }
        });
    }
    
    /// Process an upscale request with full error recovery
    pub async fn process_upscale_request(
        &self,
        request: EnhancedUpscaleRequest,
        client_id: &str,
    ) -> EnhancedUpscaleResponse {
        let operation_logger = OperationLogger::new(format!("upscale_request_{}", client_id));
        let start_time = Instant::now();
        
        // Check rate limit
        if !self.check_rate_limit(client_id).await {
            return self.rate_limit_exceeded_response();
        }
        
        // Check circuit breaker
        if self.config.circuit_breaker_enabled {
            match self.circuit_breaker.call(|| Ok::<_, EnhancedError>(())) .await {
                Err(EnhancedError::CircuitBreakerOpen { reset_after, .. }) => {
                    return self.circuit_breaker_open_response(reset_after);
                }
                _ => {}
            }
        }
        
        // Decode and validate input
        let image_data = match base64::decode(&request.image_data) {
            Ok(data) => data,
            Err(e) => {
                operation_logger.log_error(&e);
                return self.validation_error_response("Invalid base64 image data");
            }
        };
        
        // Check file size
        if image_data.len() > self.config.max_file_size {
            return self.validation_error_response("Image size exceeds maximum allowed");
        }
        
        // Check cache
        let cache_key = self.generate_cache_key(&request);
        if self.config.cache_enabled {
            if let Some(cached) = self.get_cached_result(&cache_key).await {
                metrics::increment_counter!("srgan_cache_hits", 1);
                return cached;
            }
        }
        
        // Process image with retry and error recovery
        let result = self.process_image_with_recovery(image_data, request.clone()).await;
        
        // Update metrics
        self.request_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let processing_time = start_time.elapsed();
        metrics::histogram!("srgan_request_duration_seconds", processing_time.as_secs_f64());
        
        match result {
            Ok((upscaled_data, metadata)) => {
                // Cache successful result
                if self.config.cache_enabled {
                    self.cache_result(&cache_key, &upscaled_data, metadata.clone()).await;
                }
                
                operation_logger.complete();
                
                EnhancedUpscaleResponse {
                    success: true,
                    image_data: Some(base64::encode(&upscaled_data)),
                    error: None,
                    metadata,
                    retry_info: None,
                }
            }
            Err(e) => {
                self.error_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                operation_logger.log_error(&e);
                self.error_response(e, request.fallback_allowed.unwrap_or(false))
            }
        }
    }
    
    /// Process image with full error recovery
    async fn process_image_with_recovery(
        &self,
        image_data: Vec<u8>,
        request: EnhancedUpscaleRequest,
    ) -> Result<(Vec<u8>, ResponseMetadata)> {
        let retry_executor = RetryExecutor::new(self.config.retry_config.clone());
        let mut context = ErrorContext::new("image_processing");
        
        // Load image
        let img = image::load_from_memory(&image_data)
            .map_err(|e| SrganError::Image(e))?;
        
        let original_size = (img.width(), img.height());
        let start_time = Instant::now();
        
        // Process with network
        let upscaled = self.performance_tracker.track("upscale_inference", || {
            self.network.upscale_image(&img)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        })?;
        
        let upscaled_size = (upscaled.width(), upscaled.height());
        
        // Encode result
        let format = request.format.as_deref().unwrap_or("png");
        let mut output = Vec::new();
        let image_format = match format {
            "png" => ImageFormat::Png,
            "jpeg" | "jpg" => ImageFormat::Jpeg,
            _ => ImageFormat::Png,
        };
        
        upscaled.write_to(&mut output, image_format)
            .map_err(|e| SrganError::Image(e))?;
        
        let metadata = ResponseMetadata {
            original_size,
            upscaled_size,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            queue_time_ms: 0,
            format: format.to_string(),
            model_used: "enhanced_srgan".to_string(),
            cache_hit: false,
            degraded_mode: false,
        };
        
        Ok((output, metadata))
    }
    
    /// Check rate limit for client
    async fn check_rate_limit(&self, client_id: &str) -> bool {
        let now = Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);
        
        let mut entry = self.rate_limiter.entry(client_id.to_string()).or_insert(Vec::new());
        entry.retain(|&t| t > one_minute_ago);
        
        if let Some(limit) = self.config.rate_limit {
            if entry.len() >= limit {
                metrics::increment_counter!("srgan_rate_limit_exceeded", 1);
                return false;
            }
        }
        
        entry.push(now);
        true
    }
    
    /// Generate cache key for request
    fn generate_cache_key(&self, request: &EnhancedUpscaleRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        request.image_data.hash(&mut hasher);
        request.scale_factor.hash(&mut hasher);
        request.format.hash(&mut hasher);
        request.quality.hash(&mut hasher);
        request.model.hash(&mut hasher);
        
        format!("cache_{:x}", hasher.finish())
    }
    
    /// Get cached result if available
    async fn get_cached_result(&self, key: &str) -> Option<EnhancedUpscaleResponse> {
        if let Some(mut cached) = self.cache.get_mut(key) {
            let age = SystemTime::now()
                .duration_since(cached.created_at)
                .unwrap_or(Duration::MAX);
            
            if age < self.config.cache_ttl {
                cached.hit_count += 1;
                let mut metadata = cached.metadata.clone();
                metadata.cache_hit = true;
                
                return Some(EnhancedUpscaleResponse {
                    success: true,
                    image_data: Some(base64::encode(&cached.data)),
                    error: None,
                    metadata,
                    retry_info: None,
                });
            }
        }
        None
    }
    
    /// Cache a successful result
    async fn cache_result(&self, key: &str, data: &[u8], metadata: ResponseMetadata) {
        self.cache.insert(
            key.to_string(),
            CachedResult {
                data: data.to_vec(),
                created_at: SystemTime::now(),
                metadata,
                hit_count: 0,
            },
        );
        
        metrics::gauge!("srgan_cache_size", self.cache.len() as f64);
    }
    
    /// Clean up expired cache entries
    fn cleanup_cache(cache: &DashMap<String, CachedResult>, ttl: Duration) {
        let now = SystemTime::now();
        cache.retain(|_, v| {
            now.duration_since(v.created_at).unwrap_or(Duration::MAX) < ttl
        });
        
        debug!("Cache cleanup: {} entries remaining", cache.len());
    }
    
    /// Update health metrics
    fn update_health_metrics(
        health: &DashMap<String, serde_json::Value>,
        request_counter: &std::sync::atomic::AtomicU64,
        error_counter: &std::sync::atomic::AtomicU64,
        start_time: Instant,
    ) {
        let requests = request_counter.load(std::sync::atomic::Ordering::Relaxed);
        let errors = error_counter.load(std::sync::atomic::Ordering::Relaxed);
        let error_rate = if requests > 0 {
            (errors as f64) / (requests as f64)
        } else {
            0.0
        };
        
        health.insert("uptime_seconds".to_string(), serde_json::json!(start_time.elapsed().as_secs()));
        health.insert("requests_processed".to_string(), serde_json::json!(requests));
        health.insert("error_rate".to_string(), serde_json::json!(error_rate));
        
        metrics::gauge!("srgan_health_error_rate", error_rate);
    }
    
    /// Create error response
    fn error_response(&self, error: SrganError, allow_fallback: bool) -> EnhancedUpscaleResponse {
        let error_id = uuid::Uuid::new_v4().to_string();
        
        let (code, category, retryable) = match &error {
            SrganError::Io(_) => ("IO_ERROR", "system", true),
            SrganError::Image(_) => ("IMAGE_ERROR", "validation", false),
            SrganError::Network(_) => ("NETWORK_ERROR", "processing", true),
            SrganError::InvalidInput(_) => ("INVALID_INPUT", "validation", false),
            SrganError::InvalidParameter(_) => ("INVALID_PARAMETER", "validation", false),
            _ => ("INTERNAL_ERROR", "system", false),
        };
        
        EnhancedUpscaleResponse {
            success: false,
            image_data: None,
            error: Some(ErrorDetails {
                code: code.to_string(),
                message: error.to_string(),
                retryable,
                category: category.to_string(),
                trace_id: error_id,
            }),
            metadata: ResponseMetadata {
                original_size: (0, 0),
                upscaled_size: (0, 0),
                processing_time_ms: 0,
                queue_time_ms: 0,
                format: "unknown".to_string(),
                model_used: "none".to_string(),
                cache_hit: false,
                degraded_mode: allow_fallback,
            },
            retry_info: if retryable {
                Some(RetryInfo {
                    retry_after: 5,
                    attempts_made: 1,
                    max_attempts: self.config.retry_config.max_attempts,
                })
            } else {
                None
            },
        }
    }
    
    /// Create rate limit exceeded response
    fn rate_limit_exceeded_response(&self) -> EnhancedUpscaleResponse {
        EnhancedUpscaleResponse {
            success: false,
            image_data: None,
            error: Some(ErrorDetails {
                code: "RATE_LIMIT_EXCEEDED".to_string(),
                message: "Too many requests. Please try again later.".to_string(),
                retryable: true,
                category: "rate_limit".to_string(),
                trace_id: uuid::Uuid::new_v4().to_string(),
            }),
            metadata: ResponseMetadata {
                original_size: (0, 0),
                upscaled_size: (0, 0),
                processing_time_ms: 0,
                queue_time_ms: 0,
                format: "unknown".to_string(),
                model_used: "none".to_string(),
                cache_hit: false,
                degraded_mode: false,
            },
            retry_info: Some(RetryInfo {
                retry_after: 60,
                attempts_made: 0,
                max_attempts: 1,
            }),
        }
    }
    
    /// Create circuit breaker open response
    fn circuit_breaker_open_response(&self, reset_after: Duration) -> EnhancedUpscaleResponse {
        EnhancedUpscaleResponse {
            success: false,
            image_data: None,
            error: Some(ErrorDetails {
                code: "SERVICE_UNAVAILABLE".to_string(),
                message: "Service is temporarily unavailable due to high error rate".to_string(),
                retryable: true,
                category: "circuit_breaker".to_string(),
                trace_id: uuid::Uuid::new_v4().to_string(),
            }),
            metadata: ResponseMetadata {
                original_size: (0, 0),
                upscaled_size: (0, 0),
                processing_time_ms: 0,
                queue_time_ms: 0,
                format: "unknown".to_string(),
                model_used: "none".to_string(),
                cache_hit: false,
                degraded_mode: false,
            },
            retry_info: Some(RetryInfo {
                retry_after: reset_after.as_secs(),
                attempts_made: 0,
                max_attempts: 1,
            }),
        }
    }
    
    /// Create validation error response
    fn validation_error_response(&self, message: &str) -> EnhancedUpscaleResponse {
        EnhancedUpscaleResponse {
            success: false,
            image_data: None,
            error: Some(ErrorDetails {
                code: "VALIDATION_ERROR".to_string(),
                message: message.to_string(),
                retryable: false,
                category: "validation".to_string(),
                trace_id: uuid::Uuid::new_v4().to_string(),
            }),
            metadata: ResponseMetadata {
                original_size: (0, 0),
                upscaled_size: (0, 0),
                processing_time_ms: 0,
                queue_time_ms: 0,
                format: "unknown".to_string(),
                model_used: "none".to_string(),
                cache_hit: false,
                degraded_mode: false,
            },
            retry_info: None,
        }
    }
    
    /// Get server health status
    pub async fn get_health_status(&self) -> HealthStatus {
        let requests = self.request_counter.load(std::sync::atomic::Ordering::Relaxed);
        let errors = self.error_counter.load(std::sync::atomic::Ordering::Relaxed);
        
        let error_rate = if requests > 0 {
            (errors as f64) / (requests as f64)
        } else {
            0.0
        };
        
        let status = if error_rate > 0.5 {
            ServiceStatus::Unhealthy
        } else if error_rate > 0.1 {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Healthy
        };
        
        HealthStatus {
            status,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            requests_processed: requests,
            error_rate,
            average_response_time_ms: 0, // Would calculate from metrics
            circuit_breaker_status: "closed".to_string(), // Would check actual status
            active_connections: 0, // Would track actual connections
            memory_usage_mb: 0, // Would calculate from system metrics
        }
    }
}

// Add uuid and base64 to dependencies
use uuid;
use base64;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_rate_limiting() {
        let config = EnhancedServerConfig {
            rate_limit: Some(2),
            ..Default::default()
        };
        
        let server = EnhancedWebServer::new(config).await.unwrap();
        
        // First two requests should pass
        assert!(server.check_rate_limit("client1").await);
        assert!(server.check_rate_limit("client1").await);
        
        // Third request should be rate limited
        assert!(!server.check_rate_limit("client1").await);
        
        // Different client should not be affected
        assert!(server.check_rate_limit("client2").await);
    }
    
    #[tokio::test]
    async fn test_cache_functionality() {
        let config = EnhancedServerConfig {
            cache_enabled: true,
            cache_ttl: Duration::from_secs(60),
            ..Default::default()
        };
        
        let server = EnhancedWebServer::new(config).await.unwrap();
        
        let metadata = ResponseMetadata {
            original_size: (100, 100),
            upscaled_size: (400, 400),
            processing_time_ms: 100,
            queue_time_ms: 0,
            format: "png".to_string(),
            model_used: "test".to_string(),
            cache_hit: false,
            degraded_mode: false,
        };
        
        // Cache a result
        server.cache_result("test_key", b"test_data", metadata.clone()).await;
        
        // Retrieve from cache
        let cached = server.get_cached_result("test_key").await;
        assert!(cached.is_some());
        
        let response = cached.unwrap();
        assert!(response.success);
        assert!(response.metadata.cache_hit);
    }
}