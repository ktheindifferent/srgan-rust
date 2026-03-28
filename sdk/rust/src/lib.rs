//! # srgan-client
//!
//! Minimal Rust client SDK for the SRGAN Rust image upscaling API.
//!
//! ## Quick start
//!
//! ```no_run
//! use srgan_client::SrganClient;
//!
//! let client = SrganClient::new("http://localhost:8080", "your-api-key");
//!
//! // Synchronous upscale
//! let result = client.upscale_file("input.jpg", "anime", "png").unwrap();
//! std::fs::write("output.png", &result).unwrap();
//!
//! // Async: submit, poll, download
//! let job = client.submit("input.jpg", "natural", "png").unwrap();
//! let status = client.poll_until_done(&job.job_id, 60).unwrap();
//! let output = client.download(&job.job_id).unwrap();
//! std::fs::write("output.png", &output).unwrap();
//! ```

use reqwest::blocking::{Client, multipart};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Errors returned by the SDK.
#[derive(Debug, thiserror::Error)]
pub enum SrganError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Job timed out after {0}s")]
    Timeout(u64),
}

/// Response from async job submission.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub status: String,
    #[serde(default)]
    pub priority: Option<String>,
}

/// Response from job status polling.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JobStatus {
    pub job_id: String,
    pub status: String,
    #[serde(default)]
    pub processing_time_ms: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub output_url: Option<String>,
    #[serde(default)]
    pub cdn_url: Option<String>,
    #[serde(default)]
    pub compare_url: Option<String>,
}

/// Waifu2x-specific upscale parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waifu2xOptions {
    /// Noise reduction level: 0 (none), 1 (low), 2 (medium), 3 (high).
    pub noise_level: u8,
    /// Upscaling factor: 1 (denoise-only), 2, 3, or 4.
    pub scale: u8,
    /// Content style: "anime", "photo", or "artwork".
    pub style: String,
    /// Output format: "png", "jpeg", or "webp".
    pub format: String,
}

impl Default for Waifu2xOptions {
    fn default() -> Self {
        Self {
            noise_level: 1,
            scale: 2,
            style: "anime".to_string(),
            format: "png".to_string(),
        }
    }
}

/// SRGAN API client.
pub struct SrganClient {
    base_url: String,
    api_key: String,
    client: Client,
}

impl SrganClient {
    /// Create a new client.
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// Waifu2x-specific upscale options.
    ///
    /// When `model` is `"waifu2x"` or a waifu2x variant, these options
    /// configure noise reduction, scale, and content style.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let client = SrganClient::new("http://localhost:8080", "key");
    /// let result = client.upscale_waifu2x("input.jpg", &Waifu2xOptions {
    ///     noise_level: 2,
    ///     scale: 4,
    ///     style: "anime".to_string(),
    ///     format: "png".to_string(),
    /// }).unwrap();
    /// std::fs::write("output.png", &result).unwrap();
    /// ```
    pub fn upscale_waifu2x(
        &self,
        path: &str,
        opts: &Waifu2xOptions,
    ) -> Result<Vec<u8>, SrganError> {
        let file_bytes = std::fs::read(path)?;
        let file_name = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "image.jpg".to_string());

        let part = multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("application/octet-stream")
            .unwrap();

        let model_label = format!(
            "waifu2x-noise{}-scale{}",
            opts.noise_level.min(3),
            opts.scale.max(1).min(4)
        );

        let mut form = multipart::Form::new()
            .part("image", part)
            .text("model", model_label)
            .text("format", opts.format.clone())
            .text("waifu2x_noise_level", opts.noise_level.to_string())
            .text("waifu2x_scale", opts.scale.to_string());

        if !opts.style.is_empty() {
            form = form.text("waifu2x_style", opts.style.clone());
        }

        let resp = self
            .client
            .post(format!("{}/api/v1/upscale", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()?;

        let status = resp.status().as_u16();
        if status >= 200 && status < 300 {
            Ok(resp.bytes()?.to_vec())
        } else {
            let msg = resp.text().unwrap_or_default();
            Err(SrganError::Api {
                status,
                message: msg,
            })
        }
    }

    /// Synchronous upscale: upload file and wait for result bytes.
    pub fn upscale_file(
        &self,
        path: &str,
        model: &str,
        format: &str,
    ) -> Result<Vec<u8>, SrganError> {
        let file_bytes = std::fs::read(path)?;
        let file_name = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "image.jpg".to_string());

        let part = multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("application/octet-stream")
            .unwrap();

        let form = multipart::Form::new()
            .part("image", part)
            .text("model", model.to_string())
            .text("format", format.to_string());

        let resp = self
            .client
            .post(format!("{}/api/v1/upscale", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()?;

        let status = resp.status().as_u16();
        if status >= 200 && status < 300 {
            Ok(resp.bytes()?.to_vec())
        } else {
            let msg = resp.text().unwrap_or_default();
            Err(SrganError::Api {
                status,
                message: msg,
            })
        }
    }

    /// Submit an async upscale job. Returns the job ID for polling.
    pub fn submit(
        &self,
        path: &str,
        model: &str,
        format: &str,
    ) -> Result<JobResponse, SrganError> {
        let file_bytes = std::fs::read(path)?;
        let file_name = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "image.jpg".to_string());

        let part = multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("application/octet-stream")
            .unwrap();

        let form = multipart::Form::new()
            .part("image", part)
            .text("model", model.to_string())
            .text("format", format.to_string());

        let resp = self
            .client
            .post(format!("{}/api/v1/upscale/async", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()?;

        let status = resp.status().as_u16();
        if status >= 200 && status < 300 {
            Ok(resp.json()?)
        } else {
            let msg = resp.text().unwrap_or_default();
            Err(SrganError::Api {
                status,
                message: msg,
            })
        }
    }

    /// Poll job status.
    pub fn get_status(&self, job_id: &str) -> Result<JobStatus, SrganError> {
        let resp = self
            .client
            .get(format!("{}/api/v1/jobs/{}", self.base_url, job_id))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()?;

        let status = resp.status().as_u16();
        if status >= 200 && status < 300 {
            Ok(resp.json()?)
        } else {
            let msg = resp.text().unwrap_or_default();
            Err(SrganError::Api {
                status,
                message: msg,
            })
        }
    }

    /// Poll until job completes or fails. `timeout_secs` is max wait time.
    pub fn poll_until_done(
        &self,
        job_id: &str,
        timeout_secs: u64,
    ) -> Result<JobStatus, SrganError> {
        let start = std::time::Instant::now();
        loop {
            let status = self.get_status(job_id)?;
            match status.status.as_str() {
                "completed" | "failed" => return Ok(status),
                _ => {}
            }
            if start.elapsed().as_secs() >= timeout_secs {
                return Err(SrganError::Timeout(timeout_secs));
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    /// Download the result of a completed job.
    pub fn download(&self, job_id: &str) -> Result<Vec<u8>, SrganError> {
        let resp = self
            .client
            .get(format!("{}/api/v1/jobs/{}/result", self.base_url, job_id))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()?;

        let status = resp.status().as_u16();
        if status >= 200 && status < 300 {
            Ok(resp.bytes()?.to_vec())
        } else {
            let msg = resp.text().unwrap_or_default();
            Err(SrganError::Api {
                status,
                message: msg,
            })
        }
    }
}
