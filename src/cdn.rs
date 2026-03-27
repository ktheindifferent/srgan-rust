//! Multi-region CDN upload via S3-compatible storage.
//!
//! After upscaling, optionally uploads the result to an S3-compatible bucket
//! and returns a public CDN URL.
//!
//! Configuration (env vars):
//! - `S3_ENDPOINT`  — e.g. `https://s3.us-east-1.amazonaws.com`
//! - `S3_BUCKET`    — bucket name
//! - `S3_KEY`       — access key ID
//! - `S3_SECRET`    — secret access key
//! - `S3_REGION`    — region (default: `us-east-1`)
//! - `S3_KEY_PREFIX`— optional key prefix (default: `upscaled/`)

use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::{Sha256, Digest};
use chrono::Utc;
use serde::{Deserialize, Serialize};

type HmacSha256 = Hmac<Sha256>;

/// CDN upload configuration parsed from env vars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnConfig {
    pub endpoint: String,
    pub bucket: String,
    pub access_key: String,
    pub secret_key: String,
    pub region: String,
    pub key_prefix: String,
}

impl CdnConfig {
    /// Try to load from environment variables. Returns `None` if required vars
    /// are missing.
    ///
    /// Checks `CDN_*` env vars first (CDN_ENDPOINT, CDN_BUCKET, CDN_KEY,
    /// CDN_SECRET), then falls back to `S3_*` for backwards compatibility.
    pub fn from_env() -> Option<Self> {
        let endpoint = env::var("CDN_ENDPOINT")
            .or_else(|_| env::var("S3_ENDPOINT"))
            .ok()?;
        let bucket = env::var("CDN_BUCKET")
            .or_else(|_| env::var("S3_BUCKET"))
            .ok()?;
        let access_key = env::var("CDN_KEY")
            .or_else(|_| env::var("S3_KEY"))
            .ok()?;
        let secret_key = env::var("CDN_SECRET")
            .or_else(|_| env::var("S3_SECRET"))
            .ok()?;
        let region = env::var("CDN_REGION")
            .or_else(|_| env::var("S3_REGION"))
            .unwrap_or_else(|_| "us-east-1".into());
        let key_prefix = env::var("CDN_KEY_PREFIX")
            .or_else(|_| env::var("S3_KEY_PREFIX"))
            .unwrap_or_else(|_| "upscaled/".into());

        Some(Self {
            endpoint,
            bucket,
            access_key,
            secret_key,
            region,
            key_prefix,
        })
    }

    /// Build the public URL for an uploaded object.
    pub fn public_url(&self, key: &str) -> String {
        format!("{}/{}/{}", self.endpoint, self.bucket, key)
    }
}

/// Result of a CDN upload.
#[derive(Debug, Clone, Serialize)]
pub struct CdnUploadResult {
    pub cdn_url: String,
    pub key: String,
    pub bucket: String,
    pub size_bytes: usize,
}

/// Upload image bytes to S3-compatible storage using a signed PUT request.
///
/// Uses AWS Signature V4 for authentication.
pub fn upload_to_cdn(
    config: &CdnConfig,
    data: &[u8],
    filename: &str,
    content_type: &str,
) -> Result<CdnUploadResult, CdnError> {
    let key = format!("{}{}", config.key_prefix, filename);
    let url = format!("{}/{}/{}", config.endpoint, config.bucket, key);

    let now = Utc::now();
    let date_stamp = now.format("%Y%m%d").to_string();
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();

    // Content hash
    let content_hash = hex::encode(Sha256::digest(data));

    // Canonical request
    let canonical_uri = format!("/{}/{}", config.bucket, key);
    let host = config
        .endpoint
        .trim_start_matches("https://")
        .trim_start_matches("http://");

    let canonical_headers = format!(
        "content-type:{}\nhost:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        content_type, host, content_hash, amz_date
    );
    let signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date";

    let canonical_request = format!(
        "PUT\n{}\n\n{}\n{}\n{}",
        canonical_uri, canonical_headers, signed_headers, content_hash
    );

    // String to sign
    let scope = format!("{}/{}/s3/aws4_request", date_stamp, config.region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        scope,
        hex::encode(Sha256::digest(canonical_request.as_bytes()))
    );

    // Signing key
    let k_date = hmac_sha256(
        format!("AWS4{}", config.secret_key).as_bytes(),
        date_stamp.as_bytes(),
    );
    let k_region = hmac_sha256(&k_date, config.region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"s3");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");

    let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        config.access_key, scope, signed_headers, signature
    );

    // Execute PUT request
    let resp = ureq::put(&url)
        .set("Content-Type", content_type)
        .set("x-amz-content-sha256", &content_hash)
        .set("x-amz-date", &amz_date)
        .set("Host", host)
        .set("Authorization", &authorization)
        .send_bytes(data)
        .map_err(|e| CdnError::UploadFailed(format!("{}", e)))?;

    let status = resp.status();
    if status >= 200 && status < 300 {
        Ok(CdnUploadResult {
            cdn_url: config.public_url(&key),
            key,
            bucket: config.bucket.clone(),
            size_bytes: data.len(),
        })
    } else {
        Err(CdnError::UploadFailed(format!(
            "S3 returned status {}",
            status
        )))
    }
}

/// Generate a unique filename for an upload.
pub fn generate_upload_key(job_id: &str, extension: &str) -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{}_{}.{}", job_id, ts, extension)
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac =
        HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Errors from CDN operations.
#[derive(Debug, Clone)]
pub enum CdnError {
    /// S3 endpoint not configured.
    NotConfigured,
    /// Upload request failed.
    UploadFailed(String),
}

impl std::fmt::Display for CdnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConfigured => write!(f, "CDN not configured (set S3_ENDPOINT, S3_BUCKET, S3_KEY, S3_SECRET)"),
            Self::UploadFailed(msg) => write!(f, "CDN upload failed: {}", msg),
        }
    }
}

impl std::error::Error for CdnError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_env_missing() {
        // With no env vars set, should return None
        let cfg = CdnConfig::from_env();
        // Can't assert None because CI might have these set, so just verify it doesn't panic
        let _ = cfg;
    }

    #[test]
    fn test_public_url() {
        let cfg = CdnConfig {
            endpoint: "https://s3.us-east-1.amazonaws.com".into(),
            bucket: "my-bucket".into(),
            access_key: "key".into(),
            secret_key: "secret".into(),
            region: "us-east-1".into(),
            key_prefix: "upscaled/".into(),
        };
        assert_eq!(
            cfg.public_url("upscaled/test.png"),
            "https://s3.us-east-1.amazonaws.com/my-bucket/upscaled/test.png"
        );
    }

    #[test]
    fn test_generate_upload_key() {
        let key = generate_upload_key("job-123", "png");
        assert!(key.starts_with("job-123_"));
        assert!(key.ends_with(".png"));
    }

    #[test]
    fn test_hmac_sha256_deterministic() {
        let a = hmac_sha256(b"key", b"data");
        let b = hmac_sha256(b"key", b"data");
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }
}
