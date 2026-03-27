/// S3-compatible object storage: upload results and generate presigned GET URLs.
///
/// Supports AWS S3 and Cloudflare R2 (or any S3-compatible endpoint).
/// Configured entirely via environment variables:
///   S3_BUCKET    — bucket name
///   S3_ENDPOINT  — base URL, e.g. https://s3.amazonaws.com or https://<acct>.r2.cloudflarestorage.com
///   S3_REGION    — AWS region (default "us-east-1")
///   S3_ACCESS_KEY
///   S3_SECRET_KEY
use sha2::{Digest, Sha256};
use hmac::{Hmac, Mac};
use hex;
use chrono::Utc;

type HmacSha256 = Hmac<Sha256>;

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct S3Config {
    pub bucket: String,
    pub endpoint: String,
    pub region: String,
    pub access_key: String,
    pub secret_key: String,
}

impl S3Config {
    /// Load from environment variables. Returns None if any required var is missing.
    pub fn from_env() -> Option<Self> {
        let bucket = std::env::var("S3_BUCKET").ok()?;
        let endpoint = std::env::var("S3_ENDPOINT").ok()?;
        let access_key = std::env::var("S3_ACCESS_KEY").ok()?;
        let secret_key = std::env::var("S3_SECRET_KEY").ok()?;
        let region = std::env::var("S3_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        Some(Self {
            bucket,
            endpoint,
            region,
            access_key,
            secret_key,
        })
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Upload `data` to `key` in the configured bucket, then return a presigned GET URL
/// valid for `expires_secs` seconds (default 86400 = 24 hours).
pub fn upload_result(
    config: &S3Config,
    key: &str,
    data: &[u8],
    content_type: &str,
) -> Result<String, String> {
    let now = Utc::now();
    let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();

    let url = format!("{}/{}/{}", config.endpoint.trim_end_matches('/'), config.bucket, key);
    let host = extract_host(&config.endpoint);

    let content_sha256 = hex::encode(Sha256::digest(data));

    // Build canonical request for PUT
    let canonical_headers = format!(
        "content-type:{}\nhost:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        content_type, host, content_sha256, datetime
    );
    let signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date";
    let canonical_uri = format!("/{}/{}", config.bucket, key);

    let canonical_request = format!(
        "PUT\n{}\n\n{}\n{}\n{}",
        canonical_uri, canonical_headers, signed_headers, content_sha256
    );

    let signature = compute_signature(
        &config.secret_key,
        &date,
        &config.region,
        "s3",
        &datetime,
        &canonical_request,
    )?;

    let credential_scope = format!("{}/{}/s3/aws4_request", date, config.region);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{},SignedHeaders={},Signature={}",
        config.access_key, credential_scope, signed_headers, signature
    );

    let client = reqwest::blocking::Client::new();
    let response = client
        .put(&url)
        .header("Content-Type", content_type)
        .header("x-amz-date", &datetime)
        .header("x-amz-content-sha256", &content_sha256)
        .header("Authorization", &authorization)
        .body(data.to_vec())
        .send()
        .map_err(|e| format!("S3 upload request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().unwrap_or_default();
        return Err(format!("S3 upload returned HTTP {}: {}", status, body));
    }

    generate_presigned_url(config, key, 86400)
}

/// Generate a presigned GET URL for an existing object without uploading.
pub fn generate_presigned_url(
    config: &S3Config,
    key: &str,
    expires_secs: u64,
) -> Result<String, String> {
    let now = Utc::now();
    let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();

    let host = extract_host(&config.endpoint);
    let credential_scope = format!("{}/{}/s3/aws4_request", date, config.region);
    let credential = url_encode(&format!("{}/{}", config.access_key, credential_scope));

    let query = format!(
        "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential={}&X-Amz-Date={}&X-Amz-Expires={}&X-Amz-SignedHeaders=host",
        credential, datetime, expires_secs
    );

    let canonical_uri = format!("/{}/{}", config.bucket, key);
    let canonical_request = format!(
        "GET\n{}\n{}\nhost:{}\n\nhost\nUNSIGNED-PAYLOAD",
        canonical_uri, query, host
    );

    let signature = compute_signature(
        &config.secret_key,
        &date,
        &config.region,
        "s3",
        &datetime,
        &canonical_request,
    )?;

    let base = format!(
        "{}/{}/{}",
        config.endpoint.trim_end_matches('/'),
        config.bucket,
        key
    );
    Ok(format!("{}?{}&X-Amz-Signature={}", base, query, signature))
}

// ── Internals ─────────────────────────────────────────────────────────────────

fn compute_signature(
    secret: &str,
    date: &str,
    region: &str,
    service: &str,
    datetime: &str,
    canonical_request: &str,
) -> Result<String, String> {
    let credential_scope = format!("{}/{}/{}/aws4_request", date, region, service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        datetime,
        credential_scope,
        hex::encode(Sha256::digest(canonical_request.as_bytes()))
    );

    let signing_key = derive_signing_key(secret, date, region, service)?;

    let mut mac = HmacSha256::new_from_slice(&signing_key)
        .map_err(|e| format!("HMAC key error: {}", e))?;
    mac.update(string_to_sign.as_bytes());
    Ok(hex::encode(mac.finalize().into_bytes()))
}

fn derive_signing_key(secret: &str, date: &str, region: &str, service: &str) -> Result<Vec<u8>, String> {
    let k_secret = format!("AWS4{}", secret);
    let k_date = hmac_sha256(k_secret.as_bytes(), date.as_bytes())?;
    let k_region = hmac_sha256(&k_date, region.as_bytes())?;
    let k_service = hmac_sha256(&k_region, service.as_bytes())?;
    hmac_sha256(&k_service, b"aws4_request")
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
    let mut mac =
        HmacSha256::new_from_slice(key).map_err(|e| format!("HMAC key error: {}", e))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn extract_host(endpoint: &str) -> String {
    endpoint
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap_or(endpoint)
        .to_string()
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{:02X}", byte)),
        }
    }
    out
}

// ── S3 Multipart Upload ──────────────────────────────────────────────────────

/// Minimum part size for S3 multipart upload (5 MB).
const MULTIPART_THRESHOLD: usize = 5 * 1024 * 1024;
/// Part size for splitting large uploads (5 MB).
const MULTIPART_PART_SIZE: usize = 5 * 1024 * 1024;

/// Upload data to S3, automatically using multipart upload for files > 5 MB.
/// Returns a presigned GET URL for the uploaded object.
pub fn upload_result_auto(
    config: &S3Config,
    key: &str,
    data: &[u8],
    content_type: &str,
) -> Result<String, String> {
    if data.len() <= MULTIPART_THRESHOLD {
        return upload_result(config, key, data, content_type);
    }
    multipart_upload(config, key, data, content_type)
}

/// Perform S3 multipart upload: initiate, upload parts in parallel, complete.
fn multipart_upload(
    config: &S3Config,
    key: &str,
    data: &[u8],
    content_type: &str,
) -> Result<String, String> {
    let client = reqwest::blocking::Client::new();
    let upload_id = initiate_multipart(config, &client, key, content_type)?;

    // Split data into parts
    let parts: Vec<&[u8]> = data.chunks(MULTIPART_PART_SIZE).collect();
    let mut etags: Vec<(usize, String)> = Vec::with_capacity(parts.len());

    // Upload parts (sequentially with blocking client; could use rayon for parallelism)
    for (idx, part_data) in parts.iter().enumerate() {
        let part_number = idx + 1;
        let etag = upload_part(config, &client, key, &upload_id, part_number, part_data)?;
        etags.push((part_number, etag));
    }

    // Complete multipart upload
    complete_multipart(config, &client, key, &upload_id, &etags)?;

    generate_presigned_url(config, key, 86400)
}

/// Initiate a multipart upload and return the UploadId.
fn initiate_multipart(
    config: &S3Config,
    client: &reqwest::blocking::Client,
    key: &str,
    content_type: &str,
) -> Result<String, String> {
    let now = Utc::now();
    let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();

    let url = format!(
        "{}/{}/{}?uploads",
        config.endpoint.trim_end_matches('/'),
        config.bucket,
        key
    );
    let host = extract_host(&config.endpoint);
    let content_sha256 = hex::encode(Sha256::digest(b""));

    let canonical_uri = format!("/{}/{}", config.bucket, key);
    let canonical_headers = format!(
        "content-type:{}\nhost:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        content_type, host, content_sha256, datetime
    );
    let signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date";
    let canonical_request = format!(
        "POST\n{}\nuploads=\n{}\n{}\n{}",
        canonical_uri, canonical_headers, signed_headers, content_sha256
    );

    let signature = compute_signature(
        &config.secret_key, &date, &config.region, "s3", &datetime, &canonical_request,
    )?;

    let credential_scope = format!("{}/{}/s3/aws4_request", date, config.region);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{},SignedHeaders={},Signature={}",
        config.access_key, credential_scope, signed_headers, signature
    );

    let response = client
        .post(&url)
        .header("Content-Type", content_type)
        .header("x-amz-date", &datetime)
        .header("x-amz-content-sha256", &content_sha256)
        .header("Authorization", &authorization)
        .send()
        .map_err(|e| format!("Initiate multipart failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().unwrap_or_default();
        return Err(format!("Initiate multipart returned HTTP {}: {}", status, body));
    }

    let body = response.text().unwrap_or_default();
    // Parse UploadId from XML response
    extract_xml_value(&body, "UploadId")
        .ok_or_else(|| "Failed to parse UploadId from response".to_string())
}

/// Upload a single part of a multipart upload. Returns the ETag.
fn upload_part(
    config: &S3Config,
    client: &reqwest::blocking::Client,
    key: &str,
    upload_id: &str,
    part_number: usize,
    data: &[u8],
) -> Result<String, String> {
    let now = Utc::now();
    let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();

    let url = format!(
        "{}/{}/{}?partNumber={}&uploadId={}",
        config.endpoint.trim_end_matches('/'),
        config.bucket,
        key,
        part_number,
        url_encode(upload_id)
    );
    let host = extract_host(&config.endpoint);
    let content_sha256 = hex::encode(Sha256::digest(data));

    let canonical_uri = format!("/{}/{}", config.bucket, key);
    let canonical_query = format!("partNumber={}&uploadId={}", part_number, url_encode(upload_id));
    let canonical_headers = format!(
        "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        host, content_sha256, datetime
    );
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";
    let canonical_request = format!(
        "PUT\n{}\n{}\n{}\n{}\n{}",
        canonical_uri, canonical_query, canonical_headers, signed_headers, content_sha256
    );

    let signature = compute_signature(
        &config.secret_key, &date, &config.region, "s3", &datetime, &canonical_request,
    )?;

    let credential_scope = format!("{}/{}/s3/aws4_request", date, config.region);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{},SignedHeaders={},Signature={}",
        config.access_key, credential_scope, signed_headers, signature
    );

    let response = client
        .put(&url)
        .header("x-amz-date", &datetime)
        .header("x-amz-content-sha256", &content_sha256)
        .header("Authorization", &authorization)
        .body(data.to_vec())
        .send()
        .map_err(|e| format!("Upload part {} failed: {}", part_number, e))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().unwrap_or_default();
        return Err(format!("Upload part {} returned HTTP {}: {}", part_number, status, body));
    }

    // ETag is in the response header
    let etag = response
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .trim_matches('"')
        .to_string();

    if etag.is_empty() {
        return Err(format!("No ETag returned for part {}", part_number));
    }

    Ok(etag)
}

/// Complete a multipart upload by sending the part list.
fn complete_multipart(
    config: &S3Config,
    client: &reqwest::blocking::Client,
    key: &str,
    upload_id: &str,
    etags: &[(usize, String)],
) -> Result<(), String> {
    let now = Utc::now();
    let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();

    // Build completion XML
    let mut xml = String::from("<CompleteMultipartUpload>");
    for (part_number, etag) in etags {
        xml.push_str(&format!(
            "<Part><PartNumber>{}</PartNumber><ETag>{}</ETag></Part>",
            part_number, etag
        ));
    }
    xml.push_str("</CompleteMultipartUpload>");

    let url = format!(
        "{}/{}/{}?uploadId={}",
        config.endpoint.trim_end_matches('/'),
        config.bucket,
        key,
        url_encode(upload_id)
    );
    let host = extract_host(&config.endpoint);
    let content_sha256 = hex::encode(Sha256::digest(xml.as_bytes()));

    let canonical_uri = format!("/{}/{}", config.bucket, key);
    let canonical_query = format!("uploadId={}", url_encode(upload_id));
    let canonical_headers = format!(
        "content-type:application/xml\nhost:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        host, content_sha256, datetime
    );
    let signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date";
    let canonical_request = format!(
        "POST\n{}\n{}\n{}\n{}\n{}",
        canonical_uri, canonical_query, canonical_headers, signed_headers, content_sha256
    );

    let signature = compute_signature(
        &config.secret_key, &date, &config.region, "s3", &datetime, &canonical_request,
    )?;

    let credential_scope = format!("{}/{}/s3/aws4_request", date, config.region);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{},SignedHeaders={},Signature={}",
        config.access_key, credential_scope, signed_headers, signature
    );

    let response = client
        .post(&url)
        .header("Content-Type", "application/xml")
        .header("x-amz-date", &datetime)
        .header("x-amz-content-sha256", &content_sha256)
        .header("Authorization", &authorization)
        .body(xml)
        .send()
        .map_err(|e| format!("Complete multipart failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().unwrap_or_default();
        return Err(format!("Complete multipart returned HTTP {}: {}", status, body));
    }

    Ok(())
}

/// Simple XML value extractor (avoids pulling in an XML parser dependency).
fn extract_xml_value(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_xml_value() {
        let xml = "<Root><UploadId>abc123</UploadId></Root>";
        assert_eq!(extract_xml_value(xml, "UploadId"), Some("abc123".to_string()));
        assert_eq!(extract_xml_value(xml, "Missing"), None);
    }

    #[test]
    fn test_multipart_threshold() {
        assert_eq!(MULTIPART_THRESHOLD, 5 * 1024 * 1024);
    }

    #[test]
    fn test_url_encode() {
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("a/b+c"), "a%2Fb%2Bc");
    }
}
