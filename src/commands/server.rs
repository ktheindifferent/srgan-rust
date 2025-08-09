use std::path::PathBuf;
use std::time::Duration;
use clap::ArgMatches;
use crate::error::SrganError;
use crate::web_server::{WebServer, ServerConfig};

/// Start the web API server
pub fn start_server(matches: &ArgMatches) -> Result<(), SrganError> {
    let mut config = ServerConfig::default();
    
    // Parse configuration from command line
    if let Some(host) = matches.value_of("host") {
        config.host = host.to_string();
    }
    
    if let Some(port) = matches.value_of("port") {
        config.port = port.parse()
            .map_err(|_| SrganError::InvalidInput("Invalid port number".to_string()))?;
    }
    
    if let Some(max_size) = matches.value_of("max-size") {
        config.max_file_size = parse_size(max_size)?;
    }
    
    if matches.is_present("no-cache") {
        config.cache_enabled = false;
    }
    
    if let Some(ttl) = matches.value_of("cache-ttl") {
        config.cache_ttl = Duration::from_secs(
            ttl.parse()
                .map_err(|_| SrganError::InvalidInput("Invalid cache TTL".to_string()))?
        );
    }
    
    if matches.is_present("no-cors") {
        config.cors_enabled = false;
    }
    
    if let Some(api_key) = matches.value_of("api-key") {
        config.api_key = Some(api_key.to_string());
    }
    
    if let Some(rate_limit) = matches.value_of("rate-limit") {
        config.rate_limit = Some(
            rate_limit.parse()
                .map_err(|_| SrganError::InvalidInput("Invalid rate limit".to_string()))?
        );
    }
    
    if let Some(model_path) = matches.value_of("model") {
        config.model_path = Some(PathBuf::from(model_path));
    }
    
    if matches.is_present("no-logging") {
        config.log_requests = false;
    }
    
    // Display configuration
    info!("Server configuration:");
    info!("  Host: {}", config.host);
    info!("  Port: {}", config.port);
    info!("  Max file size: {} MB", config.max_file_size / (1024 * 1024));
    info!("  Cache: {}", if config.cache_enabled { "enabled" } else { "disabled" });
    info!("  CORS: {}", if config.cors_enabled { "enabled" } else { "disabled" });
    info!("  Rate limit: {} req/min", config.rate_limit.unwrap_or(0));
    
    if config.api_key.is_some() {
        info!("  API key: configured");
    }
    
    // Create and start server
    let server = WebServer::new(config)?;
    
    info!("Starting SRGAN web server...");
    info!("Press Ctrl+C to stop");
    
    // Set up signal handler for graceful shutdown
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    
    ctrlc::set_handler(move || {
        info!("Shutting down server...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    }).map_err(|e| SrganError::InvalidInput(format!("Error setting Ctrl-C handler: {}", e)))?;
    
    // Start server
    server.start()?;
    
    Ok(())
}

/// Parse size string (e.g., "10MB", "1GB")
fn parse_size(size_str: &str) -> Result<usize, SrganError> {
    let size_str = size_str.to_uppercase();
    
    if let Some(mb_str) = size_str.strip_suffix("MB") {
        mb_str.parse::<usize>()
            .map(|mb| mb * 1024 * 1024)
            .map_err(|_| SrganError::InvalidInput(format!("Invalid size: {}", size_str)))
    } else if let Some(gb_str) = size_str.strip_suffix("GB") {
        gb_str.parse::<usize>()
            .map(|gb| gb * 1024 * 1024 * 1024)
            .map_err(|_| SrganError::InvalidInput(format!("Invalid size: {}", size_str)))
    } else if let Some(kb_str) = size_str.strip_suffix("KB") {
        kb_str.parse::<usize>()
            .map(|kb| kb * 1024)
            .map_err(|_| SrganError::InvalidInput(format!("Invalid size: {}", size_str)))
    } else {
        size_str.parse()
            .map_err(|_| SrganError::InvalidInput(format!("Invalid size: {}", size_str)))
    }
}

/// Generate example client code
pub fn generate_client_example(matches: &ArgMatches) -> Result<(), SrganError> {
    let language = matches.value_of("language").unwrap_or("python");
    let host = matches.value_of("host").unwrap_or("localhost");
    let port = matches.value_of("port").unwrap_or("8080");
    
    let example = match language.to_lowercase().as_str() {
        "python" => generate_python_client(host, port),
        "javascript" | "js" => generate_javascript_client(host, port),
        "curl" => generate_curl_client(host, port),
        "rust" => generate_rust_client(host, port),
        _ => return Err(SrganError::InvalidInput(
            format!("Unknown language: {}. Supported: python, javascript, curl, rust", language)
        )),
    };
    
    println!("{}", example);
    Ok(())
}

fn generate_python_client(host: &str, port: &str) -> String {
    format!(r#"#!/usr/bin/env python3
"""
SRGAN Web API Client Example
"""

import base64
import requests
import json
from PIL import Image
import io

API_URL = "http://{}:{}/api"

def upscale_image(image_path, output_path):
    """Upscale an image using the SRGAN API"""
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare request
    payload = {{
        'image_data': image_data,
        'format': 'png',
        'model': 'natural'
    }}
    
    # Send request
    response = requests.post(f"{{API_URL}}/upscale", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        if result['success']:
            # Decode and save result
            image_bytes = base64.b64decode(result['image_data'])
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"✓ Upscaled image saved to {{output_path}}")
            print(f"  Original size: {{result['metadata']['original_size']}}")
            print(f"  Upscaled size: {{result['metadata']['upscaled_size']}}")
            print(f"  Processing time: {{result['metadata']['processing_time_ms']}}ms")
        else:
            print(f"✗ Error: {{result['error']}}")
    else:
        print(f"✗ HTTP Error: {{response.status_code}}")

def upscale_async(image_path):
    """Upscale an image asynchronously"""
    
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Start async job
    response = requests.post(f"{{API_URL}}/upscale/async", json={{
        'image_data': image_data
    }})
    
    if response.status_code == 202:
        job = response.json()
        job_id = job['job_id']
        print(f"Job started: {{job_id}}")
        
        # Poll for completion
        import time
        while True:
            status = requests.get(f"{{API_URL}}/job/{{job_id}}").json()
            
            if status['status'] == 'Completed':
                print("✓ Job completed!")
                break
            elif 'Failed' in status['status']:
                print(f"✗ Job failed: {{status['error']}}")
                break
            else:
                print(f"  Status: {{status['status']}}")
                time.sleep(1)

if __name__ == '__main__':
    # Example usage
    upscale_image('input.jpg', 'output.png')
    # upscale_async('input.jpg')
"#, host, port)
}

fn generate_javascript_client(host: &str, port: &str) -> String {
    format!(r#"// SRGAN Web API Client Example

const API_URL = 'http://{}:{}/api';

async function upscaleImage(imageFile) {{
    // Convert image to base64
    const reader = new FileReader();
    const base64 = await new Promise((resolve) => {{
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(imageFile);
    }});
    
    // Send request
    const response = await fetch(`${{API_URL}}/upscale`, {{
        method: 'POST',
        headers: {{
            'Content-Type': 'application/json',
        }},
        body: JSON.stringify({{
            image_data: base64,
            format: 'png',
            model: 'natural'
        }})
    }});
    
    const result = await response.json();
    
    if (result.success) {{
        // Convert base64 to blob
        const imageBlob = await fetch(`data:image/png;base64,${{result.image_data}}`)
            .then(res => res.blob());
        
        // Create download link
        const url = URL.createObjectURL(imageBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'upscaled.png';
        a.click();
        
        console.log('✓ Image upscaled successfully');
        console.log(`  Processing time: ${{result.metadata.processing_time_ms}}ms`);
    }} else {{
        console.error('✗ Error:', result.error);
    }}
}}

// HTML usage example
/*
<input type="file" id="imageInput" accept="image/*">
<script>
document.getElementById('imageInput').addEventListener('change', (e) => {{
    if (e.target.files.length > 0) {{
        upscaleImage(e.target.files[0]);
    }}
}});
</script>
*/
"#, host, port)
}

fn generate_curl_client(host: &str, port: &str) -> String {
    format!(r#"#!/bin/bash
# SRGAN Web API Client Example using curl

API_URL="http://{}:{}/api"

# Function to upscale an image
upscale_image() {{
    local input_file="$1"
    local output_file="$2"
    
    # Encode image to base64
    image_data=$(base64 -i "$input_file" | tr -d '\n')
    
    # Create JSON payload
    json_payload=$(cat <<EOF
{{
    "image_data": "$image_data",
    "format": "png",
    "model": "natural"
}}
EOF
)
    
    # Send request and save response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$API_URL/upscale")
    
    # Extract base64 image from response
    echo "$response" | jq -r '.image_data' | base64 -d > "$output_file"
    
    # Display metadata
    echo "✓ Image upscaled to $output_file"
    echo "$response" | jq '.metadata'
}}

# Check server health
check_health() {{
    curl -s "$API_URL/health" | jq '.'
}}

# List available models
list_models() {{
    curl -s "$API_URL/models" | jq '.'
}}

# Example usage
# upscale_image "input.jpg" "output.png"
# check_health
# list_models
"#, host, port)
}

fn generate_rust_client(host: &str, port: &str) -> String {
    format!(r#"// SRGAN Web API Client Example in Rust

use reqwest;
use serde::{{Deserialize, Serialize}};
use base64;
use std::fs;

const API_URL: &str = "http://{}:{}/api";

#[derive(Serialize)]
struct UpscaleRequest {{
    image_data: String,
    format: Option<String>,
    model: Option<String>,
}}

#[derive(Deserialize)]
struct UpscaleResponse {{
    success: bool,
    image_data: Option<String>,
    error: Option<String>,
    metadata: ResponseMetadata,
}}

#[derive(Deserialize)]
struct ResponseMetadata {{
    original_size: (u32, u32),
    upscaled_size: (u32, u32),
    processing_time_ms: u64,
    format: String,
    model_used: String,
}}

async fn upscale_image(input_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {{
    // Read and encode image
    let image_data = fs::read(input_path)?;
    let encoded = base64::encode(&image_data);
    
    // Prepare request
    let request = UpscaleRequest {{
        image_data: encoded,
        format: Some("png".to_string()),
        model: Some("natural".to_string()),
    }};
    
    // Send request
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{{}}/upscale", API_URL))
        .json(&request)
        .send()
        .await?;
    
    let result: UpscaleResponse = response.json().await?;
    
    if result.success {{
        // Decode and save result
        if let Some(data) = result.image_data {{
            let decoded = base64::decode(data)?;
            fs::write(output_path, decoded)?;
            
            println!("✓ Upscaled image saved to {{}}", output_path);
            println!("  Original size: {{:?}}", result.metadata.original_size);
            println!("  Upscaled size: {{:?}}", result.metadata.upscaled_size);
            println!("  Processing time: {{}}ms", result.metadata.processing_time_ms);
        }}
    }} else {{
        eprintln!("✗ Error: {{}}", result.error.unwrap_or_default());
    }}
    
    Ok(())
}}

#[tokio::main]
async fn main() {{
    if let Err(e) = upscale_image("input.jpg", "output.png").await {{
        eprintln!("Error: {{}}", e);
    }}
}}
"#, host, port)
}