//! Inline HTML documentation server.
//!
//! - `GET /docs` — full API reference (dark theme, sidebar nav, code tabs)
//! - `GET /docs/openapi.json` — OpenAPI 3.0 specification
//! - `GET /pricing` — pricing page with feature comparison, FAQ, Stripe checkout

// ── OpenAPI 3.0 spec ─────────────────────────────────────────────────────────

/// Returns a JSON string containing the OpenAPI 3.0 specification.
pub fn render_openapi_spec() -> String {
    serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "SRGAN Image Super-Resolution API",
            "description": "AI-powered image and video upscaling service built in Rust.",
            "version": "1.0.0",
            "contact": { "email": "support@srgan.dev" },
            "license": { "name": "MIT" }
        },
        "servers": [
            { "url": "https://api.srgan.dev", "description": "Production" },
            { "url": "http://localhost:8080", "description": "Local development" }
        ],
        "security": [{ "BearerAuth": [] }],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "API key"
                }
            },
            "schemas": {
                "UpscaleResponse": {
                    "type": "object",
                    "properties": {
                        "job_id": { "type": "string" },
                        "status": { "type": "string", "enum": ["complete", "processing", "failed"] },
                        "output_url": { "type": "string" },
                        "width": { "type": "integer" },
                        "height": { "type": "integer" },
                        "processing_time_ms": { "type": "integer" }
                    }
                },
                "JobStatus": {
                    "type": "object",
                    "properties": {
                        "job_id": { "type": "string" },
                        "status": { "type": "string" },
                        "progress": { "type": "integer", "minimum": 0, "maximum": 100 },
                        "created_at": { "type": "string", "format": "date-time" }
                    }
                },
                "BatchResponse": {
                    "type": "object",
                    "properties": {
                        "batch_id": { "type": "string" },
                        "total": { "type": "integer" },
                        "status": { "type": "string" }
                    }
                },
                "VideoUpscaleResponse": {
                    "type": "object",
                    "properties": {
                        "job_id": { "type": "string" },
                        "status": { "type": "string" },
                        "frames_total": { "type": "integer" },
                        "output_url": { "type": "string" }
                    }
                },
                "ModelInfo": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "description": { "type": "string" },
                        "max_scale": { "type": "integer" }
                    }
                },
                "WebhookRegister": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": { "type": "string", "format": "uri" },
                        "secret": { "type": "string" },
                        "events": {
                            "type": "array",
                            "items": { "type": "string", "enum": ["job.completed", "job.failed", "batch.completed"] }
                        }
                    }
                },
                "WebhookPayload": {
                    "type": "object",
                    "properties": {
                        "event": { "type": "string" },
                        "timestamp": { "type": "integer" },
                        "data": { "type": "object" }
                    }
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": { "type": "string" },
                        "code": { "type": "string" },
                        "message": { "type": "string" }
                    }
                }
            }
        },
        "paths": {
            "/api/v1/upscale": {
                "post": {
                    "summary": "Upscale an image",
                    "operationId": "upscaleImage",
                    "tags": ["Image"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["image"],
                                    "properties": {
                                        "image": { "type": "string", "format": "binary", "description": "Image file (JPEG, PNG, WebP)" },
                                        "scale": { "type": "integer", "enum": [2, 4], "default": 4 },
                                        "model": { "type": "string", "enum": ["natural", "anime", "waifu2x", "real-esrgan"], "default": "natural" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Upscale complete", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/UpscaleResponse" } } } },
                        "400": { "description": "Invalid request" },
                        "401": { "description": "Unauthorized" },
                        "413": { "description": "File too large" },
                        "429": { "description": "Rate limited" }
                    }
                }
            },
            "/api/v1/upscale/async": {
                "post": {
                    "summary": "Upscale an image asynchronously",
                    "operationId": "upscaleImageAsync",
                    "tags": ["Image"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["image"],
                                    "properties": {
                                        "image": { "type": "string", "format": "binary" },
                                        "scale": { "type": "integer", "enum": [2, 4], "default": 4 },
                                        "model": { "type": "string", "default": "natural" },
                                        "webhook_url": { "type": "string", "format": "uri" },
                                        "webhook_secret": { "type": "string" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Job queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/JobStatus" } } } }
                    }
                }
            },
            "/api/v1/batch": {
                "post": {
                    "summary": "Batch upscale multiple images",
                    "operationId": "batchUpscale",
                    "tags": ["Batch"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["images"],
                                    "properties": {
                                        "images": { "type": "array", "items": { "type": "string", "format": "binary" } },
                                        "scale": { "type": "integer", "default": 4 },
                                        "model": { "type": "string", "default": "natural" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Batch queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/BatchResponse" } } } }
                    }
                }
            },
            "/api/v1/video/upscale": {
                "post": {
                    "summary": "Upscale a video",
                    "operationId": "upscaleVideo",
                    "tags": ["Video"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["video"],
                                    "properties": {
                                        "video": { "type": "string", "format": "binary", "description": "Video file (MP4, AVI, MKV)" },
                                        "scale": { "type": "integer", "enum": [2, 4], "default": 2 },
                                        "model": { "type": "string", "default": "natural" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Video job queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VideoUpscaleResponse" } } } }
                    }
                }
            },
            "/api/v1/models": {
                "get": {
                    "summary": "List available models",
                    "operationId": "listModels",
                    "tags": ["Models"],
                    "security": [],
                    "responses": {
                        "200": { "description": "Model list", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/ModelInfo" } } } } }
                    }
                }
            },
            "/api/v1/job/{id}": {
                "get": {
                    "summary": "Get job status",
                    "operationId": "getJobStatus",
                    "tags": ["Jobs"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Job status", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/JobStatus" } } } },
                        "404": { "description": "Job not found" }
                    }
                }
            },
            "/api/v1/result/{id}": {
                "get": {
                    "summary": "Download upscaled result",
                    "operationId": "getResult",
                    "tags": ["Jobs"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Image binary", "content": { "image/png": { "schema": { "type": "string", "format": "binary" } } } },
                        "404": { "description": "Result not found" }
                    }
                }
            },
            "/api/v1/webhooks": {
                "post": {
                    "summary": "Register a webhook endpoint",
                    "operationId": "registerWebhook",
                    "tags": ["Webhooks"],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/WebhookRegister" } } }
                    },
                    "responses": {
                        "201": { "description": "Webhook registered" }
                    }
                },
                "get": {
                    "summary": "List registered webhooks",
                    "operationId": "listWebhooks",
                    "tags": ["Webhooks"],
                    "responses": {
                        "200": { "description": "Webhook list" }
                    }
                }
            },
            "/api/v1/webhooks/test": {
                "post": {
                    "summary": "Send a test webhook ping",
                    "operationId": "testWebhook",
                    "tags": ["Webhooks"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["url"],
                                    "properties": {
                                        "url": { "type": "string", "format": "uri" },
                                        "secret": { "type": "string" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Test ping dispatched" }
                    }
                }
            },
            "/api/v1/health": {
                "get": {
                    "summary": "Health check",
                    "operationId": "healthCheck",
                    "tags": ["System"],
                    "security": [],
                    "responses": {
                        "200": { "description": "Service healthy" }
                    }
                }
            },
            "/api/v1/detect": {
                "post": {
                    "summary": "Auto-detect optimal model for an image",
                    "operationId": "detectModel",
                    "tags": ["Image"],
                    "requestBody": {
                        "required": true,
                        "content": { "multipart/form-data": { "schema": { "type": "object", "properties": { "image": { "type": "string", "format": "binary" } } } } }
                    },
                    "responses": {
                        "200": { "description": "Detection result" }
                    }
                }
            },
            "/api/register": {
                "post": {
                    "summary": "Register a new account",
                    "operationId": "register",
                    "tags": ["Auth"],
                    "security": [],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "type": "object", "required": ["email"], "properties": { "email": { "type": "string", "format": "email" } } } } }
                    },
                    "responses": {
                        "200": { "description": "API key issued" }
                    }
                }
            }
        },
        "tags": [
            { "name": "Image", "description": "Image upscaling endpoints" },
            { "name": "Video", "description": "Video upscaling endpoints" },
            { "name": "Batch", "description": "Batch processing endpoints" },
            { "name": "Jobs", "description": "Job status and result retrieval" },
            { "name": "Models", "description": "Model listing" },
            { "name": "Webhooks", "description": "Webhook management" },
            { "name": "Auth", "description": "Authentication and registration" },
            { "name": "System", "description": "System health and status" }
        ]
    }).to_string()
}

// ── Full API reference page ──────────────────────────────────────────────────

/// Returns a complete API reference page with sidebar navigation and code tabs.
pub fn render_full_docs_page() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRGAN API Reference</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--accent:#58a6ff;--accent2:#1f6feb;--text:#c9d1d9;--text-muted:#8b949e;--green:#3fb950;--yellow:#d29922;--red:#f85149;--font-mono:'SF Mono',SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;--font-sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--radius:8px}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);line-height:1.6;display:flex;min-height:100vh}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}

/* Top nav */
.topnav{position:fixed;top:0;left:0;right:0;height:52px;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 20px;z-index:100}
.topnav .logo{color:#f0f6fc;font-weight:700;font-size:1rem;margin-right:24px}
.topnav a{color:var(--text-muted);font-size:.85rem;margin-right:16px}
.topnav a:hover{color:#f0f6fc;text-decoration:none}

/* Sidebar */
.sidebar{position:fixed;top:52px;left:0;bottom:0;width:240px;background:var(--bg2);border-right:1px solid var(--border);overflow-y:auto;padding:16px 0;z-index:90}
.sidebar h4{color:var(--text-muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;padding:12px 20px 4px;margin-top:8px}
.sidebar a{display:block;padding:6px 20px;color:var(--text);font-size:.85rem;transition:background .15s}
.sidebar a:hover{background:var(--bg3);text-decoration:none}
.sidebar a.active{color:var(--accent);background:rgba(88,166,255,.08);border-right:2px solid var(--accent)}

/* Main content */
.main{margin-left:240px;margin-top:52px;flex:1;padding:32px 40px 80px;max-width:900px}
h1{font-size:1.8rem;color:#f0f6fc;margin-bottom:8px}
h2{font-size:1.4rem;color:#f0f6fc;margin:48px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border)}
h3{font-size:1.1rem;color:#e6edf3;margin:24px 0 8px}
p{margin:8px 0}
.subtitle{color:var(--text-muted);margin-bottom:32px}

/* Badges */
.badge{display:inline-block;padding:2px 10px;border-radius:4px;font-size:.75rem;font-weight:700;margin-right:8px;font-family:var(--font-mono)}
.badge-post{background:var(--accent2);color:#fff}
.badge-get{background:#238636;color:#fff}
.badge-delete{background:var(--red);color:#fff}
.badge-put{background:var(--yellow);color:#000}

/* Endpoint cards */
.endpoint{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:24px;margin:16px 0}
.endpoint-header{display:flex;align-items:center;margin-bottom:12px;font-family:var(--font-mono);font-size:.95rem}

/* Code tabs */
.code-tabs{margin:12px 0}
.tab-bar{display:flex;border-bottom:1px solid var(--border);margin-bottom:0}
.tab-bar button{background:none;border:none;color:var(--text-muted);padding:8px 16px;font-size:.8rem;cursor:pointer;border-bottom:2px solid transparent;font-family:var(--font-sans)}
.tab-bar button:hover{color:var(--text)}
.tab-bar button.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* Code blocks */
pre{background:var(--bg);border:1px solid var(--border);border-radius:0 0 var(--radius) var(--radius);padding:16px;overflow-x:auto;font-size:.82rem;line-height:1.5;margin:0}
code{font-family:var(--font-mono);font-size:.82rem}
.inline-code{background:var(--bg3);padding:2px 6px;border-radius:4px;font-size:.82rem}

/* Tables */
table{width:100%;border-collapse:collapse;margin:12px 0}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--border);font-size:.85rem}
th{color:var(--text-muted);font-weight:600;background:var(--bg2)}
td code{font-size:.8rem}

/* Callouts */
.callout{padding:12px 16px;border-radius:var(--radius);margin:12px 0;font-size:.85rem;border-left:3px solid}
.callout-info{background:rgba(88,166,255,.06);border-color:var(--accent)}
.callout-warn{background:rgba(210,153,34,.06);border-color:var(--yellow)}

@media(max-width:768px){.sidebar{display:none}.main{margin-left:0;padding:20px 16px}}
</style>
</head>
<body>

<div class="topnav">
<a class="logo" href="/">SRGAN</a>
<a href="/docs">API Docs</a>
<a href="/docs/webhooks">Webhooks</a>
<a href="/docs/sdk">SDKs</a>
<a href="/pricing">Pricing</a>
<a href="/demo">Demo</a>
<a href="/docs/openapi.json">OpenAPI Spec</a>
</div>

<nav class="sidebar">
<h4>Getting Started</h4>
<a href="#auth">Authentication</a>
<a href="#quickstart">Quick Start</a>
<a href="#errors">Error Handling</a>
<h4>Endpoints</h4>
<a href="#upscale">Image Upscale</a>
<a href="#upscale-async">Async Upscale</a>
<a href="#video">Video Upscale</a>
<a href="#batch">Batch Processing</a>
<a href="#models">List Models</a>
<a href="#detect">Auto-Detect Model</a>
<a href="#jobs">Job Status</a>
<a href="#result">Download Result</a>
<h4>Webhooks</h4>
<a href="#webhook-register">Register Webhook</a>
<a href="#webhook-list">List Webhooks</a>
<a href="#webhook-test">Test Webhook</a>
<h4>Account</h4>
<a href="#register">Register</a>
<a href="#rate-limits">Rate Limits</a>
<h4>Resources</h4>
<a href="/docs/webhooks">Webhook Guide</a>
<a href="/docs/sdk">SDK Reference</a>
<a href="/docs/openapi.json">OpenAPI 3.0</a>
</nav>

<div class="main">

<h1>API Reference</h1>
<p class="subtitle">SRGAN super-resolution image &amp; video upscaling API &mdash; v1</p>

<!-- ── Authentication ───────────────────────────────────────────── -->
<h2 id="auth">Authentication</h2>
<p>All API requests (except health check and registration) require a Bearer token in the <code class="inline-code">Authorization</code> header:</p>
<pre><code>Authorization: Bearer sk_live_your_api_key_here</code></pre>
<div class="callout callout-info">Get your API key by registering at <code class="inline-code">POST /api/register</code> or from the <a href="/admin">admin dashboard</a>.</div>

<!-- ── Quick Start ──────────────────────────────────────────────── -->
<h2 id="quickstart">Quick Start</h2>
<p>Upscale an image in one request:</p>

<div class="code-tabs" data-group="quickstart">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
<button onclick="switchTab(this,'python')">Python</button>
<button onclick="switchTab(this,'rust')">Rust</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl -X POST https://api.srgan.dev/api/v1/upscale \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F "scale=4" \
  -F "model=natural" \
  -o upscaled.png</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>import { SrganClient } from '@srgan/sdk';

const client = new SrganClient({ apiKey: 'YOUR_API_KEY' });
const result = await client.upscale({
  image: fs.readFileSync('photo.jpg'),
  scale: 4,
  model: 'natural',
});
fs.writeFileSync('upscaled.png', result.data);</code></pre>
</div>
<div class="tab-panel" data-tab="python">
<pre><code>import requests

resp = requests.post(
    "https://api.srgan.dev/api/v1/upscale",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    files={"image": open("photo.jpg", "rb")},
    data={"scale": 4, "model": "natural"},
)
with open("upscaled.png", "wb") as f:
    f.write(resp.content)</code></pre>
</div>
<div class="tab-panel" data-tab="rust">
<pre><code>use srgan_sdk::Client;

let client = Client::new("YOUR_API_KEY");
let result = client.upscale()
    .file("photo.jpg")
    .scale(4)
    .model("natural")
    .send()
    .await?;
std::fs::write("upscaled.png", result.bytes())?;</code></pre>
</div>
</div>

<!-- ── Error Handling ───────────────────────────────────────────── -->
<h2 id="errors">Error Handling</h2>
<p>Errors return JSON with a consistent shape:</p>
<pre><code>{
  "error": "rate_limited",
  "message": "Daily request quota exceeded. Upgrade at /pricing"
}</code></pre>
<table>
<tr><th>Status</th><th>Code</th><th>Description</th></tr>
<tr><td>400</td><td><code>bad_request</code></td><td>Missing or invalid parameters</td></tr>
<tr><td>401</td><td><code>unauthorized</code></td><td>Missing or invalid API key</td></tr>
<tr><td>402</td><td><code>payment_required</code></td><td>Plan upgrade required</td></tr>
<tr><td>413</td><td><code>file_too_large</code></td><td>Image exceeds plan's max file size</td></tr>
<tr><td>429</td><td><code>rate_limited</code></td><td>Request quota exceeded</td></tr>
<tr><td>500</td><td><code>internal_error</code></td><td>Server-side processing failure</td></tr>
</table>

<!-- ── POST /api/v1/upscale ─────────────────────────────────────── -->
<h2 id="upscale">Image Upscale</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span>&nbsp;<code>/api/v1/upscale</code></div>
<p>Synchronous super-resolution upscaling. Returns the upscaled image or a JSON result with a download URL.</p>
<h3>Parameters</h3>
<table>
<tr><th>Field</th><th>Type</th><th>Required</th><th>Description</th></tr>
<tr><td><code>image</code></td><td>file</td><td>Yes</td><td>JPEG, PNG, or WebP image</td></tr>
<tr><td><code>scale</code></td><td>integer</td><td>No</td><td>2 or 4 (default 4)</td></tr>
<tr><td><code>model</code></td><td>string</td><td>No</td><td>natural, anime, waifu2x, real-esrgan (default natural)</td></tr>
</table>

<div class="code-tabs" data-group="upscale">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
<button onclick="switchTab(this,'python')">Python</button>
<button onclick="switchTab(this,'rust')">Rust</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl -X POST https://api.srgan.dev/api/v1/upscale \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@photo.jpg" \
  -F "scale=4" \
  -F "model=natural"</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>const form = new FormData();
form.append('image', fs.createReadStream('photo.jpg'));
form.append('scale', '4');
form.append('model', 'natural');

const res = await fetch('https://api.srgan.dev/api/v1/upscale', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${API_KEY}` },
  body: form,
});</code></pre>
</div>
<div class="tab-panel" data-tab="python">
<pre><code>resp = requests.post(
    "https://api.srgan.dev/api/v1/upscale",
    headers={"Authorization": f"Bearer {API_KEY}"},
    files={"image": open("photo.jpg", "rb")},
    data={"scale": 4, "model": "natural"},
)</code></pre>
</div>
<div class="tab-panel" data-tab="rust">
<pre><code>let result = client.upscale()
    .file("photo.jpg")
    .scale(4)
    .model("natural")
    .send()
    .await?;</code></pre>
</div>
</div>

<h3>Response</h3>
<pre><code>{
  "job_id": "a1b2c3d4",
  "status": "complete",
  "output_url": "/api/result/a1b2c3d4",
  "width": 1920,
  "height": 1080,
  "processing_time_ms": 2340
}</code></pre>
</div>

<!-- ── POST /api/v1/upscale/async ───────────────────────────────── -->
<h2 id="upscale-async">Async Upscale</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span>&nbsp;<code>/api/v1/upscale/async</code></div>
<p>Queue an image for background processing. Returns immediately with a job ID. Optionally receive a webhook on completion.</p>
<h3>Additional Parameters</h3>
<table>
<tr><th>Field</th><th>Type</th><th>Description</th></tr>
<tr><td><code>webhook_url</code></td><td>string</td><td>URL to POST results to on completion</td></tr>
<tr><td><code>webhook_secret</code></td><td>string</td><td>HMAC secret for webhook signatures</td></tr>
</table>

<div class="code-tabs" data-group="async">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
<button onclick="switchTab(this,'python')">Python</button>
<button onclick="switchTab(this,'rust')">Rust</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl -X POST https://api.srgan.dev/api/v1/upscale/async \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@photo.jpg" \
  -F "scale=4" \
  -F "webhook_url=https://example.com/hook"</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>const result = await client.upscaleAsync({
  image: fs.readFileSync('photo.jpg'),
  scale: 4,
  webhookUrl: 'https://example.com/hook',
});
console.log(result.jobId); // poll or wait for webhook</code></pre>
</div>
<div class="tab-panel" data-tab="python">
<pre><code>resp = requests.post(
    "https://api.srgan.dev/api/v1/upscale/async",
    headers={"Authorization": f"Bearer {API_KEY}"},
    files={"image": open("photo.jpg", "rb")},
    data={"scale": 4, "webhook_url": "https://example.com/hook"},
)
job_id = resp.json()["job_id"]</code></pre>
</div>
<div class="tab-panel" data-tab="rust">
<pre><code>let job = client.upscale_async()
    .file("photo.jpg")
    .scale(4)
    .webhook_url("https://example.com/hook")
    .send()
    .await?;
println!("Job queued: {}", job.id);</code></pre>
</div>
</div>

<h3>Response (202 Accepted)</h3>
<pre><code>{
  "job_id": "f5e6d7c8",
  "status": "processing",
  "progress": 0,
  "created_at": "2026-03-27T10:00:00Z"
}</code></pre>
</div>

<!-- ── POST /api/v1/video/upscale ───────────────────────────────── -->
<h2 id="video">Video Upscale</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span>&nbsp;<code>/api/v1/video/upscale</code></div>
<p>Upload a video for frame-by-frame neural upscaling. Processing is asynchronous; use the returned job ID to track progress.</p>
<h3>Parameters</h3>
<table>
<tr><th>Field</th><th>Type</th><th>Required</th><th>Description</th></tr>
<tr><td><code>video</code></td><td>file</td><td>Yes</td><td>MP4, AVI, or MKV video</td></tr>
<tr><td><code>scale</code></td><td>integer</td><td>No</td><td>2 or 4 (default 2)</td></tr>
<tr><td><code>model</code></td><td>string</td><td>No</td><td>Model to use (default natural)</td></tr>
</table>

<div class="code-tabs" data-group="video">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
<button onclick="switchTab(this,'python')">Python</button>
<button onclick="switchTab(this,'rust')">Rust</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl -X POST https://api.srgan.dev/api/v1/video/upscale \
  -H "Authorization: Bearer $API_KEY" \
  -F "video=@clip.mp4" \
  -F "scale=2"</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>const job = await client.upscaleVideo({
  video: fs.readFileSync('clip.mp4'),
  scale: 2,
});
console.log(`Video job: ${job.jobId}`);</code></pre>
</div>
<div class="tab-panel" data-tab="python">
<pre><code>resp = requests.post(
    "https://api.srgan.dev/api/v1/video/upscale",
    headers={"Authorization": f"Bearer {API_KEY}"},
    files={"video": open("clip.mp4", "rb")},
    data={"scale": 2},
)</code></pre>
</div>
<div class="tab-panel" data-tab="rust">
<pre><code>let job = client.upscale_video()
    .file("clip.mp4")
    .scale(2)
    .send()
    .await?;</code></pre>
</div>
</div>

<h3>Response (202 Accepted)</h3>
<pre><code>{
  "job_id": "vid_a1b2c3",
  "status": "processing",
  "frames_total": 240,
  "output_url": null
}</code></pre>
</div>

<!-- ── POST /api/v1/batch ───────────────────────────────────────── -->
<h2 id="batch">Batch Processing</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span>&nbsp;<code>/api/v1/batch</code></div>
<p>Upload multiple images for batch processing. Returns a batch ID to track aggregate progress. Requires Pro plan or above.</p>

<div class="code-tabs" data-group="batch">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
<button onclick="switchTab(this,'python')">Python</button>
<button onclick="switchTab(this,'rust')">Rust</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl -X POST https://api.srgan.dev/api/v1/batch \
  -H "Authorization: Bearer $API_KEY" \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "images=@img3.jpg" \
  -F "scale=4"</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>const batch = await client.batch({
  images: ['img1.jpg', 'img2.jpg', 'img3.jpg'],
  scale: 4,
});
console.log(`Batch ${batch.batchId}: ${batch.total} images`);</code></pre>
</div>
<div class="tab-panel" data-tab="python">
<pre><code>files = [("images", open(f, "rb")) for f in ["img1.jpg", "img2.jpg", "img3.jpg"]]
resp = requests.post(
    "https://api.srgan.dev/api/v1/batch",
    headers={"Authorization": f"Bearer {API_KEY}"},
    files=files,
    data={"scale": 4},
)</code></pre>
</div>
<div class="tab-panel" data-tab="rust">
<pre><code>let batch = client.batch()
    .files(&["img1.jpg", "img2.jpg", "img3.jpg"])
    .scale(4)
    .send()
    .await?;</code></pre>
</div>
</div>

<h3>Response (202 Accepted)</h3>
<pre><code>{
  "batch_id": "batch_x9y8z7",
  "total": 3,
  "status": "processing"
}</code></pre>
</div>

<!-- ── GET /api/v1/models ───────────────────────────────────────── -->
<h2 id="models">List Models</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-get">GET</span>&nbsp;<code>/api/v1/models</code></div>
<p>List all available upscaling models. No authentication required.</p>

<div class="code-tabs" data-group="models">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
<button onclick="switchTab(this,'python')">Python</button>
<button onclick="switchTab(this,'rust')">Rust</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl https://api.srgan.dev/api/v1/models</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>const models = await client.listModels();
models.forEach(m => console.log(m.id, m.name));</code></pre>
</div>
<div class="tab-panel" data-tab="python">
<pre><code>resp = requests.get("https://api.srgan.dev/api/v1/models")
for m in resp.json():
    print(m["id"], m["name"])</code></pre>
</div>
<div class="tab-panel" data-tab="rust">
<pre><code>let models = client.list_models().await?;
for m in &models {
    println!("{}: {}", m.id, m.name);
}</code></pre>
</div>
</div>

<h3>Response</h3>
<pre><code>[
  {"id": "natural", "name": "Natural", "description": "L1-trained on natural photos", "max_scale": 4},
  {"id": "anime", "name": "Anime", "description": "L1-trained on animation frames", "max_scale": 4},
  {"id": "waifu2x", "name": "Waifu2x", "description": "VGG7 CNN denoising + upscaling", "max_scale": 2},
  {"id": "real-esrgan", "name": "Real-ESRGAN", "description": "Real-world degradation model", "max_scale": 4}
]</code></pre>
</div>

<!-- ── POST /api/v1/detect ──────────────────────────────────────── -->
<h2 id="detect">Auto-Detect Model</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span>&nbsp;<code>/api/v1/detect</code></div>
<p>Upload an image and the API recommends the best model. Uses content analysis to distinguish photos, anime, and artwork.</p>

<div class="code-tabs" data-group="detect">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl -X POST https://api.srgan.dev/api/v1/detect \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@photo.jpg"</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>const detection = await client.detect({
  image: fs.readFileSync('photo.jpg'),
});
console.log(detection.recommended_model);</code></pre>
</div>
</div>

<h3>Response</h3>
<pre><code>{
  "recommended_model": "natural",
  "confidence": 0.94,
  "analysis": {"type": "photograph", "noise_level": "low"}
}</code></pre>
</div>

<!-- ── GET /api/v1/job/:id ──────────────────────────────────────── -->
<h2 id="jobs">Job Status</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-get">GET</span>&nbsp;<code>/api/v1/job/{id}</code></div>
<p>Check the status and progress of an async upscaling job.</p>

<div class="code-tabs" data-group="job">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl https://api.srgan.dev/api/v1/job/a1b2c3d4 \
  -H "Authorization: Bearer $API_KEY"</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>const status = await client.getJob('a1b2c3d4');
console.log(status.progress + '%');</code></pre>
</div>
</div>

<h3>Response</h3>
<pre><code>{
  "job_id": "a1b2c3d4",
  "status": "processing",
  "progress": 65,
  "created_at": "2026-03-27T10:00:00Z"
}</code></pre>
</div>

<!-- ── GET /api/v1/result/:id ───────────────────────────────────── -->
<h2 id="result">Download Result</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-get">GET</span>&nbsp;<code>/api/v1/result/{id}</code></div>
<p>Download the upscaled image. Returns the binary image data (PNG by default).</p>
<pre><code>curl -o upscaled.png https://api.srgan.dev/api/v1/result/a1b2c3d4 \
  -H "Authorization: Bearer $API_KEY"</code></pre>
</div>

<!-- ── POST /api/v1/webhooks ────────────────────────────────────── -->
<h2 id="webhook-register">Register Webhook</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span>&nbsp;<code>/api/v1/webhooks</code></div>
<p>Register a URL to receive event notifications. See the <a href="/docs/webhooks">Webhook Guide</a> for payload schemas and retry behavior.</p>

<div class="code-tabs" data-group="wh-reg">
<div class="tab-bar">
<button class="active" onclick="switchTab(this,'curl')">cURL</button>
<button onclick="switchTab(this,'js')">JavaScript</button>
</div>
<div class="tab-panel active" data-tab="curl">
<pre><code>curl -X POST https://api.srgan.dev/api/v1/webhooks \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/webhook",
    "secret": "whsec_your_secret",
    "events": ["job.completed", "job.failed"]
  }'</code></pre>
</div>
<div class="tab-panel" data-tab="js">
<pre><code>await client.webhooks.register({
  url: 'https://example.com/webhook',
  secret: 'whsec_your_secret',
  events: ['job.completed', 'job.failed'],
});</code></pre>
</div>
</div>
</div>

<!-- ── GET /api/v1/webhooks ─────────────────────────────────────── -->
<h2 id="webhook-list">List Webhooks</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-get">GET</span>&nbsp;<code>/api/v1/webhooks</code></div>
<p>List all registered webhooks for the authenticated user.</p>
<pre><code>curl https://api.srgan.dev/api/v1/webhooks \
  -H "Authorization: Bearer $API_KEY"</code></pre>
</div>

<!-- ── POST /api/v1/webhooks/test ───────────────────────────────── -->
<h2 id="webhook-test">Test Webhook</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span>&nbsp;<code>/api/v1/webhooks/test</code></div>
<p>Send a test <code class="inline-code">ping</code> event to a URL. Useful for verifying your integration before going live.</p>
<pre><code>curl -X POST https://api.srgan.dev/api/v1/webhooks/test \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/webhook", "secret": "whsec_your_secret"}'</code></pre>
<h3>Response (202 Accepted)</h3>
<pre><code>{"status": "queued", "url": "https://example.com/webhook", "message": "Test ping dispatched"}</code></pre>
</div>

<!-- ── POST /api/register ───────────────────────────────────────── -->
<h2 id="register">Register</h2>
<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span>&nbsp;<code>/api/register</code></div>
<p>Create an account and receive an API key. Starts on the Free plan (100 images/month).</p>
<pre><code>curl -X POST https://api.srgan.dev/api/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'</code></pre>
<h3>Response</h3>
<pre><code>{
  "api_key": "sk_live_abc123...",
  "tier": "free",
  "email": "you@example.com"
}</code></pre>
</div>

<!-- ── Rate Limits ──────────────────────────────────────────────── -->
<h2 id="rate-limits">Rate Limits</h2>
<p>Limits are enforced per API key. Headers are included in every response:</p>
<table>
<tr><th>Header</th><th>Description</th></tr>
<tr><td><code>X-RateLimit-Limit</code></td><td>Total allowed requests per period</td></tr>
<tr><td><code>X-RateLimit-Remaining</code></td><td>Requests remaining</td></tr>
<tr><td><code>X-RateLimit-Reset</code></td><td>Unix timestamp when the limit resets</td></tr>
</table>
<table>
<tr><th>Plan</th><th>Monthly Quota</th><th>Max File Size</th><th>Max Scale</th></tr>
<tr><td>Free</td><td>100 images</td><td>5 MB</td><td>2x</td></tr>
<tr><td>Pro ($19/mo)</td><td>5,000 images</td><td>25 MB</td><td>4x</td></tr>
<tr><td>Enterprise</td><td>Unlimited</td><td>100 MB</td><td>8x</td></tr>
</table>
<div class="callout callout-warn">When you hit a rate limit, the API returns <code class="inline-code">429 Too Many Requests</code>. Back off and retry after the <code class="inline-code">X-RateLimit-Reset</code> time.</div>

</div>

<script>
function switchTab(btn, lang) {
  const group = btn.closest('.code-tabs');
  group.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
  group.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  group.querySelector('[data-tab="'+lang+'"]').classList.add('active');
}
// Smooth scroll for sidebar links
document.querySelectorAll('.sidebar a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const id = a.getAttribute('href').slice(1);
    const el = document.getElementById(id);
    if (el) { e.preventDefault(); el.scrollIntoView({behavior:'smooth',block:'start'}); }
  });
});
// Highlight active sidebar link on scroll
const sections = document.querySelectorAll('h2[id]');
const navLinks = document.querySelectorAll('.sidebar a[href^="#"]');
window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(s => { if (window.scrollY >= s.offsetTop - 80) current = s.id; });
  navLinks.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  });
});
</script>
</body>
</html>"##.to_string()
}

// ── Pricing page ─────────────────────────────────────────────────────────────

/// Returns an updated pricing page with Free (100 img/mo), Pro ($19/mo, 5000 img),
/// Enterprise (custom), feature comparison, FAQ, and Stripe checkout buttons.
pub fn render_full_pricing_page() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRGAN Pricing</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--accent:#58a6ff;--accent2:#1f6feb;--text:#c9d1d9;--text-muted:#8b949e;--green:#3fb950;--yellow:#d29922;--red:#f85149;--font-sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--radius:8px}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);line-height:1.6}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}

.topnav{background:var(--bg2);border-bottom:1px solid var(--border);padding:12px 24px;position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;align-items:center}
.topnav .left{display:flex;align-items:center;gap:16px}
.topnav .logo{color:#f0f6fc;font-weight:700;font-size:1rem}
.topnav a{color:var(--text-muted);font-size:.85rem}
.topnav a:hover{color:#f0f6fc;text-decoration:none}

.container{max-width:1100px;margin:0 auto;padding:60px 24px}
h1{font-size:2rem;color:#f0f6fc;text-align:center;margin-bottom:8px}
.subtitle{text-align:center;color:var(--text-muted);margin-bottom:48px}

/* Pricing cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px;margin-bottom:60px}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:32px 28px;display:flex;flex-direction:column}
.card.featured{border-color:var(--accent2);box-shadow:0 0 24px rgba(31,111,235,.15)}
.card .popular{background:var(--accent2);color:#fff;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;padding:4px 12px;border-radius:20px;display:inline-block;margin-bottom:12px;width:fit-content}
.card h2{color:#f0f6fc;font-size:1.3rem;margin-bottom:4px}
.card .price{font-size:2.2rem;color:#f0f6fc;font-weight:700;margin:16px 0 4px}
.card .price span{font-size:.9rem;color:var(--text-muted);font-weight:400}
.card .desc{color:var(--text-muted);font-size:.85rem;margin-bottom:20px}
.card ul{list-style:none;flex:1}
.card ul li{padding:6px 0;font-size:.9rem;color:var(--text)}
.card ul li::before{content:"\2713 ";color:var(--green);margin-right:6px}
.btn{display:block;text-align:center;padding:12px;border-radius:var(--radius);font-weight:600;font-size:.95rem;margin-top:24px;transition:all .2s;cursor:pointer;border:none}
.btn:hover{opacity:.85;text-decoration:none}
.btn-outline{border:1px solid var(--border);color:var(--text);background:transparent}
.btn-primary{background:var(--accent2);color:#fff}
.btn-enterprise{background:var(--green);color:#fff}

/* Feature table */
h3{font-size:1.3rem;color:#f0f6fc;text-align:center;margin-bottom:20px}
table{width:100%;border-collapse:collapse;margin:0 auto 60px}
th,td{text-align:center;padding:12px 16px;border-bottom:1px solid var(--border);font-size:.9rem}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-muted);font-weight:600;background:var(--bg2)}
.check{color:var(--green)}.cross{color:#484f58}

/* FAQ */
.faq{max-width:740px;margin:0 auto}
.faq h3{text-align:left;margin-bottom:24px}
.faq-item{border-bottom:1px solid var(--border);padding:16px 0}
.faq-q{font-weight:600;color:#f0f6fc;cursor:pointer;display:flex;justify-content:space-between;align-items:center}
.faq-q::after{content:"+";font-size:1.2rem;color:var(--text-muted);transition:transform .2s}
.faq-item.open .faq-q::after{transform:rotate(45deg)}
.faq-a{color:var(--text-muted);font-size:.9rem;max-height:0;overflow:hidden;transition:max-height .3s ease,padding .3s ease}
.faq-item.open .faq-a{max-height:200px;padding-top:12px}

footer{text-align:center;padding:32px;color:#484f58;font-size:.8rem;border-top:1px solid var(--border)}
</style>
</head>
<body>

<div class="topnav">
<div class="left">
<a class="logo" href="/">SRGAN</a>
<a href="/docs">Docs</a>
<a href="/pricing">Pricing</a>
<a href="/demo">Demo</a>
</div>
</div>

<div class="container">
<h1>Simple, transparent pricing</h1>
<p class="subtitle">Scale from prototype to production with a plan that fits.</p>

<div class="cards">

<div class="card">
<h2>Free</h2>
<div class="price">$0<span>/mo</span></div>
<div class="desc">For hobby projects and evaluation</div>
<ul>
<li>100 images / month</li>
<li>Max 2x upscale</li>
<li>5 MB file limit</li>
<li>Community support</li>
<li>Natural &amp; anime models</li>
</ul>
<a class="btn btn-outline" href="/api/register">Get started free</a>
</div>

<div class="card featured">
<div class="popular">Most popular</div>
<h2>Pro</h2>
<div class="price">$19<span>/mo</span></div>
<div class="desc">For teams shipping real products</div>
<ul>
<li>5,000 images / month</li>
<li>Up to 4x upscale</li>
<li>25 MB file limit</li>
<li>Priority email support</li>
<li>All models (waifu2x, Real-ESRGAN)</li>
<li>Async &amp; batch processing</li>
<li>Webhook notifications</li>
<li>Video upscaling</li>
</ul>
<a class="btn btn-primary" href="/api/v1/billing/checkout" onclick="startCheckout('pro');return false;">Upgrade to Pro</a>
</div>

<div class="card">
<h2>Enterprise</h2>
<div class="price">Custom</div>
<div class="desc">For high-volume and on-prem needs</div>
<ul>
<li>Unlimited images</li>
<li>Up to 8x upscale</li>
<li>100 MB file limit</li>
<li>Dedicated support &amp; SLA</li>
<li>Custom model training</li>
<li>Multi-tenant organizations</li>
<li>On-premises deployment</li>
<li>SSO &amp; audit logs</li>
</ul>
<a class="btn btn-enterprise" href="mailto:sales@srgan.dev">Contact sales</a>
</div>

</div>

<h3>Feature comparison</h3>
<table>
<tr><th>Feature</th><th>Free</th><th>Pro</th><th>Enterprise</th></tr>
<tr><td>Monthly quota</td><td>100</td><td>5,000</td><td>Unlimited</td></tr>
<tr><td>Max scale factor</td><td>2x</td><td>4x</td><td>8x</td></tr>
<tr><td>Max file size</td><td>5 MB</td><td>25 MB</td><td>100 MB</td></tr>
<tr><td>Models</td><td>natural, anime</td><td>All</td><td>All + custom</td></tr>
<tr><td>Async upscaling</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td><td class="check">&#x2713;</td></tr>
<tr><td>Batch processing</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td><td class="check">&#x2713;</td></tr>
<tr><td>Video upscaling</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td><td class="check">&#x2713;</td></tr>
<tr><td>Webhooks</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td><td class="check">&#x2713;</td></tr>
<tr><td>S3 / CDN upload</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td><td class="check">&#x2713;</td></tr>
<tr><td>Custom models</td><td class="cross">&#x2717;</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td></tr>
<tr><td>Organizations</td><td class="cross">&#x2717;</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td></tr>
<tr><td>On-premises</td><td class="cross">&#x2717;</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td></tr>
<tr><td>SLA</td><td class="cross">&#x2717;</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td></tr>
</table>

<div class="faq">
<h3>Frequently asked questions</h3>

<div class="faq-item">
<div class="faq-q" onclick="this.parentElement.classList.toggle('open')">What counts as an image?</div>
<div class="faq-a">Each image submitted to the upscale or batch endpoint counts as one image toward your monthly quota, regardless of resolution or scale factor.</div>
</div>

<div class="faq-item">
<div class="faq-q" onclick="this.parentElement.classList.toggle('open')">Can I change plans at any time?</div>
<div class="faq-a">Yes. Upgrades take effect immediately and are prorated. Downgrades take effect at the start of the next billing cycle.</div>
</div>

<div class="faq-item">
<div class="faq-q" onclick="this.parentElement.classList.toggle('open')">What happens if I exceed my quota?</div>
<div class="faq-a">Additional requests return HTTP 429 (rate limited). You can upgrade your plan at any time to increase your quota. Enterprise plans have no limits.</div>
</div>

<div class="faq-item">
<div class="faq-q" onclick="this.parentElement.classList.toggle('open')">Do you offer annual billing?</div>
<div class="faq-a">Yes. Annual Pro plans are $190/year (save ~17%). Contact sales for annual enterprise pricing.</div>
</div>

<div class="faq-item">
<div class="faq-q" onclick="this.parentElement.classList.toggle('open')">Is there a free trial for Pro?</div>
<div class="faq-a">The Free plan serves as your trial. Upgrade to Pro when you need higher quotas, more models, or batch processing.</div>
</div>

<div class="faq-item">
<div class="faq-q" onclick="this.parentElement.classList.toggle('open')">How does video upscaling billing work?</div>
<div class="faq-a">Each frame in a video counts as one image toward your monthly quota. A 10-second video at 24fps uses 240 images from your quota.</div>
</div>

</div>
</div>

<footer>&copy; 2026 SRGAN. All rights reserved.</footer>

<script>
function startCheckout(plan) {
  const key = sessionStorage.getItem('api_key') || prompt('Enter your API key to start checkout:');
  if (!key) return;
  sessionStorage.setItem('api_key', key);
  fetch('/api/v1/billing/checkout', {
    method: 'POST',
    headers: {'Content-Type':'application/json','Authorization':'Bearer '+key},
    body: JSON.stringify({plan: plan})
  })
  .then(r => r.json())
  .then(data => {
    if (data.checkout_url) window.location.href = data.checkout_url;
    else if (data.url) window.location.href = data.url;
    else alert(data.error || 'Checkout failed');
  })
  .catch(e => alert('Error: ' + e.message));
}
</script>
</body>
</html>"##.to_string()
}
