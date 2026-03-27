/// GET /docs — API documentation page
pub fn render_docs_page() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRGAN API Documentation</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,monospace;line-height:1.6}
a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
.container{max-width:900px;margin:0 auto;padding:40px 24px}
h1{font-size:2rem;color:#f0f6fc;margin-bottom:8px}
h2{font-size:1.4rem;color:#f0f6fc;margin:40px 0 16px;padding-bottom:8px;border-bottom:1px solid #21262d}
h3{font-size:1.1rem;color:#e6edf3;margin:24px 0 8px}
p{margin:8px 0}
nav{background:#161b22;border-bottom:1px solid #21262d;padding:12px 24px;position:sticky;top:0;z-index:10}
nav a{margin-right:20px;font-size:.9rem;color:#8b949e}nav a:hover{color:#f0f6fc}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:700;margin-right:8px}
.badge-post{background:#1f6feb;color:#fff}
.badge-get{background:#238636;color:#fff}
pre{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px;overflow-x:auto;margin:12px 0;font-size:.85rem;line-height:1.5}
code{font-family:'SF Mono',SFMono-Regular,Consolas,monospace;font-size:.85rem}
table{width:100%;border-collapse:collapse;margin:12px 0}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #21262d;font-size:.9rem}
th{color:#8b949e;font-weight:600;background:#161b22}
.endpoint{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:20px;margin:16px 0}
.endpoint-header{display:flex;align-items:center;margin-bottom:12px}
</style>
</head>
<body>
<nav>
<a href="/">Home</a>
<a href="/docs">Docs</a>
<a href="/pricing">Pricing</a>
<a href="/demo">Demo</a>
</nav>
<div class="container">
<h1>API Documentation</h1>
<p style="color:#8b949e">SRGAN super-resolution image upscaling API &mdash; v1</p>

<h2>Getting Started</h2>
<p>The SRGAN API lets you upscale images using state-of-the-art neural networks. All requests go to your server's base URL. Responses are JSON unless you request a binary image download.</p>
<pre><code>BASE_URL = https://your-server.example.com</code></pre>

<h2>Authentication</h2>
<p>Include your API key in the <code>Authorization</code> header:</p>
<pre><code>Authorization: Bearer YOUR_API_KEY</code></pre>
<p>Obtain a key by registering via <code>POST /api/register</code> with a JSON body containing <code>{"email":"you@example.com"}</code>.</p>

<h2>Endpoints</h2>

<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-post">POST</span><code>/api/v1/upscale</code></div>
<p>Upload an image for synchronous super-resolution upscaling.</p>
<h3>Request</h3>
<pre><code>curl -X POST $BASE_URL/api/v1/upscale \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F "scale=4" \
  -F "model=natural" \
  -o upscaled.png</code></pre>
<h3>Parameters</h3>
<table>
<tr><th>Field</th><th>Type</th><th>Required</th><th>Description</th></tr>
<tr><td>image</td><td>file</td><td>Yes</td><td>Image file (JPEG, PNG, WebP)</td></tr>
<tr><td>scale</td><td>integer</td><td>No</td><td>Upscale factor: 2 or 4 (default 4)</td></tr>
<tr><td>model</td><td>string</td><td>No</td><td>Model: natural, anime, waifu2x (default natural)</td></tr>
</table>
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

<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-get">GET</span><code>/api/v1/job/{id}</code></div>
<p>Check the status of an upscaling job (useful for async requests).</p>
<h3>Request</h3>
<pre><code>curl $BASE_URL/api/v1/job/a1b2c3d4 \
  -H "Authorization: Bearer YOUR_API_KEY"</code></pre>
<h3>Response</h3>
<pre><code>{
  "job_id": "a1b2c3d4",
  "status": "processing",
  "progress": 65,
  "created_at": "2026-03-27T10:00:00Z"
}</code></pre>
</div>

<div class="endpoint">
<div class="endpoint-header"><span class="badge badge-get">GET</span><code>/api/v1/health</code></div>
<p>Health check endpoint. No authentication required.</p>
<h3>Response</h3>
<pre><code>{
  "status": "ok",
  "model_loaded": true,
  "version": "0.2.0"
}</code></pre>
</div>

<h2>Rate Limits</h2>
<table>
<tr><th>Plan</th><th>Requests / day</th><th>Max file size</th><th>Max scale</th></tr>
<tr><td>Free</td><td>100</td><td>5 MB</td><td>2x</td></tr>
<tr><td>Pro</td><td>10,000</td><td>25 MB</td><td>4x</td></tr>
<tr><td>Enterprise</td><td>Unlimited</td><td>100 MB</td><td>8x</td></tr>
</table>
<p>Rate-limit headers are included in every response: <code>X-RateLimit-Remaining</code>, <code>X-RateLimit-Reset</code>.</p>

<h2>Error Codes</h2>
<table>
<tr><th>Status</th><th>Code</th><th>Description</th></tr>
<tr><td>400</td><td>bad_request</td><td>Missing or invalid parameters</td></tr>
<tr><td>401</td><td>unauthorized</td><td>Missing or invalid API key</td></tr>
<tr><td>402</td><td>payment_required</td><td>Plan upgrade required for this feature</td></tr>
<tr><td>413</td><td>file_too_large</td><td>Image exceeds plan's max file size</td></tr>
<tr><td>429</td><td>rate_limited</td><td>Daily request quota exceeded</td></tr>
<tr><td>500</td><td>internal_error</td><td>Server-side processing failure</td></tr>
</table>

</div>
</body>
</html>"##.to_string()
}

/// GET /pricing — pricing plans page
pub fn render_pricing_page() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRGAN Pricing</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.6}
a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
nav{background:#161b22;border-bottom:1px solid #21262d;padding:12px 24px;position:sticky;top:0;z-index:10}
nav a{margin-right:20px;font-size:.9rem;color:#8b949e}nav a:hover{color:#f0f6fc}
.container{max-width:1100px;margin:0 auto;padding:60px 24px}
h1{font-size:2rem;color:#f0f6fc;text-align:center;margin-bottom:8px}
.subtitle{text-align:center;color:#8b949e;margin-bottom:48px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px;margin-bottom:60px}
.card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:32px 28px;display:flex;flex-direction:column}
.card.featured{border-color:#1f6feb;box-shadow:0 0 20px rgba(31,111,235,.15)}
.card h2{color:#f0f6fc;font-size:1.3rem;margin-bottom:4px}
.card .price{font-size:2.2rem;color:#f0f6fc;font-weight:700;margin:16px 0 4px}
.card .price span{font-size:.9rem;color:#8b949e;font-weight:400}
.card .desc{color:#8b949e;font-size:.85rem;margin-bottom:20px}
.card ul{list-style:none;flex:1}
.card ul li{padding:6px 0;font-size:.9rem;color:#c9d1d9}
.card ul li::before{content:"\2713 ";color:#3fb950;margin-right:6px}
.btn{display:block;text-align:center;padding:12px;border-radius:8px;font-weight:600;font-size:.95rem;margin-top:24px;transition:opacity .2s}
.btn:hover{opacity:.85;text-decoration:none}
.btn-outline{border:1px solid #30363d;color:#c9d1d9;background:transparent}
.btn-primary{background:#1f6feb;color:#fff;border:none}
.btn-enterprise{background:#238636;color:#fff;border:none}
h3{font-size:1.3rem;color:#f0f6fc;text-align:center;margin-bottom:20px}
table{width:100%;border-collapse:collapse;margin:0 auto}
th,td{text-align:center;padding:12px 16px;border-bottom:1px solid #21262d;font-size:.9rem}
th:first-child,td:first-child{text-align:left}
th{color:#8b949e;font-weight:600;background:#161b22}
.check{color:#3fb950}.cross{color:#484f58}
</style>
</head>
<body>
<nav>
<a href="/">Home</a>
<a href="/docs">Docs</a>
<a href="/pricing">Pricing</a>
<a href="/demo">Demo</a>
</nav>
<div class="container">
<h1>Simple, transparent pricing</h1>
<p class="subtitle">Scale from prototype to production with a plan that fits.</p>

<div class="cards">
<div class="card">
<h2>Free</h2>
<div class="price">$0<span>/mo</span></div>
<div class="desc">For hobby projects and evaluation</div>
<ul>
<li>100 images / day</li>
<li>Max 2x upscale</li>
<li>5 MB file limit</li>
<li>Community support</li>
<li>natural &amp; anime models</li>
</ul>
<a class="btn btn-outline" href="/api/register">Get started free</a>
</div>

<div class="card featured">
<h2>Pro</h2>
<div class="price">$29<span>/mo</span></div>
<div class="desc">For teams shipping real products</div>
<ul>
<li>10,000 images / day</li>
<li>Up to 4x upscale</li>
<li>25 MB file limit</li>
<li>Priority support</li>
<li>All models including waifu2x</li>
<li>Async batch processing</li>
<li>Webhook notifications</li>
</ul>
<a class="btn btn-primary" href="/checkout?plan=pro">Upgrade to Pro</a>
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
<li>All models + custom training</li>
<li>Multi-tenant organizations</li>
<li>On-premises deployment</li>
<li>SSO &amp; audit logs</li>
</ul>
<a class="btn btn-enterprise" href="/checkout?plan=enterprise">Contact sales</a>
</div>
</div>

<h3>Feature comparison</h3>
<table>
<tr><th>Feature</th><th>Free</th><th>Pro</th><th>Enterprise</th></tr>
<tr><td>Daily quota</td><td>100</td><td>10,000</td><td>Unlimited</td></tr>
<tr><td>Max scale factor</td><td>2x</td><td>4x</td><td>8x</td></tr>
<tr><td>Max file size</td><td>5 MB</td><td>25 MB</td><td>100 MB</td></tr>
<tr><td>Async upscaling</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td><td class="check">&#x2713;</td></tr>
<tr><td>Batch processing</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td><td class="check">&#x2713;</td></tr>
<tr><td>Webhooks</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td><td class="check">&#x2713;</td></tr>
<tr><td>Custom models</td><td class="cross">&#x2717;</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td></tr>
<tr><td>Organizations</td><td class="cross">&#x2717;</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td></tr>
<tr><td>On-premises</td><td class="cross">&#x2717;</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td></tr>
<tr><td>SLA</td><td class="cross">&#x2717;</td><td class="cross">&#x2717;</td><td class="check">&#x2713;</td></tr>
</table>
</div>
</body>
</html>"##.to_string()
}

/// GET / or GET /landing — marketing landing page
pub fn render_landing_page() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRGAN — AI-Powered Image Super-Resolution API</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.6}
a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
nav{background:#161b22;border-bottom:1px solid #21262d;padding:12px 24px;position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;align-items:center}
nav .left a{margin-right:20px;font-size:.9rem;color:#8b949e}nav .left a:hover{color:#f0f6fc}
nav .right a{margin-left:12px}
.btn{display:inline-block;padding:10px 20px;border-radius:8px;font-weight:600;font-size:.9rem;transition:opacity .2s}
.btn:hover{opacity:.85;text-decoration:none}
.btn-primary{background:#1f6feb;color:#fff}
.btn-outline{border:1px solid #30363d;color:#c9d1d9}
.hero{text-align:center;padding:100px 24px 80px;max-width:800px;margin:0 auto}
.hero h1{font-size:2.8rem;color:#f0f6fc;line-height:1.2;margin-bottom:16px}
.hero p{font-size:1.15rem;color:#8b949e;margin-bottom:36px}
.hero .ctas a{margin:0 8px}
.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:24px;max-width:960px;margin:0 auto 80px;padding:0 24px}
.feature{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:28px 24px}
.feature h3{color:#f0f6fc;margin-bottom:8px;font-size:1.05rem}
.feature p{font-size:.9rem;color:#8b949e}
.section{max-width:800px;margin:0 auto;padding:0 24px 80px}
.section h2{font-size:1.6rem;color:#f0f6fc;text-align:center;margin-bottom:24px}
pre{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:20px;overflow-x:auto;font-size:.85rem;line-height:1.5}
code{font-family:'SF Mono',SFMono-Regular,Consolas,monospace}
.social-proof{text-align:center;padding:60px 24px;border-top:1px solid #21262d}
.social-proof h2{font-size:1.4rem;color:#f0f6fc;margin-bottom:12px}
.social-proof p{color:#8b949e;font-size:1rem;margin-bottom:24px}
.stats{display:flex;justify-content:center;gap:60px;flex-wrap:wrap}
.stat .num{font-size:2rem;font-weight:700;color:#f0f6fc}
.stat .label{font-size:.85rem;color:#8b949e}
footer{text-align:center;padding:32px;color:#484f58;font-size:.8rem;border-top:1px solid #21262d}
</style>
</head>
<body>
<nav>
<div class="left">
<a href="/" style="color:#f0f6fc;font-weight:700;font-size:1rem">SRGAN</a>
<a href="/docs">Docs</a>
<a href="/pricing">Pricing</a>
<a href="/demo">Demo</a>
</div>
<div class="right">
<a class="btn btn-outline" href="/api/register">Sign up</a>
<a class="btn btn-primary" href="/pricing">Get started</a>
</div>
</nav>

<section class="hero">
<h1>AI-Powered Image Super-Resolution API</h1>
<p>Upscale images 2&ndash;8x with neural networks trained on millions of photos. Ship sharper visuals in minutes, not months.</p>
<div class="ctas">
<a class="btn btn-primary" href="/pricing">View pricing</a>
<a class="btn btn-outline" href="/docs">Read the docs</a>
</div>
</section>

<div class="features">
<div class="feature">
<h3>Fast</h3>
<p>Sub-second upscaling for typical images. Optimized Rust backend with optional GPU acceleration.</p>
</div>
<div class="feature">
<h3>Accurate</h3>
<p>State-of-the-art PSNR and perceptual quality. Multiple models tuned for photos, anime, and art.</p>
</div>
<div class="feature">
<h3>Simple API</h3>
<p>One POST request. Send an image, get a sharper one back. SDKs and webhooks for production workflows.</p>
</div>
<div class="feature">
<h3>Scalable</h3>
<p>From 100 free images/day to unlimited enterprise volumes. Batch processing and async jobs built in.</p>
</div>
</div>

<div class="section">
<h2>Up and running in 30 seconds</h2>
<pre><code># 1. Get your API key
curl -X POST https://api.srgan.dev/api/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com"}'

# 2. Upscale an image
curl -X POST https://api.srgan.dev/api/v1/upscale \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F "scale=4" \
  -o upscaled.png</code></pre>
</div>

<section class="social-proof">
<h2>Trusted by developers worldwide</h2>
<p>Powering image pipelines from indie apps to enterprise platforms.</p>
<div class="stats">
<div class="stat"><div class="num">10M+</div><div class="label">Images upscaled</div></div>
<div class="stat"><div class="num">2,500+</div><div class="label">Developers</div></div>
<div class="stat"><div class="num">99.9%</div><div class="label">Uptime SLA</div></div>
<div class="stat"><div class="num">&lt;500ms</div><div class="label">Avg latency</div></div>
</div>
</section>

<footer>&copy; 2026 SRGAN. All rights reserved.</footer>
</body>
</html>"##.to_string()
}
