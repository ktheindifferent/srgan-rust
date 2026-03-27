//! SDK documentation page.
//!
//! `GET /docs/sdk` — SDK reference with JS/TS and Rust usage examples.

/// Returns the SDK documentation page HTML.
pub fn render_sdk_docs_page() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRGAN SDK Reference</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--accent:#58a6ff;--accent2:#1f6feb;--text:#c9d1d9;--text-muted:#8b949e;--green:#3fb950;--yellow:#d29922;--red:#f85149;--font-mono:'SF Mono',SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;--font-sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--radius:8px}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);line-height:1.6}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}

.topnav{position:fixed;top:0;left:0;right:0;height:52px;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 20px;z-index:100}
.topnav .logo{color:#f0f6fc;font-weight:700;font-size:1rem;margin-right:24px}
.topnav a{color:var(--text-muted);font-size:.85rem;margin-right:16px}
.topnav a:hover{color:#f0f6fc;text-decoration:none}

.container{max-width:860px;margin:72px auto 80px;padding:0 24px}
h1{font-size:1.8rem;color:#f0f6fc;margin-bottom:8px}
h2{font-size:1.3rem;color:#f0f6fc;margin:40px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border)}
h3{font-size:1.05rem;color:#e6edf3;margin:20px 0 8px}
p{margin:8px 0}
.subtitle{color:var(--text-muted);margin-bottom:32px}

pre{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:16px;overflow-x:auto;font-size:.82rem;line-height:1.5;margin:12px 0}
code{font-family:var(--font-mono);font-size:.82rem}
.inline-code{background:var(--bg3);padding:2px 6px;border-radius:4px;font-size:.82rem}

table{width:100%;border-collapse:collapse;margin:12px 0}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--border);font-size:.85rem}
th{color:var(--text-muted);font-weight:600;background:var(--bg2)}

.callout{padding:12px 16px;border-radius:var(--radius);margin:12px 0;font-size:.85rem;border-left:3px solid}
.callout-info{background:rgba(88,166,255,.06);border-color:var(--accent)}

/* Language tabs */
.lang-tabs{display:flex;gap:8px;margin:24px 0 16px}
.lang-tab{padding:8px 20px;border-radius:6px;font-size:.85rem;font-weight:600;cursor:pointer;border:1px solid var(--border);background:transparent;color:var(--text-muted);transition:all .15s}
.lang-tab:hover{color:var(--text);border-color:var(--text-muted)}
.lang-tab.active{background:var(--accent2);color:#fff;border-color:var(--accent2)}
.lang-section{display:none}
.lang-section.active{display:block}

.method-sig{background:var(--bg3);padding:8px 12px;border-radius:6px;font-family:var(--font-mono);font-size:.82rem;margin:8px 0;display:inline-block}

footer{text-align:center;padding:32px;color:#484f58;font-size:.8rem;border-top:1px solid var(--border);margin-top:60px}
</style>
</head>
<body>

<div class="topnav">
<a class="logo" href="/">SRGAN</a>
<a href="/docs">API Docs</a>
<a href="/docs/webhooks">Webhooks</a>
<a href="/docs/sdk">SDKs</a>
<a href="/pricing">Pricing</a>
</div>

<div class="container">

<h1>SDK Reference</h1>
<p class="subtitle">Official client libraries for JavaScript/TypeScript and Rust.</p>

<div class="lang-tabs">
<button class="lang-tab active" onclick="switchLang('js')">JavaScript / TypeScript</button>
<button class="lang-tab" onclick="switchLang('rust')">Rust</button>
</div>

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- JavaScript / TypeScript SDK                                    -->
<!-- ═══════════════════════════════════════════════════════════════ -->
<div class="lang-section active" id="lang-js">

<h2>Installation</h2>
<pre><code>npm install @srgan/sdk
# or
yarn add @srgan/sdk</code></pre>

<h2>Quick Start</h2>
<pre><code>import { SrganClient } from '@srgan/sdk';
import fs from 'fs';

const client = new SrganClient({
  apiKey: process.env.SRGAN_API_KEY,
  // baseUrl: 'http://localhost:8080',  // optional
});

// Upscale an image
const result = await client.upscale({
  image: fs.readFileSync('photo.jpg'),
  scale: 4,
  model: 'natural',
});
fs.writeFileSync('upscaled.png', result.data);
console.log(`Done in ${result.processingTimeMs}ms`);</code></pre>

<h2>Client Configuration</h2>
<pre><code>const client = new SrganClient({
  apiKey: string,          // Required. Your API key.
  baseUrl?: string,        // Server URL (default: https://api.srgan.dev)
  timeout?: number,        // Request timeout in ms (default: 120000)
  retries?: number,        // Auto-retry count (default: 2)
});</code></pre>

<h2>Methods</h2>

<h3>client.upscale(options)</h3>
<p>Synchronous image upscaling. Returns the upscaled image data.</p>
<pre><code>const result = await client.upscale({
  image: Buffer | ReadableStream,  // Image data
  scale?: 2 | 4,                   // Scale factor (default: 4)
  model?: string,                  // Model name (default: 'natural')
});

// result: {
//   jobId: string,
//   status: 'complete',
//   data: Buffer,
//   width: number,
//   height: number,
//   processingTimeMs: number,
//   outputUrl: string,
// }</code></pre>

<h3>client.upscaleAsync(options)</h3>
<p>Queue an image for background processing. Returns a job handle.</p>
<pre><code>const job = await client.upscaleAsync({
  image: Buffer,
  scale: 4,
  model: 'anime',
  webhookUrl: 'https://example.com/hook',  // optional
  webhookSecret: 'whsec_...',              // optional
});

console.log(job.jobId);  // "a1b2c3d4"
console.log(job.status); // "processing"

// Poll for completion
const status = await client.getJob(job.jobId);
if (status.status === 'complete') {
  const data = await client.getResult(job.jobId);
  fs.writeFileSync('output.png', data);
}</code></pre>

<h3>client.upscaleVideo(options)</h3>
<p>Upload a video for frame-by-frame upscaling.</p>
<pre><code>const job = await client.upscaleVideo({
  video: fs.readFileSync('clip.mp4'),
  scale: 2,
  model: 'natural',
});
console.log(`Video job: ${job.jobId}, frames: ${job.framesTotal}`);</code></pre>

<h3>client.batch(options)</h3>
<p>Process multiple images at once.</p>
<pre><code>const batch = await client.batch({
  images: [
    fs.readFileSync('img1.jpg'),
    fs.readFileSync('img2.jpg'),
    fs.readFileSync('img3.jpg'),
  ],
  scale: 4,
  model: 'natural',
});
console.log(`Batch ${batch.batchId}: ${batch.total} images`);</code></pre>

<h3>client.listModels()</h3>
<pre><code>const models = await client.listModels();
// [{ id: 'natural', name: 'Natural', ... }, ...]</code></pre>

<h3>client.detect(options)</h3>
<p>Auto-detect the best model for an image.</p>
<pre><code>const detection = await client.detect({
  image: fs.readFileSync('photo.jpg'),
});
console.log(detection.recommendedModel); // 'natural'</code></pre>

<h3>client.getJob(jobId)</h3>
<pre><code>const status = await client.getJob('a1b2c3d4');
// { jobId, status, progress, createdAt }</code></pre>

<h3>client.getResult(jobId)</h3>
<pre><code>const imageBuffer = await client.getResult('a1b2c3d4');
fs.writeFileSync('output.png', imageBuffer);</code></pre>

<h2>Webhook Helpers</h2>
<pre><code>import { verifyWebhookSignature } from '@srgan/sdk';

// Express middleware
app.post('/webhook', express.raw({type: 'application/json'}), (req, res) => {
  const isValid = verifyWebhookSignature(
    req.body,                                  // raw body buffer
    req.headers['x-srgan-signature'],          // signature header
    process.env.WEBHOOK_SECRET                 // your secret
  );
  if (!isValid) return res.status(401).send('Bad signature');

  const event = JSON.parse(req.body);
  switch (event.event) {
    case 'job.completed':
      console.log('Job done:', event.data.job_id);
      break;
    case 'job.failed':
      console.error('Job failed:', event.data.error);
      break;
  }
  res.sendStatus(200);
});</code></pre>

<h2>TypeScript Types</h2>
<pre><code>import type {
  UpscaleOptions,
  UpscaleResult,
  AsyncJob,
  BatchResult,
  VideoJob,
  ModelInfo,
  DetectionResult,
  JobStatus,
  WebhookEvent,
} from '@srgan/sdk';</code></pre>

<h2>Error Handling</h2>
<pre><code>import { SrganError, RateLimitError } from '@srgan/sdk';

try {
  await client.upscale({ image: buf });
} catch (e) {
  if (e instanceof RateLimitError) {
    console.log(`Rate limited. Retry after ${e.retryAfter}s`);
  } else if (e instanceof SrganError) {
    console.error(`API error: ${e.code} — ${e.message}`);
  }
}</code></pre>

</div>

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- Rust SDK                                                       -->
<!-- ═══════════════════════════════════════════════════════════════ -->
<div class="lang-section" id="lang-rust">

<h2>Installation</h2>
<pre><code># Cargo.toml
[dependencies]
srgan-sdk = "0.2"
tokio = { version = "1", features = ["full"] }</code></pre>

<h2>Quick Start</h2>
<pre><code>use srgan_sdk::Client;

#[tokio::main]
async fn main() -> Result&lt;(), srgan_sdk::Error&gt; {
    let client = Client::new("sk_live_your_api_key");

    // Upscale an image
    let result = client.upscale()
        .file("photo.jpg")
        .scale(4)
        .model("natural")
        .send()
        .await?;

    std::fs::write("upscaled.png", result.bytes())?;
    println!("Done in {}ms", result.processing_time_ms);
    Ok(())
}</code></pre>

<h2>Client Configuration</h2>
<pre><code>use srgan_sdk::{Client, ClientConfig};

let client = Client::with_config(ClientConfig {
    api_key: "sk_live_...".into(),
    base_url: Some("http://localhost:8080".into()),  // optional
    timeout: Some(std::time::Duration::from_secs(120)),
    retries: Some(2),
});</code></pre>

<h2>Methods</h2>

<h3>client.upscale()</h3>
<p>Synchronous image upscaling with builder pattern.</p>
<pre><code>let result = client.upscale()
    .file("photo.jpg")           // or .bytes(buf)
    .scale(4)                     // 2 or 4
    .model("natural")             // natural, anime, waifu2x, real-esrgan
    .send()
    .await?;

// result.job_id: String
// result.status: Status::Complete
// result.bytes(): Vec&lt;u8&gt;
// result.width: u32
// result.height: u32
// result.processing_time_ms: u64
// result.output_url: String</code></pre>

<h3>client.upscale_async()</h3>
<pre><code>let job = client.upscale_async()
    .file("photo.jpg")
    .scale(4)
    .webhook_url("https://example.com/hook")
    .webhook_secret("whsec_...")
    .send()
    .await?;

println!("Job queued: {}", job.id);

// Poll for completion
let status = client.get_job(&amp;job.id).await?;
if status.is_complete() {
    let data = client.get_result(&amp;job.id).await?;
    std::fs::write("output.png", data)?;
}</code></pre>

<h3>client.upscale_video()</h3>
<pre><code>let job = client.upscale_video()
    .file("clip.mp4")
    .scale(2)
    .send()
    .await?;

println!("Video job: {}, frames: {}", job.id, job.frames_total);</code></pre>

<h3>client.batch()</h3>
<pre><code>let batch = client.batch()
    .files(&amp;["img1.jpg", "img2.jpg", "img3.jpg"])
    .scale(4)
    .model("natural")
    .send()
    .await?;

println!("Batch {}: {} images", batch.batch_id, batch.total);</code></pre>

<h3>client.list_models()</h3>
<pre><code>let models = client.list_models().await?;
for m in &amp;models {
    println!("{}: {} (max {}x)", m.id, m.name, m.max_scale);
}</code></pre>

<h3>client.detect()</h3>
<pre><code>let detection = client.detect()
    .file("photo.jpg")
    .send()
    .await?;
println!("Recommended: {} ({:.0}%)", detection.model, detection.confidence * 100.0);</code></pre>

<h3>client.get_job() / client.get_result()</h3>
<pre><code>let status = client.get_job("a1b2c3d4").await?;
let image_bytes = client.get_result("a1b2c3d4").await?;</code></pre>

<h2>Webhook Verification</h2>
<pre><code>use srgan_sdk::webhooks::verify_signature;

fn handle_webhook(body: &amp;[u8], sig_header: &amp;str) -> bool {
    verify_signature(body, sig_header, "whsec_your_secret")
}

// In an Actix-web handler:
async fn webhook_handler(
    body: web::Bytes,
    req: HttpRequest,
) -> impl Responder {
    let sig = req.headers()
        .get("x-srgan-signature")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if !verify_signature(&amp;body, sig, &amp;std::env::var("WEBHOOK_SECRET").unwrap()) {
        return HttpResponse::Unauthorized().finish();
    }

    let event: srgan_sdk::WebhookEvent = serde_json::from_slice(&amp;body).unwrap();
    match event.event.as_str() {
        "job.completed" => log::info!("Job done: {:?}", event.data),
        "job.failed" => log::error!("Job failed: {:?}", event.data),
        _ => {}
    }
    HttpResponse::Ok().finish()
}</code></pre>

<h2>Error Handling</h2>
<pre><code>use srgan_sdk::{Error, ApiError};

match client.upscale().file("photo.jpg").send().await {
    Ok(result) => println!("Success: {}x{}", result.width, result.height),
    Err(Error::Api(ApiError::RateLimited { retry_after })) => {
        println!("Rate limited, retry in {}s", retry_after);
    }
    Err(Error::Api(e)) => {
        eprintln!("API error {}: {}", e.code(), e.message());
    }
    Err(e) => eprintln!("Error: {}", e),
}</code></pre>

<h2>Feature Flags</h2>
<pre><code># Cargo.toml
[dependencies]
srgan-sdk = { version = "0.2", features = ["rustls"] }  # Use rustls instead of native-tls
srgan-sdk = { version = "0.2", features = ["blocking"] } # Sync (blocking) client</code></pre>

</div>

</div>

<footer>&copy; 2026 SRGAN. All rights reserved.</footer>

<script>
function switchLang(lang) {
  document.querySelectorAll('.lang-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.lang-section').forEach(s => s.classList.remove('active'));
  document.querySelector('.lang-tab[onclick*="'+lang+'"]').classList.add('active');
  document.getElementById('lang-' + lang).classList.add('active');
}
</script>
</body>
</html>"##.to_string()
}
