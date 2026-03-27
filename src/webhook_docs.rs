//! Webhook documentation page and test UI.
//!
//! - `GET /docs/webhooks` — comprehensive webhook guide
//! - The test endpoint `POST /api/v1/webhooks/test` is already wired in web_server.rs

/// Returns the webhook documentation page HTML.
pub fn render_webhook_docs_page() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRGAN Webhook Guide</title>
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
.callout-warn{background:rgba(210,153,34,.06);border-color:var(--yellow)}

/* Test UI */
.test-ui{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:24px;margin:20px 0}
.test-ui h3{margin-top:0}
.form-group{margin-bottom:16px}
.form-group label{display:block;font-size:.85rem;color:var(--text-muted);margin-bottom:4px}
.form-group input{width:100%;padding:10px 12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:var(--font-mono);font-size:.85rem}
.form-group input:focus{outline:none;border-color:var(--accent)}
.test-btn{background:var(--accent2);color:#fff;border:none;padding:10px 24px;border-radius:6px;font-weight:600;cursor:pointer;font-size:.9rem}
.test-btn:hover{opacity:.85}
.test-result{margin-top:16px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;font-family:var(--font-mono);font-size:.82rem;white-space:pre-wrap;display:none}
.test-result.visible{display:block}
.test-result.success{border-color:var(--green)}
.test-result.error{border-color:var(--red)}

.badge{display:inline-block;padding:2px 10px;border-radius:4px;font-size:.75rem;font-weight:700;margin-right:6px;font-family:var(--font-mono)}
.badge-event{background:#238636;color:#fff}
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

<h1>Webhook Guide</h1>
<p class="subtitle">Receive real-time notifications when jobs complete, fail, or batches finish processing.</p>

<!-- Overview -->
<h2>Overview</h2>
<p>Webhooks let your server receive HTTP POST notifications for events in your SRGAN account. Instead of polling for job status, register a URL and we'll push updates to you.</p>
<div class="callout callout-info">Webhooks require a Pro or Enterprise plan. Free-tier accounts can use polling via <code class="inline-code">GET /api/v1/job/{id}</code>.</div>

<!-- Event Types -->
<h2>Event Types</h2>
<table>
<tr><th>Event</th><th>Trigger</th><th>Description</th></tr>
<tr><td><span class="badge badge-event">job.completed</span></td><td>Upscale finishes</td><td>Fired when an image or video upscale job completes successfully.</td></tr>
<tr><td><span class="badge badge-event">job.failed</span></td><td>Upscale fails</td><td>Fired when a job fails due to an error (bad input, processing error, etc.).</td></tr>
<tr><td><span class="badge badge-event">batch.completed</span></td><td>Batch finishes</td><td>Fired when all images in a batch have been processed.</td></tr>
</table>

<!-- Registration -->
<h2>Registering a Webhook</h2>
<pre><code>curl -X POST https://api.srgan.dev/api/v1/webhooks \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/srgan-webhook",
    "secret": "whsec_your_secret_here",
    "events": ["job.completed", "job.failed", "batch.completed"]
  }'</code></pre>
<p>If <code class="inline-code">events</code> is omitted, all event types are subscribed by default.</p>

<!-- Payload Schema -->
<h2>Payload Schemas</h2>

<h3>Common Fields</h3>
<p>Every webhook POST includes these fields:</p>
<pre><code>{
  "event": "job.completed",
  "timestamp": 1711526400,
  "data": { ... }
}</code></pre>

<h3>job.completed</h3>
<pre><code>{
  "event": "job.completed",
  "timestamp": 1711526400,
  "data": {
    "job_id": "a1b2c3d4",
    "output_url": "/api/result/a1b2c3d4",
    "width": 1920,
    "height": 1080,
    "processing_time_ms": 2340,
    "model": "natural",
    "scale": 4
  }
}</code></pre>

<h3>job.failed</h3>
<pre><code>{
  "event": "job.failed",
  "timestamp": 1711526400,
  "data": {
    "job_id": "e5f6g7h8",
    "error": "unsupported_format",
    "message": "Could not decode input image"
  }
}</code></pre>

<h3>batch.completed</h3>
<pre><code>{
  "event": "batch.completed",
  "timestamp": 1711526400,
  "data": {
    "batch_id": "batch_x9y8z7",
    "total": 25,
    "succeeded": 24,
    "failed": 1,
    "processing_time_ms": 58200
  }
}</code></pre>

<!-- Signature Verification -->
<h2>Signature Verification</h2>
<p>If you provide a <code class="inline-code">secret</code> when registering, every delivery includes an <code class="inline-code">X-SRGAN-Signature</code> header containing an HMAC-SHA256 hex digest of the request body.</p>

<h3>Verifying in Node.js</h3>
<pre><code>const crypto = require('crypto');

function verifyWebhook(body, signature, secret) {
  const expected = crypto
    .createHmac('sha256', secret)
    .update(body)
    .digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expected)
  );
}

// In your Express handler:
app.post('/srgan-webhook', (req, res) => {
  const sig = req.headers['x-srgan-signature'];
  if (!verifyWebhook(req.rawBody, sig, process.env.WEBHOOK_SECRET)) {
    return res.status(401).send('Invalid signature');
  }
  const event = req.body;
  console.log(`Event: ${event.event}, Job: ${event.data.job_id}`);
  res.sendStatus(200);
});</code></pre>

<h3>Verifying in Python</h3>
<pre><code>import hmac, hashlib

def verify_webhook(body: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

# In your Flask handler:
@app.route('/srgan-webhook', methods=['POST'])
def handle_webhook():
    sig = request.headers.get('X-SRGAN-Signature', '')
    if not verify_webhook(request.data, sig, WEBHOOK_SECRET):
        return 'Invalid signature', 401
    event = request.json
    print(f"Event: {event['event']}")
    return '', 200</code></pre>

<!-- Retry Behavior -->
<h2>Retry Behavior</h2>
<p>Failed deliveries (non-2xx response or timeout) are retried with exponential backoff:</p>
<table>
<tr><th>Attempt</th><th>Delay</th><th>Cumulative</th></tr>
<tr><td>1 (initial)</td><td>Immediate</td><td>0</td></tr>
<tr><td>2</td><td>1 minute</td><td>1 min</td></tr>
<tr><td>3</td><td>5 minutes</td><td>6 min</td></tr>
<tr><td>4</td><td>15 minutes</td><td>21 min</td></tr>
<tr><td>5</td><td>1 hour</td><td>~1.3 hr</td></tr>
</table>
<div class="callout callout-warn">After 5 failed attempts, the delivery is moved to a <strong>dead letter queue</strong>. Check the admin dashboard to inspect and retry dead-lettered events.</div>
<p>Your endpoint must respond with a <code class="inline-code">2xx</code> status within <strong>10 seconds</strong> to be considered successful. Process long-running work asynchronously after acknowledging the webhook.</p>

<!-- Best Practices -->
<h2>Best Practices</h2>
<ul style="margin:12px 0;padding-left:24px">
<li>Always verify the <code class="inline-code">X-SRGAN-Signature</code> header to prevent spoofed requests.</li>
<li>Respond with <code class="inline-code">200 OK</code> immediately, then process the event asynchronously.</li>
<li>Handle duplicate deliveries idempotently &mdash; use the <code class="inline-code">job_id</code> as a deduplication key.</li>
<li>Use HTTPS endpoints only in production.</li>
<li>Monitor the dead letter queue in the <a href="/admin">admin dashboard</a>.</li>
</ul>

<!-- Test UI -->
<h2>Test Your Webhook</h2>
<p>Use this tool to send a test <code class="inline-code">ping</code> event to your endpoint and verify it's working correctly.</p>

<div class="test-ui">
<h3>Send Test Ping</h3>
<div class="form-group">
<label>API Key</label>
<input type="text" id="test-key" placeholder="sk_live_your_api_key">
</div>
<div class="form-group">
<label>Webhook URL</label>
<input type="url" id="test-url" placeholder="https://example.com/srgan-webhook">
</div>
<div class="form-group">
<label>Secret (optional)</label>
<input type="text" id="test-secret" placeholder="whsec_your_secret">
</div>
<button class="test-btn" onclick="sendTestWebhook()">Send Test Ping</button>
<div id="test-result" class="test-result"></div>
</div>

</div>

<script>
async function sendTestWebhook() {
  const key = document.getElementById('test-key').value.trim();
  const url = document.getElementById('test-url').value.trim();
  const secret = document.getElementById('test-secret').value.trim();
  const result = document.getElementById('test-result');

  if (!key || !url) {
    result.textContent = 'Error: API key and webhook URL are required.';
    result.className = 'test-result visible error';
    return;
  }

  result.textContent = 'Sending test ping...';
  result.className = 'test-result visible';

  try {
    const body = { url: url };
    if (secret) body.secret = secret;

    const resp = await fetch('/api/v1/webhooks/test', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + key,
      },
      body: JSON.stringify(body),
    });
    const data = await resp.json();

    if (resp.ok) {
      result.textContent = JSON.stringify(data, null, 2);
      result.className = 'test-result visible success';
    } else {
      result.textContent = 'HTTP ' + resp.status + '\n' + JSON.stringify(data, null, 2);
      result.className = 'test-result visible error';
    }
  } catch (e) {
    result.textContent = 'Network error: ' + e.message;
    result.className = 'test-result visible error';
  }
}

// Pre-fill API key from session if available
window.addEventListener('DOMContentLoaded', () => {
  const saved = sessionStorage.getItem('api_key');
  if (saved) document.getElementById('test-key').value = saved;
});
</script>
</body>
</html>"##.to_string()
}
