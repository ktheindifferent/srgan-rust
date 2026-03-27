/// GET /docs — full API reference page (delegates to docs module)
pub fn render_docs_page() -> String {
    crate::docs::render_full_docs_page()
}

/// GET /pricing — pricing page with FAQ and Stripe checkout (delegates to docs module)
pub fn render_pricing_page() -> String {
    crate::docs::render_full_pricing_page()
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
<a href="/docs/webhooks">Webhooks</a>
<a href="/docs/sdk">SDKs</a>
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
<p>From 100 free images/month to unlimited enterprise volumes. Batch processing and async jobs built in.</p>
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
