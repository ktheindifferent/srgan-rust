# API Reference

Base URL: `http://localhost:8080` (or your deployed domain)

All endpoints under `/api/v1/` require an `Authorization: Bearer <token>` header unless noted otherwise.

---

## Authentication

API keys are issued per subscription tier (Free, Pro, Enterprise). Include the key in every request:

```
Authorization: Bearer srgan_live_xxxxxxxxxxxxxxxxxxxx
```

Errors return HTTP 401 (missing/invalid key) or HTTP 403 (tier does not permit this endpoint).

---

## GET /api/v1/health

Health check. No authentication required.

**Response 200**

```json
{
  "status": "ok",
  "version": "0.2.0",
  "model_loaded": true,
  "model": "natural",
  "model_factor": 4,
  "uptime": 1711234567
}
```

**Response 503** — model not loaded or circuit-breaker open.

---

## GET /api/v1/models

List all available models.

**Response 200**

```json
{
  "models": [
    {
      "name": "natural",
      "description": "Neural net trained on natural photographs with L1 loss",
      "architecture": "srgan",
      "scale_factor": 4,
      "psnr_db": 28.5,
      "recommended_for": ["photos", "scenery", "portraits"]
    },
    {
      "name": "anime",
      "description": "Anime/illustration-optimised model trained with L1 loss",
      "architecture": "srgan",
      "scale_factor": 4,
      "psnr_db": 29.1,
      "recommended_for": ["anime", "illustrations", "cartoons"]
    },
    {
      "name": "waifu2x",
      "description": "Noise-reducing upscaler for anime and photos",
      "architecture": "waifu2x",
      "scale_factor": 2,
      "parameters": {
        "waifu2x_noise_level": "0–3 (0 = none, 3 = aggressive; default 1)",
        "waifu2x_scale": "1 or 2 (default 2)"
      },
      "recommended_for": ["anime", "illustrations", "scanned images"]
    },
    {
      "name": "real-esrgan",
      "description": "Real-ESRGAN ×4 for general photos — trained on synthetic real-world degradations (JPEG artifacts, noise, blur)",
      "architecture": "real-esrgan",
      "scale_factor": 4,
      "psnr_db": 31.8,
      "recommended_for": ["photos", "compressed", "noisy", "real-world"]
    },
    {
      "name": "real-esrgan-anime",
      "description": "Real-ESRGAN ×4 optimised for anime/illustration content — anime-specific degradation pipeline",
      "architecture": "real-esrgan",
      "scale_factor": 4,
      "psnr_db": 32.1,
      "recommended_for": ["anime", "illustrations", "cartoons", "line-art"]
    },
    {
      "name": "real-esrgan-x2",
      "description": "Real-ESRGAN ×2 for general photos — lower memory than the ×4 variant",
      "architecture": "real-esrgan",
      "scale_factor": 2,
      "psnr_db": 32.4,
      "recommended_for": ["photos", "general", "low-memory"]
    }
  ]
}
```

---

## POST /api/v1/upscale

Synchronous upscale. Returns the upscaled image in the response body (base64) or as an S3 presigned URL when S3 is configured.

Blocks until processing is complete. For large images, prefer the async endpoint.

**Request body (JSON)**

```json
{
  "image_data": "<base64-encoded image>",
  "scale_factor": 4,
  "model": "natural",
  "format": "png",
  "quality": 90,
  "auto_detect": true
}
```

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `image_data` | string | Yes | Base64-encoded input image (JPEG, PNG, WebP, BMP). |
| `scale_factor` | integer | No | Upscale factor. Default: model's native factor (4 for `natural`/`anime`, 2 for `waifu2x`). |
| `model` | string | No | `"natural"`, `"anime"`, `"waifu2x"`, or `"waifu2x-noise{0-3}-scale{1,2}"`. Default: auto-detect. |
| `format` | string | No | Output format: `"png"`, `"jpeg"`, `"webp"`. Default: `"png"`. |
| `quality` | integer | No | JPEG/WebP quality 1–100. Default: 90. Ignored for PNG. |
| `auto_detect` | boolean | No | When `true` and `model` is omitted, classifies the image to pick the best model. Default: `true`. |
| `waifu2x_noise_level` | integer | No | Noise level 0–3. Only used when `model` is `"waifu2x"`. |
| `waifu2x_scale` | integer | No | Scale factor 1 or 2. Only used when `model` is `"waifu2x"`. |

**Response 200**

```json
{
  "success": true,
  "image_data": "<base64-encoded output image>",
  "s3_url": null,
  "error": null,
  "metadata": {
    "original_size": [256, 256],
    "upscaled_size": [1024, 1024],
    "processing_time_ms": 1243,
    "format": "png",
    "model_used": "anime"
  }
}
```

When S3 is configured, `image_data` is `null` and `s3_url` contains a presigned download URL (valid 1 hour).

**Response 400** — invalid request body or unsupported image format.

**Response 413** — input image exceeds your tier's resolution limit.

**Response 429** — rate limit exceeded. Check `Retry-After` header.

**Shell example**

```bash
curl -s -X POST http://localhost:8080/api/v1/upscale \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "'$(base64 -i input.jpg)'",
    "model": "anime",
    "format": "png"
  }' | jq -r .image_data | base64 -d > output.png
```

---

## POST /api/v1/upscale/async

Submit an upscale job to the priority queue. Returns immediately with a `job_id`.

Supports optional webhook delivery on completion (Pro and Enterprise only).

**Request body (JSON)**

Same fields as `/api/v1/upscale`, plus:

```json
{
  "image_data": "<base64-encoded image>",
  "model": "natural",
  "format": "png",
  "webhook_config": {
    "url": "https://example.com/hooks/upscale-done",
    "secret": "optional-hmac-secret",
    "max_retries": 3,
    "retry_delay_secs": 5
  }
}
```

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `webhook_config.url` | string | Yes (if webhook) | URL to POST results to on completion. |
| `webhook_config.secret` | string | No | When set, adds `X-SRGAN-Signature: sha256=<hmac>` to the webhook POST. |
| `webhook_config.max_retries` | integer | No | Retry attempts after first failure. Default: 3. |
| `webhook_config.retry_delay_secs` | integer | No | Base delay between retries in seconds; doubles each attempt. Default: 5. |

**Response 202**

```json
{
  "job_id": "01HV3K2XABCDEFGHIJKLMNOPQR",
  "status": "Pending",
  "poll_url": "/api/v1/job/01HV3K2XABCDEFGHIJKLMNOPQR"
}
```

**Shell example**

```bash
# Submit
JOB_ID=$(curl -s -X POST http://localhost:8080/api/v1/upscale/async \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"image_data": "'$(base64 -i photo.jpg)'", "model": "natural"}' \
  | jq -r .job_id)

# Poll until complete
curl -s http://localhost:8080/api/v1/job/$JOB_ID \
  -H "Authorization: Bearer $TOKEN" | jq .
```

---

## GET /api/v1/job/:id

Poll the status of an async job.

**Response 200 — pending/processing**

```json
{
  "id": "01HV3K2XABCDEFGHIJKLMNOPQR",
  "status": "Processing",
  "created_at": 1711234500,
  "updated_at": 1711234510,
  "result_url": null,
  "result_data": null,
  "error": null,
  "model": "natural",
  "input_size": "512x512",
  "output_size": null,
  "webhook_delivery": null
}
```

**Response 200 — completed**

```json
{
  "id": "01HV3K2XABCDEFGHIJKLMNOPQR",
  "status": "Completed",
  "created_at": 1711234500,
  "updated_at": 1711234518,
  "result_url": null,
  "result_data": "<base64-encoded output image>",
  "error": null,
  "model": "natural",
  "input_size": "512x512",
  "output_size": "2048x2048",
  "webhook_delivery": {
    "attempts": 1,
    "last_attempt_at": 1711234519,
    "last_status_code": 200,
    "delivered": true
  }
}
```

**Response 200 — failed**

```json
{
  "id": "01HV3K2XABCDEFGHIJKLMNOPQR",
  "status": {"Failed": "Input image corrupt or unreadable"},
  "created_at": 1711234500,
  "updated_at": 1711234512,
  "result_data": null,
  "error": "Input image corrupt or unreadable"
}
```

**Response 404** — job ID not found or expired (jobs are purged 1 hour after completion).

---

## GET /api/v1/job/:id/webhook

Returns the webhook delivery state for a job without the full result payload.

**Response 200**

```json
{
  "job_id": "01HV3K2XABCDEFGHIJKLMNOPQR",
  "webhook_delivery": {
    "attempts": 2,
    "last_attempt_at": 1711234530,
    "last_status_code": 503,
    "delivered": false
  }
}
```

---

## POST /api/v1/detect

Classify an image as `photo` or `anime` and get the recommended model.

**Request body (JSON)**

```json
{
  "image_data": "<base64-encoded image>"
}
```

**Response 200**

```json
{
  "detected_type": "anime",
  "recommended_model": "anime"
}
```

---

## POST /api/v1/batch

Batch upscale up to 10 images synchronously, or submit larger batches as async jobs.

- **≤10 images**: processes synchronously, returns all results in the response.
- **>10 images**: submits an async batch job and returns a `batch_id`.

**Request body (JSON)**

```json
{
  "images": [
    "<base64-encoded image 1>",
    "<base64-encoded image 2>"
  ],
  "model": "anime",
  "format": "png"
}
```

**Response 200 (sync, ≤10 images)**

```json
{
  "success": true,
  "results": [
    {
      "index": 0,
      "success": true,
      "image_data": "<base64-encoded output>",
      "error": null,
      "processing_time_ms": 1105
    },
    {
      "index": 1,
      "success": false,
      "image_data": null,
      "error": "Unsupported image format",
      "processing_time_ms": 3
    }
  ],
  "total": 2,
  "successful": 1,
  "failed": 1,
  "total_processing_time_ms": 1108
}
```

**Response 202 (async, >10 images)**

```json
{
  "batch_id": "batch_01HV3K2XABCDEFGHIJKLMNOPQR",
  "status": "Processing",
  "total": 42,
  "poll_url": "/api/v1/batch/batch_01HV3K2XABCDEFGHIJKLMNOPQR"
}
```

---

## GET /api/v1/batch/:id

Poll the status of an async batch job.

**Response 200**

```json
{
  "batch_id": "batch_01HV3K2XABCDEFGHIJKLMNOPQR",
  "status": {"Processing": {"completed": 17, "total": 42}},
  "results": [],
  "total": 42,
  "created_at": 1711234500,
  "updated_at": 1711234560
}
```

When `status` is `"Completed"`, `results` contains the full `BatchImageResult` array.

---

## POST /api/v1/batch/start

Start a server-side directory batch job. Requires a path accessible to the server process.

**Request body (JSON)**

```json
{
  "input_dir": "/data/input",
  "output_dir": "/data/output",
  "model": "natural",
  "scale": 4,
  "recursive": true
}
```

**Response 202**

```json
{
  "batch_id": "batch_01HV3K2XABCDEFGHIJKLMNOPQR",
  "status": "Processing",
  "poll_url": "/api/v1/batch/batch_01HV3K2XABCDEFGHIJKLMNOPQR"
}
```

---

## GET /api/v1/batch/:id/checkpoint

Query the checkpoint state of a CLI batch job that was started with `batch --resume`.

**Response 200**

```json
{
  "batch_id": "batch_01HV3K2XABCDEFGHIJKLMNOPQR",
  "processed_files": ["img001.jpg", "img002.jpg"],
  "failed_files": [],
  "total_discovered": 150,
  "last_updated": 1711234580
}
```

---

## POST /api/v1/billing/checkout

Create a Stripe checkout session to upgrade to Pro or Enterprise.

**Request body (JSON)**

```json
{
  "tier": "pro",
  "success_url": "https://example.com/billing/success",
  "cancel_url": "https://example.com/billing/cancel"
}
```

**Response 200**

```json
{
  "checkout_url": "https://checkout.stripe.com/pay/cs_live_...",
  "session_id": "cs_live_..."
}
```

---

## POST /api/v1/billing/webhook

Stripe webhook endpoint. Send Stripe events here. Requires the `Stripe-Signature` header (handled automatically by Stripe).

Not intended for direct use — configure this URL in your Stripe dashboard.

---

## GET /api/v1/billing/status

Get the current subscription status for the authenticated API key.

**Response 200**

```json
{
  "tier": "pro",
  "credits_remaining": 847,
  "credits_reset_at": 1711238400,
  "credits_issued_today": 1000,
  "credits_consumed_today": 153
}
```

---

## POST /api/v1/webhook/test

Send a test webhook delivery to a specified URL. Pro and Enterprise only.

**Request body (JSON)**

```json
{
  "url": "https://example.com/hooks/test",
  "secret": "optional-hmac-secret"
}
```

**Response 200**

```json
{
  "success": true,
  "status_code": 200,
  "latency_ms": 142
}
```

---

## GET /api/v1/admin/stats

Admin statistics. Requires the `X-Admin-Key` header (value must match `ADMIN_KEY` env var).

```bash
curl http://localhost:8080/api/v1/admin/stats \
  -H "X-Admin-Key: $ADMIN_KEY"
```

**Response 200**

```json
{
  "total_keys": 312,
  "keys_by_tier": {
    "free": 280,
    "pro": 29,
    "enterprise": 3
  },
  "usage_today_by_tier": {
    "free": 1240,
    "pro": 8340,
    "enterprise": 25000
  },
  "total_requests_today": 34580
}
```

---

## GET /dashboard

HTML admin dashboard — job monitor and live stats. Requires the `X-Admin-Key` cookie or query parameter `?key=<admin_key>`.

---

## GET /metrics

Prometheus metrics in text format. No authentication required (restrict at the network layer in production).

---

## Error responses

All errors return a JSON body:

```json
{
  "success": false,
  "error": "Human-readable description of the error",
  "code": "RATE_LIMITED"
}
```

| HTTP Status | `code` | Description |
|-------------|--------|-------------|
| 400 | `INVALID_REQUEST` | Malformed JSON, missing required fields, or invalid field values. |
| 401 | `UNAUTHORIZED` | Missing or invalid API key. |
| 403 | `FORBIDDEN` | Valid key but tier does not permit this operation. |
| 404 | `NOT_FOUND` | Resource (job, batch) not found or expired. |
| 413 | `IMAGE_TOO_LARGE` | Input resolution exceeds your tier's limit. |
| 429 | `RATE_LIMITED` | Too many requests. Check `Retry-After` header. |
| 500 | `INTERNAL_ERROR` | Unexpected server error. |
| 503 | `SERVICE_UNAVAILABLE` | Model not loaded or circuit-breaker open. |

---

## Webhook payload

When a job completes and a `webhook_config` was provided, the server POSTs this payload:

```json
{
  "event": "job.completed",
  "job_id": "01HV3K2XABCDEFGHIJKLMNOPQR",
  "status": "Completed",
  "result_data": "<base64-encoded output image>",
  "metadata": {
    "original_size": [512, 512],
    "upscaled_size": [2048, 2048],
    "processing_time_ms": 4821,
    "model_used": "natural"
  },
  "timestamp": 1711234580
}
```

When a `secret` is configured, the request includes:

```
X-SRGAN-Signature: sha256=<hex-encoded HMAC-SHA256 of raw body using secret>
```

Verify the signature in your webhook handler:

```python
import hmac, hashlib

def verify(body: bytes, secret: str, signature_header: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)
```

---

## JavaScript SDK

A minimal SDK is available at `sdk/srgan.js`:

```js
import SrganClient from './sdk/srgan.js'

const client = new SrganClient({
  baseUrl: 'https://api.example.com',
  apiKey: 'srgan_live_xxxxxxxxxxxxxxxxxxxx',
})

// Synchronous upscale
const { imageData, metadata } = await client.upscale({
  imagePath: './photo.jpg',   // Node.js
  model: 'natural',
  format: 'png',
})

// Async upscale with webhook
const job = await client.upscaleAsync({
  imageData: base64String,
  model: 'anime',
  webhookUrl: 'https://example.com/hooks/done',
})
const result = await job.poll()  // polls until complete
```

---

---

## GET /api/v1/job/:id/stream

Stream live progress for an async job via **Server-Sent Events (SSE)**.

**Headers (response)**

```
Content-Type: text/event-stream
Cache-Control: no-cache
```

**Event shapes**

All events are delivered as `data: <json>\n\n` lines.

| `event` field | When emitted | Extra fields |
|---|---|---|
| `queued` | Job accepted, waiting in queue | `position` (int) |
| `started` | Worker picked up the job | — |
| `progress` | Periodic progress update | `percent` (0–100) |
| `done` | Job finished successfully | `output_url` (string) |
| `error` | Job failed or not found | `message` (string) |

**Example stream**

```
data: {"event":"queued","job_id":"abc123","position":2}

data: {"event":"started","job_id":"abc123"}

data: {"event":"progress","job_id":"abc123","percent":10}

data: {"event":"progress","job_id":"abc123","percent":50}

data: {"event":"progress","job_id":"abc123","percent":100}

data: {"event":"done","job_id":"abc123","output_url":"/api/v1/result/abc123"}
```

**Browser (JavaScript)**

```javascript
const es = new EventSource(`/api/v1/job/${jobId}/stream`);
es.onmessage = (e) => {
  const ev = JSON.parse(e.data);
  if (ev.event === 'progress') console.log(`${ev.percent}%`);
  if (ev.event === 'done')     { console.log('Done:', ev.output_url); es.close(); }
  if (ev.event === 'error')    { console.error(ev.message); es.close(); }
};
```

**curl**

```bash
curl -N http://localhost:8080/api/v1/job/abc123/stream
```

**SDK (`upscaleStream`)**

```typescript
import { SrganClient } from "@srgan-rust/sdk";

const client = new SrganClient({ apiKey: "srgan_live_..." });

const result = await client.upscaleStream(
  imageBuffer,
  { model: "anime" },
  (ev) => {
    if (ev.event === "progress") process.stdout.write(`\r${ev.percent}%`);
  },
);
// result is UpscaleResponse with the final upscaled image
```

---

## Rate limits

| Tier | Limit |
|------|-------|
| Free | 5 requests / minute |
| Pro | 60 requests / minute |
| Enterprise | Custom (configured per key) |

When the limit is exceeded the server returns HTTP 429 with a `Retry-After: <seconds>` header.
