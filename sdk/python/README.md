# srgan-sdk · Python

Python SDK for the [SRGAN-Rust](https://github.com/srgan-rust/srgan-rust) image upscaling API.

## Requirements

- Python ≥ 3.8
- [`httpx`](https://www.python-httpx.org/) ≥ 0.24
- [`typing_extensions`](https://pypi.org/project/typing-extensions/) ≥ 4.0

## Installation

```bash
pip install srgan-sdk
```

Or from the repository root:

```bash
pip install ./sdk/python
```

## Quickstart

```python
from srgan_sdk import SrganClient

client = SrganClient("http://localhost:8080", api_key="sk-...")

# Synchronous upscale — accepts a file path, pathlib.Path, or raw bytes
result = client.upscale("photo.jpg", model="esrgan", scale=4)
result.save("photo_4x.png")
print(result)  # UpscaleResult(content_type='image/png', size=2048x2048, inference_ms=420)
```

## API Reference

### `SrganClient(base_url, api_key=None, *, timeout=120, max_retries=3)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:8080"` | Server base URL |
| `api_key` | `str \| None` | `None` | Sent as `X-API-Key` header |
| `timeout` | `float` | `120.0` | Per-request timeout (seconds) |
| `max_retries` | `int` | `3` | Retries on 429 / transient errors |

The client is a context manager — use `with SrganClient(...) as client:` for automatic connection cleanup.

---

### `upscale(image, *, model="esrgan", scale=4, webhook_url=None) → UpscaleResult`

Upscale an image **synchronously**.  For large images the server may queue
the job; this method polls automatically until the result is ready.

```python
result = client.upscale("photo.jpg", model="esrgan-anime", scale=4)
result.save("photo_4x.png")
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | `str \| Path \| bytes` | — | File path or raw image bytes |
| `model` | `str` | `"esrgan"` | Model identifier |
| `scale` | `int` | `4` | Upscaling factor |
| `webhook_url` | `str \| None` | `None` | URL to POST the result to |

---

### `upscale_async(image, *, model="esrgan", scale=4, webhook_url=None) → str`

Submit an upscale job **asynchronously**.  Returns a `job_id` string.

```python
job_id = client.upscale_async("photo.jpg")
print("Submitted:", job_id)
```

---

### `status(job_id) → JobStatus`

Poll the status of an async job.

```python
import time

job_id = client.upscale_async("photo.jpg")
while True:
    js = client.status(job_id)
    print(js.status)
    if js.is_terminal:
        break
    time.sleep(2)

if js.status == "completed":
    js.result.save("photo_4x.png")
```

**`JobStatus` attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `job_id` | `str` | Unique identifier |
| `status` | `str` | `"pending"` / `"processing"` / `"completed"` / `"failed"` / `"timed_out"` |
| `created_at` | `str` | ISO-8601 timestamp |
| `completed_at` | `str \| None` | ISO-8601 timestamp when finished |
| `result` | `UpscaleResult \| None` | Populated on completion |
| `error` | `str \| None` | Populated on failure |
| `is_terminal` | `bool` | `True` when no further state changes will occur |

---

### `health() → dict`

Return server health information.

```python
info = client.health()
print(info["status"], info["version"])
```

---

### `models() → list[ModelInfo]`

List models available on the server.

```python
for m in client.models():
    print(m.label, m.scale_factor, m.description)
```

**`ModelInfo` attributes**: `label`, `description`, `scale_factor`.

---

### `UpscaleResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `image` | `str` | Base64-encoded image bytes |
| `content_type` | `str` | e.g. `"image/png"` |
| `width` | `int` | Original width in pixels |
| `height` | `int` | Original height in pixels |
| `inference_ms` | `int` | Server inference time in ms |
| `.save(path)` | method | Decode and write the image to disk |

---

## Exceptions

| Exception | HTTP status | Description |
|-----------|-------------|-------------|
| `SrganError` | any | Base class |
| `SrganAuthError` | 401 / 403 | Bad or missing API key |
| `SrganRateLimitError` | 429 | Rate limited; check `.retry_after` |
| `SrganValidationError` | 422 | Invalid request payload |
| `SrganNotFoundError` | 404 | Resource not found |
| `SrganServerError` | 5xx | Internal server error |

```python
from srgan_sdk import SrganClient, SrganRateLimitError
import time

client = SrganClient("http://localhost:8080")
try:
    result = client.upscale("photo.jpg")
except SrganRateLimitError as e:
    print(f"Rate limited — retry after {e.retry_after}s")
    time.sleep(e.retry_after)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SRGAN_BASE_URL` | Default server URL |
| `SRGAN_API_KEY` | Default API key |

## Running the Examples

```bash
# Show server health and available models
python examples/basic.py

# Upscale a file
python examples/basic.py photo.jpg photo_4x.png
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
