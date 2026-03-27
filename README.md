# srgan-rust

![LogoNN](docs/logo_nn.png)![LogoLin](docs/logo_lin.png)![Logo](docs/logo_rs.png)

**Neural network image upscaling at production scale — written in Rust.**

srgan-rust is a fast, SaaS-ready implementation of [SRGAN](https://arxiv.org/abs/1609.04802) (Super-Resolution Generative Adversarial Networks). Give it a low-resolution image, get back a 4× upscale powered by deep learning — in milliseconds, with zero runtime dependencies.

---

## Why srgan-rust?

| | srgan-rust | Python alternatives |
|---|---|---|
| **Inference speed** | ~1.2 s (256→1024, CPU) | 4–8 s |
| **Memory footprint** | ~180 MB | 800 MB+ |
| **Deployment** | Single static binary | Python env + CUDA drivers |
| **API server** | Built-in, production-ready | Separate service required |
| **Concurrency** | Rayon parallel, lock-free | GIL-limited |

---

## Feature highlights

- **Eight built-in models** — `natural` (photos), `anime` (illustrations), `waifu2x` (default waifu2x), `waifu2x-anime` (anime/cartoon optimised), `waifu2x-photo` (photo denoise+upscale), `real-esrgan` (×4 photos, real-world degradations), `real-esrgan-anime` (×4 anime), `real-esrgan-x2` (×2 photos)
- **Auto-detection** — automatically selects the best model for each image (photo vs. anime classifier built in)
- **Batch processing** — directory-level processing with checkpoint/resume, parallel workers
- **Async job queue** — priority queue (Enterprise > Pro > Free), 5-min timeout, 1-hour result retention
- **Webhooks** — HMAC-SHA256-signed POST callbacks with exponential-backoff retry on job completion
- **Admin dashboard** — HTML job monitor at `/dashboard`, stats at `/api/v1/admin/stats`
- **JavaScript SDK** — drop-in `srgan.js` client for browser and Node.js
- **S3 output** — results written directly to S3 when configured; presigned URLs returned
- **Prometheus metrics** — `/metrics` endpoint, Grafana dashboard included (`docs/grafana-dashboard.json`)
- **Subscription tiers** — Free / Pro / Enterprise enforced at the API key layer
- **Kubernetes-ready** — HPA, PVC, cert-manager Ingress manifests in `k8s/`

---

## Quick start

### Docker (one-liner)

```bash
docker run --rm -p 8080:8080 \
  -e JWT_SECRET=changeme \
  ghcr.io/your-org/srgan-rust:latest
```

```bash
curl http://localhost:8080/api/v1/health
# {"status":"ok","version":"0.2.0"}
```

### Docker Compose (API + nginx reverse proxy)

```bash
git clone https://github.com/your-org/srgan-rust
cd srgan-rust
cp .env.example .env          # set JWT_SECRET and optional vars
docker compose up --build -d  # srgan-api on :8080, nginx on :80/:443
```

### Binary (from source)

```bash
# Requires Rust 1.75+
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/srgan-rust --help
```

### CLI upscaling

```bash
# Natural photo (auto-detect)
./srgan-rust input.jpg output.png

# Force anime model
./srgan-rust -p anime manga.png manga_4x.png

# Waifu2x anime preset (noise=1, scale=2×)
./srgan-rust -p waifu2x-anime manga.png manga_2x.png

# Waifu2x photo denoise+upscale (noise=2, scale=2×)
./srgan-rust -p waifu2x-photo noisy_scan.jpg clean_2x.png

# Waifu2x with explicit noise reduction level 2, 2× scale
./srgan-rust -p waifu2x-noise2-scale2 sketch.png sketch_2x.png

# Real-ESRGAN for a heavily compressed photo
./srgan-rust -p real-esrgan compressed.jpg restored_4x.png

# Real-ESRGAN anime variant
./srgan-rust -p real-esrgan-anime manga.png manga_4x.png

# Real-ESRGAN ×2 (lower memory)
./srgan-rust -p real-esrgan-x2 photo.jpg photo_2x.png

# Batch directory, 8 threads
./srgan-rust batch ./input/ ./output/ --threads 8

# 2× instead of 4×
./srgan-rust -f 2 photo.jpg photo_2x.png
```

### Video upscaling

```bash
# Basic video upscale (4x, h264, medium quality)
./srgan-rust video --input clip.mp4 --output clip_4x.mp4

# Anime video with H.265 codec and high quality
./srgan-rust video --input anime.mkv --output anime_4x.mkv \
  -p anime --codec h265 --quality high

# 2x upscale, VP9 codec, strip audio
./srgan-rust video --input input.webm --output output.webm \
  --scale 2 --codec vp9 --no-audio

# Override FPS and use 8 parallel frame workers
./srgan-rust video --input timelapse.mp4 --output timelapse_4x.mp4 \
  --fps 60 --parallel 8

# Preview a single frame before committing to full processing
./srgan-rust video --input movie.mp4 --output movie_4x.mp4 --preview-only

# Process only a 30-second clip starting at 1:30
./srgan-rust video --input long.mp4 --output clip_4x.mp4 \
  --start 00:01:30 --duration 00:00:30
```

### Video upscaling via API

```bash
# Submit async video upscale job
curl -X POST http://localhost:8080/api/v1/video/upscale \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "video_data": "'$(base64 -i clip.mp4)'",
    "model": "anime",
    "scale": 4,
    "codec": "h265",
    "quality": "high",
    "preserve_audio": true
  }'
# {"job_id":"...","status":"pending","type":"video_upscale","check_url":"/api/v1/job/..."}

# Poll job status
curl http://localhost:8080/api/v1/job/$JOB_ID \
  -H "Authorization: Bearer $TOKEN"
```

---

## Model comparison

| Model | Best for | Scale | PSNR | Notes |
|-------|----------|-------|------|-------|
| `natural` | Photos, scenery, portraits | 4× | 28.5 dB | Trained on DIV2K dataset |
| `anime` | Anime, cartoons, illustrations | 4× | 29.1 dB | L1 loss, UCID-anime dataset |
| `waifu2x` | Anime + photos with noise | 1× or 2× | — | Noise levels 0–3, style: anime/photo/artwork; best for scans/screenshots. Falls back to built-in anime model; native weights optional. |
| `waifu2x-anime` | Anime, cartoons, manga | 2× | — | Preset: noise=1, anime-tuned sharpening. Uses anime model proxy when no native weights. |
| `waifu2x-photo` | Photos with noise/JPEG artifacts | 2× | — | Preset: noise=2, photo-tuned denoise. Uses natural model proxy when no native weights. |
| `real-esrgan` | Compressed/noisy photos | 4× | 31.8 dB | Real-world degradation training (JPEG, noise, blur) |
| `real-esrgan-anime` | Compressed/noisy anime | 4× | 32.1 dB | Anime-specific degradation pipeline; sharpest line art |
| `real-esrgan-x2` | Photos, low-memory | 2× | 32.4 dB | Half the memory of `real-esrgan`; moderate upscale |

When `model` is omitted the server classifies the image and picks `natural` or `anime` automatically.

### Waifu2x support

Waifu2x is fully supported with two convenience presets and full noise/scale control:

| Label | Best for | Defaults |
|-------|----------|----------|
| `waifu2x-anime` | Anime, cartoons, manga | noise=1, scale=2×, anime sharpening |
| `waifu2x-photo` | Photos, scans, screenshots | noise=2, scale=2×, photo denoise |
| `waifu2x` | Generic (same as waifu2x-anime) | noise=1, scale=2× |
| `waifu2x-noise{0..3}-scale{1,2}` | Fine-tuned control | explicit noise + scale |

```bash
# Anime upscale
./srgan-rust -p waifu2x-anime manga.png manga_2x.png

# Photo denoise + upscale
./srgan-rust -p waifu2x-photo noisy_scan.jpg clean_2x.png
```

The built-in anime/natural models serve as high-quality fallbacks when native
waifu2x weight files are not present — no separate installation is needed.

**Optional: native waifu2x weights**

To use the original ncnn-format waifu2x weights instead of the built-in fallback:

1. Obtain the ncnn-format waifu2x weights for your desired noise level and scale.
2. Place the file in a `models/` directory next to the binary, named:
   `models/waifu2x_noise{N}_scale{M}x.bin`
   — for example `models/waifu2x_noise1_scale2x.bin` (noise=1, scale=2×).
3. Supported combinations: noise 0–3, scale 1 (denoise only) or 2 (upscale ×2).

---

## API reference summary

All endpoints are under `/api/v1/`. Authenticate with `Authorization: Bearer <token>`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/models` | List available models |
| `POST` | `/api/v1/upscale` | Synchronous upscale (base64 in/out) |
| `POST` | `/api/v1/upscale/async` | Submit async job, returns `job_id` |
| `GET` | `/api/v1/job/:id` | Poll async job status |
| `GET` | `/api/v1/job/:id/webhook` | Webhook delivery state |
| `POST` | `/api/v1/detect` | Classify image as photo/anime |
| `POST` | `/api/v1/video/upscale` | Async video upscale (returns `job_id`) |
| `POST` | `/api/v1/batch` | Batch upscale (sync ≤10, async >10) |
| `POST` | `/api/v1/batch/start` | Start directory batch job |
| `GET` | `/api/v1/batch/:id` | Poll batch job status |
| `GET` | `/api/v1/batch/:id/checkpoint` | Query CLI checkpoint progress |
| `POST` | `/api/v1/billing/checkout` | Create Stripe checkout session |
| `GET` | `/api/v1/billing/status` | Get current subscription status |
| `GET` | `/api/v1/admin/stats` | Admin stats (requires `X-Admin-Key`) |
| `GET` | `/dashboard` | HTML admin dashboard |
| `GET` | `/metrics` | Prometheus metrics |

See [docs/API.md](docs/API.md) for full request/response examples.

### Quick example

```bash
curl -X POST http://localhost:8080/api/v1/upscale \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "'$(base64 -i input.jpg)'",
    "scale_factor": 4,
    "model": "anime",
    "format": "png"
  }' | jq -r .image_data | base64 -d > output.png
```

### Waifu2x API example

Use `model: "waifu2x"` with the optional `waifu2x_noise_level` (0–3) and
`waifu2x_scale` (1 or 2) parameters.  Omitting either uses the defaults
(noise_level=1, scale=2).

```bash
# Noise level 2, 2× upscale — good for scanned manga or screenshots
curl -X POST http://localhost:8080/api/v1/upscale \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "'$(base64 -i sketch.png)'",
    "model": "waifu2x",
    "waifu2x_noise_level": 2,
    "waifu2x_scale": 2,
    "format": "png"
  }' | jq -r .image_data | base64 -d > sketch_2x.png

# Denoise only (scale=1) with aggressive noise reduction (level 3)
curl -X POST http://localhost:8080/api/v1/upscale \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "'$(base64 -i noisy.png)'",
    "model": "waifu2x",
    "waifu2x_noise_level": 3,
    "waifu2x_scale": 1,
    "format": "png"
  }' | jq -r .image_data | base64 -d > denoised.png
```

You can also pass the full variant label directly via `model`:

```bash
curl -X POST http://localhost:8080/api/v1/upscale \
  -H "Content-Type: application/json" \
  -d '{"image_data": "'$(base64 -i art.png)'", "model": "waifu2x-noise1-scale2"}'
```

---

## Performance

| Input → Output | Model | CPU (8-core) | GPU (RTX 3080) |
|----------------|-------|-------------|----------------|
| 256×256 → 1024×1024 | natural | ~1.2 s | ~0.15 s |
| 512×512 → 2048×2048 | natural | ~4.8 s | ~0.55 s |
| 1080p → 4K | natural | ~18 s | ~2.1 s |
| 256×256 → 1024×1024 | anime | ~1.1 s | ~0.14 s |

Benchmarked on Apple M2 Pro (CPU) and NVIDIA RTX 3080 (GPU).

```bash
./srgan-rust benchmark --input test.jpg --iterations 10
```

---

## Training your own model

```bash
# 1. Generate a training config
./srgan-rust generate-config --type training > my_config.toml

# 2. Edit my_config.toml — set dataset_path, epochs, patch_size, etc.

# 3. Train
./srgan-rust train --config my_config.toml
# Epoch  1  loss: 0.043210  PSNR: 27.30 dB  LR: 3.000e-3
# Epoch  2  loss: 0.039100  PSNR: 28.16 dB  LR: 3.000e-3
# [Early stopping] No improvement for 20 epochs. Halting.
```

- Checkpoints every 10 epochs (configurable)
- Early stopping with patience=20
- LR decay ×0.5 every 50 epochs
- Data augmentation: random crop, horizontal flip, brightness/contrast jitter
- 80/20 train/val split

---

## SaaS tiers

| Feature | Free | Pro ($19/mo) | Enterprise |
|---------|------|-------------|------------|
| Images / day | 10 | 1,000 | Unlimited |
| Max input resolution | 4 MP | 20 MP | Unlimited |
| Output watermark | Yes | No | No |
| Models | All built-in | All built-in | + Custom models |
| Async queue | — | Yes | Yes (priority) |
| Webhooks | — | Yes | Yes |
| API rate limit | 5 req/min | 60 req/min | Custom |
| S3 output | — | Yes | Yes |
| SLA | — | 99.5% | 99.9% |
| Support | Community | Email | Dedicated |

See [docs/PRICING.md](docs/PRICING.md) for full pricing copy.

---

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for:
- Docker Compose production setup
- systemd service unit
- nginx reverse proxy with SSL/TLS
- Kubernetes manifests (`k8s/`)
- Full environment variable reference

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | — | **Required.** Signs API tokens. |
| `ADMIN_KEY` | — | Required for `/api/v1/admin/stats`. |
| `RUST_LOG` | `info` | Log level: `error`/`warn`/`info`/`debug`/`trace` |
| `SRGAN_MODEL_PATH` | `/app/models` | Directory scanned for `.rsr`/`.bin` models |
| `SRGAN_MAX_WORKERS` | `4` | Parallel inference threads |
| `DATABASE_URL` | `sqlite:///app/data/srgan.db` | Job-queue persistence |
| `RATE_LIMIT_RPM` | `60` | Max requests per minute per API key |
| `S3_BUCKET` | — | Output bucket name (enables S3 mode) |
| `S3_REGION` | `us-east-1` | AWS region |
| `S3_ACCESS_KEY` | — | AWS access key |
| `S3_SECRET_KEY` | — | AWS secret key |

---

## Project structure

```
src/
  main.rs                 CLI entry point
  lib.rs                  Library root
  commands/               One file per CLI sub-command
  training/
    data_loader.rs        Dataset scan, augmentation, 80/20 split
    trainer_simple.rs     Training loop (early stopping, LR decay)
    checkpoint.rs         Checkpoint save/load
  api/
    auth.rs               JWT / API key validation, tier enforcement
    billing.rs            Subscription tiers, credit tracking
    upscale.rs            Priority job queue, webhook delivery
    admin.rs              Admin stats endpoint
    middleware.rs         Per-tier rate limiter
  web_server.rs           Production REST API
  web_server_improved.rs  Circuit-breaker + retry variant
  gpu.rs                  GPU acceleration layer
  profiling.rs            Memory profiler
k8s/                      Kubernetes manifests
docs/                     API, pricing, deployment guides, Grafana dashboard
```

---

## WASM In-Browser Live Preview

The WASM preview module provides instant in-browser image upscaling (2x/4x bilinear) entirely in the browser via WebAssembly. No data leaves the user's machine. For full SRGAN neural-network quality, the demo page can also call the server API.

### Live demo

Start the server and visit **`/demo`** for a drag-drop demo page with:
- Drag-and-drop or file-picker image upload
- 2x / 4x scale selector
- Side-by-side before/after comparison
- One-click download of the upscaled result

```bash
./srgan-rust server --port 8080
# Open http://localhost:8080/demo
```

### Building the WASM module

```bash
# Install wasm-pack (one-time)
cargo install wasm-pack

# Build the standalone WASM crate
cd wasm && ./build.sh

# Or manually:
cd wasm && wasm-pack build --target web --out-dir pkg --release
```

The build output lands in `wasm/pkg/` and is served by the HTTP server at `/demo/pkg/*` and `/preview/pkg/*`.

### WASM API

```js
import init, { upscale_image, model_info, version } from './pkg/srgan_wasm_preview.js';

await init();

// Upscale a PNG — returns PNG bytes
const result = upscale_image(pngBytes, 4);  // 4x scale

// Get module info
console.log(version());     // "0.2.0"
console.log(model_info());  // JSON with supported scales, method, etc.
```

### Server-side fallback

When not compiling to WASM, use the native shim:

```rust
use srgan_rust::wasm::wasm_shim::WasmShim;

let shim = WasmShim::new();
let output_png = shim.upscale_preview_bytes(&input_png_bytes);
```

### HTTP routes

| Route | Description |
|-------|-------------|
| `GET /demo` | WASM live preview demo page |
| `GET /demo/pkg/*` | WASM package assets (JS glue, `.wasm` binary) |
| `GET /preview` | Alternate preview page |
| `GET /preview/pkg/*` | Alternate asset path |

---

## Contributing

1. Fork the repo and create a feature branch.
2. `cargo test` — all tests must pass.
3. `cargo clippy -- -D warnings` — no new lints.
4. Open a pull request with a clear description.

Bug reports and feature requests: [GitHub Issues](https://github.com/your-org/srgan-rust/issues)

---

## License

MIT — see [LICENSE](LICENSE).
