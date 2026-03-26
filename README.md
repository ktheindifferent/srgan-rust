# srgan-rust

![LogoNN](docs/logo_nn.png)![LogoLin](docs/logo_lin.png)![Logo](docs/logo_rs.png)

**Fast, production-ready SRGAN image upscaling in Rust.**
Drop-in SaaS API, Docker-first deployment, custom model training — all in a single binary.

---

## What is it?

srgan-rust is a Rust implementation of [SRGAN](https://arxiv.org/abs/1609.04802) (Super-Resolution Generative Adversarial Networks).
Give it a low-resolution image, get back a 4× upscale powered by deep learning — in milliseconds.

- **Fast**: native Rust, `rayon` parallel processing, optional GPU acceleration
- **SaaS-ready**: JWT-authenticated REST API, rate limiting, job queue, Prometheus metrics
- **Trainable**: bring your own dataset, get a domain-specific model (anime, medical, satellite…)
- **Portable**: single static binary or Docker image, zero runtime dependencies

---

## Quick start (Docker)

```bash
# One-liner: pull, build, and start the API + nginx reverse proxy
curl -fsSL https://raw.githubusercontent.com/your-org/srgan-rust/master/deploy.sh | bash
```

Or manually:

```bash
git clone https://github.com/your-org/srgan-rust
cd srgan-rust
cp .env.example .env          # edit JWT_SECRET and other settings
docker compose up --build -d  # starts srgan-api on :8080 and nginx on :80/:443
```

Health check:

```bash
curl http://localhost:8080/api/health
# {"status":"ok","version":"0.2.0"}
```

---

## API reference

All endpoints are under `/api/v1/`. Authenticate with an `Authorization: Bearer <token>` header.

### Upscale an image

```bash
curl -X POST http://localhost:8080/api/v1/upscale \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "'$(base64 -i input.jpg)'",
    "scale_factor": 4,
    "model": "default",
    "format": "png"
  }' | jq -r .image_data | base64 -d > output.png
```

### List available models

```bash
curl http://localhost:8080/api/v1/models \
  -H "Authorization: Bearer $TOKEN"
```

```json
[
  {
    "name": "default",
    "description": "General purpose SRGAN, trained on DIV2K",
    "scale_factor": 4,
    "version": "1.0.0",
    "psnr": 28.5,
    "recommended_for": ["photos", "general"]
  }
]
```

### Async job queue

```bash
# Submit
curl -X POST http://localhost:8080/api/v1/upscale/async \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@large_photo.jpg" | jq .job_id

# Poll
curl http://localhost:8080/api/v1/jobs/<job_id> \
  -H "Authorization: Bearer $TOKEN"
```

---

## Build from source

```bash
# Requires Rust 1.75+
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/srgan-rust --help
```

**CLI upscaling:**

```bash
# Single image (natural photos)
./srgan-rust input.jpg output.png

# Anime-optimised model
./srgan-rust -p anime anime.png anime_4x.png

# Batch directory
./srgan-rust batch ./input_dir ./output_dir --threads 8

# 2× instead of 4×
./srgan-rust -f 2 photo.jpg photo_2x.png
```

---

## Training your own model

```bash
# 1. Prepare dataset — high-resolution images in one directory
./srgan-rust generate-config --type training > my_config.toml

# 2. Edit my_config.toml (dataset_path, epochs, patch_size, …)

# 3. Train
./srgan-rust train --config my_config.toml

# Training logs:
# Epoch    1  loss: 0.043210  PSNR:  27.30 dB  change: 0.000321  LR: 3.000e-3
# Epoch    2  loss: 0.039100  PSNR:  28.16 dB  change: 0.000298  LR: 3.000e-3
# ...
# [Early stopping] No improvement for 20 epochs. Halting.
```

Training features:
- **Checkpoints** every 10 epochs (configurable)
- **Early stopping** with patience=20
- **LR decay** ×0.5 every 50 epochs
- **Data augmentation**: random crop, horizontal flip, brightness/contrast jitter
- **80/20 train/val split** with progress bar

Trained models go in `models/` alongside a sidecar `.json` metadata file.

---

## Performance

| Resolution | Model | Time (CPU, 8-core) | Time (GPU) |
|------------|-------|--------------------|------------|
| 256×256 → 1024×1024 | default | ~1.2 s | ~0.15 s |
| 512×512 → 2048×2048 | default | ~4.8 s | ~0.55 s |
| 1080p → 4K | default | ~18 s | ~2.1 s |
| 256×256 → 1024×1024 | anime | ~1.1 s | ~0.14 s |

Benchmarked on Apple M2 Pro (CPU) and NVIDIA RTX 3080 (GPU).
Run your own: `./srgan-rust benchmark --input test.jpg --iterations 10`

---

## SaaS tiers

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Upscales / month | 100 | 10,000 | Unlimited |
| Max resolution | 512×512 input | 2048×2048 input | Unlimited |
| Models | Default only | All built-in | Custom models |
| API rate limit | 5 req/min | 60 req/min | Custom |
| Async queue | — | ✓ | ✓ |
| SLA | — | 99.5% | 99.9% |
| Support | Community | Email | Dedicated |

---

## Docker services

```yaml
# docker-compose.yml — three services included
srgan-api     # REST API on :8080
nginx         # Reverse proxy on :80/:443, rate-limited, gzip, 100 MB upload
srgan-batch   # Batch processor (--profile tools)
srgan-train   # Training runner  (--profile tools)
```

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | — | **Required.** Signs API tokens. |
| `RUST_LOG` | `info` | Log level (`error`/`warn`/`info`/`debug`/`trace`) |
| `SRGAN_MODEL_PATH` | `/app/models` | Directory scanned for `.rsr`/`.bin` models |
| `SRGAN_MAX_WORKERS` | `4` | Parallel inference threads |
| `DATABASE_URL` | `sqlite:///app/data/srgan.db` | Job-queue database |
| `RATE_LIMIT_RPM` | `60` | Max requests per minute per API key |

---

## Project layout

```
src/
  main.rs            CLI entry point
  lib.rs             Library root
  commands/          One file per CLI command
  training/
    data_loader.rs   Dataset scan, augmentation, 80/20 split, progress bar
    trainer_simple.rs  Training loop (early stopping, LR decay, PSNR logging)
    checkpoint.rs    Checkpoint save/load
  model_manager.rs   Scan models/ dir, load sidecar JSON, expose via API
  web_server_improved.rs  Production REST API with circuit-breaker & retry
  gpu.rs             GPU acceleration layer
  profiling.rs       Memory profiler
models/
  default.json       Sidecar metadata for the default model
```

---

## Contributing

1. Fork the repo and create a feature branch.
2. `cargo test` — all tests must pass.
3. `cargo clippy -- -D warnings` — no new lints.
4. Open a pull request with a clear description.

Bug reports and feature requests via [GitHub Issues](https://github.com/your-org/srgan-rust/issues).

---

## License

MIT — see [LICENSE](LICENSE).
