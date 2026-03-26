# Deployment Guide

This guide covers production deployment of srgan-rust: Docker Compose, bare-metal systemd, nginx TLS termination, and Kubernetes.

---

## Requirements

- Linux x86_64 or ARM64 (macOS supported for development)
- 512 MB RAM minimum; 2 GB recommended for concurrent inference
- Docker 24+ (for container deployments)
- Rust 1.75+ (if building from source)

---

## Docker Compose (recommended)

The included `docker-compose.yml` starts three services:

| Service | Port | Description |
|---------|------|-------------|
| `srgan-api` | 8080 (internal) | REST API server |
| `nginx` | 80, 443 | Reverse proxy, rate limiting, TLS |
| `srgan-batch` | — | Batch processor (opt-in via `--profile tools`) |
| `srgan-train` | — | Training runner (opt-in via `--profile tools`) |

### Setup

```bash
git clone https://github.com/your-org/srgan-rust
cd srgan-rust
cp .env.example .env
```

Edit `.env`:

```bash
# Required
JWT_SECRET=your-very-long-random-secret-here
ADMIN_KEY=your-admin-dashboard-key

# Optional — S3 output
S3_BUCKET=my-upscaled-images
S3_REGION=us-east-1
S3_ACCESS_KEY=AKIA...
S3_SECRET_KEY=...

# Optional — tuning
SRGAN_MAX_WORKERS=4
RATE_LIMIT_RPM=60
RUST_LOG=info
```

### Start

```bash
docker compose up --build -d

# Verify
curl http://localhost:8080/api/v1/health
# {"status":"ok","version":"0.2.0","model_loaded":true}
```

### Include batch/training tools

```bash
docker compose --profile tools up -d
```

### Update

```bash
docker compose pull
docker compose up -d --no-deps srgan-api
```

---

## Building from source

```bash
# Clone
git clone https://github.com/your-org/srgan-rust
cd srgan-rust

# Build optimised binary
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Binary is at:
./target/release/srgan-rust
```

### Start the API server

```bash
JWT_SECRET=changeme ./target/release/srgan-rust server \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4
```

---

## systemd service

Create `/etc/systemd/system/srgan-rust.service`:

```ini
[Unit]
Description=srgan-rust API server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=srgan
Group=srgan
WorkingDirectory=/opt/srgan-rust
ExecStart=/opt/srgan-rust/srgan-rust server --host 127.0.0.1 --port 8080 --workers 4

# Environment
Environment=RUST_LOG=info
Environment=SRGAN_MAX_WORKERS=4
EnvironmentFile=/etc/srgan-rust/env

# Restart policy
Restart=on-failure
RestartSec=5s
StartLimitIntervalSec=60
StartLimitBurst=3

# Security hardening
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/srgan-rust/data /opt/srgan-rust/models

# Resource limits
LimitNOFILE=65536
MemoryMax=2G

[Install]
WantedBy=multi-user.target
```

Create `/etc/srgan-rust/env` (mode 0600):

```bash
JWT_SECRET=your-very-long-random-secret-here
ADMIN_KEY=your-admin-key
DATABASE_URL=sqlite:///opt/srgan-rust/data/srgan.db
```

Enable and start:

```bash
sudo useradd -r -s /bin/false srgan
sudo mkdir -p /opt/srgan-rust/data /opt/srgan-rust/models /etc/srgan-rust
sudo cp target/release/srgan-rust /opt/srgan-rust/
sudo chown -R srgan:srgan /opt/srgan-rust /etc/srgan-rust
sudo chmod 0600 /etc/srgan-rust/env

sudo systemctl daemon-reload
sudo systemctl enable srgan-rust
sudo systemctl start srgan-rust
sudo systemctl status srgan-rust
```

---

## nginx reverse proxy

### HTTP-only (development)

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass         http://127.0.0.1:8080;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        client_max_body_size 100m;
    }
}
```

### HTTPS with Let's Encrypt (production)

Install certbot, obtain a certificate, then:

```nginx
# /etc/nginx/sites-available/srgan-rust

# Redirect HTTP → HTTPS
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # TLS
    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 1d;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options    nosniff always;
    add_header X-Frame-Options           SAMEORIGIN always;
    add_header X-XSS-Protection          "1; mode=block" always;

    # Upload limit for image payloads
    client_max_body_size 100m;

    # Compression
    gzip on;
    gzip_types application/json;
    gzip_min_length 1024;

    # Rate limiting (supplement server-side limiter)
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass         http://127.0.0.1:8080;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_read_timeout 120s;
        proxy_send_timeout 30s;
        proxy_connect_timeout 5s;
    }

    # Longer timeout for async endpoints
    location ~ ^/api/v1/(upscale/async|batch) {
        proxy_pass         http://127.0.0.1:8080;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        client_max_body_size 100m;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/srgan-rust /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

Renew certificates automatically (already set up by certbot, but verify):

```bash
sudo certbot renew --dry-run
```

---

## Environment variables reference

| Variable | Default | Required | Description |
|----------|---------|:--------:|-------------|
| `JWT_SECRET` | — | Yes | Signing key for API tokens. Use 32+ random bytes. |
| `ADMIN_KEY` | — | Yes | Required to access `/api/v1/admin/stats` and `/dashboard`. |
| `RUST_LOG` | `info` | No | Log verbosity: `error`, `warn`, `info`, `debug`, `trace`. |
| `SRGAN_MODEL_PATH` | `/app/models` | No | Directory scanned for `.rsr` and `.bin` model files at startup. |
| `SRGAN_MAX_WORKERS` | `4` | No | Number of parallel inference threads. Set to CPU core count. |
| `DATABASE_URL` | `sqlite:///app/data/srgan.db` | No | SQLite path for job-queue persistence. Use an absolute path. |
| `RATE_LIMIT_RPM` | `60` | No | Global default max requests per minute per API key. |
| `S3_BUCKET` | — | No | S3 bucket name. When set, results are uploaded and presigned URLs returned. |
| `S3_REGION` | `us-east-1` | No | AWS region for S3 bucket. |
| `S3_ACCESS_KEY` | — | No | AWS access key ID. |
| `S3_SECRET_KEY` | — | No | AWS secret access key. |
| `PORT` | `8080` | No | Port for the API server to bind on. |
| `HOST` | `127.0.0.1` | No | Interface to bind on. Use `0.0.0.0` in containers. |

---

## Kubernetes

All manifests are in `k8s/`. Apply in order:

```bash
# 1. Namespace
kubectl create namespace srgan

# 2. Secrets (never commit real values)
kubectl create secret generic srgan-secrets \
  --from-literal=JWT_SECRET=<jwt-secret> \
  --from-literal=ADMIN_KEY=<admin-key> \
  --from-literal=STRIPE_KEY=<stripe-key> \
  --from-literal=S3_ACCESS_KEY=<s3-access-key> \
  --from-literal=S3_SECRET_KEY=<s3-secret-key> \
  -n srgan

# 3. Manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Verify
kubectl rollout status deployment/srgan-api -n srgan
```

### Manifest overview

| File | Description |
|------|-------------|
| `deployment.yaml` | 3-replica Deployment with liveness/readiness probes, resource limits, pod anti-affinity |
| `service.yaml` | ClusterIP on port 8080 (+ port 9090 for Prometheus) |
| `ingress.yaml` | nginx Ingress with TLS via cert-manager and rate-limiting annotations |
| `hpa.yaml` | HorizontalPodAutoscaler — 2–10 pods, scales at CPU > 70% |
| `pvc.yaml` | 5 Gi ReadWriteMany PVC for the `/models` directory |
| `configmap.yaml` | Non-secret runtime config (log level, worker count, upload limits) |
| `secret.yaml` | Template for secrets — fill in and apply, never commit real values |

### Resource sizing

| | Request | Limit |
|---|---------|-------|
| CPU | 0.5 core | 2 cores |
| Memory | 512 Mi | 2 Gi |

For GPU nodes, add to `deployment.yaml`:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

---

## Monitoring

### Prometheus

Metrics are exposed at `GET /metrics` in the standard Prometheus text format.

Key metrics:

| Metric | Description |
|--------|-------------|
| `srgan_requests_total` | Total API requests by endpoint and status |
| `srgan_inference_duration_seconds` | Inference latency histogram |
| `srgan_queue_depth` | Current async job queue depth by tier |
| `srgan_images_processed_total` | Total images successfully upscaled |

### Grafana

Import `docs/grafana-dashboard.json` into your Grafana instance.

### Alert rules

Alert rules (Prometheus Alertmanager format) are in `docs/alert-rules.yaml`.

---

## Health check

```bash
curl http://localhost:8080/api/v1/health
```

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

Returns HTTP 200 when healthy, HTTP 503 when the model is not loaded or a circuit-breaker is open.

---

## Security checklist

- [ ] Set a strong random `JWT_SECRET` (32+ bytes)
- [ ] Set a strong `ADMIN_KEY` and restrict `/dashboard` access at the nginx layer if possible
- [ ] Enable HTTPS (TLS 1.2+ only, HSTS header)
- [ ] Set `client_max_body_size` in nginx to limit upload size
- [ ] Run the container/service as a non-root user
- [ ] Restrict `SRGAN_MODEL_PATH` to a read-only mount containing only trusted model files
- [ ] Rotate API keys if compromised (delete and reissue via admin endpoint)
- [ ] Review CORS settings if the API is accessed from a browser
