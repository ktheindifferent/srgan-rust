# Stage 1: builder
FROM rust:1.75-slim AS builder
WORKDIR /app

# Install system dependencies for image processing and SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Cache dependencies by building a dummy project first
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

# Build the real project
COPY src ./src
COPY res ./res
RUN RUSTFLAGS="-C target-cpu=native" cargo build --release

# Stage 2: runtime
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary and models
COPY --from=builder /app/target/release/srgan-rust /usr/local/bin/
COPY --from=builder /app/res/ /app/models/

# Create data directory and non-root user
RUN useradd -m -u 1000 srgan \
    && mkdir -p /app/data \
    && chown -R srgan:srgan /app

WORKDIR /app
USER srgan

EXPOSE 8080

CMD ["srgan-rust", "server", "--host", "0.0.0.0", "--port", "8080"]
