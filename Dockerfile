# Multi-stage build for optimized image size
FROM rust:1.75 as builder

# Set working directory
WORKDIR /usr/src/srgan-rust

# Copy Cargo files first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Create dummy main.rs to cache dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy actual source code
COPY src ./src
COPY res ./res

# Build with native CPU optimizations
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3"
RUN cargo build --release

# Runtime stage - use smaller base image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /usr/src/srgan-rust/target/release/srgan-rust /usr/local/bin/srgan-rust

# Copy pre-trained models
COPY --from=builder /usr/src/srgan-rust/res /opt/srgan-rust/res

# Set working directory for user files
WORKDIR /workspace

# Create non-root user
RUN useradd -m -u 1000 srgan && \
    chown -R srgan:srgan /workspace

USER srgan

# Set entrypoint
ENTRYPOINT ["srgan-rust"]

# Default command shows help
CMD ["--help"]