#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8080/api/health}"
HEALTH_RETRIES=10
HEALTH_DELAY=3

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

check_deps() {
    command -v docker   >/dev/null 2>&1 || die "docker not found"
    command -v curl     >/dev/null 2>&1 || die "curl not found"
}

# ── Deploy ────────────────────────────────────────────────────────────────────
main() {
    check_deps

    log "Pulling latest code..."
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git pull --ff-only
    else
        log "Not a git repo — skipping pull"
    fi

    log "Building and starting containers..."
    docker compose -f "$COMPOSE_FILE" up --build -d srgan-api nginx

    # Run DB migrations if migrate binary is present
    if docker compose -f "$COMPOSE_FILE" run --rm srgan-api srgan-rust --version >/dev/null 2>&1; then
        log "Running DB migrations (if any)..."
        # Placeholder: swap this line for your migration command
        # docker compose -f "$COMPOSE_FILE" run --rm srgan-api srgan-rust migrate
        log "No migrations defined — skipping"
    fi

    # Health check
    log "Waiting for API to become healthy..."
    for i in $(seq 1 "$HEALTH_RETRIES"); do
        if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
            log "Health check passed after $i attempt(s)."
            log "Deploy complete."
            docker compose -f "$COMPOSE_FILE" ps
            exit 0
        fi
        log "Attempt $i/$HEALTH_RETRIES — retrying in ${HEALTH_DELAY}s..."
        sleep "$HEALTH_DELAY"
    done

    die "Health check failed after $HEALTH_RETRIES attempts. Check logs: docker compose logs srgan-api"
}

main "$@"
