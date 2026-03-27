#!/usr/bin/env bash
set -euo pipefail

# Build the SRGAN WASM preview crate and copy output to wasm/pkg/
# Requires: cargo install wasm-pack

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building WASM preview crate..."
wasm-pack build --target web --out-dir pkg --release

echo ""
echo "Build complete. Output in wasm/pkg/"
echo "Files:"
ls -lh pkg/*.{js,wasm,d.ts} 2>/dev/null || ls -lh pkg/
