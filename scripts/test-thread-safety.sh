#!/bin/bash

# Script to run comprehensive thread safety tests locally
# Usage: ./scripts/test-thread-safety.sh [miri|loom|stress|all]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_miri_tests() {
    echo_info "Running Miri tests for undefined behavior detection..."
    
    if ! command -v cargo +nightly miri &> /dev/null; then
        echo_warn "Miri not installed. Installing..."
        rustup +nightly component add miri
    fi
    
    echo_info "Testing thread_safe_network with Miri..."
    cargo +nightly miri test thread_safe_network::tests:: || echo_warn "Some Miri tests failed"
    
    echo_info "Testing parallel module with Miri..."
    cargo +nightly miri test parallel::tests:: || echo_warn "Some Miri tests failed"
    
    echo_info "Testing gpu module with Miri..."
    cargo +nightly miri test gpu::tests:: || echo_warn "Some Miri tests failed"
}

run_loom_tests() {
    echo_info "Running Loom tests for concurrency verification..."
    
    # Check if loom tests exist
    if [ -f "tests/thread_safety_loom.rs" ]; then
        LOOM_MAX_THREADS=4 LOOM_CHECKPOINT_INTERVAL=100 \
            cargo test --test thread_safety_loom --features loom || echo_warn "Some Loom tests failed"
    else
        echo_warn "Loom tests not found at tests/thread_safety_loom.rs"
    fi
}

run_stress_tests() {
    echo_info "Running stress tests for high-load scenarios..."
    
    if [ -f "tests/thread_safety_stress.rs" ]; then
        echo_info "Running high contention test..."
        cargo test --test thread_safety_stress stress_test_high_contention_thread_safe_network --release
        
        echo_info "Running parallel network cloning test..."
        cargo test --test thread_safety_stress stress_test_parallel_network_rapid_cloning --release
        
        echo_info "Running GPU memory allocation test..."
        cargo test --test thread_safety_stress stress_test_gpu_memory_allocation --release
        
        echo_info "Running buffer pool race condition test..."
        cargo test --test thread_safety_stress stress_test_buffer_pool_race_conditions --release
        
        echo_info "Running mixed operations test..."
        cargo test --test thread_safety_stress stress_test_mixed_operations --release
    else
        echo_warn "Stress tests not found at tests/thread_safety_stress.rs"
    fi
}

run_sanitizer_tests() {
    echo_info "Running ThreadSanitizer tests (requires nightly Rust)..."
    
    # Check if we're on Linux (ThreadSanitizer is Linux-only)
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        echo_warn "ThreadSanitizer is only available on Linux. Skipping..."
        return
    fi
    
    export RUSTFLAGS="-Z sanitizer=thread"
    export TSAN_OPTIONS="halt_on_error=1:history_size=7"
    
    cargo +nightly test --tests --target x86_64-unknown-linux-gnu -- --test-threads=1 || echo_warn "ThreadSanitizer found issues"
    
    unset RUSTFLAGS
    unset TSAN_OPTIONS
}

run_unsafe_audit() {
    echo_info "Auditing unsafe code..."
    
    # Check for SAFETY comments in unsafe impls
    for file in src/thread_safe_network.rs src/parallel.rs src/gpu.rs; do
        if [ -f "$file" ]; then
            echo_info "Checking $file for unsafe impl documentation..."
            if grep -q "unsafe impl" "$file"; then
                if grep -B10 "unsafe impl" "$file" | grep -q "SAFETY:"; then
                    echo_info "✓ SAFETY documentation found in $file"
                else
                    echo_error "Missing SAFETY documentation in $file"
                fi
            fi
        fi
    done
    
    # Count unsafe usage
    if command -v cargo-geiger &> /dev/null; then
        echo_info "Running cargo-geiger to count unsafe code..."
        cargo geiger --print-unused
    else
        echo_warn "cargo-geiger not installed. Install with: cargo install cargo-geiger"
    fi
}

run_all_tests() {
    echo_info "Running all thread safety tests..."
    run_unsafe_audit
    run_stress_tests
    run_miri_tests
    run_loom_tests
    run_sanitizer_tests
}

# Main script logic
case "${1:-all}" in
    miri)
        run_miri_tests
        ;;
    loom)
        run_loom_tests
        ;;
    stress)
        run_stress_tests
        ;;
    sanitizer)
        run_sanitizer_tests
        ;;
    audit)
        run_unsafe_audit
        ;;
    all)
        run_all_tests
        ;;
    *)
        echo "Usage: $0 [miri|loom|stress|sanitizer|audit|all]"
        echo ""
        echo "Options:"
        echo "  miri      - Run Miri tests for undefined behavior detection"
        echo "  loom      - Run Loom tests for concurrency verification"
        echo "  stress    - Run stress tests for high-load scenarios"
        echo "  sanitizer - Run ThreadSanitizer tests (Linux only)"
        echo "  audit     - Audit unsafe code and documentation"
        echo "  all       - Run all tests (default)"
        exit 1
        ;;
esac

echo_info "Thread safety testing completed!"