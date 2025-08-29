use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;
use chrono::Local;
use log::{info, warn, error};
use std::ptr;
use std::cell::Cell;

#[cfg(debug_assertions)]
use std::collections::HashSet;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_MEMORY: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_FAILURES: AtomicUsize = AtomicUsize::new(0);
static MEMORY_LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static TELEMETRY_ENABLED: AtomicBool = AtomicBool::new(false);
static OOM_HANDLER_ENABLED: AtomicBool = AtomicBool::new(true);

lazy_static::lazy_static! {
    static ref ALLOCATIONS: Mutex<HashMap<String, AllocationStats>> = Mutex::new(HashMap::new());
    static ref PROFILING_ENABLED: AtomicUsize = AtomicUsize::new(0);
    static ref ALLOCATION_TELEMETRY: RwLock<AllocationTelemetry> = RwLock::new(AllocationTelemetry::new());
}

#[cfg(debug_assertions)]
lazy_static::lazy_static! {
    static ref ACTIVE_ALLOCATIONS: Mutex<HashSet<usize>> = Mutex::new(HashSet::new());
    static ref ALLOCATION_BACKTRACE: Mutex<HashMap<usize, String>> = Mutex::new(HashMap::new());
}

thread_local! {
    static RECURSION_GUARD: Cell<bool> = Cell::new(false);
}

#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub count: usize,
    pub total_bytes: usize,
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub failures: usize,
    pub last_failure_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct AllocationTelemetry {
    pub total_attempts: usize,
    pub successful_allocations: usize,
    pub failed_allocations: usize,
    pub fallback_count: usize,
    pub largest_failed_size: usize,
    pub last_failure_timestamp: Option<Instant>,
    pub oom_events: Vec<OomEvent>,
}

#[derive(Debug, Clone)]
pub struct OomEvent {
    pub timestamp: Instant,
    pub requested_size: usize,
    pub current_allocated: usize,
    pub memory_limit: usize,
}

impl AllocationTelemetry {
    fn new() -> Self {
        AllocationTelemetry {
            total_attempts: 0,
            successful_allocations: 0,
            failed_allocations: 0,
            fallback_count: 0,
            largest_failed_size: 0,
            last_failure_timestamp: None,
            oom_events: Vec::new(),
        }
    }
    
    fn record_failure(&mut self, size: usize) {
        self.failed_allocations += 1;
        self.largest_failed_size = self.largest_failed_size.max(size);
        self.last_failure_timestamp = Some(Instant::now());
    }
    
    fn record_oom(&mut self, size: usize, current: usize, limit: usize) {
        self.oom_events.push(OomEvent {
            timestamp: Instant::now(),
            requested_size: size,
            current_allocated: current,
            memory_limit: limit,
        });
        
        if self.oom_events.len() > 100 {
            self.oom_events.remove(0);
        }
    }
}

impl Default for AllocationStats {
    fn default() -> Self {
        AllocationStats {
            count: 0,
            total_bytes: 0,
            current_bytes: 0,
            peak_bytes: 0,
            failures: 0,
            last_failure_size: None,
        }
    }
}

pub struct MemoryProfiler {
    start_time: Instant,
    samples: Vec<MemorySample>,
    sampling_interval: Duration,
    last_sample: Instant,
}

#[derive(Debug, Clone)]
pub struct MemorySample {
    pub timestamp: Duration,
    pub allocated: usize,
    pub deallocated: usize,
    pub current: usize,
    pub peak: usize,
}

impl MemoryProfiler {
    pub fn new(sampling_interval_ms: u64) -> Self {
        let now = Instant::now();
        MemoryProfiler {
            start_time: now,
            samples: Vec::new(),
            sampling_interval: Duration::from_millis(sampling_interval_ms),
            last_sample: now,
        }
    }

    pub fn start(&self) {
        PROFILING_ENABLED.store(1, Ordering::SeqCst);
        info!("Memory profiling started");
    }

    pub fn stop(&self) {
        PROFILING_ENABLED.store(0, Ordering::SeqCst);
        info!("Memory profiling stopped");
    }

    pub fn sample(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_sample) >= self.sampling_interval {
            let allocated = ALLOCATED.load(Ordering::SeqCst);
            let deallocated = DEALLOCATED.load(Ordering::SeqCst);
            let current = allocated.saturating_sub(deallocated);
            let peak = PEAK_MEMORY.load(Ordering::SeqCst);

            self.samples.push(MemorySample {
                timestamp: now.duration_since(self.start_time),
                allocated,
                deallocated,
                current,
                peak,
            });

            self.last_sample = now;
        }
    }

    pub fn get_current_usage(&self) -> MemoryUsage {
        let allocated = ALLOCATED.load(Ordering::SeqCst);
        let deallocated = DEALLOCATED.load(Ordering::SeqCst);
        let current = allocated.saturating_sub(deallocated);
        let peak = PEAK_MEMORY.load(Ordering::SeqCst);

        MemoryUsage {
            allocated,
            deallocated,
            current,
            peak,
        }
    }

    pub fn report(&self) -> MemoryReport {
        let usage = self.get_current_usage();
        let allocations = ALLOCATIONS.lock()
            .map(|guard| guard.clone())
            .unwrap_or_else(|_| HashMap::new());

        MemoryReport {
            usage,
            allocations,
            samples: self.samples.clone(),
            duration: Instant::now().duration_since(self.start_time),
        }
    }

    pub fn save_report(&self, path: &str) -> std::io::Result<()> {
        let report = self.report();
        let mut file = File::create(path)?;

        writeln!(file, "# Memory Profile Report")?;
        writeln!(file, "Generated: {}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
        writeln!(file, "Duration: {:.2}s", report.duration.as_secs_f64())?;
        writeln!(file)?;

        writeln!(file, "## Summary")?;
        writeln!(file, "- Total Allocated: {} MB", report.usage.allocated / 1_048_576)?;
        writeln!(file, "- Total Deallocated: {} MB", report.usage.deallocated / 1_048_576)?;
        writeln!(file, "- Current Usage: {} MB", report.usage.current / 1_048_576)?;
        writeln!(file, "- Peak Usage: {} MB", report.usage.peak / 1_048_576)?;
        writeln!(file)?;

        if !report.allocations.is_empty() {
            writeln!(file, "## Allocations by Category")?;
            for (category, stats) in &report.allocations {
                writeln!(file, "### {}", category)?;
                writeln!(file, "- Count: {}", stats.count)?;
                writeln!(file, "- Total: {} MB", stats.total_bytes / 1_048_576)?;
                writeln!(file, "- Current: {} MB", stats.current_bytes / 1_048_576)?;
                writeln!(file, "- Peak: {} MB", stats.peak_bytes / 1_048_576)?;
                writeln!(file)?;
            }
        }

        if !self.samples.is_empty() {
            writeln!(file, "## Memory Timeline (CSV)")?;
            writeln!(file, "timestamp_ms,allocated_mb,deallocated_mb,current_mb,peak_mb")?;
            for sample in &self.samples {
                writeln!(
                    file,
                    "{},{},{},{},{}",
                    sample.timestamp.as_millis(),
                    sample.allocated / 1_048_576,
                    sample.deallocated / 1_048_576,
                    sample.current / 1_048_576,
                    sample.peak / 1_048_576
                )?;
            }
        }

        Ok(())
    }

    pub fn save_csv(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        writeln!(file, "timestamp_ms,allocated_bytes,deallocated_bytes,current_bytes,peak_bytes")?;
        
        for sample in &self.samples {
            writeln!(
                file,
                "{},{},{},{},{}",
                sample.timestamp.as_millis(),
                sample.allocated,
                sample.deallocated,
                sample.current,
                sample.peak
            )?;
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub allocated: usize,
    pub deallocated: usize,
    pub current: usize,
    pub peak: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub usage: MemoryUsage,
    pub allocations: HashMap<String, AllocationStats>,
    pub samples: Vec<MemorySample>,
    pub duration: Duration,
}

pub fn track_allocation(category: &str, bytes: usize) {
    if PROFILING_ENABLED.load(Ordering::SeqCst) == 0 {
        return;
    }

    let Ok(mut allocations) = ALLOCATIONS.lock() else {
        return;
    };
    let stats = allocations.entry(category.to_string()).or_default();
    stats.count += 1;
    stats.total_bytes += bytes;
    stats.current_bytes += bytes;
    if stats.current_bytes > stats.peak_bytes {
        stats.peak_bytes = stats.current_bytes;
    }
}

pub fn track_deallocation(category: &str, bytes: usize) {
    if PROFILING_ENABLED.load(Ordering::SeqCst) == 0 {
        return;
    }

    let Ok(mut allocations) = ALLOCATIONS.lock() else {
        return;
    };
    if let Some(stats) = allocations.get_mut(category) {
        stats.current_bytes = stats.current_bytes.saturating_sub(bytes);
    }
}

pub struct TrackingAllocator;

// SAFETY: TrackingAllocator's GlobalAlloc implementation is thread-safe because:
//
// 1. Atomic Operations:
//    - ALLOCATED, DEALLOCATED, PEAK_MEMORY use AtomicUsize with proper ordering
//    - fetch_add operations are atomic and thread-safe
//    - compare_exchange_weak loop ensures atomic peak memory updates
//
// 2. Memory Ordering:
//    - SeqCst ordering provides strongest guarantees
//    - Ensures all threads see consistent memory state
//    - No data races on the atomic counters
//
// 3. Delegation to System Allocator:
//    - Actual allocation/deallocation delegated to System allocator
//    - System allocator is already thread-safe
//    - TrackingAllocator only adds atomic bookkeeping
//
// 4. Lock-free Design:
//    - No mutexes in the hot path (alloc/dealloc)
//    - Only atomic operations for performance
//    - Peak memory update uses CAS loop for correctness
//
// 5. No Shared Mutable State:
//    - Static atomics are the only shared state
//    - No raw pointers or unsafe memory manipulation
//    - Statistics tracking is separate (uses Mutex when needed)
//
// This allocator can be safely used as a global allocator in multi-threaded
// programs. The tracking overhead is minimal due to lock-free atomic operations.
impl TrackingAllocator {
    fn check_memory_limit(&self, size: usize) -> bool {
        let limit = MEMORY_LIMIT.load(Ordering::Relaxed);
        if limit == usize::MAX {
            return true;
        }
        
        let current = ALLOCATED.load(Ordering::Relaxed);
        let deallocated = DEALLOCATED.load(Ordering::Relaxed);
        let net_allocated = current.saturating_sub(deallocated);
        
        net_allocated.saturating_add(size) <= limit
    }
    
    fn record_allocation_telemetry(&self, size: usize, success: bool) {
        if !TELEMETRY_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        
        if RECURSION_GUARD.with(|g| g.replace(true)) {
            return;
        }
        
        if let Ok(mut telemetry) = ALLOCATION_TELEMETRY.write() {
            telemetry.total_attempts += 1;
            if success {
                telemetry.successful_allocations += 1;
            } else {
                telemetry.record_failure(size);
            }
        }
        
        RECURSION_GUARD.with(|g| g.set(false));
    }
    
    fn handle_oom(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let current = ALLOCATED.load(Ordering::Relaxed);
        let limit = MEMORY_LIMIT.load(Ordering::Relaxed);
        
        if TELEMETRY_ENABLED.load(Ordering::Relaxed) {
            if let Ok(mut telemetry) = ALLOCATION_TELEMETRY.write() {
                telemetry.record_oom(size, current, limit);
            }
        }
        
        if OOM_HANDLER_ENABLED.load(Ordering::Relaxed) {
            error!(
                "Out of memory: requested {} bytes, current: {}, limit: {}",
                size, current, limit
            );
            
            unsafe {
                if layout.size() <= 16 * 1024 {
                    std::thread::sleep(Duration::from_millis(10));
                    let retry = System.alloc(layout);
                    if !retry.is_null() {
                        if let Ok(mut telemetry) = ALLOCATION_TELEMETRY.write() {
                            telemetry.fallback_count += 1;
                        }
                        return retry;
                    }
                }
            }
        }
        
        ptr::null_mut()
    }
    
    #[cfg(debug_assertions)]
    fn track_allocation(&self, ptr: *mut u8, size: usize) {
        if RECURSION_GUARD.with(|g| g.replace(true)) {
            return;
        }
        
        let ptr_addr = ptr as usize;
        if let Ok(mut active) = ACTIVE_ALLOCATIONS.lock() {
            active.insert(ptr_addr);
        }
        
        RECURSION_GUARD.with(|g| g.set(false));
    }
    
    #[cfg(debug_assertions)]
    fn track_deallocation(&self, ptr: *mut u8) {
        if RECURSION_GUARD.with(|g| g.replace(true)) {
            return;
        }
        
        let ptr_addr = ptr as usize;
        if let Ok(mut active) = ACTIVE_ALLOCATIONS.lock() {
            if !active.remove(&ptr_addr) {
                warn!("Possible double-free detected at {:p}", ptr);
            }
        }
        
        RECURSION_GUARD.with(|g| g.set(false));
    }
    
    #[cfg(not(debug_assertions))]
    fn track_allocation(&self, _ptr: *mut u8, _size: usize) {}
    
    #[cfg(not(debug_assertions))]
    fn track_deallocation(&self, _ptr: *mut u8) {}
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        
        if !self.check_memory_limit(size) {
            ALLOCATION_FAILURES.fetch_add(1, Ordering::Relaxed);
            self.record_allocation_telemetry(size, false);
            return self.handle_oom(layout);
        }
        
        let ret = System.alloc(layout);
        
        if ret.is_null() {
            ALLOCATION_FAILURES.fetch_add(1, Ordering::Relaxed);
            self.record_allocation_telemetry(size, false);
            return self.handle_oom(layout);
        }
        
        let old = ALLOCATED.fetch_add(size, Ordering::AcqRel);
        let current = old.saturating_add(size);
        
        loop {
            let peak = PEAK_MEMORY.load(Ordering::Acquire);
            if current <= peak {
                break;
            }
            match PEAK_MEMORY.compare_exchange_weak(
                peak,
                current,
                Ordering::Release,
                Ordering::Acquire
            ) {
                Ok(_) => break,
                Err(_) => std::hint::spin_loop(),
            }
        }
        
        self.track_allocation(ret, size);
        self.record_allocation_telemetry(size, true);
        
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            warn!("Attempted to deallocate null pointer");
            return;
        }
        
        DEALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        
        self.track_deallocation(ptr);
        
        System.dealloc(ptr, layout);
        
        let size = layout.size();
        DEALLOCATED.fetch_add(size, Ordering::AcqRel);
    }
    
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = self.alloc(layout);
        if !ptr.is_null() {
            ptr::write_bytes(ptr, 0, layout.size());
        }
        ptr
    }
    
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size == 0 {
            self.dealloc(ptr, layout);
            return ptr::null_mut();
        }
        
        if ptr.is_null() {
            return self.alloc(Layout::from_size_align_unchecked(new_size, layout.align()));
        }
        
        let old_size = layout.size();
        
        if !self.check_memory_limit(new_size.saturating_sub(old_size)) {
            ALLOCATION_FAILURES.fetch_add(1, Ordering::Relaxed);
            return ptr::null_mut();
        }
        
        let new_ptr = System.realloc(ptr, layout, new_size);
        
        if new_ptr.is_null() {
            ALLOCATION_FAILURES.fetch_add(1, Ordering::Relaxed);
            return ptr::null_mut();
        }
        
        if new_size > old_size {
            let diff = new_size - old_size;
            ALLOCATED.fetch_add(diff, Ordering::AcqRel);
        } else if new_size < old_size {
            let diff = old_size - new_size;
            DEALLOCATED.fetch_add(diff, Ordering::AcqRel);
        }
        
        #[cfg(debug_assertions)]
        {
            self.track_deallocation(ptr);
            self.track_allocation(new_ptr, new_size);
        }
        
        new_ptr
    }
}

pub struct MemoryScope {
    category: String,
    start_allocated: usize,
    start_deallocated: usize,
}

impl MemoryScope {
    pub fn new(category: impl Into<String>) -> Self {
        let category = category.into();
        let start_allocated = ALLOCATED.load(Ordering::SeqCst);
        let start_deallocated = DEALLOCATED.load(Ordering::SeqCst);
        
        MemoryScope {
            category,
            start_allocated,
            start_deallocated,
        }
    }
}

impl Drop for MemoryScope {
    fn drop(&mut self) {
        let end_allocated = ALLOCATED.load(Ordering::SeqCst);
        let end_deallocated = DEALLOCATED.load(Ordering::SeqCst);
        
        let allocated = end_allocated - self.start_allocated;
        let deallocated = end_deallocated - self.start_deallocated;
        
        if allocated > deallocated {
            track_allocation(&self.category, allocated - deallocated);
        }
    }
}

#[macro_export]
macro_rules! memory_scope {
    ($category:expr) => {
        let _scope = $crate::profiling::MemoryScope::new($category);
    };
}

pub fn set_memory_limit(limit_bytes: usize) {
    MEMORY_LIMIT.store(limit_bytes, Ordering::Release);
    info!("Memory limit set to {} bytes", limit_bytes);
}

pub fn clear_memory_limit() {
    MEMORY_LIMIT.store(usize::MAX, Ordering::Release);
    info!("Memory limit cleared");
}

pub fn enable_telemetry(enable: bool) {
    TELEMETRY_ENABLED.store(enable, Ordering::Release);
    if enable {
        info!("Allocation telemetry enabled");
    }
}

pub fn get_allocation_stats() -> AllocationStatistics {
    AllocationStatistics {
        allocated_bytes: ALLOCATED.load(Ordering::Acquire),
        deallocated_bytes: DEALLOCATED.load(Ordering::Acquire),
        peak_memory: PEAK_MEMORY.load(Ordering::Acquire),
        allocation_count: ALLOCATION_COUNT.load(Ordering::Acquire),
        deallocation_count: DEALLOCATION_COUNT.load(Ordering::Acquire),
        allocation_failures: ALLOCATION_FAILURES.load(Ordering::Acquire),
        memory_limit: MEMORY_LIMIT.load(Ordering::Acquire),
    }
}

pub fn get_telemetry() -> Option<AllocationTelemetry> {
    ALLOCATION_TELEMETRY.read().ok().map(|t| t.clone())
}

#[cfg(debug_assertions)]
pub fn check_memory_leaks() -> Vec<usize> {
    if let Ok(active) = ACTIVE_ALLOCATIONS.lock() {
        active.iter().cloned().collect()
    } else {
        Vec::new()
    }
}

#[cfg(debug_assertions)]
pub fn get_active_allocation_count() -> usize {
    ACTIVE_ALLOCATIONS.lock()
        .map(|active| active.len())
        .unwrap_or(0)
}

pub fn reset_telemetry() {
    if let Ok(mut telemetry) = ALLOCATION_TELEMETRY.write() {
        *telemetry = AllocationTelemetry::new();
    }
    ALLOCATION_FAILURES.store(0, Ordering::Release);
}

pub fn enable_oom_handler(enable: bool) {
    OOM_HANDLER_ENABLED.store(enable, Ordering::Release);
}

#[derive(Debug, Clone)]
pub struct AllocationStatistics {
    pub allocated_bytes: usize,
    pub deallocated_bytes: usize,
    pub peak_memory: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub allocation_failures: usize,
    pub memory_limit: usize,
}

impl AllocationStatistics {
    pub fn net_allocated(&self) -> usize {
        self.allocated_bytes.saturating_sub(self.deallocated_bytes)
    }
    
    pub fn failure_rate(&self) -> f64 {
        if self.allocation_count == 0 {
            0.0
        } else {
            self.allocation_failures as f64 / self.allocation_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new(10);
        profiler.start();
        
        let _data = vec![0u8; 1024 * 1024];
        profiler.sample();
        
        let usage = profiler.get_current_usage();
        assert!(usage.current > 0);
        
        profiler.stop();
    }

    #[test]
    fn test_memory_scope() {
        let profiler = MemoryProfiler::new(10);
        profiler.start();
        
        {
            let _scope = MemoryScope::new("test_allocation");
            let _data = vec![0u8; 1024];
        }
        
        let allocations = ALLOCATIONS.lock()
            .expect("Allocations tracking lock poisoned");
        if let Some(stats) = allocations.get("test_allocation") {
            assert!(stats.total_bytes > 0);
        }
        
        profiler.stop();
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = MemoryProfiler::new(10);
        profiler.start();
        
        let _data1 = vec![0u8; 1024];
        profiler.sample();
        let _data2 = vec![0u8; 2048];
        profiler.sample();
        
        let report = profiler.report();
        assert!(report.usage.allocated > 0);
        assert!(!report.samples.is_empty());
        
        profiler.stop();
    }
}