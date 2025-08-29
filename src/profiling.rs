use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;
use chrono::Local;
use log::info;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_MEMORY: AtomicUsize = AtomicUsize::new(0);

lazy_static::lazy_static! {
    static ref ALLOCATIONS: Mutex<HashMap<String, AllocationStats>> = Mutex::new(HashMap::new());
    static ref PROFILING_ENABLED: AtomicUsize = AtomicUsize::new(0);
}

#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub count: usize,
    pub total_bytes: usize,
    pub current_bytes: usize,
    pub peak_bytes: usize,
}

impl Default for AllocationStats {
    fn default() -> Self {
        AllocationStats {
            count: 0,
            total_bytes: 0,
            current_bytes: 0,
            peak_bytes: 0,
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

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            let size = layout.size();
            let old = ALLOCATED.fetch_add(size, Ordering::SeqCst);
            let current = old + size;
            
            loop {
                let peak = PEAK_MEMORY.load(Ordering::SeqCst);
                if current <= peak {
                    break;
                }
                if PEAK_MEMORY.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst
                ).is_ok() {
                    break;
                }
            }
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        let size = layout.size();
        DEALLOCATED.fetch_add(size, Ordering::SeqCst);
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