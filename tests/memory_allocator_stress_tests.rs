use srgan_rust::profiling::{
    set_memory_limit, clear_memory_limit, enable_telemetry, get_allocation_stats,
    get_telemetry, reset_telemetry, enable_oom_handler,
};
use std::thread;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[cfg(debug_assertions)]
use srgan_rust::profiling::{check_memory_leaks, get_active_allocation_count};

#[test]
fn test_memory_limit_enforcement() {
    enable_telemetry(true);
    reset_telemetry();
    
    let limit = 1024 * 1024;
    set_memory_limit(limit);
    
    let mut allocations = Vec::new();
    let allocation_size = 256 * 1024;
    
    for _ in 0..10 {
        let vec: Result<Vec<u8>, _> = (|| {
            let mut v = Vec::new();
            v.try_reserve(allocation_size).map(|_| v)
        })();
        if let Ok(v) = vec {
            allocations.push(v);
        }
    }
    
    let stats = get_allocation_stats();
    assert!(stats.net_allocated() <= limit + allocation_size);
    assert!(stats.allocation_failures > 0);
    
    clear_memory_limit();
    reset_telemetry();
}

#[test]
fn test_concurrent_allocations() {
    enable_telemetry(true);
    reset_telemetry();
    
    let thread_count = 8;
    let allocations_per_thread = 100;
    let allocation_size = 1024;
    
    let total_allocations = Arc::new(AtomicUsize::new(0));
    let total_failures = Arc::new(AtomicUsize::new(0));
    
    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let total_allocs = Arc::clone(&total_allocations);
            let total_fails = Arc::clone(&total_failures);
            
            thread::spawn(move || {
                let mut local_allocations = Vec::new();
                let mut failures = 0;
                
                for _ in 0..allocations_per_thread {
                    let mut v = Vec::new();
                    match v.try_reserve(allocation_size) {
                        Ok(_) => {
                            v.resize(allocation_size, 0);
                            local_allocations.push(v);
                        }
                        Err(_) => failures += 1,
                    }
                }
                
                total_allocs.fetch_add(local_allocations.len(), Ordering::Relaxed);
                total_fails.fetch_add(failures, Ordering::Relaxed);
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    let stats = get_allocation_stats();
    let expected_count = thread_count * allocations_per_thread;
    
    assert!(stats.allocation_count >= expected_count);
    assert_eq!(
        total_allocations.load(Ordering::Relaxed) + total_failures.load(Ordering::Relaxed),
        expected_count
    );
    
    reset_telemetry();
}

#[test]
fn test_allocation_telemetry() {
    enable_telemetry(true);
    reset_telemetry();
    
    let allocations: Vec<_> = (0..10)
        .map(|i| vec![0u8; 1024 * (i + 1)])
        .collect();
    
    let telemetry = get_telemetry().expect("Telemetry should be available");
    assert!(telemetry.successful_allocations > 0);
    assert_eq!(telemetry.failed_allocations, 0);
    
    drop(allocations);
    reset_telemetry();
}

#[test]
fn test_oom_recovery() {
    enable_telemetry(true);
    enable_oom_handler(true);
    reset_telemetry();
    
    let small_limit = 1024 * 1024;
    set_memory_limit(small_limit);
    
    let mut allocations = Vec::new();
    let mut recovered = false;
    
    for i in 0..20 {
        let size = if i < 10 { 128 * 1024 } else { 8 * 1024 };
        
        let mut v: Vec<u8> = Vec::new();
        match v.try_reserve(size) {
            Ok(_) => {
                allocations.push(v);
                if i >= 10 {
                    recovered = true;
                }
            }
            Err(_) => {}
        }
    }
    
    let telemetry = get_telemetry().expect("Telemetry should be available");
    
    if recovered {
        assert!(telemetry.fallback_count > 0 || telemetry.failed_allocations > 0);
    }
    
    clear_memory_limit();
    reset_telemetry();
}

#[test]
fn test_allocation_statistics() {
    reset_telemetry();
    
    let initial_stats = get_allocation_stats();
    
    let allocations: Vec<_> = (0..5)
        .map(|_| vec![0u8; 1024])
        .collect();
    
    let after_alloc_stats = get_allocation_stats();
    assert!(after_alloc_stats.allocated_bytes > initial_stats.allocated_bytes);
    assert!(after_alloc_stats.allocation_count > initial_stats.allocation_count);
    
    drop(allocations);
    
    let after_dealloc_stats = get_allocation_stats();
    assert!(after_dealloc_stats.deallocated_bytes > after_alloc_stats.deallocated_bytes);
    assert!(after_dealloc_stats.deallocation_count > after_alloc_stats.deallocation_count);
}

#[test]
fn test_peak_memory_tracking() {
    reset_telemetry();
    
    let initial_stats = get_allocation_stats();
    let initial_peak = initial_stats.peak_memory;
    
    let large_allocation = vec![0u8; 10 * 1024 * 1024];
    let stats_with_large = get_allocation_stats();
    assert!(stats_with_large.peak_memory >= initial_peak + 10 * 1024 * 1024);
    
    drop(large_allocation);
    
    let _small_allocation = vec![0u8; 1024];
    let final_stats = get_allocation_stats();
    
    assert!(final_stats.peak_memory >= stats_with_large.peak_memory);
}

#[test]
fn test_memory_limit_with_realloc() {
    enable_telemetry(true);
    reset_telemetry();
    set_memory_limit(2 * 1024 * 1024);
    
    let mut vec = Vec::with_capacity(1024);
    
    for i in 0..20 {
        vec.push(i);
        if i % 100 == 0 {
            vec.reserve(1024);
        }
    }
    
    let stats = get_allocation_stats();
    assert!(stats.net_allocated() <= 2 * 1024 * 1024 + 65536);
    
    clear_memory_limit();
    reset_telemetry();
}

#[cfg(debug_assertions)]
#[test]
fn test_memory_leak_detection() {
    reset_telemetry();
    
    let initial_count = get_active_allocation_count();
    
    {
        let _temp = vec![0u8; 1024];
        let during_count = get_active_allocation_count();
        assert!(during_count > initial_count);
    }
    
    thread::sleep(Duration::from_millis(10));
    
    let final_count = get_active_allocation_count();
    assert_eq!(final_count, initial_count);
}

#[cfg(debug_assertions)]
#[test]
fn test_leak_detection_with_multiple_allocations() {
    reset_telemetry();
    
    let initial_leaks = check_memory_leaks();
    let initial_count = initial_leaks.len();
    
    {
        let _v1 = vec![1u8; 512];
        let _v2 = vec![2u8; 1024];
        let _v3 = vec![3u8; 2048];
        
        let during_leaks = check_memory_leaks();
        assert!(during_leaks.len() >= initial_count + 3);
    }
    
    thread::sleep(Duration::from_millis(10));
    
    let final_leaks = check_memory_leaks();
    assert_eq!(final_leaks.len(), initial_count);
}

#[test]
fn test_stress_allocation_deallocation_pattern() {
    enable_telemetry(true);
    reset_telemetry();
    
    let iterations = 1000;
    let mut allocations = Vec::new();
    
    for i in 0..iterations {
        if i % 3 == 0 && !allocations.is_empty() {
            let idx = i % allocations.len();
            allocations.remove(idx);
        }
        
        let size = ((i * 17) % 1024) + 128;
        let mut v = Vec::new();
        if v.try_reserve(size).is_ok() {
            v.resize(size, (i % 256) as u8);
            allocations.push(v);
        }
        
        if allocations.len() > 100 {
            allocations.truncate(50);
        }
    }
    
    let stats = get_allocation_stats();
    assert!(stats.allocation_count >= iterations);
    assert!(stats.deallocation_count > 0);
    
    let telemetry = get_telemetry().expect("Telemetry should be available");
    assert!(telemetry.successful_allocations > 0);
    
    reset_telemetry();
}

#[test]
fn test_failure_rate_calculation() {
    enable_telemetry(true);
    reset_telemetry();
    set_memory_limit(1024 * 1024);
    
    let mut allocations = Vec::new();
    let large_size = 512 * 1024;
    
    for _ in 0..10 {
        let mut v: Vec<u8> = Vec::new();
        if v.try_reserve(large_size).is_ok() {
            allocations.push(v);
        }
    }
    
    let stats = get_allocation_stats();
    let failure_rate = stats.failure_rate();
    
    assert!(failure_rate > 0.0);
    assert!(failure_rate <= 1.0);
    
    if stats.allocation_failures > 0 {
        assert!(failure_rate > 0.0);
    }
    
    clear_memory_limit();
    reset_telemetry();
}

#[test]
fn test_oom_event_recording() {
    enable_telemetry(true);
    reset_telemetry();
    set_memory_limit(1024 * 1024);
    
    let mut allocations = Vec::new();
    
    for _ in 0..20 {
        let mut v: Vec<u8> = Vec::new();
        match v.try_reserve(512 * 1024) {
            Ok(_) => allocations.push(v),
            Err(_) => {}
        }
    }
    
    if let Some(telemetry) = get_telemetry() {
        if telemetry.failed_allocations > 0 {
            assert!(!telemetry.oom_events.is_empty());
            
            for event in &telemetry.oom_events {
                assert!(event.requested_size > 0);
                assert!(event.memory_limit == 1024 * 1024);
            }
        }
    }
    
    clear_memory_limit();
    reset_telemetry();
}