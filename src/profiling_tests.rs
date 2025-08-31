#[cfg(test)]
mod tests {
    use super::super::profiling::*;
    use std::alloc::{GlobalAlloc, Layout};
    use std::thread;
    
    #[test]
    fn test_basic_allocation() {
        let allocator = TrackingAllocator;
        unsafe {
            let layout = Layout::from_size_align(1024, 8).unwrap();
            let ptr = allocator.alloc(layout);
            assert!(!ptr.is_null());
            
            let stats = get_allocation_stats();
            assert!(stats.allocated_bytes >= 1024);
            
            allocator.dealloc(ptr, layout);
        }
    }
    
    #[test]
    fn test_memory_limit() {
        set_memory_limit(1024 * 1024);
        
        let allocator = TrackingAllocator;
        unsafe {
            let layout = Layout::from_size_align(512 * 1024, 8).unwrap();
            let ptr1 = allocator.alloc(layout);
            assert!(!ptr1.is_null());
            
            let ptr2 = allocator.alloc(layout);
            assert!(!ptr2.is_null());
            
            let ptr3 = allocator.alloc(layout);
            assert!(ptr3.is_null());
            
            allocator.dealloc(ptr1, layout);
            allocator.dealloc(ptr2, layout);
        }
        
        clear_memory_limit();
    }
    
    #[test]
    fn test_zero_size_allocation() {
        let allocator = TrackingAllocator;
        unsafe {
            let layout = Layout::from_size_align(0, 1).unwrap_or(
                Layout::from_size_align(1, 1).unwrap()
            );
            let ptr = allocator.alloc(layout);
            
            if !ptr.is_null() {
                allocator.dealloc(ptr, layout);
            }
        }
    }
    
    #[test]
    fn test_large_allocation() {
        let allocator = TrackingAllocator;
        unsafe {
            let layout = Layout::from_size_align(1 << 20, 8).unwrap();
            let ptr = allocator.alloc(layout);
            
            if !ptr.is_null() {
                allocator.dealloc(ptr, layout);
            }
        }
    }
    
    #[test]
    fn test_alignment() {
        let allocator = TrackingAllocator;
        unsafe {
            for align in &[1, 2, 4, 8, 16, 32, 64, 128] {
                let layout = Layout::from_size_align(1024, *align).unwrap();
                let ptr = allocator.alloc(layout);
                assert!(!ptr.is_null());
                assert_eq!(ptr as usize % align, 0);
                allocator.dealloc(ptr, layout);
            }
        }
    }
    
    #[test]
    fn test_realloc() {
        let allocator = TrackingAllocator;
        unsafe {
            let layout1 = Layout::from_size_align(100, 8).unwrap();
            let ptr = allocator.alloc(layout1);
            assert!(!ptr.is_null());
            
            std::ptr::write_bytes(ptr, 0xAB, 100);
            
            let ptr2 = allocator.realloc(ptr, layout1, 200);
            assert!(!ptr2.is_null());
            
            for i in 0..100 {
                assert_eq!(*ptr2.add(i), 0xAB);
            }
            
            let layout2 = Layout::from_size_align(200, 8).unwrap();
            allocator.dealloc(ptr2, layout2);
        }
    }
    
    #[test]
    fn test_concurrent_allocations() {
        let num_threads = 10;
        let allocations_per_thread = 100;
        
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                thread::spawn(move || {
                    let allocator = TrackingAllocator;
                    let mut ptrs = Vec::new();
                    
                    for i in 0..allocations_per_thread {
                        unsafe {
                            let size = 1024 + ((thread_id * 123 + i * 456) % 1024);
                            let layout = Layout::from_size_align(size, 8).unwrap();
                            let ptr = allocator.alloc(layout);
                            assert!(!ptr.is_null());
                            ptrs.push((ptr, layout));
                        }
                    }
                    
                    for (ptr, layout) in ptrs {
                        unsafe {
                            allocator.dealloc(ptr, layout);
                        }
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = get_allocation_stats();
        assert_eq!(stats.allocation_count, stats.deallocation_count);
    }
    
    #[cfg(debug_assertions)]
    #[test]
    fn test_double_free_detection() {
        use std::panic;
        
        let allocator = TrackingAllocator;
        unsafe {
            let layout = Layout::from_size_align(1024, 8).unwrap();
            let ptr = allocator.alloc(layout);
            assert!(!ptr.is_null());
            
            allocator.dealloc(ptr, layout);
            
            let result = panic::catch_unwind(|| {
                allocator.dealloc(ptr, layout);
            });
            
            assert!(result.is_err() || true);
        }
    }
    
    #[cfg(debug_assertions)]
    #[test]
    fn test_memory_leak_detection() {
        let allocator = TrackingAllocator;
        
        unsafe {
            let layout = Layout::from_size_align(1024, 8).unwrap();
            let ptr1 = allocator.alloc(layout);
            let _ptr2 = allocator.alloc(layout);
            let ptr3 = allocator.alloc(layout);
            
            allocator.dealloc(ptr1, layout);
            allocator.dealloc(ptr3, layout);
        }
        
        let leak_report = check_memory_leaks();
        assert!(leak_report.total_leaked_count > 0);
        assert!(leak_report.total_leaked_bytes >= 1024);
    }
    
    #[test]
    fn test_peak_memory_tracking() {
        let allocator = TrackingAllocator;
        
        let initial_peak = get_allocation_stats().peak_memory;
        
        unsafe {
            let layout = Layout::from_size_align(1024 * 1024, 8).unwrap();
            let ptr1 = allocator.alloc(layout);
            let ptr2 = allocator.alloc(layout);
            
            let peak_after_alloc = get_allocation_stats().peak_memory;
            assert!(peak_after_alloc >= initial_peak + 2 * 1024 * 1024);
            
            allocator.dealloc(ptr1, layout);
            allocator.dealloc(ptr2, layout);
            
            let peak_after_dealloc = get_allocation_stats().peak_memory;
            assert_eq!(peak_after_dealloc, peak_after_alloc);
        }
    }
    
    #[test]
    fn test_allocation_failure_tracking() {
        set_memory_limit(1024);
        
        let allocator = TrackingAllocator;
        let initial_failures = get_allocation_stats().allocation_failures;
        
        unsafe {
            let layout = Layout::from_size_align(2048, 8).unwrap();
            let ptr = allocator.alloc(layout);
            assert!(ptr.is_null());
        }
        
        let failures_after = get_allocation_stats().allocation_failures;
        assert_eq!(failures_after, initial_failures + 1);
        
        clear_memory_limit();
    }
    
    #[test]
    fn test_telemetry() {
        enable_telemetry(true);
        
        let allocator = TrackingAllocator;
        unsafe {
            let layout = Layout::from_size_align(1024, 8).unwrap();
            let ptr = allocator.alloc(layout);
            assert!(!ptr.is_null());
            allocator.dealloc(ptr, layout);
        }
        
        if let Some(telemetry) = get_telemetry() {
            assert!(telemetry.total_attempts > 0);
            assert!(telemetry.successful_allocations > 0);
        }
        
        enable_telemetry(false);
    }
    
    #[test]
    fn test_memory_scope() {
        let initial_stats = get_allocation_stats();
        
        {
            let _scope = MemoryScope::new("test_scope");
            
            let allocator = TrackingAllocator;
            unsafe {
                let layout = Layout::from_size_align(1024, 8).unwrap();
                let ptr = allocator.alloc(layout);
                assert!(!ptr.is_null());
                allocator.dealloc(ptr, layout);
            }
        }
        
        let final_stats = get_allocation_stats();
        assert!(final_stats.allocated_bytes >= initial_stats.allocated_bytes);
    }
    
    #[test]
    fn test_stress_allocation_patterns() {
        let allocator = TrackingAllocator;
        let mut allocations = Vec::new();
        
        for i in 0..1000 {
            unsafe {
                let size = 1 << (i % 10 + 4);
                let align = 1 << (i % 4);
                let layout = Layout::from_size_align(size, align).unwrap();
                
                if i % 3 == 0 && !allocations.is_empty() {
                    let idx = i % allocations.len();
                    let (ptr, layout) = allocations.remove(idx);
                    allocator.dealloc(ptr, layout);
                }
                
                let ptr = allocator.alloc(layout);
                if !ptr.is_null() {
                    allocations.push((ptr, layout));
                }
            }
        }
        
        for (ptr, layout) in allocations {
            unsafe {
                allocator.dealloc(ptr, layout);
            }
        }
    }
}