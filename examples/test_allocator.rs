use srgan_rust::profiling::*;
use std::alloc::{GlobalAlloc, Layout};

fn main() {
    println!("Testing TrackingAllocator...");
    
    // Enable telemetry
    enable_telemetry(true);
    
    // Set a memory limit
    set_memory_limit(10 * 1024 * 1024); // 10 MB
    
    let allocator = TrackingAllocator;
    
    // Test basic allocation
    unsafe {
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let ptr = allocator.alloc(layout);
        
        if ptr.is_null() {
            println!("Allocation failed!");
        } else {
            println!("Successfully allocated 1024 bytes at {:p}", ptr);
            
            // Write some data
            for i in 0..1024 {
                *ptr.add(i) = (i & 0xFF) as u8;
            }
            
            // Verify data
            let mut ok = true;
            for i in 0..1024 {
                if *ptr.add(i) != (i & 0xFF) as u8 {
                    ok = false;
                    break;
                }
            }
            
            if ok {
                println!("Data integrity check passed");
            } else {
                println!("Data integrity check failed!");
            }
            
            // Deallocate
            allocator.dealloc(ptr, layout);
            println!("Memory deallocated");
        }
    }
    
    // Test reallocation
    unsafe {
        let layout1 = Layout::from_size_align(100, 8).unwrap();
        let ptr = allocator.alloc(layout1);
        
        if !ptr.is_null() {
            println!("Allocated 100 bytes for realloc test");
            
            // Fill with pattern
            for i in 0..100 {
                *ptr.add(i) = 0xAB;
            }
            
            // Reallocate to larger size
            let ptr2 = allocator.realloc(ptr, layout1, 200);
            
            if !ptr2.is_null() {
                println!("Successfully reallocated to 200 bytes");
                
                // Check pattern preserved
                let mut ok = true;
                for i in 0..100 {
                    if *ptr2.add(i) != 0xAB {
                        ok = false;
                        break;
                    }
                }
                
                if ok {
                    println!("Data preserved after reallocation");
                } else {
                    println!("Data corrupted after reallocation!");
                }
                
                let layout2 = Layout::from_size_align(200, 8).unwrap();
                allocator.dealloc(ptr2, layout2);
            }
        }
    }
    
    // Test multiple allocations
    let mut ptrs = Vec::new();
    for i in 0..10 {
        unsafe {
            let size = 1024 * (i + 1);
            let layout = Layout::from_size_align(size, 8).unwrap();
            let ptr = allocator.alloc(layout);
            
            if !ptr.is_null() {
                println!("Allocated {} bytes", size);
                ptrs.push((ptr, layout));
            } else {
                println!("Failed to allocate {} bytes", size);
            }
        }
    }
    
    // Deallocate all
    for (ptr, layout) in ptrs {
        unsafe {
            allocator.dealloc(ptr, layout);
        }
    }
    println!("All allocations cleaned up");
    
    // Print statistics
    let stats = get_allocation_stats();
    println!("\nAllocation Statistics:");
    println!("  Total allocated: {} bytes", stats.allocated_bytes);
    println!("  Total deallocated: {} bytes", stats.deallocated_bytes);
    println!("  Peak memory: {} bytes", stats.peak_memory);
    println!("  Allocation count: {}", stats.allocation_count);
    println!("  Deallocation count: {}", stats.deallocation_count);
    println!("  Allocation failures: {}", stats.allocation_failures);
    
    if let Some(telemetry) = get_telemetry() {
        println!("\nTelemetry:");
        println!("  Total attempts: {}", telemetry.total_attempts);
        println!("  Successful allocations: {}", telemetry.successful_allocations);
        println!("  Failed allocations: {}", telemetry.failed_allocations);
        
        if !telemetry.oom_events.is_empty() {
            println!("  OOM events: {}", telemetry.oom_events.len());
        }
    }
    
    #[cfg(debug_assertions)]
    {
        let leak_report = check_memory_leaks();
        if leak_report.total_leaked_count > 0 {
            println!("\nWARNING: Memory leaks detected!");
            leak_report.print_report();
        } else {
            println!("\nNo memory leaks detected!");
        }
    }
    
    println!("\nAll tests completed!");
}