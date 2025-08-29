use crate::error::{Result, SrganError};
use ndarray::ArrayD;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    None,
    Cuda,
    OpenCL,
    Metal,
    Vulkan,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GpuBackend::None => write!(f, "CPU"),
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::OpenCL => write!(f, "OpenCL"),
            GpuBackend::Metal => write!(f, "Metal"),
            GpuBackend::Vulkan => write!(f, "Vulkan"),
        }
    }
}

impl GpuBackend {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" | "none" => Ok(GpuBackend::None),
            "cuda" => Ok(GpuBackend::Cuda),
            "opencl" | "cl" => Ok(GpuBackend::OpenCL),
            "metal" => Ok(GpuBackend::Metal),
            "vulkan" | "vk" => Ok(GpuBackend::Vulkan),
            _ => Err(SrganError::InvalidParameter(format!(
                "Unknown GPU backend: {}. Valid options: cpu, cuda, opencl, metal, vulkan",
                s
            ))),
        }
    }

    pub fn is_available(&self) -> bool {
        match self {
            GpuBackend::None => true,
            // For now, GPU backends are not implemented
            // In a real implementation, we would check for driver/library availability
            _ => false,
        }
    }
}

pub struct GpuDevice {
    backend: GpuBackend,
    device_id: usize,
    memory_mb: usize,
    name: String,
}

impl GpuDevice {
    pub fn cpu() -> Self {
        GpuDevice {
            backend: GpuBackend::None,
            device_id: 0,
            memory_mb: 0,
            name: "CPU".to_string(),
        }
    }

    pub fn list_devices() -> Vec<GpuDevice> {
        // For now, only CPU is available
        // In a real implementation, we would query available GPU devices
        vec![Self::cpu()]
    }

    pub fn select_best() -> Self {
        // Select the best available device
        // For now, this is always CPU
        Self::cpu()
    }

    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn memory_mb(&self) -> usize {
        self.memory_mb
    }
}

pub trait GpuCompute {
    fn to_device(&mut self, device: &GpuDevice) -> Result<()>;
    fn to_cpu(&mut self) -> Result<()>;
    fn is_on_device(&self) -> bool;
}

impl GpuCompute for ArrayD<f32> {
    fn to_device(&mut self, device: &GpuDevice) -> Result<()> {
        match device.backend() {
            GpuBackend::None => Ok(()),
            _ => Err(SrganError::InvalidParameter(format!(
                "GPU backend {} is not yet implemented",
                device.backend()
            ))),
        }
    }

    fn to_cpu(&mut self) -> Result<()> {
        // Already on CPU
        Ok(())
    }

    fn is_on_device(&self) -> bool {
        false
    }
}

/// Thread-safe GPU context that can be shared across threads
pub struct GpuContext {
    device: Arc<GpuDevice>,
    allocated_mb: Arc<RwLock<usize>>,
    memory_pools: Arc<RwLock<HashMap<std::thread::ThreadId, MemoryPool>>>,
}

/// Per-thread memory pool for GPU allocations
struct MemoryPool {
    allocations: Vec<GpuAllocation>,
    total_size: usize,
}

/// Individual GPU memory allocation
struct GpuAllocation {
    size: usize,
    #[allow(dead_code)]
    data: Vec<u8>,  // Placeholder for actual GPU memory
}

impl GpuContext {
    pub fn new(backend: GpuBackend) -> Result<Self> {
        if !backend.is_available() {
            return Err(SrganError::InvalidParameter(format!(
                "GPU backend {} is not available on this system",
                backend
            )));
        }

        let device = if backend == GpuBackend::None {
            GpuDevice::cpu()
        } else {
            // In a real implementation, we would select the best GPU
            return Err(SrganError::InvalidParameter(format!(
                "GPU backend {} is not yet implemented",
                backend
            )));
        };

        Ok(GpuContext {
            device: Arc::new(device),
            allocated_mb: Arc::new(RwLock::new(0)),
            memory_pools: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    pub fn allocated_mb(&self) -> usize {
        *self.allocated_mb.read().unwrap()
    }

    pub fn available_mb(&self) -> usize {
        let allocated = *self.allocated_mb.read().unwrap();
        if self.device.memory_mb > allocated {
            self.device.memory_mb - allocated
        } else {
            0
        }
    }
    
    /// Allocate memory for the current thread
    pub fn allocate(&self, size_mb: usize) -> Result<()> {
        let thread_id = std::thread::current().id();
        let mut pools = self.memory_pools.write().unwrap();
        let mut allocated = self.allocated_mb.write().unwrap();
        
        // Check if we have enough memory
        if *allocated + size_mb > self.device.memory_mb {
            return Err(SrganError::InvalidParameter(
                format!("Insufficient GPU memory: requested {} MB, available {} MB", 
                    size_mb, self.device.memory_mb - *allocated)
            ));
        }
        
        // Get or create thread pool
        let pool = pools.entry(thread_id).or_insert(MemoryPool {
            allocations: Vec::new(),
            total_size: 0,
        });
        
        // Add allocation
        pool.allocations.push(GpuAllocation {
            size: size_mb,
            data: vec![0; size_mb * 1024 * 1024],  // Placeholder
        });
        pool.total_size += size_mb;
        *allocated += size_mb;
        
        Ok(())
    }
    
    /// Free memory for the current thread
    pub fn free_thread_memory(&self) {
        let thread_id = std::thread::current().id();
        let mut pools = self.memory_pools.write().unwrap();
        let mut allocated = self.allocated_mb.write().unwrap();
        
        if let Some(pool) = pools.remove(&thread_id) {
            *allocated -= pool.total_size;
        }
    }
}

// SAFETY: GpuContext can be safely sent and shared between threads because:
//
// 1. All fields use thread-safe synchronization primitives:
//    - `device: Arc<GpuDevice>` - Arc provides thread-safe reference counting
//    - `allocated_mb: Arc<RwLock<usize>>` - RwLock ensures synchronized read/write access
//    - `memory_pools: Arc<RwLock<HashMap<ThreadId, MemoryPool>>>` - Protected by RwLock
//
// 2. Memory Pool Isolation:
//    - Each thread has its own MemoryPool indexed by ThreadId
//    - No memory pool is shared between threads
//    - RwLock ensures atomic updates to the HashMap
//
// 3. GpuDevice Safety:
//    - GpuDevice contains only primitive types (backend enum, memory sizes)
//    - No raw pointers or external resources
//    - Immutable after construction (no interior mutability)
//
// 4. Allocation Safety:
//    - All allocation/deallocation operations acquire write locks
//    - Memory accounting (allocated_mb) is protected by RwLock
//    - Thread-local pools prevent cross-thread memory corruption
//
// 5. No GPU API calls:
//    - Currently, this is a placeholder implementation
//    - When real GPU APIs are added, they must be thread-safe or protected
//    - Most GPU APIs (CUDA, OpenCL) have their own thread safety guarantees
//
// Note: When implementing actual GPU backends, ensure that:
// - GPU context handles are thread-safe or wrapped in synchronization primitives
// - Memory operations are properly synchronized
// - Command queues/streams are either thread-local or synchronized
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl Clone for GpuContext {
    fn clone(&self) -> Self {
        GpuContext {
            device: Arc::clone(&self.device),
            allocated_mb: Arc::clone(&self.allocated_mb),
            memory_pools: Arc::clone(&self.memory_pools),
        }
    }
}

// #[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    
    pub fn is_cuda_available() -> bool {
        // Check if CUDA runtime is available
        // This would require cuda-sys or similar crate
        false
    }
    
    pub fn get_cuda_devices() -> Vec<GpuDevice> {
        vec![]
    }
}

// #[cfg(feature = "opencl")]
mod opencl {
    use super::*;
    
    pub fn is_opencl_available() -> bool {
        // Check if OpenCL runtime is available
        // This would require ocl or similar crate
        false
    }
    
    pub fn get_opencl_devices() -> Vec<GpuDevice> {
        vec![]
    }
}

// #[cfg(feature = "metal")]
mod metal {
    use super::*;
    
    pub fn is_metal_available() -> bool {
        // Check if Metal is available (macOS only)
        // This would require metal-rs or similar crate
        false
    }
    
    pub fn get_metal_devices() -> Vec<GpuDevice> {
        vec![]
    }
}

// #[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    
    pub fn is_vulkan_available() -> bool {
        // Check if Vulkan runtime is available
        // This would require vulkano or similar crate
        false
    }
    
    pub fn get_vulkan_devices() -> Vec<GpuDevice> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_from_str() {
        assert_eq!(GpuBackend::from_str("cpu").unwrap(), GpuBackend::None);
        assert_eq!(GpuBackend::from_str("cuda").unwrap(), GpuBackend::Cuda);
        assert_eq!(GpuBackend::from_str("opencl").unwrap(), GpuBackend::OpenCL);
        assert_eq!(GpuBackend::from_str("metal").unwrap(), GpuBackend::Metal);
        assert_eq!(GpuBackend::from_str("vulkan").unwrap(), GpuBackend::Vulkan);
        assert!(GpuBackend::from_str("invalid").is_err());
    }

    #[test]
    fn test_gpu_device_list() {
        let devices = GpuDevice::list_devices();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].backend(), GpuBackend::None);
    }

    #[test]
    fn test_gpu_context_cpu() {
        let context = GpuContext::new(GpuBackend::None).unwrap();
        assert_eq!(context.device().backend(), GpuBackend::None);
    }

    #[test]
    fn test_gpu_compute_trait() {
        let mut arr = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
        let device = GpuDevice::cpu();
        assert!(arr.to_device(&device).is_ok());
        assert!(!arr.is_on_device());
        assert!(arr.to_cpu().is_ok());
    }
}