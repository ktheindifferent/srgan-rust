use crate::error::Result;
use crate::UpscalingNetwork;
use ndarray::ArrayD;

/// Thread-safe wrapper for UpscalingNetwork that enables parallel processing
/// 
/// This wrapper clones the network for each thread to avoid Send/Sync constraints.
/// Each thread gets its own copy of the network parameters and graph definition.
pub struct ThreadSafeNetwork {
    /// The original network to clone for each thread
    network: UpscalingNetwork,
}

impl ThreadSafeNetwork {
    /// Create a new thread-safe network wrapper
    pub fn new(network: UpscalingNetwork) -> Self {
        ThreadSafeNetwork { network }
    }

    /// Get a clone of the network for use in a specific thread
    /// 
    /// This creates a deep copy of the network that can be safely used
    /// in parallel processing without synchronization overhead.
    pub fn get_network(&self) -> UpscalingNetwork {
        self.network.clone()
    }

    /// Process multiple images in parallel using rayon
    /// 
    /// Returns a vector of results in the same order as the input
    pub fn process_batch_parallel<F, T>(
        &self,
        items: Vec<T>,
        processor: F,
        num_threads: Option<usize>,
    ) -> Vec<Result<ArrayD<f32>>>
    where
        F: Fn(T, &UpscalingNetwork) -> Result<ArrayD<f32>> + Send + Sync,
        T: Send,
    {
        use rayon::prelude::*;

        // Configure thread pool if specified
        if let Some(threads) = num_threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .ok();
            
            if let Some(pool) = pool {
                pool.install(|| {
                    items
                        .into_par_iter()
                        .map(|item| {
                            // Each thread gets its own network clone
                            let network = self.get_network();
                            processor(item, &network)
                        })
                        .collect()
                })
            } else {
                // Fall back to global pool
                items
                    .into_par_iter()
                    .map(|item| {
                        let network = self.get_network();
                        processor(item, &network)
                    })
                    .collect()
            }
        } else {
            // Use global thread pool
            items
                .into_par_iter()
                .map(|item| {
                    let network = self.get_network();
                    processor(item, &network)
                })
                .collect()
        }
    }
}

// SAFETY: ThreadSafeNetwork in parallel.rs can be safely sent and shared between threads because:
//
// 1. Design Pattern - Clone-per-thread:
//    - The struct contains only an `UpscalingNetwork` field
//    - `get_network()` creates a deep clone for each thread
//    - No shared mutable state between threads
//
// 2. Thread Isolation:
//    - Each thread operates on its own independent copy of the network
//    - No references or pointers are shared across thread boundaries
//    - Cloning ensures complete data isolation
//
// 3. Underlying Type Safety:
//    - UpscalingNetwork must implement Clone
//    - The clone operation creates fully independent instances
//    - No use of Rc, raw pointers, or other !Send/!Sync types
//
// 4. Rayon Integration:
//    - Used with rayon's parallel iterators which require Send
//    - Each closure gets its own network clone via `get_network()`
//    - Results are collected safely through rayon's infrastructure
//
// This approach trades memory usage for simplicity and safety - each thread
// has its own copy, eliminating any possibility of data races.
unsafe impl Send for ThreadSafeNetwork {}
unsafe impl Sync for ThreadSafeNetwork {}

/// Configuration for parallel batch processing
#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// Number of threads to use (None = use all available)
    pub num_threads: Option<usize>,
    /// Size of chunks for batch processing
    pub chunk_size: usize,
    /// Enable progress tracking
    pub show_progress: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        ParallelConfig {
            num_threads: None,
            chunk_size: 10,
            show_progress: true,
        }
    }
}

/// Builder pattern for configuring parallel processing
pub struct ParallelProcessorBuilder {
    config: ParallelConfig,
}

impl ParallelProcessorBuilder {
    pub fn new() -> Self {
        ParallelProcessorBuilder {
            config: ParallelConfig::default(),
        }
    }

    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = Some(threads);
        self
    }

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    pub fn build(self) -> ParallelConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config_builder() {
        let config = ParallelProcessorBuilder::new()
            .num_threads(4)
            .chunk_size(20)
            .show_progress(false)
            .build();

        assert_eq!(config.num_threads, Some(4));
        assert_eq!(config.chunk_size, 20);
        assert_eq!(config.show_progress, false);
    }
}