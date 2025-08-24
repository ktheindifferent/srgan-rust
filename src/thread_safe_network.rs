use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use alumina::graph::{GraphDef, Result as GraphResult};
use ndarray::ArrayD;
use image::DynamicImage;
use crate::error::{Result, SrganError};
use crate::{NetworkDescription, inference_sr_net, image_to_data, data_to_image};

/// Immutable network weights that can be shared across threads
#[derive(Clone, Debug)]
pub struct NetworkWeights {
    /// Network parameters (weights and biases)
    parameters: Arc<Vec<ArrayD<f32>>>,
    /// Network configuration
    factor: u32,
    width: u32,
    log_depth: u32,
    global_node_factor: u32,
    /// Display name for the network
    display: String,
}

impl NetworkWeights {
    /// Create new network weights from a NetworkDescription
    pub fn new(desc: NetworkDescription, display: &str) -> Self {
        NetworkWeights {
            parameters: Arc::new(desc.parameters),
            factor: desc.factor,
            width: desc.width,
            log_depth: desc.log_depth,
            global_node_factor: desc.global_node_factor,
            display: display.to_string(),
        }
    }

    /// Get a reference to the parameters
    pub fn parameters(&self) -> &[ArrayD<f32>] {
        &self.parameters
    }

    /// Create a new graph definition for this network
    pub fn create_graph(&self) -> GraphResult<GraphDef> {
        inference_sr_net(
            self.factor as usize,
            self.width,
            self.log_depth,
            self.global_node_factor as usize,
        )
    }
}

/// Per-thread computation buffer for network inference
struct ComputeBuffer {
    /// Graph definition (can be reused per thread)
    graph: GraphDef,
}

impl ComputeBuffer {
    /// Create a new compute buffer for the given weights
    fn new(weights: &NetworkWeights) -> GraphResult<Self> {
        let graph = weights.create_graph()?;
        Ok(ComputeBuffer {
            graph,
        })
    }

    /// Execute inference on the given input
    fn execute(&mut self, input: ArrayD<f32>, weights: &NetworkWeights) -> GraphResult<ArrayD<f32>> {
        // Prepare input vector with image and parameters
        let mut input_vec = vec![input];
        input_vec.extend(weights.parameters().iter().cloned());
        
        // Get node IDs
        let input_id = self.graph.node_id("input").value_id();
        let param_ids: Vec<_> = self.graph.parameter_ids()
            .iter()
            .map(|node_id| node_id.value_id())
            .collect();
        let mut subgraph_inputs = vec![input_id];
        subgraph_inputs.extend(param_ids);
        let output_id = self.graph.node_id("output").value_id();
        
        // Create subgraph for execution
        let mut subgraph = self.graph.subgraph(&subgraph_inputs, &[output_id.clone()])?;
        
        // Execute the subgraph
        let result = subgraph.execute(input_vec)?;
        
        // Extract output
        Ok(result.into_map().remove(&output_id).unwrap())
    }
}

/// Thread-safe network wrapper that allows concurrent inference
pub struct ThreadSafeNetwork {
    /// Immutable network weights shared across threads
    weights: Arc<NetworkWeights>,
    /// Pool of computation buffers indexed by thread ID
    buffer_pool: Arc<Mutex<HashMap<std::thread::ThreadId, ComputeBuffer>>>,
}

impl ThreadSafeNetwork {
    /// Create a new thread-safe network from a NetworkDescription
    pub fn new(desc: NetworkDescription, display: &str) -> Result<Self> {
        let weights = Arc::new(NetworkWeights::new(desc, display));
        Ok(ThreadSafeNetwork {
            weights,
            buffer_pool: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create network from built-in natural model
    pub fn load_builtin_natural() -> Result<Self> {
        let data = crate::L1_SRGB_NATURAL_PARAMS;
        let desc = crate::network_from_bytes(data)
            .map_err(|e| SrganError::Network(e))?;
        Self::new(desc, "neural net trained on natural images with an L1 loss")
    }

    /// Create network from built-in anime model
    pub fn load_builtin_anime() -> Result<Self> {
        let data = crate::L1_SRGB_ANIME_PARAMS;
        let desc = crate::network_from_bytes(data)
            .map_err(|e| SrganError::Network(e))?;
        Self::new(desc, "neural net trained on animation images with an L1 loss")
    }

    /// Load network from file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)
            .map_err(|e| SrganError::Io(e))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| SrganError::Io(e))?;
            
        let desc = crate::network_from_bytes(&data)
            .map_err(|e| SrganError::Network(e))?;
            
        Self::new(desc, "custom network")
    }

    /// Load network from label
    pub fn from_label(label: &str, bilinear_factor: Option<usize>) -> Result<Self> {
        match label {
            "natural" => Self::load_builtin_natural(),
            "anime" => Self::load_builtin_anime(),
            "bilinear" => {
                // Bilinear needs special handling
                Self::new_bilinear(bilinear_factor.unwrap_or(4))
            },
            _ => Err(SrganError::Network(format!("Unsupported network type: {}", label))),
        }
    }
    
    /// Create a bilinear upscaling network
    fn new_bilinear(factor: usize) -> Result<Self> {
        // For bilinear, we create a network with minimal configuration
        let desc = NetworkDescription {
            factor: factor as u32,
            width: 12,  // Use default width
            log_depth: 1,  // Minimal depth
            global_node_factor: 0,
            parameters: Vec::new(),
        };
        Self::new(desc, "bilinear interpolation")
    }

    /// Process a tensor through the network
    pub fn process(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        let thread_id = std::thread::current().id();
        
        // Try to get existing buffer for this thread
        let needs_creation = {
            let pool = self.buffer_pool.lock().unwrap();
            !pool.contains_key(&thread_id)
        };
        
        // Create buffer if needed (outside of lock to minimize contention)
        if needs_creation {
            let new_buffer = ComputeBuffer::new(&self.weights)
                .map_err(|e| SrganError::GraphExecution(e.to_string()))?;
            let mut pool = self.buffer_pool.lock().unwrap();
            pool.insert(thread_id, new_buffer);
        }
        
        // Execute inference
        let mut pool = self.buffer_pool.lock().unwrap();
        let buffer = pool.get_mut(&thread_id).unwrap();
        buffer.execute(input, &self.weights)
            .map_err(|e| SrganError::GraphExecution(e.to_string()))
    }

    /// Upscale an image
    pub fn upscale_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        // Convert image to tensor
        let tensor = image_to_data(img);
        let shape = tensor.shape().to_vec();
        let input = tensor.into_shape(ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]]))
            .map_err(|_| SrganError::InvalidInput("Failed to reshape image tensor".to_string()))?;
        
        // Process through network
        let output = self.process(input)?;
        
        // Convert back to image
        let upscaled_img = data_to_image(output.subview(ndarray::Axis(0), 0));
        Ok(upscaled_img)
    }

    /// Get the display name of the network
    pub fn display(&self) -> &str {
        &self.weights.display
    }

    /// Get the upscaling factor
    pub fn factor(&self) -> u32 {
        self.weights.factor
    }
}

// Implement Send and Sync for ThreadSafeNetwork
unsafe impl Send for ThreadSafeNetwork {}
unsafe impl Sync for ThreadSafeNetwork {}

impl std::fmt::Display for ThreadSafeNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.weights.display)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;

    #[test]
    fn test_thread_safe_network_creation() {
        let network = ThreadSafeNetwork::load_builtin_natural().unwrap();
        assert_eq!(network.factor(), 4);
    }

    #[test]
    fn test_concurrent_inference() {
        let network = Arc::new(ThreadSafeNetwork::load_builtin_natural().unwrap());
        let mut handles = vec![];

        // Spawn multiple threads to perform inference concurrently
        for i in 0..4 {
            let network_clone = Arc::clone(&network);
            let handle = thread::spawn(move || {
                // Create a small test image tensor
                let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
                let result = network_clone.process(input);
                assert!(result.is_ok(), "Thread {} failed: {:?}", i, result);
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_no_mutex_required() {
        // This test verifies that the network can be used without wrapping in Arc<Mutex<>>
        let network = ThreadSafeNetwork::load_builtin_natural().unwrap();
        let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
        
        // Multiple sequential calls should work without mutex
        let result1 = network.process(input.clone());
        let result2 = network.process(input);
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }
}