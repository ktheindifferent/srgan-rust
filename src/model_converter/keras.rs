use std::collections::HashMap;
use std::path::Path;
use log::{info, warn, debug};

#[cfg(feature = "keras-support")]
use hdf5::{File as H5File, Group, Dataset, Result as H5Result};
#[cfg(feature = "keras-support")]
use ndarray::{ArrayD, IxDyn};

use crate::error::SrganError;
use super::common::{TensorData, DataType, WeightExtractor, ModelInfo, validate_tensor_data, convert_nhwc_to_nchw};

pub struct KerasParser {
    layers: HashMap<String, LayerInfo>,
    weights: HashMap<String, TensorData>,
    model_info: ModelInfo,
    model_config: Option<ModelConfig>,
}

#[derive(Debug, Clone)]
struct LayerInfo {
    name: String,
    class_name: String,
    weights: Vec<String>,
    config: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct ModelConfig {
    name: String,
    layers: Vec<LayerConfig>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct LayerConfig {
    name: String,
    class_name: String,
    inbound_nodes: Vec<String>,
    config: serde_json::Value,
}

impl KerasParser {
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            weights: HashMap::new(),
            model_info: ModelInfo {
                format: "keras".into(),
                version: "2.0".into(),
                input_shape: vec![],
                output_shape: vec![],
                total_parameters: 0,
                architecture_hints: vec![],
            },
            model_config: None,
        }
    }
    
    pub fn load_h5_model(&mut self, path: &Path) -> Result<(), SrganError> {
        if !path.exists() {
            return Err(SrganError::FileNotFound(path.to_path_buf()));
        }
        
        // Open HDF5 file
        let file = H5File::open(path)
            .map_err(|e| SrganError::Parse(format!("Failed to open H5 file: {}", e)))?;
        
        // Check if it's a Keras model
        if !self.is_keras_model(&file) {
            return Err(SrganError::Parse("Not a valid Keras HDF5 model".into()));
        }
        
        // Load model configuration
        self.load_model_config(&file)?;
        
        // Load model weights
        self.load_model_weights(&file)?;
        
        // Analyze model structure
        self.analyze_model()?;
        
        info!("Successfully loaded Keras model from {:?}", path);
        Ok(())
    }
    
    fn is_keras_model(&self, file: &H5File) -> bool {
        // Check for Keras-specific attributes
        if let Ok(attrs) = file.attr("keras_version") {
            return true;
        }
        
        // Check for model_config attribute (Keras 2.x)
        if let Ok(_) = file.attr("model_config") {
            return true;
        }
        
        // Check for model_weights group
        if file.group("model_weights").is_ok() {
            return true;
        }
        
        false
    }
    
    fn load_model_config(&mut self, file: &H5File) -> Result<(), SrganError> {
        // Try to load model configuration from attributes
        if let Ok(config_attr) = file.attr("model_config") {
            if let Ok(config_str) = config_attr.read_scalar::<String>() {
                self.parse_model_config(&config_str)?;
            }
        }
        
        // Try to get Keras version
        if let Ok(version_attr) = file.attr("keras_version") {
            if let Ok(version) = version_attr.read_scalar::<String>() {
                self.model_info.version = version;
            }
        }
        
        // Try to get backend
        if let Ok(backend_attr) = file.attr("backend") {
            if let Ok(backend) = backend_attr.read_scalar::<String>() {
                self.model_info.architecture_hints.push(format!("Backend: {}", backend));
            }
        }
        
        Ok(())
    }
    
    fn parse_model_config(&mut self, config_str: &str) -> Result<(), SrganError> {
        let config_json: serde_json::Value = serde_json::from_str(config_str)
            .map_err(|e| SrganError::Parse(format!("Failed to parse model config: {}", e)))?;
        
        // Extract model information
        if let Some(config) = config_json.get("config") {
            if let Some(name) = config.get("name").and_then(|v| v.as_str()) {
                self.model_info.architecture_hints.push(format!("Model name: {}", name));
            }
            
            // Parse layers
            if let Some(layers) = config.get("layers").and_then(|v| v.as_array()) {
                for layer in layers {
                    self.parse_layer_config(layer)?;
                }
            }
            
            // Try to extract input/output shapes
            if let Some(input_layers) = config.get("input_layers").and_then(|v| v.as_array()) {
                if let Some(first_input) = input_layers.first() {
                    // Extract shape from input specification
                    self.extract_input_shape(first_input);
                }
            }
        }
        
        Ok(())
    }
    
    fn parse_layer_config(&mut self, layer_json: &serde_json::Value) -> Result<(), SrganError> {
        let class_name = layer_json.get("class_name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        
        let config = layer_json.get("config").unwrap_or(&serde_json::Value::Null);
        
        let name = config.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed");
        
        let mut layer_info = LayerInfo {
            name: name.to_string(),
            class_name: class_name.to_string(),
            weights: Vec::new(),
            config: HashMap::new(),
        };
        
        // Extract relevant configuration
        if let Some(filters) = config.get("filters").and_then(|v| v.as_u64()) {
            layer_info.config.insert("filters".into(), filters.to_string());
        }
        
        if let Some(kernel_size) = config.get("kernel_size") {
            layer_info.config.insert("kernel_size".into(), kernel_size.to_string());
        }
        
        if let Some(activation) = config.get("activation").and_then(|v| v.as_str()) {
            layer_info.config.insert("activation".into(), activation.to_string());
        }
        
        // Add architecture hints based on layer type
        match class_name {
            "Conv2D" => {
                self.model_info.architecture_hints.push(format!("Conv2D layer: {}", name));
            }
            "Conv2DTranspose" | "UpSampling2D" => {
                self.model_info.architecture_hints.push("Upsampling layer detected".into());
            }
            "Add" | "Concatenate" if name.contains("res") => {
                self.model_info.architecture_hints.push("Residual connection detected".into());
            }
            _ => {}
        }
        
        self.layers.insert(name.to_string(), layer_info);
        Ok(())
    }
    
    fn extract_input_shape(&mut self, input_spec: &serde_json::Value) {
        // Try to extract shape from various possible formats
        if let Some(shape_array) = input_spec.as_array() {
            if shape_array.len() >= 2 {
                if let Some(shape_spec) = shape_array.get(2) {
                    if let Some(shape) = shape_spec.as_array() {
                        let dims: Vec<usize> = shape.iter()
                            .filter_map(|v| v.as_u64().map(|u| u as usize))
                            .collect();
                        if !dims.is_empty() {
                            self.model_info.input_shape = dims;
                        }
                    }
                }
            }
        }
    }
    
    fn load_model_weights(&mut self, file: &H5File) -> Result<(), SrganError> {
        // Try different weight storage patterns
        let weight_groups = vec![
            "model_weights",
            "weights",
            "/",  // Sometimes weights are at root
        ];
        
        let mut total_params = 0;
        
        for group_name in weight_groups {
            if let Ok(group) = file.group(group_name) {
                debug!("Loading weights from group: {}", group_name);
                total_params += self.load_weights_from_group(&group)?;
            }
        }
        
        self.model_info.total_parameters = total_params;
        info!("Loaded {} weight tensors with {} total parameters",
              self.weights.len(), total_params);
        
        Ok(())
    }
    
    fn load_weights_from_group(&mut self, group: &Group) -> Result<usize, SrganError> {
        let mut total_params = 0;
        
        // Iterate through all members of the group
        for name in group.member_names().unwrap_or_default() {
            // Check if it's a subgroup (layer)
            if let Ok(subgroup) = group.group(&name) {
                // This is a layer group, load its weights
                total_params += self.load_layer_weights(&subgroup, &name)?;
            } else if let Ok(dataset) = group.dataset(&name) {
                // Direct weight dataset
                if let Ok(tensor_data) = self.load_tensor_from_dataset(&dataset, &name) {
                    if validate_tensor_data(&tensor_data.data) {
                        total_params += tensor_data.data.len();
                        self.weights.insert(name.clone(), tensor_data);
                    }
                }
            }
        }
        
        Ok(total_params)
    }
    
    fn load_layer_weights(&mut self, layer_group: &Group, layer_name: &str) -> Result<usize, SrganError> {
        let mut total_params = 0;
        
        // Look for weight datasets within the layer
        let weight_names = vec!["kernel", "bias", "gamma", "beta", "moving_mean", "moving_variance"];
        
        for weight_name in weight_names {
            // Try with and without the layer name prefix
            let dataset_names = vec![
                weight_name.to_string(),
                format!("{}:0", weight_name),
                format!("{}/{}", layer_name, weight_name),
                format!("{}_{}", layer_name, weight_name),
            ];
            
            for dataset_name in dataset_names {
                if let Ok(dataset) = layer_group.dataset(&dataset_name) {
                    let full_name = format!("{}/{}", layer_name, weight_name);
                    
                    if let Ok(tensor_data) = self.load_tensor_from_dataset(&dataset, &full_name) {
                        if validate_tensor_data(&tensor_data.data) {
                            total_params += tensor_data.data.len();
                            self.weights.insert(full_name, tensor_data);
                            
                            // Update layer info
                            if let Some(layer_info) = self.layers.get_mut(layer_name) {
                                layer_info.weights.push(weight_name.to_string());
                            }
                        } else {
                            warn!("Tensor {}/{} contains invalid values", layer_name, weight_name);
                        }
                    }
                    break;  // Found this weight, move to next
                }
            }
        }
        
        Ok(total_params)
    }
    
    fn load_tensor_from_dataset(&self, dataset: &Dataset, name: &str) -> H5Result<TensorData> {
        // Get dataset shape
        let shape: Vec<usize> = dataset.shape().iter().cloned().collect();
        
        // Read data based on datatype
        let dtype = dataset.dtype()?;
        
        let data = if dtype.is::<f32>() {
            let array: ArrayD<f32> = dataset.read()?;
            array.into_raw_vec()
        } else if dtype.is::<f64>() {
            let array: ArrayD<f64> = dataset.read()?;
            array.iter().map(|&x| x as f32).collect()
        } else if dtype.is::<i32>() {
            let array: ArrayD<i32> = dataset.read()?;
            array.iter().map(|&x| x as f32).collect()
        } else {
            return Err(hdf5::Error::from("Unsupported data type"));
        };
        
        Ok(TensorData {
            name: name.to_string(),
            shape,
            data,
            dtype: DataType::Float32,
        })
    }
    
    fn analyze_model(&mut self) -> Result<(), SrganError> {
        // Count layer types
        let mut conv_count = 0;
        let mut dense_count = 0;
        let mut bn_count = 0;
        let mut activation_count = 0;
        let mut upsampling_count = 0;
        
        for layer_info in self.layers.values() {
            match layer_info.class_name.as_str() {
                "Conv2D" => conv_count += 1,
                "Dense" => dense_count += 1,
                "BatchNormalization" => bn_count += 1,
                "Activation" | "ReLU" | "LeakyReLU" | "PReLU" => activation_count += 1,
                "Conv2DTranspose" | "UpSampling2D" => upsampling_count += 1,
                _ => {}
            }
        }
        
        info!("Model structure: {} Conv2D, {} Dense, {} BN, {} Activation, {} Upsampling layers",
              conv_count, dense_count, bn_count, activation_count, upsampling_count);
        
        // Detect model type
        if conv_count > 10 && upsampling_count > 0 {
            self.model_info.architecture_hints.push("Likely super-resolution model".into());
        }
        
        if self.model_info.architecture_hints.iter()
            .any(|h| h.contains("Residual") || h.contains("res")) {
            self.model_info.architecture_hints.push("Contains residual connections".into());
        }
        
        // Try to infer input/output shapes from layer configurations
        if self.model_info.input_shape.is_empty() {
            self.infer_input_shape();
        }
        
        if self.model_info.output_shape.is_empty() {
            self.infer_output_shape();
        }
        
        Ok(())
    }
    
    fn infer_input_shape(&mut self) {
        // Look for InputLayer or first Conv2D layer
        for layer in self.layers.values() {
            if layer.class_name == "InputLayer" {
                // Try to extract shape from config
                if let Some(shape_str) = layer.config.get("batch_input_shape") {
                    // Parse shape string
                    debug!("Found input shape: {}", shape_str);
                }
            } else if layer.class_name == "Conv2D" && self.model_info.input_shape.is_empty() {
                // Default SRGAN input shape (without batch dimension)
                self.model_info.input_shape = vec![256, 256, 3];
                break;
            }
        }
        
        // Default if not found
        if self.model_info.input_shape.is_empty() {
            self.model_info.input_shape = vec![256, 256, 3];
        }
    }
    
    fn infer_output_shape(&mut self) {
        // For SRGAN, output is typically 4x the input
        if !self.model_info.input_shape.is_empty() {
            let h = self.model_info.input_shape[0];
            let w = self.model_info.input_shape[1];
            let c = self.model_info.input_shape.get(2).copied().unwrap_or(3);
            
            // Check upsampling factor
            let upsampling_count = self.layers.values()
                .filter(|l| l.class_name == "Conv2DTranspose" || l.class_name == "UpSampling2D")
                .count();
            
            let scale_factor = 2_usize.pow(upsampling_count as u32).max(4);
            self.model_info.output_shape = vec![h * scale_factor, w * scale_factor, c];
        }
    }
    
    pub fn convert_keras_weights_to_nchw(&mut self) -> Result<(), SrganError> {
        // Keras uses NHWC format by default, convert Conv2D weights to NCHW
        for (name, tensor) in &mut self.weights {
            if name.contains("kernel") && tensor.shape.len() == 4 {
                // Conv2D kernel: [height, width, in_channels, out_channels]
                // Need to transpose to: [out_channels, in_channels, height, width]
                let h = tensor.shape[0];
                let w = tensor.shape[1];
                let in_c = tensor.shape[2];
                let out_c = tensor.shape[3];
                
                let mut new_data = vec![0.0; tensor.data.len()];
                
                for out_ch in 0..out_c {
                    for in_ch in 0..in_c {
                        for row in 0..h {
                            for col in 0..w {
                                let keras_idx = row * w * in_c * out_c + col * in_c * out_c + in_ch * out_c + out_ch;
                                let nchw_idx = out_ch * in_c * h * w + in_ch * h * w + row * w + col;
                                new_data[nchw_idx] = tensor.data[keras_idx];
                            }
                        }
                    }
                }
                
                tensor.data = new_data;
                tensor.shape = vec![out_c, in_c, h, w];
            }
        }
        
        Ok(())
    }
}

impl WeightExtractor for KerasParser {
    fn extract_weights(&self) -> Result<HashMap<String, TensorData>, SrganError> {
        Ok(self.weights.clone())
    }
    
    fn get_layer_names(&self) -> Vec<String> {
        self.layers.keys().cloned().collect()
    }
    
    fn get_model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_keras_parser_creation() {
        let parser = KerasParser::new();
        assert!(parser.layers.is_empty());
        assert!(parser.weights.is_empty());
    }
    
    #[test]
    fn test_model_info_initialization() {
        let parser = KerasParser::new();
        let info = parser.get_model_info();
        assert_eq!(info.format, "keras");
        assert_eq!(info.version, "2.0");
    }
    
    #[test]
    fn test_layer_info_creation() {
        let layer = LayerInfo {
            name: "conv1".into(),
            class_name: "Conv2D".into(),
            weights: vec!["kernel".into(), "bias".into()],
            config: HashMap::new(),
        };
        
        assert_eq!(layer.name, "conv1");
        assert_eq!(layer.class_name, "Conv2D");
        assert_eq!(layer.weights.len(), 2);
    }
}