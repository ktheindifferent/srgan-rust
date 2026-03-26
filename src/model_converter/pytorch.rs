use std::collections::HashMap;
use std::convert::TryInto;
use std::io::{Read, Cursor};
use std::path::Path;
use std::fs::File;
use log::{info, warn, debug};
use serde_pickle::{DeOptions, HashableValue, Value};
use num_traits::ToPrimitive;
use zip::ZipArchive;
use crate::error::SrganError;
use super::common::{TensorData, ModelInfo, tensor_statistics, DataType};

/// PyTorch model parser with enhanced capabilities
pub struct PyTorchParser {
    state_dict: HashMap<String, Value>,
    metadata: HashMap<String, String>,
    model_info: ModelInfo,
    is_torchscript: bool,
    is_compressed: bool,
}

impl PyTorchParser {
    pub fn new() -> Self {
        Self {
            state_dict: HashMap::new(),
            metadata: HashMap::new(),
            model_info: ModelInfo::default(),
            is_torchscript: false,
            is_compressed: false,
        }
    }

    /// Load a PyTorch model from file
    pub fn load_model(&mut self, path: &Path) -> Result<(), SrganError> {
        if !path.exists() {
            return Err(SrganError::FileNotFound(path.to_path_buf()));
        }

        let mut file = File::open(path)
            .map_err(|e| SrganError::Io(e))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| SrganError::Io(e))?;

        // Check file format
        if self.is_zip_format(&buffer) {
            debug!("Detected ZIP-based PyTorch model");
            self.load_zip_model(&buffer)?;
        } else if self.is_torchscript_format(&buffer) {
            debug!("Detected TorchScript model");
            self.load_torchscript_model(&buffer)?;
        } else {
            debug!("Detected pickle-based PyTorch model");
            self.load_pickle_model(&buffer)?;
        }

        // Analyze model structure
        self.analyze_model_structure()?;
        
        info!("Successfully loaded PyTorch model from {:?}", path);
        Ok(())
    }

    /// Check if data is ZIP format
    fn is_zip_format(&self, data: &[u8]) -> bool {
        data.len() >= 4 && data.starts_with(&[0x50, 0x4B, 0x03, 0x04])
    }

    /// Check if data is TorchScript format
    fn is_torchscript_format(&self, data: &[u8]) -> bool {
        // TorchScript models are ZIP files with specific structure
        if !self.is_zip_format(data) {
            return false;
        }
        
        // Check for TorchScript markers
        let cursor = Cursor::new(data);
        if let Ok(mut archive) = ZipArchive::new(cursor) {
            // TorchScript files contain constants.pkl and code/
            for i in 0..archive.len() {
                if let Ok(file) = archive.by_index(i) {
                    if file.name() == "constants.pkl" || file.name().starts_with("code/") {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Load ZIP-based PyTorch model
    fn load_zip_model(&mut self, data: &[u8]) -> Result<(), SrganError> {
        self.is_compressed = true;
        let cursor = Cursor::new(data);
        let mut archive = ZipArchive::new(cursor)
            .map_err(|e| SrganError::Parse(format!("Failed to open ZIP archive: {}", e)))?;

        // Look for the main data file (usually data.pkl)
        let mut model_data = Vec::new();
        let mut found_model = false;
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| SrganError::Parse(format!("Failed to read ZIP entry: {}", e)))?;
            
            let file_name = file.name().to_string();
            debug!("Found file in ZIP: {}", file_name);
            
            if file_name == "data.pkl" || file_name.ends_with("/data.pkl") {
                file.read_to_end(&mut model_data)
                    .map_err(|e| SrganError::Io(e))?;
                found_model = true;
                break;
            }
        }

        if !found_model {
            // Try to find any .pkl file
            let mut f = archive.by_index(0)
                .map_err(|e| SrganError::Parse(format!("No model data found in ZIP: {}", e)))?;
            f.read_to_end(&mut model_data)
                .map_err(|e| SrganError::Parse(format!("Failed to read model data: {}", e)))?;
        }

        // Parse the extracted pickle data
        self.parse_pickle_data(&model_data)?;
        
        // Also extract metadata if available
        self.extract_zip_metadata(&mut archive)?;
        
        Ok(())
    }

    /// Load TorchScript model
    fn load_torchscript_model(&mut self, data: &[u8]) -> Result<(), SrganError> {
        self.is_torchscript = true;
        self.is_compressed = true;
        
        let cursor = Cursor::new(data);
        let mut archive = ZipArchive::new(cursor)
            .map_err(|e| SrganError::Parse(format!("Failed to open TorchScript archive: {}", e)))?;

        // TorchScript structure:
        // - constants.pkl: contains tensor data
        // - code/: contains model code
        // - data/: contains additional data files
        
        // Extract constants (weights)
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| SrganError::Parse(format!("Failed to read TorchScript entry: {}", e)))?;
            
            let file_name = file.name().to_string();
            
            if file_name == "constants.pkl" {
                let mut constants_data = Vec::new();
                file.read_to_end(&mut constants_data)
                    .map_err(|e| SrganError::Io(e))?;
                
                self.parse_torchscript_constants(&constants_data)?;
            } else if file_name.starts_with("data/") && file_name.ends_with(".pkl") {
                // Additional weight files
                let mut weight_data = Vec::new();
                file.read_to_end(&mut weight_data)
                    .map_err(|e| SrganError::Io(e))?;
                
                let layer_name = file_name
                    .strip_prefix("data/")
                    .and_then(|n| n.strip_suffix(".pkl"))
                    .unwrap_or("unknown");
                
                self.parse_weight_file(&weight_data, layer_name)?;
            }
        }

        if self.state_dict.is_empty() {
            return Err(SrganError::Parse("No weights found in TorchScript model".into()));
        }

        Ok(())
    }

    /// Load pickle-based PyTorch model
    fn load_pickle_model(&mut self, data: &[u8]) -> Result<(), SrganError> {
        self.parse_pickle_data(data)?;
        Ok(())
    }

    /// Parse pickle data
    fn parse_pickle_data(&mut self, data: &[u8]) -> Result<(), SrganError> {
        let de_options = DeOptions::new();
        let value = serde_pickle::from_slice(data, de_options)
            .map_err(|e| SrganError::Parse(format!("Failed to parse pickle: {}", e)))?;
        
        self.state_dict = self.extract_state_dict(value)?;
        
        if self.state_dict.is_empty() {
            return Err(SrganError::Parse("No parameters found in model".into()));
        }
        
        Ok(())
    }

    /// Parse TorchScript constants file
    fn parse_torchscript_constants(&mut self, data: &[u8]) -> Result<(), SrganError> {
        let de_options = DeOptions::new();
        let value = serde_pickle::from_slice(data, de_options)
            .map_err(|e| SrganError::Parse(format!("Failed to parse constants: {}", e)))?;
        
        // TorchScript constants are typically stored as a list of tensors
        match value {
            Value::List(tensors) => {
                for (idx, tensor) in tensors.into_iter().enumerate() {
                    let key = format!("constant_{}", idx);
                    self.state_dict.insert(key, tensor);
                }
            }
            Value::Dict(_) => {
                self.state_dict = self.extract_state_dict(value)?;
            }
            _ => {
                return Err(SrganError::Parse("Unexpected TorchScript constants format".into()));
            }
        }
        
        Ok(())
    }

    /// Parse individual weight file
    fn parse_weight_file(&mut self, data: &[u8], layer_name: &str) -> Result<(), SrganError> {
        let de_options = DeOptions::new();
        let value = serde_pickle::from_slice(data, de_options)
            .map_err(|e| SrganError::Parse(format!("Failed to parse weight file: {}", e)))?;
        
        self.state_dict.insert(layer_name.to_string(), value);
        Ok(())
    }

    /// Extract state dict from pickle value
    fn extract_state_dict(&self, value: Value) -> Result<HashMap<String, Value>, SrganError> {
        match value {
            Value::Dict(dict) => {
                let mut state_dict = HashMap::new();
                for (key, val) in dict {
                    let key_str = self.hashable_to_string(key)?;
                    
                    // Check if this is a nested state_dict
                    if key_str == "state_dict" {
                        if let Value::Dict(inner_dict) = val {
                            for (inner_key, inner_val) in inner_dict {
                                let inner_key_str = self.hashable_to_string(inner_key)?;
                                state_dict.insert(inner_key_str, inner_val);
                            }
                        } else {
                            state_dict.insert(key_str, val);
                        }
                    } else {
                        state_dict.insert(key_str, val);
                    }
                }
                Ok(state_dict)
            }
            Value::List(list) => {
                // Some models store state as list of tuples
                let mut state_dict = HashMap::new();
                for item in list {
                    if let Value::Tuple(tuple) = item {
                        if tuple.len() == 2 {
                            let key = self.value_to_string(&tuple[0])?;
                            state_dict.insert(key, tuple[1].clone());
                        }
                    }
                }
                Ok(state_dict)
            }
            _ => Err(SrganError::Parse("Unexpected model structure".into()))
        }
    }

    /// Convert HashableValue to String
    fn hashable_to_string(&self, value: HashableValue) -> Result<String, SrganError> {
        match value {
            HashableValue::String(s) => Ok(s),
            HashableValue::Bytes(b) => String::from_utf8(b)
                .map_err(|e| SrganError::Parse(format!("Invalid UTF-8: {}", e))),
            _ => Ok(format!("{:?}", value)),
        }
    }

    /// Convert Value to String
    fn value_to_string(&self, value: &Value) -> Result<String, SrganError> {
        match value {
            Value::String(s) => Ok(s.clone()),
            Value::Bytes(b) => String::from_utf8(b.clone())
                .map_err(|e| SrganError::Parse(format!("Invalid UTF-8: {}", e))),
            _ => Ok(format!("{:?}", value)),
        }
    }

    /// Extract metadata from ZIP archive
    fn extract_zip_metadata(&mut self, archive: &mut ZipArchive<Cursor<&[u8]>>) -> Result<(), SrganError> {
        // Look for version file or metadata
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| SrganError::Parse(format!("Failed to read metadata: {}", e)))?;
            
            let file_name = file.name().to_string();
            
            if file_name == "version" || file_name == "pytorch_version" {
                let mut version_data = String::new();
                file.read_to_string(&mut version_data)
                    .map_err(|e| SrganError::Io(e))?;
                self.metadata.insert("pytorch_version".into(), version_data.trim().to_string());
            } else if file_name == "model_info.json" {
                let mut info_data = String::new();
                file.read_to_string(&mut info_data)
                    .map_err(|e| SrganError::Io(e))?;
                
                // Parse JSON metadata
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&info_data) {
                    if let Some(obj) = json.as_object() {
                        for (key, val) in obj {
                            if let Some(str_val) = val.as_str() {
                                self.metadata.insert(key.clone(), str_val.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Analyze model structure and populate model info
    fn analyze_model_structure(&mut self) -> Result<(), SrganError> {
        // Detect architecture
        self.model_info.architecture_hints = self.detect_architecture_hints();
        
        // Infer shapes
        self.model_info.input_shape = self.infer_input_shape();
        self.model_info.output_shape = self.infer_output_shape();
        
        // Set version
        self.model_info.version = self.metadata.get("pytorch_version")
            .cloned()
            .unwrap_or_else(|| {
                if self.is_torchscript {
                    "torchscript".to_string()
                } else if self.is_compressed {
                    "1.6+".to_string()
                } else {
                    "1.0+".to_string()
                }
            });
        
        // Count parameters
        let total_params: usize = self.state_dict.values()
            .filter_map(|v| self.get_tensor_size(v).ok())
            .sum();
        
        info!("Model analysis complete: {} parameters, architecture: {:?}", 
              total_params, self.model_info.architecture_hints);
        
        Ok(())
    }

    /// Detect architecture hints from layer names
    fn detect_architecture_hints(&self) -> Vec<String> {
        let mut hints = Vec::new();
        let keys: Vec<String> = self.state_dict.keys().cloned().collect();
        
        // Check for specific architectures
        if keys.iter().any(|k| k.contains("generator")) {
            hints.push("GAN".into());
        }
        if keys.iter().any(|k| k.contains("discriminator")) {
            hints.push("GAN".into());
        }
        if keys.iter().any(|k| k.contains("RRDB") || k.contains("rrdb")) {
            hints.push("ESRGAN".into());
        }
        if keys.iter().any(|k| k.contains("residual") || k.contains("res")) {
            hints.push("ResNet".into());
        }
        if keys.iter().any(|k| k.contains("dense") || k.contains("RDB")) {
            hints.push("DenseNet".into());
        }
        if keys.iter().any(|k| k.contains("upsample") || k.contains("pixelshuffle")) {
            hints.push("super-resolution".into());
        }
        
        // Architecture-specific patterns
        if keys.iter().any(|k| k.contains("conv_first") && k.contains("trunk_conv")) {
            hints.push("ESRGAN".into());
        } else if keys.iter().any(|k| k.starts_with("generator.initial")) {
            hints.push("SRGAN".into());
        }
        
        hints
    }

    /// Infer input shape from first layer
    fn infer_input_shape(&self) -> Vec<usize> {
        // Look for first conv layer
        let first_conv_candidates = [
            "conv_first.weight",
            "conv1.weight", 
            "features.0.weight",
            "initial.0.weight",
            "generator.initial.0.weight",
            "module.conv_first.weight", // For DataParallel models
        ];
        
        for candidate in &first_conv_candidates {
            if let Some(tensor) = self.state_dict.get(*candidate) {
                if let Ok(shape) = self.get_tensor_shape(tensor) {
                    if shape.len() == 4 {
                        // Conv weight: [out_channels, in_channels, height, width]
                        let in_channels = shape[1];
                        return vec![1, in_channels, 256, 256]; // Default spatial size
                    }
                }
            }
        }
        
        // Default for super-resolution
        vec![1, 3, 256, 256]
    }

    /// Infer output shape from model structure
    fn infer_output_shape(&self) -> Vec<usize> {
        let input_shape = self.infer_input_shape();
        let mut scale_factor = 4; // Default for SRGAN
        
        // Count upsampling layers
        let upsample_count = self.state_dict.keys()
            .filter(|k| k.contains("upsample") || k.contains("pixelshuffle"))
            .count();
        
        if upsample_count > 0 {
            scale_factor = 2_usize.pow(upsample_count as u32);
        }
        
        vec![
            input_shape[0],
            3, // RGB output
            input_shape[2] * scale_factor,
            input_shape[3] * scale_factor,
        ]
    }

    /// Get tensor shape from Value
    fn get_tensor_shape(&self, value: &Value) -> Result<Vec<usize>, SrganError> {
        match value {
            Value::Dict(dict) => {
                // Look for shape metadata
                let shape_keys = ["shape", "size", "_shape", "_size"];
                for key in shape_keys {
                    let hashable_key = HashableValue::String(key.into());
                    if let Some(shape_val) = dict.get(&hashable_key) {
                        return self.extract_shape_from_value(shape_val);
                    }
                }
                Err(SrganError::Parse("No shape information found".into()))
            }
            _ => Err(SrganError::Parse("Cannot extract shape from value".into()))
        }
    }

    /// Extract shape from Value
    fn extract_shape_from_value(&self, value: &Value) -> Result<Vec<usize>, SrganError> {
        match value {
            Value::List(list) => {
                list.iter()
                    .map(|v| match v {
                        Value::I64(i) => Ok(*i as usize),
                        _ => Err(SrganError::Parse("Invalid shape element".into()))
                    })
                    .collect()
            }
            Value::Tuple(tuple) => {
                tuple.iter()
                    .map(|v| match v {
                        Value::I64(i) => Ok(*i as usize),
                        _ => Err(SrganError::Parse("Invalid shape element".into()))
                    })
                    .collect()
            }
            _ => Err(SrganError::Parse("Invalid shape format".into()))
        }
    }

    /// Get tensor size
    fn get_tensor_size(&self, value: &Value) -> Result<usize, SrganError> {
        match value {
            Value::List(list) => Ok(list.len()),
            Value::Bytes(bytes) => Ok(bytes.len() / 4), // Assume float32
            Value::Dict(_) => {
                // Try to extract tensor data and count elements
                if let Ok(data) = self.extract_tensor_data(value.clone()) {
                    Ok(data.len())
                } else {
                    Ok(0)
                }
            }
            _ => Ok(0)
        }
    }

    /// Extract weights from parsed model
    pub fn extract_weights(&mut self) -> Result<HashMap<String, TensorData>, SrganError> {
        let mut weights = HashMap::new();
        
        for (name, value) in &self.state_dict {
            match self.extract_tensor_data(value.clone()) {
                Ok(data) => {
                    if !data.is_empty() {
                        // Try to get shape information
                        let shape = self.get_tensor_shape(value).unwrap_or_else(|_| {
                            // Infer shape from data length and layer name
                            self.infer_tensor_shape(&name, data.len())
                        });
                        
                        let tensor = TensorData {
                            name: name.clone(),
                            data,
                            shape,
                            dtype: DataType::Float32,
                        };
                        
                        // Validate tensor
                        let stats = tensor_statistics(&tensor.data);
                        if stats.min.is_finite() && stats.max.is_finite() {
                            weights.insert(name.clone(), tensor);
                        } else {
                            warn!("Skipping tensor {} with invalid values", name);
                        }
                    }
                }
                Err(e) => {
                    debug!("Failed to extract tensor {}: {}", name, e);
                }
            }
        }
        
        if weights.is_empty() {
            return Err(SrganError::Parse("No valid weights extracted".into()));
        }
        
        info!("Extracted {} weight tensors from PyTorch model", weights.len());
        Ok(weights)
    }

    /// Infer tensor shape from name and size
    fn infer_tensor_shape(&self, name: &str, size: usize) -> Vec<usize> {
        // Common patterns
        if name.contains("weight") && name.contains("conv") {
            // Try common conv shapes
            if size == 64 * 3 * 9 * 9 { return vec![64, 3, 9, 9]; }
            if size == 64 * 64 * 3 * 3 { return vec![64, 64, 3, 3]; }
            if size == 256 * 64 * 3 * 3 { return vec![256, 64, 3, 3]; }
        } else if name.contains("bias") {
            return vec![size];
        } else if name.contains("bn") || name.contains("batch_norm") {
            return vec![size];
        }
        
        // Default to 1D
        vec![size]
    }

    /// Extract tensor data from Value (enhanced version)
    fn extract_tensor_data(&self, value: Value) -> Result<Vec<f32>, SrganError> {
        match value {
            Value::List(list) => {
                let mut weights = Vec::new();
                for item in list {
                    weights.push(self.value_to_f32(item)?);
                }
                Ok(weights)
            }
            Value::Bytes(bytes) => {
                self.parse_tensor_bytes(&bytes)
            }
            Value::Dict(dict) => {
                self.extract_tensor_from_dict(dict)
            }
            Value::Tuple(tuple) => {
                // Try each element
                for item in tuple {
                    if let Ok(data) = self.extract_tensor_data(item) {
                        if !data.is_empty() {
                            return Ok(data);
                        }
                    }
                }
                Err(SrganError::Parse("No tensor data in tuple".into()))
            }
            _ => Err(SrganError::Parse("Unsupported tensor format".into()))
        }
    }

    /// Parse tensor bytes with multiple dtype support
    fn parse_tensor_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>, SrganError> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        // Try different interpretations based on byte alignment
        if bytes.len() % 4 == 0 {
            // Float32
            let mut weights = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                let val = f32::from_le_bytes(arr);
                if val.is_finite() || (weights.len() < 10 && !val.is_finite()) {
                    weights.push(val);
                } else if !val.is_finite() {
                    // Try big-endian
                    let val_be = f32::from_be_bytes(arr);
                    weights.push(val_be);
                }
            }
            Ok(weights)
        } else if bytes.len() % 8 == 0 {
            // Float64
            let mut weights = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks_exact(8) {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                weights.push(f64::from_le_bytes(arr) as f32);
            }
            Ok(weights)
        } else if bytes.len() % 2 == 0 {
            // Float16
            let mut weights = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let arr: [u8; 2] = chunk.try_into().unwrap();
                weights.push(self.float16_to_float32(u16::from_le_bytes(arr)));
            }
            Ok(weights)
        } else {
            // Int8 quantized
            let mut weights = Vec::with_capacity(bytes.len());
            for &byte in bytes {
                let int_val = byte as i8;
                weights.push(int_val as f32 / 127.0);
            }
            Ok(weights)
        }
    }

    /// Convert float16 to float32
    fn float16_to_float32(&self, half: u16) -> f32 {
        let sign = (half >> 15) & 1;
        let exp = (half >> 10) & 0x1F;
        let frac = half & 0x3FF;
        
        if exp == 0 {
            if frac == 0 {
                if sign == 1 { -0.0 } else { 0.0 }
            } else {
                let result = (frac as f32 / 1024.0) * 2.0_f32.powi(-14);
                if sign == 1 { -result } else { result }
            }
        } else if exp == 0x1F {
            if frac == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            let exp_f32 = (exp as i32) - 15;
            let frac_f32 = 1.0 + (frac as f32 / 1024.0);
            let result = frac_f32 * 2.0_f32.powi(exp_f32);
            if sign == 1 { -result } else { result }
        }
    }

    /// Extract tensor from dictionary
    fn extract_tensor_from_dict(&self, dict: std::collections::BTreeMap<HashableValue, Value>) -> Result<Vec<f32>, SrganError> {
        // Check for PyTorch tensor structure
        let storage_keys = ["storage", "_storage", "data", "_data"];
        
        for key in storage_keys {
            if let Some((_, storage)) = dict.iter().find(|(k, _)| {
                matches!(k, HashableValue::String(s) if s.as_str() == key)
            }) {
                if let Ok(data) = self.extract_tensor_data(storage.clone()) {
                    if !data.is_empty() {
                        return Ok(data);
                    }
                }
            }
        }

        // Check for typed storage
        for (key, val) in &dict {
            if let HashableValue::String(key_str) = key {
                if key_str.contains("Storage") {
                    if let Ok(data) = self.extract_tensor_data(val.clone()) {
                        if !data.is_empty() {
                            return Ok(data);
                        }
                    }
                }
            }
        }

        Err(SrganError::Parse("No tensor data in dict".into()))
    }

    /// Convert value to f32
    fn value_to_f32(&self, value: Value) -> Result<f32, SrganError> {
        match value {
            Value::F64(d) => Ok(d as f32),
            Value::I64(i) => Ok(i as f32),
            Value::Int(i) => {
                i.to_f32()
                    .ok_or_else(|| SrganError::Parse("Integer too large".into()))
            }
            _ => Err(SrganError::Parse("Cannot convert to f32".into()))
        }
    }

    /// Get model information
    pub fn get_model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }

    /// Validate model integrity
    pub fn validate_model(&self) -> Result<(), SrganError> {
        if self.state_dict.is_empty() {
            return Err(SrganError::Parse("Model has no parameters".into()));
        }

        // Check for required layers for super-resolution
        let has_conv = self.state_dict.keys()
            .any(|k| k.contains("conv"));
        
        if !has_conv {
            warn!("Model appears to have no convolutional layers");
        }

        // Validate tensor integrity
        let mut invalid_tensors = Vec::new();
        for (name, value) in &self.state_dict {
            if let Ok(data) = self.extract_tensor_data(value.clone()) {
                if data.iter().any(|v| !v.is_finite()) {
                    invalid_tensors.push(name.clone());
                }
            }
        }

        if !invalid_tensors.is_empty() {
            warn!("Found {} tensors with NaN/Inf values: {:?}", 
                  invalid_tensors.len(), invalid_tensors);
        }

        Ok(())
    }
}

// Re-export for compatibility

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_pytorch_parser_creation() {
        let parser = PyTorchParser::new();
        assert!(parser.state_dict.is_empty());
        assert!(!parser.is_torchscript);
        assert!(!parser.is_compressed);
    }

    #[test]
    fn test_zip_format_detection() {
        let parser = PyTorchParser::new();
        
        // ZIP magic bytes
        let zip_data = vec![0x50, 0x4B, 0x03, 0x04, 0x00, 0x00];
        assert!(parser.is_zip_format(&zip_data));
        
        // Non-ZIP data
        let pickle_data = vec![0x80, 0x02, 0x00, 0x00];
        assert!(!parser.is_zip_format(&pickle_data));
    }

    #[test]
    fn test_float16_conversion() {
        let parser = PyTorchParser::new();
        
        // Test zero
        assert_eq!(parser.float16_to_float32(0x0000), 0.0);
        assert_eq!(parser.float16_to_float32(0x8000), -0.0);
        
        // Test one
        assert_eq!(parser.float16_to_float32(0x3C00), 1.0);
        assert_eq!(parser.float16_to_float32(0xBC00), -1.0);
        
        // Test infinity
        assert_eq!(parser.float16_to_float32(0x7C00), f32::INFINITY);
        assert_eq!(parser.float16_to_float32(0xFC00), f32::NEG_INFINITY);
        
        // Test NaN
        assert!(parser.float16_to_float32(0x7C01).is_nan());
    }

    #[test]
    fn test_tensor_shape_inference() {
        let parser = PyTorchParser::new();
        
        // Conv weight
        let shape = parser.infer_tensor_shape("conv1.weight", 64 * 3 * 9 * 9);
        assert_eq!(shape, vec![64, 3, 9, 9]);
        
        // Bias
        let shape = parser.infer_tensor_shape("conv1.bias", 64);
        assert_eq!(shape, vec![64]);
        
        // Unknown
        let shape = parser.infer_tensor_shape("unknown", 1000);
        assert_eq!(shape, vec![1000]);
    }

    #[test]
    fn test_architecture_detection() {
        let mut parser = PyTorchParser::new();
        
        // ESRGAN
        parser.state_dict.insert("conv_first.weight".into(), Value::None);
        parser.state_dict.insert("trunk_conv.weight".into(), Value::None);
        parser.state_dict.insert("RRDB.0.conv1.weight".into(), Value::None);
        
        let hints = parser.detect_architecture_hints();
        assert!(hints.contains(&"ESRGAN".to_string()));
        
        // SRGAN
        parser.state_dict.clear();
        parser.state_dict.insert("generator.initial.0.weight".into(), Value::None);
        parser.state_dict.insert("discriminator.conv1.weight".into(), Value::None);
        
        let hints = parser.detect_architecture_hints();
        assert!(hints.contains(&"GAN".to_string()));
        assert!(hints.contains(&"SRGAN".to_string()));
    }

    #[test]
    fn test_model_validation() {
        let mut parser = PyTorchParser::new();
        
        // Empty model should fail
        assert!(parser.validate_model().is_err());
        
        // Add valid tensor
        parser.state_dict.insert("conv1.weight".into(), Value::List(vec![Value::F64(1.0)]));
        assert!(parser.validate_model().is_ok());
    }
}