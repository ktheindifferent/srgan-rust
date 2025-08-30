use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::convert::TryInto;
use prost::Message;
use log::{info, warn, debug};

use crate::error::SrganError;
use crate::model_converter::common::{TensorData, DataType, WeightExtractor, ModelInfo, validate_tensor_data};

pub struct TensorFlowParser {
    graph_def: Option<GraphDef>,
    variables: HashMap<String, TensorData>,
    model_info: ModelInfo,
}

// Simplified GraphDef structure (would be generated from .proto in real implementation)
#[derive(Clone, PartialEq, Message)]
pub struct GraphDef {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeDef>,
    #[prost(message, optional, tag = "4")]
    pub versions: Option<VersionDef>,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeDef {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, tag = "2")]
    pub op: String,
    #[prost(string, repeated, tag = "3")]
    pub input: Vec<String>,
    #[prost(message, repeated, tag = "5")]
    pub attr: Vec<AttrValue>,
}

#[derive(Clone, PartialEq, Message)]
pub struct AttrValue {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(oneof = "attr_value::Value", tags = "2, 3, 4, 5, 6, 7, 8")]
    pub value: Option<attr_value::Value>,
}

pub mod attr_value {
    use super::*;
    
    #[derive(Clone, PartialEq)]
    pub enum Value {
        #[prost(float, tag = "2")]
        F(f32),
        #[prost(int64, tag = "3")]
        I(i64),
        #[prost(bytes, tag = "4")]
        S(Vec<u8>),
        #[prost(message, tag = "5")]
        Tensor(TensorProto),
        #[prost(message, tag = "6")]
        Shape(TensorShapeProto),
        #[prost(enumeration = "super::DataType", tag = "7")]
        Type(i32),
        #[prost(message, tag = "8")]
        List(ListValue),
    }
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(enumeration = "DataType", tag = "1")]
    pub dtype: i32,
    #[prost(message, optional, tag = "2")]
    pub tensor_shape: Option<TensorShapeProto>,
    #[prost(int32, tag = "3")]
    pub version_number: i32,
    #[prost(bytes = "vec", tag = "4")]
    pub tensor_content: Vec<u8>,
    #[prost(float, repeated, tag = "5")]
    pub float_val: Vec<f32>,
    #[prost(double, repeated, tag = "6")]
    pub double_val: Vec<f64>,
    #[prost(int32, repeated, tag = "7")]
    pub int_val: Vec<i32>,
    #[prost(int64, repeated, tag = "10")]
    pub int64_val: Vec<i64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "2")]
    pub dim: Vec<TensorShapeDim>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeDim {
    #[prost(int64, tag = "1")]
    pub size: i64,
    #[prost(string, tag = "2")]
    pub name: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct ListValue {
    #[prost(float, repeated, tag = "2")]
    pub f: Vec<f32>,
    #[prost(int64, repeated, tag = "3")]
    pub i: Vec<i64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct VersionDef {
    #[prost(int32, tag = "1")]
    pub producer: i32,
    #[prost(int32, tag = "2")]
    pub min_consumer: i32,
}

impl TensorFlowParser {
    pub fn new() -> Self {
        Self {
            graph_def: None,
            variables: HashMap::new(),
            model_info: ModelInfo {
                format: "tensorflow".into(),
                version: "2.0".into(),
                input_shape: vec![],
                output_shape: vec![],
                total_parameters: 0,
                architecture_hints: vec![],
            },
        }
    }
    
    pub fn load_saved_model(&mut self, path: &Path) -> Result<(), SrganError> {
        if !path.exists() || !path.is_dir() {
            return Err(SrganError::FileNotFound(path.to_path_buf()));
        }
        
        // Load the main graph
        let pb_path = path.join("saved_model.pb");
        if !pb_path.exists() {
            // Try alternative paths
            let alt_path = path.join("saved_model.pbtxt");
            if !alt_path.exists() {
                return Err(SrganError::Parse("No saved_model.pb found".into()));
            }
        }
        
        self.load_graph_def(&pb_path)?;
        
        // Load variables from variables/ directory
        let vars_dir = path.join("variables");
        if vars_dir.exists() {
            self.load_variables(&vars_dir)?;
        }
        
        // Analyze the model
        self.analyze_model()?;
        
        info!("Successfully loaded TensorFlow SavedModel from {:?}", path);
        Ok(())
    }
    
    fn load_graph_def(&mut self, path: &Path) -> Result<(), SrganError> {
        let mut file = File::open(path)
            .map_err(|e| SrganError::Io(e))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| SrganError::Io(e))?;
        
        // Parse protobuf
        let graph_def = GraphDef::decode(&buffer[..])
            .map_err(|e| SrganError::Parse(format!("Failed to parse GraphDef: {}", e)))?;
        
        debug!("Loaded graph with {} nodes", graph_def.node.len());
        self.graph_def = Some(graph_def);
        
        Ok(())
    }
    
    fn load_variables(&mut self, vars_dir: &Path) -> Result<(), SrganError> {
        // Look for variables.data-* and variables.index files
        let index_path = vars_dir.join("variables.index");
        let data_path = vars_dir.join("variables.data-00000-of-00001");
        
        if !index_path.exists() || !data_path.exists() {
            warn!("Variables files not found in {:?}", vars_dir);
            return Ok(());
        }
        
        // Parse checkpoint files (simplified - would need proper checkpoint reader)
        self.parse_checkpoint_files(&index_path, &data_path)?;
        
        Ok(())
    }
    
    fn parse_checkpoint_files(&mut self, _index_path: &Path, data_path: &Path) -> Result<(), SrganError> {
        let mut file = File::open(data_path)
            .map_err(|e| SrganError::Io(e))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| SrganError::Io(e))?;
        
        // This is a simplified version - real implementation would parse TF checkpoint format
        // For now, we'll create placeholder tensors
        info!("Loaded {} bytes of variable data", buffer.len());
        
        Ok(())
    }
    
    fn analyze_model(&mut self) -> Result<(), SrganError> {
        let graph = self.graph_def.as_ref()
            .ok_or_else(|| SrganError::Parse("No graph loaded".into()))?;
        
        let mut total_params = 0;
        let mut conv_layers = 0;
        let mut dense_layers = 0;
        
        for node in &graph.node {
            match node.op.as_str() {
                "Conv2D" | "Conv2DBackpropInput" => {
                    conv_layers += 1;
                    self.model_info.architecture_hints.push(format!("Conv2D: {}", node.name));
                    
                    // Try to extract kernel shape
                    if let Some(shape) = self.extract_conv_shape(node) {
                        let params = shape.iter().product::<usize>();
                        total_params += params;
                    }
                }
                "MatMul" | "Dense" => {
                    dense_layers += 1;
                    self.model_info.architecture_hints.push(format!("Dense: {}", node.name));
                }
                "Placeholder" => {
                    // This might be an input
                    if node.name.contains("input") || node.name.contains("x") {
                        if let Some(shape) = self.extract_tensor_shape(node) {
                            self.model_info.input_shape = shape;
                        }
                    }
                }
                "Identity" => {
                    // This might be an output
                    if node.name.contains("output") || node.name.contains("y") {
                        if let Some(shape) = self.extract_tensor_shape(node) {
                            self.model_info.output_shape = shape;
                        }
                    }
                }
                _ => {}
            }
        }
        
        self.model_info.total_parameters = total_params;
        
        info!("Model analysis: {} conv layers, {} dense layers, ~{} parameters",
              conv_layers, dense_layers, total_params);
        
        // Detect if it's likely an SRGAN model
        if conv_layers > 10 && self.model_info.architecture_hints.iter()
            .any(|h| h.contains("residual") || h.contains("res")) {
            self.model_info.architecture_hints.push("Likely SRGAN/SRResNet architecture".into());
        }
        
        Ok(())
    }
    
    fn extract_conv_shape(&self, node: &NodeDef) -> Option<Vec<usize>> {
        for attr in &node.attr {
            if attr.key == "shape" {
                if let Some(attr_value::Value::Shape(shape)) = &attr.value {
                    let dims: Vec<usize> = shape.dim.iter()
                        .map(|d| d.size as usize)
                        .collect();
                    return Some(dims);
                }
            }
        }
        None
    }
    
    fn extract_tensor_shape(&self, node: &NodeDef) -> Option<Vec<usize>> {
        for attr in &node.attr {
            if attr.key == "shape" || attr.key == "_output_shapes" {
                if let Some(attr_value::Value::Shape(shape)) = &attr.value {
                    let dims: Vec<usize> = shape.dim.iter()
                        .map(|d| d.size as usize)
                        .filter(|&d| d > 0)  // Filter out -1 (unknown) dimensions
                        .collect();
                    if !dims.is_empty() {
                        return Some(dims);
                    }
                }
            }
        }
        None
    }
    
    pub fn extract_weights_from_nodes(&mut self) -> Result<(), SrganError> {
        let graph = self.graph_def.as_ref()
            .ok_or_else(|| SrganError::Parse("No graph loaded".into()))?;
        
        for node in &graph.node {
            if node.op == "Const" {
                // Constants often contain weights
                for attr in &node.attr {
                    if attr.key == "value" {
                        if let Some(attr_value::Value::Tensor(tensor)) = &attr.value {
                            let tensor_data = self.parse_tensor_proto(tensor, &node.name)?;
                            
                            if validate_tensor_data(&tensor_data.data) {
                                self.variables.insert(node.name.clone(), tensor_data);
                            } else {
                                warn!("Tensor {} contains invalid values", node.name);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn parse_tensor_proto(&self, tensor: &TensorProto, name: &str) -> Result<TensorData, SrganError> {
        let shape = tensor.tensor_shape.as_ref()
            .map(|s| s.dim.iter().map(|d| d.size as usize).collect())
            .unwrap_or_else(Vec::new);
        
        let dtype = match tensor.dtype {
            1 => DataType::Float32,
            2 => DataType::Float64,
            3 => DataType::Int32,
            9 => DataType::Int64,
            4 => DataType::Uint8,
            _ => DataType::Float32,
        };
        
        let data = if !tensor.tensor_content.is_empty() {
            // Parse raw bytes
            self.parse_tensor_content(&tensor.tensor_content, dtype)?
        } else if !tensor.float_val.is_empty() {
            tensor.float_val.clone()
        } else if !tensor.double_val.is_empty() {
            tensor.double_val.iter().map(|&d| d as f32).collect()
        } else if !tensor.int_val.is_empty() {
            tensor.int_val.iter().map(|&i| i as f32).collect()
        } else if !tensor.int64_val.is_empty() {
            tensor.int64_val.iter().map(|&i| i as f32).collect()
        } else {
            vec![]
        };
        
        Ok(TensorData {
            name: name.to_string(),
            shape,
            data,
            dtype,
        })
    }
    
    fn parse_tensor_content(&self, content: &[u8], dtype: DataType) -> Result<Vec<f32>, SrganError> {
        let mut result = Vec::new();
        
        match dtype {
            DataType::Float32 => {
                for chunk in content.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk.try_into()
                        .map_err(|_| SrganError::Parse("Invalid float32 data".into()))?;
                    result.push(f32::from_le_bytes(bytes));
                }
            }
            DataType::Float64 => {
                for chunk in content.chunks_exact(8) {
                    let bytes: [u8; 8] = chunk.try_into()
                        .map_err(|_| SrganError::Parse("Invalid float64 data".into()))?;
                    result.push(f64::from_le_bytes(bytes) as f32);
                }
            }
            DataType::Int32 => {
                for chunk in content.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk.try_into()
                        .map_err(|_| SrganError::Parse("Invalid int32 data".into()))?;
                    result.push(i32::from_le_bytes(bytes) as f32);
                }
            }
            DataType::Int64 => {
                for chunk in content.chunks_exact(8) {
                    let bytes: [u8; 8] = chunk.try_into()
                        .map_err(|_| SrganError::Parse("Invalid int64 data".into()))?;
                    result.push(i64::from_le_bytes(bytes) as f32);
                }
            }
            DataType::Uint8 => {
                for &byte in content {
                    result.push(byte as f32);
                }
            }
        }
        
        Ok(result)
    }
}

impl WeightExtractor for TensorFlowParser {
    fn extract_weights(&self) -> Result<HashMap<String, TensorData>, SrganError> {
        Ok(self.variables.clone())
    }
    
    fn get_layer_names(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
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
    fn test_tensorflow_parser_creation() {
        let parser = TensorFlowParser::new();
        assert!(parser.graph_def.is_none());
        assert!(parser.variables.is_empty());
    }
    
    #[test]
    fn test_model_info_initialization() {
        let parser = TensorFlowParser::new();
        let info = parser.get_model_info();
        assert_eq!(info.format, "tensorflow");
        assert_eq!(info.version, "2.0");
    }
}