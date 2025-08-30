use std::collections::HashMap;
use crate::error::SrganError;

/// Common tensor data structure for all parsers
#[derive(Debug, Clone)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub dtype: DataType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Uint8,
}

/// Common trait for weight extraction from different formats
pub trait WeightExtractor {
    fn extract_weights(&self) -> Result<HashMap<String, TensorData>, SrganError>;
    fn get_layer_names(&self) -> Vec<String>;
    fn get_model_info(&self) -> ModelInfo;
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub format: String,
    pub version: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub total_parameters: usize,
    pub architecture_hints: Vec<String>,
}

/// Convert tensor layout from NHWC to NCHW
pub fn convert_nhwc_to_nchw(data: &[f32], shape: &[usize]) -> Result<Vec<f32>, SrganError> {
    if shape.len() != 4 {
        return Err(SrganError::Parse("Expected 4D tensor for NHWC conversion".into()));
    }
    
    let n = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];
    
    let mut nchw_data = vec![0.0; data.len()];
    
    for batch in 0..n {
        for row in 0..h {
            for col in 0..w {
                for channel in 0..c {
                    let nhwc_idx = batch * h * w * c + row * w * c + col * c + channel;
                    let nchw_idx = batch * c * h * w + channel * h * w + row * w + col;
                    nchw_data[nchw_idx] = data[nhwc_idx];
                }
            }
        }
    }
    
    Ok(nchw_data)
}

/// Convert tensor layout from NCHW to NHWC
pub fn convert_nchw_to_nhwc(data: &[f32], shape: &[usize]) -> Result<Vec<f32>, SrganError> {
    if shape.len() != 4 {
        return Err(SrganError::Parse("Expected 4D tensor for NCHW conversion".into()));
    }
    
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    
    let mut nhwc_data = vec![0.0; data.len()];
    
    for batch in 0..n {
        for channel in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let nchw_idx = batch * c * h * w + channel * h * w + row * w + col;
                    let nhwc_idx = batch * h * w * c + row * w * c + col * c + channel;
                    nhwc_data[nhwc_idx] = data[nchw_idx];
                }
            }
        }
    }
    
    Ok(nhwc_data)
}

/// Validate tensor data for NaN or Inf values
pub fn validate_tensor_data(data: &[f32]) -> bool {
    data.iter().all(|&x| x.is_finite())
}

/// Get statistics about tensor data
pub fn tensor_statistics(data: &[f32]) -> TensorStats {
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();
    
    TensorStats {
        min,
        max,
        mean,
        std_dev,
        total_elements: data.len(),
    }
}

#[derive(Debug)]
pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub total_elements: usize,
}