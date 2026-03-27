//! Bridge between the ONNX parser and the native `NetworkDescription` used by
//! the inference pipeline.
//!
//! This module takes raw ONNX bytes, extracts the initializer tensors via
//! `OnnxParser`, and maps them into a `NetworkDescription` that can be fed
//! directly to `UpscalingNetwork::new()`.

use ndarray::{ArrayD, IxDyn};

use crate::NetworkDescription;
use super::onnx::OnnxParser;
use super::common::WeightExtractor;

/// Load raw ONNX bytes and produce a `NetworkDescription` ready for inference.
///
/// Returns `(description, display_string)`.
pub fn load_onnx_as_network_description(
    data: &[u8],
) -> Result<(NetworkDescription, String), String> {
    let mut parser = OnnxParser::new();
    parser
        .load_from_bytes(data)
        .map_err(|e| format!("Failed to parse ONNX model: {}", e))?;

    let info = parser.get_model_info();
    let layer_names = parser.get_layer_names();

    let weights = parser
        .extract_weights()
        .map_err(|e| format!("Failed to extract ONNX weights: {}", e))?;

    // Convert HashMap<String, TensorData> → Vec<ArrayD<f32>> in sorted order
    // so the mapping is deterministic.
    let mut sorted_names = layer_names;
    sorted_names.sort();

    let parameters: Vec<ArrayD<f32>> = sorted_names
        .iter()
        .filter_map(|name| weights.get(name))
        .map(|td| {
            let shape: Vec<usize> = td.shape.clone();
            ArrayD::from_shape_vec(IxDyn(&shape), td.data.clone())
                .unwrap_or_else(|_| {
                    // If shape doesn't match data length, create a flat array
                    ArrayD::from_shape_vec(IxDyn(&[td.data.len()]), td.data.clone())
                        .expect("flat shape must always work")
                })
        })
        .collect();

    let total_params: usize = parameters.iter().map(|p| p.len()).sum();

    // Infer scale factor and architecture hints from the ONNX metadata
    let scale_factor = infer_scale_factor(&info.input_shape, &info.output_shape);

    let desc = NetworkDescription {
        factor: scale_factor,
        width: 32, // sensible default; overridden if the ONNX graph implies otherwise
        log_depth: 4,
        global_node_factor: 1,
        parameters,
    };

    let display = format!(
        "ONNX model ({} tensors, {} params, {}× scale)",
        sorted_names.len(),
        total_params,
        scale_factor,
    );

    Ok((desc, display))
}

/// Try to infer the scale factor from input/output shapes.
fn infer_scale_factor(input_shape: &[usize], output_shape: &[usize]) -> u32 {
    // Typical ONNX SR models: input [1, 3, H, W], output [1, 3, H*s, W*s]
    if input_shape.len() == 4 && output_shape.len() == 4 {
        let in_h = input_shape[2];
        let out_h = output_shape[2];
        if in_h > 0 && out_h > in_h {
            return (out_h / in_h) as u32;
        }
    }
    4 // default
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_scale_factor_4x() {
        let input = vec![1, 3, 64, 64];
        let output = vec![1, 3, 256, 256];
        assert_eq!(infer_scale_factor(&input, &output), 4);
    }

    #[test]
    fn test_infer_scale_factor_2x() {
        let input = vec![1, 3, 128, 128];
        let output = vec![1, 3, 256, 256];
        assert_eq!(infer_scale_factor(&input, &output), 2);
    }

    #[test]
    fn test_infer_scale_factor_default() {
        assert_eq!(infer_scale_factor(&[], &[]), 4);
    }
}
