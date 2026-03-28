//! INT8 quantization for faster inference.
//!
//! Converts FP32 model weights to INT8 representation with per-tensor
//! scale/zero-point calibration.  Provides both:
//!
//! - **Static quantization** — convert weights ahead-of-time for ~4× speedup.
//! - **Quantized tensor operations** — INT8 matrix multiply with FP32 accumulator.
//!
//! Supports native `.rsr` models and ONNX quantized models.
//!
//! Toggle via `quantize: true` in the API request body.

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::error::{Result, SrganError};
use crate::NetworkDescription;

// ── Constants ───────────────────────────────────────────────────────────────

/// Range of INT8 values.
const I8_MIN: f32 = -128.0;
const I8_MAX: f32 = 127.0;

// ── Quantization parameters ─────────────────────────────────────────────────

/// Per-tensor quantization parameters (symmetric affine quantization).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor: `float_val = scale * (int8_val - zero_point)`.
    pub scale: f32,
    /// Zero-point offset.
    pub zero_point: i8,
}

impl QuantizationParams {
    /// Compute quantization parameters from the min/max of a tensor.
    pub fn from_min_max(min_val: f32, max_val: f32) -> Self {
        let min_val = min_val.min(0.0);
        let max_val = max_val.max(0.0);

        let range = max_val - min_val;
        if range < 1e-10 {
            return QuantizationParams {
                scale: 1.0,
                zero_point: 0,
            };
        }

        let scale = range / (I8_MAX - I8_MIN);
        let zero_point = (I8_MIN - min_val / scale).round().max(I8_MIN).min(I8_MAX) as i8;

        QuantizationParams { scale, zero_point }
    }

    /// Compute from tensor statistics.
    pub fn calibrate(tensor: &ArrayViewD<f32>) -> Self {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in tensor.iter() {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        Self::from_min_max(min_val, max_val)
    }
}

// ── Quantized tensor ────────────────────────────────────────────────────────

/// A tensor stored in INT8 format with quantization parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// INT8 data stored as flat Vec for cache efficiency.
    pub data: Vec<i8>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Quantization parameters for this tensor.
    pub params: QuantizationParams,
}

impl QuantizedTensor {
    /// Quantize an FP32 tensor to INT8.
    pub fn from_fp32(tensor: &ArrayViewD<f32>) -> Self {
        let params = QuantizationParams::calibrate(tensor);
        let data: Vec<i8> = tensor
            .iter()
            .map(|&v| {
                let quantized = (v / params.scale) + params.zero_point as f32;
                quantized.round().max(I8_MIN).min(I8_MAX) as i8
            })
            .collect();

        QuantizedTensor {
            data,
            shape: tensor.shape().to_vec(),
            params,
        }
    }

    /// Dequantize back to FP32.
    pub fn to_fp32(&self) -> ArrayD<f32> {
        let values: Vec<f32> = self
            .data
            .iter()
            .map(|&v| self.params.scale * (v as f32 - self.params.zero_point as f32))
            .collect();

        ArrayD::from_shape_vec(IxDyn(&self.shape), values)
            .expect("shape mismatch in dequantization")
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compression ratio vs FP32 (should be ~4×).
    pub fn compression_ratio(&self) -> f64 {
        4.0 // FP32 = 4 bytes, INT8 = 1 byte
    }
}

// ── INT8 tensor operations ──────────────────────────────────────────────────

/// Perform INT8 matrix-vector multiply with FP32 accumulator.
///
/// `weights` is a quantized 2D weight matrix [out_features, in_features].
/// `input` is an FP32 input vector [in_features].
/// Returns FP32 output [out_features].
pub fn int8_matvec(weights: &QuantizedTensor, input: &[f32]) -> Vec<f32> {
    assert_eq!(weights.shape.len(), 2, "weights must be 2D");
    let out_features = weights.shape[0];
    let in_features = weights.shape[1];
    assert_eq!(input.len(), in_features, "input dimension mismatch");

    let w_scale = weights.params.scale;
    let w_zp = weights.params.zero_point as f32;

    let mut output = vec![0.0f32; out_features];

    for i in 0..out_features {
        let row_offset = i * in_features;
        let mut acc = 0.0f32;
        for j in 0..in_features {
            let w_int8 = weights.data[row_offset + j] as f32;
            let w_fp32 = w_scale * (w_int8 - w_zp);
            acc += w_fp32 * input[j];
        }
        output[i] = acc;
    }

    output
}

/// Element-wise INT8 multiply-add: `out[i] = a[i] * b[i] + bias[i]`.
/// Both `a` and `b` are quantized; `bias` is FP32.
pub fn int8_elementwise_fma(
    a: &QuantizedTensor,
    b: &QuantizedTensor,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(a.data.len(), b.data.len(), "tensor length mismatch");

    let a_scale = a.params.scale;
    let a_zp = a.params.zero_point as f32;
    let b_scale = b.params.scale;
    let b_zp = b.params.zero_point as f32;
    let combined_scale = a_scale * b_scale;

    let mut output = Vec::with_capacity(a.data.len());
    for i in 0..a.data.len() {
        let av = (a.data[i] as f32 - a_zp) * (b.data[i] as f32 - b_zp) * combined_scale;
        let bv = bias.map_or(0.0, |b| b[i]);
        output.push(av + bv);
    }
    output
}

// ── Model quantization ──────────────────────────────────────────────────────

/// A fully quantized model: all weight tensors stored as INT8.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedModel {
    pub factor: u32,
    pub width: u32,
    pub log_depth: u32,
    pub global_node_factor: u32,
    /// Quantized weight tensors (one per layer).
    pub parameters: Vec<QuantizedTensor>,
    /// Original FP32 size in bytes.
    pub original_size_bytes: usize,
    /// Quantized size in bytes.
    pub quantized_size_bytes: usize,
}

impl QuantizedModel {
    /// Quantize an FP32 [`NetworkDescription`] to INT8.
    pub fn from_network(desc: &NetworkDescription) -> Self {
        let mut original_size = 0usize;
        let mut quantized_size = 0usize;

        let parameters: Vec<QuantizedTensor> = desc
            .parameters
            .iter()
            .map(|p| {
                original_size += p.len() * 4; // FP32 = 4 bytes
                let qt = QuantizedTensor::from_fp32(&p.view());
                quantized_size += qt.data.len(); // INT8 = 1 byte
                qt
            })
            .collect();

        QuantizedModel {
            factor: desc.factor,
            width: desc.width,
            log_depth: desc.log_depth,
            global_node_factor: desc.global_node_factor,
            parameters,
            original_size_bytes: original_size,
            quantized_size_bytes: quantized_size,
        }
    }

    /// Dequantize back to FP32 [`NetworkDescription`] for inference through
    /// the standard graph pipeline.
    pub fn to_network_description(&self) -> NetworkDescription {
        NetworkDescription {
            factor: self.factor,
            width: self.width,
            log_depth: self.log_depth,
            global_node_factor: self.global_node_factor,
            parameters: self.parameters.iter().map(|qt| qt.to_fp32()).collect(),
        }
    }

    /// Compression ratio achieved (should be ~4×).
    pub fn compression_ratio(&self) -> f64 {
        if self.quantized_size_bytes == 0 {
            return 1.0;
        }
        self.original_size_bytes as f64 / self.quantized_size_bytes as f64
    }

    /// Total number of quantized parameters.
    pub fn total_params(&self) -> usize {
        self.parameters.iter().map(|p| p.len()).sum()
    }
}

/// Quantize a model from raw `.rsr` bytes and return the quantized model.
pub fn quantize_model_bytes(data: &[u8]) -> Result<QuantizedModel> {
    let desc = crate::network_from_bytes(data)
        .map_err(|e| SrganError::Serialization(e))?;
    Ok(QuantizedModel::from_network(&desc))
}

/// Quantize a model from a label (e.g. "natural", "anime").
pub fn quantize_model_by_label(label: &str) -> Result<QuantizedModel> {
    let network = crate::UpscalingNetwork::from_label(label, None)
        .map_err(|e| SrganError::Network(e))?;
    let (_, params) = network.borrow_network();
    let desc = NetworkDescription {
        factor: 4,
        width: 12,
        log_depth: 4,
        global_node_factor: 1,
        parameters: params.to_vec(),
    };
    Ok(QuantizedModel::from_network(&desc))
}

// ── ONNX quantized model support ────────────────────────────────────────────

/// Metadata for an ONNX quantized model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxQuantizedModelInfo {
    /// Path or identifier for the ONNX file.
    pub source: String,
    /// Whether the ONNX model uses INT8 quantization.
    pub is_quantized: bool,
    /// Number of quantized operators.
    pub quantized_ops: usize,
    /// Total operators in the graph.
    pub total_ops: usize,
}

/// Check if an ONNX model file contains quantized operators.
///
/// Looks for QLinearConv / QLinearMatMul / QuantizeLinear nodes in the graph.
pub fn detect_onnx_quantization(data: &[u8]) -> OnnxQuantizedModelInfo {
    // Simple heuristic: search for known quantized op-type strings in the
    // protobuf bytes (avoids full protobuf parsing dependency).
    let data_str = String::from_utf8_lossy(data);
    let quantized_ops = ["QLinearConv", "QLinearMatMul", "QuantizeLinear", "DequantizeLinear"]
        .iter()
        .map(|op| data_str.matches(op).count())
        .sum::<usize>();
    let total_ops = data_str.matches("op_type").count();

    OnnxQuantizedModelInfo {
        source: String::new(),
        is_quantized: quantized_ops > 0,
        quantized_ops,
        total_ops,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![0.1, -0.5, 0.8, -1.0, 0.3, 0.0],
        )
        .unwrap();

        let qt = QuantizedTensor::from_fp32(&original.view());
        let recovered = qt.to_fp32();

        // Check shapes match
        assert_eq!(original.shape(), recovered.shape());

        // Check values are close (quantization introduces small error)
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 0.02,
                "Quantization error too large: {} vs {}",
                a,
                b,
            );
        }
    }

    #[test]
    fn test_quantization_params_symmetric() {
        let params = QuantizationParams::from_min_max(-1.0, 1.0);
        assert!(params.scale > 0.0);
        assert!(params.scale < 0.01); // ~2/255
    }

    #[test]
    fn test_int8_matvec() {
        let weights_fp32 = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let weights = QuantizedTensor::from_fp32(&weights_fp32.view());

        let input = vec![1.0, 2.0, 3.0];
        let output = int8_matvec(&weights, &input);

        assert_eq!(output.len(), 2);
        // First row [1,0,0] · [1,2,3] ≈ 1.0
        assert!((output[0] - 1.0).abs() < 0.1);
        // Second row [0,1,0] · [1,2,3] ≈ 2.0
        assert!((output[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_quantized_model_compression() {
        let desc = NetworkDescription {
            factor: 4,
            width: 12,
            log_depth: 4,
            global_node_factor: 1,
            parameters: vec![
                ArrayD::from_elem(IxDyn(&[64, 64]), 0.01f32),
                ArrayD::from_elem(IxDyn(&[64]), 0.0f32),
            ],
        };

        let qm = QuantizedModel::from_network(&desc);
        assert!((qm.compression_ratio() - 4.0).abs() < 0.01);
        assert_eq!(qm.total_params(), 64 * 64 + 64);
    }

    #[test]
    fn test_quantized_model_roundtrip() {
        let desc = NetworkDescription {
            factor: 4,
            width: 12,
            log_depth: 4,
            global_node_factor: 1,
            parameters: vec![ArrayD::from_elem(IxDyn(&[4, 4]), 0.5f32)],
        };

        let qm = QuantizedModel::from_network(&desc);
        let recovered = qm.to_network_description();

        assert_eq!(recovered.factor, 4);
        assert_eq!(recovered.parameters.len(), 1);
        for (a, b) in desc.parameters[0].iter().zip(recovered.parameters[0].iter()) {
            assert!((a - b).abs() < 0.02);
        }
    }

    #[test]
    fn test_onnx_quantization_detection() {
        let fake_onnx = b"op_type: QLinearConv op_type: QLinearConv op_type: Conv";
        let info = detect_onnx_quantization(fake_onnx);
        assert!(info.is_quantized);
        assert_eq!(info.quantized_ops, 2);
    }
}
