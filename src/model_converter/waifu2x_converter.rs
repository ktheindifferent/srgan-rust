//! Converter for waifu2x model weights to the `.rsr` format.
//!
//! Supports two input formats:
//!
//! 1. **Waifu2x JSON** — the original format distributed with waifu2x
//!    (e.g. `noise1_scale2x_model.json`).  Each file contains a JSON array
//!    of layer objects with `"weight"` and `"bias"` fields.
//!
//! 2. **Pre-converted `.rsr`** — the binary format used by this codebase
//!    (XZ-compressed bincode of `NetworkDescription`).
//!
//! ## Usage
//!
//! ```no_run
//! use srgan_rust::model_converter::waifu2x_converter::Waifu2xWeightConverter;
//! use std::path::Path;
//!
//! let mut converter = Waifu2xWeightConverter::new();
//! converter.load_waifu2x_json(Path::new("noise1_scale2x_model.json")).unwrap();
//! converter.save_rsr(Path::new("models/waifu2x/noise1_scale2_anime.rsr")).unwrap();
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use log::{info, warn};
use ndarray::{Array4, ArrayD, IxDyn};

use crate::error::SrganError;

/// VGG7 layer specification: (input_channels, output_channels).
const VGG7_LAYERS: [(usize, usize); 7] = [
    (3, 32),    // conv0
    (32, 32),   // conv1
    (32, 64),   // conv2
    (64, 64),   // conv3
    (64, 128),  // conv4
    (128, 128), // conv5
    (128, 0),   // conv6 — output channels depend on scale (3 or 12)
];

/// Converter for waifu2x model weights.
pub struct Waifu2xWeightConverter {
    /// Extracted weight tensors, keyed by layer name.
    weights: HashMap<String, Vec<f32>>,
    /// Extracted bias vectors, keyed by layer name.
    biases: HashMap<String, Vec<f32>>,
    /// Detected number of layers.
    num_layers: usize,
    /// Detected output channels of the final layer.
    output_channels: usize,
}

impl Waifu2xWeightConverter {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            biases: HashMap::new(),
            num_layers: 0,
            output_channels: 0,
        }
    }

    /// Load weights from a waifu2x JSON model file.
    ///
    /// The JSON format is an array of layer objects:
    /// ```json
    /// [
    ///   {
    ///     "nInputPlane": 3,
    ///     "nOutputPlane": 32,
    ///     "kW": 3, "kH": 3,
    ///     "weight": [ ... ],
    ///     "bias": [ ... ]
    ///   },
    ///   ...
    /// ]
    /// ```
    pub fn load_waifu2x_json(&mut self, path: &Path) -> Result<(), SrganError> {
        if !path.exists() {
            return Err(SrganError::FileNotFound(path.to_path_buf()));
        }

        let mut file = File::open(path).map_err(SrganError::Io)?;
        let mut json_str = String::new();
        file.read_to_string(&mut json_str).map_err(SrganError::Io)?;

        let layers: Vec<serde_json::Value> = serde_json::from_str(&json_str)
            .map_err(|e| SrganError::Parse(format!("Invalid waifu2x JSON: {}", e)))?;

        self.num_layers = layers.len();
        info!("Loading waifu2x JSON model with {} layers", self.num_layers);

        for (i, layer) in layers.iter().enumerate() {
            let n_in = layer["nInputPlane"].as_u64().unwrap_or(0) as usize;
            let n_out = layer["nOutputPlane"].as_u64().unwrap_or(0) as usize;
            let kw = layer["kW"].as_u64().unwrap_or(3) as usize;
            let kh = layer["kH"].as_u64().unwrap_or(3) as usize;

            if kw != 3 || kh != 3 {
                warn!("Layer {} has kernel {}x{} (expected 3x3)", i, kw, kh);
            }

            // Extract weights: waifu2x stores as [out_ch, in_ch, kH, kW] (OIHW)
            // alumina expects [batch, H, W, channels] (NHWC) for Conv ops
            let weight_arr = layer["weight"].as_array()
                .ok_or_else(|| SrganError::Parse(format!("Layer {} missing weight array", i)))?;

            let weight_data: Vec<f32> = Self::flatten_nested_array(weight_arr)?;
            let expected_size = n_out * n_in * kh * kw;
            if weight_data.len() != expected_size {
                return Err(SrganError::Parse(format!(
                    "Layer {} weight size mismatch: got {}, expected {} ({}x{}x{}x{})",
                    i, weight_data.len(), expected_size, n_out, n_in, kh, kw
                )));
            }

            // Extract biases
            let bias_arr = layer["bias"].as_array()
                .ok_or_else(|| SrganError::Parse(format!("Layer {} missing bias array", i)))?;
            let bias_data: Vec<f32> = bias_arr.iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            if bias_data.len() != n_out {
                return Err(SrganError::Parse(format!(
                    "Layer {} bias size mismatch: got {}, expected {}",
                    i, bias_data.len(), n_out
                )));
            }

            let layer_name = format!("layer{}", i);
            self.weights.insert(layer_name.clone(), weight_data);
            self.biases.insert(layer_name, bias_data);

            if i == layers.len() - 1 {
                self.output_channels = n_out;
            }
        }

        info!("Loaded {} layers, output_channels={}", self.num_layers, self.output_channels);
        Ok(())
    }

    /// Convert loaded weights to `.rsr` format and save to disk.
    pub fn save_rsr(&self, output_path: &Path) -> Result<(), SrganError> {
        if self.weights.is_empty() {
            return Err(SrganError::InvalidInput("No weights loaded".into()));
        }

        // Determine scale factor from output channels
        let scale_factor = if self.output_channels == 3 {
            1u32
        } else if self.output_channels == 12 {
            2u32
        } else {
            return Err(SrganError::InvalidInput(format!(
                "Unexpected output channels {}: expected 3 (scale=1) or 12 (scale=2)",
                self.output_channels
            )));
        };

        // Build parameter list in the order alumina expects
        let mut parameters = Vec::new();
        for i in 0..self.num_layers {
            let layer_name = format!("layer{}", i);

            let weight_data = self.weights.get(&layer_name)
                .ok_or_else(|| SrganError::InvalidInput(format!("Missing weights for {}", layer_name)))?;
            let bias_data = self.biases.get(&layer_name)
                .ok_or_else(|| SrganError::InvalidInput(format!("Missing biases for {}", layer_name)))?;

            // Determine layer dimensions
            let (n_in, n_out) = if i < VGG7_LAYERS.len() - 1 {
                VGG7_LAYERS[i]
            } else {
                (VGG7_LAYERS[VGG7_LAYERS.len() - 1].0, self.output_channels)
            };

            // Reshape weights from OIHW to the format needed by alumina
            // alumina Conv expects shape [kH, kW, in_channels, out_channels]
            let weight_oihw = Array4::from_shape_vec(
                (n_out, n_in, 3, 3),
                weight_data.clone(),
            ).map_err(|e| SrganError::Parse(format!("Weight reshape failed: {}", e)))?;

            // Transpose OIHW → HWIO
            let weight_hwio = weight_oihw.permuted_axes([2, 3, 1, 0]);
            let weight_dynamic = weight_hwio.into_dyn();
            parameters.push(weight_dynamic);

            // Bias as 1D
            let bias_dynamic = ArrayD::from_shape_vec(
                IxDyn(&[n_out]),
                bias_data.clone(),
            ).map_err(|e| SrganError::Parse(format!("Bias reshape failed: {}", e)))?;
            parameters.push(bias_dynamic);
        }

        // Build NetworkDescription
        let desc = crate::NetworkDescription {
            factor: scale_factor,
            width: 32,  // VGG7 base width
            log_depth: 3,  // ~7 layers ≈ 2^3 - 1
            global_node_factor: 0,
            parameters,
        };

        // Serialize and compress
        let serialized = bincode::serialize(&desc)
            .map_err(|e| SrganError::Parse(format!("bincode serialize failed: {}", e)))?;
        let shuffled = crate::shuffle(&serialized, 4);

        // Create parent directories
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(SrganError::Io)?;
        }

        let mut file = File::create(output_path).map_err(SrganError::Io)?;
        let mut encoder = xz2::write::XzEncoder::new(Vec::new(), 6);
        encoder.write_all(&shuffled)
            .map_err(|e| SrganError::Io(e))?;
        let compressed = encoder.finish()
            .map_err(|e| SrganError::Io(e))?;
        file.write_all(&compressed).map_err(SrganError::Io)?;

        info!("Saved waifu2x .rsr model to {}", output_path.display());
        Ok(())
    }

    /// Flatten a potentially nested JSON array of numbers into a flat Vec<f32>.
    fn flatten_nested_array(arr: &[serde_json::Value]) -> Result<Vec<f32>, SrganError> {
        let mut result = Vec::new();
        for val in arr {
            match val {
                serde_json::Value::Number(n) => {
                    result.push(n.as_f64().unwrap_or(0.0) as f32);
                }
                serde_json::Value::Array(inner) => {
                    result.extend(Self::flatten_nested_array(inner)?);
                }
                _ => {
                    return Err(SrganError::Parse(
                        "Unexpected value type in weight array".into()
                    ));
                }
            }
        }
        Ok(result)
    }

    /// Get a summary of the loaded model.
    pub fn summary(&self) -> String {
        let total_params: usize = self.weights.values().map(|w| w.len()).sum::<usize>()
            + self.biases.values().map(|b| b.len()).sum::<usize>();
        format!(
            "Waifu2x model: {} layers, {} total parameters, output_channels={}",
            self.num_layers, total_params, self.output_channels
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_converter_is_empty() {
        let conv = Waifu2xWeightConverter::new();
        assert_eq!(conv.num_layers, 0);
        assert!(conv.weights.is_empty());
    }

    #[test]
    fn save_rsr_fails_without_weights() {
        let conv = Waifu2xWeightConverter::new();
        assert!(conv.save_rsr(Path::new("/tmp/test.rsr")).is_err());
    }

    #[test]
    fn flatten_nested_array_works() {
        let arr = vec![
            serde_json::json!([1.0, 2.0]),
            serde_json::json!([3.0, 4.0]),
        ];
        let result = Waifu2xWeightConverter::flatten_nested_array(&arr).unwrap();
        assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn flatten_deeply_nested() {
        let arr = vec![
            serde_json::json!([[1.0, 2.0], [3.0, 4.0]]),
            serde_json::json!([[5.0, 6.0], [7.0, 8.0]]),
        ];
        let result = Waifu2xWeightConverter::flatten_nested_array(&arr).unwrap();
        assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}
