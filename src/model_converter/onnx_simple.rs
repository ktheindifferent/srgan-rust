use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use log::{info, warn, debug};

use crate::error::SrganError;
use crate::model_converter::common::{TensorData, DataType, WeightExtractor, ModelInfo};

// ── Protobuf wire-format helpers ────────────────────────────────────────────

/// Protobuf wire types we care about.
const WIRE_VARINT: u8 = 0;
const WIRE_64BIT: u8 = 1;
const WIRE_LEN: u8 = 2;

/// Read a varint from `buf[pos..]`, advancing `pos`.
fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64, SrganError> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        if *pos >= buf.len() {
            return Err(SrganError::Parse("Unexpected end of protobuf data".into()));
        }
        let byte = buf[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(SrganError::Parse("Varint too long".into()));
        }
    }
}

/// Skip a field value based on wire type.
fn skip_field(buf: &[u8], pos: &mut usize, wire_type: u8) -> Result<(), SrganError> {
    match wire_type {
        WIRE_VARINT => { read_varint(buf, pos)?; }
        WIRE_64BIT => { *pos += 8; }
        WIRE_LEN => {
            let len = read_varint(buf, pos)? as usize;
            *pos += len;
        }
        5 => { *pos += 4; } // 32-bit fixed
        _ => return Err(SrganError::Parse(format!("Unknown wire type {}", wire_type))),
    }
    if *pos > buf.len() {
        return Err(SrganError::Parse("Field extends past end of buffer".into()));
    }
    Ok(())
}

// ── ONNX TensorProto parser ────────────────────────────────────────────────

/// ONNX data type constants (from onnx.proto TensorProto.DataType).
const ONNX_FLOAT: i64 = 1;
const ONNX_DOUBLE: i64 = 11;
const ONNX_FLOAT16: i64 = 10;
const ONNX_INT32: i64 = 6;
const ONNX_INT64: i64 = 7;

/// Parsed ONNX tensor initializer.
struct OnnxTensor {
    name: String,
    dims: Vec<i64>,
    data_type: i64,
    float_data: Vec<f32>,
    raw_data: Vec<u8>,
}

impl OnnxTensor {
    fn new() -> Self {
        Self {
            name: String::new(),
            dims: Vec::new(),
            data_type: ONNX_FLOAT,
            float_data: Vec::new(),
            raw_data: Vec::new(),
        }
    }
}

/// Parse a TensorProto message from `buf`.
fn parse_tensor_proto(buf: &[u8]) -> Result<OnnxTensor, SrganError> {
    let mut t = OnnxTensor::new();
    let mut pos = 0;
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        let field_number = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;

        match (field_number, wire_type) {
            // field 1: dims (repeated int64) — can be packed or individual
            (1, WIRE_VARINT) => {
                let v = read_varint(buf, &mut pos)? as i64;
                t.dims.push(v);
            }
            (1, WIRE_LEN) => {
                // packed repeated int64
                let len = read_varint(buf, &mut pos)? as usize;
                let end = pos + len;
                while pos < end {
                    let v = read_varint(buf, &mut pos)? as i64;
                    t.dims.push(v);
                }
            }
            // field 2: data_type (int32)
            (2, WIRE_VARINT) => {
                t.data_type = read_varint(buf, &mut pos)? as i64;
            }
            // field 4: float_data (repeated float, packed)
            (4, WIRE_LEN) => {
                let len = read_varint(buf, &mut pos)? as usize;
                let end = pos + len;
                while pos + 4 <= end {
                    let val = f32::from_le_bytes([buf[pos], buf[pos+1], buf[pos+2], buf[pos+3]]);
                    t.float_data.push(val);
                    pos += 4;
                }
            }
            (4, 5) => {
                // individual float (wire type 5 = 32-bit)
                if pos + 4 <= buf.len() {
                    let val = f32::from_le_bytes([buf[pos], buf[pos+1], buf[pos+2], buf[pos+3]]);
                    t.float_data.push(val);
                    pos += 4;
                }
            }
            // field 5: int32_data — skip
            // field 8: name (string)
            (8, WIRE_LEN) => {
                let len = read_varint(buf, &mut pos)? as usize;
                if pos + len > buf.len() {
                    return Err(SrganError::Parse("Tensor name extends past buffer".into()));
                }
                t.name = String::from_utf8_lossy(&buf[pos..pos+len]).to_string();
                pos += len;
            }
            // field 13: raw_data (bytes)
            (13, WIRE_LEN) => {
                let len = read_varint(buf, &mut pos)? as usize;
                if pos + len > buf.len() {
                    return Err(SrganError::Parse("raw_data extends past buffer".into()));
                }
                t.raw_data = buf[pos..pos+len].to_vec();
                pos += len;
            }
            _ => {
                skip_field(buf, &mut pos, wire_type)?;
            }
        }
    }
    Ok(t)
}

/// Convert an OnnxTensor to f32 data, handling raw_data and float_data fields.
fn tensor_to_f32(t: &OnnxTensor) -> Vec<f32> {
    // Prefer float_data if present
    if !t.float_data.is_empty() {
        return t.float_data.clone();
    }
    // Otherwise decode raw_data based on data_type
    if !t.raw_data.is_empty() {
        match t.data_type {
            ONNX_FLOAT => {
                t.raw_data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            ONNX_DOUBLE => {
                t.raw_data.chunks_exact(8)
                    .map(|c| {
                        let v = f64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]]);
                        v as f32
                    })
                    .collect()
            }
            ONNX_FLOAT16 => {
                // Simple float16 decode: sign(1) + exp(5) + mantissa(10)
                t.raw_data.chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        half_to_f32(bits)
                    })
                    .collect()
            }
            ONNX_INT32 => {
                t.raw_data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()
            }
            ONNX_INT64 => {
                t.raw_data.chunks_exact(8)
                    .map(|c| {
                        let v = i64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]]);
                        v as f32
                    })
                    .collect()
            }
            _ => {
                warn!("Unknown ONNX data type {}, treating raw_data as float32", t.data_type);
                t.raw_data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        }
    } else {
        Vec::new()
    }
}

/// Convert IEEE 754 half-precision to f32.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        let val = (sign << 31) | 0;
        if mant == 0 {
            return f32::from_bits(val);
        }
        // Subnormal: convert to normalized f32
        let mut m = mant;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normalized
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

// ── ONNX GraphProto parser ─────────────────────────────────────────────────

/// Parse GraphProto, extracting only initializer tensors (field 5).
fn parse_graph_initializers(buf: &[u8]) -> Result<Vec<OnnxTensor>, SrganError> {
    let mut tensors = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        let field_number = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;

        if field_number == 5 && wire_type == WIRE_LEN {
            // initializer: repeated TensorProto
            let len = read_varint(buf, &mut pos)? as usize;
            if pos + len > buf.len() {
                return Err(SrganError::Parse("Initializer extends past graph buffer".into()));
            }
            let tensor = parse_tensor_proto(&buf[pos..pos+len])?;
            tensors.push(tensor);
            pos += len;
        } else {
            skip_field(buf, &mut pos, wire_type)?;
        }
    }
    Ok(tensors)
}

// ── ONNX ModelProto parser ─────────────────────────────────────────────────

/// Parse a top-level ONNX ModelProto, returning initializer tensors from the graph.
fn parse_onnx_model(buf: &[u8]) -> Result<Vec<OnnxTensor>, SrganError> {
    let mut pos = 0;
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        let field_number = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;

        if field_number == 7 && wire_type == WIRE_LEN {
            // field 7 = graph (GraphProto)
            let len = read_varint(buf, &mut pos)? as usize;
            if pos + len > buf.len() {
                return Err(SrganError::Parse("GraphProto extends past model buffer".into()));
            }
            return parse_graph_initializers(&buf[pos..pos+len]);
        } else {
            skip_field(buf, &mut pos, wire_type)?;
        }
    }
    Err(SrganError::Parse("No graph field found in ONNX ModelProto".into()))
}

// ── Public ONNX parser ─────────────────────────────────────────────────────

/// ONNX model parser that extracts real weights from .onnx protobuf files.
///
/// Parses the protobuf wire format directly (no codegen needed) to extract
/// graph initializer tensors, supporting float32, float64, float16, and
/// integer data types.
pub struct OnnxParser {
    weights: HashMap<String, TensorData>,
    model_info: ModelInfo,
}

impl OnnxParser {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            model_info: ModelInfo {
                format: "onnx".into(),
                version: "1.0".into(),
                input_shape: vec![],
                output_shape: vec![],
                total_parameters: 0,
                architecture_hints: vec![],
            },
        }
    }

    pub fn load_model(&mut self, path: &Path) -> Result<(), SrganError> {
        if !path.exists() {
            return Err(SrganError::FileNotFound(path.to_path_buf()));
        }

        let mut file = File::open(path).map_err(SrganError::Io)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(SrganError::Io)?;

        self.load_from_bytes(&buffer)?;

        info!(
            "Loaded ONNX model from {:?}: {} initializer tensors, {} total parameters",
            path,
            self.weights.len(),
            self.model_info.total_parameters,
        );
        Ok(())
    }

    /// Load from raw bytes (useful for in-memory ONNX data).
    pub fn load_from_bytes(&mut self, buffer: &[u8]) -> Result<(), SrganError> {
        if buffer.len() < 4 {
            return Err(SrganError::Parse("File too small to be a valid ONNX model".into()));
        }

        let tensors = parse_onnx_model(buffer)?;

        if tensors.is_empty() {
            warn!("ONNX model contains no initializer tensors");
        }

        let mut total_params = 0usize;
        for t in &tensors {
            let data = tensor_to_f32(t);
            let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
            let n_params = data.len();
            total_params += n_params;

            let dtype = match t.data_type {
                ONNX_FLOAT => DataType::Float32,
                ONNX_DOUBLE => DataType::Float64,
                ONNX_INT32 => DataType::Int32,
                ONNX_INT64 => DataType::Int64,
                _ => DataType::Float32,
            };

            debug!(
                "  tensor '{}': shape={:?}, dtype={}, params={}",
                t.name, shape, t.data_type, n_params
            );

            if !t.name.is_empty() {
                self.weights.insert(t.name.clone(), TensorData {
                    name: t.name.clone(),
                    shape,
                    data,
                    dtype,
                });
            }
        }

        self.model_info.total_parameters = total_params;
        self.model_info.architecture_hints.push(
            format!("ONNX model with {} initializer tensors", self.weights.len()),
        );

        Ok(())
    }

    /// Return extracted weights as a flat `Vec<f32>` suitable for constructing
    /// a `NetworkDescription::parameters` vector.
    ///
    /// Tensors are returned in alphabetical order by name so the mapping is
    /// deterministic.
    pub fn weights_as_flat_params(&self) -> Vec<Vec<f32>> {
        let mut keys: Vec<&String> = self.weights.keys().collect();
        keys.sort();
        keys.iter().map(|k| self.weights[*k].data.clone()).collect()
    }

    pub fn map_onnx_operators_to_srgan(&self) -> HashMap<String, String> {
        let mut mapping = HashMap::new();
        for (i, key) in self.weights.keys().enumerate() {
            mapping.insert(key.clone(), format!("param_{}", i));
        }
        mapping
    }
}

impl WeightExtractor for OnnxParser {
    fn extract_weights(&self) -> Result<HashMap<String, TensorData>, SrganError> {
        Ok(self.weights.clone())
    }

    fn get_layer_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.weights.keys().cloned().collect();
        names.sort();
        names
    }

    fn get_model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_parser_creation() {
        let parser = OnnxParser::new();
        assert!(parser.weights.is_empty());
        assert_eq!(parser.model_info.format, "onnx");
    }

    #[test]
    fn test_operator_mapping() {
        let parser = OnnxParser::new();
        let mapping = parser.map_onnx_operators_to_srgan();
        assert!(mapping.is_empty()); // no weights loaded yet
    }

    #[test]
    fn test_half_to_f32_zero() {
        assert_eq!(half_to_f32(0x0000), 0.0f32);
    }

    #[test]
    fn test_half_to_f32_one() {
        // 1.0 in fp16 = 0x3C00
        let val = half_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_half_to_f32_negative() {
        // -2.0 in fp16 = 0xC000
        let val = half_to_f32(0xC000);
        assert!((val - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_varint_roundtrip() {
        // Encode 300 as varint: 0xAC 0x02
        let buf = [0xAC, 0x02];
        let mut pos = 0;
        let v = read_varint(&buf, &mut pos).unwrap();
        assert_eq!(v, 300);
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_tensor_to_f32_float_data() {
        let t = OnnxTensor {
            name: "test".into(),
            dims: vec![2, 2],
            data_type: ONNX_FLOAT,
            float_data: vec![1.0, 2.0, 3.0, 4.0],
            raw_data: Vec::new(),
        };
        let data = tensor_to_f32(&t);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_to_f32_raw_data() {
        let mut raw = Vec::new();
        for &v in &[1.0f32, 2.0, 3.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let t = OnnxTensor {
            name: "test".into(),
            dims: vec![3],
            data_type: ONNX_FLOAT,
            float_data: Vec::new(),
            raw_data: raw,
        };
        let data = tensor_to_f32(&t);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
