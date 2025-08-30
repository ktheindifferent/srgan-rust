use srgan_rust::model_converter::ModelConverter;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;
use serde_pickle::{value_to_vec, Value, HashableValue, SerOptions};
use std::collections::BTreeMap;

/// Test PyTorch tensor v2 format with storage, offset, and size
#[test]
fn test_pytorch_tensor_v2_format() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("tensor_v2.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Create a tensor with storage format
    let mut tensor_dict = BTreeMap::new();
    
    // Storage contains the actual data
    let storage_data = vec![Value::F64(0.1), Value::F64(0.2), Value::F64(0.3), 
                            Value::F64(0.4), Value::F64(0.5), Value::F64(0.6)];
    tensor_dict.insert(
        HashableValue::String("storage".to_string()),
        Value::List(storage_data)
    );
    
    // Offset into the storage
    tensor_dict.insert(
        HashableValue::String("storage_offset".to_string()),
        Value::I64(2)
    );
    
    // Size of the tensor
    tensor_dict.insert(
        HashableValue::String("size".to_string()),
        Value::List(vec![Value::I64(2), Value::I64(2)])
    );
    
    state_dict.insert(
        HashableValue::String("layer.weight".to_string()),
        Value::Dict(tensor_dict)
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load and verify
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load tensor v2 format: {:?}", result);
}

/// Test ESRGAN model detection
#[test]
fn test_esrgan_model_detection() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("esrgan.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // ESRGAN-specific layers
    state_dict.insert(
        HashableValue::String("conv_first.weight".to_string()),
        Value::List(vec![Value::F64(0.01); 64 * 3 * 3 * 3])
    );
    
    // RRDB blocks
    for i in 0..3 {
        state_dict.insert(
            HashableValue::String(format!("RRDB_trunk.{}.RDB1.conv1.weight", i)),
            Value::List(vec![Value::F64(0.01); 32 * 64 * 3 * 3])
        );
    }
    
    state_dict.insert(
        HashableValue::String("trunk_conv.weight".to_string()),
        Value::List(vec![Value::F64(0.01); 64 * 64 * 3 * 3])
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load and check architecture detection
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load ESRGAN model: {:?}", result);
    
    let stats = converter.get_conversion_stats();
    assert_eq!(stats.get("architecture"), Some(&"esrgan".to_string()));
}

/// Test float16 weight conversion
#[test]
fn test_float16_weights() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("float16.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Create float16 weights as bytes
    // Float16: sign(1) + exponent(5) + mantissa(10)
    // 0x3C00 = 0011 1100 0000 0000 = 1.0 in float16
    // 0x4000 = 0100 0000 0000 0000 = 2.0 in float16
    let float16_bytes = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0 in little-endian
    
    state_dict.insert(
        HashableValue::String("layer.weight".to_string()),
        Value::Bytes(float16_bytes)
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load and verify
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load float16 weights: {:?}", result);
}

/// Test model with mixed precision weights
#[test]
fn test_mixed_precision_model() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("mixed_precision.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Float32 weights (stored as F64 in pickle)
    state_dict.insert(
        HashableValue::String("conv1.weight".to_string()),
        Value::List(vec![Value::F64(0.1), Value::F64(0.2)])
    );
    
    // Float64 weights
    state_dict.insert(
        HashableValue::String("conv2.weight".to_string()),
        Value::List(vec![Value::F64(0.3), Value::F64(0.4)])
    );
    
    // Int8 quantized weights
    state_dict.insert(
        HashableValue::String("conv3.weight".to_string()),
        Value::Bytes(vec![127, 64, 0, 192, 255])
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load and verify all layers are parsed
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load mixed precision model: {:?}", result);
    
    let stats = converter.get_conversion_stats();
    let param_count: usize = stats.get("param_count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    
    // Should have all 9 parameters (2 + 2 + 5)
    assert_eq!(param_count, 9, "Expected 9 parameters, got {}", param_count);
}

/// Test upscale factor detection
#[test]
fn test_upscale_factor_detection() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test x2 upscale
    {
        let model_path = temp_dir.path().join("x2_model.pth");
        let mut state_dict = BTreeMap::new();
        
        state_dict.insert(
            HashableValue::String("upsampling_x2.weight".to_string()),
            Value::List(vec![Value::F64(0.01); 256 * 64 * 3 * 3])
        );
        
        let state_dict = Value::Dict(state_dict);
        let ser_options = SerOptions::new();
        let pickled = value_to_vec(&state_dict, ser_options).unwrap();
        
        let mut file = File::create(&model_path).unwrap();
        file.write_all(&pickled).unwrap();
        
        let mut converter = ModelConverter::new();
        converter.load_pytorch(&model_path).unwrap();
        
        let stats = converter.get_conversion_stats();
        assert!(stats.get("output_shape").unwrap().contains("512"));
    }
    
    // Test x8 upscale
    {
        let model_path = temp_dir.path().join("x8_model.pth");
        let mut state_dict = BTreeMap::new();
        
        state_dict.insert(
            HashableValue::String("model_x8.upsampling.0.weight".to_string()),
            Value::List(vec![Value::F64(0.01); 256 * 64 * 3 * 3])
        );
        state_dict.insert(
            HashableValue::String("model_x8.upsampling.1.weight".to_string()),
            Value::List(vec![Value::F64(0.01); 256 * 64 * 3 * 3])
        );
        state_dict.insert(
            HashableValue::String("model_x8.upsampling.2.weight".to_string()),
            Value::List(vec![Value::F64(0.01); 256 * 64 * 3 * 3])
        );
        
        let state_dict = Value::Dict(state_dict);
        let ser_options = SerOptions::new();
        let pickled = value_to_vec(&state_dict, ser_options).unwrap();
        
        let mut file = File::create(&model_path).unwrap();
        file.write_all(&pickled).unwrap();
        
        let mut converter = ModelConverter::new();
        converter.load_pytorch(&model_path).unwrap();
        
        let stats = converter.get_conversion_stats();
        assert!(stats.get("output_shape").unwrap().contains("2048"));
    }
}

/// Test deep residual network detection
#[test]
fn test_deep_resnet_detection() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("deep_resnet.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Add many residual blocks
    for i in 0..25 {
        state_dict.insert(
            HashableValue::String(format!("residual_block_{}.conv1.weight", i)),
            Value::List(vec![Value::F64(0.01); 64 * 64 * 3 * 3])
        );
        state_dict.insert(
            HashableValue::String(format!("residual_block_{}.conv2.weight", i)),
            Value::List(vec![Value::F64(0.01); 64 * 64 * 3 * 3])
        );
    }
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    let mut converter = ModelConverter::new();
    converter.load_pytorch(&model_path).unwrap();
    
    let stats = converter.get_conversion_stats();
    assert_eq!(stats.get("architecture"), Some(&"srresnet_deep".to_string()));
}

/// Test model with storage type markers
#[test]
fn test_storage_type_markers() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("storage_types.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // FloatStorage type
    let mut float_storage = BTreeMap::new();
    float_storage.insert(
        HashableValue::String("_type".to_string()),
        Value::String("torch.FloatStorage".to_string())
    );
    float_storage.insert(
        HashableValue::String("data".to_string()),
        Value::List(vec![Value::F64(0.1), Value::F64(0.2)])
    );
    
    state_dict.insert(
        HashableValue::String("layer1.weight".to_string()),
        Value::Dict(float_storage)
    );
    
    // DoubleStorage type
    let mut double_storage = BTreeMap::new();
    double_storage.insert(
        HashableValue::String("_type".to_string()),
        Value::String("torch.DoubleStorage".to_string())
    );
    double_storage.insert(
        HashableValue::String("data".to_string()),
        Value::List(vec![Value::F64(0.3), Value::F64(0.4)])
    );
    
    state_dict.insert(
        HashableValue::String("layer2.weight".to_string()),
        Value::Dict(double_storage)
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load model with storage types: {:?}", result);
}

/// Test handling of empty tensors
#[test]
fn test_empty_tensors() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("empty_tensors.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Valid tensor
    state_dict.insert(
        HashableValue::String("valid.weight".to_string()),
        Value::List(vec![Value::F64(0.1), Value::F64(0.2)])
    );
    
    // Empty tensor
    state_dict.insert(
        HashableValue::String("empty.weight".to_string()),
        Value::List(vec![])
    );
    
    // Empty bytes
    state_dict.insert(
        HashableValue::String("empty_bytes.weight".to_string()),
        Value::Bytes(vec![])
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    // Should succeed but only have the valid tensor
    assert!(result.is_ok(), "Failed to handle empty tensors: {:?}", result);
    
    let stats = converter.get_conversion_stats();
    let param_count: usize = stats.get("param_count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    
    assert_eq!(param_count, 2, "Should only count valid parameters");
}

/// Test handling of NaN and Inf values
#[test]
fn test_nan_inf_handling() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("nan_inf.pth");
    
    let mut state_dict = BTreeMap::new();
    
    state_dict.insert(
        HashableValue::String("layer.weight".to_string()),
        Value::List(vec![
            Value::F64(0.1),
            Value::F64(f64::NAN),
            Value::F64(f64::INFINITY),
            Value::F64(f64::NEG_INFINITY),
            Value::F64(0.2)
        ])
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    // Should load but with warnings about NaN/Inf
    assert!(result.is_ok(), "Failed to handle NaN/Inf values: {:?}", result);
}