use srgan_rust::model_converter::{ModelConverter, ModelFormat};
use std::path::Path;
use std::fs::{self, File};
use std::io::Write;
use tempfile::TempDir;
use serde_pickle::{value_to_vec, Value, HashableValue, SerOptions};
use std::collections::BTreeMap;

#[test]
fn test_model_format_detection() {
    // Test PyTorch format detection
    assert!(matches!(
        ModelConverter::auto_detect_format(Path::new("model.pth")),
        Ok(ModelFormat::PyTorch)
    ));
    
    assert!(matches!(
        ModelConverter::auto_detect_format(Path::new("model.pt")),
        Ok(ModelFormat::PyTorch)
    ));
    
    // Test TensorFlow format detection
    assert!(matches!(
        ModelConverter::auto_detect_format(Path::new("model.pb")),
        Ok(ModelFormat::TensorFlow)
    ));
    
    // Test ONNX format detection
    assert!(matches!(
        ModelConverter::auto_detect_format(Path::new("model.onnx")),
        Ok(ModelFormat::ONNX)
    ));
    
    // Test Keras format detection
    assert!(matches!(
        ModelConverter::auto_detect_format(Path::new("model.h5")),
        Ok(ModelFormat::Keras)
    ));
    
    assert!(matches!(
        ModelConverter::auto_detect_format(Path::new("model.hdf5")),
        Ok(ModelFormat::Keras)
    ));
    
    // Test unknown format
    assert!(ModelConverter::auto_detect_format(Path::new("model.unknown")).is_err());
}

#[test]
fn test_model_converter_creation() {
    let converter = ModelConverter::new();
    let stats = converter.get_conversion_stats();
    assert!(stats.is_empty() || !stats.contains_key("format"));
}

#[test]
fn test_model_converter_with_metadata() {
    let mut converter = ModelConverter::new();
    
    // Initially no stats
    let stats = converter.get_conversion_stats();
    assert!(!stats.contains_key("format"));
    
    // After loading (simulated), stats should be available
    // This would require mocking or test fixtures in a real implementation
}

#[test]
fn test_invalid_model_paths() {
    let mut converter = ModelConverter::new();
    
    // Test non-existent file
    let result = converter.load_pytorch(Path::new("/nonexistent/model.pth"));
    assert!(result.is_err());
    
    // Test non-existent directory for TensorFlow
    let result = converter.load_tensorflow(Path::new("/nonexistent/saved_model"));
    assert!(result.is_err());
}

// Helper function to create a simple PyTorch model state dict
fn create_test_state_dict() -> Value {
    let mut state_dict = BTreeMap::new();
    
    // Add some test weights (3x3 conv kernel with 64 filters)
    let conv1_weights = vec![0.1_f64; 64 * 3 * 3 * 3]; // 64 filters, 3 input channels, 3x3 kernel
    state_dict.insert(
        HashableValue::String("conv1.weight".to_string()),
        Value::List(conv1_weights.into_iter().map(Value::F64).collect())
    );
    
    // Add bias
    let conv1_bias = vec![0.01_f64; 64];
    state_dict.insert(
        HashableValue::String("conv1.bias".to_string()),
        Value::List(conv1_bias.into_iter().map(Value::F64).collect())
    );
    
    // Add batch norm parameters
    state_dict.insert(
        HashableValue::String("bn1.weight".to_string()),
        Value::List(vec![Value::F64(1.0); 64])
    );
    
    state_dict.insert(
        HashableValue::String("bn1.bias".to_string()),
        Value::List(vec![Value::F64(0.0); 64])
    );
    
    Value::Dict(state_dict)
}

#[test]
fn test_pytorch_simple_model_loading() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.pth");
    
    // Create a simple state dict and serialize it
    let state_dict = create_test_state_dict();
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    // Write to file
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load with converter
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load PyTorch model: {:?}", result);
    
    // Check metadata
    let stats = converter.get_conversion_stats();
    assert!(stats.contains_key("format"));
    assert_eq!(stats.get("format"), Some(&"pytorch".to_string()));
}

#[test]
fn test_pytorch_model_with_nested_structure() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("nested_model.pth");
    
    // Create nested structure (generator + discriminator)
    let mut model_dict = BTreeMap::new();
    
    // Generator weights
    let mut gen_dict = BTreeMap::new();
    gen_dict.insert(
        HashableValue::String("conv1.weight".to_string()),
        Value::List(vec![Value::F64(0.1); 256])
    );
    gen_dict.insert(
        HashableValue::String("conv2.weight".to_string()),
        Value::List(vec![Value::F64(0.2); 512])
    );
    
    model_dict.insert(
        HashableValue::String("generator".to_string()),
        Value::Dict(gen_dict)
    );
    
    // Discriminator weights
    let mut disc_dict = BTreeMap::new();
    disc_dict.insert(
        HashableValue::String("conv1.weight".to_string()),
        Value::List(vec![Value::F64(0.3); 128])
    );
    
    model_dict.insert(
        HashableValue::String("discriminator".to_string()),
        Value::Dict(disc_dict)
    );
    
    let state_dict = Value::Dict(model_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load nested model: {:?}", result);
}

#[test]
fn test_pytorch_empty_file() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("empty.pth");
    
    // Create empty file
    File::create(&model_path).unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too small"));
}

#[test]
fn test_pytorch_corrupt_pickle() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("corrupt.pth");
    
    // Write invalid pickle data
    let mut file = File::create(&model_path).unwrap();
    file.write_all(b"This is not valid pickle data at all!").unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("parse"));
}

#[test]
fn test_pytorch_model_with_different_dtypes() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("mixed_types.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Float32 weights
    state_dict.insert(
        HashableValue::String("layer1.weight".to_string()),
        Value::List(vec![Value::F64(0.5); 100])
    );
    
    // Float64 weights (should be converted to f32)
    state_dict.insert(
        HashableValue::String("layer2.weight".to_string()),
        Value::List(vec![Value::F64(0.7); 100])
    );
    
    // Integer weights (should be converted to f32)
    state_dict.insert(
        HashableValue::String("layer3.weight".to_string()),
        Value::List(vec![Value::I64(1); 100])
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load mixed dtype model: {:?}", result);
}

#[test]
fn test_pytorch_model_with_bytes_tensor() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("bytes_tensor.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Create raw bytes representation of float32 array
    let weights: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let mut bytes = Vec::new();
    for w in weights {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    
    state_dict.insert(
        HashableValue::String("layer.weight".to_string()),
        Value::Bytes(bytes)
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load bytes tensor model: {:?}", result);
}

#[test]
fn test_pytorch_zip_format_detection() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("zipped.pth");
    
    // Write ZIP magic bytes
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&[0x50, 0x4B, 0x03, 0x04]).unwrap();
    file.write_all(b"rest of zip file...").unwrap();
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("ZIP format"));
}

#[test]
fn test_pytorch_model_architecture_detection() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test SRGAN full model
    {
        let model_path = temp_dir.path().join("srgan_full.pth");
        let mut state_dict = BTreeMap::new();
        
        state_dict.insert(
            HashableValue::String("generator.conv1.weight".to_string()),
            Value::List(vec![Value::F64(0.1); 100])
        );
        state_dict.insert(
            HashableValue::String("discriminator.conv1.weight".to_string()),
            Value::List(vec![Value::F64(0.1); 100])
        );
        
        let state_dict = Value::Dict(state_dict);
        let pickled = value_to_vec(&state_dict, SerOptions::new()).unwrap();
        fs::write(&model_path, pickled).unwrap();
        
        let mut converter = ModelConverter::new();
        converter.load_pytorch(&model_path).unwrap();
        
        let stats = converter.get_conversion_stats();
        assert_eq!(stats.get("architecture"), Some(&"srgan_full".to_string()));
    }
    
    // Test SRResNet model
    {
        let model_path = temp_dir.path().join("srresnet.pth");
        let mut state_dict = BTreeMap::new();
        
        state_dict.insert(
            HashableValue::String("residual_block1.conv1.weight".to_string()),
            Value::List(vec![Value::F64(0.1); 100])
        );
        
        let state_dict = Value::Dict(state_dict);
        let pickled = value_to_vec(&state_dict, SerOptions::new()).unwrap();
        fs::write(&model_path, pickled).unwrap();
        
        let mut converter = ModelConverter::new();
        converter.load_pytorch(&model_path).unwrap();
        
        let stats = converter.get_conversion_stats();
        assert_eq!(stats.get("architecture"), Some(&"srresnet".to_string()));
    }
}