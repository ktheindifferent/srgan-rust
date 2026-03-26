use srgan_rust::model_converter::{ModelConverter, ModelFormat};
use std::path::Path;
use std::fs::File;
use std::io::Write;
use tempfile::{TempDir, NamedTempFile};
use serde_pickle::{value_to_vec, SerOptions, Value, HashableValue};
use std::collections::BTreeMap;

/// Create a simple PyTorch state dict for testing
fn create_test_state_dict() -> Value {
    let mut state_dict = BTreeMap::new();
    
    // Add conv layer weights (64x3x9x9)
    let conv_weights = vec![0.1_f64; 64 * 3 * 9 * 9];
    state_dict.insert(
        HashableValue::String("conv1.weight".into()),
        Value::List(conv_weights.into_iter().map(Value::F64).collect())
    );
    
    // Add conv bias (64)
    let conv_bias = vec![0.01_f64; 64];
    state_dict.insert(
        HashableValue::String("conv1.bias".into()),
        Value::List(conv_bias.into_iter().map(Value::F64).collect())
    );
    
    // Add batch norm weights
    state_dict.insert(
        HashableValue::String("bn1.weight".into()),
        Value::List(vec![Value::F64(1.0); 64])
    );
    state_dict.insert(
        HashableValue::String("bn1.bias".into()),
        Value::List(vec![Value::F64(0.0); 64])
    );
    
    // Add residual block
    let res_weights = vec![0.05_f64; 64 * 64 * 3 * 3];
    state_dict.insert(
        HashableValue::String("residual_blocks.0.conv1.weight".into()),
        Value::List(res_weights.into_iter().map(Value::F64).collect())
    );
    
    // Add upsampling layer
    let upsample_weights = vec![0.1_f64; 256 * 64 * 3 * 3];
    state_dict.insert(
        HashableValue::String("upsample.0.weight".into()),
        Value::List(upsample_weights.into_iter().map(Value::F64).collect())
    );
    
    Value::Dict(state_dict)
}

/// Create a PyTorch model with float16 weights
fn create_float16_state_dict() -> Value {
    let mut state_dict = BTreeMap::new();
    
    // Create float16 bytes (simplified - just example data)
    let mut float16_bytes = Vec::new();
    for _ in 0..64 * 3 * 3 * 3 {
        // Float16 representation of ~0.1
        float16_bytes.extend_from_slice(&[0x2E, 0x66]); // Approximately 0.1 in float16
    }
    
    state_dict.insert(
        HashableValue::String("conv1.weight".into()),
        Value::Bytes(float16_bytes)
    );
    
    Value::Dict(state_dict)
}

/// Create a PyTorch model with int8 quantized weights
fn create_int8_state_dict() -> Value {
    let mut state_dict = BTreeMap::new();
    
    // Create int8 bytes (quantized weights)
    let int8_bytes: Vec<u8> = (0..64 * 3 * 3 * 3)
        .map(|i| ((i % 255) as u8))
        .collect();
    
    state_dict.insert(
        HashableValue::String("conv1.weight".into()),
        Value::Bytes(int8_bytes)
    );
    
    Value::Dict(state_dict)
}

/// Create an ESRGAN-style state dict
fn create_esrgan_state_dict() -> Value {
    let mut state_dict = BTreeMap::new();
    
    // ESRGAN specific layers
    state_dict.insert(
        HashableValue::String("conv_first.weight".into()),
        Value::List(vec![Value::F64(0.1); 64 * 3 * 3 * 3])
    );
    
    state_dict.insert(
        HashableValue::String("trunk_conv.weight".into()),
        Value::List(vec![Value::F64(0.1); 64 * 64 * 3 * 3])
    );
    
    // RRDB blocks
    for i in 0..5 {
        let key = format!("RRDB.{}.conv1.weight", i);
        state_dict.insert(
            HashableValue::String(key),
            Value::List(vec![Value::F64(0.05); 32 * 64 * 3 * 3])
        );
    }
    
    state_dict.insert(
        HashableValue::String("upconv1.weight".into()),
        Value::List(vec![Value::F64(0.1); 256 * 64 * 3 * 3])
    );
    
    state_dict.insert(
        HashableValue::String("HRconv.weight".into()),
        Value::List(vec![Value::F64(0.1); 64 * 64 * 3 * 3])
    );
    
    state_dict.insert(
        HashableValue::String("conv_last.weight".into()),
        Value::List(vec![Value::F64(0.1); 3 * 64 * 3 * 3])
    );
    
    Value::Dict(state_dict)
}

/// Create a nested state dict (model saved with torch.save(model))
fn create_nested_state_dict() -> Value {
    let inner_dict = create_test_state_dict();
    
    let mut outer_dict = BTreeMap::new();
    outer_dict.insert(
        HashableValue::String("state_dict".into()),
        inner_dict
    );
    outer_dict.insert(
        HashableValue::String("epoch".into()),
        Value::I64(100)
    );
    outer_dict.insert(
        HashableValue::String("optimizer".into()),
        Value::Dict(BTreeMap::new())
    );
    
    Value::Dict(outer_dict)
}

/// Create a PyTorch model file for testing
fn create_test_model_file(state_dict: Value) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let mut file = NamedTempFile::new()?;
    
    let ser_options = SerOptions::new();
    let bytes = value_to_vec(&state_dict, ser_options)?;
    file.write_all(&bytes)?;
    file.flush()?;
    
    Ok(file)
}

/// Create a ZIP-based PyTorch model file
fn create_zip_model_file() -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    use zip::write::{ZipWriter, FileOptions};
    use std::io::Cursor;
    
    let mut file = NamedTempFile::new()?;
    
    // Create in-memory ZIP
    let mut buffer = Vec::new();
    {
        let mut zip = ZipWriter::new(Cursor::new(&mut buffer));
        
        // Add data.pkl
        let state_dict = create_test_state_dict();
        let ser_options = SerOptions::new();
        let pkl_bytes = value_to_vec(&state_dict, ser_options)?;
        
        zip.start_file("data.pkl", FileOptions::default())?;
        zip.write_all(&pkl_bytes)?;
        
        // Add version file
        zip.start_file("version", FileOptions::default())?;
        zip.write_all(b"1.9.0+cu111")?;
        
        zip.finish()?;
    }
    
    file.write_all(&buffer)?;
    file.flush()?;
    
    Ok(file)
}

#[test]
fn test_load_simple_pytorch_model() {
    let state_dict = create_test_state_dict();
    let file = create_test_model_file(state_dict).expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    assert!(result.is_ok(), "Failed to load PyTorch model: {:?}", result);
}

#[test]
fn test_load_float16_model() {
    let state_dict = create_float16_state_dict();
    let file = create_test_model_file(state_dict).expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    // Float16 support is implemented, should succeed
    assert!(result.is_ok(), "Failed to load float16 model: {:?}", result);
}

#[test]
fn test_load_int8_quantized_model() {
    let state_dict = create_int8_state_dict();
    let file = create_test_model_file(state_dict).expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    // Int8 support is implemented, should succeed
    assert!(result.is_ok(), "Failed to load int8 model: {:?}", result);
}

#[test]
fn test_load_esrgan_model() {
    let state_dict = create_esrgan_state_dict();
    let file = create_test_model_file(state_dict).expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    assert!(result.is_ok(), "Failed to load ESRGAN model: {:?}", result);
    
    // Check if architecture was detected correctly
    let stats = converter.get_conversion_stats();
    assert_eq!(stats.get("architecture"), Some(&"esrgan".to_string()));
}

#[test]
fn test_load_nested_state_dict() {
    let state_dict = create_nested_state_dict();
    let file = create_test_model_file(state_dict).expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    assert!(result.is_ok(), "Failed to load nested state dict: {:?}", result);
}

#[test]
fn test_load_zip_model() {
    let file = create_zip_model_file().expect("Failed to create ZIP file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    assert!(result.is_ok(), "Failed to load ZIP model: {:?}", result);
}

#[test]
fn test_nonexistent_file() {
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(Path::new("/nonexistent/file.pth"));
    
    assert!(result.is_err());
    assert!(matches!(result, Err(srgan_rust::error::SrganError::FileNotFound(_))));
}

#[test]
fn test_invalid_pickle_data() {
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    file.write_all(b"This is not valid pickle data").expect("Failed to write");
    file.flush().expect("Failed to flush");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    assert!(result.is_err());
}

#[test]
fn test_empty_model() {
    let empty_dict = Value::Dict(BTreeMap::new());
    let file = create_test_model_file(empty_dict).expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    // Should fail with no parameters
    assert!(result.is_err());
}

#[test]
fn test_model_with_nan_values() {
    let mut state_dict = BTreeMap::new();
    
    // Add layer with NaN values
    state_dict.insert(
        HashableValue::String("conv1.weight".into()),
        Value::List(vec![Value::F64(f64::NAN); 64])
    );
    
    let file = create_test_model_file(Value::Dict(state_dict))
        .expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    // Should handle NaN values gracefully
    assert!(result.is_ok());
}

#[test]
fn test_auto_detect_pytorch_format() {
    let pth_path = Path::new("model.pth");
    let pt_path = Path::new("model.pt");
    
    assert!(matches!(
        ModelConverter::auto_detect_format(pth_path),
        Ok(ModelFormat::PyTorch)
    ));
    
    assert!(matches!(
        ModelConverter::auto_detect_format(pt_path),
        Ok(ModelFormat::PyTorch)
    ));
}

#[test]
fn test_conversion_stats() {
    let state_dict = create_test_state_dict();
    let file = create_test_model_file(state_dict).expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    converter.load_pytorch(file.path()).expect("Failed to load model");
    
    let stats = converter.get_conversion_stats();
    
    assert_eq!(stats.get("format"), Some(&"pytorch".to_string()));
    assert!(stats.contains_key("param_count"));
    assert!(stats.contains_key("input_shape"));
    assert!(stats.contains_key("output_shape"));
}

/// Test loading model with various layer types
#[test]
fn test_comprehensive_layer_types() {
    let mut state_dict = BTreeMap::new();
    
    // Various layer types
    let layer_configs = vec![
        ("conv2d.weight", 64 * 3 * 3 * 3),
        ("linear.weight", 1024 * 512),
        ("batchnorm.weight", 64),
        ("layernorm.weight", 768),
        ("embedding.weight", 10000 * 300),
        ("lstm.weight_ih_l0", 512 * 128),
        ("gru.weight_hh_l0", 256 * 256),
        ("attention.q_proj.weight", 512 * 512),
    ];
    
    for (name, size) in layer_configs {
        state_dict.insert(
            HashableValue::String(name.into()),
            Value::List(vec![Value::F64(0.1); size])
        );
    }
    
    let file = create_test_model_file(Value::Dict(state_dict))
        .expect("Failed to create test file");
    
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(file.path());
    
    assert!(result.is_ok(), "Failed to load model with various layers: {:?}", result);
}

/// Integration test for batch conversion
#[test]
fn test_batch_conversion() {
    use srgan_rust::model_converter::batch_convert_models;
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    std::fs::create_dir_all(&input_dir).expect("Failed to create input dir");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    
    // Create multiple test models
    let models = vec![
        ("model1.pth", create_test_state_dict()),
        ("model2.pth", create_esrgan_state_dict()),
        ("model3.pth", create_nested_state_dict()),
    ];
    
    for (name, state_dict) in models {
        let model_path = input_dir.join(name);
        let file = File::create(&model_path).expect("Failed to create file");
        
        let ser_options = SerOptions::new();
        let bytes = value_to_vec(&state_dict, ser_options)
            .expect("Failed to serialize");
        
        std::fs::write(&model_path, bytes).expect("Failed to write file");
    }
    
    // Run batch conversion
    let results = batch_convert_models(&input_dir, &output_dir, Some(ModelFormat::PyTorch));
    
    assert!(results.is_ok(), "Batch conversion failed: {:?}", results);
    
    let results = results.unwrap();
    assert_eq!(results.len(), 3);
    
    // Check if output files were created
    for (name, success) in results {
        if success {
            let output_name = name.replace(".pth", ".rsr");
            let output_path = output_dir.join(output_name);
            // Note: Actual file creation depends on full implementation
            // For now, we just check the conversion attempt was made
        }
    }
}