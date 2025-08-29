use srgan_rust::model_converter::ModelConverter;
use std::path::Path;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;
use tempfile::TempDir;
use serde_pickle::{value_to_vec, Value, HashableValue, SerOptions};
use std::collections::BTreeMap;

/// Create a realistic SRGAN generator model for testing
fn create_srgan_generator_model() -> Value {
    let mut state_dict = BTreeMap::new();
    
    // Initial convolution block
    state_dict.insert(
        HashableValue::String("conv_input.0.weight".to_string()),
        Value::List(vec![Value::F64(0.01); 64 * 3 * 9 * 9])
    );
    state_dict.insert(
        HashableValue::String("conv_input.0.bias".to_string()),
        Value::List(vec![Value::F64(0.0); 64])
    );
    
    // Residual blocks (reduced from 16 to 2 for testing)
    for i in 0..2 {
        let prefix = format!("residual_blocks.{}", i);
        
        // First conv in residual block
        state_dict.insert(
            HashableValue::String(format!("{}.conv1.weight", prefix)),
            Value::List(vec![Value::F64(0.01); 64 * 64 * 3 * 3])
        );
        state_dict.insert(
            HashableValue::String(format!("{}.conv1.bias", prefix)),
            Value::List(vec![Value::F64(0.0); 64])
        );
        
        // Batch norm 1
        state_dict.insert(
            HashableValue::String(format!("{}.bn1.weight", prefix)),
            Value::List(vec![Value::F64(1.0); 64])
        );
        state_dict.insert(
            HashableValue::String(format!("{}.bn1.bias", prefix)),
            Value::List(vec![Value::F64(0.0); 64])
        );
        state_dict.insert(
            HashableValue::String(format!("{}.bn1.running_mean", prefix)),
            Value::List(vec![Value::F64(0.0); 64])
        );
        state_dict.insert(
            HashableValue::String(format!("{}.bn1.running_var", prefix)),
            Value::List(vec![Value::F64(1.0); 64])
        );
        
        // Second conv in residual block
        state_dict.insert(
            HashableValue::String(format!("{}.conv2.weight", prefix)),
            Value::List(vec![Value::F64(0.01); 64 * 64 * 3 * 3])
        );
        state_dict.insert(
            HashableValue::String(format!("{}.conv2.bias", prefix)),
            Value::List(vec![Value::F64(0.0); 64])
        );
        
        // Batch norm 2
        state_dict.insert(
            HashableValue::String(format!("{}.bn2.weight", prefix)),
            Value::List(vec![Value::F64(1.0); 64])
        );
        state_dict.insert(
            HashableValue::String(format!("{}.bn2.bias", prefix)),
            Value::List(vec![Value::F64(0.0); 64])
        );
    }
    
    // Mid convolution
    state_dict.insert(
        HashableValue::String("conv_mid.0.weight".to_string()),
        Value::List(vec![Value::F64(0.01); 64 * 64 * 3 * 3])
    );
    state_dict.insert(
        HashableValue::String("conv_mid.0.bias".to_string()),
        Value::List(vec![Value::F64(0.0); 64])
    );
    
    // Upsampling blocks (2x for 4x upscaling)
    for i in 0..2 {
        let prefix = format!("upsampling.{}", i);
        state_dict.insert(
            HashableValue::String(format!("{}.weight", prefix)),
            Value::List(vec![Value::F64(0.01); 256 * 64 * 3 * 3])
        );
        state_dict.insert(
            HashableValue::String(format!("{}.bias", prefix)),
            Value::List(vec![Value::F64(0.0); 256])
        );
    }
    
    // Output convolution
    state_dict.insert(
        HashableValue::String("conv_output.weight".to_string()),
        Value::List(vec![Value::F64(0.01); 3 * 64 * 9 * 9])
    );
    state_dict.insert(
        HashableValue::String("conv_output.bias".to_string()),
        Value::List(vec![Value::F64(0.0); 3])
    );
    
    // Add metadata
    state_dict.insert(
        HashableValue::String("_metadata".to_string()),
        Value::Dict(BTreeMap::new())
    );
    
    Value::Dict(state_dict)
}

/// Create a realistic SRGAN discriminator model
fn create_srgan_discriminator_model() -> Value {
    let mut state_dict = BTreeMap::new();
    
    // VGG-style discriminator layers
    let channels = [(3, 64), (64, 64), (64, 128), (128, 128), 
                    (128, 256), (256, 256), (256, 512), (512, 512)];
    
    for (i, (in_ch, out_ch)) in channels.iter().enumerate() {
        let layer_name = format!("features.{}", i * 3); // Conv layers are at indices 0, 3, 6, ...
        
        // Convolution layer
        state_dict.insert(
            HashableValue::String(format!("{}.weight", layer_name)),
            Value::List(vec![Value::F64(0.01); out_ch * in_ch * 3 * 3])
        );
        state_dict.insert(
            HashableValue::String(format!("{}.bias", layer_name)),
            Value::List(vec![Value::F64(0.0); *out_ch])
        );
        
        // Batch norm (except first layer)
        if i > 0 {
            let bn_name = format!("features.{}", i * 3 + 1);
            state_dict.insert(
                HashableValue::String(format!("{}.weight", bn_name)),
                Value::List(vec![Value::F64(1.0); *out_ch])
            );
            state_dict.insert(
                HashableValue::String(format!("{}.bias", bn_name)),
                Value::List(vec![Value::F64(0.0); *out_ch])
            );
        }
    }
    
    // Classifier layers (reduced size for testing)
    state_dict.insert(
        HashableValue::String("classifier.0.weight".to_string()),
        Value::List(vec![Value::F64(0.01); 1024])
    );
    state_dict.insert(
        HashableValue::String("classifier.0.bias".to_string()),
        Value::List(vec![Value::F64(0.0); 1024])
    );
    
    state_dict.insert(
        HashableValue::String("classifier.3.weight".to_string()),
        Value::List(vec![Value::F64(0.01); 1 * 1024])
    );
    state_dict.insert(
        HashableValue::String("classifier.3.bias".to_string()),
        Value::List(vec![Value::F64(0.0); 1])
    );
    
    Value::Dict(state_dict)
}

#[test]
fn test_srgan_generator_integration() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("srgan_generator.pth");
    
    // Create and save SRGAN generator model
    let state_dict = create_srgan_generator_model();
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load with converter
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load SRGAN generator: {:?}", result);
    
    // Verify model properties
    let stats = converter.get_conversion_stats();
    assert_eq!(stats.get("format"), Some(&"pytorch".to_string()));
    assert_eq!(stats.get("architecture"), Some(&"srresnet".to_string())); // Detected as srresnet due to residual blocks
    
    // Check that we loaded the expected number of parameters
    let param_count: usize = stats.get("param_count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(param_count > 100, "Expected many parameters, got {}", param_count);
}

#[test]
fn test_srgan_discriminator_integration() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("srgan_discriminator.pth");
    
    // Create and save SRGAN discriminator model
    let state_dict = create_srgan_discriminator_model();
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load with converter
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load SRGAN discriminator: {:?}", result);
    
    // Verify model properties
    let stats = converter.get_conversion_stats();
    assert_eq!(stats.get("format"), Some(&"pytorch".to_string()));
}

#[test]
fn test_full_srgan_model_integration() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("srgan_full.pth");
    
    // Create combined model with both generator and discriminator
    let mut full_model = BTreeMap::new();
    
    // Add generator
    if let Value::Dict(gen_dict) = create_srgan_generator_model() {
        for (k, v) in gen_dict {
            if let HashableValue::String(key) = k {
                full_model.insert(
                    HashableValue::String(format!("generator.{}", key)),
                    v
                );
            }
        }
    }
    
    // Add discriminator
    if let Value::Dict(disc_dict) = create_srgan_discriminator_model() {
        for (k, v) in disc_dict {
            if let HashableValue::String(key) = k {
                full_model.insert(
                    HashableValue::String(format!("discriminator.{}", key)),
                    v
                );
            }
        }
    }
    
    let state_dict = Value::Dict(full_model);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load with converter
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load full SRGAN model: {:?}", result);
    
    // Verify model properties
    let stats = converter.get_conversion_stats();
    assert_eq!(stats.get("format"), Some(&"pytorch".to_string()));
    assert_eq!(stats.get("architecture"), Some(&"srgan_full".to_string()));
}

#[test]
fn test_model_with_quantized_weights() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("quantized.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Simulate int8 quantized weights stored as bytes
    let quantized_weights: Vec<i8> = vec![-128, -64, 0, 64, 127];
    let mut bytes = Vec::new();
    for w in &quantized_weights {
        bytes.push(*w as u8);
    }
    
    // Add quantized layer
    state_dict.insert(
        HashableValue::String("quantized_layer.weight".to_string()),
        Value::Bytes(bytes.clone())
    );
    
    // Add scale and zero point for dequantization
    state_dict.insert(
        HashableValue::String("quantized_layer.scale".to_string()),
        Value::F64(0.01)
    );
    state_dict.insert(
        HashableValue::String("quantized_layer.zero_point".to_string()),
        Value::I64(128)
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load with converter - should handle or gracefully fail on quantized models
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    // The converter should either successfully load or provide a meaningful error
    if result.is_err() {
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("byte length") || err_msg.contains("tensor"),
                "Expected tensor-related error, got: {}", err_msg);
    }
}

#[test]
#[ignore] // Skip this test by default as it uses a lot of memory
fn test_performance_large_model() {
    use std::time::Instant;
    
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("large_model.pth");
    
    let mut state_dict = BTreeMap::new();
    
    // Create a large model with many parameters
    for i in 0..100 {
        let layer_name = format!("layer_{}.weight", i);
        // Each layer has 1000x1000 parameters = 1M parameters
        state_dict.insert(
            HashableValue::String(layer_name),
            Value::List(vec![Value::F64(0.01); 1_000_000])
        );
    }
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Measure loading time
    let start = Instant::now();
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Failed to load large model: {:?}", result);
    
    // Check that loading completes in reasonable time (< 10 seconds for 100M parameters)
    assert!(duration.as_secs() < 10, 
            "Loading took too long: {:?} seconds", duration.as_secs());
    
    println!("Loaded 100M parameter model in {:?}", duration);
}