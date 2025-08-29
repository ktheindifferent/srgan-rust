use srgan_rust::model_converter::ModelConverter;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;
use serde_pickle::{value_to_vec, Value, HashableValue, SerOptions};
use std::collections::BTreeMap;

#[test]
fn test_pytorch_minimal_model() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("minimal.pth");
    
    // Create minimal model with just a few parameters
    let mut state_dict = BTreeMap::new();
    
    state_dict.insert(
        HashableValue::String("layer1.weight".to_string()),
        Value::List(vec![Value::F64(0.1), Value::F64(0.2), Value::F64(0.3)])
    );
    
    state_dict.insert(
        HashableValue::String("layer1.bias".to_string()),
        Value::List(vec![Value::F64(0.01)])
    );
    
    let state_dict = Value::Dict(state_dict);
    let ser_options = SerOptions::new();
    let pickled = value_to_vec(&state_dict, ser_options).unwrap();
    
    // Write to file
    let mut file = File::create(&model_path).unwrap();
    file.write_all(&pickled).unwrap();
    
    // Load with converter
    let mut converter = ModelConverter::new();
    let result = converter.load_pytorch(&model_path);
    
    assert!(result.is_ok(), "Failed to load minimal PyTorch model: {:?}", result);
    
    // Check metadata
    let stats = converter.get_conversion_stats();
    assert_eq!(stats.get("format"), Some(&"pytorch".to_string()));
    
    // Check parameter count
    let param_count: usize = stats.get("param_count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert_eq!(param_count, 4, "Expected 4 parameters, got {}", param_count);
}