use srgan_rust::model_converter::{ModelConverter, ModelFormat};
use std::path::Path;

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