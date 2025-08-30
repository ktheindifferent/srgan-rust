use clap::{App, Arg};
use srgan_rust::commands::convert_model;
use srgan_rust::error::SrganError;
use std::fs::File;
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_missing_input_argument() {
    // Create ArgMatches without input argument
    let app = App::new("test")
        .arg(Arg::with_name("input")
            .long("input")
            .value_name("INPUT")
            .takes_value(true))
        .arg(Arg::with_name("output")
            .long("output")
            .value_name("OUTPUT")
            .takes_value(true))
        .arg(Arg::with_name("format")
            .long("format")
            .value_name("FORMAT")
            .takes_value(true));
    
    let matches = app.get_matches_from(vec!["test"]);
    
    let result = convert_model(&matches);
    
    assert!(result.is_err());
    match result.unwrap_err() {
        SrganError::InvalidParameter(msg) => {
            assert!(msg.contains("Input path is required"), "Expected 'Input path is required', got: {}", msg);
        }
        err => panic!("Expected InvalidParameter error, got: {:?}", err),
    }
}

#[test]
fn test_invalid_format_argument() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.pth");
    File::create(&input_path).unwrap();
    
    let app = App::new("test")
        .arg(Arg::with_name("input")
            .long("input")
            .value_name("INPUT")
            .takes_value(true))
        .arg(Arg::with_name("output")
            .long("output")
            .value_name("OUTPUT")
            .takes_value(true))
        .arg(Arg::with_name("format")
            .long("format")
            .value_name("FORMAT")
            .takes_value(true));
    
    let matches = app.get_matches_from(vec![
        "test",
        "--input", input_path.to_str().unwrap(),
        "--format", "invalid_format"
    ]);
    
    let result = convert_model(&matches);
    
    assert!(result.is_err());
    match result.unwrap_err() {
        SrganError::InvalidInput(msg) => {
            assert!(msg.contains("Invalid model format 'invalid_format'"), 
                   "Expected error about invalid format, got: {}", msg);
            assert!(msg.contains("Valid formats are: pytorch, tensorflow, onnx, keras"),
                   "Expected list of valid formats in error message");
        }
        err => panic!("Expected InvalidInput error, got: {:?}", err),
    }
}

#[test]
fn test_nonexistent_input_file() {
    let app = App::new("test")
        .arg(Arg::with_name("input")
            .long("input")
            .value_name("INPUT")
            .takes_value(true))
        .arg(Arg::with_name("output")
            .long("output")
            .value_name("OUTPUT")
            .takes_value(true))
        .arg(Arg::with_name("format")
            .long("format")
            .value_name("FORMAT")
            .takes_value(true));
    
    let matches = app.get_matches_from(vec![
        "test",
        "--input", "/nonexistent/path/to/model.pth"
    ]);
    
    let result = convert_model(&matches);
    
    assert!(result.is_err());
    match result.unwrap_err() {
        SrganError::FileNotFound(path) => {
            assert_eq!(path, Path::new("/nonexistent/path/to/model.pth"));
        }
        err => panic!("Expected FileNotFound error, got: {:?}", err),
    }
}

#[test]
fn test_batch_mode_with_file_instead_of_directory() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("file.pth");
    File::create(&file_path).unwrap();
    
    let app = App::new("test")
        .arg(Arg::with_name("input")
            .long("input")
            .value_name("INPUT")
            .takes_value(true))
        .arg(Arg::with_name("output")
            .long("output")
            .value_name("OUTPUT")
            .takes_value(true))
        .arg(Arg::with_name("batch")
            .long("batch")
            .takes_value(false));
    
    let matches = app.get_matches_from(vec![
        "test",
        "--input", file_path.to_str().unwrap(),
        "--batch"
    ]);
    
    let result = convert_model(&matches);
    
    assert!(result.is_err());
    match result.unwrap_err() {
        SrganError::InvalidInput(msg) => {
            assert!(msg.contains("Batch mode requires input to be a directory"),
                   "Expected batch mode directory error, got: {}", msg);
        }
        err => panic!("Expected InvalidInput error, got: {:?}", err),
    }
}

#[test]
fn test_valid_formats() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test each valid format doesn't cause parse errors
    let valid_formats = vec!["pytorch", "tensorflow", "onnx", "keras"];
    
    for format in valid_formats {
        let input_path = temp_dir.path().join(format!("test_{}.model", format));
        File::create(&input_path).unwrap();
        
        let app = App::new("test")
            .arg(Arg::with_name("input")
                .long("input")
                .value_name("INPUT")
                .takes_value(true))
            .arg(Arg::with_name("output")
                .long("output")
                .value_name("OUTPUT")
                .takes_value(true))
            .arg(Arg::with_name("format")
                .long("format")
                .value_name("FORMAT")
                .takes_value(true));
        
        let matches = app.get_matches_from(vec![
            "test",
            "--input", input_path.to_str().unwrap(),
            "--format", format
        ]);
        
        // The function will fail because the file is empty, but it should parse the format correctly
        let result = convert_model(&matches);
        
        // We expect an error, but not a format parsing error
        assert!(result.is_err());
        match result.unwrap_err() {
            SrganError::InvalidInput(msg) if msg.contains("Invalid model format") => {
                panic!("Format '{}' should be valid but got error: {}", format, msg);
            }
            _ => {} // Other errors are expected (file parsing, etc.)
        }
    }
}

#[test]
fn test_optional_output_uses_default() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.pth");
    
    // Create a minimal valid pickle file
    File::create(&input_path).unwrap();
    
    let app = App::new("test")
        .arg(Arg::with_name("input")
            .long("input")
            .value_name("INPUT")
            .takes_value(true))
        .arg(Arg::with_name("output")
            .long("output")
            .value_name("OUTPUT")
            .takes_value(true));
    
    let matches = app.get_matches_from(vec![
        "test",
        "--input", input_path.to_str().unwrap()
    ]);
    
    // Should not panic when output is not provided (uses default "converted_model.rsr")
    let result = convert_model(&matches);
    
    // Will fail for other reasons (invalid pickle), but shouldn't panic on missing output
    assert!(result.is_err());
    assert!(!matches!(result.unwrap_err(), SrganError::InvalidParameter(_)));
}

#[test]
fn test_mixed_case_format_argument() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.pth");
    File::create(&input_path).unwrap();
    
    // Test mixed case format names (should fail with proper error)
    let mixed_cases = vec!["PyTorch", "PYTORCH", "TensorFlow", "Onnx", "KERAS"];
    
    for format in mixed_cases {
        let app = App::new("test")
            .arg(Arg::with_name("input")
                .long("input")
                .value_name("INPUT")
                .takes_value(true))
            .arg(Arg::with_name("format")
                .long("format")
                .value_name("FORMAT")
                .takes_value(true));
        
        let matches = app.get_matches_from(vec![
            "test",
            "--input", input_path.to_str().unwrap(),
            "--format", format
        ]);
        
        let result = convert_model(&matches);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            SrganError::InvalidInput(msg) => {
                assert!(msg.contains(&format!("Invalid model format '{}'", format)),
                       "Expected error for format '{}', got: {}", format, msg);
            }
            err => panic!("Expected InvalidInput error for format '{}', got: {:?}", format, err),
        }
    }
}

#[test]
fn test_empty_format_string() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.pth");
    File::create(&input_path).unwrap();
    
    let app = App::new("test")
        .arg(Arg::with_name("input")
            .long("input")
            .value_name("INPUT")
            .takes_value(true))
        .arg(Arg::with_name("format")
            .long("format")
            .value_name("FORMAT")
            .takes_value(true)
            .empty_values(false));
    
    // Clap should prevent empty values if configured properly
    let result = app.get_matches_from_safe(vec![
        "test",
        "--input", input_path.to_str().unwrap(),
        "--format", ""
    ]);
    
    assert!(result.is_err()); // Clap should reject empty format value
}