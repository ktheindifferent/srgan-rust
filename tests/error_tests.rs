use srgan_rust::error::{Result, SrganError};
use std::io;
use std::path::PathBuf;

#[test]
fn test_io_error_conversion() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "test error");
    let srgan_err: SrganError = io_err.into();
    
    match srgan_err {
        SrganError::Io(_) => (),
        _ => panic!("Expected Io error variant"),
    }
}

#[test]
fn test_error_display() {
    let err = SrganError::InvalidParameter("test parameter".to_string());
    let display = format!("{}", err);
    assert_eq!(display, "Invalid parameter: test parameter");
    
    let err = SrganError::FileNotFound(PathBuf::from("/test/path"));
    let display = format!("{}", err);
    assert_eq!(display, "File not found: /test/path");
    
    let err = SrganError::Training("training failed".to_string());
    let display = format!("{}", err);
    assert_eq!(display, "Training error: training failed");
}

#[test]
fn test_parse_int_error_conversion() {
    let parse_err = "not a number".parse::<i32>().unwrap_err();
    let srgan_err: SrganError = parse_err.into();
    
    match srgan_err {
        SrganError::Parse(msg) => assert!(msg.contains("Failed to parse integer")),
        _ => panic!("Expected Parse error variant"),
    }
}

#[test]
fn test_parse_float_error_conversion() {
    let parse_err = "not a float".parse::<f32>().unwrap_err();
    let srgan_err: SrganError = parse_err.into();
    
    match srgan_err {
        SrganError::Parse(msg) => assert!(msg.contains("Failed to parse float")),
        _ => panic!("Expected Parse error variant"),
    }
}

#[test]
fn test_result_type() {
    fn test_function() -> Result<i32> {
        Ok(42)
    }
    
    let result = test_function();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_result_error() {
    fn test_function() -> Result<i32> {
        Err(SrganError::InvalidInput("test error".to_string()))
    }
    
    let result = test_function();
    assert!(result.is_err());
    
    match result.unwrap_err() {
        SrganError::InvalidInput(msg) => assert_eq!(msg, "test error"),
        _ => panic!("Expected InvalidInput error"),
    }
}