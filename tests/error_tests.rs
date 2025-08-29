use srgan_rust::error::{Result, SrganError};
use std::io;
use std::path::PathBuf;

// Import test helpers for better error handling
#[path = "test_helpers.rs"]
mod test_helpers;
use test_helpers::*;

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
    let parse_result = "not a number".parse::<i32>();
    let parse_err = assert_err(parse_result, "parsing invalid integer");
    let srgan_err: SrganError = parse_err.into();
    
    match srgan_err {
        SrganError::Parse(msg) => assert_contains(&msg, "Failed to parse integer", "parse error message"),
        _ => panic!("Expected Parse error variant"),
    }
}

#[test]
fn test_parse_float_error_conversion() {
    let parse_result = "not a float".parse::<f32>();
    let parse_err = assert_err(parse_result, "parsing invalid float");
    let srgan_err: SrganError = parse_err.into();
    
    match srgan_err {
        SrganError::Parse(msg) => assert_contains(&msg, "Failed to parse float", "parse error message"),
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
    let value = assert_ok(result, "test function returning Ok");
    assert_eq!(value, 42);
}

#[test]
fn test_result_error() {
    fn test_function() -> Result<i32> {
        Err(SrganError::InvalidInput("test error".to_string()))
    }
    
    let result = test_function();
    assert!(result.is_err());
    
    let err = assert_err(result, "test function returning Err");
    match err {
        SrganError::InvalidInput(msg) => assert_eq!(msg, "test error"),
        _ => panic!("Expected InvalidInput error"),
    }
}

#[test]
fn test_error_chaining() {
    fn inner_function() -> Result<()> {
        Err(SrganError::InvalidInput("inner error".to_string()))
    }
    
    fn outer_function() -> Result<()> {
        inner_function()?;
        Ok(())
    }
    
    let result = outer_function();
    let err = assert_err(result, "chained error propagation");
    
    match err {
        SrganError::InvalidInput(msg) => assert_eq!(msg, "inner error"),
        _ => panic!("Expected InvalidInput error to propagate"),
    }
}

#[test]
fn test_custom_error_variants() {
    // Test all error variants for proper construction and display
    let errors = vec![
        SrganError::Io(io::Error::new(io::ErrorKind::Other, "io error")),
        SrganError::Image("image error".to_string()),
        SrganError::InvalidInput("invalid input".to_string()),
        SrganError::InvalidParameter("invalid param".to_string()),
        SrganError::FileNotFound(PathBuf::from("/missing")),
        SrganError::ShapeError("shape mismatch".to_string()),
        SrganError::GraphExecution("graph error".to_string()),
        SrganError::Training("training error".to_string()),
        SrganError::Parse("parse error".to_string()),
        SrganError::Conversion("conversion error".to_string()),
        SrganError::Network("network error".to_string()),
        SrganError::Other("other error".to_string()),
    ];
    
    for error in errors {
        // Verify each error can be displayed
        let display = format!("{}", error);
        assert!(!display.is_empty(), "Error display should not be empty");
        
        // Verify Debug formatting works
        let debug = format!("{:?}", error);
        assert!(!debug.is_empty(), "Error debug should not be empty");
    }
}

#[test]
fn test_error_conversion_from_string() {
    let string_err = String::from("string error");
    let srgan_err: SrganError = SrganError::Other(string_err);
    
    match srgan_err {
        SrganError::Other(msg) => assert_eq!(msg, "string error"),
        _ => panic!("Expected Other error variant"),
    }
}

#[test]
fn test_result_map_operations() {
    fn get_value() -> Result<i32> {
        Ok(42)
    }
    
    // Test map
    let result = get_value().map(|x| x * 2);
    let value = assert_ok(result, "mapped result");
    assert_eq!(value, 84);
    
    // Test map_err
    fn get_error() -> Result<i32> {
        Err(SrganError::Other("original".to_string()))
    }
    
    let result = get_error().map_err(|_| SrganError::Other("mapped".to_string()));
    let err = assert_err(result, "mapped error");
    
    match err {
        SrganError::Other(msg) => assert_eq!(msg, "mapped"),
        _ => panic!("Expected mapped error"),
    }
}

#[test]
fn test_result_and_or_operations() {
    fn success() -> Result<i32> {
        Ok(42)
    }
    
    fn failure() -> Result<i32> {
        Err(SrganError::Other("failed".to_string()))
    }
    
    // Test and_then with success
    let result = success().and_then(|x| Ok(x + 1));
    let value = assert_ok(result, "and_then with success");
    assert_eq!(value, 43);
    
    // Test and_then with failure
    let result = failure().and_then(|x| Ok(x + 1));
    assert_result_err(result, "and_then with failure");
    
    // Test or_else with success
    let result = success().or_else(|_| Ok(0));
    let value = assert_ok(result, "or_else with success");
    assert_eq!(value, 42);
    
    // Test or_else with failure
    let result = failure().or_else(|_| Ok(0));
    let value = assert_ok(result, "or_else with failure recovery");
    assert_eq!(value, 0);
}