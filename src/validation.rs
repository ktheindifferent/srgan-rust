use crate::error::{Result, SrganError};
use std::fs;
use std::path::{Path, PathBuf};

/// Validates that a file exists and is readable
pub fn validate_input_file(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    
    if !path.exists() {
        return Err(SrganError::FileNotFound(path.to_path_buf()));
    }
    
    if !path.is_file() {
        return Err(SrganError::InvalidInput(format!(
            "{} is not a file",
            path.display()
        )));
    }
    
    // Check if file is readable
    fs::metadata(path)
        .map_err(|e| SrganError::Io(e))?;
    
    Ok(path.to_path_buf())
}

/// Validates that the output path is writable
pub fn validate_output_path(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    
    // Check if parent directory exists
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            return Err(SrganError::InvalidInput(format!(
                "Parent directory {} does not exist",
                parent.display()
            )));
        }
        
        if !parent.is_dir() {
            return Err(SrganError::InvalidInput(format!(
                "{} is not a directory",
                parent.display()
            )));
        }
    }
    
    // If file exists, check if it's writable
    if path.exists() && !path.is_file() {
        return Err(SrganError::InvalidInput(format!(
            "{} exists but is not a file",
            path.display()
        )));
    }
    
    Ok(path.to_path_buf())
}

/// Validates that a directory exists and is readable
pub fn validate_directory(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    
    if !path.exists() {
        return Err(SrganError::FileNotFound(path.to_path_buf()));
    }
    
    if !path.is_dir() {
        return Err(SrganError::InvalidInput(format!(
            "{} is not a directory",
            path.display()
        )));
    }
    
    Ok(path.to_path_buf())
}

/// Validates that an image file has a supported extension
pub fn validate_image_extension(path: &Path) -> Result<()> {
    let valid_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"];
    
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    
    match extension {
        Some(ext) if valid_extensions.contains(&ext.as_str()) => Ok(()),
        Some(ext) => Err(SrganError::InvalidInput(format!(
            "Unsupported image format: .{}. Supported formats: {}",
            ext,
            valid_extensions.join(", ")
        ))),
        None => Err(SrganError::InvalidInput(
            "File has no extension. Please specify an image file with a valid extension".to_string()
        )),
    }
}

/// Validates a positive integer parameter
pub fn validate_positive_int(value: &str, param_name: &str) -> Result<usize> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| SrganError::Parse(format!("{} must be a positive integer", param_name)))?;
    
    if parsed == 0 {
        return Err(SrganError::InvalidParameter(format!(
            "{} must be greater than 0",
            param_name
        )));
    }
    
    Ok(parsed)
}

/// Validates a positive float parameter
pub fn validate_positive_float(value: &str, param_name: &str) -> Result<f32> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| SrganError::Parse(format!("{} must be a valid number", param_name)))?;
    
    if parsed <= 0.0 {
        return Err(SrganError::InvalidParameter(format!(
            "{} must be greater than 0",
            param_name
        )));
    }
    
    if !parsed.is_finite() {
        return Err(SrganError::InvalidParameter(format!(
            "{} must be a finite number",
            param_name
        )));
    }
    
    Ok(parsed)
}

/// Validates the upscaling factor
pub fn validate_factor(factor: usize) -> Result<()> {
    match factor {
        1..=8 => Ok(()),
        _ => Err(SrganError::InvalidParameter(format!(
            "Factor {} is out of range. Must be between 1 and 8",
            factor
        ))),
    }
}

/// Validates batch size
pub fn validate_batch_size(size: usize) -> Result<()> {
    match size {
        1..=256 => Ok(()),
        _ => Err(SrganError::InvalidParameter(format!(
            "Batch size {} is out of range. Must be between 1 and 256",
            size
        ))),
    }
}

/// Validates patch size
pub fn validate_patch_size(size: usize) -> Result<()> {
    match size {
        8..=512 => Ok(()),
        _ => Err(SrganError::InvalidParameter(format!(
            "Patch size {} is out of range. Must be between 8 and 512",
            size
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_validate_input_file_exists() {
        let dir = TempDir::new().expect("Failed to create temp dir for test");
        let file_path = dir.path().join("test.png");
        File::create(&file_path).expect("Failed to create test file");
        
        let result = validate_input_file(file_path.to_str().expect("Failed to convert path to str"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_input_file_not_exists() {
        let result = validate_input_file("/nonexistent/file.png");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_output_path_valid() {
        let dir = TempDir::new().expect("Failed to create temp dir for test");
        let file_path = dir.path().join("output.png");
        
        let result = validate_output_path(file_path.to_str().expect("Failed to convert path to str"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_directory_exists() {
        let dir = TempDir::new().expect("Failed to create temp dir for test");
        
        let result = validate_directory(dir.path().to_str().expect("Failed to convert path to str"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_image_extension_valid() {
        let path = Path::new("image.png");
        assert!(validate_image_extension(path).is_ok());
        
        let path = Path::new("image.jpg");
        assert!(validate_image_extension(path).is_ok());
    }

    #[test]
    fn test_validate_image_extension_invalid() {
        let path = Path::new("file.txt");
        assert!(validate_image_extension(path).is_err());
        
        let path = Path::new("file_without_extension");
        assert!(validate_image_extension(path).is_err());
    }

    #[test]
    fn test_validate_positive_int() {
        assert_eq!(validate_positive_int("5", "test").expect("Should parse valid positive int"), 5);
        assert!(validate_positive_int("0", "test").is_err());
        assert!(validate_positive_int("-1", "test").is_err());
        assert!(validate_positive_int("abc", "test").is_err());
    }

    #[test]
    fn test_validate_positive_float() {
        assert_eq!(validate_positive_float("3.5", "test").expect("Should parse valid positive float"), 3.5);
        assert!(validate_positive_float("0", "test").is_err());
        assert!(validate_positive_float("-1.5", "test").is_err());
        assert!(validate_positive_float("inf", "test").is_err());
    }

    #[test]
    fn test_validate_factor() {
        assert!(validate_factor(4).is_ok());
        assert!(validate_factor(0).is_err());
        assert!(validate_factor(10).is_err());
    }
}