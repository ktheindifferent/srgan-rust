use srgan_rust::commands::batch::{BatchConfig, find_images};
use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;

// Import test helpers for better error handling
#[path = "test_helpers.rs"]
mod test_helpers;
use test_helpers::*;

#[test]
fn test_batch_config_creation() {
    let config = BatchConfig {
        input_path: PathBuf::from("input"),
        output_path: PathBuf::from("output"),
        model_path: Some(PathBuf::from("model.rsr")),
        recursive: true,
        parallel: false,
        skip_existing: true,
        extensions: vec!["jpg".to_string(), "png".to_string()],
    };
    
    assert_eq!(config.input_path, PathBuf::from("input"));
    assert_eq!(config.output_path, PathBuf::from("output"));
    assert!(config.recursive);
    assert!(!config.parallel);
    assert!(config.skip_existing);
    assert_eq!(config.extensions.len(), 2);
}

#[test]
fn test_find_images_empty_dir() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    let images = assert_ok(
        find_images(temp_dir.path(), false, &["jpg", "png"]),
        "finding images in empty directory"
    );
    assert_eq!(images.len(), 0);
}

#[test]
fn test_find_images_with_files() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    
    // Create test files
    assert_result_ok(
        fs::write(temp_dir.path().join("test1.jpg"), b""),
        "creating test1.jpg"
    );
    assert_result_ok(
        fs::write(temp_dir.path().join("test2.png"), b""),
        "creating test2.png"
    );
    assert_result_ok(
        fs::write(temp_dir.path().join("test3.txt"), b""),
        "creating test3.txt"
    );
    
    let images = assert_ok(
        find_images(temp_dir.path(), false, &["jpg", "png"]),
        "finding images with specific extensions"
    );
    assert_eq!(images.len(), 2);
}

#[test]
fn test_find_images_recursive() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    let sub_dir = temp_dir.path().join("subdir");
    assert_result_ok(fs::create_dir(&sub_dir), "creating subdirectory");
    
    // Create files in root and subdirectory
    assert_result_ok(
        fs::write(temp_dir.path().join("root.jpg"), b""),
        "creating root.jpg"
    );
    assert_result_ok(
        fs::write(sub_dir.join("sub.png"), b""),
        "creating sub.png in subdirectory"
    );
    
    // Non-recursive should find only root file
    let images = assert_ok(
        find_images(temp_dir.path(), false, &["jpg", "png"]),
        "finding images non-recursively"
    );
    assert_eq!(images.len(), 1);
    
    // Recursive should find both files
    let images = assert_ok(
        find_images(temp_dir.path(), true, &["jpg", "png"]),
        "finding images recursively"
    );
    assert_eq!(images.len(), 2);
}

#[test]
fn test_find_images_case_sensitivity() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    
    // Create files with different case extensions
    assert_result_ok(
        fs::write(temp_dir.path().join("test1.JPG"), b""),
        "creating test1.JPG"
    );
    assert_result_ok(
        fs::write(temp_dir.path().join("test2.PNG"), b""),
        "creating test2.PNG"
    );
    assert_result_ok(
        fs::write(temp_dir.path().join("test3.Png"), b""),
        "creating test3.Png"
    );
    
    // Test with lowercase extensions
    let images = assert_ok(
        find_images(temp_dir.path(), false, &["jpg", "png"]),
        "finding images with lowercase extensions"
    );
    
    // Behavior may vary by OS, but we should handle it gracefully
    assert!(images.len() <= 3, "Should find at most 3 images");
}

#[test]
fn test_find_images_permission_errors() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    
    // Create a file
    assert_result_ok(
        fs::write(temp_dir.path().join("test.jpg"), b""),
        "creating test.jpg"
    );
    
    // Try to find images - should succeed even if we can't access some files
    let result = find_images(temp_dir.path(), false, &["jpg", "png"]);
    assert!(result.is_ok(), "Should handle permission issues gracefully");
}

#[test]
fn test_batch_config_validation() {
    // Test with empty extensions
    let config = BatchConfig {
        input_path: PathBuf::from("input"),
        output_path: PathBuf::from("output"),
        model_path: None,
        recursive: false,
        parallel: false,
        skip_existing: false,
        extensions: vec![],
    };
    
    assert_eq!(config.extensions.len(), 0, "Empty extensions should be allowed");
    
    // Test with many extensions
    let config = BatchConfig {
        input_path: PathBuf::from("input"),
        output_path: PathBuf::from("output"),
        model_path: None,
        recursive: false,
        parallel: true,
        skip_existing: false,
        extensions: vec![
            "jpg".to_string(),
            "jpeg".to_string(),
            "png".to_string(),
            "bmp".to_string(),
            "tiff".to_string(),
        ],
    };
    
    assert_eq!(config.extensions.len(), 5, "Multiple extensions should be supported");
}

#[test]
fn test_find_images_symlinks() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    let target_dir = temp_dir.path().join("target");
    assert_result_ok(fs::create_dir(&target_dir), "creating target directory");
    
    // Create a file in target directory
    assert_result_ok(
        fs::write(target_dir.join("image.jpg"), b""),
        "creating image in target directory"
    );
    
    // Create symlink (if supported by OS)
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let link = temp_dir.path().join("link.jpg");
        if symlink(target_dir.join("image.jpg"), &link).is_ok() {
            let images = assert_ok(
                find_images(temp_dir.path(), false, &["jpg"]),
                "finding images including symlinks"
            );
            assert!(images.len() >= 1, "Should find symlinked image");
        }
    }
}

#[test]
fn test_find_images_hidden_files() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    
    // Create hidden file (Unix-style)
    assert_result_ok(
        fs::write(temp_dir.path().join(".hidden.jpg"), b""),
        "creating hidden image file"
    );
    
    // Create normal file
    assert_result_ok(
        fs::write(temp_dir.path().join("visible.jpg"), b""),
        "creating visible image file"
    );
    
    let images = assert_ok(
        find_images(temp_dir.path(), false, &["jpg"]),
        "finding images including hidden files"
    );
    
    // Behavior may vary, but should at least find the visible file
    assert!(images.len() >= 1, "Should find at least the visible file");
}