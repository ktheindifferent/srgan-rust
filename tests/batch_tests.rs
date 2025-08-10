use srgan_rust::commands::batch::{BatchConfig, find_images};
use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;

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
    let temp_dir = TempDir::new().unwrap();
    let images = find_images(temp_dir.path(), false, &["jpg", "png"]).unwrap();
    assert_eq!(images.len(), 0);
}

#[test]
fn test_find_images_with_files() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create test files
    fs::write(temp_dir.path().join("test1.jpg"), b"").unwrap();
    fs::write(temp_dir.path().join("test2.png"), b"").unwrap();
    fs::write(temp_dir.path().join("test3.txt"), b"").unwrap();
    
    let images = find_images(temp_dir.path(), false, &["jpg", "png"]).unwrap();
    assert_eq!(images.len(), 2);
}

#[test]
fn test_find_images_recursive() {
    let temp_dir = TempDir::new().unwrap();
    let sub_dir = temp_dir.path().join("subdir");
    fs::create_dir(&sub_dir).unwrap();
    
    // Create files in root and subdirectory
    fs::write(temp_dir.path().join("root.jpg"), b"").unwrap();
    fs::write(sub_dir.join("sub.png"), b"").unwrap();
    
    // Non-recursive should find only root file
    let images = find_images(temp_dir.path(), false, &["jpg", "png"]).unwrap();
    assert_eq!(images.len(), 1);
    
    // Recursive should find both files
    let images = find_images(temp_dir.path(), true, &["jpg", "png"]).unwrap();
    assert_eq!(images.len(), 2);
}