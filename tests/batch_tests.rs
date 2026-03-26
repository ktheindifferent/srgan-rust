use srgan_rust::commands::batch::collect_image_files;
use std::fs;
use tempfile::TempDir;

// Import test helpers for better error handling
#[path = "test_helpers.rs"]
mod test_helpers;
use test_helpers::*;

#[test]
fn test_collect_image_files_empty_dir() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    let images = assert_ok(
        collect_image_files(temp_dir.path(), "*.jpg", false),
        "finding images in empty directory"
    );
    assert_eq!(images.len(), 0);
}

#[test]
fn test_collect_image_files_with_files() {
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
        collect_image_files(temp_dir.path(), "*.jpg", false),
        "finding jpg images"
    );
    assert_eq!(images.len(), 1);
}

#[test]
fn test_collect_image_files_recursive() {
    let temp_dir = assert_ok(TempDir::new(), "creating temporary directory");
    let sub_dir = temp_dir.path().join("subdir");
    assert_result_ok(fs::create_dir(&sub_dir), "creating subdirectory");

    // Create files in root and subdirectory
    assert_result_ok(
        fs::write(temp_dir.path().join("root.jpg"), b""),
        "creating root.jpg"
    );
    assert_result_ok(
        fs::write(sub_dir.join("sub.jpg"), b""),
        "creating sub.jpg in subdirectory"
    );

    // Non-recursive should find only root file
    let images = assert_ok(
        collect_image_files(temp_dir.path(), "*.jpg", false),
        "finding images non-recursively"
    );
    assert_eq!(images.len(), 1);

    // Recursive should find both files
    let images = assert_ok(
        collect_image_files(temp_dir.path(), "*.jpg", true),
        "finding images recursively"
    );
    assert_eq!(images.len(), 2);
}

#[test]
fn test_collect_image_files_hidden_files() {
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
        collect_image_files(temp_dir.path(), "*.jpg", false),
        "finding images including hidden files"
    );

    // Behavior may vary, but should at least find the visible file
    assert!(images.len() >= 1, "Should find at least the visible file");
}
