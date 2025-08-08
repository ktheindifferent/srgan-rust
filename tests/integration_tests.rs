use std::process::Command;
use std::fs;
use std::path::Path;

#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .args(&["run", "--", "--help"])
        .output()
        .expect("Failed to execute command");
    
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Rusty SR"));
    assert!(stdout.contains("upscale images"));
}

#[test]
fn test_train_help() {
    let output = Command::new("cargo")
        .args(&["run", "--", "train", "--help"])
        .output()
        .expect("Failed to execute command");
    
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Train"));
    assert!(stdout.contains("neural parameters"));
}

#[test]
fn test_downscale_help() {
    let output = Command::new("cargo")
        .args(&["run", "--", "downscale", "--help"])
        .output()
        .expect("Failed to execute command");
    
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Downscale"));
}

#[test]
fn test_psnr_help() {
    let output = Command::new("cargo")
        .args(&["run", "--", "psnr", "--help"])
        .output()
        .expect("Failed to execute command");
    
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PSNR"));
}

#[test]
fn test_invalid_command() {
    let output = Command::new("cargo")
        .args(&["run", "--", "invalid_command"])
        .output()
        .expect("Failed to execute command");
    
    // Should fail or show usage
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error") || stderr.contains("INPUT_FILE"));
}

#[test]
fn test_missing_arguments() {
    let output = Command::new("cargo")
        .args(&["run", "--"])
        .output()
        .expect("Failed to execute command");
    
    // Should show error about missing input file
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("required") || stderr.contains("INPUT_FILE"));
}

#[test]
fn test_parameter_validation() {
    let output = Command::new("cargo")
        .args(&["run", "--", "-p", "invalid_param", "input.png", "output.png"])
        .output()
        .expect("Failed to execute command");
    
    // Should show error about invalid parameter
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("possible values") || stderr.contains("invalid"));
}

#[test]
fn test_downscale_missing_factor() {
    let output = Command::new("cargo")
        .args(&["run", "--", "downscale"])
        .output()
        .expect("Failed to execute command");
    
    // Should show error about missing factor
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("required") || stderr.contains("FACTOR"));
}

#[test]
fn test_psnr_missing_images() {
    let output = Command::new("cargo")
        .args(&["run", "--", "psnr", "image1.png"])
        .output()
        .expect("Failed to execute command");
    
    // Should show error about missing second image
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("required") || stderr.contains("IMAGE2"));
}