use std::process::Command;
use std::fs;
use std::path::Path;

// Import test helpers for better error handling
#[path = "test_helpers.rs"]
mod test_helpers;
use test_helpers::*;

#[test]
fn test_cli_help() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "--help"])
            .output(),
        "executing cargo run help command"
    );
    
    assert_command_success(&output, "cargo run --help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_contains(&stdout, "Rusty SR", "help output");
    assert_contains(&stdout, "upscale images", "help output");
}

#[test]
fn test_train_help() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "train", "--help"])
            .output(),
        "executing cargo run train help command"
    );
    
    assert_command_success(&output, "cargo run train --help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_contains(&stdout, "Train", "train help output");
    assert_contains(&stdout, "neural parameters", "train help output");
}

#[test]
fn test_downscale_help() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "downscale", "--help"])
            .output(),
        "executing cargo run downscale help command"
    );
    
    assert_command_success(&output, "cargo run downscale --help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_contains(&stdout, "Downscale", "downscale help output");
}

#[test]
fn test_psnr_help() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "psnr", "--help"])
            .output(),
        "executing cargo run psnr help command"
    );
    
    assert_command_success(&output, "cargo run psnr --help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_contains(&stdout, "PSNR", "psnr help output");
}

#[test]
fn test_invalid_command() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "invalid_command"])
            .output(),
        "executing cargo run with invalid command"
    );
    
    // Should fail or show usage
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_contains_any(
        &stderr,
        &["Error", "INPUT_FILE"],
        "invalid command error message"
    );
}

#[test]
fn test_missing_arguments() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--"])
            .output(),
        "executing cargo run without arguments"
    );
    
    // Should show error about missing input file
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_contains_any(
        &stderr,
        &["required", "INPUT_FILE"],
        "missing arguments error message"
    );
}

#[test]
fn test_parameter_validation() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "-p", "invalid_param", "input.png", "output.png"])
            .output(),
        "executing cargo run with invalid parameter"
    );
    
    // Should show error about invalid parameter
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_contains_any(
        &stderr,
        &["possible values", "invalid"],
        "invalid parameter error message"
    );
}

#[test]
fn test_downscale_missing_factor() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "downscale"])
            .output(),
        "executing downscale without factor"
    );
    
    // Should show error about missing factor
    assert_command_failure(&output, "downscale without factor");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_contains_any(
        &stderr,
        &["required", "FACTOR"],
        "missing downscale factor error message"
    );
}

#[test]
fn test_psnr_missing_images() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "psnr", "image1.png"])
            .output(),
        "executing psnr with missing second image"
    );
    
    // Should show error about missing second image
    assert_command_failure(&output, "psnr with missing second image");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_contains_any(
        &stderr,
        &["required", "IMAGE2"],
        "missing second image error message"
    );
}

// Additional error condition tests

#[test]
fn test_invalid_input_file() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "nonexistent_file.png", "output.png"])
            .output(),
        "executing with non-existent input file"
    );
    
    // Should fail with file not found error
    assert_command_failure(&output, "processing non-existent file");
}

#[test]
fn test_invalid_output_directory() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "tests/test_helpers.rs", "/invalid/path/output.png"])
            .output(),
        "executing with invalid output directory"
    );
    
    // Should fail with directory error
    assert_command_failure(&output, "invalid output directory");
}

#[test]
fn test_downscale_invalid_factor() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "downscale", "not_a_number", "input.png", "output.png"])
            .output(),
        "executing downscale with invalid factor"
    );
    
    // Should fail with parse error
    assert_command_failure(&output, "downscale with invalid factor");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid") || stderr.contains("parse") || stderr.contains("number"),
        "Expected error about invalid number in stderr: {}",
        stderr
    );
}

#[test]
fn test_train_without_dataset() {
    let output = assert_ok(
        Command::new("cargo")
            .args(&["run", "--", "train"])
            .output(),
        "executing train without dataset"
    );
    
    // Should fail with missing dataset error
    assert_command_failure(&output, "train without dataset");
}

#[test]
fn test_concurrent_cli_execution() {
    use std::thread;
    
    let handles: Vec<_> = (0..4).map(|i| {
        thread::spawn(move || {
            let output = assert_ok(
                Command::new("cargo")
                    .args(&["run", "--", "--help"])
                    .output(),
                &format!("concurrent help execution {}", i)
            );
            assert_command_success(&output, &format!("concurrent help {}", i));
        })
    }).collect();
    
    for (idx, handle) in handles.into_iter().enumerate() {
        assert_thread_success(handle.join(), &format!("concurrent CLI thread {}", idx));
    }
}

#[test]
fn test_help_subcommands_all() {
    let subcommands = ["train", "downscale", "psnr"];
    
    for subcommand in &subcommands {
        let output = assert_ok(
            Command::new("cargo")
                .args(&["run", "--", subcommand, "--help"])
                .output(),
            &format!("executing {} help", subcommand)
        );
        
        assert_command_success(&output, &format!("{} --help", subcommand));
        
        // Each subcommand should have its name in the help text
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}{}", stdout, stderr);
        
        assert!(
            combined.to_lowercase().contains(subcommand) || 
            combined.contains("USAGE") ||
            combined.contains("Usage"),
            "Help for {} should contain the command name or usage info. Got: {}",
            subcommand,
            combined
        );
    }
}