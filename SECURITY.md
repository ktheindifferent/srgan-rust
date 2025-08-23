# Security Policy

## Overview

This document outlines the security measures implemented in SRGAN-Rust to protect against various attack vectors, particularly command injection vulnerabilities in video processing operations.

## Security Features

### 1. Command Injection Prevention

The video processing module (`src/video.rs`) has been hardened against command injection attacks through multiple layers of defense:

#### Input Validation
- **Path Validation**: All file paths are validated before use
- **Forbidden Characters**: Paths containing shell metacharacters are rejected
  - Semicolons (`;`), pipes (`|`), ampersands (`&`)
  - Command substitution characters (`` ` ``, `$()`, `${}`)
  - Redirection operators (`<`, `>`)
  - Quotes (`'`, `"`) and escape characters (`\`)
  - Wildcards (`*`, `?`, `[`, `]`)
  - Other special characters that could be interpreted by shells

#### Directory Traversal Prevention
- Paths containing `..` sequences are rejected
- Both Unix (`../`) and Windows (`..\\`) style traversal attempts are blocked
- Paths starting with dashes (`-`) are rejected to prevent flag injection

#### Safe Command Execution
- **Argument Separation**: All arguments are passed as separate parameters to `Command::new()`
- **No Shell Interpretation**: Commands are executed directly without shell interpretation
- **Explicit Argument Passing**: Each argument is passed via `.arg()` method, not concatenated
- **Stdio Control**: All commands use `Stdio::null()` to prevent stdin injection

Example of safe command execution:
```rust
// SAFE - Arguments passed separately
Command::new("ffmpeg")
    .arg("-i")
    .arg(&validated_path)  // Each argument is separate
    .stdin(Stdio::null())   // Prevent stdin injection
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .status()

// UNSAFE - Never do this!
Command::new("ffmpeg")
    .arg(format!("-i {}", user_input))  // Vulnerable to injection!
    .status()
```

### 2. Path Allowlisting

- **Canonicalization**: All paths are canonicalized to resolve symlinks and get absolute paths
- **Directory Restrictions**: Files are only processed within allowed directories:
  - Current working directory
  - System temporary directory
  - User home directory (if available)
- **Validation Before Use**: Paths are validated both syntactically and against the allowlist

### 3. Input Sanitization

#### Time String Validation
- Time parameters for video processing are validated to contain only:
  - Digits (0-9)
  - Colons (`:`) for time format
  - Periods (`.`) for decimal seconds
- Maximum length enforced to prevent buffer overflows

#### FPS Validation
- Frame rate values are validated to be:
  - Positive numbers
  - Within reasonable bounds (0 < fps ≤ 240)
  - Finite values (no NaN or Infinity)

### 4. Audit Logging

All command executions are logged for security auditing:

```rust
[AUDIT] Command execution at 1234567890: ffmpeg -version
[AUDIT] Command execution at 1234567891: ffprobe -v error | Input: /safe/path/video.mp4
```

Features:
- **Timestamp**: Unix timestamp for each command
- **Command Details**: Full command and arguments logged
- **Input Tracking**: Input files are logged for traceability
- **Optional File Logging**: Can be configured to write to `/var/log/srgan_audit.log`

Enable file-based audit logging with the `audit-log` feature:
```toml
[features]
audit-log = []
```

### 5. Resource Limits

- **Command Timeout**: 5-minute default timeout for ffmpeg operations
- **Memory Limits**: Configurable through system ulimits
- **File Size Validation**: Can be configured to reject files above certain sizes

## Security Testing

Comprehensive security tests are implemented in `tests/security_tests.rs`:

### Test Coverage
1. **Command Injection**: Tests for semicolons, pipes, ampersands, command substitution
2. **Directory Traversal**: Tests for `../` and `..\\` patterns
3. **Special Characters**: Tests for newlines, null bytes, quotes, brackets
4. **Path Prefix Attacks**: Tests for dash-prefixed paths that could be interpreted as flags
5. **Wildcard Patterns**: Tests for glob patterns and tilde expansion
6. **Unicode Attacks**: Tests for special Unicode characters and encoding tricks
7. **Combined Attacks**: Tests for multiple attack vectors combined

### Running Security Tests
```bash
cargo test --test security_tests
```

## Best Practices for Contributors

When adding new features that involve:

1. **File Operations**:
   - Always use the `validate_path()` function
   - Use `validate_and_canonicalize_path()` for paths that will be accessed
   - Never concatenate user input into commands

2. **Command Execution**:
   - Always pass arguments using `.arg()` method
   - Never use shell expansion or string formatting with user input
   - Always set `stdin(Stdio::null())` unless input is required
   - Log all command executions using `log_command_execution()`

3. **User Input**:
   - Validate all user-provided strings before use
   - Use enums for predefined options instead of strings
   - Implement proper bounds checking for numeric inputs

## Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** open a public issue
2. Email security concerns to: [security@example.com]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## Security Checklist for Code Review

- [ ] All file paths are validated using `validate_path()`
- [ ] Commands use separate `.arg()` calls, not string concatenation
- [ ] User input is never directly interpolated into commands
- [ ] All commands have `stdin(Stdio::null())`
- [ ] Time strings and numeric inputs are validated
- [ ] Path canonicalization is performed before file access
- [ ] Command executions are logged for auditing
- [ ] Error messages don't leak sensitive information
- [ ] Resource limits are enforced where appropriate

## Updates and Maintenance

This security policy is maintained alongside the codebase. Security measures are continuously improved based on:
- New threat discoveries
- Security audit results
- Community feedback
- Industry best practices

Last Updated: 2024
Version: 1.0.0