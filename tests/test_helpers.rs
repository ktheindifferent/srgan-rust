/// Test helper functions for result-based assertions
/// 
/// These helpers provide better error messages and more explicit error handling
/// compared to using .unwrap() directly in tests.

use std::fmt::Debug;

/// Assert that a Result is Ok and return the unwrapped value.
/// Provides a descriptive error message on failure.
pub fn assert_ok<T, E: Debug>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(err) => panic!("Expected Ok result for {}, but got error: {:?}", context, err),
    }
}

/// Assert that a Result is Ok without returning the value.
/// Useful when you only need to verify success.
pub fn assert_result_ok<T, E: Debug>(result: Result<T, E>, context: &str) {
    if let Err(err) = result {
        panic!("Expected Ok result for {}, but got error: {:?}", context, err);
    }
}

/// Assert that a Result is Err and return the error.
/// Provides a descriptive error message on unexpected success.
pub fn assert_err<T: Debug, E>(result: Result<T, E>, context: &str) -> E {
    match result {
        Err(err) => err,
        Ok(val) => panic!("Expected Err result for {}, but got Ok: {:?}", context, val),
    }
}

/// Assert that a Result is Err without returning the error.
/// Useful when you only need to verify failure.
pub fn assert_result_err<T: Debug, E>(result: Result<T, E>, context: &str) {
    if let Ok(val) = result {
        panic!("Expected Err result for {}, but got Ok: {:?}", context, val);
    }
}

/// Assert that an Option is Some and return the unwrapped value.
/// Provides a descriptive error message on None.
pub fn assert_some<T>(option: Option<T>, context: &str) -> T {
    match option {
        Some(value) => value,
        None => panic!("Expected Some value for {}, but got None", context),
    }
}

/// Assert that an Option is None.
/// Provides a descriptive error message on unexpected Some.
pub fn assert_none<T: Debug>(option: Option<T>, context: &str) {
    if let Some(val) = option {
        panic!("Expected None for {}, but got Some: {:?}", context, val);
    }
}

/// Helper for thread join operations in tests.
/// Provides better error messages than raw unwrap.
pub fn assert_thread_success<T>(join_result: std::thread::Result<T>, thread_name: &str) -> T {
    match join_result {
        Ok(value) => value,
        Err(_) => panic!("Thread '{}' panicked during execution", thread_name),
    }
}

/// Helper for command execution assertions.
/// Checks both execution success and status code.
pub fn assert_command_success(output: &std::process::Output, command_desc: &str) {
    assert!(
        output.status.success(),
        "Command '{}' failed with exit code: {:?}\nStderr: {}",
        command_desc,
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Helper for command execution that's expected to fail.
pub fn assert_command_failure(output: &std::process::Output, command_desc: &str) {
    assert!(
        !output.status.success(),
        "Command '{}' unexpectedly succeeded when it should have failed\nStdout: {}",
        command_desc,
        String::from_utf8_lossy(&output.stdout)
    );
}

/// Assert that a string contains a substring, with helpful error message.
pub fn assert_contains(haystack: &str, needle: &str, context: &str) {
    assert!(
        haystack.contains(needle),
        "{}: Expected to find '{}' in output, but got: '{}'",
        context,
        needle,
        haystack
    );
}

/// Assert that a string contains any of the given substrings.
pub fn assert_contains_any(haystack: &str, needles: &[&str], context: &str) {
    let found = needles.iter().any(|needle| haystack.contains(needle));
    assert!(
        found,
        "{}: Expected to find one of {:?} in output, but got: '{}'",
        context,
        needles,
        haystack
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_ok_success() {
        let result: Result<i32, String> = Ok(42);
        let value = assert_ok(result, "test operation");
        assert_eq!(value, 42);
    }

    #[test]
    #[should_panic(expected = "Expected Ok result for test operation")]
    fn test_assert_ok_failure() {
        let result: Result<i32, String> = Err("error".to_string());
        assert_ok(result, "test operation");
    }

    #[test]
    fn test_assert_result_ok_success() {
        let result: Result<i32, String> = Ok(42);
        assert_result_ok(result, "test operation");
    }

    #[test]
    fn test_assert_err_success() {
        let result: Result<i32, String> = Err("error".to_string());
        let err = assert_err(result, "test operation");
        assert_eq!(err, "error");
    }

    #[test]
    #[should_panic(expected = "Expected Err result for test operation")]
    fn test_assert_err_failure() {
        let result: Result<i32, String> = Ok(42);
        assert_err(result, "test operation");
    }

    #[test]
    fn test_assert_some_success() {
        let option = Some(42);
        let value = assert_some(option, "test value");
        assert_eq!(value, 42);
    }

    #[test]
    #[should_panic(expected = "Expected Some value for test value")]
    fn test_assert_some_failure() {
        let option: Option<i32> = None;
        assert_some(option, "test value");
    }

    #[test]
    fn test_assert_none_success() {
        let option: Option<i32> = None;
        assert_none(option, "test value");
    }

    #[test]
    fn test_assert_contains_success() {
        assert_contains("hello world", "world", "greeting check");
    }

    #[test]
    #[should_panic(expected = "Expected to find 'foo' in output")]
    fn test_assert_contains_failure() {
        assert_contains("hello world", "foo", "greeting check");
    }

    #[test]
    fn test_assert_contains_any_success() {
        assert_contains_any("hello world", &["foo", "world", "bar"], "greeting check");
    }
}