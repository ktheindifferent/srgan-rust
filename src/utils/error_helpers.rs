use crate::error::{SrganError, Result};
use std::io;

pub trait IoErrorMapper<T> {
    fn map_io_err(self) -> Result<T>;
}

impl<T> IoErrorMapper<T> for std::result::Result<T, io::Error> {
    fn map_io_err(self) -> Result<T> {
        self.map_err(|e| SrganError::Io(e))
    }
}

pub trait ParseErrorMapper<T> {
    fn map_parse_err(self, context: &str) -> Result<T>;
}

impl<T, E: std::fmt::Display> ParseErrorMapper<T> for std::result::Result<T, E> {
    fn map_parse_err(self, context: &str) -> Result<T> {
        self.map_err(|e| SrganError::Parse(format!("{}: {}", context, e)))
    }
}

pub trait InvalidParameterMapper<T> {
    fn map_invalid_param(self, param_name: &str, expected: &str) -> Result<T>;
}

impl<T, E> InvalidParameterMapper<T> for std::result::Result<T, E> {
    fn map_invalid_param(self, param_name: &str, expected: &str) -> Result<T> {
        self.map_err(|_| SrganError::InvalidParameter(
            format!("{} must be {}", param_name, expected)
        ))
    }
}

#[macro_export]
macro_rules! parse_param {
    ($app_m:expr, $param:expr, $type:ty, $desc:expr) => {
        if let Some(value) = $app_m.value_of($param) {
            Some(value.parse::<$type>()
                .map_err(|_| $crate::error::SrganError::InvalidParameter(
                    format!("{} must be {}", $param, $desc)
                ))?)
        } else {
            None
        }
    };
}