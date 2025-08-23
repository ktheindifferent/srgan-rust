use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Debug)]
pub enum SrganError {
	Io(io::Error),
	Image(image::ImageError),
	Parse(String),
	Network(String),
	Training(String),
	Validation(String),
	Serialization(String),
	InvalidParameter(String),
	FileNotFound(PathBuf),
	GraphConstruction(String),
	GraphExecution(String),
	CheckpointSave(String),
	InvalidInput(String),
	ShapeError(String),
	MissingFolder(String),
}

impl fmt::Display for SrganError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			SrganError::Io(err) => write!(f, "IO error: {}", err),
			SrganError::Image(err) => write!(f, "Image processing error: {}", err),
			SrganError::Parse(msg) => write!(f, "Parse error: {}", msg),
			SrganError::Network(msg) => write!(f, "Network error: {}", msg),
			SrganError::Training(msg) => write!(f, "Training error: {}", msg),
			SrganError::Validation(msg) => write!(f, "Validation error: {}", msg),
			SrganError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
			SrganError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
			SrganError::FileNotFound(path) => write!(f, "File not found: {}", path.display()),
			SrganError::GraphConstruction(msg) => write!(f, "Graph construction error: {}", msg),
			SrganError::GraphExecution(msg) => write!(f, "Graph execution error: {}", msg),
			SrganError::CheckpointSave(msg) => write!(f, "Failed to save checkpoint: {}", msg),
			SrganError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
			SrganError::ShapeError(msg) => write!(f, "Shape error: {}", msg),
			SrganError::MissingFolder(msg) => write!(f, "Missing required folder: {}", msg),
		}
	}
}

impl StdError for SrganError {}

impl From<io::Error> for SrganError {
	fn from(err: io::Error) -> Self {
		SrganError::Io(err)
	}
}

impl From<image::ImageError> for SrganError {
	fn from(err: image::ImageError) -> Self {
		SrganError::Image(err)
	}
}

impl From<String> for SrganError {
	fn from(err: String) -> Self {
		SrganError::Network(err)
	}
}

impl From<alumina::graph::Error> for SrganError {
	fn from(err: alumina::graph::Error) -> Self {
		SrganError::GraphConstruction(format!("{}", err))
	}
}

impl From<bincode::Error> for SrganError {
	fn from(err: bincode::Error) -> Self {
		SrganError::Serialization(format!("Bincode error: {}", err))
	}
}

impl From<std::num::ParseIntError> for SrganError {
	fn from(err: std::num::ParseIntError) -> Self {
		SrganError::Parse(format!("Failed to parse integer: {}", err))
	}
}

impl From<std::num::ParseFloatError> for SrganError {
	fn from(err: std::num::ParseFloatError) -> Self {
		SrganError::Parse(format!("Failed to parse float: {}", err))
	}
}

pub type Result<T> = std::result::Result<T, SrganError>;
