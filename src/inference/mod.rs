//! Inference engine — batch-optimised model serving.
//!
//! Re-exports the [`BatchInferenceEngine`] which accumulates incoming requests
//! into batches (up to 32 images, max 5 s wait) before submitting them to a
//! worker thread-pool for parallel upscaling.

pub mod batch;

pub use batch::{
    BatchInferenceEngine, BatchRequest, BatchResponse, BatchResult,
};
