//! WASM browser preview module.
//!
//! Provides a lightweight 2x nearest-neighbor + sharpening upscaler compiled
//! to WebAssembly (wasm-bindgen compatible). The server serves the compiled
//! `.wasm` and JS glue at `GET /preview.js` and `GET /preview_bg.wasm`, plus
//! a single-page demo at `GET /demo`.

pub mod preview;

/// Inline demo HTML served at `GET /wasm-demo`.
pub const PREVIEW_DEMO_HTML: &str = include_str!("../../static/preview-demo.html");
