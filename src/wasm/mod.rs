//! WASM browser preview module.
//!
//! Provides a lightweight 2x bicubic/Lanczos upscaler compiled to WebAssembly
//! (wasm-bindgen compatible). The server serves the compiled `.wasm` and JS glue
//! at `GET /preview.js` and `GET /preview_bg.wasm`, plus a single-page demo at
//! `GET /demo`.
//!
//! ## Interpolation methods
//!
//! - **Lanczos3** (default): High-quality sinc-windowed interpolation
//! - **Bicubic**: Fast cubic spline interpolation
//! - **NearestNeighbor**: Fastest, pixelated output
//!
//! ## WASM entry point
//!
//! When compiled with `--target wasm32-unknown-unknown` under the `wasm` feature:
//! ```js
//! import init, { upscale_preview } from './preview.js';
//! await init();
//! const pngBytes = upscale_preview(inputImageBytes);
//! ```
//!
//! ## Server-side fallback
//!
//! When not targeting WASM, use `wasm_shim::WasmShim` for the same API:
//! ```rust,ignore
//! let shim = WasmShim::new();
//! let png = shim.upscale_preview_bytes(&input_png);
//! ```

pub mod preview;
pub mod wasm_shim;

/// Inline demo HTML served at `GET /wasm-demo`.
pub const PREVIEW_DEMO_HTML: &str = include_str!("../../static/preview-demo.html");
