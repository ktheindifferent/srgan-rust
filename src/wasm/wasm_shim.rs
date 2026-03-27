//! Server-side fallback implementation for the WASM preview module.
//!
//! When not compiling to WASM (i.e. running on the server), this module
//! provides the same `upscale_preview` API using the native Rust implementation
//! from `crate::wasm::preview`. This allows server-side rendering of previews
//! for clients that don't support WebAssembly.

use super::preview::{self, InterpolationMethod, PreviewConfig, PreviewResult};

/// Server-side preview upscaler — uses the same algorithms as the WASM module
/// but runs natively on the server.
pub struct WasmShim {
    config: PreviewConfig,
}

impl WasmShim {
    /// Create a new shim with default configuration (Lanczos3 2x).
    pub fn new() -> Self {
        Self {
            config: PreviewConfig::default(),
        }
    }

    /// Create a shim with custom configuration.
    pub fn with_config(config: PreviewConfig) -> Self {
        Self { config }
    }

    /// Upscale raw RGBA pixel data, returning the result with timing info.
    pub fn upscale_preview(&self, rgba: &[u8], width: u32, height: u32) -> PreviewResult {
        preview::upscale_preview(rgba, width, height, &self.config)
    }

    /// Upscale from encoded image bytes (PNG with raw RGBA fallback), returning PNG bytes.
    /// This matches the WASM-bindgen `upscale_preview(image_data: &[u8]) -> Vec<u8>` signature.
    pub fn upscale_preview_bytes(&self, image_data: &[u8]) -> Vec<u8> {
        preview::upscale_preview_bytes(image_data)
    }

    /// Get the current interpolation method.
    pub fn method(&self) -> InterpolationMethod {
        self.config.method
    }

    /// Get the current scale factor.
    pub fn scale(&self) -> u32 {
        self.config.scale
    }

    /// Return a version string matching the WASM module.
    pub fn version(&self) -> &'static str {
        "0.2.0-native-shim"
    }
}

impl Default for WasmShim {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shim_default() {
        let shim = WasmShim::new();
        assert_eq!(shim.scale(), 2);
        assert_eq!(shim.method(), InterpolationMethod::Lanczos3);
    }

    #[test]
    fn test_shim_upscale() {
        let shim = WasmShim::new();
        let rgba = vec![128u8; 4 * 4 * 4]; // 4x4 image
        let result = shim.upscale_preview(&rgba, 4, 4);
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_shim_upscale_bytes_empty() {
        let shim = WasmShim::new();
        let result = shim.upscale_preview_bytes(&[]);
        assert!(result.is_empty());
    }
}
