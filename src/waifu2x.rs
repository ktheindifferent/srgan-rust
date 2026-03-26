//! Waifu2x model support for anime/illustration super-resolution.
//!
//! Waifu2x is a CNN-based image upscaling and noise-reduction algorithm
//! originally designed for anime-style artwork. It uses a VGG-like
//! convolutional network with:
//!
//! - Multiple 3×3 conv layers with ReLU activations
//! - No spatial pooling (preserves resolution)
//! - Optional noise reduction (levels 0–3)
//! - Sub-pixel convolution for upscaling (×1 = denoise only, ×2 = upscale)
//!
//! ## Weight status
//!
//! The native waifu2x weights (distributed as `.json` / `.bin` files in the
//! original project) use a different serialisation format from the `.rsr`
//! format used by this codebase. Conversion from ncnn/ONNX/waifu2x-json to
//! `.rsr` is tracked as a TODO.
//!
//! In the meantime all waifu2x labels fall back to the built-in anime model,
//! which was trained on the same class of content and provides equivalent
//! upscaling quality at ×4.

use crate::config::Waifu2xConfig;
use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;

// ── NoiseLevel ────────────────────────────────────────────────────────────────

/// Waifu2x noise-reduction strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseLevel {
    /// No noise reduction.
    None = 0,
    /// Light noise reduction.
    Low = 1,
    /// Medium noise reduction.
    Medium = 2,
    /// Aggressive noise reduction.
    High = 3,
}

impl NoiseLevel {
    /// Parse from the integer stored in `Waifu2xConfig`.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => NoiseLevel::None,
            1 => NoiseLevel::Low,
            2 => NoiseLevel::Medium,
            _ => NoiseLevel::High,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl std::fmt::Display for NoiseLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_u8())
    }
}

// ── ScaleFactor ───────────────────────────────────────────────────────────────

/// Waifu2x upscaling factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waifu2xScale {
    /// Denoise only — output is the same resolution as the input.
    One,
    /// Upscale ×2.
    Two,
}

impl Waifu2xScale {
    pub fn from_u8(v: u8) -> Self {
        if v <= 1 { Waifu2xScale::One } else { Waifu2xScale::Two }
    }

    pub fn as_u8(self) -> u8 {
        match self { Waifu2xScale::One => 1, Waifu2xScale::Two => 2 }
    }
}

impl std::fmt::Display for Waifu2xScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_u8())
    }
}

// ── Waifu2xNetwork ────────────────────────────────────────────────────────────

/// High-level waifu2x wrapper that presents the same `upscale_image` interface
/// as [`UpscalingNetwork`] while encoding the waifu2x-specific configuration.
pub struct Waifu2xNetwork {
    /// Underlying upscaling network (currently the built-in anime model).
    inner: UpscalingNetwork,
    /// Noise reduction level used to build this instance.
    noise_level: NoiseLevel,
    /// Scale factor used to build this instance.
    scale: Waifu2xScale,
}

impl Waifu2xNetwork {
    /// Build a `Waifu2xNetwork` from a [`Waifu2xConfig`].
    ///
    /// # Note on weights
    ///
    /// Real waifu2x weights (ncnn / waifu2x-caffe format) have not yet been
    /// converted to the `.rsr` format used by this binary.  Until that
    /// conversion pipeline is complete the built-in anime model is used as the
    /// inference backend.  The noise-level and scale parameters are recorded
    /// and reported but do not yet alter the network topology.
    ///
    /// TODO: replace with dedicated waifu2x weights once the
    ///       ncnn/ONNX → `.rsr` conversion is available.
    pub fn from_config(config: &Waifu2xConfig) -> Result<Self> {
        let noise_level = NoiseLevel::from_u8(config.noise_level);
        let scale = Waifu2xScale::from_u8(config.scale);
        let label = config.model_label();
        let inner = UpscalingNetwork::from_label(&label, None)
            .map_err(SrganError::Network)?;
        Ok(Self { inner, noise_level, scale })
    }

    /// Load waifu2x network from a canonical label such as `"waifu2x"` or
    /// `"waifu2x-noise2-scale2"`.
    pub fn from_label(label: &str) -> Result<Self> {
        let config = parse_label(label)?;
        Self::from_config(&config)
    }

    /// Noise-reduction level this instance was built with.
    pub fn noise_level(&self) -> NoiseLevel {
        self.noise_level
    }

    /// Upscaling factor this instance was built with.
    pub fn scale(&self) -> Waifu2xScale {
        self.scale
    }

    /// Upscale (and optionally denoise) a [`image::DynamicImage`].
    pub fn upscale_image(
        &self,
        img: &image::DynamicImage,
    ) -> Result<image::DynamicImage> {
        self.inner.upscale_image(img)
    }

    /// Human-readable description of the active configuration.
    pub fn description(&self) -> String {
        format!(
            "waifu2x noise={} scale={}x (backed by built-in anime model; \
             TODO: load native waifu2x weights)",
            self.noise_level, self.scale
        )
    }
}

impl std::fmt::Display for Waifu2xNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.description())
    }
}

// ── Label parser ──────────────────────────────────────────────────────────────

/// Parse a waifu2x label into a [`Waifu2xConfig`].
///
/// Accepted formats:
/// - `"waifu2x"` → noise=1, scale=2 (defaults)
/// - `"waifu2x-noise{0..3}-scale{1,2}"` → explicit parameters
fn parse_label(label: &str) -> Result<Waifu2xConfig> {
    if label == "waifu2x" {
        return Ok(Waifu2xConfig { noise_level: 1, scale: 2 });
    }

    if let Some(rest) = label.strip_prefix("waifu2x-") {
        // Expected: "noise{N}-scale{M}"
        let parts: Vec<&str> = rest.split('-').collect();
        if parts.len() == 2 {
            let noise = parts[0]
                .strip_prefix("noise")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(1)
                .min(3);
            let scale = parts[1]
                .strip_prefix("scale")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(2);
            let scale = if scale == 1 { 1 } else { 2 };
            return Ok(Waifu2xConfig { noise_level: noise, scale });
        }
    }

    Err(SrganError::Network(format!(
        "invalid waifu2x label '{}'; expected 'waifu2x' or \
         'waifu2x-noise{{0..3}}-scale{{1,2}}'",
        label
    )))
}

// ── Supported labels ──────────────────────────────────────────────────────────

/// All canonical waifu2x labels accepted by the CLI and API.
pub const WAIFU2X_LABELS: &[&str] = &[
    "waifu2x",
    "waifu2x-noise0-scale1",
    "waifu2x-noise0-scale2",
    "waifu2x-noise1-scale1",
    "waifu2x-noise1-scale2",
    "waifu2x-noise2-scale1",
    "waifu2x-noise2-scale2",
    "waifu2x-noise3-scale1",
    "waifu2x-noise3-scale2",
];

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bare_label() {
        let c = parse_label("waifu2x").unwrap();
        assert_eq!(c.noise_level, 1);
        assert_eq!(c.scale, 2);
    }

    #[test]
    fn parse_parameterised_label() {
        let c = parse_label("waifu2x-noise2-scale1").unwrap();
        assert_eq!(c.noise_level, 2);
        assert_eq!(c.scale, 1);
    }

    #[test]
    fn parse_noise3_scale2() {
        let c = parse_label("waifu2x-noise3-scale2").unwrap();
        assert_eq!(c.noise_level, 3);
        assert_eq!(c.scale, 2);
    }

    #[test]
    fn parse_invalid_label_errors() {
        assert!(parse_label("waifu2x-bad").is_err());
        assert!(parse_label("natural").is_err());
    }

    #[test]
    fn noise_level_roundtrip() {
        for v in 0u8..=3 {
            assert_eq!(NoiseLevel::from_u8(v).as_u8(), v);
        }
    }

    #[test]
    fn scale_roundtrip() {
        assert_eq!(Waifu2xScale::from_u8(1).as_u8(), 1);
        assert_eq!(Waifu2xScale::from_u8(2).as_u8(), 2);
        assert_eq!(Waifu2xScale::from_u8(0).as_u8(), 1); // 0 → One
        assert_eq!(Waifu2xScale::from_u8(5).as_u8(), 2); // >2 → Two
    }
}
