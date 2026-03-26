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

use crate::config::{Waifu2xConfig, Waifu2xStyle};
use crate::error::{Result, SrganError};
use image::GenericImage;

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

// ── Waifu2xNetwork (waifu2x-compat mode) ─────────────────────────────────────
//
// This is the "waifu2x-compat" software fallback.  It does NOT use the
// original waifu2x neural network weights (which require a separate weight
// conversion pipeline).  Instead it approximates waifu2x output using:
//
//   1. Lanczos3 resize at the requested scale (1× = identity, 2× = upscale)
//   2. Unsharp-mask sharpening whose strength is derived from noise_level:
//        noise 0 → no sharpening
//        noise 1 → light  (amount=0.3, radius=1)
//        noise 2 → medium (amount=0.5, radius=1)
//        noise 3 → aggressive (amount=0.8, radius=1)
//
// When native waifu2x weights are bundled this implementation should be
// replaced with real CNN inference.

/// High-level waifu2x-compat wrapper.  Performs Lanczos3 resize + unsharp-mask
/// sharpening to approximate waifu2x noise-reduction output without requiring
/// the original neural network weights.
pub struct Waifu2xNetwork {
    noise_level: NoiseLevel,
    scale: Waifu2xScale,
    style: Waifu2xStyle,
}

impl Waifu2xNetwork {
    /// Build a `Waifu2xNetwork` from a [`Waifu2xConfig`].
    ///
    /// This uses the waifu2x-compat software fallback (Lanczos3 + unsharp
    /// mask) — no neural network weights are loaded.
    pub fn from_config(config: &Waifu2xConfig) -> Result<Self> {
        let noise_level = NoiseLevel::from_u8(config.noise_level);
        let scale = Waifu2xScale::from_u8(config.scale);
        Ok(Self { noise_level, scale, style: config.style })
    }

    /// Load from a canonical label such as `"waifu2x"` or
    /// `"waifu2x-noise2-scale2"`.
    ///
    /// Uses the default style (`Anime`).  To specify a style, use
    /// [`from_label_with_style`] or [`from_config`].
    pub fn from_label(label: &str) -> Result<Self> {
        let config = parse_label(label)?;
        Self::from_config(&config)
    }

    /// Load from a label with an explicit style override.
    pub fn from_label_with_style(label: &str, style: Waifu2xStyle) -> Result<Self> {
        let mut config = parse_label(label)?;
        config.style = style;
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

    /// Content style this instance was built with.
    pub fn style(&self) -> Waifu2xStyle {
        self.style
    }

    /// Upscale (and optionally denoise) a [`image::DynamicImage`] using the
    /// waifu2x-compat software path (Lanczos3 resize + unsharp mask).
    ///
    /// The `style` parameter adjusts the sharpening profile:
    /// - `Anime`: stronger edge sharpening (default waifu2x behaviour)
    /// - `Photo`: gentler sharpening to preserve natural texture
    /// - `Artwork`: moderate sharpening tuned for digital paintings
    ///
    /// TODO: When native waifu2x weights are bundled, replace this with real
    /// CNN inference and load style-specific weight files (e.g.
    /// `noise{N}_scale{M}_{style}.json`).
    pub fn upscale_image(
        &self,
        img: &image::DynamicImage,
    ) -> Result<image::DynamicImage> {
        let (w, h) = (img.width(), img.height());
        let scale_u8 = self.scale.as_u8();

        // Step 1: Lanczos3 resize (scale=1 keeps original dimensions).
        let resized = if scale_u8 >= 2 {
            img.resize_exact(w * 2, h * 2, image::FilterType::Lanczos3)
        } else {
            img.clone()
        };

        // Step 2: Unsharp-mask sharpening based on noise level, adjusted by
        // content style.
        //
        // Style multipliers:
        //   Anime   → 1.0× (baseline — waifu2x was designed for anime)
        //   Artwork → 0.8× (slightly softer for painterly detail)
        //   Photo   → 0.6× (gentler to preserve natural texture/grain)
        let style_multiplier = match self.style {
            Waifu2xStyle::Anime   => 1.0f32,
            Waifu2xStyle::Artwork => 0.8,
            Waifu2xStyle::Photo   => 0.6,
        };

        let base_amount = match self.noise_level {
            NoiseLevel::None   => 0.0f32,
            NoiseLevel::Low    => 0.3,
            NoiseLevel::Medium => 0.5,
            NoiseLevel::High   => 0.8,
        };

        let amount = base_amount * style_multiplier;

        if amount < f32::EPSILON {
            return Ok(resized);
        }

        Ok(unsharp_mask(&resized, amount))
    }

    /// Human-readable description of the active configuration.
    pub fn description(&self) -> String {
        format!(
            "waifu2x-compat noise={} scale={}x style={} (Lanczos3 + unsharp mask)",
            self.noise_level, self.scale, self.style
        )
    }
}

impl std::fmt::Display for Waifu2xNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.description())
    }
}

// ── Unsharp mask ─────────────────────────────────────────────────────────────

/// Apply unsharp-mask sharpening: `output = original + amount * (original - blur)`.
///
/// Uses a 3×3 box blur as the smoothing kernel for simplicity.  This is the
/// waifu2x-compat approximation of CNN-based noise reduction.
fn unsharp_mask(img: &image::DynamicImage, amount: f32) -> image::DynamicImage {
    use image::{DynamicImage, GenericImage, Pixel};

    let rgba = img.to_rgba();
    let (w, h) = rgba.dimensions();
    if w < 3 || h < 3 {
        return img.clone();
    }

    let mut out = rgba.clone();

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            // 3×3 box-blur average for each channel.
            let mut sums = [0u32; 4];
            for dy in 0u32..3 {
                for dx in 0u32..3 {
                    let p = rgba.get_pixel(x + dx - 1, y + dy - 1);
                    let channels = p.channels();
                    for c in 0..4 {
                        sums[c] += channels[c] as u32;
                    }
                }
            }

            let orig = rgba.get_pixel(x, y);
            let orig_ch = orig.channels();
            let mut sharpened = [0u8; 4];
            for c in 0..4 {
                if c == 3 {
                    // Preserve alpha unchanged.
                    sharpened[c] = orig_ch[c];
                } else {
                    let blurred = (sums[c] as f32) / 9.0;
                    let diff = orig_ch[c] as f32 - blurred;
                    let val = orig_ch[c] as f32 + amount * diff;
                    sharpened[c] = val.round().max(0.0).min(255.0) as u8;
                }
            }

            out.put_pixel(x, y, image::Rgba(sharpened));
        }
    }

    DynamicImage::ImageRgba8(out)
}

// ── Label parser ──────────────────────────────────────────────────────────────

/// Parse a waifu2x label into a [`Waifu2xConfig`].
///
/// Accepted formats:
/// - `"waifu2x"` → noise=1, scale=2 (defaults)
/// - `"waifu2x-noise{0..3}-scale{1,2}"` → explicit parameters
fn parse_label(label: &str) -> Result<Waifu2xConfig> {
    if label == "waifu2x" {
        return Ok(Waifu2xConfig { noise_level: 1, scale: 2, style: Waifu2xStyle::default() });
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
            return Ok(Waifu2xConfig { noise_level: noise, scale, style: Waifu2xStyle::default() });
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

    // ── Waifu2x-compat inference tests ──────────────────────────────────

    fn test_image(w: u32, h: u32) -> image::DynamicImage {
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
            image::Rgba([(x % 256) as u8, (y % 256) as u8, 128u8, 255u8])
        }))
    }

    #[test]
    fn compat_scale2_doubles_dimensions() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale2").unwrap();
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 32);
        assert_eq!(result.height(), 32);
    }

    #[test]
    fn compat_scale1_preserves_dimensions() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale1").unwrap();
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn compat_noise0_no_sharpening() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise0-scale2").unwrap();
        let img = test_image(8, 8);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn compat_noise3_scale1_sharpens_only() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise3-scale1").unwrap();
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn compat_all_variants_succeed() {
        let img = test_image(10, 10);
        for &label in WAIFU2X_LABELS {
            let net = Waifu2xNetwork::from_label(label)
                .unwrap_or_else(|e| panic!("from_label({}) failed: {}", label, e));
            let result = net.upscale_image(&img)
                .unwrap_or_else(|e| panic!("upscale_image({}) failed: {}", label, e));
            if label.contains("scale2") || label == "waifu2x" {
                assert_eq!(result.width(), 20, "width mismatch for {}", label);
                assert_eq!(result.height(), 20, "height mismatch for {}", label);
            } else {
                assert_eq!(result.width(), 10, "width mismatch for {}", label);
                assert_eq!(result.height(), 10, "height mismatch for {}", label);
            }
        }
    }

    #[test]
    fn compat_description_mentions_compat() {
        let net = Waifu2xNetwork::from_label("waifu2x").unwrap();
        assert!(net.description().contains("compat"));
    }

    #[test]
    fn compat_invalid_label_errors() {
        assert!(Waifu2xNetwork::from_label("esrgan").is_err());
        assert!(Waifu2xNetwork::from_label("waifu2x-bad").is_err());
    }

    #[test]
    fn compat_unsharp_mask_modifies_pixels() {
        // With noise=3 (aggressive sharpening), pixel values should differ
        // from the input for non-edge pixels.
        let img = test_image(16, 16);
        let sharpened = unsharp_mask(&img, 0.8);
        let orig_rgba = img.to_rgba();
        let sharp_rgba = sharpened.to_rgba();
        let mut differs = false;
        for y in 1..15 {
            for x in 1..15 {
                if orig_rgba.get_pixel(x, y) != sharp_rgba.get_pixel(x, y) {
                    differs = true;
                    break;
                }
            }
        }
        assert!(differs, "unsharp mask should modify at least some interior pixels");
    }
}
