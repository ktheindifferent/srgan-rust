//! Waifu2x model integration for the `models` module.
//!
//! Re-exports the core waifu2x types from [`crate::waifu2x`] so that all
//! model implementations live under a single `models` namespace, consistent
//! with the [`crate::models::real_esrgan`] module.
//!
//! ## Quick reference
//!
//! | Type               | Purpose                                              |
//! |--------------------|------------------------------------------------------|
//! | [`Waifu2xVariant`] | Enum over all noise × scale combinations             |
//! | [`Waifu2xNetwork`] | High-level wrapper — `upscale_image` entry point     |
//! | [`NoiseLevel`]     | Noise-reduction strength (0 = none … 3 = aggressive) |
//! | [`Waifu2xScale`]   | Output scale factor (×1 = denoise-only, ×2 = upscale)|
//! | [`WAIFU2X_LABELS`] | All canonical CLI/API label strings                  |
//!
//! ## Usage
//!
//! ```no_run
//! use srgan_rust::models::waifu2x::{Waifu2xNetwork, Waifu2xVariant};
//!
//! // Build from the default label ("waifu2x" → noise=1, scale=2)
//! let net = Waifu2xNetwork::from_label("waifu2x").unwrap();
//!
//! // Build from an explicit variant
//! let variant = Waifu2xVariant::new(2, 2).unwrap();
//! let net = Waifu2xNetwork::from_label(&variant.label()).unwrap();
//! ```

pub use crate::waifu2x::{NoiseLevel, Waifu2xNetwork, Waifu2xScale, WAIFU2X_LABELS};

// ── Waifu2xVariant ────────────────────────────────────────────────────────────

/// A concrete waifu2x configuration: one noise level combined with one scale
/// factor.
///
/// Mirrors the role of [`crate::models::real_esrgan::RealEsrganVariant`] for
/// Real-ESRGAN.  Use [`Waifu2xVariant::label`] to obtain the canonical
/// CLI/API label that can be passed to
/// [`crate::thread_safe_network::ThreadSafeNetwork::from_label`] or
/// [`Waifu2xNetwork::from_label`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Waifu2xVariant {
    /// Noise-reduction level (0–3).
    pub noise_level: u8,
    /// Upscaling factor (1 or 2).
    pub scale: u8,
}

impl Waifu2xVariant {
    /// Build a variant, validating that `noise_level` is 0–3 and `scale` is
    /// 1 or 2.
    ///
    /// Returns `None` for out-of-range values.
    pub fn new(noise_level: u8, scale: u8) -> Option<Self> {
        if noise_level > 3 { return None; }
        if scale != 1 && scale != 2 { return None; }
        Some(Self { noise_level, scale })
    }

    /// The default variant used when `model = "waifu2x"` is requested without
    /// explicit noise/scale parameters: noise=1, scale=2.
    pub fn default_variant() -> Self {
        Self { noise_level: 1, scale: 2 }
    }

    /// Return the canonical CLI/API label string for this variant.
    ///
    /// Examples: `"waifu2x-noise1-scale2"`, `"waifu2x-noise0-scale1"`.
    pub fn label(self) -> String {
        format!("waifu2x-noise{}-scale{}", self.noise_level, self.scale)
    }

    /// Nominal scale factor advertised by this variant (1 or 2).
    pub fn scale_factor(self) -> u32 {
        self.scale as u32
    }

    /// Noise reduction level as [`NoiseLevel`].
    pub fn noise(self) -> NoiseLevel {
        NoiseLevel::from_u8(self.noise_level)
    }

    /// Human-readable description.
    pub fn description(self) -> String {
        let noise_desc = match self.noise_level {
            0 => "no noise reduction",
            1 => "light noise reduction",
            2 => "medium noise reduction",
            _ => "aggressive noise reduction",
        };
        let scale_desc = match self.scale {
            1 => "denoise-only (×1)",
            _ => "upscale ×2",
        };
        format!(
            "waifu2x — {} + {} (backed by built-in anime model; \
             TODO: load native waifu2x weights)",
            noise_desc, scale_desc
        )
    }

    /// Attempt to parse a label string into a `Waifu2xVariant`.
    ///
    /// Accepts `"waifu2x"` (returns the default variant) or
    /// `"waifu2x-noise{N}-scale{M}"`.
    pub fn from_label(label: &str) -> Option<Self> {
        if label == "waifu2x" {
            return Some(Self::default_variant());
        }
        let rest = label.strip_prefix("waifu2x-")?;
        let parts: Vec<&str> = rest.split('-').collect();
        if parts.len() != 2 { return None; }
        let noise = parts[0].strip_prefix("noise")?.parse::<u8>().ok()?.min(3);
        let scale = parts[1].strip_prefix("scale")?.parse::<u8>().ok()?;
        let scale = if scale == 1 { 1 } else { 2 };
        Some(Self { noise_level: noise, scale })
    }
}

impl std::fmt::Display for Waifu2xVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_variant_label() {
        assert_eq!(Waifu2xVariant::default_variant().label(), "waifu2x-noise1-scale2");
    }

    #[test]
    fn new_out_of_range() {
        assert!(Waifu2xVariant::new(4, 2).is_none()); // noise > 3
        assert!(Waifu2xVariant::new(1, 3).is_none()); // scale must be 1 or 2
        assert!(Waifu2xVariant::new(0, 0).is_none()); // scale 0 invalid
    }

    #[test]
    fn new_valid() {
        let v = Waifu2xVariant::new(2, 1).unwrap();
        assert_eq!(v.label(), "waifu2x-noise2-scale1");
    }

    #[test]
    fn from_label_bare() {
        let v = Waifu2xVariant::from_label("waifu2x").unwrap();
        assert_eq!(v.noise_level, 1);
        assert_eq!(v.scale, 2);
    }

    #[test]
    fn from_label_parameterised() {
        let v = Waifu2xVariant::from_label("waifu2x-noise3-scale1").unwrap();
        assert_eq!(v.noise_level, 3);
        assert_eq!(v.scale, 1);
    }

    #[test]
    fn from_label_unknown() {
        assert!(Waifu2xVariant::from_label("waifu2x-bad").is_none());
        assert!(Waifu2xVariant::from_label("natural").is_none());
    }

    #[test]
    fn all_waifu2x_labels_parse() {
        for &label in WAIFU2X_LABELS {
            assert!(
                Waifu2xVariant::from_label(label).is_some(),
                "should parse: {}",
                label
            );
        }
    }

    #[test]
    fn scale_factor() {
        assert_eq!(Waifu2xVariant::new(0, 1).unwrap().scale_factor(), 1);
        assert_eq!(Waifu2xVariant::new(0, 2).unwrap().scale_factor(), 2);
    }
}
