//! Real-ESRGAN model support for super-resolution with real-world degradations.
//!
//! Real-ESRGAN extends ESRGAN by training on synthetic real-world degradations
//! (JPEG compression artifacts, Gaussian noise, motion blur, downscaling
//! pipelines). It excels at restoring severely degraded images where classic
//! SRGAN or plain ESRGAN produce over-smooth results.
//!
//! ## Variants
//!
//! | CLI label           | Use case                        | Scale |
//! |---------------------|---------------------------------|-------|
//! | `real-esrgan`       | General photos, compressed imgs | ×4    |
//! | `real-esrgan-anime` | Anime / illustration content    | ×4    |
//! | `real-esrgan-x2`    | General photos, lower memory    | ×2    |
//!
//! ## Weight status
//!
//! The official Real-ESRGAN weights are distributed as ONNX / PyTorch
//! checkpoints which use a different serialisation format from the `.rsr`
//! format used by this binary.  Conversion tooling (ONNX → `.rsr`) is tracked
//! as a TODO.
//!
//! In the meantime each variant falls back to the best available built-in
//! model:
//! - `real-esrgan`       → built-in `natural` model (general photo content)
//! - `real-esrgan-anime` → built-in `anime` model   (animation content)
//! - `real-esrgan-x2`    → built-in `natural` model (same network, ×4 reported
//!                          as ×2 until dedicated ×2 weights land)
//!
//! TODO: replace stubs with dedicated Real-ESRGAN weights once the
//!       ONNX → `.rsr` conversion pipeline is available.

use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;

// ── Variant ───────────────────────────────────────────────────────────────────

/// The three supported Real-ESRGAN model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealEsrganVariant {
    /// General-purpose model trained on real-world photo degradations (×4).
    X4Plus,
    /// Anime/illustration-optimised variant (×4).
    X4PlusAnime,
    /// Lighter general-purpose model for lower memory usage (×2).
    X2Plus,
}

impl RealEsrganVariant {
    /// Parse from the canonical CLI/API label.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "real-esrgan" => Some(RealEsrganVariant::X4Plus),
            "real-esrgan-anime" => Some(RealEsrganVariant::X4PlusAnime),
            "real-esrgan-x2" => Some(RealEsrganVariant::X2Plus),
            _ => None,
        }
    }

    /// Return the canonical CLI/API label for this variant.
    pub fn label(self) -> &'static str {
        match self {
            RealEsrganVariant::X4Plus => "real-esrgan",
            RealEsrganVariant::X4PlusAnime => "real-esrgan-anime",
            RealEsrganVariant::X2Plus => "real-esrgan-x2",
        }
    }

    /// Nominal scale factor advertised by this variant.
    pub fn scale_factor(self) -> u32 {
        match self {
            RealEsrganVariant::X4Plus | RealEsrganVariant::X4PlusAnime => 4,
            RealEsrganVariant::X2Plus => 2,
        }
    }

    /// Human-readable description.
    pub fn description(self) -> &'static str {
        match self {
            RealEsrganVariant::X4Plus =>
                "Real-ESRGAN ×4 — general photos; trained on synthetic real-world \
                 degradations (JPEG artifacts, noise, blur). \
                 TODO: load native Real-ESRGAN weights.",
            RealEsrganVariant::X4PlusAnime =>
                "Real-ESRGAN ×4 Anime — anime/illustration content; uses the \
                 anime-optimised degradation pipeline. \
                 TODO: load native Real-ESRGAN-Anime weights.",
            RealEsrganVariant::X2Plus =>
                "Real-ESRGAN ×2 — general photos at ×2 scale; lower memory than \
                 the ×4 variant. \
                 TODO: load dedicated ×2 Real-ESRGAN weights.",
        }
    }
}

impl std::fmt::Display for RealEsrganVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ── RealEsrganModel ───────────────────────────────────────────────────────────

/// High-level Real-ESRGAN wrapper.
///
/// Presents the same `upscale_image` interface as [`UpscalingNetwork`] while
/// encoding the Real-ESRGAN variant configuration.  The `inner` network is
/// currently one of the built-in stub models; swap it for dedicated
/// Real-ESRGAN weights once the conversion pipeline is ready.
pub struct RealEsrganModel {
    /// Which Real-ESRGAN variant this instance represents.
    variant: RealEsrganVariant,
    /// Underlying inference network (stub — currently a built-in model).
    inner: UpscalingNetwork,
}

impl RealEsrganModel {
    /// Build a [`RealEsrganModel`] from a variant enum value.
    pub fn from_variant(variant: RealEsrganVariant) -> Result<Self> {
        // Select the best available built-in proxy until native weights land.
        let inner_label = match variant {
            RealEsrganVariant::X4PlusAnime => "anime",
            RealEsrganVariant::X4Plus | RealEsrganVariant::X2Plus => "natural",
        };

        let inner = UpscalingNetwork::from_label(inner_label, None)
            .map_err(SrganError::Network)?;

        Ok(Self { variant, inner })
    }

    /// Build from a canonical CLI/API label (`"real-esrgan"`,
    /// `"real-esrgan-anime"`, `"real-esrgan-x2"`).
    pub fn from_label(label: &str) -> Result<Self> {
        let variant = RealEsrganVariant::from_label(label).ok_or_else(|| {
            SrganError::Network(format!(
                "invalid Real-ESRGAN label '{}'; expected one of: {}",
                label,
                REAL_ESRGAN_LABELS.join(", ")
            ))
        })?;
        Self::from_variant(variant)
    }

    /// The variant this model was built for.
    pub fn variant(&self) -> RealEsrganVariant {
        self.variant
    }

    /// Upscale a [`image::DynamicImage`] using this model.
    pub fn upscale_image(
        &self,
        img: &image::DynamicImage,
    ) -> Result<image::DynamicImage> {
        self.inner.upscale_image(img)
    }

    /// Human-readable description of the active configuration.
    pub fn description(&self) -> String {
        self.variant.description().to_string()
    }
}

impl std::fmt::Display for RealEsrganModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.variant.description())
    }
}

// ── Supported labels ──────────────────────────────────────────────────────────

/// All canonical Real-ESRGAN labels accepted by the CLI and API.
pub const REAL_ESRGAN_LABELS: &[&str] = &[
    "real-esrgan",
    "real-esrgan-anime",
    "real-esrgan-x2",
];

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_roundtrip() {
        for &label in REAL_ESRGAN_LABELS {
            let v = RealEsrganVariant::from_label(label).expect("known label");
            assert_eq!(v.label(), label);
        }
    }

    #[test]
    fn unknown_label_returns_none() {
        assert!(RealEsrganVariant::from_label("natural").is_none());
        assert!(RealEsrganVariant::from_label("esrgan").is_none());
        assert!(RealEsrganVariant::from_label("real-esrgan-x8").is_none());
    }

    #[test]
    fn scale_factors() {
        assert_eq!(RealEsrganVariant::X4Plus.scale_factor(), 4);
        assert_eq!(RealEsrganVariant::X4PlusAnime.scale_factor(), 4);
        assert_eq!(RealEsrganVariant::X2Plus.scale_factor(), 2);
    }

    #[test]
    fn from_label_error_message_contains_valid_labels() {
        let err = RealEsrganModel::from_label("bad-label").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("real-esrgan"), "error: {}", msg);
    }
}
