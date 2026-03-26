//! Output quality and format options.
//!
//! Provides [`OutputConfig`] — a set of knobs that control how the
//! upscaled result is encoded and delivered.

use crate::error::SrganError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-request output options sent as JSON in the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output image format: `"png"`, `"jpeg"`, or `"webp"`.
    /// Defaults to `"png"`.
    #[serde(default = "default_format")]
    pub format: String,

    /// JPEG quality (1–100).  Only used when `format` is `"jpeg"`.
    /// Defaults to 85.
    #[serde(default = "default_quality")]
    pub quality: u8,

    /// Desired scale factor (2, 3, or 4).  The neural network always
    /// produces a 4× result; factors < 4 are achieved by downscaling
    /// the 4× output with Lanczos3.  Defaults to 4.
    #[serde(default = "default_scale")]
    pub scale: u8,

    /// When `true`, attempt to copy EXIF metadata from the input image
    /// into the output.  Defaults to `false`.
    #[serde(default)]
    pub preserve_metadata: bool,
}

fn default_format() -> String {
    "png".into()
}
fn default_quality() -> u8 {
    85
}
fn default_scale() -> u8 {
    4
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: default_format(),
            quality: default_quality(),
            scale: default_scale(),
            preserve_metadata: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl OutputConfig {
    /// Validate the config, returning a descriptive error on bad values.
    pub fn validate(&self) -> Result<(), SrganError> {
        let fmt = self.format.to_lowercase();
        if fmt != "png" && fmt != "jpeg" && fmt != "jpg" && fmt != "webp" {
            return Err(SrganError::InvalidParameter(format!(
                "unsupported output format '{}'; expected png, jpeg, or webp",
                self.format
            )));
        }
        if self.quality == 0 || self.quality > 100 {
            return Err(SrganError::InvalidParameter(format!(
                "quality must be 1–100, got {}",
                self.quality
            )));
        }
        if self.scale == 0 || self.scale > 4 {
            return Err(SrganError::InvalidParameter(format!(
                "scale must be 1–4, got {}",
                self.scale
            )));
        }
        Ok(())
    }

    /// Return the validated, normalised format string (`"png"`, `"jpeg"`, or `"webp"`).
    pub fn effective_format(&self) -> &str {
        match self.format.to_lowercase().as_str() {
            "jpg" | "jpeg" => "jpeg",
            "webp" => "webp",
            _ => "png",
        }
    }
}

// ---------------------------------------------------------------------------
// Post-upscale rescaling
// ---------------------------------------------------------------------------

/// If the requested scale is less than the native 4×, downscale the
/// 4× result to the target size using Lanczos3.
pub fn apply_output_scale(
    img: image::DynamicImage,
    original_width: u32,
    original_height: u32,
    target_scale: u8,
) -> image::DynamicImage {
    if target_scale >= 4 {
        return img;
    }
    let target_w = original_width * target_scale as u32;
    let target_h = original_height * target_scale as u32;
    img.resize_exact(target_w, target_h, image::FilterType::Lanczos3)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImage;

    #[test]
    fn default_config_is_valid() {
        OutputConfig::default().validate().unwrap();
    }

    #[test]
    fn validates_format() {
        let cfg = OutputConfig { format: "bmp".into(), ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validates_quality_zero() {
        let cfg = OutputConfig { quality: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validates_quality_over_100() {
        let cfg = OutputConfig { quality: 101, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validates_scale_zero() {
        let cfg = OutputConfig { scale: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validates_scale_over_4() {
        let cfg = OutputConfig { scale: 5, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn effective_format_normalises() {
        assert_eq!(OutputConfig { format: "jpg".into(), ..Default::default() }.effective_format(), "jpeg");
        assert_eq!(OutputConfig { format: "JPEG".into(), ..Default::default() }.effective_format(), "jpeg");
        assert_eq!(OutputConfig { format: "png".into(), ..Default::default() }.effective_format(), "png");
        assert_eq!(OutputConfig { format: "webp".into(), ..Default::default() }.effective_format(), "webp");
        assert_eq!(OutputConfig { format: "unknown".into(), ..Default::default() }.effective_format(), "png");
    }

    #[test]
    fn apply_output_scale_4x_noop() {
        let img = image::DynamicImage::new_rgb8(40, 40);
        let result = apply_output_scale(img, 10, 10, 4);
        assert_eq!(result.width(), 40);
        assert_eq!(result.height(), 40);
    }

    #[test]
    fn apply_output_scale_2x() {
        let img = image::DynamicImage::new_rgb8(40, 40);
        let result = apply_output_scale(img, 10, 10, 2);
        assert_eq!(result.width(), 20);
        assert_eq!(result.height(), 20);
    }

    #[test]
    fn deserialize_defaults() {
        let json = "{}";
        let cfg: OutputConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.format, "png");
        assert_eq!(cfg.quality, 85);
        assert_eq!(cfg.scale, 4);
        assert!(!cfg.preserve_metadata);
    }

    #[test]
    fn deserialize_partial() {
        let json = r#"{"format": "jpeg", "quality": 90}"#;
        let cfg: OutputConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.format, "jpeg");
        assert_eq!(cfg.quality, 90);
        assert_eq!(cfg.scale, 4);
    }
}
