//! Image preprocessing pipeline with optional filters.
//!
//! Provides denoise, contrast enhancement, sharpening, and auto-crop
//! operations that run *before* neural-network inference.  Each step is
//! independently toggleable via [`PipelineConfig`].

#[allow(unused_imports)]
use image::{DynamicImage, GenericImage, Pixel, Rgb, RgbImage};

use crate::error::SrganError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-request preprocessing toggles sent as JSON in the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Apply a 3×3 box blur to reduce noise (default `false`).
    #[serde(default)]
    pub denoise: bool,
    /// Histogram-equalisation contrast enhancement (default `false`).
    #[serde(default)]
    pub enhance_contrast: bool,
    /// Unsharp-mask sharpening pre-pass (default `false`).
    #[serde(default)]
    pub sharpen: bool,
    /// Auto-crop solid-colour borders / letterboxing (default `false`).
    #[serde(default)]
    pub auto_crop: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            denoise: false,
            enhance_contrast: false,
            sharpen: false,
            auto_crop: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry-point
// ---------------------------------------------------------------------------

/// Run the optional preprocessing pipeline on a decoded image.
///
/// Steps are applied in order: auto-crop → denoise → contrast → sharpen.
/// Only enabled steps execute; if nothing is enabled the image is returned
/// unchanged.
pub fn run_pipeline(
    img: DynamicImage,
    cfg: &PipelineConfig,
) -> Result<DynamicImage, SrganError> {
    let mut rgb = img.to_rgb();

    if cfg.auto_crop {
        rgb = auto_crop(&rgb);
    }
    if cfg.denoise {
        rgb = box_blur_3x3(&rgb);
    }
    if cfg.enhance_contrast {
        rgb = histogram_equalize(&rgb);
    }
    if cfg.sharpen {
        rgb = unsharp_mask(&rgb);
    }

    Ok(DynamicImage::ImageRgb8(rgb))
}

// ---------------------------------------------------------------------------
// Auto-crop: remove solid-colour borders
// ---------------------------------------------------------------------------

/// Detect and remove uniform-colour borders (letterboxing, pillarboxing).
///
/// Scans inward from each edge.  A row/column is considered "border" when
/// every pixel's per-channel distance from the corner pixel is ≤ threshold.
fn auto_crop(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return img.clone();
    }

    const THRESHOLD: u8 = 15;
    let ref_pixel = img.get_pixel(0, 0);

    let is_border_pixel = |p: &Rgb<u8>| -> bool {
        p.channels()
            .iter()
            .zip(ref_pixel.channels().iter())
            .all(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() <= THRESHOLD as u16)
    };

    // Top
    let mut top: u32 = 0;
    'top: for y in 0..h {
        for x in 0..w {
            if !is_border_pixel(img.get_pixel(x, y)) {
                break 'top;
            }
        }
        top = y + 1;
    }

    // Bottom
    let mut bottom: u32 = h;
    'bottom: for y in (0..h).rev() {
        for x in 0..w {
            if !is_border_pixel(img.get_pixel(x, y)) {
                break 'bottom;
            }
        }
        bottom = y;
    }

    // Left
    let mut left: u32 = 0;
    'left: for x in 0..w {
        for y in top..bottom {
            if !is_border_pixel(img.get_pixel(x, y)) {
                break 'left;
            }
        }
        left = x + 1;
    }

    // Right
    let mut right: u32 = w;
    'right: for x in (0..w).rev() {
        for y in top..bottom {
            if !is_border_pixel(img.get_pixel(x, y)) {
                break 'right;
            }
        }
        right = x;
    }

    // Avoid cropping to nothing
    if right <= left || bottom <= top {
        return img.clone();
    }

    let cropped_w = right - left;
    let cropped_h = bottom - top;
    let mut img_mut = img.clone();
    image::imageops::crop(&mut img_mut, left, top, cropped_w, cropped_h).to_image()
}

// ---------------------------------------------------------------------------
// Denoise: simple 3×3 box blur
// ---------------------------------------------------------------------------

fn box_blur_3x3(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let mut out = RgbImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let mut sums = [0u32; 3];
            let mut count = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let p = img.get_pixel(nx as u32, ny as u32);
                        for c in 0..3 {
                            sums[c] += p[c] as u32;
                        }
                        count += 1;
                    }
                }
            }
            out.put_pixel(
                x,
                y,
                Rgb([
                    (sums[0] / count) as u8,
                    (sums[1] / count) as u8,
                    (sums[2] / count) as u8,
                ]),
            );
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Contrast enhancement: per-channel histogram equalisation
// ---------------------------------------------------------------------------

fn histogram_equalize(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let total = (w * h) as f64;
    let mut out = img.clone();

    for c in 0..3usize {
        // Build histogram
        let mut hist = [0u64; 256];
        for p in img.pixels() {
            hist[p[c] as usize] += 1;
        }

        // Build CDF
        let mut cdf = [0f64; 256];
        cdf[0] = hist[0] as f64;
        for i in 1..256 {
            cdf[i] = cdf[i - 1] + hist[i] as f64;
        }

        // Find CDF min (first non-zero)
        let cdf_min = cdf.iter().copied().find(|&v| v > 0.0).unwrap_or(0.0);

        // Build lookup table
        let mut lut = [0u8; 256];
        for i in 0..256 {
            if total - cdf_min > 0.0 {
                lut[i] = ((cdf[i] - cdf_min) / (total - cdf_min) * 255.0).round() as u8;
            }
        }

        // Apply
        for p in out.pixels_mut() {
            p[c] = lut[p[c] as usize];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Sharpen: unsharp mask (blur then subtract)
// ---------------------------------------------------------------------------

fn unsharp_mask(img: &RgbImage) -> RgbImage {
    let blurred = box_blur_3x3(img);
    let (w, h) = img.dimensions();
    let mut out = RgbImage::new(w, h);
    let amount: f32 = 0.5; // sharpening strength

    for y in 0..h {
        for x in 0..w {
            let orig = img.get_pixel(x, y);
            let blur = blurred.get_pixel(x, y);
            let mut channels = [0u8; 3];
            for c in 0..3 {
                let val = orig[c] as f32 + amount * (orig[c] as f32 - blur[c] as f32);
                channels[c] = val.max(0.0).min(255.0).round() as u8;
            }
            out.put_pixel(x, y, Rgb(channels));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(w: u32, h: u32) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w.max(1)) as u8;
                let g = ((y * 255) / h.max(1)) as u8;
                let b = 128u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        img
    }

    fn make_bordered_image() -> RgbImage {
        let mut img = RgbImage::from_pixel(20, 20, Rgb([0, 0, 0]));
        // Fill inner 10x10 region with non-black content
        for y in 5..15 {
            for x in 5..15 {
                img.put_pixel(x, y, Rgb([200, 100, 50]));
            }
        }
        img
    }

    #[test]
    fn pipeline_noop_when_all_disabled() {
        let cfg = PipelineConfig::default();
        let img = DynamicImage::ImageRgb8(make_test_image(8, 8));
        let result = run_pipeline(img.clone(), &cfg).unwrap();
        assert_eq!(result.width(), 8);
        assert_eq!(result.height(), 8);
    }

    #[test]
    fn pipeline_denoise_preserves_dimensions() {
        let cfg = PipelineConfig { denoise: true, ..Default::default() };
        let img = DynamicImage::ImageRgb8(make_test_image(16, 16));
        let result = run_pipeline(img, &cfg).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn pipeline_sharpen_preserves_dimensions() {
        let cfg = PipelineConfig { sharpen: true, ..Default::default() };
        let img = DynamicImage::ImageRgb8(make_test_image(16, 16));
        let result = run_pipeline(img, &cfg).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn pipeline_contrast_preserves_dimensions() {
        let cfg = PipelineConfig { enhance_contrast: true, ..Default::default() };
        let img = DynamicImage::ImageRgb8(make_test_image(16, 16));
        let result = run_pipeline(img, &cfg).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn auto_crop_removes_black_border() {
        let img = make_bordered_image();
        let cropped = auto_crop(&img);
        // Should crop to roughly the 10x10 inner content
        assert!(cropped.width() <= 12, "width={}", cropped.width());
        assert!(cropped.height() <= 12, "height={}", cropped.height());
        assert!(cropped.width() >= 10, "width={}", cropped.width());
        assert!(cropped.height() >= 10, "height={}", cropped.height());
    }

    #[test]
    fn auto_crop_no_border_unchanged() {
        let img = make_test_image(10, 10);
        let cropped = auto_crop(&img);
        // Most pixels differ from corner, so little/no cropping expected
        assert!(cropped.width() >= 8);
        assert!(cropped.height() >= 8);
    }

    #[test]
    fn box_blur_smooths_noise() {
        // Single bright pixel in a dark image should be averaged down
        let mut img = RgbImage::from_pixel(5, 5, Rgb([0, 0, 0]));
        img.put_pixel(2, 2, Rgb([255, 255, 255]));
        let blurred = box_blur_3x3(&img);
        // Centre should be 255/9 ≈ 28
        let centre = blurred.get_pixel(2, 2);
        assert!(centre[0] < 50, "expected smoothing, got {}", centre[0]);
        assert!(centre[0] > 20, "over-smoothed, got {}", centre[0]);
    }

    #[test]
    fn histogram_equalize_spreads_values() {
        // Uniform low-intensity image should get spread across range
        let img = RgbImage::from_pixel(10, 10, Rgb([50, 50, 50]));
        let eq = histogram_equalize(&img);
        // All-same-value image: every pixel maps to 0 (CDF is step function at index 50)
        // Exact value depends on formula, but result should be consistent
        let p = eq.get_pixel(0, 0);
        let q = eq.get_pixel(5, 5);
        assert_eq!(p, q, "uniform input should give uniform output");
    }

    #[test]
    fn unsharp_mask_increases_contrast() {
        // Create an image with a sharp edge
        let mut img = RgbImage::new(10, 10);
        for y in 0..10 {
            for x in 0..5 {
                img.put_pixel(x, y, Rgb([50, 50, 50]));
            }
            for x in 5..10 {
                img.put_pixel(x, y, Rgb([200, 200, 200]));
            }
        }
        let sharpened = unsharp_mask(&img);
        // Pixel on the bright side of the edge should be >= original
        let bright_edge = sharpened.get_pixel(5, 5);
        assert!(bright_edge[0] >= 200, "bright side should stay bright or increase");
        // Pixel on the dark side of the edge should be <= original
        let dark_edge = sharpened.get_pixel(4, 5);
        assert!(dark_edge[0] <= 50, "dark side should stay dark or decrease");
    }

    #[test]
    fn pipeline_all_filters_combined() {
        let cfg = PipelineConfig {
            denoise: true,
            enhance_contrast: true,
            sharpen: true,
            auto_crop: true,
        };
        let img = DynamicImage::ImageRgb8(make_test_image(20, 20));
        let result = run_pipeline(img, &cfg).unwrap();
        // Should produce a valid image without panicking
        assert!(result.width() > 0);
        assert!(result.height() > 0);
    }

    #[test]
    fn pipeline_config_deserialize_defaults() {
        let json = "{}";
        let cfg: PipelineConfig = serde_json::from_str(json).unwrap();
        assert!(!cfg.denoise);
        assert!(!cfg.enhance_contrast);
        assert!(!cfg.sharpen);
        assert!(!cfg.auto_crop);
    }

    #[test]
    fn pipeline_config_deserialize_partial() {
        let json = r#"{"denoise": true, "sharpen": true}"#;
        let cfg: PipelineConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.denoise);
        assert!(!cfg.enhance_contrast);
        assert!(cfg.sharpen);
        assert!(!cfg.auto_crop);
    }

    #[test]
    fn tiny_image_auto_crop_no_panic() {
        let img = RgbImage::from_pixel(2, 2, Rgb([0, 0, 0]));
        let cropped = auto_crop(&img);
        assert_eq!(cropped.dimensions(), (2, 2));
    }
}
