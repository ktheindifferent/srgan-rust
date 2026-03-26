//! Heuristic image type detection for automatic model selection.
//!
//! Classifies images as [`ImageType::Photo`], [`ImageType::Anime`], or
//! [`ImageType::Illustration`] using three signals:
//!
//! | Signal | Description |
//! |--------|-------------|
//! | Mean HSV saturation | High → vivid anime/illustration; moderate → photo |
//! | Saturation std-dev  | Low  → flat fills (anime); high → varied gradients (photo) |
//! | Flat-region fraction | High → anime/illustration; low → photo |
//! | Palette diversity    | Low  → limited color count (illustration/anime) |

use image::{DynamicImage, GenericImage};
use serde::{Deserialize, Serialize};

// ── ImageType ─────────────────────────────────────────────────────────────────

/// Detected image content type used to select the best upscaling model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageType {
    /// Natural photograph — real-world scene with camera noise and gradients.
    Photo,
    /// Anime-style — flat color fills, high saturation, clean outline edges.
    Anime,
    /// Illustration — drawn artwork with moderate color variation and limited palette.
    Illustration,
}

impl ImageType {
    /// Lowercase string identifier (e.g., `"photo"`, `"anime"`, `"illustration"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageType::Photo => "photo",
            ImageType::Anime => "anime",
            ImageType::Illustration => "illustration",
        }
    }
}

impl std::fmt::Display for ImageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Analyze `image` and return its detected [`ImageType`].
///
/// The heuristic combines mean HSV saturation, saturation standard deviation,
/// flat-region fraction, and sampled palette diversity.  No neural network is
/// used — analysis runs in a few milliseconds on typical images.
///
/// Model recommendations:
/// * `Photo`        → `"natural"` (Real-ESRGAN style)
/// * `Anime`        → `"waifu2x"`
/// * `Illustration` → `"anime"`
pub fn detect_image_type(image: &DynamicImage) -> ImageType {
    let f = extract_features(image);

    // Anime: high/moderate saturation + lots of flat fills + uniform saturation
    if f.mean_saturation > 0.28 && f.flat_fraction > 0.38 && f.std_saturation < 0.30 {
        return ImageType::Anime;
    }

    // Illustration: moderate saturation + limited palette + some flat regions
    if f.mean_saturation > 0.18 && f.palette_diversity < 0.35 && f.flat_fraction > 0.25 {
        return ImageType::Illustration;
    }

    // Default: photograph
    ImageType::Photo
}

/// Return the built-in model label that works best for the given image type.
///
/// | Type | Model label |
/// |------|-------------|
/// | `Photo` | `"natural"` |
/// | `Anime` | `"waifu2x"` |
/// | `Illustration` | `"anime"` |
pub fn recommended_model_for(image_type: &ImageType) -> &'static str {
    match image_type {
        ImageType::Photo => "natural",
        ImageType::Anime => "waifu2x",
        ImageType::Illustration => "anime",
    }
}

// ── Internal feature extraction ───────────────────────────────────────────────

struct Features {
    mean_saturation: f32,
    std_saturation: f32,
    flat_fraction: f32,
    /// Relative palette diversity: unique sampled colors / total samples (capped at 1).
    palette_diversity: f32,
}

fn extract_features(img: &DynamicImage) -> Features {
    let rgb = img.to_rgb();
    let (w, h) = img.dimensions();

    let total = (w * h).max(1);
    // Aim for ~4 000 samples regardless of image size.
    let stride = ((total / 4000) as u32).max(1);

    let mut saturations: Vec<f32> = Vec::with_capacity(4200);
    // 6-bit-per-channel palette key to reduce noise while keeping color diversity signal.
    let mut palette = std::collections::HashSet::<u32>::new();
    let mut sample_count = 0u32;

    let mut y = 0u32;
    while y < h {
        let mut x = 0u32;
        while x < w {
            let p = rgb.get_pixel(x, y);
            let r = p[0] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[2] as f32 / 255.0;

            saturations.push(rgb_saturation(r, g, b));

            let key = ((p[0] >> 2) as u32) << 12
                | ((p[1] >> 2) as u32) << 6
                | (p[2] >> 2) as u32;
            palette.insert(key);

            sample_count += 1;
            x += stride;
        }
        y += stride;
    }

    let n = saturations.len() as f32;
    let mean_sat = saturations.iter().sum::<f32>() / n;
    let std_sat = {
        let var = saturations.iter().map(|s| (s - mean_sat).powi(2)).sum::<f32>() / n;
        var.sqrt()
    };

    let palette_diversity = if sample_count > 0 {
        (palette.len() as f32 / sample_count as f32).min(1.0)
    } else {
        1.0
    };

    let flat_fraction = compute_flat_fraction(&rgb, w, h);

    Features {
        mean_saturation: mean_sat,
        std_saturation: std_sat,
        flat_fraction,
        palette_diversity,
    }
}

// ── Pixel helpers ─────────────────────────────────────────────────────────────

fn rgb_saturation(r: f32, g: f32, b: f32) -> f32 {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    if max < 1e-6 {
        0.0
    } else {
        (max - min) / max
    }
}

/// Fraction of sampled pixels whose 5-point cross neighbourhood has low luma variance
/// (i.e. flat-fill regions typical of anime/illustration).
fn compute_flat_fraction(rgb: &image::RgbImage, w: u32, h: u32) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let stride = ((w * h / 2000) as u32).max(2);
    let mut flat_count = 0u32;
    let mut total = 0u32;

    let mut y = 1u32;
    while y < h - 1 {
        let mut x = 1u32;
        while x < w - 1 {
            let lumas: [f32; 5] = [
                pixel_luma(rgb, x, y),
                pixel_luma(rgb, x - 1, y),
                pixel_luma(rgb, x + 1, y),
                pixel_luma(rgb, x, y - 1),
                pixel_luma(rgb, x, y + 1),
            ];
            let mean = lumas.iter().sum::<f32>() / 5.0;
            let var = lumas.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / 5.0;
            if var < 0.002 {
                flat_count += 1;
            }
            total += 1;
            x += stride;
        }
        y += stride;
    }

    if total == 0 {
        0.0
    } else {
        flat_count as f32 / total as f32
    }
}

fn pixel_luma(rgb: &image::RgbImage, x: u32, y: u32) -> f32 {
    let p = rgb.get_pixel(x, y);
    (p[0] as f32 * 0.299 + p[1] as f32 * 0.587 + p[2] as f32 * 0.114) / 255.0
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturation_gray() {
        assert!(rgb_saturation(0.5, 0.5, 0.5) < 1e-4);
    }

    #[test]
    fn test_saturation_red() {
        assert!(rgb_saturation(1.0, 0.0, 0.0) > 0.99);
    }

    #[test]
    fn test_gray_image_is_photo() {
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
            8,
            8,
            image::Rgb([128u8, 128, 128]),
        ));
        assert_eq!(detect_image_type(&img), ImageType::Photo);
    }

    #[test]
    fn test_vivid_flat_image_is_anime() {
        // A solid vivid-green image: max saturation, completely flat → anime
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
            16,
            16,
            image::Rgb([0u8, 200, 50]),
        ));
        assert_eq!(detect_image_type(&img), ImageType::Anime);
    }

    #[test]
    fn test_recommended_model_photo() {
        assert_eq!(recommended_model_for(&ImageType::Photo), "natural");
    }

    #[test]
    fn test_recommended_model_anime() {
        assert_eq!(recommended_model_for(&ImageType::Anime), "waifu2x");
    }

    #[test]
    fn test_recommended_model_illustration() {
        assert_eq!(recommended_model_for(&ImageType::Illustration), "anime");
    }
}
