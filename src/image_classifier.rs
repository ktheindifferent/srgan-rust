//! Heuristic-based image type classifier.
//!
//! Classifies an image as one of several types using simple statistical
//! features derived from pixel values — no neural network required.
//!
//! ## Detection strategy
//! | Feature | Description |
//! |---------|-------------|
//! | Mean HSV saturation | High → vivid anime/illustration; low → document/screenshot |
//! | Saturation std-dev | High → natural photo; low → flat anime fill regions |
//! | Edge density | Fraction of pixels with strong local gradient |
//! | Skin-tone fraction | Fraction of pixels matching a broad skin-tone range |
//! | Brightness mean | Low overall → document (dark text on white differs) |

use image::{DynamicImage, GenericImage};
use serde::{Deserialize, Serialize};

// ── ImageType ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImageType {
    Photograph,
    Anime,
    Screenshot,
    Document,
    FaceHeavy,
}

impl ImageType {
    pub fn slug(&self) -> &'static str {
        match self {
            ImageType::Photograph => "photograph",
            ImageType::Anime => "anime",
            ImageType::Screenshot => "screenshot",
            ImageType::Document => "document",
            ImageType::FaceHeavy => "face-heavy",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ImageType::Photograph => "Photograph",
            ImageType::Anime => "Anime / Illustration",
            ImageType::Screenshot => "Screenshot",
            ImageType::Document => "Document",
            ImageType::FaceHeavy => "Face-heavy Photo",
        }
    }

    /// The built-in model label that works best for this image type.
    pub fn recommended_model(&self) -> &'static str {
        match self {
            ImageType::Photograph => "natural",
            ImageType::Anime => "anime",
            ImageType::Screenshot => "natural",
            ImageType::Document => "natural",
            ImageType::FaceHeavy => "natural",
        }
    }
}

// ── ClassificationResult ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub detected_type: ImageType,
    /// Rough confidence in [0, 1].  Currently a simple rule-based score.
    pub confidence: f32,
    pub recommended_model: String,
    /// Human-readable explanation of which features drove the decision.
    pub reasoning: String,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Classify `img` and return a [`ClassificationResult`].
pub fn classify_image(img: &DynamicImage) -> ClassificationResult {
    let features = extract_features(img);
    classify_from_features(features)
}

/// Classify from a file path, opening the image internally.
pub fn classify_path(path: &std::path::Path) -> crate::error::Result<ClassificationResult> {
    let img = image::open(path).map_err(crate::error::SrganError::Image)?;
    Ok(classify_image(&img))
}

// ── Feature extraction ────────────────────────────────────────────────────────

struct ImageFeatures {
    mean_saturation: f32,
    std_saturation: f32,
    mean_brightness: f32,
    edge_density: f32,
    skin_fraction: f32,
    /// Fraction of pixels whose local neighbourhood has low variance (flat fill).
    flat_fraction: f32,
}

fn extract_features(img: &DynamicImage) -> ImageFeatures {
    let rgb = img.to_rgb();
    let (w, h) = img.dimensions();

    // Sample stride — aim for ~4000 samples regardless of image size.
    let total = (w * h).max(1);
    let stride = ((total / 4000) as u32).max(1);

    let mut saturations: Vec<f32> = Vec::with_capacity(4200);
    let mut brightnesses: Vec<f32> = Vec::with_capacity(4200);
    let mut skin_count = 0u32;
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
            brightnesses.push((r + g + b) / 3.0);

            if is_skin_tone(p[0], p[1], p[2]) {
                skin_count += 1;
            }
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
    let mean_brightness = brightnesses.iter().sum::<f32>() / n;
    let skin_fraction = if sample_count > 0 {
        skin_count as f32 / sample_count as f32
    } else {
        0.0
    };

    let edge_density = compute_edge_density(&rgb, w, h);
    let flat_fraction = compute_flat_fraction(&rgb, w, h);

    ImageFeatures {
        mean_saturation: mean_sat,
        std_saturation: std_sat,
        mean_brightness,
        edge_density,
        skin_fraction,
        flat_fraction,
    }
}

// ── Classification rules ──────────────────────────────────────────────────────

fn classify_from_features(f: ImageFeatures) -> ClassificationResult {
    // --- Document ---
    // Very desaturated overall, mostly light background (high brightness).
    if f.mean_saturation < 0.08 && f.mean_brightness > 0.65 {
        return ClassificationResult {
            detected_type: ImageType::Document,
            confidence: score(f.mean_saturation, 0.0, 0.08) * 0.9 + 0.1,
            recommended_model: ImageType::Document.recommended_model().to_string(),
            reasoning: format!(
                "Very low saturation ({:.2}) and high brightness ({:.2}) indicate a document or text image.",
                f.mean_saturation, f.mean_brightness
            ),
        };
    }

    // --- Screenshot ---
    // Low saturation, high edge density (UI widgets, text), low brightness variance.
    if f.mean_saturation < 0.20 && f.edge_density > 0.18 && f.flat_fraction > 0.45 {
        return ClassificationResult {
            detected_type: ImageType::Screenshot,
            confidence: 0.75,
            recommended_model: ImageType::Screenshot.recommended_model().to_string(),
            reasoning: format!(
                "Low saturation ({:.2}), high edge density ({:.2}), and large flat regions ({:.0}%) suggest a UI screenshot.",
                f.mean_saturation, f.edge_density, f.flat_fraction * 100.0
            ),
        };
    }

    // --- Anime / Illustration ---
    // High or moderate saturation, large flat-color regions, clean sparse edges.
    if f.mean_saturation > 0.28 && f.flat_fraction > 0.40 && f.std_saturation < 0.28 {
        return ClassificationResult {
            detected_type: ImageType::Anime,
            confidence: 0.80,
            recommended_model: ImageType::Anime.recommended_model().to_string(),
            reasoning: format!(
                "High saturation ({:.2}) with large flat regions ({:.0}%) and uniform saturation (std {:.2}) match anime/illustration style.",
                f.mean_saturation, f.flat_fraction * 100.0, f.std_saturation
            ),
        };
    }

    // --- Face-heavy ---
    // Significant skin-tone coverage.
    if f.skin_fraction > 0.22 {
        return ClassificationResult {
            detected_type: ImageType::FaceHeavy,
            confidence: (f.skin_fraction - 0.22) / 0.28 + 0.6,
            recommended_model: ImageType::FaceHeavy.recommended_model().to_string(),
            reasoning: format!(
                "High skin-tone pixel fraction ({:.0}%) suggests a portrait or face-heavy image.",
                f.skin_fraction * 100.0
            ),
        };
    }

    // --- Photograph (default) ---
    ClassificationResult {
        detected_type: ImageType::Photograph,
        confidence: 0.65,
        recommended_model: ImageType::Photograph.recommended_model().to_string(),
        reasoning: format!(
            "Natural saturation distribution (mean {:.2}, std {:.2}) and edge density ({:.2}) match a photograph.",
            f.mean_saturation, f.std_saturation, f.edge_density
        ),
    }
}

// ── Image feature helpers ─────────────────────────────────────────────────────

/// Compute HSV saturation for an RGB pixel (all channels in [0, 1]).
fn rgb_saturation(r: f32, g: f32, b: f32) -> f32 {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    if max < 1e-6 {
        0.0
    } else {
        (max - min) / max
    }
}

/// True if the RGB byte values fall within a broad skin-tone range.
fn is_skin_tone(r: u8, g: u8, b: u8) -> bool {
    // Simple empirical rule (works for diverse skin tones under typical lighting):
    // R > 95, G > 40, B > 20
    // max - min > 15 (not gray)
    // R > G > B
    let max = r.max(g).max(b) as i32;
    let min = r.min(g).min(b) as i32;
    r > 95 && g > 40 && b > 20 && (max - min) > 15 && r > g && g > b
}

/// Approximate edge density using a sampled Sobel-like gradient.
/// Samples a coarse grid to keep cost low.
fn compute_edge_density(rgb: &image::RgbImage, w: u32, h: u32) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let stride = ((w * h / 2000) as u32).max(2);
    let mut edge_count = 0u32;
    let mut total = 0u32;

    let mut y = 1u32;
    while y < h - 1 {
        let mut x = 1u32;
        while x < w - 1 {
            // Luma of 3×3 neighbourhood corners for gradient approximation
            let luma = |px: u32, py: u32| -> f32 {
                let p = rgb.get_pixel(px, py);
                (p[0] as f32 * 0.299 + p[1] as f32 * 0.587 + p[2] as f32 * 0.114) / 255.0
            };
            let gx = luma(x + 1, y) - luma(x - 1, y);
            let gy = luma(x, y + 1) - luma(x, y - 1);
            let mag = (gx * gx + gy * gy).sqrt();
            if mag > 0.12 {
                edge_count += 1;
            }
            total += 1;
            x += stride;
        }
        y += stride;
    }
    if total == 0 {
        0.0
    } else {
        edge_count as f32 / total as f32
    }
}

/// Fraction of sampled pixels whose 3×3 neighbourhood has low color variance
/// (i.e. flat-fill areas typical of illustrations).
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
            // Sample 5-point cross neighbourhood
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

/// Map `value` from range [lo, hi] to a confidence-like score in [0, 1].
fn score(value: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return 0.5;
    }
    ((value - lo) / (hi - lo)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_saturation_gray() {
        assert!(rgb_saturation(0.5, 0.5, 0.5) < 1e-4);
    }

    #[test]
    fn test_rgb_saturation_red() {
        assert!(rgb_saturation(1.0, 0.0, 0.0) > 0.99);
    }

    #[test]
    fn test_skin_tone() {
        assert!(is_skin_tone(210, 140, 100));
        assert!(!is_skin_tone(10, 10, 10)); // black
        assert!(!is_skin_tone(200, 200, 200)); // gray
    }
}
