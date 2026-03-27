//! Unified image quality metrics: PSNR and SSIM.
//!
//! Provides a single entry point for computing quality scores between a
//! reference image and an upscaled image, suitable for the admin dashboard
//! and the benchmark binary.

use ndarray::ArrayViewD;
use serde::Serialize;

use crate::psnr::psnr_calculation;
use crate::ssim::ssim_calculation;

/// Combined quality score for an image pair.
#[derive(Debug, Clone, Serialize)]
pub struct QualityScore {
    /// Peak Signal-to-Noise Ratio in dB (higher is better).
    pub psnr_db: f32,
    /// Luma PSNR in dB (Y-channel only).
    pub psnr_luma_db: f32,
    /// Structural Similarity Index in [-1, 1] (higher is better).
    pub ssim: f32,
}

impl std::fmt::Display for QualityScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PSNR: {:.2} dB | Luma PSNR: {:.2} dB | SSIM: {:.4}",
            self.psnr_db, self.psnr_luma_db, self.ssim
        )
    }
}

/// Compute PSNR (RGB + luma) and SSIM between two image tensors.
///
/// Both images should be `[H, W, 3]` or `[1, H, W, 3]` with values in `[0, 1]`.
pub fn compute_quality(
    reference: ArrayViewD<f32>,
    upscaled: ArrayViewD<f32>,
) -> QualityScore {
    let (rgb_err, luma_err, pixel_count) = psnr_calculation(reference.view(), upscaled.view());

    let psnr_db = if pixel_count > 0.0 && rgb_err > 0.0 {
        -10.0 * (rgb_err / pixel_count).log10()
    } else if pixel_count > 0.0 {
        f32::INFINITY
    } else {
        0.0
    };

    let psnr_luma_db = if pixel_count > 0.0 && luma_err > 0.0 {
        -10.0 * (luma_err / pixel_count).log10()
    } else if pixel_count > 0.0 {
        f32::INFINITY
    } else {
        0.0
    };

    let ssim = ssim_calculation(reference, upscaled);

    QualityScore {
        psnr_db,
        psnr_luma_db,
        ssim,
    }
}

/// Compute quality metrics from raw image byte buffers (8-bit RGB).
///
/// Images are converted to f32 tensors in [0, 1] range before comparison.
/// Both images must have the same dimensions.
pub fn compute_quality_from_rgb(
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    upsample_data: &[u8],
    ups_width: usize,
    ups_height: usize,
) -> QualityScore {
    let ref_tensor = rgb_bytes_to_tensor(ref_data, ref_width, ref_height);
    let ups_tensor = rgb_bytes_to_tensor(upsample_data, ups_width, ups_height);
    compute_quality(ref_tensor.view(), ups_tensor.view())
}

/// Convert raw 8-bit RGB bytes to [H, W, 3] f32 tensor.
fn rgb_bytes_to_tensor(
    data: &[u8],
    width: usize,
    height: usize,
) -> ndarray::ArrayD<f32> {
    let mut arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[height, width, 3]));
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                arr[[y, x, 0]] = data[idx] as f32 / 255.0;
                arr[[y, x, 1]] = data[idx + 1] as f32 / 255.0;
                arr[[y, x, 2]] = data[idx + 2] as f32 / 255.0;
            }
        }
    }
    arr
}

/// Format a quality score as an HTML snippet for the admin dashboard.
pub fn quality_score_html(score: &QualityScore) -> String {
    let ssim_class = if score.ssim > 0.95 {
        "excellent"
    } else if score.ssim > 0.85 {
        "good"
    } else {
        "poor"
    };

    format!(
        r#"<div class="quality-metrics">
  <span class="metric">PSNR: <strong>{:.2} dB</strong></span>
  <span class="metric">Luma PSNR: <strong>{:.2} dB</strong></span>
  <span class="metric ssim-{}">SSIM: <strong>{:.4}</strong></span>
</div>"#,
        score.psnr_db, score.psnr_luma_db, ssim_class, score.ssim
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_quality_identical_images() {
        let img = ArrayD::<f32>::from_elem(vec![8, 8, 3], 0.5);
        let score = compute_quality(img.view(), img.view());
        assert!(score.psnr_db.is_infinite(), "PSNR should be inf for identical images");
        assert!((score.ssim - 1.0).abs() < 1e-4, "SSIM should be ~1.0");
    }

    #[test]
    fn test_quality_different_images() {
        let img1 = ArrayD::<f32>::zeros(vec![8, 8, 3]);
        let img2 = ArrayD::<f32>::from_elem(vec![8, 8, 3], 1.0);
        let score = compute_quality(img1.view(), img2.view());
        assert!(score.psnr_db > 0.0 && score.psnr_db < 10.0);
        assert!(score.ssim < 0.5);
    }

    #[test]
    fn test_quality_from_rgb_bytes() {
        let white = vec![255u8; 4 * 4 * 3];
        let black = vec![0u8; 4 * 4 * 3];
        let score = compute_quality_from_rgb(&white, 4, 4, &black, 4, 4);
        assert!(score.psnr_db > 0.0);
        assert!(score.ssim < 0.5);
    }

    #[test]
    fn test_quality_display() {
        let score = QualityScore {
            psnr_db: 35.5,
            psnr_luma_db: 37.2,
            ssim: 0.9801,
        };
        let s = format!("{}", score);
        assert!(s.contains("35.50"));
        assert!(s.contains("0.9801"));
    }

    #[test]
    fn test_quality_score_html() {
        let score = QualityScore {
            psnr_db: 35.5,
            psnr_luma_db: 37.2,
            ssim: 0.9801,
        };
        let html = quality_score_html(&score);
        assert!(html.contains("excellent"));
        assert!(html.contains("35.50"));
    }
}
