//! Model ensemble inference — runs an image through multiple models (e.g.
//! SRGAN + Real-ESRGAN + waifu2x), then combines the outputs by either
//! averaging or selecting the best result by SSIM.
//!
//! Exposed via `POST /api/v1/upscale/ensemble`.

use std::time::Instant;

use image::DynamicImage;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::error::{Result, SrganError};
use crate::ssim::ssim_calculation;
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::{data_to_image, image_to_data};

// ── Configuration ───────────────────────────────────────────────────────────

/// Strategy for combining ensemble outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleStrategy {
    /// Pixel-wise average of all model outputs.
    Average,
    /// Select the single output with the highest SSIM vs. the average.
    BestSsim,
    /// Weighted average using SSIM scores as weights.
    WeightedSsim,
}

impl Default for EnsembleStrategy {
    fn default() -> Self {
        EnsembleStrategy::Average
    }
}

/// Default model set for ensemble.
pub const DEFAULT_ENSEMBLE_MODELS: &[&str] = &["natural", "anime"];

// ── Request / Response ──────────────────────────────────────────────────────

/// Request body for `POST /api/v1/upscale/ensemble`.
#[derive(Debug, Deserialize)]
pub struct EnsembleRequest {
    /// Base64-encoded input image.
    pub image_data: String,
    /// Models to use (2–3). Defaults to `["natural", "anime"]`.
    #[serde(default)]
    pub models: Vec<String>,
    /// Combination strategy. Defaults to `average`.
    #[serde(default)]
    pub strategy: EnsembleStrategy,
    /// Optional output format ("png", "jpeg").
    #[serde(default)]
    pub format: Option<String>,
}

/// Response for ensemble upscaling.
#[derive(Debug, Serialize)]
pub struct EnsembleResponse {
    pub job_id: String,
    pub status: String,
    /// Which models were used.
    pub models_used: Vec<String>,
    /// Strategy applied.
    pub strategy: EnsembleStrategy,
    /// Per-model SSIM scores (vs. ensemble average).
    pub model_scores: Vec<ModelScore>,
    /// Total processing time in ms.
    pub processing_time_ms: u64,
}

/// SSIM score for a single model's output.
#[derive(Debug, Clone, Serialize)]
pub struct ModelScore {
    pub model: String,
    pub ssim: f32,
    pub processing_time_ms: u64,
}

// ── Engine ──────────────────────────────────────────────────────────────────

/// Run ensemble inference on a single image.
///
/// 1. Upscales the image through each model in `models`.
/// 2. Combines outputs according to `strategy`.
/// 3. Returns the combined image + per-model SSIM scores.
pub fn ensemble_upscale(
    input: &DynamicImage,
    models: &[String],
    strategy: EnsembleStrategy,
) -> Result<(DynamicImage, Vec<ModelScore>)> {
    if models.len() < 2 {
        return Err(SrganError::InvalidInput(
            "Ensemble requires at least 2 models".into(),
        ));
    }
    if models.len() > 5 {
        return Err(SrganError::InvalidInput(
            "Ensemble supports at most 5 models".into(),
        ));
    }

    // ── 1. Upscale through each model ───────────────────────────────────────
    let mut outputs: Vec<(String, ArrayD<f32>, u64)> = Vec::with_capacity(models.len());

    for label in models {
        let start = Instant::now();
        let network = ThreadSafeNetwork::from_label(label, None)?;
        let upscaled = network.upscale_image(input)?;
        let elapsed = start.elapsed().as_millis() as u64;

        let tensor = image_to_data(&upscaled);
        outputs.push((label.clone(), tensor, elapsed));
    }

    // Ensure all outputs have the same spatial dimensions (trim to min).
    let min_h = outputs.iter().map(|(_, t, _)| t.shape()[0]).min().unwrap();
    let min_w = outputs.iter().map(|(_, t, _)| t.shape()[1]).min().unwrap();

    // ── 2. Compute pixel-wise average ───────────────────────────────────────
    let n = outputs.len() as f32;
    let mut average = ArrayD::<f32>::zeros(IxDyn(&[min_h, min_w, 3]));
    for (_, tensor, _) in &outputs {
        for y in 0..min_h {
            for x in 0..min_w {
                for c in 0..3usize {
                    average[[y, x, c]] += tensor[[y, x, c]] / n;
                }
            }
        }
    }

    // ── 3. Compute per-model SSIM scores vs. average ────────────────────────
    let mut scores: Vec<ModelScore> = Vec::with_capacity(outputs.len());
    for (label, tensor, time_ms) in &outputs {
        // Build a cropped ArrayD for SSIM comparison
        let mut cropped = ArrayD::<f32>::zeros(IxDyn(&[min_h, min_w, 3]));
        for y in 0..min_h {
            for x in 0..min_w {
                for c in 0..3usize {
                    cropped[[y, x, c]] = tensor[[y, x, c]];
                }
            }
        }
        let ssim = ssim_calculation(cropped.view(), average.view());
        scores.push(ModelScore {
            model: label.clone(),
            ssim,
            processing_time_ms: *time_ms,
        });
    }

    // ── 4. Combine according to strategy ────────────────────────────────────
    let combined = match strategy {
        EnsembleStrategy::Average => average.clone(),

        EnsembleStrategy::BestSsim => {
            let best_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.ssim.partial_cmp(&b.ssim).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let (_, tensor, _) = &outputs[best_idx];
            let mut result = ArrayD::<f32>::zeros(IxDyn(&[min_h, min_w, 3]));
            for y in 0..min_h {
                for x in 0..min_w {
                    for c in 0..3usize {
                        result[[y, x, c]] = tensor[[y, x, c]];
                    }
                }
            }
            result
        }

        EnsembleStrategy::WeightedSsim => {
            let total_ssim: f32 = scores.iter().map(|s| s.ssim.max(0.0)).sum();
            if total_ssim < 1e-6 {
                average.clone()
            } else {
                let mut weighted = ArrayD::<f32>::zeros(IxDyn(&[min_h, min_w, 3]));
                for (i, (_, tensor, _)) in outputs.iter().enumerate() {
                    let w = scores[i].ssim.max(0.0) / total_ssim;
                    for y in 0..min_h {
                        for x in 0..min_w {
                            for c in 0..3usize {
                                weighted[[y, x, c]] += tensor[[y, x, c]] * w;
                            }
                        }
                    }
                }
                weighted
            }
        }
    };

    let result_image = data_to_image(combined.view());
    Ok((result_image, scores))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn tiny_image() -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::new(32, 32))
    }

    #[test]
    fn test_ensemble_too_few_models() {
        let img = tiny_image();
        let models = vec!["natural".to_string()];
        let result = ensemble_upscale(&img, &models, EnsembleStrategy::Average);
        assert!(result.is_err());
    }

    #[test]
    fn test_ensemble_too_many_models() {
        let img = tiny_image();
        let models: Vec<String> = (0..6).map(|i| format!("model-{}", i)).collect();
        let result = ensemble_upscale(&img, &models, EnsembleStrategy::Average);
        assert!(result.is_err());
    }

    #[test]
    fn test_ensemble_strategy_default() {
        assert_eq!(EnsembleStrategy::default(), EnsembleStrategy::Average);
    }

    #[test]
    fn test_default_ensemble_models() {
        assert!(DEFAULT_ENSEMBLE_MODELS.len() >= 2);
        assert!(DEFAULT_ENSEMBLE_MODELS.contains(&"natural"));
        assert!(DEFAULT_ENSEMBLE_MODELS.contains(&"anime"));
    }
}
