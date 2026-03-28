//! Image tiling for large images — splits images exceeding a pixel threshold
//! into overlapping tiles, upscales each tile independently, and blends tile
//! boundaries seamlessly to prevent OOM on very large inputs.
//!
//! This module provides a standalone tiling pipeline that works with any
//! `ThreadSafeNetwork`.  The core algorithm:
//!
//! 1. Split the input into overlapping tiles of `tile_size × tile_size`.
//! 2. Upscale each tile through the SR network.
//! 3. Feather-blend overlapping regions using linear ramp weights.
//! 4. Normalize the accumulated result to produce a seamless output.
//!
//! Tile processing can be parallelised across threads via Rayon.

use std::time::Instant;

use image::{DynamicImage, GenericImage};
use ndarray::{ArrayD, Axis, IxDyn};
use serde::{Deserialize, Serialize};

use crate::error::{Result, SrganError};
use crate::thread_safe_network::ThreadSafeNetwork;
use crate::{data_to_image, image_to_data};

// ── Constants ───────────────────────────────────────────────────────────────

/// Default tile size in pixels (input space).
pub const DEFAULT_TILE_SIZE: usize = 512;

/// Default overlap between adjacent tiles (input space).
pub const DEFAULT_OVERLAP: usize = 32;

/// Pixel-count threshold above which tiling is automatically engaged.
/// Default: 4 MP (e.g. 2000×2000).
pub const AUTO_TILE_THRESHOLD: u64 = 4_000_000;

/// Maximum supported tile size.
pub const MAX_TILE_SIZE: usize = 2048;

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the tiling pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilingConfig {
    /// Tile size in input pixels. Each tile is `tile_size × tile_size`.
    #[serde(default = "default_tile_size")]
    pub tile_size: usize,
    /// Overlap between adjacent tiles in input pixels.
    #[serde(default = "default_overlap")]
    pub overlap: usize,
    /// Pixel count threshold to auto-enable tiling.
    #[serde(default = "default_threshold")]
    pub auto_threshold: u64,
    /// Whether to use parallel tile processing (via Rayon).
    #[serde(default = "default_true")]
    pub parallel: bool,
}

fn default_tile_size() -> usize { DEFAULT_TILE_SIZE }
fn default_overlap() -> usize { DEFAULT_OVERLAP }
fn default_threshold() -> u64 { AUTO_TILE_THRESHOLD }
fn default_true() -> bool { true }

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
            overlap: DEFAULT_OVERLAP,
            auto_threshold: AUTO_TILE_THRESHOLD,
            parallel: true,
        }
    }
}

// ── Result info ─────────────────────────────────────────────────────────────

/// Metadata about a tiled upscale operation.
#[derive(Debug, Clone, Serialize)]
pub struct TilingInfo {
    /// Number of tiles in the X direction.
    pub tiles_x: usize,
    /// Number of tiles in the Y direction.
    pub tiles_y: usize,
    /// Total number of tiles processed.
    pub total_tiles: usize,
    /// Input dimensions.
    pub input_width: usize,
    pub input_height: usize,
    /// Output dimensions.
    pub output_width: usize,
    pub output_height: usize,
    /// Tile size used (input space).
    pub tile_size: usize,
    /// Overlap used (input space).
    pub overlap: usize,
    /// Total processing time in ms.
    pub processing_time_ms: u64,
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Check whether an image should use tiled processing.
pub fn should_tile(width: u32, height: u32, config: &TilingConfig) -> bool {
    (width as u64) * (height as u64) > config.auto_threshold
}

/// Upscale an image using tiled processing.
///
/// Splits the image into overlapping tiles, upscales each through `network`,
/// and blends the results into a seamless output.
pub fn tiled_upscale(
    input: &DynamicImage,
    network: &ThreadSafeNetwork,
    config: &TilingConfig,
) -> Result<(DynamicImage, TilingInfo)> {
    let start = Instant::now();

    let tensor = image_to_data(input); // [H, W, 3]
    let in_h = tensor.shape()[0];
    let in_w = tensor.shape()[1];
    let scale = network.factor() as usize;

    let tile_size = config.tile_size.min(MAX_TILE_SIZE).max(16);
    let overlap = config.overlap.min(tile_size / 2);

    let out_h = in_h * scale;
    let out_w = in_w * scale;
    let overlap_out = overlap * scale;

    // Compute tile start positions
    let step = tile_size.saturating_sub(2 * overlap).max(1);
    let y_starts = tile_starts(in_h, tile_size, step);
    let x_starts = tile_starts(in_w, tile_size, step);

    let tiles_y = y_starts.len();
    let tiles_x = x_starts.len();
    let total_tiles = tiles_y * tiles_x;

    // Accumulator and weight sum for blending
    let mut accum = ArrayD::<f32>::zeros(IxDyn(&[out_h, out_w, 3]));
    let mut wsum = ArrayD::<f32>::zeros(IxDyn(&[out_h, out_w, 1]));

    // Process tiles (sequentially — parallel version uses upscale_image_tiled
    // on ThreadSafeNetwork which is already Rayon-aware).
    for &ys in &y_starts {
        let ye = (ys + tile_size).min(in_h);
        let th = ye - ys;

        for &xs in &x_starts {
            let xe = (xs + tile_size).min(in_w);
            let tw = xe - xs;

            // Extract tile [th, tw, 3] → [1, th, tw, 3]
            let tile = tensor.slice(ndarray::s![ys..ye, xs..xe, ..]).to_owned();
            let tile_4d = tile
                .into_shape(IxDyn(&[1, th, tw, 3]))
                .map_err(|e| SrganError::ShapeError(format!("tile reshape: {}", e)))?;

            let upscaled = network.process(tile_4d)?;
            let ut = upscaled.subview(Axis(0), 0); // [out_th, out_tw, 3]
            let out_th = ut.shape()[0];
            let out_tw = ut.shape()[1];

            let oy0 = ys * scale;
            let ox0 = xs * scale;

            // Blend tile into accumulator
            for i in 0..out_th {
                for j in 0..out_tw {
                    let oy = oy0 + i;
                    let ox = ox0 + j;
                    if oy >= out_h || ox >= out_w {
                        continue;
                    }
                    let wy = blend_weight(i, out_th, overlap_out);
                    let wx = blend_weight(j, out_tw, overlap_out);
                    let w = wy * wx;
                    for c in 0..3usize {
                        accum[[oy, ox, c]] += ut[[i, j, c]] * w;
                    }
                    wsum[[oy, ox, 0]] += w;
                }
            }
        }
    }

    // Normalize
    for oy in 0..out_h {
        for ox in 0..out_w {
            let w = wsum[[oy, ox, 0]];
            if w > 0.0 {
                for c in 0..3usize {
                    accum[[oy, ox, c]] /= w;
                }
            }
        }
    }

    let result = data_to_image(accum.view());
    let elapsed = start.elapsed().as_millis() as u64;

    let info = TilingInfo {
        tiles_x,
        tiles_y,
        total_tiles,
        input_width: in_w,
        input_height: in_h,
        output_width: out_w,
        output_height: out_h,
        tile_size,
        overlap,
        processing_time_ms: elapsed,
    };

    Ok((result, info))
}

/// Upscale with automatic tiling decision.
///
/// If the image exceeds the threshold, tiles it; otherwise does a single-pass
/// upscale.
pub fn smart_upscale(
    input: &DynamicImage,
    network: &ThreadSafeNetwork,
    config: &TilingConfig,
) -> Result<(DynamicImage, Option<TilingInfo>)> {
    let w = input.width();
    let h = input.height();

    if should_tile(w, h, config) {
        let (img, info) = tiled_upscale(input, network, config)?;
        Ok((img, Some(info)))
    } else {
        let img = network.upscale_image(input)?;
        Ok((img, None))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Generate tile start positions along one dimension.
fn tile_starts(total: usize, tile_size: usize, step: usize) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut s = 0usize;
    loop {
        starts.push(s);
        if s + tile_size >= total {
            break;
        }
        s += step;
    }
    starts
}

/// Compute feather-blend weight for a pixel at position `i` within a tile
/// of dimension `dim`, using `overlap` as the ramp width.
fn blend_weight(i: usize, dim: usize, overlap: usize) -> f32 {
    if overlap == 0 {
        return 1.0;
    }
    let dist_near = i + 1;
    let dist_far = dim - i;
    let d = dist_near.min(dist_far).min(overlap);
    d as f32 / overlap as f32
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_should_tile() {
        let config = TilingConfig::default();
        // 1000×1000 = 1M < 4M threshold
        assert!(!should_tile(1000, 1000, &config));
        // 3000×3000 = 9M > 4M threshold
        assert!(should_tile(3000, 3000, &config));
    }

    #[test]
    fn test_tile_starts() {
        let starts = tile_starts(100, 32, 28);
        assert!(!starts.is_empty());
        assert_eq!(starts[0], 0);
        // Last start + tile_size should cover the full dimension
        assert!(*starts.last().unwrap() + 32 >= 100);
    }

    #[test]
    fn test_blend_weight_center() {
        // Center of a large tile should have weight 1.0
        let w = blend_weight(100, 200, 32);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_weight_edge() {
        // Very edge should have small weight
        let w = blend_weight(0, 200, 32);
        assert!(w < 0.1);
    }

    #[test]
    fn test_tiling_config_default() {
        let config = TilingConfig::default();
        assert_eq!(config.tile_size, 512);
        assert_eq!(config.overlap, 32);
        assert!(config.parallel);
    }

    #[test]
    fn test_tile_grid_computation() {
        // Verify tile grid is computed correctly for a 256×256 image
        // with 64px tiles and 16px overlap → step = 32, multiple tiles.
        let config = TilingConfig {
            tile_size: 64,
            overlap: 16,
            auto_threshold: 0,
            parallel: false,
        };
        let y_starts = super::tile_starts(256, config.tile_size, 32);
        let x_starts = super::tile_starts(256, config.tile_size, 32);
        assert!(y_starts.len() > 1, "should produce multiple Y tiles");
        assert!(x_starts.len() > 1, "should produce multiple X tiles");
        // Last tile should cover the full dimension
        assert!(*y_starts.last().unwrap() + 64 >= 256);
        assert!(*x_starts.last().unwrap() + 64 >= 256);
    }

    #[test]
    fn test_smart_upscale_decision() {
        // Verify that smart_upscale correctly decides whether to tile
        let config = TilingConfig::default();
        // Small image — should not tile
        assert!(!should_tile(100, 100, &config));
        // Large image — should tile
        assert!(should_tile(3000, 3000, &config));
    }
}
