//! Content-aware region analysis and per-region upscaling.
//!
//! Splits an image into a grid of tiles, classifies each tile by content type
//! (face/skin, text/edge, flat-color, photorealistic), and applies the best
//! upscaling model per region.  The results are blended back into a single
//! output image.
//!
//! Exposed as `--auto-enhance` in the CLI and `auto_enhance: true` in the API.

use image::{DynamicImage, GenericImage, RgbImage};
use log::info;

use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;

// ── Region types ────────────────────────────────────────────────────────────

/// Content class detected for a single tile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegionKind {
    /// High edge density — text, line art, or fine detail.
    TextEdge,
    /// Skin-tone dominant — faces or skin regions.
    SkinTone,
    /// Low variance, high saturation — flat fills typical of anime/illustration.
    FlatColor,
    /// Everything else — natural photograph content.
    Photo,
}

impl RegionKind {
    /// Model label best suited for this region kind.
    pub fn model_label(&self) -> &'static str {
        match self {
            RegionKind::FlatColor => "anime",
            RegionKind::Photo | RegionKind::SkinTone => "natural",
            RegionKind::TextEdge => "natural",
        }
    }

    /// Whether this region should receive an extra sharpening pass after upscaling.
    pub fn needs_sharpening(&self) -> bool {
        matches!(self, RegionKind::TextEdge)
    }
}

// ── Tile analysis ───────────────────────────────────────────────────────────

/// Size of analysis grid tiles (in pixels). Smaller = finer region boundaries
/// but more overhead from model switching.
const ANALYSIS_TILE: usize = 64;

/// Sobel edge-density threshold above which a tile is classified as text/edge.
const EDGE_DENSITY_THRESHOLD: f32 = 0.25;

/// Fraction of pixels that must be skin-toned to classify the tile as skin.
const SKIN_FRACTION_THRESHOLD: f32 = 0.35;

/// Flat-region luma-variance ceiling (same metric as detection.rs).
const FLAT_VARIANCE_CEILING: f32 = 0.003;

/// Fraction of pixels with low local variance needed to call the tile flat-color.
const FLAT_FRACTION_THRESHOLD: f32 = 0.55;

/// Analyse a single tile and return its [`RegionKind`].
pub fn classify_tile(rgb: &RgbImage, x0: u32, y0: u32, tw: u32, th: u32) -> RegionKind {
    let edge_density = sobel_edge_density(rgb, x0, y0, tw, th);
    if edge_density > EDGE_DENSITY_THRESHOLD {
        return RegionKind::TextEdge;
    }

    let skin_frac = skin_tone_fraction(rgb, x0, y0, tw, th);
    if skin_frac > SKIN_FRACTION_THRESHOLD {
        return RegionKind::SkinTone;
    }

    let flat_frac = flat_fraction(rgb, x0, y0, tw, th);
    if flat_frac > FLAT_FRACTION_THRESHOLD {
        return RegionKind::FlatColor;
    }

    RegionKind::Photo
}

// ── Sobel edge density ──────────────────────────────────────────────────────

/// Compute the fraction of pixels in the tile whose Sobel gradient magnitude
/// exceeds a threshold.  This is a fast proxy for text / line-art content.
fn sobel_edge_density(rgb: &RgbImage, x0: u32, y0: u32, tw: u32, th: u32) -> f32 {
    if tw < 3 || th < 3 {
        return 0.0;
    }
    let grad_threshold: f32 = 0.15; // normalised gradient magnitude
    let mut edge_count = 0u32;
    let mut total = 0u32;

    // Sample every other pixel for speed.
    let mut y = y0 + 1;
    while y < y0 + th - 1 {
        let mut x = x0 + 1;
        while x < x0 + tw - 1 {
            let g = sobel_magnitude(rgb, x, y);
            if g > grad_threshold {
                edge_count += 1;
            }
            total += 1;
            x += 2;
        }
        y += 2;
    }
    if total == 0 {
        0.0
    } else {
        edge_count as f32 / total as f32
    }
}

/// 3×3 Sobel gradient magnitude on luma, normalised to [0, 1].
fn sobel_magnitude(rgb: &RgbImage, x: u32, y: u32) -> f32 {
    let l = |dx: i32, dy: i32| -> f32 {
        let px = rgb.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32);
        (px[0] as f32 * 0.299 + px[1] as f32 * 0.587 + px[2] as f32 * 0.114) / 255.0
    };

    let gx = -l(-1, -1) - 2.0 * l(-1, 0) - l(-1, 1) + l(1, -1) + 2.0 * l(1, 0) + l(1, 1);
    let gy = -l(-1, -1) - 2.0 * l(0, -1) - l(1, -1) + l(-1, 1) + 2.0 * l(0, 1) + l(1, 1);
    (gx * gx + gy * gy).sqrt().min(1.0)
}

// ── Skin-tone detection ─────────────────────────────────────────────────────

/// Fraction of sampled pixels whose RGB values fall in a skin-tone range.
///
/// Uses the simple rule from Peer et al. (2003):
///   R > 95, G > 40, B > 20,
///   max(R,G,B) - min(R,G,B) > 15,
///   |R - G| > 15, R > G, R > B.
fn skin_tone_fraction(rgb: &RgbImage, x0: u32, y0: u32, tw: u32, th: u32) -> f32 {
    let mut skin = 0u32;
    let mut total = 0u32;
    let step = 2u32;

    let mut y = y0;
    while y < y0 + th {
        let mut x = x0;
        while x < x0 + tw {
            let p = rgb.get_pixel(x, y);
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            if r > 95
                && g > 40
                && b > 20
                && (mx - mn) > 15
                && (r - g).abs() > 15
                && r > g
                && r > b
            {
                skin += 1;
            }
            total += 1;
            x += step;
        }
        y += step;
    }
    if total == 0 {
        0.0
    } else {
        skin as f32 / total as f32
    }
}

// ── Flat-region fraction ────────────────────────────────────────────────────

/// Fraction of sampled pixels with very low local luma variance (5-point cross).
fn flat_fraction(rgb: &RgbImage, x0: u32, y0: u32, tw: u32, th: u32) -> f32 {
    if tw < 3 || th < 3 {
        return 0.0;
    }
    let mut flat = 0u32;
    let mut total = 0u32;
    let step = 2u32;

    let mut y = y0 + 1;
    while y < y0 + th - 1 {
        let mut x = x0 + 1;
        while x < x0 + tw - 1 {
            let luma_at = |dx: i32, dy: i32| -> f32 {
                let px = rgb.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32);
                (px[0] as f32 * 0.299 + px[1] as f32 * 0.587 + px[2] as f32 * 0.114) / 255.0
            };
            let lumas = [
                luma_at(0, 0),
                luma_at(-1, 0),
                luma_at(1, 0),
                luma_at(0, -1),
                luma_at(0, 1),
            ];
            let mean = lumas.iter().sum::<f32>() / 5.0;
            let var = lumas.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / 5.0;
            if var < FLAT_VARIANCE_CEILING {
                flat += 1;
            }
            total += 1;
            x += step;
        }
        y += step;
    }
    if total == 0 {
        0.0
    } else {
        flat as f32 / total as f32
    }
}

// ── Unsharp-mask sharpening ─────────────────────────────────────────────────

/// Apply unsharp-mask sharpening to an image region.
/// `amount` controls the strength (1.0 = moderate, 2.0 = strong).
pub fn sharpen_image(img: &DynamicImage, amount: f32) -> DynamicImage {
    let rgb = img.to_rgb();
    let (w, h) = (rgb.width(), rgb.height());
    let mut out = rgb.clone();

    if w < 3 || h < 3 {
        return DynamicImage::ImageRgb8(out);
    }

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            for c in 0..3usize {
                let center = rgb.get_pixel(x, y)[c] as f32;
                // Simple 3×3 box-blur approximation for the unsharp mask.
                let blur = (rgb.get_pixel(x - 1, y)[c] as f32
                    + rgb.get_pixel(x + 1, y)[c] as f32
                    + rgb.get_pixel(x, y - 1)[c] as f32
                    + rgb.get_pixel(x, y + 1)[c] as f32)
                    / 4.0;
                let sharpened = center + amount * (center - blur);
                out.get_pixel_mut(x, y)[c] = sharpened.max(0.0).min(255.0) as u8;
            }
        }
    }
    DynamicImage::ImageRgb8(out)
}

// ── Content-aware upscaling orchestrator ────────────────────────────────────

/// Build a region map: for each tile in a grid, classify its content.
pub fn build_region_map(img: &DynamicImage) -> Vec<Vec<RegionKind>> {
    let rgb = img.to_rgb();
    let (w, h) = (rgb.width(), rgb.height());
    let tile = ANALYSIS_TILE as u32;

    let cols = ((w + tile - 1) / tile) as usize;
    let rows = ((h + tile - 1) / tile) as usize;

    let mut map = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut row_vec = Vec::with_capacity(cols);
        let y0 = row as u32 * tile;
        let th = tile.min(h - y0);
        for col in 0..cols {
            let x0 = col as u32 * tile;
            let tw = tile.min(w - x0);
            row_vec.push(classify_tile(&rgb, x0, y0, tw, th));
        }
        map.push(row_vec);
    }
    map
}

/// Log a summary of the region map.
fn log_region_summary(map: &[Vec<RegionKind>]) {
    let (mut text, mut skin, mut flat, mut photo) = (0u32, 0u32, 0u32, 0u32);
    for row in map {
        for kind in row {
            match kind {
                RegionKind::TextEdge => text += 1,
                RegionKind::SkinTone => skin += 1,
                RegionKind::FlatColor => flat += 1,
                RegionKind::Photo => photo += 1,
            }
        }
    }
    let total = text + skin + flat + photo;
    info!(
        "Region map: {} tiles — text/edge: {}, skin: {}, flat-color: {}, photo: {}",
        total, text, skin, flat, photo
    );
}

/// Determine if the region map is homogeneous (all tiles have the same kind).
/// Returns `Some(kind)` when uniform, `None` when mixed.
fn uniform_region(map: &[Vec<RegionKind>]) -> Option<RegionKind> {
    let first = map.first()?.first()?;
    for row in map {
        for kind in row {
            if kind != first {
                return None;
            }
        }
    }
    Some(*first)
}

/// Run content-aware upscaling on `img`.
///
/// 1. Build a region map by classifying tiles.
/// 2. If all regions are the same type, upscale the whole image with that model.
/// 3. Otherwise, upscale the whole image once with the "natural" model and once
///    with "anime", then blend per-tile based on the region map.  Text/edge
///    regions also receive a sharpening pass.
///
/// The `factor` parameter is used to construct the networks.
pub fn auto_enhance_upscale(img: &DynamicImage, factor: usize) -> Result<DynamicImage> {
    let region_map = build_region_map(img);
    log_region_summary(&region_map);

    // Fast path: homogeneous content.
    if let Some(kind) = uniform_region(&region_map) {
        info!("Uniform region type: {:?} — using single model '{}'", kind, kind.model_label());
        let net = UpscalingNetwork::from_label(kind.model_label(), Some(factor))
            .map_err(|e| SrganError::Network(e))?;
        let result = net.upscale_image(img)
            .map_err(|e| SrganError::GraphExecution(e.to_string()))?;
        if kind.needs_sharpening() {
            return Ok(sharpen_image(&result, 1.5));
        }
        return Ok(result);
    }

    // Mixed content: upscale with both models and blend per-tile.
    info!("Mixed content detected — upscaling with both 'natural' and 'anime' models");

    let net_photo = UpscalingNetwork::from_label("natural", Some(factor))
        .map_err(|e| SrganError::Network(e))?;
    let net_anime = UpscalingNetwork::from_label("anime", Some(factor))
        .map_err(|e| SrganError::Network(e))?;

    let upscaled_photo = net_photo
        .upscale_image(img)
        .map_err(|e| SrganError::GraphExecution(e.to_string()))?;
    let upscaled_anime = net_anime
        .upscale_image(img)
        .map_err(|e| SrganError::GraphExecution(e.to_string()))?;

    // Sharpen a copy of the photo result for text/edge regions.
    let upscaled_sharp = sharpen_image(&upscaled_photo, 1.5);

    // Blend based on region map.
    let (out_w, out_h) = (upscaled_photo.width(), upscaled_photo.height());
    let scale = factor as u32;
    let tile_out = ANALYSIS_TILE as u32 * scale;

    let rgb_photo = upscaled_photo.to_rgb();
    let rgb_anime = upscaled_anime.to_rgb();
    let rgb_sharp = upscaled_sharp.to_rgb();

    let rows = region_map.len();
    let cols = region_map[0].len();

    let mut output = RgbImage::new(out_w, out_h);

    for row in 0..rows {
        for col in 0..cols {
            let kind = region_map[row][col];
            let src = match kind {
                RegionKind::FlatColor => &rgb_anime,
                RegionKind::TextEdge => &rgb_sharp,
                RegionKind::Photo | RegionKind::SkinTone => &rgb_photo,
            };

            let oy0 = row as u32 * tile_out;
            let ox0 = col as u32 * tile_out;
            let oy_end = (oy0 + tile_out).min(out_h);
            let ox_end = (ox0 + tile_out).min(out_w);

            for y in oy0..oy_end {
                for x in ox0..ox_end {
                    output.put_pixel(x, y, *src.get_pixel(x, y));
                }
            }
        }
    }

    info!("Content-aware blending complete ({}×{} output)", out_w, out_h);
    Ok(DynamicImage::ImageRgb8(output))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn test_classify_flat_green_tile() {
        let img = ImageBuffer::from_pixel(16, 16, Rgb([0u8, 200, 50]));
        let kind = classify_tile(&img, 0, 0, 16, 16);
        assert_eq!(kind, RegionKind::FlatColor);
    }

    #[test]
    fn test_classify_high_edge_tile() {
        // Checkerboard pattern → high edge density.
        let mut img = ImageBuffer::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let kind = classify_tile(&img, 0, 0, 32, 32);
        assert_eq!(kind, RegionKind::TextEdge);
    }

    #[test]
    fn test_sharpen_does_not_panic() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([128, 128, 128])));
        let _ = sharpen_image(&img, 1.0);
    }

    #[test]
    fn test_region_map_dimensions() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::new(200, 150));
        let map = build_region_map(&img);
        // 200/64 = 4 cols (ceil), 150/64 = 3 rows (ceil)
        assert_eq!(map.len(), 3);
        assert_eq!(map[0].len(), 4);
    }

    #[test]
    fn test_uniform_region_detects_homogeneous() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(64, 64, Rgb([128, 128, 128])));
        let map = build_region_map(&img);
        assert!(uniform_region(&map).is_some());
    }

    #[test]
    fn test_skin_tone_detection() {
        // A tile full of skin-toned pixels.
        let img = ImageBuffer::from_pixel(16, 16, Rgb([200u8, 150, 120]));
        let frac = skin_tone_fraction(&img, 0, 0, 16, 16);
        assert!(frac > 0.5, "Expected skin fraction > 0.5, got {}", frac);
    }
}
