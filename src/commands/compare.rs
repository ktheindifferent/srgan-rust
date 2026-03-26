//! Quality comparison between an original image and its upscaled counterpart.
//!
//! Usage: `srgan-rust compare <input> <upscaled> [--output comparison.jpg]`
//!
//! Computes PSNR, SSIM, file-size ratio, and a pixel-difference histogram.
//! Saves a side-by-side centre-crop comparison image and prints a recommendation.

use crate::error::{Result, SrganError};
use alumina::data::image_folder::image_to_data;
use clap::ArgMatches;
use image::GenericImage;
use std::fs;
use std::path::Path;

// ── Public entry point ────────────────────────────────────────────────────────

pub fn compare(app_m: &ArgMatches) -> Result<()> {
    let input_path = app_m
        .value_of("INPUT")
        .ok_or_else(|| SrganError::InvalidParameter("No INPUT file given".to_string()))?;
    let upscaled_path = app_m
        .value_of("UPSCALED")
        .ok_or_else(|| SrganError::InvalidParameter("No UPSCALED file given".to_string()))?;
    let output_path = app_m.value_of("OUTPUT").unwrap_or("comparison.jpg");

    let input_img = image::open(Path::new(input_path))?;
    let upscaled_img = image::open(Path::new(upscaled_path))?;

    let (in_w, in_h) = input_img.dimensions();
    let (up_w, up_h) = upscaled_img.dimensions();

    println!("Input:    {} ({}x{})", input_path, in_w, in_h);
    println!("Upscaled: {} ({}x{})", upscaled_path, up_w, up_h);
    if in_w > 0 {
        println!("Scale:    {:.2}x", up_w as f32 / in_w as f32);
    }

    // ── File size ratio ───────────────────────────────────────────────────────
    if let Some(r) = compute_size_ratio(input_path, upscaled_path) {
        println!(
            "File size ratio: {:.2}x ({} -> {})",
            r.ratio,
            human_bytes(r.input_bytes),
            human_bytes(r.output_bytes)
        );
    }

    // ── PSNR / SSIM ──────────────────────────────────────────────────────────
    let input_data = image_to_data(&input_img);
    let upscaled_data = image_to_data(&upscaled_img);

    let (err, y_err, pix) = crate::psnr::psnr_calculation(input_data.view(), upscaled_data.view());
    let ssim = crate::ssim::ssim_calculation(input_data.view(), upscaled_data.view());

    use crate::constants::psnr as psnr_constants;
    let srgb_psnr = psnr_constants::LOG10_MULTIPLIER * (err / pix).log10();
    let luma_psnr = psnr_constants::LOG10_MULTIPLIER * (y_err / pix).log10();

    println!("\nQuality metrics (overlapping region):");
    println!("  sRGB PSNR: {:.2} dB", srgb_psnr);
    println!("  Luma PSNR: {:.2} dB", luma_psnr);
    println!("  SSIM:      {:.4}", ssim);

    // ── Pixel-difference histogram ────────────────────────────────────────────
    let histogram = compute_diff_histogram(&input_img, &upscaled_img);
    print_diff_histogram(&histogram);

    // ── Side-by-side comparison image ─────────────────────────────────────────
    match save_comparison(&input_img, &upscaled_img, output_path, 256) {
        Ok(_) => println!("\nComparison image saved to: {}", output_path),
        Err(e) => eprintln!("Warning: could not save comparison image: {}", e),
    }

    // ── Recommendation ────────────────────────────────────────────────────────
    let recommendation = make_recommendation(srgb_psnr, ssim, &histogram);
    println!("\nRecommendation: {}", recommendation);

    Ok(())
}

// ── File size ratio ───────────────────────────────────────────────────────────

struct SizeRatio {
    input_bytes: u64,
    output_bytes: u64,
    ratio: f64,
}

fn compute_size_ratio(input_path: &str, output_path: &str) -> Option<SizeRatio> {
    let in_bytes = fs::metadata(input_path).ok()?.len();
    let out_bytes = fs::metadata(output_path).ok()?.len();
    if in_bytes == 0 {
        return None;
    }
    Some(SizeRatio {
        input_bytes: in_bytes,
        output_bytes: out_bytes,
        ratio: out_bytes as f64 / in_bytes as f64,
    })
}

fn human_bytes(b: u64) -> String {
    if b < 1024 {
        format!("{}B", b)
    } else if b < 1024 * 1024 {
        format!("{:.1}KB", b as f64 / 1024.0)
    } else {
        format!("{:.1}MB", b as f64 / (1024.0 * 1024.0))
    }
}

// ── Pixel-difference histogram ────────────────────────────────────────────────

struct DiffHistogram {
    /// 10 buckets covering [0,25), [25,50), …, [225,256).
    buckets: [u32; 10],
    total: u32,
    mean_diff: f32,
    max_diff: u8,
}

fn compute_diff_histogram(a: &image::DynamicImage, b: &image::DynamicImage) -> DiffHistogram {
    let a_rgb = a.to_rgb();
    let b_rgb = b.to_rgb();
    let (aw, ah) = a.dimensions();
    let (bw, bh) = b.dimensions();
    let w = aw.min(bw);
    let h = ah.min(bh);

    let mut buckets = [0u32; 10];
    let mut total = 0u32;
    let mut sum_diff = 0u64;
    let mut max_diff = 0u8;

    let stride = (((w * h) / 40_000) as u32).max(1);

    let mut y = 0u32;
    while y < h {
        let mut x = 0u32;
        while x < w {
            let pa = a_rgb.get_pixel(x, y);
            let pb = b_rgb.get_pixel(x, y);
            for c in 0..3usize {
                let diff = if pa[c] > pb[c] { pa[c] - pb[c] } else { pb[c] - pa[c] };
                let bucket = (diff as usize / 25).min(9);
                buckets[bucket] += 1;
                sum_diff += diff as u64;
                if diff > max_diff {
                    max_diff = diff;
                }
                total += 1;
            }
            x += stride;
        }
        y += stride;
    }

    let mean_diff = if total > 0 { sum_diff as f32 / total as f32 } else { 0.0 };
    DiffHistogram { buckets, total, mean_diff, max_diff }
}

fn print_diff_histogram(h: &DiffHistogram) {
    println!("\nPixel-difference distribution (per channel):");
    println!("  Mean diff: {:.1}/255  Max diff: {}/255", h.mean_diff, h.max_diff);
    let labels = [
        "  0-24 ", " 25-49 ", " 50-74 ", " 75-99 ", "100-124",
        "125-149", "150-174", "175-199", "200-224", "225-255",
    ];
    let max_count = h.buckets.iter().copied().max().unwrap_or(1).max(1);
    for (i, &count) in h.buckets.iter().enumerate() {
        // Skip trailing empty buckets except the first two.
        if i >= 2 && count == 0 && h.buckets[i.saturating_sub(1)] == 0 {
            continue;
        }
        let bar_len = (count as u64 * 30 / max_count as u64) as usize;
        let bar: String = "#".repeat(bar_len);
        let pct = if h.total > 0 { count as f32 / h.total as f32 * 100.0 } else { 0.0 };
        println!("  {} | {:5.1}% {}", labels[i], pct, bar);
    }
}

// ── Side-by-side comparison image ─────────────────────────────────────────────

fn save_comparison(
    input: &image::DynamicImage,
    upscaled: &image::DynamicImage,
    output_path: &str,
    crop_size: u32,
) -> Result<()> {
    let a_rgb = input.to_rgb();
    let b_rgb = upscaled.to_rgb();

    let (aw, ah) = input.dimensions();
    let (bw, bh) = upscaled.dimensions();

    let a_crop_w = crop_size.min(aw);
    let a_crop_h = crop_size.min(ah);
    let b_crop_w = crop_size.min(bw);
    let b_crop_h = crop_size.min(bh);

    let a_x = aw.saturating_sub(a_crop_w) / 2;
    let a_y = ah.saturating_sub(a_crop_h) / 2;
    let b_x = bw.saturating_sub(b_crop_w) / 2;
    let b_y = bh.saturating_sub(b_crop_h) / 2;

    let sep = 4u32;
    let out_w = a_crop_w + sep + b_crop_w;
    let out_h = a_crop_h.max(b_crop_h);

    let mut out: image::RgbImage = image::ImageBuffer::new(out_w, out_h);

    // Left: input crop
    for dy in 0..a_crop_h {
        for dx in 0..a_crop_w {
            let p = *a_rgb.get_pixel(a_x + dx, a_y + dy);
            out.put_pixel(dx, dy, p);
        }
    }

    // Separator
    for dy in 0..out_h {
        for dx in 0..sep {
            out.put_pixel(a_crop_w + dx, dy, image::Rgb([128u8, 128u8, 128u8]));
        }
    }

    // Right: upscaled crop
    for dy in 0..b_crop_h {
        for dx in 0..b_crop_w {
            let p = *b_rgb.get_pixel(b_x + dx, b_y + dy);
            out.put_pixel(a_crop_w + sep + dx, dy, p);
        }
    }

    image::DynamicImage::ImageRgb8(out)
        .save(Path::new(output_path))
        .map_err(SrganError::Io)?;

    Ok(())
}

// ── Recommendation ────────────────────────────────────────────────────────────

fn make_recommendation(psnr: f32, ssim: f32, hist: &DiffHistogram) -> &'static str {
    if psnr.is_nan() || psnr.is_infinite() {
        return "Images appear identical or cannot be compared.";
    }
    // Over-sharpening: high PSNR but structural dissimilarity
    if psnr > 34.0 && ssim < 0.82 {
        return "Over-sharpened — high PSNR but low structural similarity; consider a smoother model.";
    }
    // Artifacts: many pixels with large channel differences
    let high_diff_frac = if hist.total > 0 {
        let high: u32 = hist.buckets[4..].iter().sum();
        high as f32 / hist.total as f32
    } else {
        0.0
    };
    if high_diff_frac > 0.05 || hist.max_diff > 150 {
        return "Artifacts detected — significant pixel differences present; check for ringing or noise amplification.";
    }
    if psnr >= 38.0 && ssim >= 0.96 {
        return "Excellent upscale — very high fidelity, minimal distortion.";
    }
    if psnr >= 32.0 && ssim >= 0.90 {
        return "Good upscale — strong detail recovery with low artifact level.";
    }
    if psnr >= 26.0 && ssim >= 0.80 {
        return "Acceptable upscale — noticeable improvement with minor artifacts.";
    }
    "Poor quality — consider a different model or verify the input image quality."
}
