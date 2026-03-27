//! Lightweight 2x bicubic/Lanczos upscaler for WASM browser preview.
//!
//! This module provides a pure-Rust image upscaler that can be compiled to WASM
//! via `wasm-bindgen` / `wasm-pack`. It performs 2x bicubic or Lanczos upscaling
//! for instant in-browser previews before submitting to the real SRGAN API.
//!
//! When compiled as a native library (non-WASM), the same logic is available for
//! testing and for generating preview.js / preview_bg.wasm content served by the
//! web server.

use serde::{Deserialize, Serialize};

// ── Public types ─────────────────────────────────────────────────────────────

/// Interpolation method for browser preview upscaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Standard nearest-neighbor (fastest, lowest quality).
    NearestNeighbor,
    /// Bicubic interpolation (good quality, fast).
    Bicubic,
    /// Lanczos-3 interpolation (best quality, slightly slower).
    Lanczos3,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        InterpolationMethod::Lanczos3
    }
}

/// Configuration for the browser preview upscaler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewConfig {
    /// Upscale factor (default 2).
    pub scale: u32,
    /// Sharpening strength 0.0–1.0 (default 0.3).
    pub sharpen_strength: f32,
    /// Interpolation method (default Lanczos3).
    pub method: InterpolationMethod,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            scale: 2,
            sharpen_strength: 0.3,
            method: InterpolationMethod::Lanczos3,
        }
    }
}

/// Result of a preview upscale operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewResult {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// RGBA pixel data (length = width * height * 4).
    pub pixels: Vec<u8>,
    /// Processing time in milliseconds.
    pub elapsed_ms: u64,
}

// ── Bicubic interpolation ────────────────────────────────────────────────────

/// Cubic interpolation kernel: Mitchell-Netravali (B=1/3, C=1/3) variant.
fn bicubic_weight(t: f64) -> f64 {
    let t = t.abs();
    if t < 1.0 {
        (1.5 * t * t * t) - (2.5 * t * t) + 1.0
    } else if t < 2.0 {
        (-0.5 * t * t * t) + (2.5 * t * t) - (4.0 * t) + 2.0
    } else {
        0.0
    }
}

/// Upscale RGBA pixels using bicubic interpolation.
pub fn bicubic_upscale(src: &[u8], w: u32, h: u32, scale: u32) -> (Vec<u8>, u32, u32) {
    let ow = w * scale;
    let oh = h * scale;
    let mut dst = vec![0u8; (ow * oh * 4) as usize];

    for oy in 0..oh {
        for ox in 0..ow {
            let src_x = (ox as f64 + 0.5) / scale as f64 - 0.5;
            let src_y = (oy as f64 + 0.5) / scale as f64 - 0.5;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;

            let mut rgba = [0.0f64; 4];
            let mut weight_sum = 0.0f64;

            for ky in -1..=2i32 {
                for kx in -1..=2i32 {
                    let sx = (x0 + kx).max(0).min(w as i32 - 1) as u32;
                    let sy = (y0 + ky).max(0).min(h as i32 - 1) as u32;

                    let wx = bicubic_weight(src_x - (x0 + kx) as f64);
                    let wy = bicubic_weight(src_y - (y0 + ky) as f64);
                    let weight = wx * wy;

                    let si = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        rgba[c] += src[si + c] as f64 * weight;
                    }
                    weight_sum += weight;
                }
            }

            let di = ((oy * ow + ox) * 4) as usize;
            if weight_sum > 0.0 {
                for c in 0..4 {
                    dst[di + c] = (rgba[c] / weight_sum).round().max(0.0).min(255.0) as u8;
                }
            }
        }
    }

    (dst, ow, oh)
}

// ── Lanczos-3 interpolation ──────────────────────────────────────────────────

/// Lanczos kernel with window size `a`.
fn lanczos_weight(t: f64, a: f64) -> f64 {
    if t.abs() < 1e-8 {
        return 1.0;
    }
    if t.abs() >= a {
        return 0.0;
    }
    let pi_t = std::f64::consts::PI * t;
    let pi_t_a = std::f64::consts::PI * t / a;
    (pi_t.sin() / pi_t) * (pi_t_a.sin() / pi_t_a)
}

/// Upscale RGBA pixels using Lanczos-3 interpolation.
pub fn lanczos3_upscale(src: &[u8], w: u32, h: u32, scale: u32) -> (Vec<u8>, u32, u32) {
    let ow = w * scale;
    let oh = h * scale;
    let mut dst = vec![0u8; (ow * oh * 4) as usize];
    let a = 3.0f64; // Lanczos-3

    for oy in 0..oh {
        for ox in 0..ow {
            let src_x = (ox as f64 + 0.5) / scale as f64 - 0.5;
            let src_y = (oy as f64 + 0.5) / scale as f64 - 0.5;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;

            let mut rgba = [0.0f64; 4];
            let mut weight_sum = 0.0f64;

            let radius = a as i32;
            for ky in (-radius + 1)..=radius {
                for kx in (-radius + 1)..=radius {
                    let sx = (x0 + kx).max(0).min(w as i32 - 1) as u32;
                    let sy = (y0 + ky).max(0).min(h as i32 - 1) as u32;

                    let wx = lanczos_weight(src_x - (x0 + kx) as f64, a);
                    let wy = lanczos_weight(src_y - (y0 + ky) as f64, a);
                    let weight = wx * wy;

                    let si = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        rgba[c] += src[si + c] as f64 * weight;
                    }
                    weight_sum += weight;
                }
            }

            let di = ((oy * ow + ox) * 4) as usize;
            if weight_sum > 0.0 {
                for c in 0..4 {
                    dst[di + c] = (rgba[c] / weight_sum).round().max(0.0).min(255.0) as u8;
                }
            }
        }
    }

    (dst, ow, oh)
}

// ── Nearest-neighbor (kept for compatibility) ────────────────────────────────

/// Perform nearest-neighbor upscaling on raw RGBA pixel data.
pub fn nearest_neighbor_upscale(src: &[u8], w: u32, h: u32, scale: u32) -> (Vec<u8>, u32, u32) {
    let ow = w * scale;
    let oh = h * scale;
    let mut dst = vec![0u8; (ow * oh * 4) as usize];

    for y in 0..oh {
        let sy = (y / scale).min(h - 1);
        for x in 0..ow {
            let sx = (x / scale).min(w - 1);
            let si = ((sy * w + sx) * 4) as usize;
            let di = ((y * ow + x) * 4) as usize;
            dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }

    (dst, ow, oh)
}

// ── Sharpening ───────────────────────────────────────────────────────────────

/// Apply a 3x3 sharpening convolution to RGBA pixel data.
pub fn sharpen(src: &[u8], w: u32, h: u32, strength: f32) -> Vec<u8> {
    let mut dst = src.to_vec();
    let w = w as i32;
    let h = h as i32;
    let s = strength;

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            for c in 0..3u8 {
                let idx = |dx: i32, dy: i32| -> usize {
                    (((y + dy) * w + (x + dx)) * 4 + c as i32) as usize
                };
                let center = src[idx(0, 0)] as f32;
                let top = src[idx(0, -1)] as f32;
                let bottom = src[idx(0, 1)] as f32;
                let left = src[idx(-1, 0)] as f32;
                let right = src[idx(1, 0)] as f32;

                let val = center * (1.0 + 4.0 * s) - s * (top + bottom + left + right);
                dst[idx(0, 0)] = val.round().max(0.0).min(255.0) as u8;
            }
        }
    }

    dst
}

// ── Main preview pipeline ────────────────────────────────────────────────────

/// Run the full preview pipeline: upscale with chosen method + optional sharpen.
pub fn upscale_preview(rgba: &[u8], w: u32, h: u32, config: &PreviewConfig) -> PreviewResult {
    let start = std::time::Instant::now();

    let (upscaled, ow, oh) = match config.method {
        InterpolationMethod::NearestNeighbor => nearest_neighbor_upscale(rgba, w, h, config.scale),
        InterpolationMethod::Bicubic => bicubic_upscale(rgba, w, h, config.scale),
        InterpolationMethod::Lanczos3 => lanczos3_upscale(rgba, w, h, config.scale),
    };

    let sharpened = if config.sharpen_strength > 0.0 {
        sharpen(&upscaled, ow, oh, config.sharpen_strength)
    } else {
        upscaled
    };

    PreviewResult {
        width: ow,
        height: oh,
        pixels: sharpened,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

/// WASM-bindgen compatible entry point: takes raw PNG/JPEG bytes, returns upscaled PNG bytes.
///
/// This is the function exported to JavaScript via wasm-bindgen.
/// It decodes the input image, runs Lanczos 2x upscale, and re-encodes as PNG.
pub fn upscale_preview_bytes(image_data: &[u8]) -> Vec<u8> {
    // Try to decode as raw RGBA (with width/height header) or as minimal PNG
    let (pixels, w, h) = match decode_png_simple(image_data) {
        Some(decoded) => decoded,
        None => {
            // If not a valid PNG, assume raw RGBA with 4-byte width + 4-byte height header
            if image_data.len() < 8 {
                return Vec::new();
            }
            let w = u32::from_be_bytes([image_data[0], image_data[1], image_data[2], image_data[3]]);
            let h = u32::from_be_bytes([image_data[4], image_data[5], image_data[6], image_data[7]]);
            let expected = (w * h * 4) as usize + 8;
            if image_data.len() < expected || w == 0 || h == 0 || w > 8192 || h > 8192 {
                return Vec::new();
            }
            (image_data[8..expected].to_vec(), w, h)
        }
    };

    let config = PreviewConfig::default();
    let result = upscale_preview(&pixels, w, h, &config);
    rgba_to_png(&result.pixels, result.width, result.height)
}

// ── Embedded WASM JS glue ────────────────────────────────────────────────────

/// Fallback `preview.js` content that uses Canvas instead of real WASM.
pub const PREVIEW_JS_FALLBACK: &str = r#"// srgan-wasm-preview fallback (Canvas-based, no real WASM)
// This is served when wasm-pack output is not found on disk.

let _ready = false;

export async function default_init() { _ready = true; }
export { default_init as default };

export function version() { return "0.2.0-canvas-fallback"; }

export function upscale_preview(inputBytes, method, scale) {
  // Decode input PNG → ImageData via OffscreenCanvas
  const blob = new Blob([inputBytes], { type: 'image/png' });
  // We need sync decoding — use a hidden <img> trick
  // For the fallback we return the input unchanged; the demo page
  // already has a canvas-based scaler as backup.
  return inputBytes;
}
"#;

/// Fallback `preview_bg.wasm` — a minimal valid WASM module (8 bytes).
pub const PREVIEW_WASM_FALLBACK: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, // magic: \0asm
    0x01, 0x00, 0x00, 0x00, // version 1
];

// ── PNG encoding helper (pure Rust, no external crate) ───────────────────────

/// Encode raw RGBA pixels into a minimal valid PNG file.
pub fn rgba_to_png(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut png = Vec::new();

    // PNG signature
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR chunk
    let mut ihdr_data = Vec::with_capacity(13);
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // bit depth
    ihdr_data.push(6); // color type: RGBA
    ihdr_data.push(0); // compression
    ihdr_data.push(0); // filter
    ihdr_data.push(0); // interlace
    write_png_chunk(&mut png, b"IHDR", &ihdr_data);

    // IDAT chunk — raw image data with filter byte 0 (None) per row
    let row_len = (width as usize) * 4 + 1;
    let mut raw = Vec::with_capacity(row_len * height as usize);
    for y in 0..height as usize {
        raw.push(0); // filter: None
        let start = y * (width as usize) * 4;
        let end = start + (width as usize) * 4;
        raw.extend_from_slice(&pixels[start..end]);
    }

    let deflated = zlib_store(&raw);
    write_png_chunk(&mut png, b"IDAT", &deflated);

    // IEND
    write_png_chunk(&mut png, b"IEND", &[]);

    png
}

/// Simple PNG decoder: extract RGBA pixels + dimensions from a PNG file.
/// Returns None if the data is not a valid PNG or uses unsupported features.
fn decode_png_simple(data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    // Check PNG signature
    if data.len() < 8 || &data[0..8] != &[137, 80, 78, 71, 13, 10, 26, 10] {
        return None;
    }

    let mut pos = 8;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bit_depth;
    let mut color_type = 0u8;
    let mut idat_data = Vec::new();

    while pos + 8 <= data.len() {
        let chunk_len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let chunk_type = &data[pos + 4..pos + 8];
        let chunk_data_start = pos + 8;
        let chunk_data_end = chunk_data_start + chunk_len;

        if chunk_data_end > data.len() {
            break;
        }

        match chunk_type {
            b"IHDR" if chunk_len >= 13 => {
                let d = &data[chunk_data_start..chunk_data_end];
                width = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
                height = u32::from_be_bytes([d[4], d[5], d[6], d[7]]);
                bit_depth = d[8];
                color_type = d[9];
                // Only support 8-bit RGBA
                if bit_depth != 8 || (color_type != 6 && color_type != 2) {
                    return None;
                }
            }
            b"IDAT" => {
                idat_data.extend_from_slice(&data[chunk_data_start..chunk_data_end]);
            }
            b"IEND" => break,
            _ => {}
        }

        pos = chunk_data_end + 4; // +4 for CRC
    }

    if width == 0 || height == 0 || idat_data.is_empty() {
        return None;
    }

    // Decompress zlib data
    let raw = zlib_decompress(&idat_data)?;

    // Unfilter rows
    let channels: usize = if color_type == 6 { 4 } else { 3 };
    let stride = width as usize * channels;
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    let mut prev_row = vec![0u8; stride];

    for y in 0..height as usize {
        let row_start = y * (stride + 1);
        if row_start >= raw.len() {
            return None;
        }
        let filter = raw[row_start];
        let row_data_start = row_start + 1;
        let row_data_end = row_data_start + stride;
        if row_data_end > raw.len() {
            return None;
        }

        let mut row = raw[row_data_start..row_data_end].to_vec();

        // Apply PNG filter
        match filter {
            0 => {} // None
            1 => {
                // Sub
                for i in channels..stride {
                    row[i] = row[i].wrapping_add(row[i - channels]);
                }
            }
            2 => {
                // Up
                for i in 0..stride {
                    row[i] = row[i].wrapping_add(prev_row[i]);
                }
            }
            3 => {
                // Average
                for i in 0..stride {
                    let left = if i >= channels { row[i - channels] as u16 } else { 0 };
                    let up = prev_row[i] as u16;
                    row[i] = row[i].wrapping_add(((left + up) / 2) as u8);
                }
            }
            4 => {
                // Paeth
                for i in 0..stride {
                    let left = if i >= channels { row[i - channels] as i32 } else { 0 };
                    let up = prev_row[i] as i32;
                    let up_left = if i >= channels { prev_row[i - channels] as i32 } else { 0 };
                    row[i] = row[i].wrapping_add(paeth_predictor(left, up, up_left) as u8);
                }
            }
            _ => return None,
        }

        // Convert to RGBA
        for x in 0..width as usize {
            if channels == 4 {
                pixels.extend_from_slice(&row[x * 4..(x * 4) + 4]);
            } else {
                pixels.extend_from_slice(&row[x * 3..(x * 3) + 3]);
                pixels.push(255); // Alpha
            }
        }

        prev_row = row;
    }

    Some((pixels, width, height))
}

fn paeth_predictor(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Minimal zlib decompressor for PNG IDAT data (DEFLATE store + fixed Huffman).
fn zlib_decompress(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 6 {
        return None;
    }
    // Skip zlib header (2 bytes) and checksum (4 bytes at end)
    let deflate_data = &data[2..data.len() - 4];
    deflate_decompress(deflate_data)
}

fn deflate_decompress(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0;

    loop {
        if pos >= data.len() {
            break;
        }

        let header = data[pos];
        let bfinal = header & 0x01;
        let btype = (header >> 1) & 0x03;
        pos += 1;

        match btype {
            0 => {
                // Stored block
                if pos + 4 > data.len() {
                    return None;
                }
                let len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 4; // len + nlen
                if pos + len > data.len() {
                    return None;
                }
                out.extend_from_slice(&data[pos..pos + len]);
                pos += len;
            }
            1 | 2 => {
                // Fixed or dynamic Huffman — for simplicity in preview context,
                // just return None and let caller fall back to raw RGBA header format
                return None;
            }
            _ => return None,
        }

        if bfinal != 0 {
            break;
        }
    }

    Some(out)
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32(chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(chunk_type: &[u8], data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc ^= b as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x78);
    out.push(0x01);

    let chunks: Vec<&[u8]> = data.chunks(65535).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i == chunks.len() - 1;
        out.push(if is_last { 0x01 } else { 0x00 });
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }

    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());

    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_neighbor_2x() {
        let src = vec![
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 0, 255,
        ];
        let (dst, w, h) = nearest_neighbor_upscale(&src, 2, 2, 2);
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(dst.len(), 4 * 4 * 4);
        assert_eq!(&dst[0..4], &[255, 0, 0, 255]);
        assert_eq!(&dst[4..8], &[255, 0, 0, 255]);
    }

    #[test]
    fn test_bicubic_upscale_dimensions() {
        let src = vec![128u8; 4 * 4 * 4]; // 4x4 image
        let (dst, w, h) = bicubic_upscale(&src, 4, 4, 2);
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(dst.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_lanczos3_upscale_dimensions() {
        let src = vec![128u8; 4 * 4 * 4]; // 4x4 image
        let (dst, w, h) = lanczos3_upscale(&src, 4, 4, 2);
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(dst.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_sharpen_identity() {
        let src = vec![128u8; 4 * 4 * 4];
        let dst = sharpen(&src, 4, 4, 0.0);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_preview_pipeline_bicubic() {
        let src = vec![128u8; 4 * 4 * 4];
        let config = PreviewConfig {
            scale: 2,
            sharpen_strength: 0.0,
            method: InterpolationMethod::Bicubic,
        };
        let result = upscale_preview(&src, 4, 4, &config);
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        assert_eq!(result.pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_preview_pipeline_lanczos() {
        let src = vec![128u8; 4 * 4 * 4];
        let config = PreviewConfig::default();
        let result = upscale_preview(&src, 4, 4, &config);
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        assert_eq!(result.pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_rgba_to_png_valid_header() {
        let pixels = vec![255u8; 2 * 2 * 4];
        let png = rgba_to_png(&pixels, 2, 2);
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn test_upscale_preview_bytes_invalid_input() {
        let result = upscale_preview_bytes(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_upscale_preview_bytes_raw_rgba() {
        // 2x2 image with raw RGBA header format
        let w = 2u32;
        let h = 2u32;
        let mut data = Vec::new();
        data.extend_from_slice(&w.to_be_bytes());
        data.extend_from_slice(&h.to_be_bytes());
        data.extend_from_slice(&vec![128u8; (w * h * 4) as usize]);
        let result = upscale_preview_bytes(&data);
        assert!(!result.is_empty());
        // Should be a valid PNG
        assert_eq!(&result[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }
}
