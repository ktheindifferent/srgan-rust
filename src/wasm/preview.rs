//! Lightweight 2x nearest-neighbor + sharpening upscaler for WASM browser preview.
//!
//! This module provides a pure-Rust image upscaler that can be compiled to WASM
//! via `wasm-bindgen` / `wasm-pack`. It performs 2x nearest-neighbor upscaling
//! followed by a 3x3 sharpening convolution — fast enough for real-time browser
//! previews without needing the full SRGAN model.
//!
//! When compiled as a native library (non-WASM), the same logic is available for
//! testing and for generating preview.js / preview_bg.wasm content served by the
//! web server.

use serde::{Deserialize, Serialize};

// ── Public types ─────────────────────────────────────────────────────────────

/// Configuration for the browser preview upscaler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewConfig {
    /// Upscale factor (default 2).
    pub scale: u32,
    /// Sharpening strength 0.0–1.0 (default 0.5).
    pub sharpen_strength: f32,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            scale: 2,
            sharpen_strength: 0.5,
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

// ── Core upscaling logic ─────────────────────────────────────────────────────

/// Perform nearest-neighbor upscaling on raw RGBA pixel data.
///
/// `src` must have length `w * h * 4`.
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

/// Apply a 3x3 sharpening convolution to RGBA pixel data.
///
/// Uses the kernel:
/// ```text
///  0  -s   0
/// -s 1+4s -s
///  0  -s   0
/// ```
/// where `s = strength`.
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
            // Alpha channel: copy through unchanged
        }
    }

    dst
}

/// Run the full preview pipeline: nearest-neighbor 2x + sharpen.
pub fn upscale_preview(rgba: &[u8], w: u32, h: u32, config: &PreviewConfig) -> PreviewResult {
    let start = std::time::Instant::now();

    let (upscaled, ow, oh) = nearest_neighbor_upscale(rgba, w, h, config.scale);
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

// ── Embedded WASM JS glue ────────────────────────────────────────────────────
// When real wasm-pack output is not available on disk, the server serves this
// minimal JS module that implements the same API using Canvas.

/// Fallback `preview.js` content that uses Canvas instead of real WASM.
pub const PREVIEW_JS_FALLBACK: &str = r#"// srgan-wasm-preview fallback (Canvas-based, no real WASM)
// This is served when wasm-pack output is not found on disk.

let _ready = false;

export async function default_init() { _ready = true; }
export { default_init as default };

export function version() { return "0.1.0-canvas-fallback"; }

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
/// This lets the browser fetch succeed without 404 even when wasm-pack
/// hasn't been run.
pub const PREVIEW_WASM_FALLBACK: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, // magic: \0asm
    0x01, 0x00, 0x00, 0x00, // version 1
];

// ── PNG encoding helper (pure Rust, no external crate) ───────────────────────

/// Encode raw RGBA pixels into a minimal valid PNG file.
/// Used by the WASM preview to return results without depending on the `image` crate.
pub fn rgba_to_png(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    // We use an uncompressed (store) DEFLATE approach for simplicity.
    // For preview images this is acceptable.
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
    let row_len = (width as usize) * 4 + 1; // +1 for filter byte
    let mut raw = Vec::with_capacity(row_len * height as usize);
    for y in 0..height as usize {
        raw.push(0); // filter: None
        let start = y * (width as usize) * 4;
        let end = start + (width as usize) * 4;
        raw.extend_from_slice(&pixels[start..end]);
    }

    // Wrap in zlib (DEFLATE store blocks)
    let deflated = zlib_store(&raw);
    write_png_chunk(&mut png, b"IDAT", &deflated);

    // IEND
    write_png_chunk(&mut png, b"IEND", &[]);

    png
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    // CRC32 over type + data
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
    // zlib header (no compression)
    out.push(0x78);
    out.push(0x01);

    // DEFLATE store blocks (max 65535 bytes each)
    let chunks: Vec<&[u8]> = data.chunks(65535).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i == chunks.len() - 1;
        out.push(if is_last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }

    // Adler-32 checksum
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
        // 2x2 red image → 4x4
        let src = vec![
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 0, 255,
        ];
        let (dst, w, h) = nearest_neighbor_upscale(&src, 2, 2, 2);
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(dst.len(), 4 * 4 * 4);
        // Top-left 2x2 block should all be red
        assert_eq!(&dst[0..4], &[255, 0, 0, 255]);
        assert_eq!(&dst[4..8], &[255, 0, 0, 255]);
    }

    #[test]
    fn test_sharpen_identity() {
        // With strength 0, output should equal input
        let src = vec![128u8; 4 * 4 * 4]; // 4x4 uniform grey
        let dst = sharpen(&src, 4, 4, 0.0);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_preview_pipeline() {
        let src = vec![128u8; 4 * 4 * 4]; // 4x4 image
        let config = PreviewConfig::default();
        let result = upscale_preview(&src, 4, 4, &config);
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        assert_eq!(result.pixels.len(), (8 * 8 * 4) as usize);
    }

    #[test]
    fn test_rgba_to_png_valid_header() {
        let pixels = vec![255u8; 2 * 2 * 4];
        let png = rgba_to_png(&pixels, 2, 2);
        // Check PNG signature
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }
}
