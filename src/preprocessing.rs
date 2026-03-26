//! Image preprocessing pipeline.
//!
//! Runs before every inference call:
//! 1. Detect and reject corrupted / undecodable images.
//! 2. Enforce a maximum input pixel count with a helpful error message.
//! 3. Convert CMYK images to RGB (the neural network expects RGB).
//! 4. Apply EXIF orientation correction.
//! 5. Pad dimensions to the nearest multiple of 4 (required by the SR network).
//! 6. Negotiate the best output format based on the input format.

use image::{DynamicImage, GenericImage, ImageFormat};
use std::io::Cursor;

use crate::error::SrganError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum pixels (width × height).  Images above this limit are
/// rejected before inference to avoid OOM.  Callers may override via
/// [`PreprocessingConfig`].
pub const DEFAULT_MAX_PIXELS: u64 = 25_000_000; // 25 MP

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Knobs that control the preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Pixel-count ceiling.  `None` means unlimited.
    pub max_pixels: Option<u64>,
    /// Whether to apply EXIF orientation correction (requires raw bytes).
    pub apply_exif_rotation: bool,
    /// Pad dimensions to a multiple of this value (default 4).
    pub align: u32,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            max_pixels: Some(DEFAULT_MAX_PIXELS),
            apply_exif_rotation: true,
            align: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Output format negotiation
// ---------------------------------------------------------------------------

/// The format that should be used when encoding the final result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Png,
    Jpeg,
}

impl OutputFormat {
    /// Choose the best output format given what we detected on input.
    pub fn negotiate(input_format: Option<ImageFormat>) -> Self {
        match input_format {
            Some(ImageFormat::JPEG) => OutputFormat::Jpeg,
            // PNG, BMP, GIF, TIFF → preserve lossless quality
            _ => OutputFormat::Png,
        }
    }

    pub fn mime_type(&self) -> &'static str {
        match self {
            OutputFormat::Png => "image/png",
            OutputFormat::Jpeg => "image/jpeg",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpg",
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// The processed image together with metadata the caller needs downstream.
pub struct PreprocessedImage {
    pub image: DynamicImage,
    pub output_format: OutputFormat,
    /// Width after alignment padding (may differ from original).
    pub padded_width: u32,
    /// Height after alignment padding (may differ from original).
    pub padded_height: u32,
    /// Original width before any padding.
    pub original_width: u32,
    /// Original height before any padding.
    pub original_height: u32,
}

// ---------------------------------------------------------------------------
// Public entry-point
// ---------------------------------------------------------------------------

/// Run the full preprocessing pipeline on raw image bytes.
///
/// Returns a [`PreprocessedImage`] ready to feed into inference, or a
/// descriptive [`SrganError`] if the input is invalid or oversized.
pub fn preprocess(raw: &[u8], cfg: &PreprocessingConfig) -> Result<PreprocessedImage, SrganError> {
    // ── 1. Detect corruption / decode ────────────────────────────────────────
    let fmt = detect_format(raw);
    let mut img = decode_image(raw)?;

    // ── 2. CMYK → RGB ────────────────────────────────────────────────────────
    img = convert_colorspace(img);

    // ── 3. EXIF orientation ──────────────────────────────────────────────────
    if cfg.apply_exif_rotation {
        img = apply_exif_orientation(raw, img);
    }

    // ── 4. Enforce max pixel count ───────────────────────────────────────────
    let w = img.width();
    let h = img.height();
    if let Some(max_px) = cfg.max_pixels {
        let px = w as u64 * h as u64;
        if px > max_px {
            return Err(SrganError::InvalidInput(format!(
                "image is {}×{} = {} pixels, which exceeds the maximum of {} pixels. \
                 Please downscale the image before submitting.",
                w, h, px, max_px
            )));
        }
    }

    // ── 5. Pad to alignment multiple ─────────────────────────────────────────
    let aligned = align_dimensions(img, cfg.align)?;

    // ── 6. Negotiate output format ────────────────────────────────────────────
    let output_format = OutputFormat::negotiate(fmt);

    Ok(PreprocessedImage {
        padded_width: aligned.width(),
        padded_height: aligned.height(),
        original_width: w,
        original_height: h,
        image: aligned,
        output_format,
    })
}

// ---------------------------------------------------------------------------
// Step implementations
// ---------------------------------------------------------------------------

/// Attempt to detect the image format from raw bytes without a full decode.
fn detect_format(raw: &[u8]) -> Option<ImageFormat> {
    image::guess_format(raw).ok()
}

/// Decode image bytes, returning a useful error on failure.
fn decode_image(raw: &[u8]) -> Result<DynamicImage, SrganError> {
    image::load_from_memory(raw).map_err(|e| {
        // Distinguish "I can't decode this" from "the file is corrupt".
        SrganError::InvalidInput(format!(
            "could not decode image (it may be corrupt or in an unsupported format): {}",
            e
        ))
    })
}

/// Convert the image to RGB8 so the neural network always receives RGB input.
///
/// This handles CMYK (rare but common in print-originated JPEGs) by converting
/// via `to_rgb()`, which the `image` crate will do automatically when the
/// underlying decoder exposes a CMYK buffer.
fn convert_colorspace(img: DynamicImage) -> DynamicImage {
    // `to_rgb()` handles Luma, LumaA, RGBA, and CMYK → RGB.
    // Wrap back in DynamicImage so the rest of the pipeline stays generic.
    DynamicImage::ImageRgb8(img.to_rgb())
}

/// Read the EXIF orientation tag from raw bytes and rotate/flip the image
/// accordingly so the pixels match what a viewer would show.
///
/// If EXIF data is absent or unreadable the image is returned unchanged.
fn apply_exif_orientation(raw: &[u8], img: DynamicImage) -> DynamicImage {
    let orientation = read_exif_orientation(raw).unwrap_or(1);
    rotate_by_orientation(img, orientation)
}

/// Returns the EXIF orientation value (1–8), or `None` if unavailable.
fn read_exif_orientation(raw: &[u8]) -> Option<u16> {
    // Parse EXIF using the kamadak-exif crate.
    let mut cursor = Cursor::new(raw);
    let reader = exif::Reader::new();
    let exif_data = reader.read_from_container(&mut cursor).ok()?;
    let field = exif_data.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    match field.value {
        exif::Value::Short(ref v) => v.first().copied(),
        _ => None,
    }
}

/// Apply a rotation/flip according to the EXIF orientation integer (1–8).
fn rotate_by_orientation(img: DynamicImage, orientation: u16) -> DynamicImage {
    match orientation {
        // 1 = normal, 2 = flipped horizontally, 3 = 180°, 4 = flipped vertically,
        // 5 = transposed, 6 = 90° CW, 7 = transverse, 8 = 90° CCW
        2 => DynamicImage::ImageRgb8(image::imageops::flip_horizontal(&img.to_rgb())),
        3 => DynamicImage::ImageRgb8(image::imageops::rotate180(&img.to_rgb())),
        4 => DynamicImage::ImageRgb8(image::imageops::flip_vertical(&img.to_rgb())),
        5 => {
            let t = image::imageops::rotate90(&img.to_rgb());
            DynamicImage::ImageRgb8(image::imageops::flip_horizontal(&t))
        }
        6 => DynamicImage::ImageRgb8(image::imageops::rotate90(&img.to_rgb())),
        7 => {
            let t = image::imageops::rotate270(&img.to_rgb());
            DynamicImage::ImageRgb8(image::imageops::flip_horizontal(&t))
        }
        8 => DynamicImage::ImageRgb8(image::imageops::rotate270(&img.to_rgb())),
        _ => img, // 1 or unknown: no transform
    }
}

/// Pad the image so both dimensions are a multiple of `align`.
///
/// Padding is filled with black (zero) pixels on the right and bottom edges.
fn align_dimensions(img: DynamicImage, align: u32) -> Result<DynamicImage, SrganError> {
    if align == 0 {
        return Ok(img);
    }
    let w = img.width();
    let h = img.height();
    let new_w = round_up(w, align);
    let new_h = round_up(h, align);

    if new_w == w && new_h == h {
        return Ok(img);
    }

    let mut canvas = DynamicImage::new_rgb8(new_w, new_h);
    image::imageops::overlay(&mut canvas, &img, 0, 0);
    Ok(canvas)
}

fn round_up(n: u32, multiple: u32) -> u32 {
    ((n + multiple - 1) / multiple) * multiple
}

// ---------------------------------------------------------------------------
// Encode result
// ---------------------------------------------------------------------------

/// Encode a `DynamicImage` into bytes using the negotiated output format.
pub fn encode_result(
    img: &DynamicImage,
    fmt: OutputFormat,
    jpeg_quality: u8,
) -> Result<Vec<u8>, SrganError> {
    let mut buf = Cursor::new(Vec::new());
    match fmt {
        OutputFormat::Png => {
            img.write_to(&mut buf, ImageFormat::PNG)
                .map_err(SrganError::Image)?;
        }
        OutputFormat::Jpeg => {
            let rgb = img.to_rgb();
            let (w, h) = (rgb.width(), rgb.height());
            image::jpeg::JPEGEncoder::new_with_quality(&mut buf, jpeg_quality)
                .encode(rgb.as_ref(), w, h, image::ColorType::RGB(8))
                .map_err(SrganError::Io)?;
        }
    }
    Ok(buf.into_inner())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_png_bytes() -> Vec<u8> {
        // 1×1 white PNG
        let img = DynamicImage::new_rgb8(1, 1);
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::PNG).unwrap();
        buf.into_inner()
    }

    #[test]
    fn rejects_corrupt_bytes() {
        let cfg = PreprocessingConfig::default();
        let err = preprocess(b"not an image", &cfg).unwrap_err();
        assert!(matches!(err, SrganError::InvalidInput(_)));
    }

    #[test]
    fn rejects_oversized_image() {
        // Create a 4×4 image and set max_pixels = 4 (4×4 = 16 > 4).
        let img = DynamicImage::new_rgb8(4, 4);
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::PNG).unwrap();
        let raw = buf.into_inner();

        let cfg = PreprocessingConfig { max_pixels: Some(4), ..Default::default() };
        let err = preprocess(&raw, &cfg).unwrap_err();
        assert!(matches!(err, SrganError::InvalidInput(_)));
    }

    #[test]
    fn pads_to_multiple_of_4() {
        // 5×7 input → 8×8 after alignment
        let img = DynamicImage::new_rgb8(5, 7);
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::PNG).unwrap();
        let raw = buf.into_inner();

        let cfg = PreprocessingConfig { max_pixels: None, ..Default::default() };
        let result = preprocess(&raw, &cfg).unwrap();
        assert_eq!(result.padded_width, 8);
        assert_eq!(result.padded_height, 8);
        assert_eq!(result.original_width, 5);
        assert_eq!(result.original_height, 7);
    }

    #[test]
    fn png_in_gives_png_out() {
        let raw = tiny_png_bytes();
        let cfg = PreprocessingConfig { max_pixels: None, ..Default::default() };
        let result = preprocess(&raw, &cfg).unwrap();
        assert_eq!(result.output_format, OutputFormat::Png);
    }

    #[test]
    fn negotiate_format_jpeg() {
        assert_eq!(OutputFormat::negotiate(Some(ImageFormat::JPEG)), OutputFormat::Jpeg);
    }

    #[test]
    fn negotiate_format_unknown_gives_png() {
        assert_eq!(OutputFormat::negotiate(None), OutputFormat::Png);
    }

    #[test]
    fn already_aligned_image_not_padded() {
        let img = DynamicImage::new_rgb8(8, 8);
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::PNG).unwrap();
        let raw = buf.into_inner();

        let cfg = PreprocessingConfig { max_pixels: None, ..Default::default() };
        let result = preprocess(&raw, &cfg).unwrap();
        assert_eq!(result.padded_width, 8);
        assert_eq!(result.padded_height, 8);
    }
}
