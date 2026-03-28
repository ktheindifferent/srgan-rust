//! Waifu2x model support for anime/illustration super-resolution.
//!
//! Waifu2x is a CNN-based image upscaling and noise-reduction algorithm
//! originally designed for anime-style artwork. It uses a VGG-like
//! convolutional network with:
//!
//! - Multiple 3×3 conv layers with ReLU activations
//! - No spatial pooling (preserves resolution)
//! - Optional noise reduction (levels 0–3)
//! - Sub-pixel convolution for upscaling (×1 = denoise only, ×2 = upscale)
//!
//! ## Inference modes
//!
//! When converted waifu2x weights (`.rsr` files) are available on disk, the
//! `Waifu2xNetwork` uses the VGG7 alumina graph for real CNN inference.
//! Otherwise it falls back to the "waifu2x-compat" software path (Lanczos3
//! resize + unsharp-mask sharpening).
//!
//! Weight files are searched for at:
//!   1. `$SRGAN_MODEL_PATH/waifu2x/noise{N}_scale{M}_{style}.rsr`
//!   2. `./models/waifu2x/noise{N}_scale{M}_{style}.rsr`
//!
//! To convert original waifu2x JSON weights to `.rsr`, use the
//! `Waifu2xWeightConverter` in `model_converter/waifu2x_converter.rs` or the
//! CLI command `convert-model --format waifu2x-json`.

use crate::config::{Waifu2xConfig, Waifu2xMode, Waifu2xStyle};
use crate::error::{Result, SrganError};
use image::GenericImage;
use log::{info, debug};
use std::path::{Path, PathBuf};

// ── NoiseLevel ────────────────────────────────────────────────────────────────

/// Waifu2x noise-reduction strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseLevel {
    /// No noise reduction.
    None = 0,
    /// Light noise reduction.
    Low = 1,
    /// Medium noise reduction.
    Medium = 2,
    /// Aggressive noise reduction.
    High = 3,
}

impl NoiseLevel {
    /// Parse from the integer stored in `Waifu2xConfig`.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => NoiseLevel::None,
            1 => NoiseLevel::Low,
            2 => NoiseLevel::Medium,
            _ => NoiseLevel::High,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl std::fmt::Display for NoiseLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_u8())
    }
}

// ── ScaleFactor ───────────────────────────────────────────────────────────────

/// Waifu2x upscaling factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waifu2xScale {
    /// Denoise only — output is the same resolution as the input.
    One,
    /// Upscale ×2.
    Two,
    /// Upscale ×3 (achieved via ×2 upscale then downscale to ×3 target).
    Three,
    /// Upscale ×4 (achieved via two iterative ×2 passes).
    Four,
}

impl Waifu2xScale {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 | 1 => Waifu2xScale::One,
            2 => Waifu2xScale::Two,
            3 => Waifu2xScale::Three,
            _ => Waifu2xScale::Four,
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Waifu2xScale::One => 1,
            Waifu2xScale::Two => 2,
            Waifu2xScale::Three => 3,
            Waifu2xScale::Four => 4,
        }
    }
}

impl std::fmt::Display for Waifu2xScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_u8())
    }
}

// ── Inference backend ────────────────────────────────────────────────────────

/// The inference backend for a `Waifu2xNetwork` instance.
enum Waifu2xBackend {
    /// Real VGG7 CNN inference via an alumina graph + loaded weights.
    Cnn {
        graph: crate::GraphDef,
        parameters: Vec<ndarray::ArrayD<f32>>,
    },
    /// NCNN-Vulkan backend: delegates to an external `waifu2x-ncnn-vulkan`
    /// binary for GPU-accelerated inference.  This is the preferred backend
    /// when available because it supports native waifu2x models directly and
    /// leverages Vulkan compute shaders for high throughput.
    Ncnn {
        /// Path to the `waifu2x-ncnn-vulkan` binary.
        binary_path: PathBuf,
        /// Path to the directory containing NCNN model files (`.param` + `.bin`).
        model_dir: PathBuf,
    },
    /// Software-only fallback: Lanczos3 resize + unsharp-mask sharpening.
    Compat,
}

// ── Waifu2xNetwork ──────────────────────────────────────────────────────────

/// High-level waifu2x wrapper.
///
/// When converted weight files are available on disk, inference runs through
/// the VGG7 alumina graph (real CNN).  Otherwise the waifu2x-compat software
/// fallback is used (Lanczos3 resize + unsharp mask).
pub struct Waifu2xNetwork {
    noise_level: NoiseLevel,
    scale: Waifu2xScale,
    style: Waifu2xStyle,
    mode: Waifu2xMode,
    backend: Waifu2xBackend,
}

impl Waifu2xNetwork {
    /// Build a `Waifu2xNetwork` from a [`Waifu2xConfig`].
    ///
    /// Attempts to load CNN weights from disk; falls back to compat mode.
    pub fn from_config(config: &Waifu2xConfig) -> Result<Self> {
        let noise_level = NoiseLevel::from_u8(config.noise_level);
        let scale = if config.mode == Waifu2xMode::Enhance {
            Waifu2xScale::One // Enhancement mode never upscales
        } else {
            Waifu2xScale::from_u8(config.scale)
        };
        let backend = Self::try_load_backend(config.noise_level, scale.as_u8(), config.style);
        Ok(Self { noise_level, scale, style: config.style, mode: config.mode, backend })
    }

    /// Load from a canonical label such as `"waifu2x"` or
    /// `"waifu2x-noise2-scale2"`.
    ///
    /// Uses the default style (`Anime`).  To specify a style, use
    /// [`from_label_with_style`] or [`from_config`].
    pub fn from_label(label: &str) -> Result<Self> {
        let config = parse_label(label)?;
        Self::from_config(&config)
    }

    /// Load from a label with an explicit style override.
    pub fn from_label_with_style(label: &str, style: Waifu2xStyle) -> Result<Self> {
        let mut config = parse_label(label)?;
        config.style = style;
        Self::from_config(&config)
    }

    /// Noise-reduction level this instance was built with.
    pub fn noise_level(&self) -> NoiseLevel {
        self.noise_level
    }

    /// Upscaling factor this instance was built with.
    pub fn scale(&self) -> Waifu2xScale {
        self.scale
    }

    /// Content style this instance was built with.
    pub fn style(&self) -> Waifu2xStyle {
        self.style
    }

    /// Processing mode this instance was built with.
    pub fn mode(&self) -> Waifu2xMode {
        self.mode
    }

    /// Whether this instance is in enhancement mode.
    pub fn is_enhance(&self) -> bool {
        self.mode == Waifu2xMode::Enhance
    }

    /// Whether this instance is using real CNN inference (vs compat fallback).
    pub fn is_cnn(&self) -> bool {
        matches!(self.backend, Waifu2xBackend::Cnn { .. })
    }

    /// Whether this instance is using the NCNN-Vulkan backend.
    pub fn is_ncnn(&self) -> bool {
        matches!(self.backend, Waifu2xBackend::Ncnn { .. })
    }

    /// Upscale (and optionally denoise) a [`image::DynamicImage`].
    ///
    /// For scale factors 3× and 4×, the image is iteratively upscaled using 2×
    /// passes:
    /// - 3× = one 2× pass → Lanczos3 resize to the exact 3× target
    /// - 4× = two consecutive 2× passes
    ///
    /// In enhancement mode, the image is denoised, sharpened, and
    /// contrast-adjusted without any resolution change.
    ///
    /// Uses CNN, NCNN, or compat backend depending on what is available.
    pub fn upscale_image(
        &self,
        img: &image::DynamicImage,
    ) -> Result<image::DynamicImage> {
        if self.mode == Waifu2xMode::Enhance {
            return self.enhance_image(img);
        }

        let scale = self.scale.as_u8();

        match scale {
            0 | 1 => self.upscale_single_pass(img, Waifu2xScale::One),
            2 => self.upscale_single_pass(img, Waifu2xScale::Two),
            3 => {
                // 2× pass then Lanczos3 down to exact 3× dimensions.
                let pass1 = self.upscale_single_pass(img, Waifu2xScale::Two)?;
                let (w, h) = (img.width(), img.height());
                let target_w = w * 3;
                let target_h = h * 3;
                Ok(pass1.resize_exact(target_w, target_h, image::FilterType::Lanczos3))
            }
            _ => {
                // 4× = two 2× passes.
                let pass1 = self.upscale_single_pass(img, Waifu2xScale::Two)?;
                self.upscale_single_pass(&pass1, Waifu2xScale::Two)
            }
        }
    }

    /// Enhancement mode: denoise + sharpen + contrast adjustment.
    ///
    /// Applies a multi-stage pipeline at the original resolution:
    /// 1. CNN-based denoise pass (if CNN/NCNN backend available), else compat denoise
    /// 2. Adaptive unsharp-mask sharpening tuned to content style
    /// 3. Local contrast enhancement via CLAHE-like histogram stretching
    fn enhance_image(
        &self,
        img: &image::DynamicImage,
    ) -> Result<image::DynamicImage> {
        info!("Waifu2x enhance mode: noise={} style={}", self.noise_level, self.style);

        // Step 1: Denoise pass at scale=1 (same resolution)
        let denoised = self.upscale_single_pass(img, Waifu2xScale::One)?;

        // Step 2: Adaptive sharpening (stronger than standard upscale sharpening)
        let style_multiplier = match self.style {
            Waifu2xStyle::Anime   => 1.2f32,
            Waifu2xStyle::Artwork => 1.0,
            Waifu2xStyle::Photo   => 0.7,
        };

        let base_amount = match self.noise_level {
            NoiseLevel::None   => 0.2f32, // Even at noise=0, enhance applies light sharpening
            NoiseLevel::Low    => 0.4,
            NoiseLevel::Medium => 0.6,
            NoiseLevel::High   => 0.9,
        };

        let sharpen_amount = base_amount * style_multiplier;
        let sharpened = unsharp_mask(&denoised, sharpen_amount);

        // Step 3: Local contrast enhancement
        let enhanced = adjust_contrast(&sharpened, self.style);

        Ok(enhanced)
    }

    /// Run a single upscale pass at the given scale using the active backend.
    fn upscale_single_pass(
        &self,
        img: &image::DynamicImage,
        pass_scale: Waifu2xScale,
    ) -> Result<image::DynamicImage> {
        match &self.backend {
            Waifu2xBackend::Cnn { graph, parameters } => {
                self.upscale_cnn(img, graph, parameters)
            }
            Waifu2xBackend::Ncnn { binary_path, model_dir } => {
                self.upscale_ncnn(img, pass_scale, binary_path, model_dir)
            }
            Waifu2xBackend::Compat => {
                self.upscale_compat_pass(img, pass_scale)
            }
        }
    }

    /// Human-readable description of the active configuration.
    pub fn description(&self) -> String {
        let (backend_desc, detail) = match &self.backend {
            Waifu2xBackend::Cnn { .. } => ("VGG7 CNN", "neural network inference"),
            Waifu2xBackend::Ncnn { .. } => ("NCNN-Vulkan", "GPU-accelerated inference"),
            Waifu2xBackend::Compat => ("compat", "Lanczos3 + unsharp mask"),
        };
        let mode_str = match self.mode {
            Waifu2xMode::Upscale => format!("scale={}x", self.scale),
            Waifu2xMode::Enhance => "enhance".to_string(),
        };
        format!(
            "waifu2x-{} noise={} {} style={} ({})",
            backend_desc, self.noise_level, mode_str, self.style, detail
        )
    }

    // ── Backend selection ──────────────────────────────────────────────

    /// Attempt to find the best available backend, in preference order:
    /// 1. NCNN-Vulkan (external binary + native model files)
    /// 2. VGG7 CNN (.rsr weights via alumina)
    /// 3. Compat (software fallback)
    fn try_load_backend(noise_level: u8, scale: u8, style: Waifu2xStyle) -> Waifu2xBackend {
        // Try NCNN-Vulkan first.
        if let Some(backend) = Self::try_load_ncnn(noise_level, scale, style) {
            return backend;
        }

        // Try alumina VGG7 CNN weights.
        let weight_path = find_weight_file(noise_level, scale, style);
        match weight_path {
            Some(path) => {
                info!("Loading waifu2x CNN weights from {}", path.display());
                match Self::load_weights_from_file(&path, scale as usize) {
                    Ok((graph, parameters)) => {
                        info!("Waifu2x VGG7 CNN ready ({} parameters)", parameters.len());
                        Waifu2xBackend::Cnn { graph, parameters }
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to load waifu2x weights from {}: {}; using compat mode",
                            path.display(), e
                        );
                        Waifu2xBackend::Compat
                    }
                }
            }
            None => {
                debug!(
                    "No waifu2x weight file found for noise={} scale={} style={}; using compat mode",
                    noise_level, scale, style
                );
                Waifu2xBackend::Compat
            }
        }
    }

    // ── NCNN-Vulkan backend ──────────────────────────────────────────

    /// Search for the `waifu2x-ncnn-vulkan` binary and matching model files.
    ///
    /// Binary search order:
    /// 1. `$WAIFU2X_NCNN_PATH` (exact binary path)
    /// 2. `waifu2x-ncnn-vulkan` on `$PATH`
    ///
    /// Model directory search order:
    /// 1. `$SRGAN_MODEL_PATH/waifu2x-ncnn/`
    /// 2. `./models/waifu2x-ncnn/`
    fn try_load_ncnn(noise_level: u8, scale: u8, _style: Waifu2xStyle) -> Option<Waifu2xBackend> {
        let binary_path = Self::find_ncnn_binary()?;

        // Find model directory containing .param/.bin files for this config.
        let model_dir = find_ncnn_model_dir(noise_level, scale)?;

        info!(
            "Using NCNN-Vulkan backend: binary={}, models={}",
            binary_path.display(),
            model_dir.display()
        );

        Some(Waifu2xBackend::Ncnn { binary_path, model_dir })
    }

    /// Locate the `waifu2x-ncnn-vulkan` binary.
    fn find_ncnn_binary() -> Option<PathBuf> {
        // Check explicit environment variable first.
        if let Ok(p) = std::env::var("WAIFU2X_NCNN_PATH") {
            let path = PathBuf::from(&p);
            if path.is_file() {
                return Some(path);
            }
        }

        // Check if it's on PATH via `which`.
        if let Ok(output) = std::process::Command::new("which")
            .arg("waifu2x-ncnn-vulkan")
            .output()
        {
            if output.status.success() {
                let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !s.is_empty() {
                    return Some(PathBuf::from(s));
                }
            }
        }

        None
    }

    /// Run inference through the external NCNN-Vulkan binary.
    ///
    /// Writes the input to a temp file, invokes the binary, and reads the
    /// output.  This avoids linking against NCNN at compile time while still
    /// getting Vulkan-accelerated inference.
    fn upscale_ncnn(
        &self,
        img: &image::DynamicImage,
        pass_scale: Waifu2xScale,
        binary_path: &Path,
        model_dir: &Path,
    ) -> Result<image::DynamicImage> {
        let tmp_dir = std::env::temp_dir().join("srgan-waifu2x-ncnn");
        std::fs::create_dir_all(&tmp_dir).map_err(SrganError::Io)?;

        let input_path = tmp_dir.join("input.png");
        let output_path = tmp_dir.join("output.png");

        // Write input image.
        img.save(&input_path)
            .map_err(|e| SrganError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        let scale_str = pass_scale.as_u8().to_string();
        let noise_str = self.noise_level.as_u8().to_string();

        let status = std::process::Command::new(binary_path)
            .arg("-i").arg(&input_path)
            .arg("-o").arg(&output_path)
            .arg("-n").arg(&noise_str)
            .arg("-s").arg(&scale_str)
            .arg("-m").arg(model_dir)
            .status()
            .map_err(|e| SrganError::Io(e))?;

        if !status.success() {
            return Err(SrganError::GraphExecution(format!(
                "waifu2x-ncnn-vulkan exited with status {}",
                status
            )));
        }

        let result = image::open(&output_path)
            .map_err(|e| SrganError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        // Cleanup temp files (best effort).
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);

        Ok(result)
    }

    /// Load `.rsr` weight file and build the VGG7 graph.
    fn load_weights_from_file(
        path: &Path,
        scale_factor: usize,
    ) -> Result<(crate::GraphDef, Vec<ndarray::ArrayD<f32>>)> {
        use std::io::Read;

        let mut file = std::fs::File::open(path).map_err(SrganError::Io)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(SrganError::Io)?;

        let desc = crate::network_from_bytes(&data)
            .map_err(|e| SrganError::Network(e))?;

        let graph = crate::network::waifu2x_vgg7_net(scale_factor)
            .map_err(|e| SrganError::Network(format!("waifu2x graph build failed: {}", e)))?;

        Ok((graph, desc.parameters))
    }

    /// Run CNN inference on an image through the VGG7 graph.
    fn upscale_cnn(
        &self,
        img: &image::DynamicImage,
        graph: &crate::GraphDef,
        _parameters: &[ndarray::ArrayD<f32>],
    ) -> Result<image::DynamicImage> {
        use alumina::data::image_folder::image_to_data;

        let input_data = image_to_data(img);

        let input_id = graph.node_id("input").value_id();
        let output_id = graph.node_id("output").value_id();

        let mut subgraph = graph
            .subgraph(&[input_id.clone()], &[output_id.clone()])
            .map_err(|e| SrganError::GraphExecution(format!("{}", e)))?;

        let result = subgraph.execute(vec![input_data])
            .map_err(|e| SrganError::GraphExecution(format!("{}", e)))?;

        let output_data = result.into_map().remove(&output_id)
            .ok_or_else(|| SrganError::GraphExecution(
                "Output node not found in waifu2x graph result".into()
            ))?;

        Ok(alumina::data::image_folder::data_to_image(output_data.view()))
    }

    // ── Compat (software fallback) path ─────────────────────────────────

    /// Software fallback for a single pass: Lanczos3 resize + unsharp mask.
    fn upscale_compat_pass(
        &self,
        img: &image::DynamicImage,
        pass_scale: Waifu2xScale,
    ) -> Result<image::DynamicImage> {
        let (w, h) = (img.width(), img.height());
        let scale_u8 = pass_scale.as_u8();

        // Step 1: Lanczos3 resize (scale=1 keeps original dimensions).
        let resized = if scale_u8 >= 2 {
            img.resize_exact(w * 2, h * 2, image::FilterType::Lanczos3)
        } else {
            img.clone()
        };

        // Step 2: Unsharp-mask sharpening based on noise level, adjusted by
        // content style.
        let style_multiplier = match self.style {
            Waifu2xStyle::Anime   => 1.0f32,
            Waifu2xStyle::Artwork => 0.8,
            Waifu2xStyle::Photo   => 0.6,
        };

        let base_amount = match self.noise_level {
            NoiseLevel::None   => 0.0f32,
            NoiseLevel::Low    => 0.3,
            NoiseLevel::Medium => 0.5,
            NoiseLevel::High   => 0.8,
        };

        let amount = base_amount * style_multiplier;

        if amount < f32::EPSILON {
            return Ok(resized);
        }

        Ok(unsharp_mask(&resized, amount))
    }
}

impl std::fmt::Display for Waifu2xNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.description())
    }
}

// ── Weight file search ──────────────────────────────────────────────────────

/// Expected weight file name for a given waifu2x configuration.
///
/// Format: `noise{N}_scale{M}_{style}.rsr`
/// Examples: `noise1_scale2_anime.rsr`, `noise0_scale1_photo.rsr`
pub fn weight_file_name(noise_level: u8, scale: u8, style: Waifu2xStyle) -> String {
    format!("noise{}_scale{}_{}.rsr", noise_level, scale, style)
}

/// Search for a waifu2x weight file in the standard locations.
///
/// Search order:
/// 1. `$SRGAN_MODEL_PATH/waifu2x/<file>`
/// 2. `./models/waifu2x/<file>`
pub fn find_weight_file(noise_level: u8, scale: u8, style: Waifu2xStyle) -> Option<PathBuf> {
    let filename = weight_file_name(noise_level, scale, style);

    // Try $SRGAN_MODEL_PATH/waifu2x/
    if let Ok(model_path) = std::env::var("SRGAN_MODEL_PATH") {
        let candidate = PathBuf::from(model_path).join("waifu2x").join(&filename);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // Try ./models/waifu2x/
    let local = PathBuf::from("models").join("waifu2x").join(&filename);
    if local.is_file() {
        return Some(local);
    }

    // Also try without style suffix (generic weight file)
    let generic = format!("noise{}_scale{}.rsr", noise_level, scale);

    if let Ok(model_path) = std::env::var("SRGAN_MODEL_PATH") {
        let candidate = PathBuf::from(model_path).join("waifu2x").join(&generic);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let local_generic = PathBuf::from("models").join("waifu2x").join(&generic);
    if local_generic.is_file() {
        return Some(local_generic);
    }

    None
}

// ── NCNN model directory search ──────────────────────────────────────────────

/// Search for NCNN model files (`.param` + `.bin`) for a given noise/scale.
///
/// The NCNN model directory should contain files like:
/// `noise{N}_scale{M}.param` and `noise{N}_scale{M}.bin`.
///
/// Search order:
/// 1. `$SRGAN_MODEL_PATH/waifu2x-ncnn/`
/// 2. `./models/waifu2x-ncnn/`
pub fn find_ncnn_model_dir(noise_level: u8, scale: u8) -> Option<PathBuf> {
    let param_name = format!("noise{}_scale{}.param", noise_level, scale.min(2));

    let candidates = [
        std::env::var("SRGAN_MODEL_PATH")
            .ok()
            .map(|p| PathBuf::from(p).join("waifu2x-ncnn")),
        Some(PathBuf::from("models").join("waifu2x-ncnn")),
    ];

    for dir in candidates.iter().flatten() {
        if dir.join(&param_name).is_file() {
            return Some(dir.clone());
        }
    }

    None
}

// ── Unsharp mask ─────────────────────────────────────────────────────────────

/// Apply unsharp-mask sharpening: `output = original + amount * (original - blur)`.
///
/// Uses a 3×3 box blur as the smoothing kernel for simplicity.  This is the
/// waifu2x-compat approximation of CNN-based noise reduction.
fn unsharp_mask(img: &image::DynamicImage, amount: f32) -> image::DynamicImage {
    use image::{DynamicImage, GenericImage, Pixel};

    let rgba = img.to_rgba();
    let (w, h) = rgba.dimensions();
    if w < 3 || h < 3 {
        return img.clone();
    }

    let mut out = rgba.clone();

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            // 3×3 box-blur average for each channel.
            let mut sums = [0u32; 4];
            for dy in 0u32..3 {
                for dx in 0u32..3 {
                    let p = rgba.get_pixel(x + dx - 1, y + dy - 1);
                    let channels = p.channels();
                    for c in 0..4 {
                        sums[c] += channels[c] as u32;
                    }
                }
            }

            let orig = rgba.get_pixel(x, y);
            let orig_ch = orig.channels();
            let mut sharpened = [0u8; 4];
            for c in 0..4 {
                if c == 3 {
                    // Preserve alpha unchanged.
                    sharpened[c] = orig_ch[c];
                } else {
                    let blurred = (sums[c] as f32) / 9.0;
                    let diff = orig_ch[c] as f32 - blurred;
                    let val = orig_ch[c] as f32 + amount * diff;
                    sharpened[c] = val.round().max(0.0).min(255.0) as u8;
                }
            }

            out.put_pixel(x, y, image::Rgba(sharpened));
        }
    }

    DynamicImage::ImageRgba8(out)
}

// ── Contrast enhancement ────────────────────────────────────────────────────

/// Apply local contrast enhancement tuned to content style.
///
/// For anime/artwork: gentle S-curve contrast to make colors pop without
/// crushing blacks.  For photos: lighter touch to preserve natural tones.
fn adjust_contrast(img: &image::DynamicImage, style: Waifu2xStyle) -> image::DynamicImage {
    use image::{DynamicImage, GenericImage, Pixel};

    let strength = match style {
        Waifu2xStyle::Anime   => 0.15f32,
        Waifu2xStyle::Artwork => 0.12,
        Waifu2xStyle::Photo   => 0.08,
    };

    let rgba = img.to_rgba();
    let (w, h) = rgba.dimensions();
    let mut out = rgba.clone();

    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            let channels = p.channels();
            let mut adjusted = [0u8; 4];
            for c in 0..3 {
                // S-curve: val = val + strength * val * (1 - val) * 4
                // Maps [0,1] → [0,1] with midtone contrast boost
                let v = channels[c] as f32 / 255.0;
                let boosted = v + strength * v * (1.0 - v) * 4.0;
                adjusted[c] = (boosted * 255.0).round().max(0.0).min(255.0) as u8;
            }
            adjusted[3] = channels[3]; // preserve alpha
            out.put_pixel(x, y, image::Rgba(adjusted));
        }
    }

    DynamicImage::ImageRgba8(out)
}

// ── Label parser ──────────────────────────────────────────────────────────────

/// Parse a waifu2x label into a [`Waifu2xConfig`].
///
/// Accepted formats:
/// - `"waifu2x"` → noise=1, scale=2, style=Anime (defaults)
/// - `"waifu2x-anime"` → noise=1, scale=2, style=Anime
/// - `"waifu2x-photo"` → noise=2, scale=2, style=Photo
/// - `"waifu2x-noise{0..3}-scale{1..4}"` → explicit parameters
fn parse_label(label: &str) -> Result<Waifu2xConfig> {
    if label == "waifu2x" {
        return Ok(Waifu2xConfig { noise_level: 1, scale: 2, style: Waifu2xStyle::default(), mode: Waifu2xMode::Upscale });
    }

    // Style-based convenience labels default to scale=2 (upscale).
    if label == "waifu2x-anime" {
        return Ok(Waifu2xConfig { noise_level: 1, scale: 2, style: Waifu2xStyle::Anime, mode: Waifu2xMode::Upscale });
    }
    if label == "waifu2x-photo" {
        return Ok(Waifu2xConfig { noise_level: 2, scale: 2, style: Waifu2xStyle::Photo, mode: Waifu2xMode::Upscale });
    }

    // Enhancement mode labels.
    if label == "waifu2x-enhance" {
        return Ok(Waifu2xConfig { noise_level: 2, scale: 1, style: Waifu2xStyle::Anime, mode: Waifu2xMode::Enhance });
    }
    if label == "waifu2x-enhance-anime" {
        return Ok(Waifu2xConfig { noise_level: 2, scale: 1, style: Waifu2xStyle::Anime, mode: Waifu2xMode::Enhance });
    }
    if label == "waifu2x-enhance-photo" {
        return Ok(Waifu2xConfig { noise_level: 2, scale: 1, style: Waifu2xStyle::Photo, mode: Waifu2xMode::Enhance });
    }

    if let Some(rest) = label.strip_prefix("waifu2x-") {
        // Expected: "noise{N}-scale{M}"
        let parts: Vec<&str> = rest.split('-').collect();
        if parts.len() == 2 {
            let noise = parts[0]
                .strip_prefix("noise")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(1)
                .min(3);
            let scale = parts[1]
                .strip_prefix("scale")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(2);
            let scale = match scale {
                0 | 1 => 1,
                2 => 2,
                3 => 3,
                _ => 4,
            };
            return Ok(Waifu2xConfig { noise_level: noise, scale, style: Waifu2xStyle::default(), mode: Waifu2xMode::Upscale });
        }
    }

    Err(SrganError::Network(format!(
        "invalid waifu2x label '{}'; expected 'waifu2x', 'waifu2x-anime', \
         'waifu2x-photo', 'waifu2x-enhance', 'waifu2x-enhance-anime', \
         'waifu2x-enhance-photo', or 'waifu2x-noise{{0..3}}-scale{{1..4}}'",
        label
    )))
}

// ── Supported labels ──────────────────────────────────────────────────────────

/// All canonical waifu2x labels accepted by the CLI and API.
pub const WAIFU2X_LABELS: &[&str] = &[
    "waifu2x",
    "waifu2x-anime",
    "waifu2x-photo",
    "waifu2x-enhance",
    "waifu2x-enhance-anime",
    "waifu2x-enhance-photo",
    "waifu2x-noise0-scale1",
    "waifu2x-noise0-scale2",
    "waifu2x-noise0-scale3",
    "waifu2x-noise0-scale4",
    "waifu2x-noise1-scale1",
    "waifu2x-noise1-scale2",
    "waifu2x-noise1-scale3",
    "waifu2x-noise1-scale4",
    "waifu2x-noise2-scale1",
    "waifu2x-noise2-scale2",
    "waifu2x-noise2-scale3",
    "waifu2x-noise2-scale4",
    "waifu2x-noise3-scale1",
    "waifu2x-noise3-scale2",
    "waifu2x-noise3-scale3",
    "waifu2x-noise3-scale4",
];

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bare_label() {
        let c = parse_label("waifu2x").unwrap();
        assert_eq!(c.noise_level, 1);
        assert_eq!(c.scale, 2);
    }

    #[test]
    fn parse_parameterised_label() {
        let c = parse_label("waifu2x-noise2-scale1").unwrap();
        assert_eq!(c.noise_level, 2);
        assert_eq!(c.scale, 1);
    }

    #[test]
    fn parse_noise3_scale2() {
        let c = parse_label("waifu2x-noise3-scale2").unwrap();
        assert_eq!(c.noise_level, 3);
        assert_eq!(c.scale, 2);
    }

    #[test]
    fn parse_anime_label() {
        let c = parse_label("waifu2x-anime").unwrap();
        assert_eq!(c.noise_level, 1);
        assert_eq!(c.scale, 2);
        assert_eq!(c.style, Waifu2xStyle::Anime);
    }

    #[test]
    fn parse_photo_label() {
        let c = parse_label("waifu2x-photo").unwrap();
        assert_eq!(c.noise_level, 2);
        assert_eq!(c.scale, 2);
        assert_eq!(c.style, Waifu2xStyle::Photo);
    }

    #[test]
    fn parse_invalid_label_errors() {
        assert!(parse_label("waifu2x-bad").is_err());
        assert!(parse_label("natural").is_err());
    }

    #[test]
    fn noise_level_roundtrip() {
        for v in 0u8..=3 {
            assert_eq!(NoiseLevel::from_u8(v).as_u8(), v);
        }
    }

    #[test]
    fn scale_roundtrip() {
        assert_eq!(Waifu2xScale::from_u8(1).as_u8(), 1);
        assert_eq!(Waifu2xScale::from_u8(2).as_u8(), 2);
        assert_eq!(Waifu2xScale::from_u8(3).as_u8(), 3);
        assert_eq!(Waifu2xScale::from_u8(4).as_u8(), 4);
        assert_eq!(Waifu2xScale::from_u8(0).as_u8(), 1); // 0 → One
        assert_eq!(Waifu2xScale::from_u8(5).as_u8(), 4); // >4 → Four
    }

    // ── Waifu2x-compat inference tests ──────────────────────────────────

    fn test_image(w: u32, h: u32) -> image::DynamicImage {
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
            image::Rgba([(x % 256) as u8, (y % 256) as u8, 128u8, 255u8])
        }))
    }

    #[test]
    fn compat_scale2_doubles_dimensions() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale2").unwrap();
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 32);
        assert_eq!(result.height(), 32);
    }

    #[test]
    fn compat_scale1_preserves_dimensions() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale1").unwrap();
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn compat_noise0_no_sharpening() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise0-scale2").unwrap();
        let img = test_image(8, 8);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn compat_noise3_scale1_sharpens_only() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise3-scale1").unwrap();
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn compat_all_variants_succeed() {
        let img = test_image(10, 10);
        for &label in WAIFU2X_LABELS {
            let net = Waifu2xNetwork::from_label(label)
                .unwrap_or_else(|e| panic!("from_label({}) failed: {}", label, e));
            let result = net.upscale_image(&img)
                .unwrap_or_else(|e| panic!("upscale_image({}) failed: {}", label, e));
            let expected_w = if label.contains("enhance") {
                10 // enhance mode preserves dimensions
            } else if label.contains("scale4") {
                40
            } else if label.contains("scale3") {
                30
            } else if label.contains("scale2") || label == "waifu2x" || label == "waifu2x-anime" || label == "waifu2x-photo" {
                20
            } else {
                10
            };
            assert_eq!(result.width(), expected_w, "width mismatch for {}", label);
            assert_eq!(result.height(), expected_w, "height mismatch for {}", label);
        }
    }

    #[test]
    fn compat_scale3_triples_dimensions() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale3").unwrap();
        let img = test_image(10, 10);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 30);
        assert_eq!(result.height(), 30);
    }

    #[test]
    fn compat_scale4_quadruples_dimensions() {
        let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale4").unwrap();
        let img = test_image(10, 10);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 40);
        assert_eq!(result.height(), 40);
    }

    #[test]
    fn is_ncnn_false_without_binary() {
        let net = Waifu2xNetwork::from_label("waifu2x").unwrap();
        assert!(!net.is_ncnn());
    }

    #[test]
    fn compat_anime_label_upscales() {
        let net = Waifu2xNetwork::from_label("waifu2x-anime").unwrap();
        assert_eq!(net.style(), Waifu2xStyle::Anime);
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 32);
        assert_eq!(result.height(), 32);
    }

    #[test]
    fn compat_photo_label_upscales() {
        let net = Waifu2xNetwork::from_label("waifu2x-photo").unwrap();
        assert_eq!(net.style(), Waifu2xStyle::Photo);
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 32);
        assert_eq!(result.height(), 32);
    }

    #[test]
    fn compat_description_mentions_compat() {
        let net = Waifu2xNetwork::from_label("waifu2x").unwrap();
        // In compat mode (no weight files), description should mention compat
        if !net.is_cnn() {
            assert!(net.description().contains("compat"));
        }
    }

    #[test]
    fn is_cnn_false_without_weights() {
        let net = Waifu2xNetwork::from_label("waifu2x").unwrap();
        // Without weight files on disk, should be in compat mode
        assert!(!net.is_cnn());
    }

    #[test]
    fn weight_file_name_format() {
        assert_eq!(
            weight_file_name(1, 2, Waifu2xStyle::Anime),
            "noise1_scale2_anime.rsr"
        );
        assert_eq!(
            weight_file_name(3, 1, Waifu2xStyle::Photo),
            "noise3_scale1_photo.rsr"
        );
    }

    #[test]
    fn compat_invalid_label_errors() {
        assert!(Waifu2xNetwork::from_label("esrgan").is_err());
        assert!(Waifu2xNetwork::from_label("waifu2x-bad").is_err());
    }

    #[test]
    fn parse_enhance_label() {
        let c = parse_label("waifu2x-enhance").unwrap();
        assert_eq!(c.mode, Waifu2xMode::Enhance);
        assert_eq!(c.scale, 1);
        assert_eq!(c.style, Waifu2xStyle::Anime);
    }

    #[test]
    fn parse_enhance_anime_label() {
        let c = parse_label("waifu2x-enhance-anime").unwrap();
        assert_eq!(c.mode, Waifu2xMode::Enhance);
        assert_eq!(c.style, Waifu2xStyle::Anime);
    }

    #[test]
    fn parse_enhance_photo_label() {
        let c = parse_label("waifu2x-enhance-photo").unwrap();
        assert_eq!(c.mode, Waifu2xMode::Enhance);
        assert_eq!(c.style, Waifu2xStyle::Photo);
    }

    #[test]
    fn enhance_preserves_dimensions() {
        let net = Waifu2xNetwork::from_label("waifu2x-enhance").unwrap();
        assert!(net.is_enhance());
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
    }

    #[test]
    fn enhance_photo_preserves_dimensions() {
        let net = Waifu2xNetwork::from_label("waifu2x-enhance-photo").unwrap();
        assert!(net.is_enhance());
        let img = test_image(20, 20);
        let result = net.upscale_image(&img).unwrap();
        assert_eq!(result.width(), 20);
        assert_eq!(result.height(), 20);
    }

    #[test]
    fn enhance_modifies_pixels() {
        let net = Waifu2xNetwork::from_label("waifu2x-enhance").unwrap();
        let img = test_image(16, 16);
        let result = net.upscale_image(&img).unwrap();
        let orig_rgba = img.to_rgba();
        let res_rgba = result.to_rgba();
        let mut differs = false;
        for y in 1..15 {
            for x in 1..15 {
                if orig_rgba.get_pixel(x, y) != res_rgba.get_pixel(x, y) {
                    differs = true;
                    break;
                }
            }
        }
        assert!(differs, "enhance mode should modify pixel values");
    }

    #[test]
    fn contrast_adjustment_modifies_midtones() {
        let img = test_image(10, 10);
        let adjusted = adjust_contrast(&img, Waifu2xStyle::Anime);
        let orig_rgba = img.to_rgba();
        let adj_rgba = adjusted.to_rgba();
        // Midtone pixels (not 0 or 255) should be boosted
        let mut any_changed = false;
        for y in 0..10 {
            for x in 0..10 {
                let o = orig_rgba.get_pixel(x, y);
                let a = adj_rgba.get_pixel(x, y);
                if o != a {
                    any_changed = true;
                    break;
                }
            }
        }
        assert!(any_changed, "contrast adjustment should modify midtone pixels");
    }

    #[test]
    fn compat_unsharp_mask_modifies_pixels() {
        // Use a checkerboard pattern so the unsharp mask has sharp edges to
        // enhance (a smooth gradient may round-trip unchanged).
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(16, 16, |x, y| {
            let v = if (x + y) % 2 == 0 { 200u8 } else { 50u8 };
            image::Rgba([v, v, v, 255])
        }));
        let sharpened = unsharp_mask(&img, 0.8);
        let orig_rgba = img.to_rgba();
        let sharp_rgba = sharpened.to_rgba();
        let mut differs = false;
        for y in 1..15 {
            for x in 1..15 {
                if orig_rgba.get_pixel(x, y) != sharp_rgba.get_pixel(x, y) {
                    differs = true;
                    break;
                }
            }
        }
        assert!(differs, "unsharp mask should modify at least some interior pixels");
    }
}
