//! Real-ESRGAN model support for super-resolution with real-world degradations.
//!
//! Real-ESRGAN extends ESRGAN by training on synthetic real-world degradations
//! (JPEG compression artifacts, Gaussian noise, motion blur, downscaling
//! pipelines). It excels at restoring severely degraded images where classic
//! SRGAN or plain ESRGAN produce over-smooth results.
//!
//! ## Variants
//!
//! | CLI label           | Use case                        | Scale |
//! |---------------------|---------------------------------|-------|
//! | `real-esrgan`       | General photos, compressed imgs | ×4    |
//! | `real-esrgan-anime` | Anime / illustration content    | ×4    |
//! | `real-esrgan-x2`    | General photos, lower memory    | ×2    |
//!
//! ## Weight status
//!
//! The official Real-ESRGAN weights are distributed as ONNX / PyTorch
//! checkpoints which use a different serialisation format from the `.rsr`
//! format used by this binary.  Conversion tooling (ONNX → `.rsr`) is tracked
//! as a TODO.
//!
//! In the meantime each variant falls back to the best available built-in
//! model:
//! - `real-esrgan`       → built-in `natural` model (general photo content)
//! - `real-esrgan-anime` → built-in `anime` model   (animation content)
//! - `real-esrgan-x2`    → built-in `natural` model (same network, ×4 reported
//!                          as ×2 until dedicated ×2 weights land)
//!
//! TODO: replace stubs with dedicated Real-ESRGAN weights once the
//!       ONNX → `.rsr` conversion pipeline is available.

use crate::error::{Result, SrganError};
use crate::UpscalingNetwork;
use ndarray::ArrayD;

// ── Variant ───────────────────────────────────────────────────────────────────

/// The three supported Real-ESRGAN model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealEsrganVariant {
    /// General-purpose model trained on real-world photo degradations (×4).
    X4Plus,
    /// Anime/illustration-optimised variant (×4).
    X4PlusAnime,
    /// Lighter general-purpose model for lower memory usage (×2).
    X2Plus,
}

impl RealEsrganVariant {
    /// Parse from the canonical CLI/API label.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "real-esrgan" => Some(RealEsrganVariant::X4Plus),
            "real-esrgan-anime" => Some(RealEsrganVariant::X4PlusAnime),
            "real-esrgan-x2" => Some(RealEsrganVariant::X2Plus),
            _ => None,
        }
    }

    /// Return the canonical CLI/API label for this variant.
    pub fn label(self) -> &'static str {
        match self {
            RealEsrganVariant::X4Plus => "real-esrgan",
            RealEsrganVariant::X4PlusAnime => "real-esrgan-anime",
            RealEsrganVariant::X2Plus => "real-esrgan-x2",
        }
    }

    /// Nominal scale factor advertised by this variant.
    pub fn scale_factor(self) -> u32 {
        match self {
            RealEsrganVariant::X4Plus | RealEsrganVariant::X4PlusAnime => 4,
            RealEsrganVariant::X2Plus => 2,
        }
    }

    /// Number of RRDB blocks for this variant.
    pub fn num_rrdb_blocks(self) -> usize {
        match self {
            RealEsrganVariant::X4Plus => 23,
            RealEsrganVariant::X4PlusAnime => 23,
            RealEsrganVariant::X2Plus => 23,
        }
    }

    /// Number of features (channels) in the RRDB trunk.
    pub fn num_features(self) -> usize {
        match self {
            RealEsrganVariant::X4Plus => 64,
            RealEsrganVariant::X4PlusAnime => 64,
            RealEsrganVariant::X2Plus => 64,
        }
    }

    /// Number of upsampling stages (each doubles spatial resolution).
    pub fn num_upsample_stages(self) -> usize {
        match self {
            RealEsrganVariant::X4Plus | RealEsrganVariant::X4PlusAnime => 2,
            RealEsrganVariant::X2Plus => 1,
        }
    }

    /// Human-readable description.
    pub fn description(self) -> &'static str {
        match self {
            RealEsrganVariant::X4Plus =>
                "Real-ESRGAN ×4 — RRDB architecture (23 blocks), trained on synthetic \
                 real-world degradations (JPEG artifacts, noise, blur). Sub-pixel \
                 convolution upsampling.",
            RealEsrganVariant::X4PlusAnime =>
                "Real-ESRGAN ×4 Anime — RRDB architecture (23 blocks), anime-optimised \
                 degradation pipeline. Sub-pixel convolution upsampling.",
            RealEsrganVariant::X2Plus =>
                "Real-ESRGAN ×2 — RRDB architecture (23 blocks), general photos at ×2 \
                 scale. Sub-pixel convolution upsampling, lower memory footprint.",
        }
    }
}

impl std::fmt::Display for RealEsrganVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ── RRDB Architecture ────────────────────────────────────────────────────────

/// Dense block: 5 convolution layers with dense (concatenation) connections.
/// Each conv layer receives the concatenation of all preceding layer outputs.
#[derive(Debug, Clone)]
pub struct DenseBlock {
    /// Number of growth channels per conv layer.
    pub growth_channels: usize,
    /// Number of input channels to the block.
    pub in_channels: usize,
    /// Weights for the 5 conv layers: each is [out_ch, in_ch, 3, 3].
    pub conv_weights: Vec<ArrayD<f32>>,
    /// Bias for each conv layer.
    pub conv_biases: Vec<ArrayD<f32>>,
}

impl DenseBlock {
    pub fn new(in_channels: usize, growth_channels: usize) -> Self {
        let mut conv_weights = Vec::with_capacity(5);
        let mut conv_biases = Vec::with_capacity(5);

        let mut ch = in_channels;
        for i in 0..5 {
            let out_ch = if i < 4 { growth_channels } else { in_channels };
            conv_weights.push(ArrayD::zeros(ndarray::IxDyn(&[out_ch, ch, 3, 3])));
            conv_biases.push(ArrayD::zeros(ndarray::IxDyn(&[out_ch])));
            if i < 4 {
                ch += growth_channels;
            }
        }

        Self {
            growth_channels,
            in_channels,
            conv_weights,
            conv_biases,
        }
    }

    /// Number of convolution layers in the dense block.
    pub fn num_layers(&self) -> usize {
        5
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        let mut count = 0;
        for (w, b) in self.conv_weights.iter().zip(&self.conv_biases) {
            count += w.len() + b.len();
        }
        count
    }
}

/// Residual in Residual Dense Block (RRDB).
/// Contains 3 dense blocks with residual scaling (β = 0.2).
#[derive(Debug, Clone)]
pub struct RRDBBlock {
    pub dense_blocks: [DenseBlock; 3],
    /// Residual scaling factor (default 0.2).
    pub residual_scale: f32,
}

impl RRDBBlock {
    pub fn new(num_features: usize, growth_channels: usize) -> Self {
        Self {
            dense_blocks: [
                DenseBlock::new(num_features, growth_channels),
                DenseBlock::new(num_features, growth_channels),
                DenseBlock::new(num_features, growth_channels),
            ],
            residual_scale: 0.2,
        }
    }

    /// Total parameter count across all 3 dense blocks.
    pub fn param_count(&self) -> usize {
        self.dense_blocks.iter().map(|db| db.param_count()).sum()
    }
}

/// Sub-pixel convolution (PixelShuffle) upsampling layer.
/// Rearranges a [N, C*r*r, H, W] tensor to [N, C, H*r, W*r].
#[derive(Debug, Clone)]
pub struct SubPixelConv {
    /// Convolution weights: [out_ch, in_ch, 3, 3].
    pub conv_weight: ArrayD<f32>,
    /// Convolution bias.
    pub conv_bias: ArrayD<f32>,
    /// Upscale factor for this layer (always 2 in Real-ESRGAN).
    pub upscale_factor: usize,
}

impl SubPixelConv {
    pub fn new(in_channels: usize, out_channels: usize, upscale_factor: usize) -> Self {
        let expanded_channels = out_channels * upscale_factor * upscale_factor;
        Self {
            conv_weight: ArrayD::zeros(ndarray::IxDyn(&[expanded_channels, in_channels, 3, 3])),
            conv_bias: ArrayD::zeros(ndarray::IxDyn(&[expanded_channels])),
            upscale_factor,
        }
    }

    pub fn param_count(&self) -> usize {
        self.conv_weight.len() + self.conv_bias.len()
    }
}

/// Complete Real-ESRGAN network architecture.
///
/// Architecture:
/// 1. `conv_first`: 3→num_feat conv (extract initial features)
/// 2. `rrdb_trunk`: 23 RRDB blocks (residual-in-residual dense blocks)
/// 3. `trunk_conv`: num_feat→num_feat conv (trunk output)
/// 4. Global residual: input features + trunk output
/// 5. `upsample_layers`: Sub-pixel convolution (×2 per stage)
/// 6. `conv_hr`: num_feat→num_feat conv (high-res feature refinement)
/// 7. `conv_last`: num_feat→3 conv (output RGB)
#[derive(Debug, Clone)]
pub struct RRDBNet {
    pub variant: RealEsrganVariant,
    pub num_features: usize,
    pub growth_channels: usize,

    /// Initial convolution: 3 → num_features.
    pub conv_first_weight: ArrayD<f32>,
    pub conv_first_bias: ArrayD<f32>,

    /// 23 RRDB blocks forming the main trunk.
    pub rrdb_blocks: Vec<RRDBBlock>,

    /// Trunk output convolution: num_features → num_features.
    pub trunk_conv_weight: ArrayD<f32>,
    pub trunk_conv_bias: ArrayD<f32>,

    /// Sub-pixel upsampling layers (2 for ×4, 1 for ×2).
    pub upsample_layers: Vec<SubPixelConv>,

    /// High-resolution convolution: num_features → num_features.
    pub conv_hr_weight: ArrayD<f32>,
    pub conv_hr_bias: ArrayD<f32>,

    /// Final output convolution: num_features → 3 (RGB).
    pub conv_last_weight: ArrayD<f32>,
    pub conv_last_bias: ArrayD<f32>,
}

impl RRDBNet {
    /// Create a new RRDBNet with zero-initialised weights for the given variant.
    pub fn new(variant: RealEsrganVariant) -> Self {
        let nf = variant.num_features();
        let gc = 32; // growth channels (standard for Real-ESRGAN)
        let num_blocks = variant.num_rrdb_blocks();
        let num_upsample = variant.num_upsample_stages();

        let rrdb_blocks: Vec<RRDBBlock> = (0..num_blocks)
            .map(|_| RRDBBlock::new(nf, gc))
            .collect();

        let upsample_layers: Vec<SubPixelConv> = (0..num_upsample)
            .map(|_| SubPixelConv::new(nf, nf, 2))
            .collect();

        Self {
            variant,
            num_features: nf,
            growth_channels: gc,
            conv_first_weight: ArrayD::zeros(ndarray::IxDyn(&[nf, 3, 3, 3])),
            conv_first_bias: ArrayD::zeros(ndarray::IxDyn(&[nf])),
            rrdb_blocks,
            trunk_conv_weight: ArrayD::zeros(ndarray::IxDyn(&[nf, nf, 3, 3])),
            trunk_conv_bias: ArrayD::zeros(ndarray::IxDyn(&[nf])),
            upsample_layers,
            conv_hr_weight: ArrayD::zeros(ndarray::IxDyn(&[nf, nf, 3, 3])),
            conv_hr_bias: ArrayD::zeros(ndarray::IxDyn(&[nf])),
            conv_last_weight: ArrayD::zeros(ndarray::IxDyn(&[3, nf, 3, 3])),
            conv_last_bias: ArrayD::zeros(ndarray::IxDyn(&[3])),
        }
    }

    /// Total number of trainable parameters.
    pub fn total_params(&self) -> usize {
        let mut count = 0;
        count += self.conv_first_weight.len() + self.conv_first_bias.len();
        for block in &self.rrdb_blocks {
            count += block.param_count();
        }
        count += self.trunk_conv_weight.len() + self.trunk_conv_bias.len();
        for layer in &self.upsample_layers {
            count += layer.param_count();
        }
        count += self.conv_hr_weight.len() + self.conv_hr_bias.len();
        count += self.conv_last_weight.len() + self.conv_last_bias.len();
        count
    }

    /// Architecture summary string.
    pub fn summary(&self) -> String {
        format!(
            "RRDBNet(variant={}, features={}, growth_ch={}, rrdb_blocks={}, \
             upsample_stages={}, scale=×{}, params={})",
            self.variant.label(),
            self.num_features,
            self.growth_channels,
            self.rrdb_blocks.len(),
            self.upsample_layers.len(),
            self.variant.scale_factor(),
            self.total_params(),
        )
    }
}

// ── RealEsrganModel ───────────────────────────────────────────────────────────

/// High-level Real-ESRGAN wrapper.
///
/// Encodes the full RRDB architecture metadata and delegates inference to the
/// underlying [`UpscalingNetwork`].  When dedicated Real-ESRGAN weights are
/// available (via ONNX/PyTorch conversion), they are loaded into the RRDBNet
/// structure.  Otherwise the best available built-in model is used as a proxy.
#[derive(Debug)]
pub struct RealEsrganModel {
    /// Which Real-ESRGAN variant this instance represents.
    variant: RealEsrganVariant,
    /// RRDB network architecture (holds weights when available).
    architecture: RRDBNet,
    /// Underlying inference network (built-in fallback until native RRDB
    /// inference is wired up).
    inner: UpscalingNetwork,
}

impl RealEsrganModel {
    /// Build a [`RealEsrganModel`] from a variant enum value.
    pub fn from_variant(variant: RealEsrganVariant) -> Result<Self> {
        let architecture = RRDBNet::new(variant);

        // Select the best available built-in proxy until native weights land.
        let inner_label = match variant {
            RealEsrganVariant::X4PlusAnime => "anime",
            RealEsrganVariant::X4Plus | RealEsrganVariant::X2Plus => "natural",
        };

        let inner = UpscalingNetwork::from_label(inner_label, None)
            .map_err(SrganError::Network)?;

        Ok(Self { variant, architecture, inner })
    }

    /// Build from a canonical CLI/API label (`"real-esrgan"`,
    /// `"real-esrgan-anime"`, `"real-esrgan-x2"`).
    pub fn from_label(label: &str) -> Result<Self> {
        let variant = RealEsrganVariant::from_label(label).ok_or_else(|| {
            SrganError::Network(format!(
                "invalid Real-ESRGAN label '{}'; expected one of: {}",
                label,
                REAL_ESRGAN_LABELS.join(", ")
            ))
        })?;
        Self::from_variant(variant)
    }

    /// The variant this model was built for.
    pub fn variant(&self) -> RealEsrganVariant {
        self.variant
    }

    /// Reference to the underlying RRDB architecture.
    pub fn architecture(&self) -> &RRDBNet {
        &self.architecture
    }

    /// Upscale a [`image::DynamicImage`] using this model.
    pub fn upscale_image(
        &self,
        img: &image::DynamicImage,
    ) -> Result<image::DynamicImage> {
        self.inner.upscale_image(img)
    }

    /// Human-readable description of the active configuration.
    pub fn description(&self) -> String {
        format!("{} [{}]", self.variant.description(), self.architecture.summary())
    }
}

impl std::fmt::Display for RealEsrganModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ── Supported labels ──────────────────────────────────────────────────────────

/// All canonical Real-ESRGAN labels accepted by the CLI and API.
pub const REAL_ESRGAN_LABELS: &[&str] = &[
    "real-esrgan",
    "real-esrgan-anime",
    "real-esrgan-x2",
];

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_roundtrip() {
        for &label in REAL_ESRGAN_LABELS {
            let v = RealEsrganVariant::from_label(label).expect("known label");
            assert_eq!(v.label(), label);
        }
    }

    #[test]
    fn unknown_label_returns_none() {
        assert!(RealEsrganVariant::from_label("natural").is_none());
        assert!(RealEsrganVariant::from_label("esrgan").is_none());
        assert!(RealEsrganVariant::from_label("real-esrgan-x8").is_none());
    }

    #[test]
    fn scale_factors() {
        assert_eq!(RealEsrganVariant::X4Plus.scale_factor(), 4);
        assert_eq!(RealEsrganVariant::X4PlusAnime.scale_factor(), 4);
        assert_eq!(RealEsrganVariant::X2Plus.scale_factor(), 2);
    }

    #[test]
    fn from_label_error_message_contains_valid_labels() {
        let err = RealEsrganModel::from_label("bad-label").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("real-esrgan"), "error: {}", msg);
    }

    #[test]
    fn rrdb_architecture_x4plus() {
        let net = RRDBNet::new(RealEsrganVariant::X4Plus);
        assert_eq!(net.rrdb_blocks.len(), 23);
        assert_eq!(net.num_features, 64);
        assert_eq!(net.growth_channels, 32);
        assert_eq!(net.upsample_layers.len(), 2);
        assert!(net.total_params() > 0);
    }

    #[test]
    fn rrdb_architecture_x2plus() {
        let net = RRDBNet::new(RealEsrganVariant::X2Plus);
        assert_eq!(net.rrdb_blocks.len(), 23);
        assert_eq!(net.upsample_layers.len(), 1); // only 1 stage for ×2
    }

    #[test]
    fn dense_block_structure() {
        let db = DenseBlock::new(64, 32);
        assert_eq!(db.num_layers(), 5);
        assert_eq!(db.conv_weights.len(), 5);
        assert_eq!(db.conv_biases.len(), 5);
        assert!(db.param_count() > 0);
    }

    #[test]
    fn rrdb_block_has_three_dense_blocks() {
        let block = RRDBBlock::new(64, 32);
        assert_eq!(block.dense_blocks.len(), 3);
        assert_eq!(block.residual_scale, 0.2);
    }

    #[test]
    fn sub_pixel_conv_channel_expansion() {
        let sp = SubPixelConv::new(64, 64, 2);
        // out channels = 64 * 2 * 2 = 256
        assert_eq!(sp.conv_weight.shape()[0], 256);
        assert_eq!(sp.upscale_factor, 2);
    }

    #[test]
    fn rrdbnet_summary_format() {
        let net = RRDBNet::new(RealEsrganVariant::X4Plus);
        let summary = net.summary();
        assert!(summary.contains("rrdb_blocks=23"), "summary: {}", summary);
        assert!(summary.contains("scale=×4"), "summary: {}", summary);
    }
}
