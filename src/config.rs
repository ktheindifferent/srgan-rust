use crate::constants::{network, training};
use crate::error::{Result, SrganError};

#[derive(Debug, Clone)]
pub struct NetworkConfig {
	pub factor: usize,
	pub width: u32,
	pub log_depth: u32,
	pub global_node_factor: usize,
}

impl Default for NetworkConfig {
	fn default() -> Self {
		Self {
			factor: network::DEFAULT_FACTOR,
			width: network::DEFAULT_WIDTH,
			log_depth: network::DEFAULT_LOG_DEPTH,
			global_node_factor: network::DEFAULT_GLOBAL_NODE_FACTOR,
		}
	}
}

impl NetworkConfig {
	pub fn builder() -> NetworkConfigBuilder {
		NetworkConfigBuilder::default()
	}

	pub fn validate(&self) -> Result<()> {
		if self.factor == 0 {
			return Err(SrganError::InvalidParameter(
				"Factor must be greater than 0".into(),
			));
		}
		if self.width == 0 {
			return Err(SrganError::InvalidParameter("Width must be greater than 0".into()));
		}
		if self.log_depth == 0 {
			return Err(SrganError::InvalidParameter(
				"Log depth must be greater than 0".into(),
			));
		}
		Ok(())
	}
}

#[derive(Default)]
pub struct NetworkConfigBuilder {
	factor: Option<usize>,
	width: Option<u32>,
	log_depth: Option<u32>,
	global_node_factor: Option<usize>,
}

impl NetworkConfigBuilder {
	pub fn factor(mut self, factor: usize) -> Self {
		self.factor = Some(factor);
		self
	}

	pub fn width(mut self, width: u32) -> Self {
		self.width = Some(width);
		self
	}

	pub fn log_depth(mut self, log_depth: u32) -> Self {
		self.log_depth = Some(log_depth);
		self
	}

	pub fn global_node_factor(mut self, global_node_factor: usize) -> Self {
		self.global_node_factor = Some(global_node_factor);
		self
	}

	pub fn build(self) -> NetworkConfig {
		NetworkConfig {
			factor: self.factor.unwrap_or(network::DEFAULT_FACTOR),
			width: self.width.unwrap_or(network::DEFAULT_WIDTH),
			log_depth: self.log_depth.unwrap_or(network::DEFAULT_LOG_DEPTH),
			global_node_factor: self.global_node_factor.unwrap_or(network::DEFAULT_GLOBAL_NODE_FACTOR),
		}
	}
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
	pub learning_rate: f32,
	pub patch_size: usize,
	pub batch_size: usize,
	pub loss_type: LossType,
	pub srgb_downscale: bool,
	pub quantise: bool,
	pub recurse: bool,
	pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
	fn default() -> Self {
		Self {
			learning_rate: training::DEFAULT_LEARNING_RATE,
			patch_size: training::DEFAULT_PATCH_SIZE,
			batch_size: training::DEFAULT_BATCH_SIZE,
			loss_type: LossType::L1,
			srgb_downscale: true,
			quantise: false,
			recurse: false,
			checkpoint_interval: training::CHECKPOINT_INTERVAL,
		}
	}
}

impl TrainingConfig {
	pub fn builder() -> TrainingConfigBuilder {
		TrainingConfigBuilder::default()
	}

	pub fn validate(&self) -> Result<()> {
		if self.learning_rate <= 0.0 {
			return Err(SrganError::InvalidParameter(format!(
				"Learning rate ({}) must be greater than 0",
				self.learning_rate
			)));
		}
		if self.patch_size == 0 {
			return Err(SrganError::InvalidParameter(format!(
				"Patch size ({}) must be greater than 0",
				self.patch_size
			)));
		}
		if self.batch_size == 0 {
			return Err(SrganError::InvalidParameter(format!(
				"Batch size ({}) must be greater than 0",
				self.batch_size
			)));
		}
		Ok(())
	}

	pub fn get_loss_params(&self) -> (f32, f32) {
		match self.loss_type {
			LossType::L1 => (training::L1_LOSS_POWER, training::L1_LOSS_SCALE),
			LossType::L2 => (training::L2_LOSS_POWER, training::L2_LOSS_SCALE),
		}
	}
}

#[derive(Default)]
pub struct TrainingConfigBuilder {
	learning_rate: Option<f32>,
	patch_size: Option<usize>,
	batch_size: Option<usize>,
	loss_type: Option<LossType>,
	srgb_downscale: Option<bool>,
	quantise: Option<bool>,
	recurse: Option<bool>,
	checkpoint_interval: Option<usize>,
}

impl TrainingConfigBuilder {
	pub fn learning_rate(mut self, rate: f32) -> Self {
		self.learning_rate = Some(rate);
		self
	}

	pub fn patch_size(mut self, size: usize) -> Self {
		self.patch_size = Some(size);
		self
	}

	pub fn batch_size(mut self, size: usize) -> Self {
		self.batch_size = Some(size);
		self
	}

	pub fn loss_type(mut self, loss_type: LossType) -> Self {
		self.loss_type = Some(loss_type);
		self
	}

	pub fn srgb_downscale(mut self, srgb: bool) -> Self {
		self.srgb_downscale = Some(srgb);
		self
	}

	pub fn quantise(mut self, quantise: bool) -> Self {
		self.quantise = Some(quantise);
		self
	}

	pub fn recurse(mut self, recurse: bool) -> Self {
		self.recurse = Some(recurse);
		self
	}

	pub fn checkpoint_interval(mut self, interval: usize) -> Self {
		self.checkpoint_interval = Some(interval);
		self
	}

	pub fn build(self) -> TrainingConfig {
		TrainingConfig {
			learning_rate: self.learning_rate.unwrap_or(training::DEFAULT_LEARNING_RATE),
			patch_size: self.patch_size.unwrap_or(training::DEFAULT_PATCH_SIZE),
			batch_size: self.batch_size.unwrap_or(training::DEFAULT_BATCH_SIZE),
			loss_type: self.loss_type.unwrap_or(LossType::L1),
			srgb_downscale: self.srgb_downscale.unwrap_or(true),
			quantise: self.quantise.unwrap_or(false),
			recurse: self.recurse.unwrap_or(false),
			checkpoint_interval: self.checkpoint_interval.unwrap_or(training::CHECKPOINT_INTERVAL),
		}
	}
}

#[derive(Debug, Clone, Copy)]
pub enum LossType {
	L1,
	L2,
}

impl LossType {
	pub fn from_str(s: &str) -> Result<Self> {
		match s {
			"L1" => Ok(LossType::L1),
			"L2" => Ok(LossType::L2),
			_ => Err(SrganError::InvalidParameter(format!("Unknown loss type: {}", s))),
		}
	}
}

/// Waifu2x content style hint.
///
/// Different waifu2x model weights are trained on different content types.
/// The style parameter selects the weight set best suited for the input
/// image.  When native waifu2x weights are available, this controls which
/// weight file is loaded; in the current software-fallback path it adjusts
/// the sharpening profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waifu2xStyle {
    /// Optimised for anime and manga illustrations (default).
    Anime,
    /// Optimised for photographs and realistic images.
    Photo,
    /// Optimised for digital artwork, paintings, and mixed media.
    Artwork,
}

impl Default for Waifu2xStyle {
    fn default() -> Self {
        Waifu2xStyle::Anime
    }
}

impl Waifu2xStyle {
    /// Parse from a string (case-insensitive).
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "anime" => Ok(Waifu2xStyle::Anime),
            "photo" => Ok(Waifu2xStyle::Photo),
            "artwork" => Ok(Waifu2xStyle::Artwork),
            _ => Err(SrganError::InvalidParameter(format!(
                "Waifu2x style must be 'anime', 'photo', or 'artwork', got '{}'",
                s
            ))),
        }
    }

    /// Return the canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Waifu2xStyle::Anime => "anime",
            Waifu2xStyle::Photo => "photo",
            Waifu2xStyle::Artwork => "artwork",
        }
    }
}

impl std::fmt::Display for Waifu2xStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Waifu2x-specific configuration options.
///
/// Waifu2x supports noise reduction (levels 0–3), two scale factors, and
/// a content style hint (anime, photo, artwork).  These options are passed
/// alongside the model label `"waifu2x"` and map to the label variants
/// used by `UpscalingNetwork::from_label` / `ThreadSafeNetwork::from_label`.
#[derive(Debug, Clone)]
pub struct Waifu2xConfig {
    /// Noise-reduction level (0 = none, 1 = light, 2 = medium, 3 = aggressive).
    pub noise_level: u8,
    /// Upscaling factor (1 or 2).
    pub scale: u8,
    /// Content style hint for weight selection.
    pub style: Waifu2xStyle,
}

impl Default for Waifu2xConfig {
    fn default() -> Self {
        Self {
            noise_level: 1,
            scale: 2,
            style: Waifu2xStyle::default(),
        }
    }
}

impl Waifu2xConfig {
    pub fn new(noise_level: u8, scale: u8) -> Result<Self> {
        if noise_level > 3 {
            return Err(SrganError::InvalidParameter(format!(
                "Waifu2x noise_level must be 0–3, got {}",
                noise_level
            )));
        }
        if scale != 1 && scale != 2 {
            return Err(SrganError::InvalidParameter(format!(
                "Waifu2x scale must be 1 or 2, got {}",
                scale
            )));
        }
        Ok(Self { noise_level, scale, style: Waifu2xStyle::default() })
    }

    /// Build with an explicit style.
    pub fn with_style(noise_level: u8, scale: u8, style: Waifu2xStyle) -> Result<Self> {
        let mut cfg = Self::new(noise_level, scale)?;
        cfg.style = style;
        Ok(cfg)
    }

    /// Return the model-label string understood by `from_label`.
    ///
    /// Examples: `"waifu2x-noise1-scale2"`, `"waifu2x-noise0-scale1"`.
    pub fn model_label(&self) -> String {
        format!("waifu2x-noise{}-scale{}", self.noise_level, self.scale)
    }
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
	pub folder: Option<String>,
	pub max_images: Option<usize>,
	pub recurse: bool,
}

impl ValidationConfig {
	pub fn new(folder: Option<String>, max_images: Option<usize>, recurse: bool) -> Self {
		Self {
			folder,
			max_images,
			recurse,
		}
	}
}
