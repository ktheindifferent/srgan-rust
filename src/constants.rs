pub mod network {
	pub const DEFAULT_FACTOR: usize = 4;
	pub const DEFAULT_WIDTH: u32 = 16;
	pub const DEFAULT_LOG_DEPTH: u32 = 4;
	pub const DEFAULT_GLOBAL_NODE_FACTOR: usize = 2;
	pub const CHANNELS: usize = 3;
	pub const WEIGHT_INIT_PREDICTION_LAYER: f32 = 0.01;
	pub const WEIGHT_INIT_HIDDEN_LAYER: f32 = 1.0;
	pub const WEIGHT_INIT_SECONDARY: f32 = 0.5;
}

pub mod training {
	pub const DEFAULT_LEARNING_RATE: f32 = 3e-3;
	pub const DEFAULT_PATCH_SIZE: usize = 48;
	pub const DEFAULT_BATCH_SIZE: usize = 4;
	pub const CHECKPOINT_INTERVAL: usize = 1000;
	pub const L1_LOSS_SCALE: f32 = 1.0 / 255.0;
	pub const L2_LOSS_SCALE: f32 = 1.0 / 255.0;
	pub const L1_LOSS_POWER: f32 = 1.0;
	pub const L2_LOSS_POWER: f32 = 2.0;
	pub const ADAM_BETA1: f32 = 0.95;
	pub const ADAM_BETA2: f32 = 0.995;
	pub const REGULARIZATION_EPSILON: f32 = 1e-5;
}

pub mod quantization {
	pub const QUANTIZE_MASK_HIGH: u8 = 0xF0;
	pub const QUANTIZE_MASK_LOW: u8 = 0x00;
	pub const QUANTIZE_BIT_REDUCTION: usize = 12;
}

pub mod io {
	pub const BUFFER_THREADS: usize = 2;
	pub const VALIDATION_BUFFER_THREADS: usize = 4;
}

pub mod psnr {
	pub const LOG10_MULTIPLIER: f32 = -10.0;
}

pub mod file {
	pub const RSR_EXTENSION: &str = ".rsr";
	pub const PNG_EXTENSION: &str = ".png";
}
