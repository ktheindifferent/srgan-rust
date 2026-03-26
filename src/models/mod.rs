//! Model implementations for various super-resolution architectures.

pub mod real_esrgan;
pub mod waifu2x;

pub use real_esrgan::{RealEsrganModel, RealEsrganVariant, REAL_ESRGAN_LABELS};
pub use waifu2x::{NoiseLevel, Waifu2xNetwork, Waifu2xScale, Waifu2xVariant, WAIFU2X_LABELS};
