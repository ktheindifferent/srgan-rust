pub mod checkpoint;
pub mod data_loader;
pub mod trainer_simple;
pub mod validation_simple;

pub use self::checkpoint::CheckpointManager;
pub use self::data_loader::DataLoader;
pub use self::trainer_simple::train_network;
pub use self::validation_simple::Validator;

use std::path::PathBuf;

/// High-level training run configuration, used by the training pipeline.
/// Distinct from `crate::config::TrainingConfig` which holds per-step optimizer settings.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Path to the directory of training images (PNG/JPG/WebP).
    pub dataset_path: PathBuf,
    /// Directory where checkpoints and the final model are written.
    pub output_path: PathBuf,
    /// Number of full passes over the dataset.
    pub epochs: usize,
    /// Images per gradient update step.
    pub batch_size: usize,
    /// Initial learning rate for the Adam optimizer.
    pub learning_rate: f64,
    /// Side length of the high-resolution patch in pixels (e.g. 96).
    pub patch_size: u32,
    /// Upscaling factor: 2, 4, or 8.
    pub scale_factor: u32,
    /// Save a checkpoint every N epochs.
    pub checkpoint_interval: usize,
    /// Stop training if validation loss does not improve for this many epochs.
    pub early_stopping_patience: usize,
    /// Fraction of images reserved for validation (0.0–1.0, default 0.2).
    pub validation_split: f64,
    /// Enable random crop, flip, and color-jitter augmentation.
    pub augmentation: bool,
    /// Resume from an existing checkpoint file.
    pub resume_from: Option<PathBuf>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::from("./data"),
            output_path: PathBuf::from("./output"),
            epochs: 100,
            batch_size: 16,
            learning_rate: 1e-4,
            patch_size: 96,
            scale_factor: 4,
            checkpoint_interval: 10,
            early_stopping_patience: 20,
            validation_split: 0.2,
            augmentation: true,
            resume_from: None,
        }
    }
}
