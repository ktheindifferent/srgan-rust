use crate::config::{LossType, NetworkConfig, TrainingConfig};
use crate::error::{Result, SrganError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Configuration structure for training parameters that can be loaded from a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigFile {
    /// Network architecture configuration
    #[serde(default)]
    pub network: NetworkConfigSection,
    
    /// Training hyperparameters
    #[serde(default)]
    pub training: TrainingConfigSection,
    
    /// Data configuration
    #[serde(default)]
    pub data: DataConfigSection,
    
    /// Validation configuration
    #[serde(default)]
    pub validation: ValidationConfigSection,
    
    /// Output configuration
    #[serde(default)]
    pub output: OutputConfigSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NetworkConfigSection {
    /// Upscaling factor (default: 4)
    pub factor: usize,
    
    /// Minimum number of channels in hidden layers (default: 16)
    pub width: u32,
    
    /// Network depth: 2^(log_depth)-1 layers (default: 4)
    pub log_depth: u32,
    
    /// Relative size of global nodes (default: 2)
    pub global_node_factor: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfigSection {
    /// Learning rate for Adam optimizer (default: 0.003)
    pub learning_rate: f32,
    
    /// Training batch size (default: 4)
    pub batch_size: usize,
    
    /// Training patch size (default: 48)
    pub patch_size: usize,
    
    /// Loss function: "L1" or "L2" (default: "L1")
    pub loss_type: String,
    
    /// Use sRGB colorspace for downscaling (default: true)
    pub srgb_downscale: bool,
    
    /// Quantise weights to reduce file size (default: false)
    pub quantise: bool,
    
    /// Checkpoint interval in steps (default: 1000)
    pub checkpoint_interval: usize,
    
    /// Maximum training steps (optional)
    pub max_steps: Option<usize>,
    
    /// Early stopping patience in validation rounds (optional)
    pub early_stopping_patience: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataConfigSection {
    /// Training data directory
    pub training_folder: String,
    
    /// Recurse into subdirectories (default: true)
    pub recurse: bool,
    
    /// Data augmentation settings
    pub augmentation: DataAugmentation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataAugmentation {
    /// Random horizontal flip (default: true)
    pub horizontal_flip: bool,
    
    /// Random vertical flip (default: false)
    pub vertical_flip: bool,
    
    /// Random 90-degree rotations (default: true)
    pub rotate_90: bool,
    
    /// Random brightness adjustment range (default: 0.1)
    pub brightness_range: f32,
    
    /// Random contrast adjustment range (default: 0.1)
    pub contrast_range: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationConfigSection {
    /// Validation data directory (optional)
    pub validation_folder: Option<String>,
    
    /// Maximum validation images per round (optional)
    pub max_images: Option<usize>,
    
    /// Validation frequency in steps (default: 100)
    pub frequency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfigSection {
    /// Output parameter file path
    pub parameter_file: String,
    
    /// Checkpoint directory (default: "./checkpoints")
    pub checkpoint_dir: String,
    
    /// TensorBoard log directory (optional)
    pub tensorboard_dir: Option<String>,
    
    /// Save best model based on validation loss (default: true)
    pub save_best_model: bool,
}

// Default implementations
impl Default for NetworkConfigSection {
    fn default() -> Self {
        Self {
            factor: 4,
            width: 16,
            log_depth: 4,
            global_node_factor: 2,
        }
    }
}

impl Default for TrainingConfigSection {
    fn default() -> Self {
        Self {
            learning_rate: 0.003,
            batch_size: 4,
            patch_size: 48,
            loss_type: "L1".to_string(),
            srgb_downscale: true,
            quantise: false,
            checkpoint_interval: 1000,
            max_steps: None,
            early_stopping_patience: None,
        }
    }
}

impl Default for DataConfigSection {
    fn default() -> Self {
        Self {
            training_folder: "./training_data".to_string(),
            recurse: true,
            augmentation: DataAugmentation::default(),
        }
    }
}

impl Default for DataAugmentation {
    fn default() -> Self {
        Self {
            horizontal_flip: true,
            vertical_flip: false,
            rotate_90: true,
            brightness_range: 0.1,
            contrast_range: 0.1,
        }
    }
}

impl Default for ValidationConfigSection {
    fn default() -> Self {
        Self {
            validation_folder: None,
            max_images: None,
            frequency: 100,
        }
    }
}

impl Default for OutputConfigSection {
    fn default() -> Self {
        Self {
            parameter_file: "./model.rsr".to_string(),
            checkpoint_dir: "./checkpoints".to_string(),
            tensorboard_dir: None,
            save_best_model: true,
        }
    }
}

impl TrainingConfigFile {
    /// Load configuration from a TOML file
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path)
            .map_err(|e| SrganError::Io(e))?;
        
        toml::from_str(&contents)
            .map_err(|e| SrganError::Parse(format!("Failed to parse TOML config: {}", e)))
    }
    
    /// Load configuration from a JSON file
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path)
            .map_err(|e| SrganError::Io(e))?;
        
        serde_json::from_str(&contents)
            .map_err(|e| SrganError::Parse(format!("Failed to parse JSON config: {}", e)))
    }
    
    /// Save configuration to a TOML file
    pub fn to_toml_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = toml::to_string_pretty(self)
            .map_err(|e| SrganError::Serialization(format!("Failed to serialize to TOML: {}", e)))?;
        
        fs::write(path, contents)
            .map_err(|e| SrganError::Io(e))
    }
    
    /// Save configuration to a JSON file
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = serde_json::to_string_pretty(self)
            .map_err(|e| SrganError::Serialization(format!("Failed to serialize to JSON: {}", e)))?;
        
        fs::write(path, contents)
            .map_err(|e| SrganError::Io(e))
    }
    
    /// Convert to NetworkConfig
    pub fn to_network_config(&self) -> NetworkConfig {
        NetworkConfig {
            factor: self.network.factor,
            width: self.network.width,
            log_depth: self.network.log_depth,
            global_node_factor: self.network.global_node_factor,
        }
    }
    
    /// Convert to TrainingConfig
    pub fn to_training_config(&self) -> Result<TrainingConfig> {
        let loss_type = match self.training.loss_type.to_uppercase().as_str() {
            "L1" => LossType::L1,
            "L2" => LossType::L2,
            _ => return Err(SrganError::InvalidParameter(
                format!("Invalid loss type: {}. Must be L1 or L2", self.training.loss_type)
            )),
        };
        
        Ok(TrainingConfig {
            learning_rate: self.training.learning_rate,
            patch_size: self.training.patch_size,
            batch_size: self.training.batch_size,
            loss_type,
            srgb_downscale: self.training.srgb_downscale,
            quantise: self.training.quantise,
            recurse: self.data.recurse,
            checkpoint_interval: self.training.checkpoint_interval,
        })
    }
    
    /// Generate a default configuration file
    pub fn generate_default() -> Self {
        Self {
            network: NetworkConfigSection::default(),
            training: TrainingConfigSection::default(),
            data: DataConfigSection::default(),
            validation: ValidationConfigSection::default(),
            output: OutputConfigSection::default(),
        }
    }
    
    /// Create an example configuration file with comments
    pub fn create_example_toml() -> String {
        r#"# SRGAN-Rust Training Configuration File
# This file configures all aspects of neural network training

[network]
# Upscaling factor (typically 2, 4, or 8)
factor = 4

# Minimum number of channels in hidden layers
# Higher values = more capacity but slower
width = 16

# Network depth: 2^(log_depth)-1 hidden layers
# log_depth=4 means 15 hidden layers
log_depth = 4

# Relative size of global (non-spatial) nodes
global_node_factor = 2

[training]
# Learning rate for Adam optimizer
learning_rate = 0.003

# Number of image patches per batch
batch_size = 4

# Size of training patches (before downscaling)
patch_size = 48

# Loss function: "L1" (sharper) or "L2" (smoother)
loss_type = "L1"

# Use sRGB colorspace for downscaling
srgb_downscale = true

# Quantise weights to reduce file size
quantise = false

# Save checkpoint every N steps
checkpoint_interval = 1000

# Maximum training steps (optional, remove to train indefinitely)
# max_steps = 100000

# Early stopping patience (optional)
# early_stopping_patience = 10

[data]
# Directory containing training images
training_folder = "./training_data"

# Recurse into subdirectories
recurse = true

# Data augmentation settings
[data.augmentation]
horizontal_flip = true
vertical_flip = false
rotate_90 = true
brightness_range = 0.1
contrast_range = 0.1

[validation]
# Validation folder (optional)
# validation_folder = "./validation_data"

# Maximum validation images per round
# max_images = 100

# Run validation every N steps
frequency = 100

[output]
# Where to save the final model
parameter_file = "./model.rsr"

# Directory for checkpoint files
checkpoint_dir = "./checkpoints"

# TensorBoard log directory (optional)
# tensorboard_dir = "./logs"

# Save the best model based on validation loss
save_best_model = true
"#.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = TrainingConfigFile::generate_default();
        assert_eq!(config.network.factor, 4);
        assert_eq!(config.training.learning_rate, 0.003);
        assert_eq!(config.data.training_folder, "./training_data");
    }

    #[test]
    fn test_to_network_config() {
        let config_file = TrainingConfigFile::generate_default();
        let network_config = config_file.to_network_config();
        assert_eq!(network_config.factor, 4);
        assert_eq!(network_config.width, 16);
    }

    #[test]
    fn test_to_training_config() {
        let config_file = TrainingConfigFile::generate_default();
        let training_config = config_file.to_training_config().unwrap();
        assert_eq!(training_config.learning_rate, 0.003);
        assert_eq!(training_config.batch_size, 4);
    }

    #[test]
    fn test_save_load_toml() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        
        let config = TrainingConfigFile::generate_default();
        config.to_toml_file(&path).unwrap();
        
        let loaded = TrainingConfigFile::from_toml_file(&path).unwrap();
        assert_eq!(loaded.network.factor, config.network.factor);
    }

    #[test]
    fn test_save_load_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        
        let config = TrainingConfigFile::generate_default();
        config.to_json_file(&path).unwrap();
        
        let loaded = TrainingConfigFile::from_json_file(&path).unwrap();
        assert_eq!(loaded.network.factor, config.network.factor);
    }
}