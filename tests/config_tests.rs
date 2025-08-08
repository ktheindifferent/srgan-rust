use srgan_rust::config::{NetworkConfig, TrainingConfig, LossType};

#[test]
fn test_network_config_default() {
    let config = NetworkConfig::default();
    assert_eq!(config.factor, 4);
    assert_eq!(config.width, 16);
    assert_eq!(config.log_depth, 4);
    assert_eq!(config.global_node_factor, 2);
}

#[test]
fn test_network_config_validate_success() {
    let config = NetworkConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_network_config_validate_zero_factor() {
    let config = NetworkConfig {
        factor: 0,
        width: 16,
        log_depth: 4,
        global_node_factor: 2,
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Factor must be greater than 0"));
}

#[test]
fn test_network_config_validate_zero_width() {
    let config = NetworkConfig {
        factor: 4,
        width: 0,
        log_depth: 4,
        global_node_factor: 2,
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Width must be greater than 0"));
}

#[test]
fn test_network_config_validate_zero_log_depth() {
    let config = NetworkConfig {
        factor: 4,
        width: 16,
        log_depth: 0,
        global_node_factor: 2,
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Log depth must be greater than 0"));
}

#[test]
fn test_network_config_builder() {
    let config = NetworkConfig::builder()
        .factor(8)
        .width(32)
        .log_depth(5)
        .global_node_factor(3)
        .build();
    
    assert_eq!(config.factor, 8);
    assert_eq!(config.width, 32);
    assert_eq!(config.log_depth, 5);
    assert_eq!(config.global_node_factor, 3);
}

#[test]
fn test_training_config_default() {
    let config = TrainingConfig::default();
    assert_eq!(config.learning_rate, 3e-3);
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.patch_size, 48);
    assert!(matches!(config.loss_type, LossType::L1));
    assert_eq!(config.srgb_downscale, true);
    assert_eq!(config.recurse, false);
    assert_eq!(config.quantise, false);
}

#[test]
fn test_training_config_validate_success() {
    let config = TrainingConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_training_config_validate_zero_batch_size() {
    let mut config = TrainingConfig::default();
    config.batch_size = 0;
    let result = config.validate();
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Batch size"));
}

#[test]
fn test_training_config_validate_zero_patch_size() {
    let mut config = TrainingConfig::default();
    config.patch_size = 0;
    let result = config.validate();
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Patch size"));
}

#[test]
fn test_training_config_validate_zero_learning_rate() {
    let mut config = TrainingConfig::default();
    config.learning_rate = 0.0;
    let result = config.validate();
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Learning rate"));
}

#[test]
fn test_training_config_builder() {
    let config = TrainingConfig::builder()
        .learning_rate(1e-4)
        .batch_size(8)
        .patch_size(64)
        .loss_type(LossType::L2)
        .srgb_downscale(false)
        .recurse(true)
        .quantise(true)
        .build();
    
    assert_eq!(config.learning_rate, 1e-4);
    assert_eq!(config.batch_size, 8);
    assert_eq!(config.patch_size, 64);
    assert!(matches!(config.loss_type, LossType::L2));
    assert_eq!(config.srgb_downscale, false);
    assert_eq!(config.recurse, true);
    assert_eq!(config.quantise, true);
}