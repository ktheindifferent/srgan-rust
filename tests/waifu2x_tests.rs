//! Tests for the Waifu2x model variant.

use srgan_rust::config::Waifu2xConfig;
use srgan_rust::model_downloader::{list_available_models, REMOTE_MODELS};
use srgan_rust::model_manager::ModelArchitecture;
use srgan_rust::thread_safe_network::ThreadSafeNetwork;

// ── Waifu2xConfig ─────────────────────────────────────────────────────────────

#[test]
fn test_waifu2x_config_default() {
    let cfg = Waifu2xConfig::default();
    assert_eq!(cfg.noise_level, 1);
    assert_eq!(cfg.scale, 2);
}

#[test]
fn test_waifu2x_config_new_valid() {
    let cfg = Waifu2xConfig::new(0, 1).unwrap();
    assert_eq!(cfg.noise_level, 0);
    assert_eq!(cfg.scale, 1);

    let cfg = Waifu2xConfig::new(3, 2).unwrap();
    assert_eq!(cfg.noise_level, 3);
    assert_eq!(cfg.scale, 2);
}

#[test]
fn test_waifu2x_config_invalid_noise_level() {
    assert!(Waifu2xConfig::new(4, 2).is_err());
}

#[test]
fn test_waifu2x_config_invalid_scale() {
    assert!(Waifu2xConfig::new(1, 0).is_err());
    assert!(Waifu2xConfig::new(1, 3).is_err());
    assert!(Waifu2xConfig::new(1, 4).is_err());
}

#[test]
fn test_waifu2x_config_model_label() {
    assert_eq!(
        Waifu2xConfig::new(1, 2).unwrap().model_label(),
        "waifu2x-noise1-scale2"
    );
    assert_eq!(
        Waifu2xConfig::new(0, 1).unwrap().model_label(),
        "waifu2x-noise0-scale1"
    );
    assert_eq!(
        Waifu2xConfig::new(3, 2).unwrap().model_label(),
        "waifu2x-noise3-scale2"
    );
}

// ── ModelArchitecture ─────────────────────────────────────────────────────────

#[test]
fn test_waifu2x_architecture_fallback_label() {
    assert_eq!(ModelArchitecture::Waifu2x.fallback_label(), "anime");
}

#[test]
fn test_waifu2x_architecture_display_name() {
    assert_eq!(ModelArchitecture::Waifu2x.display_name(), "Waifu2x");
}

#[test]
fn test_waifu2x_architecture_serde_round_trip() {
    let arch = ModelArchitecture::Waifu2x;
    let json = serde_json::to_string(&arch).unwrap();
    assert_eq!(json, "\"waifu2x\"");
    let decoded: ModelArchitecture = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, ModelArchitecture::Waifu2x);
}

// ── Model catalogue ───────────────────────────────────────────────────────────

#[test]
fn test_remote_models_include_waifu2x() {
    let waifu2x_names: Vec<&str> = REMOTE_MODELS
        .iter()
        .filter(|m| m.name.starts_with("waifu2x"))
        .map(|m| m.name)
        .collect();
    assert!(!waifu2x_names.is_empty(), "no waifu2x entries in REMOTE_MODELS");
    assert!(waifu2x_names.contains(&"waifu2x-noise1-scale2"));
    assert!(waifu2x_names.contains(&"waifu2x-noise2-scale2"));
    assert!(waifu2x_names.contains(&"waifu2x-noise0-scale1"));
    assert!(waifu2x_names.contains(&"waifu2x-noise3-scale2"));
}

#[test]
fn test_list_available_models_includes_waifu2x() {
    let models = list_available_models();
    let waifu2x: Vec<_> = models
        .iter()
        .filter(|(name, _, _, _)| name.starts_with("waifu2x"))
        .collect();
    assert!(!waifu2x.is_empty());
}

// ── ThreadSafeNetwork label loading ──────────────────────────────────────────

#[test]
fn test_thread_safe_network_from_label_waifu2x_bare() {
    let net = ThreadSafeNetwork::from_label("waifu2x", None);
    assert!(net.is_ok(), "expected waifu2x label to load");
}

#[test]
fn test_thread_safe_network_from_label_waifu2x_noise1_scale2() {
    let net = ThreadSafeNetwork::from_label("waifu2x-noise1-scale2", None);
    assert!(net.is_ok());
}

#[test]
fn test_thread_safe_network_from_label_waifu2x_noise0_scale1() {
    let net = ThreadSafeNetwork::from_label("waifu2x-noise0-scale1", None);
    assert!(net.is_ok());
}

#[test]
fn test_thread_safe_network_from_label_waifu2x_noise3_scale2() {
    let net = ThreadSafeNetwork::from_label("waifu2x-noise3-scale2", None);
    assert!(net.is_ok());
}

#[test]
fn test_thread_safe_network_waifu2x_display_contains_label() {
    let net = ThreadSafeNetwork::from_label("waifu2x-noise2-scale2", None).unwrap();
    assert!(
        net.display().contains("waifu2x"),
        "display '{}' should mention waifu2x",
        net.display()
    );
}

#[test]
fn test_thread_safe_network_waifu2x_factor_is_4() {
    // Backed by the built-in anime model which is 4×.
    let net = ThreadSafeNetwork::from_label("waifu2x", None).unwrap();
    assert_eq!(net.factor(), 4);
}
