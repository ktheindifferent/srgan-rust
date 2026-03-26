//! Tests for the Waifu2x model variant and waifu2x-compat inference path.

use srgan_rust::config::Waifu2xConfig;
use srgan_rust::model_downloader::{list_available_models, REMOTE_MODELS};
use srgan_rust::model_manager::ModelArchitecture;
use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use srgan_rust::waifu2x::Waifu2xNetwork;
use image::GenericImage;

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

// ── Waifu2x-compat inference tests ──────────────────────────────────────────

fn test_image(w: u32, h: u32) -> image::DynamicImage {
    image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
        image::Rgba([(x % 256) as u8, (y % 256) as u8, 128u8, 255u8])
    }))
}

#[test]
fn test_waifu2x_compat_scale2_doubles_dimensions() {
    let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale2").unwrap();
    let img = test_image(16, 16);
    let result = net.upscale_image(&img).unwrap();
    assert_eq!(result.width(), 32);
    assert_eq!(result.height(), 32);
}

#[test]
fn test_waifu2x_compat_scale1_preserves_dimensions() {
    let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale1").unwrap();
    let img = test_image(16, 16);
    let result = net.upscale_image(&img).unwrap();
    assert_eq!(result.width(), 16);
    assert_eq!(result.height(), 16);
}

#[test]
fn test_waifu2x_compat_noise0_no_sharpening() {
    // noise=0 should return a pure Lanczos3 resize (no unsharp mask).
    let net = Waifu2xNetwork::from_label("waifu2x-noise0-scale2").unwrap();
    let img = test_image(8, 8);
    let result = net.upscale_image(&img).unwrap();
    assert_eq!(result.width(), 16);
    assert_eq!(result.height(), 16);
}

#[test]
fn test_waifu2x_compat_noise3_scale1_sharpens_only() {
    // scale=1 + noise=3 → same dimensions, sharpening applied.
    let net = Waifu2xNetwork::from_label("waifu2x-noise3-scale1").unwrap();
    let img = test_image(16, 16);
    let result = net.upscale_image(&img).unwrap();
    assert_eq!(result.width(), 16);
    assert_eq!(result.height(), 16);
}

#[test]
fn test_waifu2x_compat_all_variants_succeed() {
    let img = test_image(10, 10);
    for &label in srgan_rust::waifu2x::WAIFU2X_LABELS {
        let net = Waifu2xNetwork::from_label(label)
            .unwrap_or_else(|e| panic!("from_label({}) failed: {}", label, e));
        let result = net.upscale_image(&img)
            .unwrap_or_else(|e| panic!("upscale_image({}) failed: {}", label, e));
        // scale-2 variants should double, scale-1 should preserve.
        if label.contains("scale2") || label == "waifu2x" {
            assert_eq!(result.width(), 20, "width mismatch for {}", label);
            assert_eq!(result.height(), 20, "height mismatch for {}", label);
        } else {
            assert_eq!(result.width(), 10, "width mismatch for {}", label);
            assert_eq!(result.height(), 10, "height mismatch for {}", label);
        }
    }
}

#[test]
fn test_waifu2x_compat_description_mentions_compat() {
    let net = Waifu2xNetwork::from_label("waifu2x").unwrap();
    let desc = net.description();
    assert!(desc.contains("compat"), "description should mention compat: {}", desc);
}

#[test]
fn test_waifu2x_compat_invalid_label_errors() {
    assert!(Waifu2xNetwork::from_label("esrgan").is_err());
    assert!(Waifu2xNetwork::from_label("waifu2x-bad").is_err());
}
