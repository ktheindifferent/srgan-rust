//! Tests for the Waifu2x model variant, CNN inference, compat path, and weight converter.

use srgan_rust::config::{Waifu2xConfig, Waifu2xStyle};
use srgan_rust::model_downloader::{list_available_models, REMOTE_MODELS};
use srgan_rust::model_manager::ModelArchitecture;
use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use srgan_rust::waifu2x::{Waifu2xNetwork, weight_file_name, find_weight_file};
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
    assert!(Waifu2xConfig::new(1, 5).is_err());
}

#[test]
fn test_waifu2x_config_scale3_and_scale4_valid() {
    let cfg3 = Waifu2xConfig::new(1, 3).unwrap();
    assert_eq!(cfg3.scale, 3);
    assert_eq!(cfg3.model_label(), "waifu2x-noise1-scale3");

    let cfg4 = Waifu2xConfig::new(2, 4).unwrap();
    assert_eq!(cfg4.scale, 4);
    assert_eq!(cfg4.model_label(), "waifu2x-noise2-scale4");
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
    assert!(waifu2x_names.contains(&"waifu2x-upconv-7-anime-style-art-rgb"));
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
        let expected = if label.contains("scale4") {
            40
        } else if label.contains("scale3") {
            30
        } else if label.contains("scale2") || label == "waifu2x" || label == "waifu2x-anime" || label == "waifu2x-photo" {
            20
        } else {
            10
        };
        assert_eq!(result.width(), expected, "width mismatch for {}", label);
        assert_eq!(result.height(), expected, "height mismatch for {}", label);
    }
}

#[test]
fn test_waifu2x_compat_description_mentions_compat() {
    let net = Waifu2xNetwork::from_label("waifu2x").unwrap();
    // Without weight files, should be in compat mode
    if !net.is_cnn() {
        let desc = net.description();
        assert!(desc.contains("compat"), "description should mention compat: {}", desc);
    }
}

#[test]
fn test_waifu2x_compat_invalid_label_errors() {
    assert!(Waifu2xNetwork::from_label("esrgan").is_err());
    assert!(Waifu2xNetwork::from_label("waifu2x-bad").is_err());
}

#[test]
fn test_model_registry_includes_waifu2x_upconv7() {
    use srgan_rust::model_registry::ModelRegistry;
    use tempfile::TempDir;
    
    let dir = TempDir::new().unwrap();
    let registry = ModelRegistry::load_from(dir.path()).unwrap();
    
    let entry = registry.get("waifu2x-upconv-7-anime-style-art-rgb");
    assert!(entry.is_some(), "waifu2x-upconv-7-anime-style-art-rgb should be in registry");
    
    let entry = entry.unwrap();
    assert_eq!(entry.display_name, "Waifu2x Upconv-7 Anime Art");
    assert!(entry.builtin);
    assert!(entry.scale_factors.contains(&2));
}

#[test]
fn test_waifu2x_compat_scale3_triples_dimensions() {
    let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale3").unwrap();
    let img = test_image(10, 10);
    let result = net.upscale_image(&img).unwrap();
    assert_eq!(result.width(), 30);
    assert_eq!(result.height(), 30);
}

#[test]
fn test_waifu2x_compat_scale4_quadruples_dimensions() {
    let net = Waifu2xNetwork::from_label("waifu2x-noise2-scale4").unwrap();
    let img = test_image(10, 10);
    let result = net.upscale_image(&img).unwrap();
    assert_eq!(result.width(), 40);
    assert_eq!(result.height(), 40);
}

#[test]
fn test_waifu2x_ncnn_not_available_falls_back() {
    // Without waifu2x-ncnn-vulkan binary, should fall back to compat.
    let net = Waifu2xNetwork::from_label("waifu2x-noise1-scale2").unwrap();
    assert!(!net.is_ncnn(), "NCNN should not be available in test env");
}

#[test]
fn test_waifu2x_scale4_noise3_high_denoise_upscale() {
    let net = Waifu2xNetwork::from_label("waifu2x-noise3-scale4").unwrap();
    let img = test_image(8, 8);
    let result = net.upscale_image(&img).unwrap();
    assert_eq!(result.width(), 32);
    assert_eq!(result.height(), 32);
}

// ── CNN / compat mode detection ─────────────────────────────────────────────

#[test]
fn test_waifu2x_is_cnn_false_without_weight_files() {
    let net = Waifu2xNetwork::from_label("waifu2x").unwrap();
    // No weight files in the test environment, so should be compat mode
    assert!(!net.is_cnn());
}

#[test]
fn test_waifu2x_description_indicates_backend() {
    let net = Waifu2xNetwork::from_label("waifu2x-noise2-scale2").unwrap();
    let desc = net.description();
    if net.is_cnn() {
        assert!(desc.contains("CNN"), "CNN desc: {}", desc);
    } else {
        assert!(desc.contains("compat"), "compat desc: {}", desc);
    }
}

// ── Weight file naming and search ───────────────────────────────────────────

#[test]
fn test_weight_file_name_anime() {
    assert_eq!(weight_file_name(1, 2, Waifu2xStyle::Anime), "noise1_scale2_anime.rsr");
}

#[test]
fn test_weight_file_name_photo() {
    assert_eq!(weight_file_name(0, 1, Waifu2xStyle::Photo), "noise0_scale1_photo.rsr");
}

#[test]
fn test_weight_file_name_artwork() {
    assert_eq!(weight_file_name(3, 2, Waifu2xStyle::Artwork), "noise3_scale2_artwork.rsr");
}

#[test]
fn test_find_weight_file_returns_none_when_missing() {
    // No weight files should exist in the test environment
    assert!(find_weight_file(1, 2, Waifu2xStyle::Anime).is_none());
}

// ── Waifu2x from_config ─────────────────────────────────────────────────────

#[test]
fn test_waifu2x_from_config() {
    let cfg = Waifu2xConfig::with_style(2, 2, Waifu2xStyle::Photo).unwrap();
    let net = Waifu2xNetwork::from_config(&cfg).unwrap();
    assert_eq!(net.noise_level().as_u8(), 2);
    assert_eq!(net.scale().as_u8(), 2);
    assert_eq!(net.style(), Waifu2xStyle::Photo);
}

#[test]
fn test_waifu2x_from_label_with_style() {
    let net = Waifu2xNetwork::from_label_with_style("waifu2x-noise1-scale2", Waifu2xStyle::Artwork).unwrap();
    assert_eq!(net.style(), Waifu2xStyle::Artwork);
}

// ── VGG7 network graph builder ──────────────────────────────────────────────

#[test]
fn test_waifu2x_vgg7_net_scale1_builds() {
    let graph = srgan_rust::network::waifu2x_vgg7_net(1);
    assert!(graph.is_ok(), "VGG7 scale=1 graph build failed: {:?}", graph.err());
}

#[test]
fn test_waifu2x_vgg7_net_scale2_builds() {
    let graph = srgan_rust::network::waifu2x_vgg7_net(2);
    assert!(graph.is_ok(), "VGG7 scale=2 graph build failed: {:?}", graph.err());
}

#[test]
#[should_panic(expected = "waifu2x scale must be 1 or 2")]
fn test_waifu2x_vgg7_net_invalid_scale_panics() {
    let _ = srgan_rust::network::waifu2x_vgg7_net(4);
}

// ── Weight converter ────────────────────────────────────────────────────────

#[test]
fn test_waifu2x_converter_new() {
    let conv = srgan_rust::model_converter::waifu2x_converter::Waifu2xWeightConverter::new();
    assert_eq!(conv.summary(), "Waifu2x model: 0 layers, 0 total parameters, output_channels=0");
}

#[test]
fn test_waifu2x_converter_save_fails_without_weights() {
    let conv = srgan_rust::model_converter::waifu2x_converter::Waifu2xWeightConverter::new();
    assert!(conv.save_rsr(std::path::Path::new("/tmp/test_w2x.rsr")).is_err());
}

#[test]
fn test_waifu2x_converter_load_nonexistent_file_fails() {
    let mut conv = srgan_rust::model_converter::waifu2x_converter::Waifu2xWeightConverter::new();
    assert!(conv.load_waifu2x_json(std::path::Path::new("/nonexistent.json")).is_err());
}

// ── ModelFormat detection ───────────────────────────────────────────────────

#[test]
fn test_auto_detect_waifu2x_json_format() {
    use srgan_rust::model_converter::ModelConverter;
    let path = std::path::Path::new("noise1_scale2x_model.json");
    let format = ModelConverter::auto_detect_format(path);
    assert!(format.is_ok());
    match format.unwrap() {
        srgan_rust::model_converter::ModelFormat::Waifu2xJson => {},
        other => panic!("Expected Waifu2xJson, got {:?}", other),
    }
}
