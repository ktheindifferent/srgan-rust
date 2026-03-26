use crate::aligned_crop::AlignedCrop;
use crate::constants::io;
use crate::error::{Result, SrganError};
use alumina::data::{image_folder::ImageFolder, Cropping, DataSet, DataStream};
use image::DynamicImage;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use std::path::{Path, PathBuf};

// ── Augmentation ─────────────────────────────────────────────────────────────

/// Parameters controlling data augmentation applied during training.
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// Apply random horizontal flip with 50% probability.
    pub horizontal_flip: bool,
    /// Apply random vertical flip with 50% probability.
    pub vertical_flip: bool,
    /// Randomly adjust brightness by ±`brightness_delta` (0.0–1.0).
    pub brightness_jitter: bool,
    /// Maximum brightness delta when `brightness_jitter` is enabled.
    pub brightness_delta: i32,
    /// Randomly adjust contrast when `color_jitter` is enabled.
    pub color_jitter: bool,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            horizontal_flip: true,
            vertical_flip: false,
            brightness_jitter: true,
            brightness_delta: 20,
            color_jitter: true,
        }
    }
}

/// Apply configured augmentation to a single image.
pub fn augment_image(img: DynamicImage, config: &AugmentationConfig) -> DynamicImage {
    let mut rng = rand::thread_rng();
    let mut out = img;

    if config.horizontal_flip && rng.gen::<bool>() {
        out = DynamicImage::ImageRgba8(image::imageops::flip_horizontal(&out));
    }

    if config.vertical_flip && rng.gen::<bool>() {
        out = DynamicImage::ImageRgba8(image::imageops::flip_vertical(&out));
    }

    if config.brightness_jitter {
        let delta = rng.gen_range(-(config.brightness_delta)..=config.brightness_delta);
        out = out.brighten(delta);
    }

    if config.color_jitter {
        let contrast: f32 = rng.gen_range(-0.3_f32..0.3_f32);
        out = out.adjust_contrast(contrast);
    }

    out
}

// ── Validation split ──────────────────────────────────────────────────────────

/// Supported image extensions for dataset scanning.
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

/// Scan `dir` for images and split into (train, validation) path lists.
///
/// `validation_fraction` should be in the range 0.0–1.0 (default 0.2).
/// Files are sorted for reproducibility before splitting.
pub fn split_dataset(dir: &Path, validation_fraction: f64) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| SrganError::MissingFolder(format!("Cannot read dataset dir: {}", e)))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() {
                let ext = path.extension()?.to_string_lossy().to_lowercase();
                if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                    return Some(path);
                }
            }
            None
        })
        .collect();

    if paths.is_empty() {
        return Err(SrganError::MissingFolder(format!(
            "No images found in {}",
            dir.display()
        )));
    }

    paths.sort();

    let val_count = ((paths.len() as f64) * validation_fraction.clamp(0.0, 1.0)) as usize;
    let val_count = val_count.max(1).min(paths.len() - 1);
    let train_paths = paths[val_count..].to_vec();
    let val_paths = paths[..val_count].to_vec();

    Ok((train_paths, val_paths))
}

/// Return a progress bar styled for dataset scanning.
pub fn dataset_progress_bar(len: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
    );
    pb.set_message(message.to_owned());
    pb
}

// ── DataLoader ────────────────────────────────────────────────────────────────

pub struct DataLoader;

impl DataLoader {
    /// Create a training data stream from a folder of high-resolution images.
    ///
    /// Each image is randomly cropped to `patch_size * factor` pixels, then
    /// shuffled, batched, and buffered for parallel loading.
    pub fn create_training_stream(
        training_folder: &str,
        recurse: bool,
        patch_size: usize,
        factor: usize,
        batch_size: usize,
    ) -> Box<dyn DataStream> {
        Box::new(
            ImageFolder::new(training_folder, recurse)
                .crop(0, &[patch_size * factor, patch_size * factor, 3], Cropping::Random)
                .shuffle_random()
                .batch(batch_size)
                .buffered(io::BUFFER_THREADS),
        )
    }

    /// Create a training stream from paired (input, target) image folders.
    pub fn create_prescaled_training_stream(
        input_folders: Vec<&str>,
        recurse: bool,
        patch_size: usize,
        factor: usize,
        batch_size: usize,
    ) -> Result<Box<dyn DataStream>> {
        let mut folders_iter = input_folders.into_iter();
        let first_folder = folders_iter
            .next()
            .ok_or_else(|| SrganError::MissingFolder("At least one training folder required".to_string()))?;

        let input_folder = std::path::Path::new(first_folder);
        let mut target_folder = input_folder
            .parent()
            .ok_or_else(|| SrganError::InvalidInput("Don't use root as a training folder".to_string()))?
            .to_path_buf();
        target_folder.push("Base");

        let initial_set = ImageFolder::new(input_folder, recurse)
            .concat_components(ImageFolder::new(target_folder, recurse))
            .boxed();

        let set = folders_iter.fold(initial_set, |set, input_folder| {
            let input_path = std::path::Path::new(input_folder);
            let mut target_folder = input_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| std::path::PathBuf::from("."));
            target_folder.push("Base");

            ImageFolder::new(input_path, recurse)
                .concat_components(ImageFolder::new(target_folder, recurse))
                .concat_elements(set)
                .boxed()
        });

        Ok(Box::new(
            set.aligned_crop(0, &[patch_size, patch_size, 3], Cropping::Random)
                .and_crop(1, &[factor, factor, 1])
                .shuffle_random()
                .batch(batch_size)
                .buffered(io::BUFFER_THREADS),
        ))
    }

    /// Create a validation data stream from a folder of images.
    pub fn create_validation_stream(validation_folder: &str, recurse: bool) -> Box<dyn DataStream> {
        Box::new(
            ImageFolder::new(validation_folder, recurse)
                .shuffle_random()
                .batch(1)
                .buffered(io::VALIDATION_BUFFER_THREADS),
        )
    }
}
