use crate::aligned_crop::AlignedCrop;
use crate::constants::io;
use crate::error::{Result, SrganError};
use alumina::data::{image_folder::ImageFolder, Cropping, DataSet, DataStream};

pub struct DataLoader;

impl DataLoader {
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

	pub fn create_prescaled_training_stream(
		input_folders: Vec<&str>,
		recurse: bool,
		patch_size: usize,
		factor: usize,
		batch_size: usize,
	) -> Result<Box<dyn DataStream>> {
		let mut folders_iter = input_folders.into_iter();
		let first_folder = folders_iter.next()
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

	pub fn create_validation_stream(validation_folder: &str, recurse: bool) -> Box<dyn DataStream> {
		Box::new(
			ImageFolder::new(validation_folder, recurse)
				.shuffle_random()
				.batch(1)
				.buffered(io::VALIDATION_BUFFER_THREADS),
		)
	}
}
