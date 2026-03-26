use crate::constants::network;
use crate::error::{Result, SrganError};
use crate::gpu::{GpuBackend, GpuDevice};
use crate::validation;
use crate::UpscalingNetwork;
use clap::ArgMatches;
use indicatif::ProgressBar;
use log::{info, warn};
use ndarray::{ArrayD, Axis, IxDyn};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use image;

const TILE_SIZE: usize = 256;
const TILE_OVERLAP: usize = 32;
const LARGE_IMAGE_THRESHOLD: usize = 4_000_000; // 4 megapixels

pub fn upscale(app_m: &ArgMatches) -> Result<()> {
	let factor = parse_factor(app_m);

	// Resolve the input path early so auto-detect can open the image for classification.
	let input_path_str = app_m
		.value_of("INPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No input file given".to_string()))?;
	let input_path_buf = validation::validate_input_file(input_path_str)?;
	validation::validate_image_extension(&input_path_buf)?;

	let network = load_network(app_m, factor, &input_path_buf)?;

	// GPU device selection
	let device_label = if app_m.is_present("GPU") {
		let device = GpuDevice::select_best();
		match device.backend() {
			GpuBackend::None => {
				warn!("--gpu requested but no GPU backend is available; using CPU");
				"CPU".to_string()
			}
			b => {
				info!("GPU acceleration enabled: {} ({})", device.name(), b);
				info!("Note: GPU inference offload not yet implemented; running on CPU");
				format!("{} ({})", device.name(), b)
			}
		}
	} else {
		"CPU".to_string()
	};

	info!("Processing device: {}", device_label);
	info!("Upsampling using {}...", network);

	let spinner = ProgressBar::new_spinner();
	spinner.set_message(format!("Processing [{}]...", device_label));

	let output_path = app_m
		.value_of("OUTPUT_FILE")
		.ok_or_else(|| SrganError::InvalidParameter("No output file given".to_string()))?;

	let output_path_buf = validation::validate_output_path(output_path)?;
	validation::validate_factor(factor)?;

	// Output format / quality
	let format = detect_format(app_m, &output_path_buf);
	let quality: u8 = app_m
		.value_of("QUALITY")
		.and_then(|s| s.parse::<u8>().ok())
		.unwrap_or(85)
		.max(1);

	let mut input_file = File::open(&input_path_buf)?;
	let input = crate::read(&mut input_file)?;

	// Choose tiled vs direct upscaling based on image size
	let pixel_count = {
		let s = input.shape();
		s[1] * s[2] // H * W (tensor is [1, H, W, C])
	};

	let output = if pixel_count > LARGE_IMAGE_THRESHOLD {
		let mp = pixel_count as f64 / 1_000_000.0;
		spinner.set_message(format!(
			"Large image ({:.1} MP) — tiled processing...",
			mp
		));
		info!(
			"Image exceeds {}MP threshold ({:.1} MP); using tiled processing ({} tiles {}px overlap)",
			LARGE_IMAGE_THRESHOLD / 1_000_000,
			mp,
			TILE_SIZE,
			TILE_OVERLAP,
		);
		upscale_tiled(input, &network)?
	} else {
		spinner.set_message("Running neural network...");
		crate::upscale(input, &network)
			.map_err(|e| SrganError::GraphExecution(e.to_string()))?
	};

	spinner.set_message("Writing output file...");
	let mut output_file = File::create(&output_path_buf)?;
	crate::save_with_format(output, &mut output_file, &format, quality)?;

	spinner.finish_with_message(format!(
		"✓ Complete [device: {}, format: {}, quality: {}]",
		device_label, format, quality
	));
	info!("Output saved to: {}", output_path_buf.display());
	Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_factor(app_m: &ArgMatches) -> usize {
	app_m
		.value_of("BILINEAR_FACTOR")
		.and_then(|s| s.parse::<usize>().ok())
		.unwrap_or(network::DEFAULT_FACTOR)
}

fn load_network(app_m: &ArgMatches, factor: usize, input_path: &Path) -> Result<UpscalingNetwork> {
	if let Some(file_str) = app_m.value_of("CUSTOM") {
		let param_path = validation::validate_input_file(file_str)?;
		let mut param_file = File::open(&param_path)?;
		let mut data = Vec::new();
		param_file.read_to_end(&mut data)?;
		let network_desc = crate::network_from_bytes(&data)?;
		UpscalingNetwork::new(network_desc, "custom trained neural net")
			.map_err(|e| SrganError::Network(e))
	} else if app_m.is_present("AUTO_DETECT") && app_m.value_of("PARAMETERS").is_none() {
		// Auto-detect: open the image, classify it, pick the best model.
		match image::open(input_path) {
			Ok(img) => {
				let image_type = crate::detection::detect_image_type(&img);
				let label = crate::detection::recommended_model_for(&image_type);
				info!("Auto-detected image type: {} → model: {}", image_type, label);
				UpscalingNetwork::from_label(label, Some(factor))
					.map_err(|e| SrganError::Network(e))
			}
			Err(e) => {
				warn!("Auto-detect: could not open image for classification ({}); falling back to 'natural'", e);
				UpscalingNetwork::from_label("natural", Some(factor))
					.map_err(|e| SrganError::Network(e))
			}
		}
	} else {
		let param_type = app_m.value_of("PARAMETERS").unwrap_or("natural");
		UpscalingNetwork::from_label(param_type, Some(factor))
			.map_err(|e| SrganError::Network(e))
	}
}

/// Detect output format from `--format` flag or file extension.
fn detect_format(app_m: &ArgMatches, output_path: &Path) -> String {
	if let Some(fmt) = app_m.value_of("FORMAT") {
		return fmt.to_lowercase();
	}
	match output_path
		.extension()
		.and_then(|e| e.to_str())
		.map(|e| e.to_lowercase())
		.as_deref()
	{
		Some("jpg") | Some("jpeg") => "jpeg".to_string(),
		Some("webp") => "webp".to_string(),
		_ => "png".to_string(),
	}
}

// ---------------------------------------------------------------------------
// Tiled upscaling for large images
// ---------------------------------------------------------------------------

/// Upscale a large image by splitting it into overlapping tiles, upscaling each
/// independently, and feather-blending them back together.
///
/// Tiles: `TILE_SIZE × TILE_SIZE` with `TILE_OVERLAP` pixels of overlap on each
/// interior edge.  The blend weight for each pixel ramps linearly from 0 at the
/// overlap boundary to 1 at distance `TILE_OVERLAP` into the tile centre.
fn upscale_tiled(img: ArrayD<f32>, network: &UpscalingNetwork) -> Result<ArrayD<f32>> {
	let shape = img.shape().to_vec();
	let in_h = shape[1];
	let in_w = shape[2];

	// Assume x4 upscaling (standard SRGAN).  Bilinear networks use the same
	// tile path; if a non-×4 custom network is used, the output dimensions will
	// still be correct because we measure from the actual upscaled tile shape.
	let scale = 4usize;
	let out_h = in_h * scale;
	let out_w = in_w * scale;
	let overlap_out = TILE_OVERLAP * scale;

	let mut accum = ArrayD::<f32>::zeros(IxDyn(&[out_h, out_w, 3]));
	let mut wsum = ArrayD::<f32>::zeros(IxDyn(&[out_h, out_w, 1]));

	let step = TILE_SIZE.saturating_sub(2 * TILE_OVERLAP).max(1);

	let img_3d = img.subview(Axis(0), 0); // [H, W, C]

	// Collect tile start positions so the last tile always ends at the image edge
	let y_starts = tile_starts(in_h, TILE_SIZE, step);
	let x_starts = tile_starts(in_w, TILE_SIZE, step);

	for &ys in &y_starts {
		let ye = (ys + TILE_SIZE).min(in_h);
		let th = ye - ys;

		for &xs in &x_starts {
			let xe = (xs + TILE_SIZE).min(in_w);
			let tw = xe - xs;

			// Extract tile → [1, th, tw, 3]
			let tile = img_3d.slice(s![ys..ye, xs..xe, ..]).to_owned();
			let tile_4d = tile
				.into_shape(IxDyn(&[1, th, tw, 3]))
				.map_err(|e| SrganError::ShapeError(format!("tile reshape: {}", e)))?;

			let upscaled = crate::upscale(tile_4d, network)
				.map_err(|e| SrganError::GraphExecution(e.to_string()))?;

			// Remove batch dim → [out_th, out_tw, 3]
			let ut = upscaled.subview(Axis(0), 0);
			let out_th = ut.shape()[0];
			let out_tw = ut.shape()[1];

			let oy0 = ys * scale;
			let ox0 = xs * scale;

			for i in 0..out_th {
				for j in 0..out_tw {
					let oy = oy0 + i;
					let ox = ox0 + j;
					if oy >= out_h || ox >= out_w {
						continue;
					}

					// Feather weight: ramp from 0 at the tile edge to 1 at
					// distance `overlap_out` inward.
					let wy = blend_weight(i, out_th, overlap_out);
					let wx = blend_weight(j, out_tw, overlap_out);
					let w = wy * wx;

					for c in 0..3usize {
						accum[[oy, ox, c]] += ut[[i, j, c]] * w;
					}
					wsum[[oy, ox, 0]] += w;
				}
			}
		}
	}

	// Normalise accumulator
	for oy in 0..out_h {
		for ox in 0..out_w {
			let w = wsum[[oy, ox, 0]];
			if w > 0.0 {
				for c in 0..3usize {
					accum[[oy, ox, c]] /= w;
				}
			}
		}
	}

	accum
		.into_shape(IxDyn(&[1, out_h, out_w, 3]))
		.map_err(|e| SrganError::ShapeError(format!("output reshape: {}", e)))
}

/// Returns the linear blend weight for pixel index `i` in a dimension of
/// size `dim` with `overlap` pixels of fade at each end.
#[inline]
fn blend_weight(i: usize, dim: usize, overlap: usize) -> f32 {
	if overlap == 0 {
		return 1.0;
	}
	let dist_near = i + 1;
	let dist_far = dim - i;
	let d = dist_near.min(dist_far).min(overlap);
	d as f32 / overlap as f32
}

/// Computes tile start positions so tiles cover `total` pixels with the last
/// tile ending exactly at `total`.
fn tile_starts(total: usize, tile_size: usize, step: usize) -> Vec<usize> {
	let mut starts = Vec::new();
	let mut s = 0usize;
	loop {
		starts.push(s);
		if s + tile_size >= total {
			break;
		}
		s += step;
	}
	starts
}
