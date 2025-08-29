use profiling::TrackingAllocator;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

#[macro_use]
extern crate alumina;
extern crate clap;
extern crate image;
extern crate rand;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate serde_derive;
extern crate bincode;
extern crate byteorder;
extern crate indexmap;
extern crate serde;
extern crate smallvec;
extern crate xz2;

pub mod aligned_crop;
pub mod benchmarks;
pub mod cli;
pub mod commands;
pub mod config;
pub mod config_file;
pub mod constants;
pub mod error;
pub mod gpu;
pub mod logging;
pub mod model_converter;
pub mod network;
pub mod parallel;
pub mod psnr;
pub mod profiling;
pub mod thread_safe_network;
pub mod training;
pub mod utils;
pub mod validation;
pub mod video;
pub mod web_server;

#[cfg(test)]
mod error_handling_tests;

use std::{
	fmt,
	fs::*,
	io::{stdout, Read, Write},
	num::FpCategory,
};

use bincode::{deserialize, serialize, DefaultOptions, Options};
use image::{ImageFormat, ImageResult};

pub use network::*;

use byteorder::{BigEndian, ByteOrder};
use ndarray::{ArrayD, Axis, IxDyn};
use xz2::read::{XzDecoder, XzEncoder};

pub use alumina::{
	data::image_folder::{data_to_image, image_to_data},
	graph::GraphDef,
};

pub const L1_SRGB_NATURAL_PARAMS: &[u8] = include_bytes!("res/L1_x4_UCID_x1node.rsr");
pub const L1_SRGB_ANIME_PARAMS: &[u8] = include_bytes!("res/L1_x4_Anime_x1node.rsr");

/// A struct containing the network parameters and hyperparameters.
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkDescriptionOld {
	pub factor: u32,
	pub log_depth: u32,
	pub global_node_factor: u32,
	pub parameters: Vec<ArrayD<f32>>,
}

/// Decompresses and deserialises the NetworkDescription from the byte format used in .rsr. files
pub fn old_network_from_bytes(data: &[u8]) -> ::std::result::Result<NetworkDescriptionOld, String> {
	let decompressed = XzDecoder::new(data)
		.bytes()
		.collect::<::std::result::Result<Vec<_>, _>>()
		.map_err(|e| format!("{}", e))?;
	let unshuffled = unshuffle(&decompressed, 4);
	let deserialized: NetworkDescriptionOld =
		deserialize(&unshuffled).map_err(|e| format!("NetworkDescription decoding failed: {}", e))?;
	Ok(deserialized)
}

/// A struct containing the network parameters and hyperparameters.
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkDescription {
	pub factor: u32,
	pub width: u32,
	pub log_depth: u32,
	pub global_node_factor: u32,
	pub parameters: Vec<ArrayD<f32>>,
}

/// Decompresses and deserialises the NetworkDescription from the byte format used in .rsr. files
pub fn network_from_bytes(data: &[u8]) -> ::std::result::Result<NetworkDescription, String> {
	let decompressed = XzDecoder::new(data)
		.bytes()
		.collect::<::std::result::Result<Vec<_>, _>>()
		.map_err(|e| format!("{}", e))?;
	let unshuffled = unshuffle(&decompressed, 4);
	
	// Try with different bincode configurations to handle version compatibility
	let options = DefaultOptions::new()
		.with_fixint_encoding()
		.allow_trailing_bytes();
	
	// Try to deserialize with compatibility options
	let deserialized: NetworkDescription = options
		.deserialize(&unshuffled)
		.or_else(|_| {
			// Fallback to standard deserialize
			deserialize(&unshuffled)
		})
		.or_else(|_| {
			// Try as old format and convert
			if let Ok(old_desc) = old_network_from_bytes(data) {
				Ok(NetworkDescription {
					factor: old_desc.factor,
					width: 32, // Default width for old models
					log_depth: old_desc.log_depth,
					global_node_factor: old_desc.global_node_factor,
					parameters: old_desc.parameters,
				})
			} else {
				Err("Failed to deserialize with any method".to_string())
			}
		})
		.map_err(|e| format!("NetworkDescription decoding failed: {}", e))?;
	Ok(deserialized)
}

/// Serialises and compresses the NetworkDescription returning the byte format used in .rsr files
/// If quantise = true, then the least significant 12 bits are zeroed to improve compression.
pub fn network_to_bytes(mut desc: NetworkDescription, quantise: bool) -> ::std::result::Result<Vec<u8>, String> {
	for arr in &mut desc.parameters {
		for e in arr.iter_mut() {
			if let FpCategory::Subnormal = e.classify() {
				*e = 0.0;
			}
			if quantise {
				let mut bytes = [0; 4];
				BigEndian::write_f32(&mut bytes, *e);
				bytes[2] &= 0xF0;
				bytes[3] &= 0x00;
				*e = BigEndian::read_f32(&bytes);
			}
		}
	}

	let serialized: Vec<u8> = serialize(&desc).map_err(|e| format!("NetworkDescription encoding failed: {}", e))?;
	let shuffled = shuffle(&serialized, 4);
	let compressed = XzEncoder::new(shuffled.as_slice(), 7)
		.bytes()
		.collect::<::std::result::Result<Vec<_>, _>>()
		.map_err(|e| format!("{}", e))?;
	Ok(compressed)
}

/// Shuffle f32 bytes so that all first bytes are contiguous etc
/// Improves compression of floating point data
fn shuffle(data: &[u8], stride: usize) -> Vec<u8> {
	let mut vec = Vec::with_capacity(data.len());
	for offset in 0..stride {
		for i in 0..(data.len() - offset + stride - 1) / stride {
			vec.push(data[offset + i * stride])
		}
	}
	debug_assert_eq!(vec.len(), data.len());
	vec
}

/// Inverts `shuffle()`
fn unshuffle(data: &[u8], stride: usize) -> Vec<u8> {
	let mut vec = vec![0; data.len()];
	let mut inc = 0;
	for offset in 0..stride {
		for i in 0..(data.len() - offset + stride - 1) / stride {
			vec[offset + i * stride] = data[inc];
			inc += 1;
		}
	}
	debug_assert_eq!(inc, data.len());
	vec
}

/// Returns an rgb image converted to a tensor of floats in the range of [0, 1] and of shape, [1, H, W, 3];
pub fn read(file: &mut File) -> ImageResult<ArrayD<f32>> {
	let mut vec = vec![];
	file.read_to_end(&mut vec)
		.map_err(|err| image::ImageError::IoError(err))?;
	let input_image = image::load_from_memory(&vec)?;
	let input = image_to_data(&input_image);
	let shape = input.shape().to_vec();
	let input = input.into_shape(IxDyn(&[1, shape[0], shape[1], shape[2]]))
		.map_err(|_| image::ImageError::DimensionError)?;
	Ok(input)
}

/// Save tensor of shape [1, H, W, 3] as .png image. Converts floats in range of [0, 1] to bytes in range [0, 255].
pub fn save(image: ArrayD<f32>, file: &mut File) -> ImageResult<()> {
	stdout().flush().ok();
	data_to_image(image.subview(Axis(0), 0)).write_to(file, ImageFormat::PNG)
}

/// Takes an image tensor of shape [1, H, W, 3] and returns one of shape [1, H/factor, W/factor, 3].
///
/// If `sRGB` is `true` the pooling operation averages over the image values as stored,
/// if `false` then the sRGB values are temporarily converted to linear RGB, pooled, then converted back.
#[allow(non_snake_case)]
pub fn downscale(image: ArrayD<f32>, factor: usize, sRGB: bool) -> alumina::graph::Result<ArrayD<f32>> {
	let graph = if sRGB {
		downscale_srgb_net(factor)?
	} else {
		downscale_lin_net(factor)?
	};

	let input_id = graph.node_id("input").value_id();
	let output_id = graph.node_id("output").value_id();
	let mut subgraph = graph.subgraph(&[input_id.clone()], &[output_id.clone()])?;
	let result = subgraph.execute(vec![image])?;

	result.into_map().remove(&output_id)
		.ok_or_else(|| alumina::graph::Error::from("Output node not found in downscale graph result"))
}

/// A container type for upscaling networks
#[derive(Clone, Debug)]
pub struct UpscalingNetwork {
	graph: GraphDef,
	parameters: Vec<ArrayD<f32>>,
	display: String,
}

impl UpscalingNetwork {
	pub fn new(desc: NetworkDescription, display: &str) -> ::std::result::Result<Self, String> {
		Ok(UpscalingNetwork {
			graph: inference_sr_net(
				desc.factor as usize,
				desc.width,
				desc.log_depth,
				desc.global_node_factor as usize,
			)
			.map_err(|e| format!("{}", e))?,
			parameters: desc.parameters,
			display: display.into(),
		})
	}
	
	/// Create network from config  
	pub fn new_from_config(config: config::NetworkConfig) -> ::std::result::Result<Self, crate::error::SrganError> {
		let desc = NetworkDescription {
			factor: 4,  // Default scale factor
			width: config.width,
			log_depth: config.log_depth,
			global_node_factor: 1,
			parameters: Vec::new(),
		};
		Self::new(desc, "custom network")
			.map_err(|e| crate::error::SrganError::Network(e))
	}

	/// Accepts labels: [natural, natural_L1, natural_rgb, anime, anime_L1, bilinear]
	pub fn from_label(label: &str, bilinear_factor: Option<usize>) -> ::std::result::Result<Self, String> {
		match label {
			"natural" => {
				let desc = network_from_bytes(L1_SRGB_NATURAL_PARAMS)?;
				Ok(UpscalingNetwork {
					graph: inference_sr_net(
						desc.factor as usize,
						desc.width,
						desc.log_depth,
						desc.global_node_factor as usize,
					)
					.map_err(|e| format!("{}", e))?,
					parameters: desc.parameters,
					display: "neural net trained on natural images with an L1 loss".into(),
				})
			},
			"anime" => {
				let desc = network_from_bytes(L1_SRGB_ANIME_PARAMS)?;
				Ok(UpscalingNetwork {
					graph: inference_sr_net(
						desc.factor as usize,
						desc.width,
						desc.log_depth,
						desc.global_node_factor as usize,
					)
					.map_err(|e| format!("{}", e))?,
					parameters: desc.parameters,
					display: "neural net trained on animation images with an L1 loss".into(),
				})
			},
			"bilinear" => Ok(UpscalingNetwork {
				graph: bilinear_net(bilinear_factor.unwrap_or(4)).map_err(|e| format!("{}", e))?,
				parameters: Vec::new(),
				display: "bilinear interpolation".into(),
			}),
			_ => Err(format!("Unsupported network type. Could not parse: {}", label)),
		}
	}

	pub fn borrow_network(&self) -> (&GraphDef, &[ArrayD<f32>]) {
		(&self.graph, &self.parameters)
	}
	
	/// Load network from file
	pub fn load_from_file(path: &std::path::Path) -> ::std::result::Result<Self, crate::error::SrganError> {
		use std::fs::File;
		use std::io::Read;
		
		let mut file = File::open(path)
			.map_err(|e| crate::error::SrganError::Io(e))?;
		let mut data = Vec::new();
		file.read_to_end(&mut data)
			.map_err(|e| crate::error::SrganError::Io(e))?;
			
		let desc = network_from_bytes(&data)
			.map_err(|e| crate::error::SrganError::Network(e))?;
			
		Self::new(desc, "custom network")
			.map_err(|e| crate::error::SrganError::Network(e))
	}
	
	/// Save network to file
	pub fn save_to_file(&self, path: &std::path::Path) -> ::std::result::Result<(), crate::error::SrganError> {
		use std::fs::File;
		use std::io::Write;
		
		let desc = NetworkDescription {
			factor: 4,  // Default
			width: 12,  // Default
			log_depth: 4,  // Default  
			global_node_factor: 1,
			parameters: self.parameters.clone(),
		};
		
		let data = network_to_bytes(desc, false)
			.map_err(|e| crate::error::SrganError::Serialization(e))?;
			
		let mut file = File::create(path)
			.map_err(|e| crate::error::SrganError::Io(e))?;
		file.write_all(&data)
			.map_err(|e| crate::error::SrganError::Io(e))?;
			
		Ok(())
	}
	
	/// Load built-in natural model
	pub fn load_builtin_natural() -> ::std::result::Result<Self, crate::error::SrganError> {
		Self::from_label("natural", None)
			.map_err(|e| crate::error::SrganError::Network(e))
	}
	
	/// Load built-in anime model
	pub fn load_builtin_anime() -> ::std::result::Result<Self, crate::error::SrganError> {
		Self::from_label("anime", None)
			.map_err(|e| crate::error::SrganError::Network(e))
	}
	
	/// Upscale an image
	pub fn upscale_image(&self, img: &image::DynamicImage) -> ::std::result::Result<image::DynamicImage, crate::error::SrganError> {
		// Convert image to tensor
		let tensor = image_to_data(img);
		
		// Upscale
		let result = upscale(tensor, self)
			.map_err(|e| crate::error::SrganError::GraphExecution(format!("{}", e)))?;
		
		// Convert back to image
		let upscaled_img = data_to_image(result.view());
		
		Ok(upscaled_img)
	}
}

impl fmt::Display for UpscalingNetwork {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", self.display)
	}
}

/// Takes an image tensor of shape [1, H, W, 3] and returns one of shape [1, H*factor, W*factor, 3], where factor is
/// determined by the network definition.
///
/// The exact results of the upscaling depend on the content on which the network being used was trained, and what loss
/// it was trained to minimise. L2 loss maximises PSNR, where as L1 loss results in sharper edges.
/// `bilinear_factor` is ignored unless the network is Bilinear.
pub fn upscale(image: ArrayD<f32>, network: &UpscalingNetwork) -> alumina::graph::Result<ArrayD<f32>> {
	let (graph, params) = network.borrow_network();

	let mut input_vec = vec![image];
	input_vec.extend(params.iter().cloned());
	let input_id = graph.node_id("input").value_id();
	let param_ids: Vec<_> = graph.parameter_ids().iter().map(|node_id| node_id.value_id()).collect();
	let mut subgraph_inputs = vec![input_id];
	subgraph_inputs.extend(param_ids);
	let output_id = graph.node_id("output").value_id();
	let mut subgraph = graph.subgraph(&subgraph_inputs, &[output_id.clone()])?;
	let result = subgraph.execute(input_vec)?;

	result.into_map().remove(&output_id)
		.ok_or_else(|| alumina::graph::Error::from("Output node not found in upscale graph result"))
}
