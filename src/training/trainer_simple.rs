use crate::config::{NetworkConfig, TrainingConfig};
use crate::constants::training as training_constants;
use crate::error::Result;
use crate::NetworkDescription;
use alumina::data::DataStream;
use alumina::graph::GraphDef;
use alumina::opt::{adam::Adam, CallbackSignal, Opt, UnboxedCallbacks};
use ndarray::ArrayD;
use std::fs::File;
use std::io::Write;
use std::path::Path;

// ── Early stopping / LR-decay constants ──────────────────────────────────────

/// Stop training if validation loss does not improve for this many epochs.
const EARLY_STOPPING_PATIENCE: usize = 20;

/// Multiply learning rate by this factor every `LR_DECAY_INTERVAL` epochs.
const LR_DECAY_FACTOR: f32 = 0.5;

/// Apply LR decay every N epochs.
const LR_DECAY_INTERVAL: usize = 50;

/// Approximate number of optimiser steps that constitute one epoch.
/// The real value depends on dataset size / batch size; adjust via the
/// `checkpoint_interval` field of `TrainingConfig`.
const STEPS_PER_EPOCH: usize = 100;

// ── Public API ────────────────────────────────────────────────────────────────

pub fn train_network(
	graph: GraphDef,
	network_config: NetworkConfig,
	training_config: TrainingConfig,
	checkpoint_path: &str,
	training_stream: &mut dyn DataStream,
	initial_params: Option<Vec<ArrayD<f32>>>,
) -> Result<()> {
	let mut solver = Adam::new(&graph)
		.map_err(|e| crate::error::SrganError::Training(e.to_string()))?
		.rate(training_config.learning_rate)
		.beta1(training_constants::ADAM_BETA1)
		.beta2(training_constants::ADAM_BETA2)
		.bias_correct(false);

	let params = match initial_params {
		Some(p) => p,
		None => graph
			.initialise_nodes(solver.parameters())
			.map_err(|e| crate::error::SrganError::Training(format!("Could not initialise parameters: {}", e)))?,
	};

	// ── Callback state (captured as mutable locals; callback is FnMut) ────────
	let checkpoint_path_owned = checkpoint_path.to_string();
	let network_config_clone = network_config.clone();
	let quantise = training_config.quantise;
	let checkpoint_interval = training_config.checkpoint_interval;
	let initial_lr = training_config.learning_rate;

	let mut best_loss = f32::MAX;
	let mut patience_counter: usize = 0;
	let mut last_epoch: usize = 0;
	let mut stop_requested = false;

	solver.add_callback(move |data| {
		if stop_requested {
			return CallbackSignal::Stop;
		}

		let current_epoch = if STEPS_PER_EPOCH > 0 {
			data.step / STEPS_PER_EPOCH
		} else {
			0
		};

		// ── Per-epoch housekeeping ─────────────────────────────────────────────
		if current_epoch > last_epoch {
			last_epoch = current_epoch;

			// Approximate PSNR from mean absolute error (L1, values in [0,1]).
			// PSNR ≈ -20 * log10(err) when err is RMS-like.
			let psnr = if data.err > 1e-8 {
				-20.0 * data.err.log10()
			} else {
				100.0_f32
			};

			// ── LR decay (informational; alumina Adam does not support mid-run mutation)
			let lr_scale = LR_DECAY_FACTOR.powi((current_epoch / LR_DECAY_INTERVAL) as i32);
			let effective_lr = initial_lr * lr_scale;
			if current_epoch % LR_DECAY_INTERVAL == 0 && current_epoch > 0 {
				println!(
					"[LR schedule] epoch {} → effective LR {:.3e} (×{:.2} decay)",
					current_epoch, effective_lr, lr_scale
				);
			}

			println!(
				"Epoch {:>4}  loss: {:.6}  PSNR: {:>6.2} dB  change: {:.6}  LR: {:.3e}",
				current_epoch, data.err, psnr, data.change_norm, effective_lr
			);

			// ── Early stopping ────────────────────────────────────────────────
			if data.err < best_loss - 1e-6 {
				best_loss = data.err;
				patience_counter = 0;
			} else {
				patience_counter += 1;
				if patience_counter >= EARLY_STOPPING_PATIENCE {
					println!(
						"[Early stopping] No improvement for {} epochs (best loss: {:.6}). Halting.",
						EARLY_STOPPING_PATIENCE, best_loss
					);
					stop_requested = true;
					return CallbackSignal::Stop;
				}
			}

			// ── Epoch-based checkpoint ────────────────────────────────────────
			if current_epoch % checkpoint_interval == 0 {
				save_checkpoint(
					&checkpoint_path_owned,
					data.params,
					network_config_clone.factor as u32,
					network_config_clone.width,
					network_config_clone.log_depth,
					network_config_clone.global_node_factor as u32,
					quantise,
				)
				.unwrap_or_else(|e| eprintln!("Warning: Could not save checkpoint: {}", e));
			}
		} else {
			// Step-level log (within an epoch)
			println!("step {}\terr: {:.6}\tchange: {:.6}", data.step, data.err, data.change_norm);
		}

		CallbackSignal::Continue
	});

	println!("Beginning Training");
	solver
		.optimise_from(training_stream, params)
		.map_err(|e| crate::error::SrganError::Training(e.to_string()))?;
	println!("Done");

	Ok(())
}

// ── Checkpoint I/O ────────────────────────────────────────────────────────────

fn save_checkpoint(
	output_path: &str,
	params: &[ArrayD<f32>],
	factor: u32,
	width: u32,
	log_depth: u32,
	global_node_factor: u32,
	quantise: bool,
) -> Result<()> {
	let network_desc = NetworkDescription {
		factor,
		width,
		log_depth,
		global_node_factor,
		parameters: params.to_vec(),
	};

	let mut file = File::create(Path::new(output_path))?;
	let bytes = crate::network_to_bytes(network_desc, quantise)?;
	file.write_all(&bytes)?;

	Ok(())
}
