use crate::config_file::TrainingConfigFile;
use crate::error::{Result, SrganError};
use clap::ArgMatches;
use log::info;
use std::fs;
use std::path::Path;

pub fn generate_config(app_m: &ArgMatches) -> Result<()> {
    let output_path = app_m
        .value_of("OUTPUT_FILE")
        .unwrap_or("training_config.toml");
    
    let format = app_m.value_of("FORMAT").unwrap_or("toml");
    let example = app_m.is_present("EXAMPLE");
    
    // Check if file exists and warn
    if Path::new(output_path).exists() && !app_m.is_present("FORCE") {
        return Err(SrganError::InvalidInput(format!(
            "File {} already exists. Use --force to overwrite",
            output_path
        )));
    }
    
    if example {
        // Generate example configuration with comments (TOML only)
        if format != "toml" {
            return Err(SrganError::InvalidParameter(
                "Example configuration with comments is only available in TOML format".to_string()
            ));
        }
        
        let example_config = TrainingConfigFile::create_example_toml();
        fs::write(output_path, example_config)
            .map_err(|e| SrganError::Io(e))?;
        
        info!("Generated example configuration file with comments: {}", output_path);
    } else {
        // Generate default configuration
        let config = TrainingConfigFile::generate_default();
        
        match format {
            "toml" => {
                config.to_toml_file(output_path)?;
                info!("Generated TOML configuration file: {}", output_path);
            }
            "json" => {
                config.to_json_file(output_path)?;
                info!("Generated JSON configuration file: {}", output_path);
            }
            _ => {
                return Err(SrganError::InvalidParameter(format!(
                    "Unknown format: {}. Use 'toml' or 'json'",
                    format
                )));
            }
        }
    }
    
    info!("You can now edit the configuration file and use it with:");
    info!("  srgan-rust train-config {}", output_path);
    
    Ok(())
}

pub fn train_with_config(app_m: &ArgMatches) -> Result<()> {
    let config_path = app_m
        .value_of("CONFIG_FILE")
        .ok_or_else(|| SrganError::InvalidParameter("No configuration file specified".to_string()))?;
    
    // Detect format from extension
    let config = if config_path.ends_with(".json") {
        TrainingConfigFile::from_json_file(config_path)?
    } else {
        TrainingConfigFile::from_toml_file(config_path)?
    };
    
    info!("Loaded configuration from: {}", config_path);
    
    // Convert to internal config structures
    let network_config = config.to_network_config();
    let training_config = config.to_training_config()?;
    
    // Validate configurations
    network_config.validate()?;
    training_config.validate()?;
    
    // Log configuration details
    info!("Network configuration:");
    info!("  Factor: {}", network_config.factor);
    info!("  Width: {}", network_config.width);
    info!("  Log depth: {}", network_config.log_depth);
    info!("  Global node factor: {}", network_config.global_node_factor);
    
    info!("Training configuration:");
    info!("  Learning rate: {}", training_config.learning_rate);
    info!("  Batch size: {}", training_config.batch_size);
    info!("  Patch size: {}", training_config.patch_size);
    info!("  Loss type: {:?}", training_config.loss_type);
    
    // Create the training graph
    let (power, scale) = training_config.get_loss_params();
    let graph = crate::network::training_sr_net(
        network_config.factor,
        network_config.width,
        network_config.log_depth,
        0,
        crate::constants::training::REGULARIZATION_EPSILON,
        power,
        scale,
        training_config.srgb_downscale,
    )
    .map_err(|e| SrganError::GraphConstruction(e.to_string()))?;
    
    // Create checkpoint directory if it doesn't exist
    let checkpoint_dir = Path::new(&config.output.checkpoint_dir);
    if !checkpoint_dir.exists() {
        fs::create_dir_all(checkpoint_dir)
            .map_err(|e| SrganError::Io(e))?;
        info!("Created checkpoint directory: {}", checkpoint_dir.display());
    }
    
    // Load initial parameters if specified
    let initial_params = if let Some(start_params) = app_m.value_of("START_PARAMETERS") {
        info!("Loading initial parameters from: {}", start_params);
        let mut param_file = fs::File::open(start_params)?;
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut param_file, &mut data)?;
        let network_desc = crate::network_from_bytes(&data)?;
        Some(network_desc.1)
    } else {
        None
    };
    
    // Create training data stream
    let mut training_stream = crate::training::DataLoader::create_training_stream(
        &config.data.training_folder,
        config.data.recurse,
        training_config.patch_size,
        network_config.factor,
        training_config.batch_size,
    );
    
    // Run training
    info!("Starting training with configuration file...");
    crate::training::train_network(
        graph,
        network_config,
        training_config,
        &config.output.parameter_file,
        &mut *training_stream,
        initial_params,
    )?;
    
    info!("Training completed. Model saved to: {}", config.output.parameter_file);
    
    Ok(())
}