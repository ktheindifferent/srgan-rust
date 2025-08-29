use std::path::Path;
use clap::ArgMatches;
use log::{info, warn};
use crate::error::SrganError;
use crate::model_converter::{ModelConverter, ModelFormat, batch_convert_models};

/// Convert external model to SRGAN-Rust format
pub fn convert_model(matches: &ArgMatches) -> Result<(), SrganError> {
    let input_path = Path::new(matches.value_of("input")
        .ok_or_else(|| SrganError::InvalidParameter("Input path is required".to_string()))?);
    let output_path = Path::new(matches.value_of("output").unwrap_or("converted_model.rsr"));
    
    // Parse format if specified
    let format = matches.value_of("format").map(|f| match f {
        "pytorch" => ModelFormat::PyTorch,
        "tensorflow" => ModelFormat::TensorFlow,
        "onnx" => ModelFormat::ONNX,
        "keras" => ModelFormat::Keras,
        _ => unreachable!(),
    });
    
    // Check if batch mode
    if matches.is_present("batch") {
        info!("Starting batch conversion from {:?}", input_path);
        
        if !input_path.is_dir() {
            return Err(SrganError::InvalidInput(
                "Batch mode requires input to be a directory".to_string()
            ));
        }
        
        let output_dir = if output_path.exists() && output_path.is_dir() {
            output_path
        } else {
            Path::new("converted_models")
        };
        
        // Create output directory if needed
        std::fs::create_dir_all(output_dir)
            .map_err(|e| SrganError::Io(e))?;
        
        let results = batch_convert_models(input_path, output_dir, format)?;
        
        // Print summary
        let successful = results.iter().filter(|(_, success)| *success).count();
        let failed = results.len() - successful;
        
        info!("Batch conversion complete:");
        info!("  Successful: {}", successful);
        info!("  Failed: {}", failed);
        
        for (file, success) in &results {
            if !success {
                warn!("  Failed: {}", file);
            }
        }
        
        if failed > 0 {
            return Err(SrganError::InvalidInput(
                format!("{} models failed to convert", failed)
            ));
        }
    } else {
        // Single file conversion
        if !input_path.exists() {
            return Err(SrganError::FileNotFound(input_path.to_path_buf()));
        }
        
        info!("Converting model from {:?} to {:?}", input_path, output_path);
        
        let mut converter = ModelConverter::new();
        
        // Auto-detect or use specified format
        let model_format = format.unwrap_or(ModelConverter::auto_detect_format(input_path)?);
        
        info!("Detected format: {:?}", model_format);
        
        // Load the model
        match model_format {
            ModelFormat::PyTorch => {
                info!("Loading PyTorch model...");
                converter.load_pytorch(input_path)?;
            },
            ModelFormat::TensorFlow => {
                info!("Loading TensorFlow model...");
                converter.load_tensorflow(input_path)?;
            },
            ModelFormat::ONNX => {
                info!("Loading ONNX model...");
                converter.load_onnx(input_path)?;
            },
            ModelFormat::Keras => {
                info!("Loading Keras model...");
                converter.load_keras(input_path)?;
            },
        }
        
        // Display conversion statistics
        let stats = converter.get_conversion_stats();
        info!("Model statistics:");
        for (key, value) in &stats {
            info!("  {}: {}", key, value);
        }
        
        // Validate if requested
        if matches.is_present("validate") {
            info!("Validating conversion...");
            let network = converter.convert_to_srgan()?;
            let valid = converter.validate_conversion(input_path, &network)?;
            
            if valid {
                info!("✓ Conversion validated successfully");
            } else {
                warn!("⚠ Conversion validation failed - results may be incorrect");
            }
        }
        
        // Save converted model
        converter.save_converted(output_path)?;
        
        info!("✓ Model converted successfully to {:?}", output_path);
    }
    
    Ok(())
}

/// List supported model formats
pub fn list_formats(_matches: &ArgMatches) -> Result<(), SrganError> {
    println!("Supported model formats for conversion:");
    println!();
    println!("  pytorch    (.pth, .pt)   - PyTorch saved models");
    println!("  tensorflow (.pb)         - TensorFlow SavedModel format");
    println!("  onnx       (.onnx)       - Open Neural Network Exchange format");
    println!("  keras      (.h5, .hdf5)  - Keras/TensorFlow H5 format");
    println!();
    println!("Usage examples:");
    println!("  # Auto-detect format");
    println!("  srgan-rust convert-model model.pth converted.rsr");
    println!();
    println!("  # Specify format explicitly");
    println!("  srgan-rust convert-model model.pb converted.rsr --format tensorflow");
    println!();
    println!("  # Batch conversion");
    println!("  srgan-rust convert-model ./models/ ./converted/ --batch");
    println!();
    println!("  # With validation");
    println!("  srgan-rust convert-model model.onnx converted.rsr --validate");
    
    Ok(())
}