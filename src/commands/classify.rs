use crate::error::{Result, SrganError};
use crate::image_classifier;
use clap::ArgMatches;
use std::path::Path;

pub fn classify(app_m: &ArgMatches) -> Result<()> {
    let image_path = app_m
        .value_of("IMAGE")
        .ok_or_else(|| SrganError::InvalidParameter("No IMAGE file given".to_string()))?;

    let result = image_classifier::classify_path(Path::new(image_path))?;

    if app_m.is_present("json") {
        // JSON output
        let json = serde_json::json!({
            "file": image_path,
            "detected_type": result.detected_type.slug(),
            "display_name": result.detected_type.display_name(),
            "confidence": result.confidence,
            "recommended_model": result.recommended_model,
            "reasoning": result.reasoning,
        });
        println!("{}", serde_json::to_string_pretty(&json)
            .unwrap_or_else(|_| "{}".to_string()));
    } else {
        println!("File:              {}", image_path);
        println!("Detected type:     {}", result.detected_type.display_name());
        println!("Confidence:        {:.0}%", result.confidence * 100.0);
        println!("Recommended model: {}", result.recommended_model);
        println!("Reasoning:         {}", result.reasoning);
    }

    Ok(())
}
