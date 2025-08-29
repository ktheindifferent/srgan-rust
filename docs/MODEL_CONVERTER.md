# Model Converter Documentation

The SRGAN-Rust model converter allows you to import pre-trained models from popular deep learning frameworks into the native SRGAN-Rust format. This enables you to leverage existing models trained in PyTorch, TensorFlow, ONNX, or Keras.

## Supported Formats

### PyTorch Models (.pth, .pt)

The converter fully supports PyTorch model files saved using `torch.save()`. 

#### Supported Features:
- **State Dict Format**: Models saved as state dictionaries (recommended)
- **Pickle Protocol**: All pickle protocol versions (0-5)
- **Data Types**: float32, float64 (converted to f32), int64 (converted to f32)
- **Tensor Storage**: List format, bytes format, nested dictionary format
- **Architecture Detection**: Automatic detection of SRGAN, SRResNet, and custom architectures
- **Batch Normalization**: Full support for batch norm layers with running statistics

#### File Format Requirements:
- Files must be in pickle format (not ZIP archives)
- If your model is saved with `torch.save(..., _use_new_zipfile_serialization=True)`, extract it first
- Minimum file size: 16 bytes

#### Usage Example:
```rust
use srgan_rust::model_converter::ModelConverter;
use std::path::Path;

let mut converter = ModelConverter::new();
converter.load_pytorch(Path::new("model.pth"))?;

// Get model metadata
let stats = converter.get_conversion_stats();
println!("Architecture: {}", stats.get("architecture").unwrap());
println!("Parameters: {}", stats.get("parameter_count").unwrap());
```

### TensorFlow Models (.pb)

Support for TensorFlow SavedModel format (protobuf).

#### Requirements:
- SavedModel directory structure with `saved_model.pb`
- Currently supports inference graphs only

### ONNX Models (.onnx)

Support for ONNX (Open Neural Network Exchange) format.

#### Features:
- Cross-framework compatibility
- Optimized for inference

### Keras Models (.h5, .hdf5)

Support for Keras HDF5 format models.

## Model Architecture Detection

The converter automatically detects the model architecture based on layer naming patterns:

| Architecture | Detection Criteria | Description |
|-------------|-------------------|-------------|
| `srgan_full` | Contains both "generator" and "discriminator" layers | Complete SRGAN model with both networks |
| `srgan_generator` | Contains "generator" layers only | Generator network for inference |
| `srresnet` | Contains "residual" or "res" blocks | Super-Resolution ResNet architecture |
| `srgan` | Default for unrecognized patterns | Generic SRGAN architecture |

## Error Handling

The converter provides comprehensive error handling for common issues:

### Common Errors and Solutions:

1. **"PyTorch file too small to be valid"**
   - File is empty or corrupted
   - Solution: Re-save the model properly

2. **"Detected ZIP format (likely torch.save with compression)"**
   - Model saved with ZIP compression
   - Solution: Extract the model first or save without compression

3. **"Failed to parse PyTorch pickle"**
   - Corrupted pickle file or unsupported format
   - Solution: Verify the model file integrity

4. **"No parameters found in PyTorch model"**
   - Empty state dictionary
   - Solution: Ensure model was saved with state_dict

5. **"Invalid tensor byte length"**
   - Corrupted tensor data
   - Solution: Re-save the model

## Performance Considerations

- **Memory Usage**: The converter loads the entire model into memory
- **Large Models**: Models with 100M+ parameters are supported and load in < 10 seconds
- **Validation**: All weights are validated for NaN and Inf values
- **Logging**: Detailed logging available with `RUST_LOG=debug`

## PyTorch Model Preparation

To prepare your PyTorch model for conversion:

```python
import torch

# Save just the state dict (recommended)
model = YourModel()
torch.save(model.state_dict(), 'model.pth')

# Or save the entire model
torch.save(model, 'model_full.pth')

# Avoid ZIP format for better compatibility
torch.save(model.state_dict(), 'model.pth', 
           _use_new_zipfile_serialization=False)
```

## Supported Layer Types

### Fully Supported:
- Convolutional layers (Conv2d, Conv3d)
- Linear/Dense layers
- Batch Normalization
- Instance Normalization
- Activation functions
- Pooling layers
- Upsampling layers
- Residual blocks

### Partial Support:
- Quantized layers (int8) - experimental
- Custom layers - may require manual mapping

## Integration with SRGAN-Rust

After conversion, the model can be used directly with the SRGAN-Rust upscaling network:

```rust
let mut converter = ModelConverter::new();
converter.load_pytorch(Path::new("model.pth"))?;

// Convert to native format
let network = converter.to_upscaling_network()?;

// Use for inference
let upscaled = network.upscale(&input_image)?;
```

## Troubleshooting

### Debug Mode
Enable debug logging to see detailed parsing information:
```bash
RUST_LOG=debug cargo run
```

### Validation
The converter validates:
- File format and magic bytes
- Tensor dimensions and shapes
- Weight values (checks for NaN/Inf)
- Layer compatibility

### Testing
Run the comprehensive test suite:
```bash
cargo test model_converter
cargo test pytorch_integration
```

## Future Enhancements

Planned improvements:
- [ ] Support for ZIP-compressed PyTorch models
- [ ] Direct TensorFlow 2.x support
- [ ] ONNX operator coverage expansion
- [ ] Automatic quantization support
- [ ] Model optimization during conversion
- [ ] Batch conversion tools

## Contributing

To add support for new model formats or improve existing converters:
1. Implement the parsing logic in `src/model_converter.rs`
2. Add comprehensive tests in `tests/model_converter_tests.rs`
3. Update this documentation
4. Submit a pull request

For questions or issues, please open a GitHub issue.