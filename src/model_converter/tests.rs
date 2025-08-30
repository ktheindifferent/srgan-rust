use super::*;
use tempfile::{TempDir, NamedTempFile};
use std::io::Write;

#[cfg(test)]
mod tensorflow_tests {
    use super::super::tensorflow::*;
    use super::*;
    
    #[test]
    fn test_tensorflow_parser_initialization() {
        let parser = TensorFlowParser::new();
        assert!(parser.graph_def.is_none());
        assert!(parser.variables.is_empty());
    }
    
    #[test]
    fn test_tensor_proto_parsing() {
        let parser = TensorFlowParser::new();
        
        let tensor = TensorProto {
            dtype: 1,  // Float32
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    TensorShapeDim { size: 2, name: String::new() },
                    TensorShapeDim { size: 3, name: String::new() },
                ],
            }),
            version_number: 0,
            tensor_content: vec![],
            float_val: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_val: vec![],
            int_val: vec![],
            int64_val: vec![],
        };
        
        let result = parser.parse_tensor_proto(&tensor, "test_tensor");
        assert!(result.is_ok());
        
        let tensor_data = result.unwrap();
        assert_eq!(tensor_data.name, "test_tensor");
        assert_eq!(tensor_data.shape, vec![2, 3]);
        assert_eq!(tensor_data.data.len(), 6);
    }
    
    #[test]
    fn test_raw_bytes_parsing() {
        let parser = TensorFlowParser::new();
        
        // Create float32 bytes
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut bytes = Vec::new();
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        
        let result = parser.parse_tensor_content(&bytes, super::super::common::DataType::Float32);
        assert!(result.is_ok());
        
        let parsed = result.unwrap();
        assert_eq!(parsed, values);
    }
    
    #[test]
    fn test_model_info_extraction() {
        let parser = TensorFlowParser::new();
        let info = parser.get_model_info();
        
        assert_eq!(info.format, "tensorflow");
        assert_eq!(info.version, "2.0");
        assert_eq!(info.total_parameters, 0);
    }
}

#[cfg(test)]
mod onnx_tests {
    use super::super::onnx::*;
    use super::*;
    
    #[test]
    fn test_onnx_parser_initialization() {
        let parser = OnnxParser::new();
        assert!(parser.model.is_none());
        assert!(parser.weights.is_empty());
    }
    
    #[test]
    fn test_tensor_proto_conversion() {
        let parser = OnnxParser::new();
        
        let tensor = TensorProto {
            dims: vec![1, 3, 256, 256],
            data_type: 1,  // Float32
            name: "conv1.weight".into(),
            raw_data: vec![],
            float_data: vec![0.1, 0.2, 0.3],
            int32_data: vec![],
            int64_data: vec![],
            double_data: vec![],
        };
        
        let result = parser.parse_tensor_proto(&tensor);
        assert!(result.is_ok());
        
        let tensor_data = result.unwrap();
        assert_eq!(tensor_data.name, "conv1.weight");
        assert_eq!(tensor_data.shape, vec![1, 3, 256, 256]);
        assert_eq!(tensor_data.data, vec![0.1, 0.2, 0.3]);
    }
    
    #[test]
    fn test_operator_mapping() {
        let mut parser = OnnxParser::new();
        
        // Create a simple model
        parser.model = Some(ModelProto {
            ir_version: 7,
            producer_name: "test".into(),
            producer_version: "1.0".into(),
            model_version: 1,
            graph: Some(GraphProto {
                node: vec![
                    NodeProto {
                        input: vec!["input".into()],
                        output: vec!["conv1_out".into()],
                        name: "conv1".into(),
                        op_type: "Conv".into(),
                        attribute: vec![],
                    },
                    NodeProto {
                        input: vec!["conv1_out".into()],
                        output: vec!["relu1_out".into()],
                        name: "relu1".into(),
                        op_type: "Relu".into(),
                        attribute: vec![],
                    },
                ],
                name: "test_graph".into(),
                initializer: vec![],
                input: vec![],
                output: vec![],
                value_info: vec![],
            }),
            opset_import: vec![],
        });
        
        let mapping = parser.map_onnx_operators_to_srgan();
        assert_eq!(mapping.get("conv1"), Some(&"conv_0".to_string()));
        assert_eq!(mapping.get("relu1"), Some(&"relu_1".to_string()));
    }
    
    #[test]
    fn test_data_type_conversion() {
        let parser = OnnxParser::new();
        
        // Test float64 to float32 conversion
        let double_data = vec![1.0f64, 2.0, 3.0];
        let mut raw_bytes = Vec::new();
        for v in &double_data {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        
        let result = parser.parse_raw_data(&raw_bytes, super::super::common::DataType::Float64);
        assert!(result.is_ok());
        
        let parsed = result.unwrap();
        assert_eq!(parsed.len(), 3);
        assert!((parsed[0] - 1.0).abs() < 1e-6);
        assert!((parsed[1] - 2.0).abs() < 1e-6);
        assert!((parsed[2] - 3.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod keras_tests {
    use super::super::keras::{KerasParser, LayerInfo};
    use super::super::common::*;
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_keras_parser_initialization() {
        let parser = KerasParser::new();
        assert!(parser.layers.is_empty());
        assert!(parser.weights.is_empty());
    }
    
    #[test]
    fn test_layer_info_structure() {
        let mut config = HashMap::new();
        config.insert("filters".into(), "64".into());
        config.insert("kernel_size".into(), "(3, 3)".into());
        config.insert("activation".into(), "relu".into());
        
        let layer = LayerInfo {
            name: "conv2d_1".into(),
            class_name: "Conv2D".into(),
            weights: vec!["kernel".into(), "bias".into()],
            config,
        };
        
        assert_eq!(layer.name, "conv2d_1");
        assert_eq!(layer.class_name, "Conv2D");
        assert_eq!(layer.weights.len(), 2);
        assert_eq!(layer.config.get("filters"), Some(&"64".to_string()));
    }
    
    
    #[test]
    fn test_model_config_parsing() {
        let mut parser = KerasParser::new();
        
        let config_json = r#"{
            "config": {
                "name": "srgan_generator",
                "layers": [
                    {
                        "class_name": "InputLayer",
                        "config": {
                            "name": "input_1",
                            "batch_input_shape": [null, 256, 256, 3]
                        }
                    },
                    {
                        "class_name": "Conv2D",
                        "config": {
                            "name": "conv2d_1",
                            "filters": 64,
                            "kernel_size": [3, 3],
                            "activation": "relu"
                        }
                    }
                ]
            }
        }"#;
        
        let result = parser.parse_model_config(config_json);
        assert!(result.is_ok());
        
        assert_eq!(parser.layers.len(), 2);
        assert!(parser.layers.contains_key("input_1"));
        assert!(parser.layers.contains_key("conv2d_1"));
    }
}

#[cfg(test)]
mod common_tests {
    use super::super::common::*;
    use super::*;
    
    #[test]
    fn test_nhwc_to_nchw_conversion() {
        // Create a simple 2x2x2x2 tensor
        let nhwc_data = vec![
            0.0, 1.0,  // pixel (0,0), channels 0,1
            2.0, 3.0,  // pixel (0,1), channels 0,1
            4.0, 5.0,  // pixel (1,0), channels 0,1
            6.0, 7.0,  // pixel (1,1), channels 0,1
        ];
        let shape = vec![1, 2, 2, 2];  // [N, H, W, C]
        
        let result = convert_nhwc_to_nchw(&nhwc_data, &shape);
        assert!(result.is_ok());
        
        let nchw_data = result.unwrap();
        // Expected: channel 0 then channel 1
        let expected = vec![
            0.0, 2.0,  // channel 0, row 0
            4.0, 6.0,  // channel 0, row 1
            1.0, 3.0,  // channel 1, row 0
            5.0, 7.0,  // channel 1, row 1
        ];
        assert_eq!(nchw_data, expected);
    }
    
    #[test]
    fn test_nchw_to_nhwc_conversion() {
        let nchw_data = vec![
            0.0, 2.0,  // channel 0, row 0
            4.0, 6.0,  // channel 0, row 1
            1.0, 3.0,  // channel 1, row 0
            5.0, 7.0,  // channel 1, row 1
        ];
        let shape = vec![1, 2, 2, 2];  // [N, C, H, W]
        
        let result = convert_nchw_to_nhwc(&nchw_data, &shape);
        assert!(result.is_ok());
        
        let nhwc_data = result.unwrap();
        let expected = vec![
            0.0, 1.0,  // pixel (0,0), channels 0,1
            2.0, 3.0,  // pixel (0,1), channels 0,1
            4.0, 5.0,  // pixel (1,0), channels 0,1
            6.0, 7.0,  // pixel (1,1), channels 0,1
        ];
        assert_eq!(nhwc_data, expected);
    }
    
    #[test]
    fn test_tensor_validation() {
        let valid_data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(validate_tensor_data(&valid_data));
        
        let invalid_data = vec![1.0, f32::NAN, 3.0, 4.0];
        assert!(!validate_tensor_data(&invalid_data));
        
        let inf_data = vec![1.0, 2.0, f32::INFINITY, 4.0];
        assert!(!validate_tensor_data(&inf_data));
    }
    
    #[test]
    fn test_tensor_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = tensor_statistics(&data);
        
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.total_elements, 5);
        
        // Check std dev is approximately 1.414
        assert!((stats.std_dev - 1.414).abs() < 0.01);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_model_format_detection() {
        use super::super::super::ModelConverter;
        
        // Test PyTorch format
        let pth_path = std::path::Path::new("model.pth");
        let result = ModelConverter::auto_detect_format(pth_path);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), super::super::super::ModelFormat::PyTorch));
        
        // Test TensorFlow format
        let pb_path = std::path::Path::new("model.pb");
        let result = ModelConverter::auto_detect_format(pb_path);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), super::super::super::ModelFormat::TensorFlow));
        
        // Test ONNX format
        let onnx_path = std::path::Path::new("model.onnx");
        let result = ModelConverter::auto_detect_format(onnx_path);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), super::super::super::ModelFormat::ONNX));
        
        // Test Keras format
        let h5_path = std::path::Path::new("model.h5");
        let result = ModelConverter::auto_detect_format(h5_path);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), super::super::super::ModelFormat::Keras));
        
        // Test unknown format
        let unknown_path = std::path::Path::new("model.unknown");
        let result = ModelConverter::auto_detect_format(unknown_path);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_weight_extractor_trait() {
        // Test that all parsers implement WeightExtractor
        let tf_parser = super::super::tensorflow::TensorFlowParser::new();
        let _ = tf_parser.extract_weights();
        let _ = tf_parser.get_layer_names();
        let _ = tf_parser.get_model_info();
        
        let onnx_parser = super::super::onnx::OnnxParser::new();
        let _ = onnx_parser.extract_weights();
        let _ = onnx_parser.get_layer_names();
        let _ = onnx_parser.get_model_info();
        
        let keras_parser = super::super::keras::KerasParser::new();
        let _ = keras_parser.extract_weights();
        let _ = keras_parser.get_layer_names();
        let _ = keras_parser.get_model_info();
    }
}