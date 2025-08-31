use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use srgan_rust::model_converter::ModelConverter;
use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;
use serde_pickle::{SerOptions, Value, HashableValue};
use std::collections::HashMap;

/// Create a PyTorch model of specified size for benchmarking
fn create_benchmark_model(num_layers: usize, params_per_layer: usize) -> Value {
    let mut state_dict = HashMap::new();
    
    for i in 0..num_layers {
        let layer_name = format!("layer_{}.weight", i);
        let weights = vec![0.1_f64; params_per_layer];
        state_dict.insert(
            HashableValue::String(layer_name),
            Value::List(weights.into_iter().map(Value::F64).collect())
        );
        
        let bias_name = format!("layer_{}.bias", i);
        let bias = vec![0.01_f64; params_per_layer / 100];
        state_dict.insert(
            HashableValue::String(bias_name),
            Value::List(bias.into_iter().map(Value::F64).collect())
        );
    }
    
    Value::Dict(state_dict)
}

/// Create a model file for benchmarking
fn create_benchmark_file(num_layers: usize, params_per_layer: usize) -> NamedTempFile {
    let state_dict = create_benchmark_model(num_layers, params_per_layer);
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    
    let ser_options = SerOptions::new();
    let bytes = serde_pickle::to_vec(&state_dict, ser_options)
        .expect("Failed to serialize");
    
    file.write_all(&bytes).expect("Failed to write");
    file.flush().expect("Failed to flush");
    
    file
}

/// Create a ZIP-based model file for benchmarking
fn create_zip_benchmark_file(num_layers: usize, params_per_layer: usize) -> NamedTempFile {
    use zip::write::{ZipWriter, FileOptions};
    use std::io::Cursor;
    
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    
    let state_dict = create_benchmark_model(num_layers, params_per_layer);
    
    let mut buffer = Vec::new();
    {
        let mut zip = ZipWriter::new(Cursor::new(&mut buffer));
        
        let ser_options = SerOptions::new();
        let pkl_bytes = serde_pickle::to_vec(&state_dict, ser_options)
            .expect("Failed to serialize");
        
        zip.start_file("data.pkl", FileOptions::default())
            .expect("Failed to start file");
        zip.write_all(&pkl_bytes).expect("Failed to write");
        
        zip.finish().expect("Failed to finish ZIP");
    }
    
    file.write_all(&buffer).expect("Failed to write");
    file.flush().expect("Failed to flush");
    
    file
}

/// Benchmark parsing small models
fn bench_small_model(c: &mut Criterion) {
    let file = create_benchmark_file(10, 1000);
    
    c.bench_function("pytorch_parse_small", |b| {
        b.iter(|| {
            let mut converter = ModelConverter::new();
            converter.load_pytorch(black_box(file.path()))
        });
    });
}

/// Benchmark parsing medium models
fn bench_medium_model(c: &mut Criterion) {
    let file = create_benchmark_file(50, 10000);
    
    c.bench_function("pytorch_parse_medium", |b| {
        b.iter(|| {
            let mut converter = ModelConverter::new();
            converter.load_pytorch(black_box(file.path()))
        });
    });
}

/// Benchmark parsing large models
fn bench_large_model(c: &mut Criterion) {
    let file = create_benchmark_file(100, 50000);
    
    c.bench_function("pytorch_parse_large", |b| {
        b.iter(|| {
            let mut converter = ModelConverter::new();
            converter.load_pytorch(black_box(file.path()))
        });
    });
}

/// Benchmark parsing ZIP-based models
fn bench_zip_model(c: &mut Criterion) {
    let file = create_zip_benchmark_file(50, 10000);
    
    c.bench_function("pytorch_parse_zip", |b| {
        b.iter(|| {
            let mut converter = ModelConverter::new();
            converter.load_pytorch(black_box(file.path()))
        });
    });
}

/// Benchmark parsing models with different data types
fn bench_dtype_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("pytorch_dtype_parsing");
    
    // Float32 model
    {
        let mut state_dict = HashMap::new();
        let float32_bytes: Vec<u8> = (0..10000)
            .flat_map(|i| {
                let val = (i as f32) * 0.001;
                val.to_le_bytes().to_vec()
            })
            .collect();
        
        state_dict.insert(
            HashableValue::String("conv.weight".into()),
            Value::Bytes(float32_bytes)
        );
        
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let ser_options = SerOptions::new();
        let bytes = serde_pickle::to_vec(&Value::Dict(state_dict), ser_options)
            .expect("Failed to serialize");
        file.write_all(&bytes).expect("Failed to write");
        file.flush().expect("Failed to flush");
        
        group.bench_function("float32", |b| {
            b.iter(|| {
                let mut converter = ModelConverter::new();
                converter.load_pytorch(black_box(file.path()))
            });
        });
    }
    
    // Float16 model
    {
        let mut state_dict = HashMap::new();
        let float16_bytes: Vec<u8> = (0..10000)
            .flat_map(|_| vec![0x3C, 0x00]) // 1.0 in float16
            .collect();
        
        state_dict.insert(
            HashableValue::String("conv.weight".into()),
            Value::Bytes(float16_bytes)
        );
        
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let ser_options = SerOptions::new();
        let bytes = serde_pickle::to_vec(&Value::Dict(state_dict), ser_options)
            .expect("Failed to serialize");
        file.write_all(&bytes).expect("Failed to write");
        file.flush().expect("Failed to flush");
        
        group.bench_function("float16", |b| {
            b.iter(|| {
                let mut converter = ModelConverter::new();
                converter.load_pytorch(black_box(file.path()))
            });
        });
    }
    
    // Int8 quantized model
    {
        let mut state_dict = HashMap::new();
        let int8_bytes: Vec<u8> = (0..10000)
            .map(|i| (i % 256) as u8)
            .collect();
        
        state_dict.insert(
            HashableValue::String("conv.weight".into()),
            Value::Bytes(int8_bytes)
        );
        
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let ser_options = SerOptions::new();
        let bytes = serde_pickle::to_vec(&Value::Dict(state_dict), ser_options)
            .expect("Failed to serialize");
        file.write_all(&bytes).expect("Failed to write");
        file.flush().expect("Failed to flush");
        
        group.bench_function("int8", |b| {
            b.iter(|| {
                let mut converter = ModelConverter::new();
                converter.load_pytorch(black_box(file.path()))
            });
        });
    }
    
    group.finish();
}

/// Benchmark model size scaling
fn bench_model_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("pytorch_model_scaling");
    
    for size in [10, 50, 100, 200] {
        let file = create_benchmark_file(size, 10000);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut converter = ModelConverter::new();
                    converter.load_pytorch(black_box(file.path()))
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark architecture detection
fn bench_architecture_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pytorch_architecture_detection");
    
    // SRGAN model
    {
        let mut state_dict = HashMap::new();
        for i in 0..16 {
            let key = format!("generator.residual_blocks.{}.conv1.weight", i);
            state_dict.insert(
                HashableValue::String(key),
                Value::List(vec![Value::F64(0.1); 64 * 64 * 3 * 3])
            );
        }
        state_dict.insert(
            HashableValue::String("generator.upsample.0.weight".into()),
            Value::List(vec![Value::F64(0.1); 256 * 64 * 3 * 3])
        );
        
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let ser_options = SerOptions::new();
        let bytes = serde_pickle::to_vec(&Value::Dict(state_dict), ser_options)
            .expect("Failed to serialize");
        file.write_all(&bytes).expect("Failed to write");
        file.flush().expect("Failed to flush");
        
        group.bench_function("srgan", |b| {
            b.iter(|| {
                let mut converter = ModelConverter::new();
                converter.load_pytorch(black_box(file.path()))
            });
        });
    }
    
    // ESRGAN model
    {
        let mut state_dict = HashMap::new();
        state_dict.insert(
            HashableValue::String("conv_first.weight".into()),
            Value::List(vec![Value::F64(0.1); 64 * 3 * 3 * 3])
        );
        
        for i in 0..23 {
            let key = format!("RRDB.{}.conv1.weight", i);
            state_dict.insert(
                HashableValue::String(key),
                Value::List(vec![Value::F64(0.1); 32 * 64 * 3 * 3])
            );
        }
        
        state_dict.insert(
            HashableValue::String("trunk_conv.weight".into()),
            Value::List(vec![Value::F64(0.1); 64 * 64 * 3 * 3])
        );
        
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let ser_options = SerOptions::new();
        let bytes = serde_pickle::to_vec(&Value::Dict(state_dict), ser_options)
            .expect("Failed to serialize");
        file.write_all(&bytes).expect("Failed to write");
        file.flush().expect("Failed to flush");
        
        group.bench_function("esrgan", |b| {
            b.iter(|| {
                let mut converter = ModelConverter::new();
                converter.load_pytorch(black_box(file.path()))
            });
        });
    }
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("pytorch_memory_patterns");
    
    // Many small tensors
    {
        let mut state_dict = HashMap::new();
        for i in 0..1000 {
            let key = format!("param_{}", i);
            state_dict.insert(
                HashableValue::String(key),
                Value::List(vec![Value::F64(0.1); 100])
            );
        }
        
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let ser_options = SerOptions::new();
        let bytes = serde_pickle::to_vec(&Value::Dict(state_dict), ser_options)
            .expect("Failed to serialize");
        file.write_all(&bytes).expect("Failed to write");
        file.flush().expect("Failed to flush");
        
        group.bench_function("many_small_tensors", |b| {
            b.iter(|| {
                let mut converter = ModelConverter::new();
                converter.load_pytorch(black_box(file.path()))
            });
        });
    }
    
    // Few large tensors
    {
        let mut state_dict = HashMap::new();
        for i in 0..10 {
            let key = format!("param_{}", i);
            state_dict.insert(
                HashableValue::String(key),
                Value::List(vec![Value::F64(0.1); 100000])
            );
        }
        
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let ser_options = SerOptions::new();
        let bytes = serde_pickle::to_vec(&Value::Dict(state_dict), ser_options)
            .expect("Failed to serialize");
        file.write_all(&bytes).expect("Failed to write");
        file.flush().expect("Failed to flush");
        
        group.bench_function("few_large_tensors", |b| {
            b.iter(|| {
                let mut converter = ModelConverter::new();
                converter.load_pytorch(black_box(file.path()))
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_small_model,
    bench_medium_model,
    bench_large_model,
    bench_zip_model,
    bench_dtype_parsing,
    bench_model_scaling,
    bench_architecture_detection,
    bench_memory_patterns
);

criterion_main!(benches);