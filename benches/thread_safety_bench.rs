use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use srgan_rust::thread_safe_network::ThreadSafeNetwork;
use ndarray::ArrayD;
use std::sync::Arc;
use std::thread;
use rayon::prelude::*;

fn bench_single_thread_inference(c: &mut Criterion) {
    let network = ThreadSafeNetwork::load_builtin_natural()
        .expect("Failed to load built-in natural network for benchmark");
    
    let mut group = c.benchmark_group("single_thread");
    for size in [16, 32, 64].iter() {
        let input = ArrayD::<f32>::zeros(vec![1, *size, *size, 3]);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &input,
            |b, input| {
                b.iter(|| {
                    network.process(black_box(input.clone()))
                });
            },
        );
    }
    group.finish();
}

fn bench_multi_thread_inference(c: &mut Criterion) {
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural()
        .expect("Failed to load built-in natural network for multi-thread benchmark"));
    
    let mut group = c.benchmark_group("multi_thread");
    for num_threads in [2, 4, 8].iter() {
        let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
        group.throughput(Throughput::Elements(*num_threads as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &(*num_threads, &input),
            |b, (num_threads, input)| {
                b.iter(|| {
                    let mut handles = vec![];
                    for _ in 0..*num_threads {
                        let network_clone = Arc::clone(&network);
                        let input_clone = (*input).clone();
                        let handle = thread::spawn(move || {
                            network_clone.process(input_clone)
                        });
                        handles.push(handle);
                    }
                    for handle in handles {
                        handle.join()
                            .expect("Thread panicked during multi-thread benchmark")
                            .expect("Network processing failed during multi-thread benchmark");
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_rayon_parallel(c: &mut Criterion) {
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural()
        .expect("Failed to load built-in natural network for rayon benchmark"));
    
    let mut group = c.benchmark_group("rayon_parallel");
    for batch_size in [8, 16, 32].iter() {
        let inputs: Vec<_> = (0..*batch_size)
            .map(|_| ArrayD::<f32>::zeros(vec![1, 32, 32, 3]))
            .collect();
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &inputs,
            |b, inputs| {
                b.iter(|| {
                    inputs.par_iter().for_each(|input| {
                        network.process(black_box(input.clone()))
                            .expect("Network processing failed in rayon benchmark");
                    });
                });
            },
        );
    }
    group.finish();
}

fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");
    
    // Benchmark network creation
    group.bench_function("network_creation", |b| {
        b.iter(|| {
            ThreadSafeNetwork::load_builtin_natural()
                .expect("Failed to load network in creation benchmark")
        });
    });
    
    // Benchmark first inference (includes buffer creation)
    let network = ThreadSafeNetwork::load_builtin_natural()
        .expect("Failed to load network for memory benchmark");
    let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
    group.bench_function("first_inference", |b| {
        b.iter(|| {
            // Create new network each time to measure first inference overhead
            let network = ThreadSafeNetwork::load_builtin_natural()
                .expect("Failed to load network in first inference benchmark");
            network.process(black_box(input.clone()))
                .expect("Network processing failed in first inference benchmark")
        });
    });
    
    // Benchmark subsequent inference (reuses buffer)
    network.process(input.clone())
        .expect("Failed to warm up network for subsequent inference benchmark"); // Warm up
    group.bench_function("subsequent_inference", |b| {
        b.iter(|| {
            network.process(black_box(input.clone()))
                .expect("Network processing failed in subsequent inference benchmark")
        });
    });
    
    group.finish();
}

fn bench_thread_contention(c: &mut Criterion) {
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural()
        .expect("Failed to load network for contention benchmark"));
    
    let mut group = c.benchmark_group("thread_contention");
    
    // Low contention - different input sizes
    group.bench_function("low_contention", |b| {
        b.iter(|| {
            let mut handles = vec![];
            for i in 0..4 {
                let network_clone = Arc::clone(&network);
                let size = 16 + i * 8;  // Different sizes
                let handle = thread::spawn(move || {
                    let input = ArrayD::<f32>::zeros(vec![1, size, size, 3]);
                    network_clone.process(black_box(input))
                        .expect("Network processing failed in contention benchmark")
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join()
                    .expect("Thread panicked in contention benchmark");
            }
        });
    });
    
    // High contention - same input size
    group.bench_function("high_contention", |b| {
        b.iter(|| {
            let mut handles = vec![];
            for _ in 0..4 {
                let network_clone = Arc::clone(&network);
                let handle = thread::spawn(move || {
                    let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
                    network_clone.process(black_box(input))
                        .expect("Network processing failed in contention benchmark")
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join()
                    .expect("Thread panicked in contention benchmark");
            }
        });
    });
    
    group.finish();
}

fn bench_scaling_efficiency(c: &mut Criterion) {
    let network = Arc::new(ThreadSafeNetwork::load_builtin_natural()
        .expect("Failed to load network for scaling benchmark"));
    let input = ArrayD::<f32>::zeros(vec![1, 32, 32, 3]);
    
    let mut group = c.benchmark_group("scaling_efficiency");
    
    // Measure how performance scales with thread count
    for num_threads in [1, 2, 4, 8, 16].iter() {
        group.throughput(Throughput::Elements(*num_threads as u64));
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    if num_threads == 1 {
                        // Single thread baseline
                        network.process(black_box(input.clone()))
                            .expect("Network processing failed in single-thread scaling benchmark");
                    } else {
                        // Multi-threaded
                        let mut handles = vec![];
                        for _ in 0..num_threads {
                            let network_clone = Arc::clone(&network);
                            let input_clone = input.clone();
                            let handle = thread::spawn(move || {
                                network_clone.process(input_clone)
                                    .expect("Network processing failed in multi-thread scaling benchmark")
                            });
                            handles.push(handle);
                        }
                        for handle in handles {
                            handle.join()
                                .expect("Thread panicked in scaling benchmark");
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread_inference,
    bench_multi_thread_inference,
    bench_rayon_parallel,
    bench_memory_overhead,
    bench_thread_contention,
    bench_scaling_efficiency
);
criterion_main!(benches);