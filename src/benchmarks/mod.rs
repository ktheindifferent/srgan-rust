pub mod parallel_bench;

pub use parallel_bench::{
    BenchmarkResult,
    run_benchmark_suite,
    print_benchmark_table,
    benchmark_sequential,
    benchmark_parallel,
};