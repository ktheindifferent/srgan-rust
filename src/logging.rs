use chrono::Local;
use env_logger::{Builder, Target};
use indicatif::{ProgressBar, ProgressStyle};
use log::{LevelFilter, info};
use std::io::Write;
use std::time::Duration;

/// Initialize the logging system with the specified verbosity level
pub fn init_logger(verbosity: u64) {
    let level = match verbosity {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };

    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {} {}:{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .filter(None, level)
        .target(Target::Stderr)
        .init();
}

/// Initialize a simple logger without timestamps for CLI output
pub fn init_simple_logger() {
    Builder::new()
        .format(|buf, record| {
            let level_str = match record.level() {
                log::Level::Error => "ERROR",
                log::Level::Warn => "WARN",
                log::Level::Info => "",
                log::Level::Debug => "DEBUG",
                log::Level::Trace => "TRACE",
            };
            
            if record.level() == log::Level::Info {
                writeln!(buf, "{}", record.args())
            } else {
                writeln!(buf, "[{}] {}", level_str, record.args())
            }
        })
        .filter(None, LevelFilter::Info)
        .target(Target::Stderr)
        .init();
}

/// Create a progress bar for file processing
pub fn create_file_progress_bar(total_files: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total_files);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Create a progress bar for training epochs
pub fn create_training_progress_bar(total_steps: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_steps);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("Training [{elapsed_precise}] [{bar:40.green/white}] Step {pos}/{len} | Loss: {msg}")
            .expect("Failed to set progress bar template")
            .progress_chars("=>-"),
    );
    pb
}

/// Create a spinner for indefinite operations
pub fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Failed to set spinner template")
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Log training metrics
pub fn log_training_metrics(step: usize, loss: f32, learning_rate: f32, duration: Duration) {
    info!(
        "Step: {} | Loss: {:.6} | LR: {:.2e} | Time: {:.2}s",
        step,
        loss,
        learning_rate,
        duration.as_secs_f32()
    );
}

/// Log validation metrics
pub fn log_validation_metrics(epoch: usize, psnr: f32, loss: f32) {
    info!(
        "Validation - Epoch: {} | PSNR: {:.2} dB | Loss: {:.6}",
        epoch, psnr, loss
    );
}

/// Create a multi-progress bar for parallel operations
pub struct MultiProgress {
    bars: Vec<ProgressBar>,
}

impl MultiProgress {
    pub fn new() -> Self {
        Self { bars: Vec::new() }
    }

    pub fn add_bar(&mut self, total: u64, message: &str) -> usize {
        let pb = create_file_progress_bar(total, message);
        self.bars.push(pb);
        self.bars.len() - 1
    }

    pub fn inc(&mut self, index: usize) {
        if let Some(bar) = self.bars.get(index) {
            bar.inc(1);
        }
    }

    pub fn finish(&mut self, index: usize) {
        if let Some(bar) = self.bars.get(index) {
            bar.finish_with_message("Complete");
        }
    }

    pub fn finish_all(&mut self) {
        for bar in &self.bars {
            bar.finish();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_progress_bar() {
        let pb = create_file_progress_bar(100, "Processing");
        assert_eq!(pb.length(), Some(100));
        pb.finish();
    }

    #[test]
    fn test_create_training_progress_bar() {
        let pb = create_training_progress_bar(1000);
        assert_eq!(pb.length(), Some(1000));
        pb.finish();
    }

    #[test]
    fn test_multi_progress() {
        let mut mp = MultiProgress::new();
        let idx = mp.add_bar(10, "Test");
        mp.inc(idx);
        mp.finish(idx);
        mp.finish_all();
    }
}