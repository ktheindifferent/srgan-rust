use crate::error::{Result, SrganError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Options captured at the start of a batch run so a resumed run can report
/// what settings were originally used.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchOptions {
    pub parameters: Option<String>,
    pub custom_model: Option<String>,
    pub factor: usize,
    pub recursive: bool,
    pub parallel: bool,
    pub skip_existing: bool,
}

/// Persistent checkpoint for a batch upscale run.
///
/// Saved as `.srgan_checkpoint_<batch_id>.json` in the output directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCheckpoint {
    pub batch_id: String,
    pub input_dir: String,
    pub output_dir: String,
    pub total_images: usize,
    /// Output paths that have been successfully written.
    pub completed_images: Vec<String>,
    /// Input paths that failed processing.
    pub failed_images: Vec<String>,
    /// Unix timestamp (seconds) when the batch was first started.
    pub started_at: u64,
    /// Unix timestamp (seconds) of the most recent checkpoint save.
    pub last_updated: u64,
    pub options: BatchOptions,
}

impl BatchCheckpoint {
    pub fn new(
        batch_id: String,
        input_dir: String,
        output_dir: String,
        total_images: usize,
        options: BatchOptions,
    ) -> Self {
        let now = unix_now();
        Self {
            batch_id,
            input_dir,
            output_dir,
            total_images,
            completed_images: Vec::new(),
            failed_images: Vec::new(),
            started_at: now,
            last_updated: now,
            options,
        }
    }
}

/// Returns the path of the checkpoint file for the given batch ID.
pub fn checkpoint_path(batch_id: &str, checkpoint_dir: &Path) -> PathBuf {
    checkpoint_dir.join(format!(".srgan_checkpoint_{}.json", batch_id))
}

/// Persist `checkpoint` to disk.  Updates `last_updated` before writing.
pub fn save_checkpoint(checkpoint: &mut BatchCheckpoint, checkpoint_dir: &Path) -> Result<()> {
    checkpoint.last_updated = unix_now();
    let path = checkpoint_path(&checkpoint.batch_id, checkpoint_dir);
    let json = serde_json::to_string_pretty(checkpoint)
        .map_err(|e| SrganError::Serialization(e.to_string()))?;
    fs::write(&path, json).map_err(SrganError::Io)?;
    Ok(())
}

/// Load the checkpoint for `batch_id` from `checkpoint_dir`.
/// Returns `Ok(None)` when no checkpoint file exists yet.
pub fn load_checkpoint(batch_id: &str, checkpoint_dir: &Path) -> Result<Option<BatchCheckpoint>> {
    let path = checkpoint_path(batch_id, checkpoint_dir);
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path).map_err(SrganError::Io)?;
    let checkpoint = serde_json::from_str::<BatchCheckpoint>(&content)
        .map_err(|e| SrganError::Serialization(e.to_string()))?;
    Ok(Some(checkpoint))
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Global checkpoint store: ~/.srgan-rust/checkpoints/<id>.json ─────────────

/// Returns `~/.srgan-rust/checkpoints/`.
pub fn global_checkpoint_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".srgan-rust").join("checkpoints")
}

/// Returns the path for a globally-stored checkpoint file.
pub fn global_checkpoint_path(batch_id: &str) -> PathBuf {
    global_checkpoint_dir().join(format!("{}.json", batch_id))
}

/// Persist `checkpoint` to `~/.srgan-rust/checkpoints/<id>.json`.
pub fn save_global_checkpoint(checkpoint: &mut BatchCheckpoint) -> Result<()> {
    checkpoint.last_updated = unix_now();
    let dir = global_checkpoint_dir();
    fs::create_dir_all(&dir).map_err(SrganError::Io)?;
    let path = global_checkpoint_path(&checkpoint.batch_id);
    let json = serde_json::to_string_pretty(checkpoint)
        .map_err(|e| SrganError::Serialization(e.to_string()))?;
    fs::write(&path, json).map_err(SrganError::Io)?;
    Ok(())
}

/// Load a globally-stored checkpoint by batch ID.
/// Returns `Ok(None)` when the checkpoint file does not exist.
pub fn load_global_checkpoint(batch_id: &str) -> Result<Option<BatchCheckpoint>> {
    let path = global_checkpoint_path(batch_id);
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path).map_err(SrganError::Io)?;
    let cp = serde_json::from_str::<BatchCheckpoint>(&content)
        .map_err(|e| SrganError::Serialization(e.to_string()))?;
    Ok(Some(cp))
}

/// List all globally-stored checkpoints sorted by `started_at` (oldest first).
pub fn list_global_checkpoints() -> Result<Vec<BatchCheckpoint>> {
    let dir = global_checkpoint_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<BatchCheckpoint> = Vec::new();
    for entry in fs::read_dir(&dir).map_err(SrganError::Io)? {
        let entry = entry.map_err(SrganError::Io)?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(cp) = serde_json::from_str::<BatchCheckpoint>(&content) {
                    entries.push(cp);
                }
            }
        }
    }
    entries.sort_by_key(|c| c.started_at);
    Ok(entries)
}
