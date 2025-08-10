use std::process::{Command, Stdio};
use crate::error::{SrganError, Result};

pub struct CommandBuilder {
    command: Command,
}

impl CommandBuilder {
    pub fn new(program: &str) -> Self {
        Self {
            command: Command::new(program),
        }
    }
    
    pub fn arg<S: AsRef<std::ffi::OsStr>>(mut self, arg: S) -> Self {
        self.command.arg(arg);
        self
    }
    
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        self.command.args(args);
        self
    }
    
    pub fn stdout(mut self, cfg: Stdio) -> Self {
        self.command.stdout(cfg);
        self
    }
    
    pub fn stderr(mut self, cfg: Stdio) -> Self {
        self.command.stderr(cfg);
        self
    }
    
    pub fn execute(mut self) -> Result<()> {
        let status = self.command.status()
            .map_err(|e| SrganError::InvalidInput(
                format!("Failed to execute command: {}", e)
            ))?;
            
        if !status.success() {
            return Err(SrganError::InvalidInput(
                format!("Command failed with exit code: {:?}", status.code())
            ));
        }
        
        Ok(())
    }
    
    pub fn output(mut self) -> Result<std::process::Output> {
        self.command.output()
            .map_err(|e| SrganError::InvalidInput(
                format!("Failed to execute command: {}", e)
            ))
    }
}

pub fn ffmpeg() -> CommandBuilder {
    CommandBuilder::new("ffmpeg")
}

pub fn ffprobe() -> CommandBuilder {
    CommandBuilder::new("ffprobe")
}

#[macro_export]
macro_rules! log_config {
    ($($field:expr => $value:expr),* $(,)?) => {
        $(
            log::info!("  {}: {:?}", $field, $value);
        )*
    };
}