use crate::error::{Result, SrganError};
use clap::ArgMatches;

pub fn completions(app_m: &ArgMatches) -> Result<()> {
	let shell = app_m
		.value_of("shell")
		.ok_or_else(|| SrganError::InvalidParameter("No shell specified".to_string()))?;

	let script = match shell {
		"bash" => BASH_COMPLETION,
		"zsh" => ZSH_COMPLETION,
		"fish" => FISH_COMPLETION,
		"powershell" => POWERSHELL_COMPLETION,
		other => {
			return Err(SrganError::InvalidParameter(format!(
				"Unknown shell '{}'. Supported: bash, zsh, fish, powershell",
				other
			)))
		}
	};

	print!("{}", script);
	Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Static completion scripts
// ─────────────────────────────────────────────────────────────────────────────

static BASH_COMPLETION: &str = r#"# srgan-rust bash completion
# Source this file or add to ~/.bash_completion:
#   source <(srgan-rust completions bash)

_srgan_rust() {
    local cur prev words cword
    _init_completion || return

    local subcommands="upscale upscale-gpu train train_prescaled batch downscale
        quantise psnr set_width benchmark parallel-benchmark generate-config
        profile-memory analyze-memory server models download-model list-gpus
        compare completions"

    local models="natural anime bilinear"
    local formats="png jpeg webp"
    local shells="bash zsh fish powershell"

    case "$prev" in
        srgan-rust)
            COMPREPLY=( $(compgen -W "$subcommands" -- "$cur") )
            return ;;
        -p|--parameters)
            COMPREPLY=( $(compgen -W "$models" -- "$cur") )
            return ;;
        --format)
            COMPREPLY=( $(compgen -W "$formats" -- "$cur") )
            return ;;
        completions)
            COMPREPLY=( $(compgen -W "$shells" -- "$cur") )
            return ;;
        models)
            COMPREPLY=( $(compgen -W "list download" -- "$cur") )
            return ;;
    esac

    COMPREPLY=( $(compgen -f -- "$cur") )
}

complete -F _srgan_rust srgan-rust
"#;

static ZSH_COMPLETION: &str = r#"#compdef srgan-rust
# srgan-rust zsh completion
# Place in a directory listed in $fpath, e.g.:
#   mkdir -p ~/.zfunc && srgan-rust completions zsh > ~/.zfunc/_srgan-rust
#   # Add to ~/.zshrc: fpath=(~/.zfunc $fpath)

_srgan-rust() {
    local -a subcommands
    subcommands=(
        'upscale:Upscale an image (CPU)'
        'upscale-gpu:Upscale an image using GPU acceleration'
        'train:Train a new model'
        'train_prescaled:Train using pre-scaled images'
        'batch:Batch process a directory of images'
        'downscale:Downscale images'
        'quantise:Quantise model weights'
        'psnr:Calculate PSNR between two images'
        'set_width:Set model channel width'
        'benchmark:Run performance benchmarks'
        'parallel-benchmark:Benchmark parallel processing'
        'generate-config:Generate a training config file'
        'profile-memory:Profile memory usage'
        'analyze-memory:Analyze memory of a command'
        'server:Start the web API server'
        'models:Manage pre-trained models'
        'download-model:Download a model to disk'
        'list-gpus:List available GPU devices'
        'compare:Compare two images (PSNR/SSIM)'
        'completions:Generate shell completion scripts'
    )

    _arguments \
        '(-h --help)'{-h,--help}'[Show help]' \
        '(-V --version)'{-V,--version}'[Show version]' \
        '1: :->subcommand' \
        '*:: :->args'

    case $state in
        subcommand)
            _describe 'command' subcommands ;;
        args)
            case $words[1] in
                completions)
                    _values 'shell' bash zsh fish powershell ;;
                models)
                    _values 'subcommand' list download ;;
                upscale|upscale-gpu|batch|train)
                    _arguments \
                        '(-p --parameters)'{-p,--parameters}'[Model preset]:preset:(natural anime bilinear)' \
                        '--format[Output format]:format:(png jpeg webp)' \
                        '*:file:_files' ;;
                compare)
                    _arguments \
                        '--json[Output JSON]' \
                        '--no-ssim[Skip SSIM]' \
                        '(-o --output)'{-o,--output}'[Report file]:file:_files' \
                        '1:reference image:_files' \
                        '2:test image:_files' ;;
                *)
                    _files ;;
            esac ;;
    esac
}

_srgan-rust "$@"
"#;

static FISH_COMPLETION: &str = r#"# srgan-rust fish completion
# Place in ~/.config/fish/completions/srgan-rust.fish

set -l cmds upscale upscale-gpu train train_prescaled batch downscale quantise \
    psnr set_width benchmark parallel-benchmark generate-config profile-memory \
    analyze-memory server models download-model list-gpus compare completions

complete -c srgan-rust -f -n "not __fish_seen_subcommand_from $cmds" \
    -a "$cmds"

# completions subcommand
complete -c srgan-rust -f -n "__fish_seen_subcommand_from completions" \
    -a "bash zsh fish powershell"

# models subcommand
complete -c srgan-rust -f -n "__fish_seen_subcommand_from models" \
    -a "list download"

# shared --parameters / -p flag
for cmd in upscale upscale-gpu batch train
    complete -c srgan-rust -n "__fish_seen_subcommand_from $cmd" \
        -l parameters -s p -a "natural anime bilinear" -d "Model preset"
end

# --format flag
for cmd in upscale upscale-gpu batch
    complete -c srgan-rust -n "__fish_seen_subcommand_from $cmd" \
        -l format -a "png jpeg webp" -d "Output format"
end

# compare flags
complete -c srgan-rust -n "__fish_seen_subcommand_from compare" \
    -l json -d "Output JSON"
complete -c srgan-rust -n "__fish_seen_subcommand_from compare" \
    -l no-ssim -d "Skip SSIM computation"
complete -c srgan-rust -n "__fish_seen_subcommand_from compare" \
    -l output -s o -r -d "Write JSON report to file"
"#;

static POWERSHELL_COMPLETION: &str = r#"# srgan-rust PowerShell completion
# Add to your $PROFILE:
#   srgan-rust completions powershell | Out-String | Invoke-Expression

Register-ArgumentCompleter -Native -CommandName srgan-rust -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $subcommands = @(
        'upscale','upscale-gpu','train','train_prescaled','batch','downscale',
        'quantise','psnr','set_width','benchmark','parallel-benchmark',
        'generate-config','profile-memory','analyze-memory','server','models',
        'download-model','list-gpus','compare','completions'
    )
    $tokens = $commandAst.CommandElements
    if ($tokens.Count -eq 2) {
        $subcommands | Where-Object { $_ -like "$wordToComplete*" } |
            ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
    } elseif ($tokens[1] -eq 'completions') {
        @('bash','zsh','fish','powershell') | Where-Object { $_ -like "$wordToComplete*" } |
            ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
    } elseif ($tokens[1] -eq 'models') {
        @('list','download') | Where-Object { $_ -like "$wordToComplete*" } |
            ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
    }
}
"#;
