//! Where model files live -- which mary deliberately does not know.
//!
//! mary is a library and a set of probes, not an installation, so it has no
//! business holding an opinion about anyone's disk layout. A baked-in default
//! path is such an opinion, and it is wrong on every machine but the one it was
//! written on. Worse, it fails silently: the probe looks in the guessed place,
//! does not find the file, and reports "not found" as though the model were
//! missing rather than the path being a guess. The user then goes looking for a
//! file that was on disk the whole time.
//!
//! So: pass the path explicitly, or set `MARY_MODELS` to the directory holding
//! them. With neither, these functions fail loudly and say both ways to fix it.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

/// The directory holding model files, from `$MARY_MODELS`. No fallback.
pub fn models_dir() -> Result<PathBuf> {
    match std::env::var_os("MARY_MODELS") {
        Some(d) if !d.is_empty() => Ok(PathBuf::from(d)),
        _ => bail!(
            "MARY_MODELS is not set, and mary has no default model directory.\n\
             Set it to wherever this machine keeps its model piles, e.g.\n\
             \x20   export MARY_MODELS=/path/to/models"
        ),
    }
}

/// Resolve one model file. An explicit `arg` wins outright; otherwise the file
/// is looked up as `$MARY_MODELS/<name>`. Never guesses.
pub fn model(arg: Option<&str>, name: &str) -> Result<PathBuf> {
    if let Some(p) = arg {
        return Ok(PathBuf::from(p));
    }
    let dir = models_dir().map_err(|e| {
        anyhow::anyhow!("{e}\n\nOr pass the path to `{name}` directly on the command line.")
    })?;
    Ok(dir.join(name))
}

/// The test/probe form: `None` instead of an error, so a harness can SKIP on a
/// machine that simply does not have the model rather than fail. Callers are
/// expected to print their own skip line -- see [`skip_reason`].
pub fn model_opt(arg: Option<&str>, name: &str) -> Option<PathBuf> {
    let p = model(arg, name).ok()?;
    p.exists().then_some(p)
}

/// The line to print when [`model_opt`] returns `None`, so every probe words it
/// the same way and always names the fix.
pub fn skip_reason(name: &str) -> String {
    match models_dir() {
        Ok(d) => format!("no {name} under {} (set MARY_MODELS or pass a path)", d.display()),
        Err(_) => format!("MARY_MODELS unset, so {name} cannot be located (set it, or pass a path)"),
    }
}

/// True when `p` is inside `dir` after both are canonicalized -- for probes that
/// accept a user path and want to report where it resolved.
pub fn under(p: &Path, dir: &Path) -> bool {
    match (p.canonicalize(), dir.canonicalize()) {
        (Ok(p), Ok(d)) => p.starts_with(d),
        _ => false,
    }
}
