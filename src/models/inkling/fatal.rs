//! A panic on ANY thread ends the process, loudly and with a nonzero status.
//!
//! # The failure this exists for
//!
//! The dense lane no longer materialises a whole `[heads, n, n]`, so the
//! particular allocation below is not one any run makes now. That does not
//! retire this module: the refusal is a property of the ALLOCATOR, it lands on
//! whatever the largest buffer happens to be, and what made it dangerous was
//! never the size -- it was that a refusal on a worker thread does not end the
//! process.
//!
//! As it stood: above roughly 15,600 tokens the device refused one attention
//! layer's `[heads, n, n]` f32 score matrix. The refusal is not a shortage of
//! memory:
//! cubecl's CUDA runtime sets its largest single allocation to
//! `cuDeviceTotalMem / 4`, which is 29.9 GiB on a 119.6 GiB node, and
//! `[32, 16384, 16384]` f32 is 32 GiB exactly. Every pool's `max_alloc_size` is
//! below the request, so `MemoryManagement::reserve` returns
//! `can't allocate buffer of size: 34359738368`.
//!
//! That `Result` is unwrapped in `initialize_memory`, on a `DSD-*` worker
//! thread inside the CUDA stream, and **a panic on a worker thread does not
//! end the process**. Measured at n = 16,384: the buffer was refused 224 times,
//! the main thread went on reading memory that had never been written, the run
//! **exited 0**, printed a plausible top-5 and a layer-RMS ladder of
//! 36.7 .. 80.3 where the coherent one is 1.5 .. 14.7. Two independent runs
//! agreed on the same garbage, which is what makes it dangerous: it is
//! reproducible, so it looks like an answer.
//!
//! A numerical program has no worse failure mode than a plausible number. So
//! any panic, anywhere, on any thread, takes the process down.
//!
//! # Why `abort` rather than `exit`
//!
//! The hook runs on the panicking thread while the CUDA driver's own threads
//! are still live. `std::process::exit` runs libc's atexit chain through them
//! and can hang; `abort` is immediate. The cost is that the status is SIGABRT
//! rather than a chosen code, which is nonzero either way and is what a shell
//! `$?` and a `timeout`-wrapped harness both need.

use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The sequence length the process is working at.
///
/// A global rather than an argument because the panic that matters happens on a
/// thread this crate never created, three crates down, with no path back to the
/// forward's locals. It is written once per run and read once per crash.
static TOKENS: AtomicUsize = AtomicUsize::new(0);

/// Record the sequence length, so a crash can name what caused it.
pub fn note_tokens(n: usize) {
    TOKENS.store(n, Ordering::Relaxed);
}

/// Install the hook. Call once, from `main`, before anything touches a device.
pub fn arm() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // The default hook first: it names the thread, the file and the line,
        // and the backtrace if one was asked for.
        previous(info);

        let text = payload(info);
        let mut err = std::io::stderr().lock();
        let _ = writeln!(err);
        let _ = writeln!(err, "=== FATAL: this run cannot produce an answer ===");
        if let Some(bytes) = refused_bytes(&text) {
            let n = TOKENS.load(Ordering::Relaxed);
            let _ = writeln!(
                err,
                "  The device REFUSED a {bytes}-byte ({:.2} GiB) allocation.",
                bytes as f64 / (1u64 << 30) as f64
            );
            if n > 0 {
                let _ = writeln!(
                    err,
                    "  Sequence length: {n} tokens. One attention layer's [heads, n, n] f32 score\n  \
                     matrix is that buffer, and it grows as n^2: every 41% more tokens doubles it."
                );
            }
            let _ = writeln!(
                err,
                "  This is a CEILING, not a shortage: cubecl's CUDA runtime caps a single\n  \
                 allocation at cuDeviceTotalMem / 4, and no amount of free memory raises it.\n  \
                 Shorten the input, or run a lane that never materialises [heads, n, n]."
            );
        }
        let _ = writeln!(
            err,
            "  Aborting. The alternative is exiting 0 with numbers read out of a buffer that\n  \
             was never written, which is what this build did before."
        );
        let _ = err.flush();
        let _ = std::io::stdout().flush();
        std::process::abort();
    }));
}

/// The panic payload as a string, for the two shapes `panic!` produces.
fn payload(info: &std::panic::PanicHookInfo<'_>) -> String {
    if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = info.payload().downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        String::new()
    }
}

/// The byte count out of cubecl's allocation-failure message, if that is what
/// this panic is.
///
/// Matched on the text because that is the only thing that crosses the thread
/// boundary: the `Err` value is unwrapped inside cubecl and its type never
/// reaches us. Brittle by construction, so a miss costs the tailored paragraph
/// and nothing else -- the abort happens either way.
fn refused_bytes(text: &str) -> Option<u64> {
    const MARK: &str = "can't allocate buffer of size: ";
    let rest = text.split(MARK).nth(1)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::refused_bytes;

    #[test]
    fn reads_the_refused_size() {
        let msg = "called `Result::unwrap()` on an `Err` value: can't allocate buffer of size: \
                   34359738368\nstack backtrace: ...";
        assert_eq!(refused_bytes(msg), Some(34_359_738_368));
    }

    #[test]
    fn ignores_other_panics() {
        assert_eq!(refused_bytes("index out of bounds: the len is 3"), None);
    }
}
