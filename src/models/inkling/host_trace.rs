//! Opt-in host-scope timing for Session forward and its KV maintenance.
//!
//! No GPU queries or fences. Durations are inclusive (nested spans must not be
//! summed) and include host blocking in calls that already existed. Only scope
//! exits of at least 100 ms are logged; an indefinitely blocked call cannot emit
//! a completed span. Context is thread-local and scoped to a Session pass.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Context {
    position: usize,
    rows: usize,
    rank: usize,
    layer: Option<usize>,
}

thread_local! {
    static CURRENT: Cell<Option<Context>> = const { Cell::new(None) };
}

pub(crate) struct ContextGuard {
    previous: Option<Context>,
    // Restoring thread-local context on a different thread would be wrong.
    _local: std::marker::PhantomData<std::rc::Rc<()>>,
}

fn enter(context: Context) -> ContextGuard {
    ContextGuard { previous: CURRENT.replace(Some(context)), _local: std::marker::PhantomData }
}

impl Drop for ContextGuard {
    fn drop(&mut self) { CURRENT.set(self.previous); }
}

pub(crate) fn pass(position: usize, rows: usize, rank: usize) -> Option<ContextGuard> {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| enabled(std::env::var("INK_HOST_STALL_TRACE").ok().as_deref()));
    on.then(|| enter(Context { position, rows, rank, layer: None }))
}

pub(crate) fn layer(layer: usize) -> Option<ContextGuard> {
    CURRENT.get().map(|context| enter(Context { layer: Some(layer), ..context }))
}

pub(crate) struct Span {
    context: Context,
    phase: &'static str,
    start: Instant,
}

pub(crate) fn span(phase: &'static str) -> Option<Span> {
    CURRENT.get().map(|context| Span { context, phase, start: Instant::now() })
}

pub(crate) fn call<T>(phase: &'static str, call: impl FnOnce() -> T) -> T {
    let _span = span(phase);
    call()
}

fn enabled(value: Option<&str>) -> bool { value == Some("1") }
fn slow(elapsed: Duration) -> bool { elapsed >= Duration::from_millis(100) }

impl Drop for Span {
    fn drop(&mut self) {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        static SLOW: AtomicU64 = AtomicU64::new(0);
        let elapsed = self.start.elapsed();
        let calls_total = CALLS.fetch_add(1, Relaxed) + 1;
        if !slow(elapsed) { return; }
        let slow_calls_total = SLOW.fetch_add(1, Relaxed) + 1;
        let end_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_millis();
        let context = self.context;
        let layer = context.layer.map(|n| n.to_string()).unwrap_or_else(|| "none".to_owned());
        eprintln!("[session-host-stall] end_unix_ms={end_unix_ms} pid={} rank={} position_start={} position_end={} rows={} layer={layer} phase={} host_seconds={:.6} inclusive=true calls_total={calls_total} slow_calls_total={slow_calls_total} counters=all_phase_spans timing=host_scope_not_gpu unwinding={}",
            std::process::id(), context.rank, context.position,
            context.position.saturating_add(context.rows), context.rows, self.phase,
            elapsed.as_secs_f64(), std::thread::panicking());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_opt_in_and_slow_threshold_are_exact() {
        assert!(enabled(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!enabled(value));
        }
        assert!(!slow(Duration::from_micros(99_999)));
        assert!(slow(Duration::from_millis(100)));
    }

    #[test]
    fn layer_context_restores_position_and_does_not_leak_after_pass() {
        assert!(CURRENT.get().is_none());
        {
            let context = Context { position: 9728, rows: 512, rank: 1, layer: None };
            let _pass = enter(context);
            {
                let _layer = layer(17);
                let timed = span("kv_append").unwrap();
                assert_eq!(timed.context, Context { layer: Some(17), ..context });
            }
            assert_eq!(CURRENT.get(), Some(context));
        }
        assert!(CURRENT.get().is_none());
        assert!(span("outside_session").is_none());
    }
}
