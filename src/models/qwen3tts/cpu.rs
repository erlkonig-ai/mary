//! Minimal Accelerate (CBLAS) bindings + scalar helpers for the stages that
//! run on the **CPU**. The code predictor's 15 autoregressive steps are ~1 GB
//! of strictly sequential matvecs per frame — on the GPU the per-op submission
//! overhead (~15-25 µs × hundreds of ops per step) dominated the actual math
//! by an order of magnitude; Accelerate's sgemv does the same work directly.
//!
//! Non-Apple targets use plain reference loops instead of Accelerate — same
//! math, unoptimized; the realtime numbers in PORT_NOTES are the Apple path.

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

/// Pin the calling thread to `QOS_CLASS_USER_INTERACTIVE` (0x21). The decode
/// loop is CPU-bound (burn op submission + the Accelerate predictor) and this
/// machine's background daemons (mediaanalysisd bursts, etc.) otherwise steal
/// enough cores to swing the loop 4-10× — background QoS loses to us once we
/// declare interactive.
pub fn set_interactive_qos() {
    #[cfg(target_os = "macos")]
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x21, 0);
    }
}

/// `y = W·x` for row-major `W: [m, n]` (the safetensors layout `[out, in]`).
pub fn sgemv(w: &[f32], m: usize, n: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), m * n);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    #[cfg(target_os = "macos")]
    unsafe {
        // 101 = CblasRowMajor, 111 = CblasNoTrans
        cblas_sgemv(
            101,
            111,
            m as i32,
            n as i32,
            1.0,
            w.as_ptr(),
            n as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..m {
        y[i] = w[i * n..(i + 1) * n].iter().zip(x).map(|(&a, &b)| a * b).sum();
    }
}

/// `C = A·B` for row-major `A: [m, k]`, `B: [k, n]`, `C: [m, n]`.
pub fn sgemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    #[cfg(target_os = "macos")]
    unsafe {
        // 101 = CblasRowMajor, 111 = CblasNoTrans
        cblas_sgemm(
            101,
            111,
            111,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|t| a[i * k + t] * b[t * n + j]).sum();
        }
    }
}

/// `C = A·Bᵀ` for row-major `A: [m, k]`, `B: [n, k]` (the safetensors
/// `[out, in]` layout applied to a `[T, in]` activation matrix), `C: [m, n]`.
pub fn sgemm_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    #[cfg(target_os = "macos")]
    unsafe {
        // 101 = CblasRowMajor, 111 = CblasNoTrans, 112 = CblasTrans
        cblas_sgemm(
            101,
            111,
            112,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] =
                a[i * k..(i + 1) * k].iter().zip(&b[j * k..(j + 1) * k]).map(|(&x, &y)| x * y).sum();
        }
    }
}

// ── row-parallel sgemv ──────────────────────────────────────────────────────
//
// The code predictor is ~5 GB of strictly sequential weight traffic per frame
// through one Accelerate thread. Splitting each gemv's ROWS across a few
// threads multiplies effective bandwidth without touching the math: row i's
// dot product is computed by the same cblas kernel whether it arrives in a
// full-matrix call or a row-block call — verified BIT-IDENTICAL for every
// shape this pool serves at every split 2..=8 (see PORT_NOTES). The one
// exception is the predictor's `down` [1024×3072]: at n=3072 the full-matrix
// call selects a different (column-blocked) kernel than any row-block call,
// so `down` must NOT go through the pool — it stays a plain `sgemv`, keeping
// the threaded path byte-identical to the serial one.
//
// Pool shape: `MARY_PRED_THREADS` ways total (0/1 disables), WORK-STEALING
// over a fixed chunk grid: rows are cut into ~4·ways equal chunks (min 64
// rows) and every participant — caller included — pulls the next chunk off an
// atomic counter. A preempted thread therefore delays only its current chunk,
// not an m/ways slice: on this shared machine the straggler tail is the
// entire cost of wider splits (measured: fixed slices at 4/6 ways ran SLOWER
// than serial under ambient load; chunk-grids at 64..512 rows are in the
// bit-exactness-verified set). Per-dispatch cost is one atomic bump + a
// (usually uncontended) notify. Workers spin briefly between jobs (the gaps
// inside a frame are µs-scale) and park on a condvar when the burst ends, so
// they cost nothing while the GPU talker runs.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

#[derive(Clone, Copy)]
struct Job {
    w: *const f32,
    x: *const f32,
    y: *mut f32,
    m: usize,
    n: usize,
    /// Rows per work-stealing chunk.
    chunk: usize,
}

struct Pool {
    epoch: AtomicUsize,
    /// Generation-tagged steal counter: `epoch << CHUNK_BITS | next_chunk`.
    /// Claimed by CAS (never blind fetch_add) so a stale worker that wakes
    /// after the pool moved on observes the generation mismatch and exits
    /// WITHOUT consuming a chunk of the new job or touching its buffers.
    next: AtomicUsize,
    /// Chunks completed for the current job (compute finished, results
    /// visible). All increments for job k happen before dispatcher k's wait
    /// releases, so a later job's reset cannot race them.
    done: AtomicUsize,
    stop: AtomicBool,
    job: std::cell::UnsafeCell<Job>,
    lock: Mutex<()>,
    cv: Condvar,
    /// Total ways including the calling thread.
    ways: usize,
}

// The job pointer is published strictly before the epoch bump (Release) and
// read strictly after (Acquire); the dispatcher owns the slot until all
// workers report done. Raw pointers never outlive the dispatch call.
unsafe impl Sync for Pool {}
unsafe impl Send for Pool {}

/// Chunk-index bits in the generation-tagged steal counter (max ~1M chunks
/// per gemv; the generation uses the remaining high bits).
const CHUNK_BITS: u32 = 20;

/// Steal chunks off `pool.next` for generation `gen` and compute them until
/// the grid is empty or the pool has moved to a newer generation.
fn steal(pool: &Pool, job: &Job, gen: usize) {
    let n_chunks = job.m.div_ceil(job.chunk);
    loop {
        // claim by CAS: a stale thread (gen mismatch) must not consume an
        // index from the new job's grid.
        let v = pool.next.load(Ordering::Acquire);
        let (g, i) = (v >> CHUNK_BITS, v & ((1 << CHUNK_BITS) - 1));
        if g != gen || i >= n_chunks {
            return;
        }
        if pool
            .next
            .compare_exchange_weak(v, v + 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            continue;
        }
        let start = i * job.chunk;
        let len = job.chunk.min(job.m - start);
        unsafe {
            let w = std::slice::from_raw_parts(job.w.add(start * job.n), len * job.n);
            let x = std::slice::from_raw_parts(job.x, job.n);
            let y = std::slice::from_raw_parts_mut(job.y.add(start), len);
            sgemv(w, len, job.n, x, y);
        }
        pool.done.fetch_add(1, Ordering::AcqRel);
    }
}

fn worker(pool: &'static Pool, _idx: usize) {
    set_interactive_qos();
    // epoch is 0 at spawn; a dispatch may already have bumped it by the time
    // this thread runs, and that job must not be skipped.
    let mut seen = 0usize;
    loop {
        // spin briefly for the next job, then park until dispatched
        let mut spins = 0u32;
        loop {
            if pool.stop.load(Ordering::Acquire) {
                return;
            }
            let e = pool.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            spins += 1;
            if spins > 50_000 {
                let mut g = pool.lock.lock().unwrap();
                loop {
                    if pool.stop.load(Ordering::Acquire) {
                        return;
                    }
                    let e = pool.epoch.load(Ordering::Acquire);
                    if e != seen {
                        seen = e;
                        break;
                    }
                    g = pool.cv.wait(g).unwrap();
                }
                break;
            }
            std::hint::spin_loop();
        }
        let job = unsafe { *pool.job.get() };
        steal(pool, &job, seen);
    }
}

fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        // Default 2: Accelerate's sgemv runs on the AMX/SME units (one per
        // P-cluster, two on an M4 Max), so two concurrent streams saturate
        // them — the measured sweep is monotone worse past 2 ways
        // (ws-chunked predictor ms/frame: 2→30.0, 3→33.5, 4→31-34, 6→35-41;
        // serial 36) and wider only adds queueing + spin pressure.
        let ways = std::env::var("MARY_PRED_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2)
            .min(8); // splits 2..=8 are the bit-exactness-verified range
        if ways < 2 {
            return None;
        }
        let p: &'static Pool = Box::leak(Box::new(Pool {
            epoch: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: std::cell::UnsafeCell::new(Job {
                w: std::ptr::null(),
                x: std::ptr::null(),
                y: std::ptr::null_mut(),
                m: 0,
                n: 0,
                chunk: 1,
            }),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            ways,
        }));
        for i in 1..ways {
            std::thread::Builder::new()
                .name(format!("sgemv-pool-{i}"))
                .spawn(move || worker(p, i))
                .expect("spawn sgemv pool worker");
        }
        Some(p)
    })
}

/// `y = W·x` like [`sgemv`], work-stealing row chunks across the pool.
/// Bit-identical to the serial call for the shapes it serves (row blocks at
/// every split 2..=8 AND chunk grids at 64..512 rows verified); do NOT route
/// matrices with n ≥ 3072 here without re-running the split test — wide-n
/// full-matrix calls can select a different cblas kernel.
pub fn sgemv_mt(w: &[f32], m: usize, n: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), m * n);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    let Some(pool) = pool() else { return sgemv(w, m, n, x, y) };
    // ~4 chunks per way (min 64 rows, 64-row aligned) balances steal
    // granularity against per-chunk call overhead.
    let chunk = (m.div_ceil(4 * pool.ways)).next_multiple_of(64).max(64);
    let n_chunks = m.div_ceil(chunk);
    static DISPATCH: Mutex<()> = Mutex::new(());
    let _d = DISPATCH.lock().unwrap();
    unsafe {
        *pool.job.get() = Job { w: w.as_ptr(), x: x.as_ptr(), y: y.as_mut_ptr(), m, n, chunk };
    }
    let gen = pool.epoch.load(Ordering::Relaxed) + 1;
    pool.done.store(0, Ordering::Release);
    pool.next.store(gen << CHUNK_BITS, Ordering::Release);
    pool.epoch.store(gen, Ordering::Release);
    {
        let _g = pool.lock.lock().unwrap();
        pool.cv.notify_all();
    }
    let job = unsafe { *pool.job.get() };
    steal(pool, &job, gen);
    while pool.done.load(Ordering::Acquire) < n_chunks {
        std::hint::spin_loop();
    }
}

/// Weighted RMSNorm: `out = x · rsqrt(mean(x²)+eps) · w`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f64, out: &mut [f32]) {
    let mean: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let s = ((mean + eps).sqrt().recip()) as f32;
    for ((o, &v), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = v * s * wi;
    }
}

/// In-place softmax (max-subtracted).
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::MIN, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}
