//! PersonaPlex packaged voice-prompt (`.pt`) reader — Phase 5.
//!
//! A packaged voice is `torch.save({"embeddings": [N,1,1,4096], "cache":
//! [1,17,4]})` (moshi `lm.py` `_step_voice_prompt_core`'s
//! `save_voice_prompt_embeddings` branch): the temporal stack's INPUT
//! embeddings for each voice frame (replayed via `step_embeddings`, bypassing
//! the embedding tables) plus the token-ring snapshot that overwrites the
//! `StreamCache` after the replay. NVIDIA's stock voices (`voices.tgz`:
//! NATM0.pt, …) store bf16 embeddings with `cuda:0` LongStorage caches (the
//! upstream `load_voice_prompt_embeddings` CPU `map_location` trap — a device
//! *string* in the pickle, irrelevant to this parser); voices built by
//! `golden/build_voice_prompt.py` (the upstream WAV→.pt flow on CPU f32,
//! e.g. ref_voice.pt) store f32. bf16→f32 is a bit-shift, so stock voices
//! load **bit-exactly** (gated vs `vp_embeddings.npy` in `personaplex_probe
//! prompt`).
//!
//! Torch `.pt` format: an UNCOMPRESSED zip (`<stem>/data.pkl` + per-storage
//! `<stem>/data/<key>` blobs, little-endian) whose pickle builds the dict via
//! `torch._utils._rebuild_tensor_v2(persid((storage, <Type>Storage, key,
//! device, numel)), offset, shape, strides, requires_grad, hooks)`. The
//! stack machine below interprets exactly that opcode surface (protocol ≤ 4
//! saves of tensor dicts) — not a general unpickler, and deliberately so:
//! anything unexpected panics with the offending opcode instead of guessing.

use std::collections::HashMap;
use std::path::Path;

use super::config as cfg;
use super::lmgen::CT;

pub struct VoicePrompt {
    /// Embedding-replay steps (rows of `[cfg::DIM]`).
    pub n_frames: usize,
    /// `[n_frames * cfg::DIM]` row-major f32.
    pub embeddings: Vec<f32>,
    /// Token-ring snapshot `[NUM_STREAMS * CT]` row-major (overwrites the
    /// `StreamCache` after the replay).
    pub cache: Vec<i64>,
}

// ─────────────────────────────── zip reader ────────────────────────────────

/// Entry name → raw (STORED) payload. Torch writes uncompressed zips; a
/// deflated entry panics (none observed in torch saves).
fn zip_entries(data: &[u8]) -> HashMap<String, &[u8]> {
    // EOCD: scan back for PK\x05\x06 (comment ≤ 64 KiB).
    let eocd = (0..data.len().saturating_sub(21))
        .rev()
        .find(|&i| data[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .expect("zip: no end-of-central-directory");
    let u16at = |i: usize| u16::from_le_bytes(data[i..i + 2].try_into().unwrap()) as usize;
    let u32at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
    let n_entries = u16at(eocd + 10);
    let mut cd = u32at(eocd + 16);

    let mut map = HashMap::new();
    for _ in 0..n_entries {
        assert_eq!(
            &data[cd..cd + 4],
            b"PK\x01\x02",
            "zip: central directory header"
        );
        let method = u16at(cd + 10);
        let csize = u32at(cd + 20);
        let usize_ = u32at(cd + 24);
        let name_len = u16at(cd + 28);
        let extra_len = u16at(cd + 30);
        let comment_len = u16at(cd + 32);
        let lho = u32at(cd + 42);
        let name = std::str::from_utf8(&data[cd + 46..cd + 46 + name_len])
            .expect("zip: entry name utf8")
            .to_string();
        assert_eq!(
            method, 0,
            "zip: {name}: only STORED entries (torch saves) supported"
        );
        assert_eq!(csize, usize_, "zip: {name}: stored size mismatch");
        // Local header: sizes may be in a data descriptor; name/extra lengths
        // are authoritative here.
        assert_eq!(&data[lho..lho + 4], b"PK\x03\x04", "zip: local header");
        let lname = u16at(lho + 26);
        let lextra = u16at(lho + 28);
        let start = lho + 30 + lname + lextra;
        map.insert(name, &data[start..start + usize_]);
        cd += 46 + name_len + extra_len + comment_len;
    }
    map
}

// ────────────────────────── targeted pickle machine ────────────────────────

#[derive(Clone, Debug)]
enum V {
    None,
    /// Payload only read via `Debug` in panic messages (requires_grad flags).
    #[allow(dead_code)]
    Bool(bool),
    Int(i64),
    Str(String),
    Tuple(Vec<V>),
    Dict(Vec<(V, V)>),
    /// `GLOBAL 'module name'` as "module name".
    Global(String),
    Tensor(TensorStub),
    /// Stack sentinel for MARK.
    Mark,
}

#[derive(Clone, Debug)]
struct TensorStub {
    storage_key: String,
    /// Storage class name, e.g. "BFloat16Storage" / "FloatStorage" /
    /// "LongStorage".
    storage_type: String,
    numel: usize,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl V {
    fn as_int(&self) -> i64 {
        match self {
            V::Int(v) => *v,
            _ => panic!("pickle: expected int, got {self:?}"),
        }
    }
    fn as_usize_vec(&self) -> Vec<usize> {
        match self {
            V::Tuple(t) => t.iter().map(|v| v.as_int() as usize).collect(),
            _ => panic!("pickle: expected tuple, got {self:?}"),
        }
    }
}

fn unpickle(b: &[u8]) -> V {
    let mut stack: Vec<V> = Vec::new();
    let mut memo: HashMap<u64, V> = HashMap::new();
    let mut i = 0usize;
    macro_rules! pop {
        () => {
            stack.pop().expect("pickle: stack underflow")
        };
    }
    loop {
        let op = b[i];
        i += 1;
        match op {
            0x80 => i += 1,                           // PROTO n
            b'}' => stack.push(V::Dict(Vec::new())),  // EMPTY_DICT
            b')' => stack.push(V::Tuple(Vec::new())), // EMPTY_TUPLE
            b'(' => stack.push(V::Mark),              // MARK
            b'N' => stack.push(V::None),
            0x88 => stack.push(V::Bool(true)),  // NEWTRUE
            0x89 => stack.push(V::Bool(false)), // NEWFALSE
            b'K' => {
                stack.push(V::Int(b[i] as i64)); // BININT1
                i += 1;
            }
            b'M' => {
                stack.push(V::Int(
                    u16::from_le_bytes(b[i..i + 2].try_into().unwrap()) as i64
                )); // BININT2
                i += 2;
            }
            b'J' => {
                stack.push(V::Int(
                    i32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as i64
                )); // BININT
                i += 4;
            }
            b'X' => {
                // BINUNICODE
                let n = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
                i += 4;
                stack.push(V::Str(String::from_utf8(b[i..i + n].to_vec()).unwrap()));
                i += n;
            }
            0x8c => {
                // SHORT_BINUNICODE (protocol 4)
                let n = b[i] as usize;
                i += 1;
                stack.push(V::Str(String::from_utf8(b[i..i + n].to_vec()).unwrap()));
                i += n;
            }
            b'c' => {
                // GLOBAL 'module\nname\n'
                let s = i;
                while b[i] != b'\n' {
                    i += 1;
                }
                let module = std::str::from_utf8(&b[s..i]).unwrap();
                i += 1;
                let s = i;
                while b[i] != b'\n' {
                    i += 1;
                }
                let name = std::str::from_utf8(&b[s..i]).unwrap();
                i += 1;
                stack.push(V::Global(format!("{module} {name}")));
            }
            0x93 => {
                // STACK_GLOBAL (protocol 4): module, name on the stack
                let name = match pop!() {
                    V::Str(s) => s,
                    v => panic!("pickle: STACK_GLOBAL name {v:?}"),
                };
                let module = match pop!() {
                    V::Str(s) => s,
                    v => panic!("pickle: STACK_GLOBAL module {v:?}"),
                };
                stack.push(V::Global(format!("{module} {name}")));
            }
            b'q' => {
                memo.insert(b[i] as u64, stack.last().expect("memo").clone()); // BINPUT
                i += 1;
            }
            b'r' => {
                let k = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as u64; // LONG_BINPUT
                i += 4;
                memo.insert(k, stack.last().expect("memo").clone());
            }
            0x94 => {
                memo.insert(memo.len() as u64, stack.last().expect("memo").clone());
                // MEMOIZE
            }
            b'h' => {
                stack.push(memo[&(b[i] as u64)].clone()); // BINGET
                i += 1;
            }
            b'j' => {
                let k = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as u64; // LONG_BINGET
                i += 4;
                stack.push(memo[&k].clone());
            }
            b't' => {
                // TUPLE (to MARK)
                let mut items = Vec::new();
                loop {
                    match pop!() {
                        V::Mark => break,
                        v => items.push(v),
                    }
                }
                items.reverse();
                stack.push(V::Tuple(items));
            }
            0x85 | 0x86 | 0x87 => {
                // TUPLE1/2/3
                let n = (op - 0x84) as usize;
                let mut items: Vec<V> = (0..n).map(|_| pop!()).collect();
                items.reverse();
                stack.push(V::Tuple(items));
            }
            b'Q' => {
                // BINPERSID: persistent id tuple ('storage', Global, key, device, numel)
                let pid = match pop!() {
                    V::Tuple(t) => t,
                    v => panic!("pickle: BINPERSID arg {v:?}"),
                };
                assert!(
                    matches!(&pid[0], V::Str(s) if s == "storage"),
                    "persid kind"
                );
                let storage_type = match &pid[1] {
                    V::Global(g) => g
                        .strip_prefix("torch ")
                        .unwrap_or_else(|| panic!("storage class {g}"))
                        .to_string(),
                    v => panic!("pickle: storage class {v:?}"),
                };
                let key = match &pid[2] {
                    V::Str(s) => s.clone(),
                    v => panic!("pickle: storage key {v:?}"),
                };
                let numel = pid[4].as_int() as usize;
                // Represent the pending storage as a stub with no view yet.
                stack.push(V::Tensor(TensorStub {
                    storage_key: key,
                    storage_type,
                    numel,
                    offset: 0,
                    shape: Vec::new(),
                    strides: Vec::new(),
                }));
            }
            b'R' => {
                // REDUCE
                let args = match pop!() {
                    V::Tuple(t) => t,
                    v => panic!("pickle: REDUCE args {v:?}"),
                };
                let callable = match pop!() {
                    V::Global(g) => g,
                    v => panic!("pickle: REDUCE callable {v:?}"),
                };
                match callable.as_str() {
                    "torch._utils _rebuild_tensor_v2" => {
                        let mut stub = match &args[0] {
                            V::Tensor(t) => t.clone(),
                            v => panic!("pickle: _rebuild_tensor_v2 storage {v:?}"),
                        };
                        stub.offset = args[1].as_int() as usize;
                        stub.shape = args[2].as_usize_vec();
                        stub.strides = args[3].as_usize_vec();
                        stack.push(V::Tensor(stub));
                    }
                    "collections OrderedDict" => stack.push(V::Dict(Vec::new())),
                    other => panic!("pickle: unsupported callable {other}"),
                }
            }
            b's' => {
                // SETITEM
                let v = pop!();
                let k = pop!();
                match stack.last_mut() {
                    Some(V::Dict(d)) => d.push((k, v)),
                    v => panic!("pickle: SETITEM target {v:?}"),
                }
            }
            b'u' => {
                // SETITEMS (to MARK)
                let mut kv = Vec::new();
                loop {
                    match pop!() {
                        V::Mark => break,
                        v => kv.push(v),
                    }
                }
                kv.reverse();
                match stack.last_mut() {
                    Some(V::Dict(d)) => {
                        for pair in kv.chunks_exact(2) {
                            d.push((pair[0].clone(), pair[1].clone()));
                        }
                    }
                    v => panic!("pickle: SETITEMS target {v:?}"),
                }
            }
            b'.' => return pop!(), // STOP
            other => panic!("pickle: unsupported opcode 0x{other:02x} at {}", i - 1),
        }
    }
}

// ─────────────────────────────── materialize ───────────────────────────────

fn contiguous(shape: &[usize], strides: &[usize]) -> bool {
    let mut expect = 1usize;
    for (d, s) in shape.iter().zip(strides).rev() {
        if *d != 1 && *s != expect {
            return false;
        }
        expect *= d;
    }
    true
}

fn to_f32(stub: &TensorStub, raw: &[u8]) -> Vec<f32> {
    let n: usize = stub.shape.iter().product();
    assert_eq!(stub.offset, 0, "voice prompt: nonzero storage offset");
    assert!(
        contiguous(&stub.shape, &stub.strides),
        "voice prompt: non-contiguous tensor"
    );
    match stub.storage_type.as_str() {
        "FloatStorage" => {
            assert_eq!(raw.len(), n * 4, "f32 storage size");
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        "BFloat16Storage" => {
            assert_eq!(raw.len(), n * 2, "bf16 storage size");
            raw.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect()
        }
        "HalfStorage" => {
            assert_eq!(raw.len(), n * 2, "f16 storage size");
            raw.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect()
        }
        other => panic!("voice prompt: unsupported embedding storage {other}"),
    }
}

impl VoicePrompt {
    pub fn load(path: &Path) -> Self {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("voice prompt {path:?}: {e}"));
        let entries = zip_entries(&data);
        let pkl_name = entries
            .keys()
            .find(|k| k.ends_with("/data.pkl"))
            .unwrap_or_else(|| panic!("voice prompt: no data.pkl in {path:?}"))
            .clone();
        let prefix = pkl_name.strip_suffix("data.pkl").unwrap().to_string();
        if let Some(bo) = entries.get(&format!("{prefix}byteorder")) {
            assert_eq!(
                &bo[..6.min(bo.len())],
                b"little",
                "voice prompt: big-endian save"
            );
        }

        let root = unpickle(entries[&pkl_name]);
        let V::Dict(kv) = root else {
            panic!("voice prompt: root is not a dict")
        };
        let get = |name: &str| -> &TensorStub {
            kv.iter()
                .find_map(|(k, v)| match (k, v) {
                    (V::Str(s), V::Tensor(t)) if s == name => Some(t),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("voice prompt: no '{name}' tensor"))
        };

        // embeddings [N,1,1,4096] → [N*4096] f32
        let emb = get("embeddings");
        assert_eq!(emb.shape.len(), 4, "embeddings rank");
        assert_eq!(&emb.shape[1..], &[1, 1, cfg::DIM], "embeddings shape");
        let n_frames = emb.shape[0];
        assert_eq!(emb.numel, n_frames * cfg::DIM, "embeddings numel");
        let embeddings = to_f32(emb, entries[&format!("{prefix}data/{}", emb.storage_key)]);

        // cache [1,17,CT] i64
        let c = get("cache");
        assert_eq!(c.shape, vec![1, cfg::NUM_STREAMS, CT], "cache shape");
        assert_eq!(c.storage_type, "LongStorage", "cache storage");
        assert!(contiguous(&c.shape, &c.strides), "cache strides");
        assert_eq!(c.offset, 0, "cache offset");
        let raw = entries[&format!("{prefix}data/{}", c.storage_key)];
        assert_eq!(raw.len(), cfg::NUM_STREAMS * CT * 8, "cache storage size");
        let cache: Vec<i64> = raw
            .chunks_exact(8)
            .map(|ch| i64::from_le_bytes(ch.try_into().unwrap()))
            .collect();

        Self {
            n_frames,
            embeddings,
            cache,
        }
    }
}
