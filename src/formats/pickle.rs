//! PyTorch `.bin` / `.pth` (pickled `state_dict`) → `(name, f32-data, shape)`
//! extractor for the import path.
//!
//! Many older / embedder / tiny models ship ONLY as a pickled PyTorch
//! `state_dict` (`pytorch_model.bin`), never safetensors. The modern PyTorch
//! save format (torch >= 1.6) is a ZIP: `data.pkl` pickles the tensor metadata
//! and `data/<key>` members hold the raw, uncompressed storage bytes.
//!
//! Rather than depend on a fragile third-party pickle crate (the ones on
//! crates.io either reject the plain-dict `torch.save(state_dict)` shape or choke
//! on torch's `persistent_id` storage refs), this module walks `data.pkl` with a
//! small, self-contained pickle-opcode interpreter tailored to torch
//! `state_dict`s. The torch opcode set is small and stable: it stores a dict of
//! `name -> _rebuild_tensor_v2(storage, offset, shape, stride, requires_grad,
//! ...)`, where `storage` is a `persistent_id` tuple `('storage', <DtypeStorage>,
//! <key>, <device>, <numel>)`. We recover `(dtype, shape, storage key, element
//! offset)` from that, read the bytes from the ZIP, and convert to `Vec<f32>` —
//! feeding the SAME content-addressed path safetensors uses.
//!
//! Only float tensors are imported (the forward never loads int buffers).
//! f32/f16/bf16/f64 storages are supported; the legacy (non-ZIP, torch < 1.6)
//! pickle format is not — practically every HuggingFace `pytorch_model.bin` is
//! the modern ZIP form.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};

/// Read a PyTorch pickle `.bin`/`.pth` and return every float tensor as
/// `(name, f32-data, row-major shape)`.
pub fn extract_tensors(path: &Path) -> Result<Vec<(String, Vec<f32>, Vec<usize>)>> {
    let mut zip = zip::ZipArchive::new(File::open(path)?)
        .map_err(|e| anyhow!("open {path:?} as zip: {e}"))?;

    // Locate the `.../data.pkl` member and its sibling storage prefix.
    let data_pkl = zip
        .file_names()
        .find(|s| s.ends_with("/data.pkl") || *s == "data.pkl")
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("no data.pkl in {path:?} — not a torch>=1.6 checkpoint"))?;
    let prefix = data_pkl.strip_suffix("data.pkl").unwrap_or("").to_string();

    let pkl_bytes = {
        let mut f = zip
            .by_name(&data_pkl)
            .map_err(|e| anyhow!("read data.pkl: {e}"))?;
        let mut buf = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut buf)?;
        buf
    };

    let tensors = parse_state_dict(&pkl_bytes).context("parse torch state_dict pickle")?;

    // Read (and cache) each referenced storage member, slice by element offset.
    let mut storage_cache: HashMap<String, Vec<u8>> = HashMap::new();
    let mut out = Vec::new();
    for t in tensors {
        let itemsize = t.dtype.size();
        if itemsize == 0 || !t.dtype.is_float() {
            continue; // skip int/unknown buffers
        }
        let numel: usize = t.shape.iter().product::<usize>().max(1);
        let member = format!("{prefix}data/{}", t.storage_key);
        if !storage_cache.contains_key(&member) {
            let mut f = zip
                .by_name(&member)
                .map_err(|e| anyhow!("storage member {member:?}: {e}"))?;
            let mut buf = Vec::with_capacity(f.size() as usize);
            f.read_to_end(&mut buf)?;
            storage_cache.insert(member.clone(), buf);
        }
        let bytes = &storage_cache[&member];
        let start = t.storage_offset * itemsize;
        let end = start + numel * itemsize;
        if end > bytes.len() {
            bail!(
                "tensor {:?}: storage slice {start}..{end} exceeds member {member:?} ({} bytes)",
                t.name,
                bytes.len()
            );
        }
        let f32s = to_f32(t.dtype, &bytes[start..end])
            .with_context(|| format!("convert pytorch tensor {:?}", t.name))?;
        out.push((t.name, f32s, t.shape));
    }
    if out.is_empty() {
        bail!(
            "no float tensors found in pytorch pickle {path:?} (empty or all-integer state_dict?)"
        );
    }
    Ok(out)
}

/// A torch storage dtype, recovered from the `<Dtype>Storage` global in the
/// tensor's `persistent_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    fn from_storage_class(s: &str) -> Option<Self> {
        // e.g. "FloatStorage", "HalfStorage", "BFloat16Storage", "LongStorage".
        let s = s.strip_suffix("Storage").unwrap_or(s);
        Some(match s {
            "Double" => Dtype::F64,
            "Float" => Dtype::F32,
            "Half" => Dtype::F16,
            "BFloat16" => Dtype::BF16,
            "Long" => Dtype::I64,
            "Int" => Dtype::I32,
            "Short" => Dtype::I16,
            "Char" => Dtype::I8,
            "Byte" => Dtype::U8,
            "Bool" => Dtype::Bool,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }
    fn is_float(self) -> bool {
        matches!(self, Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16)
    }
}

fn to_f32(dtype: Dtype, raw: &[u8]) -> Result<Vec<f32>> {
    let v = match dtype {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::F64 => raw
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        other => bail!("unsupported pytorch tensor dtype {other:?}"),
    };
    Ok(v)
}

/// One tensor recovered from the pickle.
#[derive(Debug)]
struct TorchTensor {
    name: String,
    dtype: Dtype,
    storage_key: String,
    storage_offset: usize,
    shape: Vec<usize>,
}

// ---- minimal pickle interpreter -----------------------------------------

/// A pickle value on the interpreter stack. We only model what a torch
/// `state_dict` needs.
#[derive(Clone, Debug)]
enum Val {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    /// A `bytes` object. Torch tensor metadata never carries a bytes value, so
    /// we keep only a placeholder (the raw contents are skipped when parsing the
    /// opcode) — enough to keep the stack shape correct.
    Bytes,
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Mark,
    /// A resolved `GLOBAL modname.name` (we only care about the name for torch).
    Global(String, String),
    /// The result of `persistent_id` — the raw argument tuple.
    Persid(Box<Val>),
    /// The result of a `REDUCE`/`BUILD` we don't specifically model: keep the
    /// callable + args so `_rebuild_tensor_v2` can be recognized downstream.
    Reduce(Box<Val>, Vec<Val>),
}

/// Walk `data.pkl` and return every tensor entry of the toplevel state dict.
fn parse_state_dict(buf: &[u8]) -> Result<Vec<TorchTensor>> {
    let mut stack: Vec<Val> = Vec::new();
    let mut memo: HashMap<u32, Val> = HashMap::new();
    let mut i = 0usize;
    let mut memo_counter: u32 = 0;

    macro_rules! pop {
        () => {
            stack
                .pop()
                .ok_or_else(|| anyhow!("pickle stack underflow"))?
        };
    }
    // Pop everything back to (and discarding) the topmost Mark; return the popped
    // values in stack order.
    fn pop_to_mark(stack: &mut Vec<Val>) -> Result<Vec<Val>> {
        let mut items = Vec::new();
        loop {
            match stack.pop() {
                Some(Val::Mark) => break,
                Some(v) => items.push(v),
                None => bail!("pickle: no MARK found"),
            }
        }
        items.reverse();
        Ok(items)
    }

    while i < buf.len() {
        let op = buf[i];
        i += 1;
        match op {
            b'\x80' => {
                i += 1; // PROTO <version byte>
            }
            b'.' => break,                              // STOP
            b'(' => stack.push(Val::Mark),              // MARK
            b'}' => stack.push(Val::Dict(Vec::new())),  // EMPTY_DICT
            b']' => stack.push(Val::List(Vec::new())),  // EMPTY_LIST
            b')' => stack.push(Val::Tuple(Vec::new())), // EMPTY_TUPLE
            b'N' => stack.push(Val::None),              // NONE
            b'\x88' => stack.push(Val::Bool(true)),     // NEWTRUE
            b'\x89' => stack.push(Val::Bool(false)),    // NEWFALSE
            // ---- ints ----
            b'K' => {
                let v = buf[i] as i64;
                i += 1;
                stack.push(Val::Int(v));
            } // BININT1
            b'M' => {
                let v = u16::from_le_bytes([buf[i], buf[i + 1]]) as i64;
                i += 2;
                stack.push(Val::Int(v));
            } // BININT2
            b'J' => {
                let v = i32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as i64;
                i += 4;
                stack.push(Val::Int(v));
            } // BININT
            b'\x8a' => {
                // LONG1: 1-byte length + little-endian signed
                let n = buf[i] as usize;
                i += 1;
                let mut v: i64 = 0;
                for k in 0..n {
                    v |= (buf[i + k] as i64) << (8 * k);
                }
                if n > 0 && (buf[i + n - 1] & 0x80) != 0 {
                    v -= 1i64 << (8 * n); // sign-extend
                }
                i += n;
                stack.push(Val::Int(v));
            }
            // ---- strings ----
            b'X' => {
                let len = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
                i += 4;
                let s = String::from_utf8_lossy(&buf[i..i + len]).into_owned();
                i += len;
                stack.push(Val::Str(s));
            } // BINUNICODE
            b'\x8c' => {
                let len = buf[i] as usize;
                i += 1;
                let s = String::from_utf8_lossy(&buf[i..i + len]).into_owned();
                i += len;
                stack.push(Val::Str(s));
            } // SHORT_BINUNICODE
            b'\x8d' => {
                let len = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let s = String::from_utf8_lossy(&buf[i..i + len]).into_owned();
                i += len;
                stack.push(Val::Str(s));
            } // BINUNICODE8
            // ---- bytes ----
            b'C' => {
                let len = buf[i] as usize;
                i += 1;
                i += len; // skip the bytes payload
                stack.push(Val::Bytes);
            } // SHORT_BINBYTES
            b'B' => {
                let len = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
                i += 4;
                i += len; // skip the bytes payload
                stack.push(Val::Bytes);
            } // BINBYTES
            // ---- memo ----
            b'q' => {
                let idx = buf[i] as u32;
                i += 1;
                memo.insert(idx, stack.last().cloned().unwrap_or(Val::None));
            } // BINPUT
            b'r' => {
                let idx = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
                i += 4;
                memo.insert(idx, stack.last().cloned().unwrap_or(Val::None));
            } // LONG_BINPUT
            b'\x94' => {
                memo.insert(memo_counter, stack.last().cloned().unwrap_or(Val::None));
                memo_counter += 1;
            } // MEMOIZE
            b'h' => {
                let idx = buf[i] as u32;
                i += 1;
                stack.push(
                    memo.get(&idx)
                        .cloned()
                        .ok_or_else(|| anyhow!("BINGET miss {idx}"))?,
                );
            } // BINGET
            b'j' => {
                let idx = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
                i += 4;
                stack.push(
                    memo.get(&idx)
                        .cloned()
                        .ok_or_else(|| anyhow!("LONG_BINGET miss {idx}"))?,
                );
            } // LONG_BINGET
            // ---- tuples ----
            b'\x85' => {
                let a = pop!();
                stack.push(Val::Tuple(vec![a]));
            } // TUPLE1
            b'\x86' => {
                let b = pop!();
                let a = pop!();
                stack.push(Val::Tuple(vec![a, b]));
            } // TUPLE2
            b'\x87' => {
                let c = pop!();
                let b = pop!();
                let a = pop!();
                stack.push(Val::Tuple(vec![a, b, c]));
            } // TUPLE3
            b't' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(Val::Tuple(items));
            } // TUPLE
            b'l' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(Val::List(items));
            } // LIST
            b'e' => {
                // APPENDS: extend the list below the mark
                let items = pop_to_mark(&mut stack)?;
                if let Some(Val::List(l)) = stack.last_mut() {
                    l.extend(items);
                } else {
                    bail!("APPENDS with no list on stack");
                }
            }
            b'a' => {
                // APPEND
                let v = pop!();
                if let Some(Val::List(l)) = stack.last_mut() {
                    l.push(v);
                } else {
                    bail!("APPEND with no list on stack");
                }
            }
            // ---- dict building ----
            b'u' => {
                // SETITEMS: pop k,v pairs back to mark, insert into dict below
                let items = pop_to_mark(&mut stack)?;
                let pairs: Vec<(Val, Val)> = items
                    .chunks_exact(2)
                    .map(|c| (c[0].clone(), c[1].clone()))
                    .collect();
                if let Some(Val::Dict(d)) = stack.last_mut() {
                    d.extend(pairs);
                } else {
                    bail!("SETITEMS with no dict on stack");
                }
            }
            b's' => {
                // SETITEM
                let v = pop!();
                let k = pop!();
                if let Some(Val::Dict(d)) = stack.last_mut() {
                    d.push((k, v));
                } else {
                    bail!("SETITEM with no dict on stack");
                }
            }
            // ---- globals / reduce / build / persid ----
            b'c' => {
                // GLOBAL: two newline-terminated strings (module\nname)
                let start = i;
                while buf[i] != b'\n' {
                    i += 1;
                }
                let module = String::from_utf8_lossy(&buf[start..i]).into_owned();
                i += 1;
                let s2 = i;
                while buf[i] != b'\n' {
                    i += 1;
                }
                let name = String::from_utf8_lossy(&buf[s2..i]).into_owned();
                i += 1;
                stack.push(Val::Global(module, name));
            }
            b'\x93' => {
                // STACK_GLOBAL: name and module are on the stack (name on top)
                let name = pop!();
                let module = pop!();
                let (m, n) = match (module, name) {
                    (Val::Str(m), Val::Str(n)) => (m, n),
                    _ => bail!("STACK_GLOBAL with non-string operands"),
                };
                stack.push(Val::Global(m, n));
            }
            b'Q' => {
                // BINPERSID: persistent id — the argument is the tuple on top
                let arg = pop!();
                stack.push(Val::Persid(Box::new(arg)));
            }
            b'R' => {
                // REDUCE: callable + argtuple
                let args = pop!();
                let callable = pop!();
                let argv = match args {
                    Val::Tuple(v) => v,
                    other => vec![other],
                };
                // `torch.save(state_dict)` uses a zero-argument
                // `collections.OrderedDict()` as the mutable toplevel mapping,
                // then fills it with SETITEMS. Normalize exactly that known
                // reduction into our dict representation; other reductions
                // retain their callable and arguments for tensor recognition.
                if argv.is_empty()
                    && matches!(&callable, Val::Global(module, name)
                        if module == "collections" && name == "OrderedDict")
                {
                    stack.push(Val::Dict(Vec::new()));
                } else {
                    stack.push(Val::Reduce(Box::new(callable), argv));
                }
            }
            b'\x81' => {
                // NEWOBJ: cls + argtuple -> object (model as Reduce for our purposes)
                let args = pop!();
                let cls = pop!();
                let argv = match args {
                    Val::Tuple(v) => v,
                    other => vec![other],
                };
                stack.push(Val::Reduce(Box::new(cls), argv));
            }
            b'b' => {
                // BUILD: state on top, object below. For torch state_dicts the
                // object is the dict (via __setstate__); keep the object as-is.
                let _state = pop!();
                // leave the object on the stack unchanged
            }
            b'0' => {
                let _ = pop!();
            } // POP
            b'2' => {
                stack.push(stack.last().cloned().unwrap_or(Val::None));
            } // DUP
            b'G' => {
                i += 8; // BINFLOAT (unused by tensor metadata)
                stack.push(Val::None);
            }
            other => bail!("unhandled pickle opcode 0x{other:02x} at offset {}", i - 1),
        }
    }

    // The toplevel result is the state dict (last value on the stack).
    let root = stack.pop().ok_or_else(|| anyhow!("empty pickle result"))?;
    let dict = collect_dict(&root)
        .ok_or_else(|| anyhow!("toplevel pickle value is not a dict-like state_dict"))?;

    let mut tensors = Vec::new();
    for (k, v) in dict {
        let name = match k {
            Val::Str(s) => s,
            _ => continue,
        };
        if let Some(t) = tensor_from_val(&name, &v) {
            tensors.push(t);
        }
    }
    Ok(tensors)
}

/// Extract a dict's `(key, value)` pairs from a `Dict` or an `OrderedDict`
/// built via reduce (`collections.OrderedDict` REDUCE-then-BUILD).
fn collect_dict(v: &Val) -> Option<Vec<(Val, Val)>> {
    match v {
        Val::Dict(d) => Some(d.clone()),
        Val::Reduce(_, args) => args.iter().find_map(collect_dict),
        _ => None,
    }
}

/// Recognize `torch._utils._rebuild_tensor_v2(storage, storage_offset, size,
/// stride, requires_grad, ...)` and pull out the tensor's dtype/shape/storage.
fn tensor_from_val(name: &str, v: &Val) -> Option<TorchTensor> {
    let Val::Reduce(callable, args) = v else {
        return None;
    };
    let is_rebuild = matches!(callable.as_ref(), Val::Global(m, n)
        if m == "torch._utils" && (n == "_rebuild_tensor_v2" || n == "_rebuild_tensor"));
    if !is_rebuild {
        return None;
    }
    // args: [ persid, storage_offset, size(tuple), stride(tuple), requires_grad, ... ]
    let persid = args.first()?;
    let storage_offset = as_int(args.get(1)?)? as usize;
    let shape = as_usize_tuple(args.get(2)?)?;

    // persid arg is Persid(Tuple(['storage', <DtypeStorage global>, key, device, numel]))
    let tup = match persid {
        Val::Persid(inner) => match inner.as_ref() {
            Val::Tuple(t) => t,
            _ => return None,
        },
        _ => return None,
    };
    // tup[0] == "storage", tup[1] == Global(_, "<Dtype>Storage"), tup[2] == key
    let dtype = match tup.get(1)? {
        Val::Global(_, cls) => Dtype::from_storage_class(cls)?,
        _ => return None,
    };
    let storage_key = match tup.get(2)? {
        Val::Str(s) => s.clone(),
        Val::Int(k) => k.to_string(),
        _ => return None,
    };

    Some(TorchTensor {
        name: name.to_string(),
        dtype,
        storage_key,
        storage_offset,
        shape,
    })
}

fn as_int(v: &Val) -> Option<i64> {
    match v {
        Val::Int(i) => Some(*i),
        Val::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

fn as_usize_tuple(v: &Val) -> Option<Vec<usize>> {
    match v {
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .map(|x| as_int(x).map(|n| n as usize))
            .collect(),
        _ => None,
    }
}
