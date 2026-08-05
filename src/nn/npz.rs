//! `.npz` archives (a zip of `.npy` members), and the dtype-general `.npy`
//! parsing [`super::npy`] deliberately does not do.
//!
//! [`super::npy`] is the f32 workhorse for goldens we write ourselves. Oracle
//! vectors captured out of *someone else's* runtime arrive in whatever dtype
//! that runtime used — float64 references, float32 kernel output, uint16
//! holding raw bfloat16 bit patterns — bundled into one `.npz`. Being able to
//! read that bundle is the whole reason a port can be gated against a third
//! party at all, so the reader belongs in the shared toolkit rather than
//! inside whichever probe binary needed it first.
//!
//! `numpy.savez` stores its members uncompressed (`ZIP_STORED`), so this walks
//! the central directory and slices the buffer. No inflate, no dependency; a
//! compressed member is refused by name rather than silently misread.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

/// A `.npy` member's elements, kept in the dtype the file actually holds.
///
/// Widening on load would be convenient and wrong: a comparison's tolerance
/// depends on which precision produced the numbers, and `u16` in particular is
/// ambiguous (a bfloat16 bit pattern, not a small integer), so it is never
/// converted implicitly.
#[derive(Debug, Clone)]
pub enum NpyData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    U16(Vec<u16>),
}

/// One array out of an `.npz`: its shape and its raw elements.
#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    /// Element count (the product of `shape`; 1 for a 0-d scalar array).
    pub fn len(&self) -> usize {
        match &self.data {
            NpyData::F64(v) => v.len(),
            NpyData::F32(v) => v.len(),
            NpyData::I64(v) => v.len(),
            NpyData::U16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Elements as f64. `u16` is refused: it means bfloat16 bits here, and
    /// reading it as an integer would give a silently plausible garbage array
    /// (values in 0..65535 instead of the ±0.05 the tensor actually holds).
    pub fn to_f64(&self) -> Vec<f64> {
        match &self.data {
            NpyData::F64(v) => v.clone(),
            NpyData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U16(_) => panic!(
                "uint16 array read as float: these hold raw bfloat16 bit patterns \
                 — use bf16_to_f64()"
            ),
        }
    }

    /// Elements as f32.
    pub fn to_f32(&self) -> Vec<f32> {
        self.to_f64().iter().map(|&x| x as f32).collect()
    }

    /// Reinterpret a `uint16` array as bfloat16: the 16 bits are the *top* half
    /// of an f32, so the widening is a shift, not a cast.
    pub fn bf16_to_f64(&self) -> Vec<f64> {
        match &self.data {
            NpyData::U16(v) => v
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16) as f64)
                .collect(),
            other => panic!("bf16_to_f64 on a non-uint16 array: {:?}", other_kind(other)),
        }
    }

    /// The scalar of a 0-d array.
    pub fn scalar(&self) -> f64 {
        assert_eq!(
            self.len(),
            1,
            "scalar() on an array of {} elements (shape {:?})",
            self.len(),
            self.shape
        );
        self.to_f64()[0]
    }
}

fn other_kind(d: &NpyData) -> &'static str {
    match d {
        NpyData::F64(_) => "f64",
        NpyData::F32(_) => "f32",
        NpyData::I64(_) => "i64",
        NpyData::U16(_) => "u16",
    }
}

/// A loaded `.npz`, members keyed by name without the `.npy` suffix (the name
/// `np.savez` was given).
pub struct Npz {
    members: HashMap<String, NpyArray>,
}

impl Npz {
    /// Read and parse every member. The whole archive is held in memory: these
    /// are reference vectors, sized to be read whole.
    pub fn open(path: &Path) -> io::Result<Self> {
        let buf = fs::read(path)?;
        let mut members = HashMap::new();
        for (name, bytes) in zip_stored_members(&buf, path)? {
            let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
            members.insert(key, parse_npy(bytes, &name));
        }
        Ok(Self { members })
    }

    /// Look up a member, failing with the available names rather than a bare
    /// `None` — a typo'd oracle key is otherwise a very quiet way to gate
    /// against nothing.
    pub fn get(&self, name: &str) -> &NpyArray {
        self.members.get(name).unwrap_or_else(|| {
            let mut have: Vec<&str> = self.members.keys().map(|s| s.as_str()).collect();
            have.sort_unstable();
            panic!("no array '{}' in npz; have: {}", name, have.join(", "))
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.members.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

fn le16(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes([b[o], b[o + 1]]) as usize
}

fn le32(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

/// Walk the zip central directory and return `(name, data)` for every member.
fn zip_stored_members<'a>(buf: &'a [u8], path: &Path) -> io::Result<Vec<(String, &'a [u8])>> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);

    // End-of-central-directory: the last 0x06054b50 within the trailing 64 KiB
    // (the comment field's maximum) plus the record's own 22 bytes.
    let eocd = (0..buf.len().saturating_sub(21))
        .rev()
        .take(66_000)
        .find(|&i| le32(buf, i) == 0x0605_4b50)
        .ok_or_else(|| bad(format!("{}: not a zip archive", path.display())))?;

    let n_entries = le16(buf, eocd + 10);
    let cd_off = le32(buf, eocd + 16);
    if n_entries == 0xFFFF || cd_off == 0xFFFF_FFFF {
        return Err(bad(format!(
            "{}: zip64 archive, unsupported (>4 GiB or >65535 members)",
            path.display()
        )));
    }

    let mut out = Vec::with_capacity(n_entries);
    let mut p = cd_off;
    for _ in 0..n_entries {
        if le32(buf, p) != 0x0201_4b50 {
            return Err(bad(format!("{}: corrupt central directory", path.display())));
        }
        let method = le16(buf, p + 10);
        let csize = le32(buf, p + 20);
        let nlen = le16(buf, p + 28);
        let elen = le16(buf, p + 30);
        let clen = le16(buf, p + 32);
        let lho = le32(buf, p + 42);
        let name = String::from_utf8_lossy(&buf[p + 46..p + 46 + nlen]).into_owned();
        if method != 0 {
            return Err(bad(format!(
                "{}: member '{}' is compressed (method {}); only ZIP_STORED \
                 (np.savez, not np.savez_compressed) is supported",
                path.display(),
                name,
                method
            )));
        }
        // The local header repeats the name and carries its own extra field,
        // which is generally a different length from the central one.
        let data_off = lho + 30 + le16(buf, lho + 26) + le16(buf, lho + 28);
        out.push((name, &buf[data_off..data_off + csize]));
        p += 46 + nlen + elen + clen;
    }
    Ok(out)
}

/// Parse one in-memory `.npy` (v1 or v2 header, C-order, little-endian).
fn parse_npy(bytes: &[u8], name: &str) -> NpyArray {
    assert!(
        bytes.len() > 10 && bytes[0] == 0x93 && &bytes[1..6] == b"NUMPY",
        "member '{}' is not a .npy file",
        name
    );
    let (hlen, hstart) = if bytes[6] == 1 {
        (le16(bytes, 8), 10)
    } else {
        (le32(bytes, 8), 12)
    };
    let header = String::from_utf8_lossy(&bytes[hstart..hstart + hlen]);

    // Fortran order would be read scrambled by the C-order slicing below, so
    // refuse it loudly rather than transpose silently (see nn::npy).
    assert!(
        !header.contains("'fortran_order': True"),
        "member '{}': Fortran-order .npy not supported: {}",
        name,
        header.trim()
    );

    let shape = parse_shape(&header);
    let raw = &bytes[hstart + hlen..];
    let data = if header.contains("'<f8'") {
        NpyData::F64(chunks(raw, 8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    } else if header.contains("'<f4'") {
        NpyData::F32(chunks(raw, 4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    } else if header.contains("'<i8'") {
        NpyData::I64(chunks(raw, 8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    } else if header.contains("'<u2'") || header.contains("'|u2'") {
        NpyData::U16(chunks(raw, 2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect())
    } else {
        panic!("member '{}': unsupported dtype in header: {}", name, header.trim())
    };

    let n: usize = shape.iter().product();
    let got = match &data {
        NpyData::F64(v) => v.len(),
        NpyData::F32(v) => v.len(),
        NpyData::I64(v) => v.len(),
        NpyData::U16(v) => v.len(),
    };
    assert_eq!(got, n, "member '{}': shape {:?} vs {} elements", name, shape, got);
    NpyArray { shape, data }
}

fn chunks<'a>(raw: &'a [u8], w: usize) -> impl Iterator<Item = &'a [u8]> + 'a {
    raw.chunks_exact(w)
}

/// Parse the shape tuple out of an `.npy` header dict. A 0-d array's `()` is
/// an empty shape, whose product is 1 — one scalar element.
fn parse_shape(header: &str) -> Vec<usize> {
    let start = header.find("'shape':").expect("no shape in .npy header") + 8;
    let open = header[start..].find('(').expect("no shape tuple") + start;
    let close = header[open..].find(')').expect("unterminated shape tuple") + open;
    header[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().expect("bad shape dimension"))
        .collect()
}
