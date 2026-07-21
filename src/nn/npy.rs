//! Minimal .npy (NumPy) file I/O for f32 tensors (plus an int64 reader for
//! token goldens).
//!
//! Supports reading and writing .npy v1.0 format with little-endian data.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// Save a flat f32 slice as a .npy file with the given shape.
pub fn save_npy(path: &Path, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;

    // Build header dict
    let shape_str: String = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let header_dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );

    // Pad header to align to 64 bytes (magic=6 + version=2 + header_len=2 + header)
    let preamble_len = 10; // 6 (magic) + 2 (version) + 2 (header_len)
    let total_unpadded = preamble_len + header_dict.len() + 1; // +1 for newline
    let padding = (64 - (total_unpadded % 64)) % 64;
    let header_len = header_dict.len() + padding + 1; // +1 for trailing newline

    // Write magic
    file.write_all(&[0x93])?;
    file.write_all(b"NUMPY")?;

    // Version 1.0
    file.write_all(&[1, 0])?;

    // Header length (little-endian u16)
    file.write_all(&(header_len as u16).to_le_bytes())?;

    // Header dict + padding + newline
    file.write_all(header_dict.as_bytes())?;
    for _ in 0..padding {
        file.write_all(b" ")?;
    }
    file.write_all(b"\n")?;

    // Raw float32 data (little-endian)
    let byte_data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    file.write_all(&byte_data)?;

    Ok(())
}

/// Load a .npy file, returning (data, shape).
/// Only supports float32 little-endian, C-order arrays.
pub fn load_npy(path: &Path) -> std::io::Result<(Vec<f32>, Vec<usize>)> {
    let mut file = fs::File::open(path)?;

    // Read magic (6 bytes)
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    assert!(
        magic[0] == 0x93 && &magic[1..6] == b"NUMPY",
        "Not a valid .npy file"
    );

    // Read version
    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;

    // Read header length
    let header_len = if version[0] == 1 {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf)?;
        u16::from_le_bytes(buf) as usize
    } else {
        // Version 2.0 uses 4-byte header length
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        u32::from_le_bytes(buf) as usize
    };

    // Read header
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header = String::from_utf8_lossy(&header_bytes);

    // Parse shape from header
    let shape = parse_shape(&header);

    // Verify dtype is float32
    assert!(
        header.contains("<f4") || header.contains("float32"),
        "Only float32 .npy files are supported, got: {}",
        header.trim()
    );

    // Fail LOUD on Fortran-order files: this loader reads the raw data as
    // C-order, so silently accepting `fortran_order: True` would scramble
    // every row (np.save writes it for F-contiguous arrays, e.g. `.T` views
    // or `.astype()` of transposed tensors — re-save with
    // `np.ascontiguousarray`). This exact silent scramble turned a v2
    // ref-code kit into deterministic babble on 2026-07-03.
    assert!(
        !header.contains("'fortran_order': True"),
        "Fortran-order .npy not supported (data would be read scrambled); \
         re-save with np.ascontiguousarray: {}",
        header.trim()
    );

    // Read remaining data
    let mut raw_data = Vec::new();
    file.read_to_end(&mut raw_data)?;

    // Convert bytes to f32
    let num_floats = raw_data.len() / 4;
    let data: Vec<f32> = (0..num_floats)
        .map(|i| {
            let bytes = [
                raw_data[i * 4],
                raw_data[i * 4 + 1],
                raw_data[i * 4 + 2],
                raw_data[i * 4 + 3],
            ];
            f32::from_le_bytes(bytes)
        })
        .collect();

    Ok((data, shape))
}

/// Load an int64 (`<i8`) .npy file, returning (data, shape) — token streams
/// (numpy default int dtype) come off capture scripts in this format.
pub fn load_npy_i64(path: &Path) -> std::io::Result<(Vec<i64>, Vec<usize>)> {
    let mut file = fs::File::open(path)?;

    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    assert!(
        magic[0] == 0x93 && &magic[1..6] == b"NUMPY",
        "Not a valid .npy file"
    );

    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;
    let header_len = if version[0] == 1 {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf)?;
        u16::from_le_bytes(buf) as usize
    } else {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        u32::from_le_bytes(buf) as usize
    };

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header = String::from_utf8_lossy(&header_bytes);
    let shape = parse_shape(&header);

    assert!(
        header.contains("<i8") || header.contains("int64"),
        "Only int64 .npy files are supported by load_npy_i64, got: {}",
        header.trim()
    );
    assert!(
        !header.contains("'fortran_order': True"),
        "Fortran-order .npy not supported (data would be read scrambled): {}",
        header.trim()
    );

    let mut raw_data = Vec::new();
    file.read_to_end(&mut raw_data)?;
    let n = raw_data.len() / 8;
    let data: Vec<i64> = (0..n)
        .map(|i| i64::from_le_bytes(raw_data[i * 8..i * 8 + 8].try_into().unwrap()))
        .collect();

    Ok((data, shape))
}

/// Parse the shape tuple from the .npy header string.
fn parse_shape(header: &str) -> Vec<usize> {
    // Find 'shape': (...) in the header
    let shape_start = header.find("'shape':").expect("No shape in header") + 8;
    let paren_start = header[shape_start..]
        .find('(')
        .expect("No opening paren for shape")
        + shape_start;
    let paren_end = header[paren_start..]
        .find(')')
        .expect("No closing paren for shape")
        + paren_start;

    let shape_str = &header[paren_start + 1..paren_end];
    shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().expect("Invalid shape dimension"))
        .collect()
}
