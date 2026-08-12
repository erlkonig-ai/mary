//! What a converted Inkling pile actually holds, by name.
//!
//! Written because "the dense import took everything without `.experts.`, so
//! the MTP heads must be in there" is an inference about a filter, not an
//! observation of a pile — and a filter is exactly the kind of thing that is
//! right about the case you pictured and wrong about the one you did not.
//!
//! Reports the dense/expert split the pile itself records (`expert_index` is a
//! fact, not a substring of the name), then groups dense names by collapsing
//! digit runs, so 42 layers of the same tensor read as one line with a count
//! instead of 42 lines. An optional filter substring narrows it.
//!
//!   inkling_pile_names <pile> [branch] [--grep SUBSTR] [--raw]

use anyhow::{Context, Result};
use mary::models::inkling::pile::PileSource;
use std::collections::BTreeMap;

fn collapse(name: &str) -> String {
    // "layers.17.attn.wq" -> "layers.N.attn.wq": one line per shape of name.
    let mut out = String::with_capacity(name.len());
    let mut in_digits = false;
    for c in name.chars() {
        if c.is_ascii_digit() {
            if !in_digits {
                out.push('N');
                in_digits = true;
            }
        } else {
            out.push(c);
            in_digits = false;
        }
    }
    out
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let pile = args.next().context("usage: inkling_pile_names <pile> [branch] [--grep S] [--raw]")?;
    let mut branch = "inkling".to_string();
    let mut grep: Option<String> = None;
    let mut raw = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--grep" => grep = Some(args.next().context("--grep needs a substring")?),
            "--raw" => raw = true,
            other => branch = other.to_string(),
        }
    }

    let src = PileSource::open(std::path::Path::new(&pile), &branch)
        .with_context(|| format!("opening {pile} on branch {branch}"))?;

    println!("pile     {pile}");
    println!("branch   {branch}");
    println!("leaves   {} total = {} dense + {} expert",
             src.len(), src.dense_len(), src.expert_len());

    let names = src.names();
    let kept: Vec<&String> = match &grep {
        Some(s) => names.iter().filter(|n| n.contains(s.as_str())).collect(),
        None => names.iter().collect(),
    };
    if let Some(s) = &grep {
        println!("matching {:?}: {} of {}", s, kept.len(), names.len());
    }

    if raw {
        let mut v: Vec<&&String> = kept.iter().collect();
        v.sort();
        for n in v {
            println!("  {n}");
        }
        return Ok(());
    }

    let mut groups: BTreeMap<String, usize> = BTreeMap::new();
    for n in &kept {
        *groups.entry(collapse(n)).or_default() += 1;
    }
    println!("{} distinct name shapes:", groups.len());
    for (shape, count) in &groups {
        println!("  {count:5}  {shape}");
    }
    Ok(())
}
