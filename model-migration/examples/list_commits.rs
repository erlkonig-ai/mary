//! READ-ONLY: walk a legacy pile's `main` commit DAG and report, for each
//! commit, the size of its OWN content and of its full ancestor closure.
//!
//! The PersonaPlex bundle adoption selects one exact legacy commit whose
//! ancestor closure must have an exact fact count, so the history has to be
//! enumerated rather than guessed at.

use anyhow::{anyhow, Context, Result};
use ed25519_dalek::SigningKey;
use std::collections::{BTreeSet, HashMap};
use triblespace::core::repo::{Repository, BlobStore, BlobStoreGet};
use triblespace::core::repo::{parent, content};
use triblespace::macros::{find, pattern};
use triblespace::prelude::*;

type Commit = Inline<inlineencodings::Handle<blobencodings::SimpleArchive>>;

fn parents_and_content(
    reader: &impl BlobStoreGet,
    commit: Commit,
) -> Result<(Vec<Commit>, Option<Commit>)> {
    let meta: TribleSet = reader
        .get(commit)
        .map_err(|e| anyhow!("read commit: {e:?}"))?;
    let contents: Vec<Commit> = find!((c: Inline<_>), pattern!(&meta, [{ content: ?c }]))
        .map(|(c,)| c)
        .collect();
    let parents: Vec<Commit> = find!((p: Inline<_>), pattern!(&meta, [{ parent: ?p }]))
        .map(|(p,)| p)
        .collect();
    Ok((parents, contents.first().copied()))
}

fn closure(reader: &impl BlobStoreGet, root: Commit) -> Result<(usize, usize)> {
    let mut seen = BTreeSet::new();
    let mut pending = vec![root];
    let mut facts = TribleSet::new();
    let mut commits = 0usize;
    while let Some(c) = pending.pop() {
        if !seen.insert(c) {
            continue;
        }
        commits += 1;
        let (parents, content_handle) = parents_and_content(reader, c)?;
        if let Some(h) = content_handle {
            let contribution: TribleSet =
                reader.get(h).map_err(|e| anyhow!("read content: {e:?}"))?;
            facts += contribution;
        }
        pending.extend(parents);
    }
    Ok((facts.len(), commits))
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: list_commits <pile>")?;
    let path = std::path::Path::new(&path);
    let mut pile = Pile::open(path).map_err(|e| anyhow!("open: {e:?}"))?;
    pile.refresh().map_err(|e| anyhow!("refresh: {e:?}"))?;
    let key = SigningKey::from_bytes(&[0x11u8; 32]);
    let mut repo = Repository::new(&mut pile, key, TribleSet::new())
        .map_err(|e| anyhow!("repo: {e:?}"))?;
    let branch = repo
        .lookup_branch("main")
        .map_err(|e| anyhow!("lookup: {e:?}"))?
        .ok_or_else(|| anyhow!("no main"))?;
    let ws = repo.pull(branch).map_err(|e| anyhow!("pull: {e:?}"))?;
    let head = ws.head().ok_or_else(|| anyhow!("no head"))?;
    let reader = repo.storage_mut().reader().context("reader")?;

    // Enumerate the DAG first, then size each node's closure.
    let mut order: Vec<Commit> = Vec::new();
    let mut seen = BTreeSet::new();
    let mut pending = vec![head];
    let mut parent_map: HashMap<Commit, Vec<Commit>> = HashMap::new();
    while let Some(c) = pending.pop() {
        if !seen.insert(c) {
            continue;
        }
        order.push(c);
        let (parents, _) = parents_and_content(&reader, c)?;
        parent_map.insert(c, parents.clone());
        pending.extend(parents);
    }
    println!("pile {}", path.display());
    println!("main head {head:?}");
    println!("{} commits reachable\n", order.len());
    println!("  {:<66} {:>10} {:>8} {:>8}", "COMMIT", "CLOSURE", "COMMITS", "PARENTS");
    for c in &order {
        let (facts, commits) = closure(&reader, *c)?;
        let hash = inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(*c);
        println!(
            "  {:<66} {facts:>10} {commits:>8} {:>8}",
            format!("{hash:?}"),
            parent_map.get(c).map(|p| p.len()).unwrap_or(0)
        );
    }
    Ok(())
}
