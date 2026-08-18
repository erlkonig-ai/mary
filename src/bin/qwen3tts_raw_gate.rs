//! Gate for the zero-copy talker lane (the `<stem>_folded_f16.pile`
//! sibling): weights and forward identity across the two ways the raw f16
//! talker comes to life.
//!
//! (The FUSED talker lane was removed 2026-07-12 — raw is the ONLY talker
//! lane; an ear A/B picked raw ("better tempo") at measured perf parity.
//! The gate's fused arm went with it; the fused lane's final green
//! comparison — weights 173/173 bit-identical across
//! fused/raw-fold/folded-alias — is on record at fa03334, the pre-removal
//! HEAD (the run itself: PORT_NOTES.md, "the zero-copy RAW talker lane",
//! merged at 5d93a95).)
//!
//! Each lane runs in its OWN process and a compare pass reads the dumps.
//! One-process-per-lane is kept deliberately, because it isolates the two load
//! paths. It also happens to observe that the raw backend reproduced itself
//! across processes on this graph — record that as a DIAGNOSTIC, never as a
//! requirement. Run-to-run determinism was killed as a gate on 2026-08-18
//! along with bit-equality against a previous lane
//! (wiki:f5dcc88988bb28e472e50fa030332adb): concurrency is nondeterministic by
//! definition, so demanding it of a parallel lane is incoherent, and demanding
//! it here would forbid ever retiling or reassociating the talker's GEMMs.
//!
//!   qwen3tts_raw_gate --lane raw|folded --out <dir> [<pile>] \
//!       [--prefill 154] [--steps 32]
//!   qwen3tts_raw_gate --compare <dir>
//!
//! Lanes: `raw` = BHalf through the ordinary loader (leaf-alias + fold at
//! load — the surviving fallback for piles without the folded sibling);
//! `folded` = BHalf aliased zero-copy from the folded sibling pile (the
//! production path).
//!
//! Per lane the dump holds: sha256 of every fold-transformed GPU tensor's
//! f16 bytes (`<lane>_weights.txt`), the read-back hidden state of a
//! synthetic prefill + teacher-forced decode steps (`<lane>_hidden.npy`),
//! and each step's codebook-0 argmax (`<lane>_argmax.txt`).
//!
//! Compare gates. Both are LOAD-PATH gates, and that is exactly why bit
//! equality is the right bar for them: neither lane computes anything the
//! other does not, in a different order. Same weights, same kernels, same
//! accumulation — one of them just arrives by an mmap alias instead of a
//! transform at load. Nothing here licenses bit-equality for a change that
//! moves arithmetic.
//!
//!   1. weights — both lanes BIT-IDENTICAL per tensor (the sibling holds
//!      exactly the weights the fold-at-load path computes).
//!   2. raw ≡ folded hidden BIT-EXACT per step, argmax streams identical
//!      (the zero-copy load may not change the voice relative to the
//!      ordinary load).
//!
//! The E2E render receipt (`speak_check` on the seeded fixture,
//! `MARY_SPEAK_SEED=7`, run twice by hand) was banked as an "E2E determinism
//! gate" — see PORT_NOTES.md, "the zero-copy RAW talker lane". It never was a
//! gate in code and as of 2026-08-18 it must not become one: treat a differing
//! wav as a TRIPWIRE, an unclassified event worth looking at (it is how a
//! corruption bug would announce itself), never a build failure. The voice's
//! real bar is capability — the ear A/B that picked this lane in the first
//! place.

#[cfg(target_os = "macos")]
mod imp {

    use burn::prelude::*;
    use mary::models::qwen3tts::talker::Talker;
    use mary::nn::backend::BHalf;
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};

    const LANES: [&str; 2] = ["raw", "folded"];

    fn arg_usize(name: &str, default: usize) -> usize {
        let args: Vec<String> = std::env::args().collect();
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn arg_str(name: &str) -> Option<String> {
        let args: Vec<String> = std::env::args().collect();
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    }

    /// Deterministic pseudo-random codec ids in [0, 2048) (the megakernel
    /// probe's generator).
    fn synth_ids(n: usize, mut seed: u64) -> Vec<u32> {
        (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) % 2048) as u32
            })
            .collect()
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }

    /// Prefill + teacher-forced steps; hidden per step + codebook-0 argmax.
    fn run_steps<B: Backend>(
        talker: &Talker<B>,
        prefill_ids: &[u32],
        step_ids: &[u32],
        dev: &B::Device,
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        // The prefill entry mimics production's op shape (codec-side embeds +
        // text-side embeds, `pipeline::build_prefill`'s sum). Historical: this
        // shape was originally forced by a burn-fusion 0.21 codegen bug on the
        // now-removed fused lane (FusedReduceLaunch strides OOB on any bare
        // entry); kept because mirroring production's entry is the honest
        // fixture anyway.
        let text_ids: Vec<u32> = prefill_ids.iter().map(|&i| (i * 97) % 50000).collect();
        let embeds = talker.embed_codec(prefill_ids, dev) + talker.embed_text(&text_ids, dev);
        let mut caches = talker.new_caches();
        let h = talker.forward(embeds, &mut caches, dev);
        let mut hs = vec![talker.last_hidden(h)];
        for &id in step_ids {
            let row = talker.codec_row(id).to_vec();
            let e = Tensor::<B, 1>::from_floats(row.as_slice(), dev).reshape([1, 1, talker.hidden]);
            let h = talker.forward(e, &mut caches, dev);
            hs.push(talker.last_hidden(h));
        }
        let am = hs.iter().map(|h| argmax(&talker.logits_from(h))).collect();
        (hs, am)
    }

    fn dump_weights<B: Backend>(talker: &Talker<B>, lane: &str, out: &Path) -> anyhow::Result<()> {
        let weights = mary::persist::qwen3tts_folded_readback(talker);
        let mut wtxt = String::new();
        for (name, bits, dims) in &weights {
            let mut hasher = Sha256::new();
            for h in bits {
                hasher.update(h.to_le_bytes());
            }
            wtxt.push_str(&format!(
                "{name} {:x} {} {dims:?}\n",
                hasher.finalize(),
                bits.len()
            ));
        }
        std::fs::write(out.join(format!("{lane}_weights.txt")), wtxt)?;
        eprintln!("[{lane}] dumped {} weight hashes → {out:?}", weights.len());
        Ok(())
    }

    fn dump_forward<B: Backend>(
        talker: &Talker<B>,
        lane: &str,
        out: &Path,
        prefill: usize,
        steps: usize,
        dev: &B::Device,
    ) -> anyhow::Result<()> {
        let (hs, am) = run_steps(talker, &synth_ids(prefill, 7), &synth_ids(steps, 99), dev);
        let flat: Vec<f32> = hs.iter().flatten().copied().collect();
        mary::nn::npy::save_npy(
            &out.join(format!("{lane}_hidden.npy")),
            &flat,
            &[hs.len(), talker.hidden],
        )?;
        let amtxt: String = am.iter().map(|a| format!("{a}\n")).collect();
        std::fs::write(out.join(format!("{lane}_argmax.txt")), amtxt)?;
        eprintln!("[{lane}] dumped {} forward steps → {out:?}", hs.len());
        Ok(())
    }

    fn dump<B: Backend>(
        talker: &Talker<B>,
        lane: &str,
        out: &Path,
        prefill: usize,
        steps: usize,
        dev: &B::Device,
    ) -> anyhow::Result<()> {
        dump_weights(talker, lane, out)?;
        dump_forward(talker, lane, out, prefill, steps, dev)
    }

    fn compare(dir: &Path) -> anyhow::Result<()> {
        // ── gate 1: weights sha equality across lanes ──
        let read_w = |lane: &str| -> anyhow::Result<Vec<(String, String)>> {
            Ok(
                std::fs::read_to_string(dir.join(format!("{lane}_weights.txt")))?
                    .lines()
                    .map(|l| {
                        let mut it = l.split_whitespace();
                        (
                            it.next().unwrap_or("").to_string(),
                            it.next().unwrap_or("").to_string(),
                        )
                    })
                    .collect(),
            )
        };
        let (wr, wo) = (read_w("raw")?, read_w("folded")?);
        anyhow::ensure!(wr.len() == wo.len(), "weight dump lengths differ");
        let mut w_ok = true;
        for (r, o) in wr.iter().zip(&wo) {
            anyhow::ensure!(r.0 == o.0, "name order mismatch: {} {}", r.0, o.0);
            if r.1 != o.1 {
                w_ok = false;
                println!("  ✗ {}: raw-fold ≠ folded-alias", r.0);
            }
        }
        println!(
            "gate 1 weights ({} tensors): raw-fold / folded-alias {}",
            wr.len(),
            if w_ok {
                "BIT-IDENTICAL ✓"
            } else {
                "DIVERGED ✗"
            }
        );

        // ── gate 2: hidden bit-exact per step + argmax streams identical ──
        let read_h = |lane: &str| -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
            Ok(mary::nn::npy::load_npy(
                &dir.join(format!("{lane}_hidden.npy")),
            )?)
        };
        let (hr, sr) = read_h("raw")?;
        let (ho, so) = read_h("folded")?;
        anyhow::ensure!(sr == so, "hidden dump shapes differ: {sr:?} {so:?}");
        let (steps, hidden) = (sr[0], sr[1]);
        let mut exact_ok = true;
        for s in 0..steps {
            let (r, o) = (&hr[s * hidden..][..hidden], &ho[s * hidden..][..hidden]);
            let mism = r
                .iter()
                .zip(o)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            if mism > 0 {
                exact_ok = false;
                println!("  ✗ step {s}: raw-fold vs folded-alias hidden differs in {mism} floats");
            }
        }
        let read_a = |lane: &str| -> anyhow::Result<Vec<usize>> {
            Ok(
                std::fs::read_to_string(dir.join(format!("{lane}_argmax.txt")))?
                    .lines()
                    .filter_map(|l| l.parse().ok())
                    .collect(),
            )
        };
        let (ar, ao) = (read_a("raw")?, read_a("folded")?);
        let am_ok = ar.len() == ao.len() && ar == ao;
        println!(
            "gate 2 raw-fold ≡ folded-alias ({steps} steps, two processes): hidden {} | argmax {}",
            if exact_ok {
                "BIT-EXACT ✓"
            } else {
                "DIVERGED ✗"
            },
            if am_ok {
                format!("{}/{} identical ✓", ar.len(), ao.len())
            } else {
                "DIVERGED ✗".into()
            }
        );

        anyhow::ensure!(w_ok, "weights gate failed");
        anyhow::ensure!(
            exact_ok && am_ok,
            "zero-copy load changed the raw lane's forward — investigate"
        );
        println!("qwen3tts_raw_gate PASSED");
        Ok(())
    }

    pub fn run() -> anyhow::Result<()> {
        mary::models::qwen3tts::cpu::set_interactive_qos();
        if let Some(dir) = arg_str("--compare") {
            return compare(Path::new(&dir));
        }
        let lane = arg_str("--lane").ok_or_else(|| {
            anyhow::anyhow!(
                "usage: qwen3tts_raw_gate --lane raw|folded --out <dir> | --compare <dir>"
            )
        })?;
        anyhow::ensure!(LANES.contains(&lane.as_str()), "unknown lane {lane}");
        let out = PathBuf::from(
            arg_str("--out").ok_or_else(|| anyhow::anyhow!("--out <dir> required with --lane"))?,
        );
        std::fs::create_dir_all(&out)?;
        let pile =
            mary::paths::model(std::env::var("QWEN3TTS_PILE").ok().as_deref(), "qwen3tts.pile")?;
        let pile = pile.as_path();
        let (prefill, steps) = (arg_usize("--prefill", 154), arg_usize("--steps", 32));

        match lane.as_str() {
            "raw" => {
                let loader = mary::persist::load_aliased_loader_from_pile(pile, "talker_f16")?;
                let dev = Default::default();
                let talker = Talker::<BHalf>::load(&loader, &dev);
                drop(loader);
                dump(&talker, "raw", &out, prefill, steps, &dev)?;
            }
            "folded" => {
                let folded = mary::persist::qwen3tts_folded_sibling_path(pile);
                anyhow::ensure!(
                folded.exists(),
                "folded sibling {folded:?} missing — derive it first: qwen3tts_persist --fold-derive {pile:?}"
            );
                let talker = mary::persist::load_qwen3tts_talker_folded(pile, &folded)?;
                let dev = Default::default();
                dump(&talker, "folded", &out, prefill, steps, &dev)?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    imp::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen3tts_raw_gate: macOS-only lane (folded-sibling zero-copy talker).");
    std::process::exit(2);
}
