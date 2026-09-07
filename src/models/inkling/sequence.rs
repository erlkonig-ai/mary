//! Independent sequence ownership on one resident set of weights.

use super::*;
use super::super::{budget, config::InklingTextConfig, devplan::ExpertTable, kvpages};

/// A conversation's mutable state, bound to the Session that admitted it.
///
/// Not Clone: copying a capsule would create another unpriced KV lifetime.
/// Reset and reuse admitted capsules; dropping one releases its state but does
/// not mint another admission. Checkpoints follow the sequence across swaps.
pub struct Sequence {
    owner: u64,
    caches: Vec<LayerCache>,
    pos: usize,
    last: Option<usize>,
    torn: bool,
    seq: u64,
    context_budget: usize,
    audio: Option<SlotQueue>,
    vision: Option<SlotQueue>,
    #[cfg(feature = "inkling-cuda")]
    anchor: Vec<super::super::learn::AnchorRow>,
    teacher: Option<(usize, ExpertTable)>,
    teacher_version: Option<u64>,
}

impl Sequence {
    pub fn position(&self) -> usize { self.pos }
    pub fn last_token(&self) -> Option<usize> { self.last }
    pub fn identity(&self) -> u64 { self.seq }
    pub fn is_poisoned(&self) -> bool { self.torn }
    pub fn context_budget(&self) -> usize { self.context_budget }

    /// Discard only this conversation, including media slot identities and
    /// learning anchors. Warm weights and every other capsule are untouched.
    pub fn reset(&mut self) {
        self.caches.clear();
        self.pos = 0;
        self.last = None;
        self.torn = false;
        self.seq = next_seq();
        for queue in [&mut self.audio, &mut self.vision].into_iter().flatten() {
            queue.pending.clear();
            queue.slot = None;
        }
        #[cfg(feature = "inkling-cuda")]
        self.anchor.clear();
        self.teacher = None;
        self.teacher_version = None;
    }

    fn bank(&self) -> Option<(usize, u64)> {
        self.teacher.as_ref().map(|(layer, _)| {
            (*layer, self.teacher_version.expect("teacher tables have a version"))
        })
    }
}

/// Persistent KV, page slack and all four convolution histories per extra
/// sequence. Activation width is separately bounded by prefill_budget. Anchors
/// are charged at their bound, even if a particular background never keeps any.
pub(super) fn extra_sequence_bytes(
    t: &InklingTextConfig,
    layers: std::ops::Range<usize>,
    capacities: &[usize],
    policy: budget::AdmissionPolicy,
    tp: Option<Tp>,
) -> Result<u64> {
    let mut bytes = 0u128;
    if !capacities.is_empty() {
        bytes += SCORE_ROWS as u128 * t.effective_vocab() as u128 * 8;
    }
    for &capacity in capacities {
        anyhow::ensure!(capacity > 0 && capacity <= u32::MAX as usize,
            "an independent context capacity must be in 1..=u32::MAX, got {capacity}");
        bytes += budget::kv_cache_bytes(t, layers.clone(), capacity, policy) as u128;
        for layer in layers.clone() {
            let (_, heads, dim) = t.heads(t.attn_kind(layer));
            let heads = match tp {
                Some(tp) => tp.share("kv_heads", heads).map_err(|e| anyhow::anyhow!(e))?,
                None => heads,
            };
            let kv_width = heads as u128 * dim as u128;
            // The paged stores can retain a cut first page and a partly filled
            // last page. Price both rather than only the logical retained rows.
            bytes += 2 * 2 * kvpages::PAGE as u128 * kv_width * 9 / 16;
            // K/V pre-convolutions plus attention/MLP output convolutions.
            bytes += 2 * t.sconv_kernel_size.saturating_sub(1) as u128
                * (kv_width + t.hidden_size as u128) * 4;
        }
        #[cfg(feature = "inkling-cuda")]
        {
            bytes += super::super::learn::ANCHOR_ROWS as u128 * 4
                * (t.effective_vocab() as u128
                    + (2 + t.sconv_kernel_size.saturating_sub(1)) as u128 * t.hidden_size as u128);
        }
    }
    u64::try_from(bytes).context("independent sequence reservation overflow")
}

impl Session {
    pub fn next_token(&self) -> Option<usize> { self.last }

    fn empty_sequence(&self, context_budget: usize) -> Sequence {
        Sequence {
            owner: self.sequence_owner,
            caches: Vec::new(), pos: 0, last: None, torn: false, seq: next_seq(),
            context_budget,
            audio: self.audio.as_ref().map(|a| SlotQueue::new(a.queue.what, a.queue.unit)),
            vision: self.vision.as_ref().map(|v| SlotQueue::new(v.queue.what, v.queue.unit)),
            #[cfg(feature = "inkling-cuda")]
            anchor: Vec::new(),
            teacher: None,
            teacher_version: None,
        }
    }

    /// Take the next explicitly admitted extra sequence. Does not allocate KV
    /// until its first prefill, and never reloads model or frontend weights.
    pub fn new_sequence(&mut self) -> Result<Sequence> {
        let capacity = self.sequence_context_budgets.pop_front()
            .context("no unused sequence admission; set sequence_context_budgets before load")?;
        Ok(self.empty_sequence(capacity))
    }

    /// Exchange the active conversation and an admitted capsule. Every mutable
    /// temporal field travels together, including teacher selection and media.
    pub fn swap_sequence(&mut self, sequence: &mut Sequence) -> Result<()> {
        anyhow::ensure!(sequence.owner == self.sequence_owner,
            "sequence belongs to a different resident Session");
        use std::mem::swap;
        swap(&mut self.caches, &mut sequence.caches);
        swap(&mut self.pos, &mut sequence.pos);
        swap(&mut self.last, &mut sequence.last);
        swap(&mut self.torn, &mut sequence.torn);
        swap(&mut self.seq, &mut sequence.seq);
        swap(&mut self.context_budget, &mut sequence.context_budget);
        swap(&mut self.moe.teacher, &mut sequence.teacher);
        swap(&mut self.sequence_teacher_version, &mut sequence.teacher_version);
        if let (Some(input), Some(queue)) = (&mut self.audio, &mut sequence.audio) {
            swap(&mut input.queue, queue);
        }
        if let (Some(input), Some(queue)) = (&mut self.vision, &mut sequence.vision) {
            swap(&mut input.queue, queue);
        }
        #[cfg(feature = "inkling-cuda")]
        {
            swap(&mut self.anchor, &mut sequence.anchor);
            // A LearnKeep belongs to a single forward, never to another state.
            self.moe.learn = None;
        }
        Ok(())
    }

    /// Run existing Session operations on a capsule and restore the foreground
    /// on either success or an ordinary error. A failed pass poisons its own
    /// capsule, not the foreground being held aside.
    pub fn with_sequence<R>(
        &mut self, sequence: &mut Sequence, f: impl FnOnce(&mut Self) -> Result<R>,
    ) -> Result<R> {
        self.swap_sequence(sequence)?;
        let result = f(self);
        self.swap_sequence(sequence)?;
        result
    }

    /// A refreshed teacher cannot inherit old K/V or convolution state. Select
    /// its version only on an empty sequence, then prefill that version's prompt.
    pub fn set_teacher_bank(&mut self, layer: usize, version: u64, table: ExpertTable) -> Result<()> {
        anyhow::ensure!(self.pos == 0 && self.caches.is_empty() && !self.torn,
            "select a teacher bank only after resetting its independent sequence");
        anyhow::ensure!((self.lo..self.hi).contains(&layer) && !self.cfg.text_config.is_dense(layer),
            "teacher layer {layer} is not a resident routed layer");
        self.moe.teacher = Some((layer, table));
        self.sequence_teacher_version = Some(version);
        Ok(())
    }

    /// Record the caller's chosen token after an explicit logits pass. This is
    /// metadata only: the token enters KV when step/extend/batch actually runs.
    pub fn set_next_token(&mut self, token: usize) -> Result<()> {
        anyhow::ensure!(self.pos > 0 && !self.torn && token < self.cfg.text_config.vocab_size,
            "cannot select token {token} for an empty, poisoned, or incompatible sequence");
        self.last = Some(token);
        Ok(())
    }

    pub fn set_sequence_next_token(&self, sequence: &mut Sequence, token: usize) -> Result<()> {
        anyhow::ensure!(sequence.owner == self.sequence_owner && sequence.pos > 0
            && !sequence.torn && token < self.cfg.text_config.vocab_size,
            "cannot select token {token} for this sequence");
        sequence.last = Some(token);
        Ok(())
    }

    /// Append sequential known tokens and return the final next-token logits.
    /// Does not score/train, keep anchors, sample, or read logits back to host.
    pub fn extend_logits(&mut self, ids: &[usize]) -> Result<T2> {
        anyhow::ensure!(!ids.is_empty(), "a logits pass needs at least one input token");
        #[cfg(feature = "inkling-cuda")]
        let learn_layer = self.moe.learn_layer.take();
        #[cfg(feature = "inkling-cuda")]
        { self.moe.learn = None; }
        let result = (|| {
            let mut logits = None;
            for chunk in ids.chunks(self.extend_batch) {
                logits = Some(self.observe_forward(chunk, false)?.0);
            }
            self.senses_drained()?;
            Ok(logits.expect("nonempty input"))
        })();
        #[cfg(feature = "inkling-cuda")]
        { self.moe.learn_layer = learn_layer; self.moe.learn = None; }
        result
    }

    /// One nonupdating sequential pass with every prediction row exposed.
    /// Bounded by the head cap as well as the admitted activation width.
    pub fn observe_logits(&mut self, ids: &[usize]) -> Result<T2> {
        anyhow::ensure!(ids.len() <= SCORE_ROWS, "observed head exceeds {SCORE_ROWS} rows");
        #[cfg(feature = "inkling-cuda")]
        let previous = self.moe.learn_layer.take();
        #[cfg(feature = "inkling-cuda")]
        { self.moe.learn = None; }
        let result = self.observe_forward(ids, true).and_then(|(logits, _)| {
            self.senses_drained()?;
            Ok(logits)
        });
        #[cfg(feature = "inkling-cuda")]
        { self.moe.learn_layer = previous; self.moe.learn = None; }
        result
    }

    fn observe_forward(&mut self, ids: &[usize], all: bool) -> Result<(T2, T2)> {
        anyhow::ensure!(!ids.is_empty() && ids.len() <= self.prefill_budget,
            "observed pass width exceeds admission or is empty");
        anyhow::ensure!(!self.torn, "the active sequence is poisoned");
        anyhow::ensure!(self.pos.checked_add(ids.len()).is_some_and(|end| end <= self.context_budget),
            "observed pass exceeds the active sequence's context admission");
        anyhow::ensure!(ids.iter().all(|&id| id < self.cfg.text_config.vocab_size),
            "observed input contains an out-of-vocabulary token");
        self.check_senses(ids)?;
        let mode = if all { PassMode::ObserveAll } else { PassMode::Observe };
        let result = match self.forward_pass(ids, mode) {
            Ok(PassOutput::Observed { logits, residual }) =>
                self.validate_cache_completeness().map(|_| (logits, residual)),
            Ok(_) => unreachable!("Observe returns its head and residual"),
            Err(error) => Err(error),
        };
        if result.is_err() { self.torn = true; }
        result
    }

    #[cfg(feature = "inkling-cuda")]
    pub fn learning_step(&self) -> Option<u64> {
        self.learner.as_ref().map(super::super::learn::Learner::step)
    }

    /// Consume the one EMA reservation after a student routed forward has bound
    /// its table. The returned bank owns teacher weights, not sequence state.
    #[cfg(feature = "inkling-cuda")]
    pub fn create_teacher_bank(&mut self, beta: f32) -> Result<super::super::learn::ema::EmaBank> {
        use super::super::learn::ema::EmaBank;
        anyhow::ensure!(self.teacher_bank_available, "no unused typed EMA bank admission");
        let step = self.learning_step().context("EMA requires an admitted learner")?;
        let t = &self.cfg.text_config;
        let inter = match self.group.as_ref().map(Group::tp) {
            Some(tp) => tp.share("intermediate_size", t.intermediate_size)
                .map_err(|e| anyhow::anyhow!(e))?,
            None => t.intermediate_size,
        };
        let layer = self.trainable_layer.context("EMA requires a configured trainable layer")?;
        let table = self.moe.route.as_ref().and_then(|route| route.tabs.get(&(layer, false)))
            .and_then(Option::as_ref).context("prefill a student before initializing its teacher bank")?;
        let bank = EmaBank::new(&self.client, table, t.hidden_size, inter,
            self.src.experts_swizzled(), beta, step)?;
        self.teacher_bank_available = false;
        Ok(bank)
    }

    /// Teacher capsules must be reset/re-prefilled after this boundary. Cloned
    /// tables alias this bank and do not represent immutable version snapshots.
    #[cfg(feature = "inkling-cuda")]
    pub fn advance_teacher_bank(
        &mut self, bank: &mut super::super::learn::ema::EmaBank, expected_version: u64,
    ) -> Result<super::super::learn::ema::EmaReport> {
        anyhow::ensure!(self.moe.teacher.is_none(), "cannot refresh a teacher with its sequence active");
        let step = self.learning_step().context("EMA requires an admitted learner")?;
        bank.advance(&self.client, expected_version, step)
    }

    /// One explicit sequential soft-target update. Row zero of dist belongs to
    /// residual row start; other rows carry no head target. Independent batch
    /// rows must NOT be passed here: the backward convolution is temporal.
    ///
    /// This appends ids under the pre-update weights, then enqueues one update.
    /// The caller owns resetting/re-prefilling a student's training capsule and
    /// advancing the teacher; no automatic SFT, frozen control or anchor runs.
    #[cfg(feature = "inkling-cuda")]
    pub fn learn_distribution(
        &mut self, ids: &[usize], dist: &T2, start: usize, weight: f32,
    ) -> Result<super::super::learn::LearnReport> {
        use super::super::learn;
        anyhow::ensure!(self.moe.teacher.is_none(), "cannot update through a teacher sequence");
        anyhow::ensure!(self.learner.is_some(), "explicit learning was not admitted at load");
        let layer = self.trainable_layer.context("explicit learning has no configured trainable layer")?;
        anyhow::ensure!(!ids.is_empty() && ids.len() <= self.prefill_budget,
            "soft-target input must fit one admitted sequential pass");
        learn::validate_soft_targets(dist, start, ids.len(), self.cfg.text_config.effective_vocab(), weight)?;
        let previous = self.moe.learn_layer.replace(layer);
        self.moe.learn = None;
        let result = (|| {
            let (_, residual) = self.observe_forward(ids, false)?;
            self.senses_drained()?;
            let keep = self.moe.learn.take().context("no routed forward kept for soft-target learning")?;
            let t = &self.cfg.text_config;
            let inter = match self.group.as_ref().map(Group::tp) {
                Some(tp) => tp.share("intermediate_size", t.intermediate_size)
                    .map_err(|e| anyhow::anyhow!(e))?,
                None => t.intermediate_size,
            };
            let route = self.moe.route.as_ref().context("soft-target layer has no route")?;
            let table = route.tabs.get(&(keep.layer, false)).and_then(Option::as_ref)
                .context("soft-target layer has no student expert table")?;
            let vocab = t.effective_vocab();
            let mup = t.logits_mup_width_multiplier as f32;
            let final_norm = &self.final_norm;
            let unembed = &self.unembed;
            let forbidden = &self.forbidden;
            let head = |x: T2| {
                let rows = x.dims()[0];
                let hs = dev_lane_resid::rms_norm(x, final_norm.clone(), t.rms_norm_eps).div_scalar(mup);
                let mut logits = dev_lane::linear_w(hs, unembed).slice([0..rows, 0..vocab]);
                for &id in forbidden.iter().filter(|&&id| id < vocab) {
                    let column = logits.clone().slice([0..rows, id..id + 1]);
                    logits = logits.slice_assign([0..rows, id..id + 1],
                        column.mul_scalar(0.0).sub_scalar(f32::INFINITY));
                }
                logits
            };
            learn::learn_last_layer(&self.client, &self.dev, self.learner.as_mut().expect("checked"),
                keep, route, table, self.src.experts_swizzled(), &residual, final_norm, &head,
                learn::Target::DistRange { dist, start, weight }, mup, t.rms_norm_eps,
                vocab, t.vocab_size, t.n_routed_experts, inter, SCORE_ROWS)
        })();
        self.moe.learn_layer = previous;
        self.moe.learn = None;
        result
    }

    /// One input token per independent, already-prefilled sequence. If present,
    /// `active` is the foreground token and its result comes first. Other results
    /// preserve capsule order. Inputs are NEVER consecutive time rows.
    ///
    /// Attention has separate positions and histories; dense/routed MLPs share
    /// a row batch. Mixed teacher/student banks are refused. No automatic SFT or
    /// anchor update runs here: temporal backward is not independent-row backward.
    pub fn batch_step(
        &mut self, active: Option<usize>, sequences: &mut [&mut Sequence], tokens: &[usize],
    ) -> Result<Vec<usize>> {
        let logits = self.batch_logits(active, sequences, tokens)?;
        let best = argmax_rows_dev(logits);
        let offset = usize::from(active.is_some());
        if active.is_some() { self.last = Some(best[0]); }
        for (sequence, &token) in sequences.iter_mut().zip(&best[offset..]) {
            sequence.last = Some(token);
        }
        Ok(best)
    }

    /// Same independent forward, returning one next-token logit row per input.
    /// No host readback and no implicit sample. Use explicit input tokens on the
    /// next batch, or record a chosen token with set_next_token before step().
    /// Width is bounded by both prefill admission and SCORE_ROWS (64).
    pub fn batch_logits(
        &mut self, active: Option<usize>, sequences: &mut [&mut Sequence], tokens: &[usize],
    ) -> Result<T2> {
        anyhow::ensure!(sequences.len() == tokens.len(), "one token is required per capsule");
        let mut foreground = self.empty_sequence(self.context_budget);
        if active.is_some() { self.swap_sequence(&mut foreground)?; }
        let result = {
            let mut rows: Vec<&mut Sequence> = Vec::with_capacity(sequences.len() + usize::from(active.is_some()));
            let mut ids = Vec::with_capacity(rows.capacity());
            if let Some(token) = active {
                rows.push(&mut foreground);
                ids.push(token);
            }
            rows.extend(sequences.iter_mut().map(|s| &mut **s));
            ids.extend_from_slice(tokens);
            self.batch_sequences(&mut rows, &ids)
        };
        if active.is_some() { self.swap_sequence(&mut foreground)?; }
        result
    }

    fn batch_sequences(&mut self, rows: &mut [&mut Sequence], ids: &[usize]) -> Result<T2> {
        let n = rows.len();
        anyhow::ensure!(n > 0 && n <= self.prefill_budget.min(SCORE_ROWS),
            "independent batch width {n} is outside 1..={}", self.prefill_budget.min(SCORE_ROWS));
        let bank = rows[0].bank();
        for (row, &id) in rows.iter().zip(ids) {
            anyhow::ensure!(row.owner == self.sequence_owner, "foreign sequence in batch");
            anyhow::ensure!(!row.torn, "sequence {} is poisoned", row.seq);
            anyhow::ensure!(row.pos > 0 && row.caches.len() == self.hi - self.lo,
                "batch decode requires each sequence to be prefilled");
            anyhow::ensure!(row.pos < row.context_budget, "sequence {} reached its context budget", row.seq);
            anyhow::ensure!(id < self.cfg.text_config.vocab_size, "token {id} is outside the embedding");
            anyhow::ensure!(row.bank() == bank, "a batch cannot mix teacher/student banks or teacher versions");
            for queue in [&row.audio, &row.vision].into_iter().flatten() { queue.check(&[id])?; }
        }
        let teacher = std::mem::replace(&mut self.moe.teacher, rows[0].teacher.take());
        let packed = self.moe.pack_rows;
        self.moe.pack_rows = packed || n > 1;
        #[cfg(feature = "inkling-cuda")]
        let learn_layer = self.moe.learn_layer.take();
        #[cfg(feature = "inkling-cuda")]
        { self.moe.learn = None; }
        let result = self.batch_forward(rows, ids);
        rows[0].teacher = std::mem::replace(&mut self.moe.teacher, teacher);
        self.moe.pack_rows = packed;
        #[cfg(feature = "inkling-cuda")]
        { self.moe.learn_layer = learn_layer; self.moe.learn = None; }
        if result.is_err() {
            for row in rows { row.torn = true; }
        }
        result
    }

    fn batch_forward(&mut self, rows: &mut [&mut Sequence], ids: &[usize]) -> Result<T2> {
        let t = &self.cfg.text_config;
        let h = t.hidden_size;
        let n = ids.len();
        let tp = self.group.as_ref().map(Group::tp);
        let group = self.group.as_ref();
        let dev = &self.dev;
        let reduce = |x: T2| match group {
            Some(group) => super::super::tpcomm::reduce_activation(group, dev, x),
            None => x,
        };
        self.cleanup_gate.begin_pass();
        let mut input = Vec::with_capacity(n * h);
        for (row, &id) in rows.iter_mut().zip(ids) {
            let mut x = embed_and_norm_bf16(&[id], &self.embed, &self.embed_norm,
                t.rms_norm_eps, t.vocab_size, h);
            if let (Some(audio), Some(queue)) = (&mut self.audio, &mut row.audio) {
                std::mem::swap(&mut audio.queue, queue);
                let result = audio_rows(audio, &[id], x, h);
                std::mem::swap(&mut audio.queue, queue);
                x = result?;
            }
            if let (Some(vision), Some(queue)) = (&mut self.vision, &mut row.vision) {
                std::mem::swap(&mut vision.queue, queue);
                let result = vision_rows(vision, &[id], x, h, dev);
                std::mem::swap(&mut vision.queue, queue);
                x = result?;
            }
            input.extend(x);
        }
        let mut xd = dev_lane_resid::as_resid(up2::<Bk>(input, n, h, dev));
        let router_arm = RouterArm::from_env();
        let t_read = std::cell::Cell::new(0.0);
        for layer in self.lo..self.hi {
            let slot = layer - self.lo;
            let p = format!("model.llm.layers.{layer}.");
            if !self.layers.contains_key(&p) {
                let bound = bind_layer(&self.src, dev, &self.client, self.aliases.as_ref(),
                    &p, layer, t, tp, router_arm, false, &t_read)?;
                self.layers.insert(p.clone(), bound.layer);
            }
            let ld = self.layers.get(&p).expect("bound layer");
            let kind = t.attn_kind(layer);
            let (heads, kv_heads, head_dim) = t.heads(kind);
            let (heads, kv_heads) = match tp {
                Some(tp) => (tp.share("q_heads", heads).map_err(|e| anyhow::anyhow!(e))?,
                    tp.share("kv_heads", kv_heads).map_err(|e| anyhow::anyhow!(e))?),
                None => (heads, kv_heads),
            };
            let dims = AttnDims { hidden: h, heads, kv_heads, head_dim, d_rel: t.d_rel,
                rel_extent: t.rel_span(kind), kernel: t.sconv_kernel_size,
                rms_eps: t.rms_norm_eps, kind };
            let window = (kind == AttnKind::Local).then_some(t.sliding_window_size);
            let ls = LogScaling { n_floor: t.log_scaling_n_floor as f32, alpha: t.log_scaling_alpha as f32 };
            let hn = dev_lane_resid::rms_norm(xd.clone(), ld.attn_norm.clone(), t.rms_norm_eps);
            let mut attention = Vec::with_capacity(n);
            for (i, row) in rows.iter_mut().enumerate() {
                attention.push(dev_lane::attention_step(hn.clone().slice([i..i + 1, 0..h]),
                    &ld.attn, &dims, Some(ls), row.pos, window, &mut row.caches[slot].attn));
            }
            // ONE collective, before each independent output convolution.
            let y = reduce(BT::<Bk, 2>::cat(attention, 0));
            let mut outputs = Vec::with_capacity(n);
            for (i, row) in rows.iter_mut().enumerate() {
                let cache = &mut row.caches[slot];
                let (out, history) = dev_lane::short_conv_step(cache.attn_sconv.clone(),
                    y.clone().slice([i..i + 1, 0..h]), ld.attn_sconv.clone());
                cache.attn_sconv = history;
                outputs.push(out);
            }
            xd = dev_lane_resid::add_resid(xd, BT::<Bk, 2>::cat(outputs, 0));
            let hn = dev_lane_resid::rms_norm(xd.clone(), ld.mlp_norm.clone(), t.rms_norm_eps);
            let y = if t.is_dense(layer) {
                let weights = self.dense.dense_for(&self.src, &self.client,
                    self.aliases.as_ref(), &p, h, tp)?;
                dense_mlp_bf16(hn, weights)
            } else {
                moe_layer(&self.src, &self.client, self.aliases.as_ref(), &mut self.dense,
                    &mut self.moe, dev, &p, layer, t, ld.router.as_ref().expect("routed layer"),
                    hn, n, self.shared_halved, tp, false)?
            };
            let y = reduce(y);
            let mut outputs = Vec::with_capacity(n);
            for (i, row) in rows.iter_mut().enumerate() {
                let cache = &mut row.caches[slot];
                let history = cache.mlp_sconv.clone().context("prefill did not seed MLP history")?;
                let (out, history) = dev_lane::short_conv_step(history,
                    y.clone().slice([i..i + 1, 0..h]), ld.mlp_sconv.clone());
                cache.mlp_sconv = Some(history);
                outputs.push(out);
            }
            xd = dev_lane_resid::add_resid(xd, BT::<Bk, 2>::cat(outputs, 0));
            let client = &self.client;
            if self.cleanup_gate.at_layer(layer + 1 == self.hi, || {
                client.memory_usage().map(|usage| super::super::pool::stranded_bytes(
                    usage.bytes_reserved, usage.bytes_in_use, usage.bytes_padding)).unwrap_or(0)
            }) {
                <Bk as burn::tensor::backend::Backend>::sync(dev)
                    .map_err(|e| anyhow::anyhow!("sync before batch cleanup: {e:?}"))?;
                client.memory_cleanup();
            }
        }
        let vocab = t.effective_vocab();
        let mut heads = Vec::new();
        // Bound head transients like the scored path, rather than materializing
        // an arbitrary batch x vocabulary allocation.
        for lo in (0..n).step_by(SCORE_ROWS) {
            let hi = (lo + SCORE_ROWS).min(n);
            let hs = dev_lane_resid::rms_norm(xd.clone().slice([lo..hi, 0..h]),
                self.final_norm.clone(), t.rms_norm_eps).div_scalar(t.logits_mup_width_multiplier as f32);
            let mut logits = dev_lane::linear_w(hs, &self.unembed).slice([0..hi - lo, 0..vocab]);
            for &id in self.forbidden.iter().filter(|&&id| id < vocab) {
                let column = logits.clone().slice([0..hi - lo, id..id + 1]);
                logits = logits.slice_assign([0..hi - lo, id..id + 1],
                    column.mul_scalar(0.0).sub_scalar(f32::INFINITY));
            }
            heads.push(logits);
        }
        for row in rows.iter_mut() {
            row.pos += 1;
            row.last = None;
            for (slot, cache) in row.caches.iter().enumerate() {
                let (base, len) = required_cache_span(t.attn_kind(self.lo + slot), row.pos, t.sliding_window_size);
                let evicted: usize = cache.attn.evicted().iter().map(|&(a, b)| b - a).sum();
                anyhow::ensure!(cache.attn.base() == base && cache.attn.len() == len.saturating_sub(evicted),
                    "batch sequence {} layer {} has incomplete cache", row.seq, self.lo + slot);
            }
        }
        Ok(BT::<Bk, 2>::cat(heads, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty(owner: u64, capacity: usize) -> Sequence {
        Sequence { owner, caches: Vec::new(), pos: 0, last: None, torn: false,
            seq: next_seq(), context_budget: capacity,
            audio: Some(SlotQueue::new("audio", 2)), vision: Some(SlotQueue::new("image", 4)),
            #[cfg(feature = "inkling-cuda")]
            anchor: Vec::new(), teacher: None, teacher_version: None }
    }

    #[test]
    fn reset_one_capsule_does_not_reset_another_or_its_media() {
        let mut live = empty(7, 1024);
        live.pos = 91; live.last = Some(42);
        live.audio.as_mut().unwrap().stage(5, &[1, 2]).unwrap();
        let live_id = live.identity();
        let mut shadow = empty(7, 2048);
        shadow.pos = 311; shadow.last = Some(88); shadow.torn = true;
        shadow.vision.as_mut().unwrap().stage(6, &[1, 2, 3, 4]).unwrap();
        let old_shadow_id = shadow.identity();
        shadow.reset();
        assert_eq!((live.position(), live.last_token(), live.identity()), (91, Some(42), live_id));
        assert_eq!(live.audio.as_ref().unwrap().staged(), 1);
        assert_eq!((shadow.position(), shadow.last_token(), shadow.is_poisoned()), (0, None, false));
        assert_ne!(shadow.identity(), old_shadow_id);
        assert_eq!(shadow.context_budget(), 2048);
        assert_eq!(shadow.vision.as_ref().unwrap().slot, None);
        assert_eq!(shadow.vision.as_ref().unwrap().staged(), 0);
    }

    #[test]
    fn capsules_and_reset_never_reuse_checkpoint_identity() {
        let a = empty(1, 128);
        let mut b = empty(1, 128);
        let before = b.identity();
        assert_ne!(a.identity(), before);
        b.reset();
        assert_ne!(a.identity(), b.identity());
        assert_ne!(before, b.identity());
    }

    fn tiny_config() -> InklingTextConfig {
        serde_json::from_str(r#"{
            "hidden_size":128,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
            "vocab_size":256,"d_rel":4,"rel_extent":16,"rms_norm_eps":0.000001,
            "sconv_kernel_size":4,"sliding_window_size":8,"local_layer_ids":[0],
            "dense_mlp_idx":0,"dense_intermediate_size":256,"intermediate_size":128,
            "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
            "route_scale":1.0,"gate_activation":"softmax"
        }"#).unwrap()
    }

    fn admission() -> budget::AdmissionPolicy {
        budget::AdmissionPolicy::new(super::super::super::pool::AllocatorConfig::ExclusivePages,
            budget::StorageDType::Bf16, budget::StorageDType::Bf16, budget::StorageDType::Bf16)
    }

    #[test]
    fn independent_admission_prices_each_cache_and_bounded_head_once() {
        let t = tiny_config();
        let policy = admission();
        assert_eq!(extra_sequence_bytes(&t, 0..2, &[], policy, None).unwrap(), 0);
        assert!(extra_sequence_bytes(&t, 0..2, &[0], policy, None).is_err());
        let one = extra_sequence_bytes(&t, 0..2, &[128], policy, None).unwrap();
        let two = extra_sequence_bytes(&t, 0..2, &[128, 128], policy, None).unwrap();
        let head = SCORE_ROWS as u64 * t.effective_vocab() as u64 * 8;
        assert_eq!(two, 2 * one - head);
        assert!(one > budget::kv_cache_bytes(&t, 0..2, 128, policy));
        let longer = extra_sequence_bytes(&t, 0..2, &[256], policy, None).unwrap();
        assert_eq!(longer - one,
            budget::kv_cache_bytes(&t, 0..2, 256, policy) - budget::kv_cache_bytes(&t, 0..2, 128, policy));
    }

    #[test]
    fn tp_shares_kv_not_per_sequence_hidden_histories() {
        let t = tiny_config();
        let single = extra_sequence_bytes(&t, 0..2, &[128], admission(), None).unwrap();
        let tp = Tp::new(0, 2).unwrap();
        let split = extra_sequence_bytes(&t, 0..2, &[128], admission().with_tp_world(2), Some(tp)).unwrap();
        assert!(split < single);
        assert!(2 * split > single, "replicated histories/head/anchors do not halve");
    }
}
