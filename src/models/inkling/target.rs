//! Backend-free settlement for one speculative target extension.
//!
//! The device cache owns the rows; this module owns the transaction law around
//! them. A target pass appends `proposed` pending rows to every layer while the
//! session's committed position and next token stay at `base_position` and
//! `base_last`. Settlement keeps one common prefix in every cache, then moves
//! the two scalar fields to the same boundary. Keeping this arithmetic outside
//! the CUDA module makes the state transition testable without a model or GPU.

use anyhow::{Result, ensure};

/// The committed session state on which one target pass was started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Boundary {
    base_position: usize,
    base_last: Option<usize>,
    proposed: usize,
}

/// The scalar half of a settled target pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Settlement {
    pub(crate) accepted: usize,
    pub(crate) position: usize,
    pub(crate) last: Option<usize>,
}

/// The cache operations settlement needs, deliberately independent of tensors.
///
/// `end_position` includes pending rows. `commit_prefix` is infallible after
/// validation: implementations validate their own pending shape in
/// `pending_rows`, then discard or retain exactly the requested prefix.
pub(crate) trait PrefixCache {
    fn pending_rows(&self) -> Option<usize>;
    fn end_position(&self) -> usize;
    fn commit_prefix(&mut self, keep: usize);
}

impl Boundary {
    pub(crate) fn new(
        base_position: usize,
        base_last: Option<usize>,
        proposed: usize,
    ) -> Result<Self> {
        ensure!(proposed > 0, "a target extension proposes at least one row");
        ensure!(
            base_position.checked_add(proposed).is_some(),
            "target position overflows usize"
        );
        Ok(Self {
            base_position,
            base_last,
            proposed,
        })
    }

    #[cfg(feature = "inkling-cuda")]
    pub(crate) fn base_position(self) -> usize {
        self.base_position
    }

    #[cfg(feature = "inkling-cuda")]
    pub(crate) fn base_last(self) -> Option<usize> {
        self.base_last
    }

    #[cfg(feature = "inkling-cuda")]
    pub(crate) fn proposed(self) -> usize {
        self.proposed
    }

    /// Keep `accepted` leading rows and discard the rest.
    pub(crate) fn accept<C: PrefixCache>(
        self,
        caches: &mut [C],
        position: &mut usize,
        last: &mut Option<usize>,
        predictions: &[usize],
        accepted: usize,
    ) -> Result<Settlement> {
        self.settle(caches, position, last, predictions, accepted)
    }

    /// Discard the whole pass. This is intentionally distinct from `accept(0)`
    /// at the API boundary even though both restore the same cache state.
    pub(crate) fn abort<C: PrefixCache>(
        self,
        caches: &mut [C],
        position: &mut usize,
        last: &mut Option<usize>,
        predictions: &[usize],
    ) -> Result<Settlement> {
        self.settle(caches, position, last, predictions, 0)
    }

    fn settle<C: PrefixCache>(
        self,
        caches: &mut [C],
        position: &mut usize,
        last: &mut Option<usize>,
        predictions: &[usize],
        accepted: usize,
    ) -> Result<Settlement> {
        ensure!(
            accepted <= self.proposed,
            "accepted {accepted} of only {} proposed rows",
            self.proposed
        );
        ensure!(
            predictions.len() == self.proposed,
            "{} target predictions for {} proposed rows",
            predictions.len(),
            self.proposed
        );
        ensure!(!caches.is_empty(), "a target extension has no layer caches");
        ensure!(
            *position == self.base_position && *last == self.base_last,
            "the committed session moved while a target extension was pending"
        );

        let pending_end = self.base_position + self.proposed;
        // Validate the whole stack before changing any layer. A width mismatch
        // is an internal tear; discovering it after the first commit would turn
        // a diagnosable refusal into a newly torn stack.
        for (layer, cache) in caches.iter().enumerate() {
            ensure!(
                cache.pending_rows() == Some(self.proposed),
                "target layer {layer} does not hold exactly {} pending rows",
                self.proposed
            );
            ensure!(
                cache.end_position() == pending_end,
                "target layer {layer} ends at {}, expected {pending_end}",
                cache.end_position()
            );
        }

        for cache in caches.iter_mut() {
            cache.commit_prefix(accepted);
        }

        let committed_end = self.base_position + accepted;
        for (layer, cache) in caches.iter().enumerate() {
            assert_eq!(
                cache.pending_rows(),
                None,
                "target layer {layer} remained pending after settlement"
            );
            assert_eq!(
                cache.end_position(),
                committed_end,
                "target layer {layer} committed a different prefix"
            );
        }

        *position = committed_end;
        if accepted > 0 {
            *last = Some(predictions[accepted - 1]);
        }
        Ok(Settlement {
            accepted,
            position: committed_end,
            last: *last,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Cache {
        committed: usize,
        pending: Option<usize>,
    }

    impl Cache {
        fn target(base: usize, rows: usize) -> Self {
            Self {
                committed: base,
                pending: Some(rows),
            }
        }
    }

    impl PrefixCache for Cache {
        fn pending_rows(&self) -> Option<usize> {
            self.pending
        }

        fn end_position(&self) -> usize {
            self.committed + self.pending.unwrap_or(0)
        }

        fn commit_prefix(&mut self, keep: usize) {
            let rows = self.pending.take().expect("validated before mutation");
            assert!(keep <= rows);
            self.committed += keep;
        }
    }

    fn begun() -> (Boundary, Vec<Cache>, usize, Option<usize>, Vec<usize>) {
        (
            Boundary::new(11, Some(7), 4).unwrap(),
            vec![Cache::target(11, 4), Cache::target(11, 4)],
            11,
            Some(7),
            vec![101, 102, 103, 104],
        )
    }

    #[test]
    fn accepting_zero_restores_every_cache_and_both_scalars() {
        let (b, mut caches, mut pos, mut last, predictions) = begun();
        let settled = b
            .accept(&mut caches, &mut pos, &mut last, &predictions, 0)
            .unwrap();

        assert_eq!(settled.accepted, 0);
        assert_eq!((pos, last), (11, Some(7)));
        assert!(
            caches
                .iter()
                .all(|c| c.committed == 11 && c.pending.is_none())
        );
    }

    #[test]
    fn accepting_a_prefix_moves_every_cache_and_last_to_that_prefix() {
        let (b, mut caches, mut pos, mut last, predictions) = begun();
        let settled = b
            .accept(&mut caches, &mut pos, &mut last, &predictions, 2)
            .unwrap();

        assert_eq!(
            settled,
            Settlement {
                accepted: 2,
                position: 13,
                last: Some(102)
            }
        );
        assert!(
            caches
                .iter()
                .all(|c| c.committed == 13 && c.pending.is_none())
        );
    }

    #[test]
    fn accepting_all_is_a_commit_of_the_whole_extension() {
        let (b, mut caches, mut pos, mut last, predictions) = begun();
        let settled = b
            .accept(&mut caches, &mut pos, &mut last, &predictions, 4)
            .unwrap();

        assert_eq!(settled.position, 15);
        assert_eq!(settled.last, Some(104));
        assert!(caches.iter().all(|c| c.committed == 15));
    }

    #[test]
    fn abort_discards_the_whole_extension() {
        let (b, mut caches, mut pos, mut last, predictions) = begun();
        let settled = b
            .abort(&mut caches, &mut pos, &mut last, &predictions)
            .unwrap();

        assert_eq!(
            settled,
            Settlement {
                accepted: 0,
                position: 11,
                last: Some(7)
            }
        );
        assert!(
            caches
                .iter()
                .all(|c| c.committed == 11 && c.pending.is_none())
        );
    }

    #[test]
    fn a_cross_layer_width_mismatch_refuses_before_mutating_anything() {
        let (b, mut caches, mut pos, mut last, predictions) = begun();
        caches[1].pending = Some(3);
        let before = caches.clone();

        let err = b
            .accept(&mut caches, &mut pos, &mut last, &predictions, 2)
            .unwrap_err();

        assert!(err.to_string().contains("target layer 1"));
        assert_eq!(caches, before);
        assert_eq!((pos, last), (11, Some(7)));
    }

    #[test]
    fn accepting_past_the_proposal_refuses_before_mutating_anything() {
        let (b, mut caches, mut pos, mut last, predictions) = begun();
        let before = caches.clone();

        let err = b
            .accept(&mut caches, &mut pos, &mut last, &predictions, 5)
            .unwrap_err();

        assert!(err.to_string().contains("accepted 5 of only 4"));
        assert_eq!(caches, before);
        assert_eq!((pos, last), (11, Some(7)));
    }
}
