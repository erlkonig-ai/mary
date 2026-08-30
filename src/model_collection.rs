//! Native collection persistence and compatibility projection for Mary models.
//!
//! New model fragments live in one fixed, append-only `SimpleArchive` union.
//! Publication takes an already-open [`Pile`] and a caller-supplied signing
//! key; reads first discover the authority-admitted opaque payload cover and
//! then materialize exactly that cover. Signatures and metadata remain
//! queryable provenance rather than coordinates of the model value. There is
//! no repository, mutable head, fallback store, repair, reopen, or implicit
//! durability flush in this surface.
//!
//! TribleSpace commit `6b65f278` changed `"hex" as attribute: Encoding` from a
//! literal attribute id to an encoding-aware id derived from `(hex, Encoding)`.
//! Model piles written before that epoch contain the literal ids. This module
//! preserves those facts and adds their canonical anchored-attribute aliases;
//! it does not choose a repository/collection migration policy or alter runtime
//! query declarations.

use std::collections::{BTreeSet, HashMap};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::path::Path;

use ed25519_dalek::{SigningKey, VerifyingKey};
use triblespace::core::attribute::Attribute;
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::{Blob, IntoBlob, TryFromBlob};
use triblespace::core::collection::simplearchive_union::{
    self, FactViewError, PreparationError, PreparedCollectionCommit, PublicationError,
};
use triblespace::core::collection::{
    CollectionAdmissionError, CollectionCommit, CollectionCoverError, CollectionRead,
    CollectionReadError, CollectionRecord, FactCover, FactMaterializationError,
    SimpleArchiveCollection,
};
use triblespace::core::collection::{descriptor, reach};
use triblespace::core::inline::encodings::UnknownInline;
use triblespace::core::metadata;
use triblespace::core::repo::pile::{
    CollectionInsertError, FlushError, GetBlobError, InsertError as PileInsertError, PileSnapshot,
    PileWriteError, ReadError,
};
use triblespace::core::repo::{OfferCaptureInsertError, SnapshotSource};
use triblespace::core::trible::TribleSet;
use triblespace::prelude::inlineencodings::{F64, U256BE};
use triblespace::prelude::*;

/// The name Mary's canonical model-graph collection is known by.
///
/// This replaces a minted scope id. A root collection is anchored by a name
/// within a team rather than by an opaque anchor, so the pile now says what
/// this collection is instead of that being answerable only by someone
/// holding this source file. The descriptor additionally fixes
/// `SimpleArchive` as its representation and TribleSpace's version-1
/// trible-set union recipe as its algebra.
pub const fn mary_model_graph_name() -> &'static str {
    "mary-model-graph"
}

/// The root collection of immutable model-bundle tokens.
///
/// One signed membership root is exactly one trible
/// `(model_root, metadata::archive, H)`. `H` is the canonical archive of the
/// complete model facts; that archive and every attachment carried by the
/// source fragment are staged before the token COMMIT. The token signature is
/// the model authority -- there is no second model-graph COMMIT to trust.
/// Runtime derivations use this tiny union as their truthful source lattice.
pub const fn mary_model_bundle_name() -> &'static str {
    "mary-model-bundles"
}

/// Concrete failure produced by publishing one model fragment to a pile.
pub type ModelFragmentPublicationError = PublicationError<
    PileInsertError,
    OfferCaptureInsertError<PileWriteError, CollectionInsertError>,
>;

/// Concrete failure produced while exactly materializing Mary's collection
/// from one frozen pile observation.
///
/// One error parameter fewer than before the store-snapshot epoch: a
/// `ReaderError` arm existed only because acquiring the blob reader was a
/// separate fallible step from enumerating records. Both now come out of the
/// same [`PileSnapshot`], so the only remaining failures are record discovery
/// and blob fetch.
pub type ModelCollectionMaterializationError =
    FactMaterializationError<ReadError, GetBlobError<Infallible>>;

/// Concrete failure produced while discovering one authorized model cover.
pub type ModelCollectionCoverError =
    CollectionCoverError<ReadError, ReadError, GetBlobError<Infallible>>;

/// Concrete failure produced by one capability-aware model read that also
/// retains the exact admitted signed roots of that observation.
///
/// This replaces the retired `CollectionSnapshotError`. The upstream name for
/// "materialization, plus the typed discovery of the admission evidence that
/// chose the cover" is now [`CollectionReadError`].
pub type ModelCollectionSnapshotError =
    CollectionReadError<ReadError, ReadError, GetBlobError<Infallible>, FactViewError>;

/// Concrete failure produced while deciding whether a signer may publish to
/// an already-founded model collection.
pub type ModelCollectionAdmissionError =
    CollectionAdmissionError<ReadError, GetBlobError<Infallible>>;

/// Failure while selecting the collection authority a model writer may use.
#[derive(Debug)]
pub enum ModelCollectionWriterSelectionError {
    /// Candidate collection discovery could not complete.
    Discovery(ModelCollectionTeamDiscoveryError),
    /// More than one authority publishes the requested collection name.
    Several {
        /// Canonical collection name whose authority is ambiguous.
        collection: &'static str,
        /// Every admitted authority found in the coherent pile prefix.
        teams: Vec<[u8; 32]>,
    },
    /// The existing collection's writer policy could not be evaluated.
    Admission(ModelCollectionAdmissionError),
    /// The supplied signer has no resident write capability from the existing
    /// collection authority.
    NotAdmitted {
        /// Canonical collection name the caller attempted to publish to.
        collection: &'static str,
        /// Authority fixed by the existing collection descriptor.
        authority: [u8; 32],
        /// Supplied signer's public key.
        signer: [u8; 32],
    },
}

impl fmt::Display for ModelCollectionWriterSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discovery(source) => source.fmt(formatter),
            Self::Several { collection, teams } => write!(
                formatter,
                "{} authorities publish `{collection}` here; name the intended collection explicitly",
                teams.len(),
            ),
            Self::Admission(source) => {
                write!(
                    formatter,
                    "check model collection writer admission: {source}"
                )
            }
            Self::NotAdmitted {
                collection,
                authority,
                signer,
            } => write!(
                formatter,
                "signer {signer:02X?} is not admitted to existing `{collection}` under authority {authority:02X?}",
            ),
        }
    }
}

impl Error for ModelCollectionWriterSelectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Discovery(source) => Some(source),
            Self::Admission(source) => Some(source),
            Self::Several { .. } | Self::NotAdmitted { .. } => None,
        }
    }
}

/// Failure while discovering which authorities actually publish one of
/// Mary's canonical collections in a pile.
#[derive(Debug)]
pub enum ModelCollectionTeamDiscoveryError {
    /// The pile's structural record or blob observation failed.
    Read(ReadError),
    /// A matching canonical descriptor could not complete authority admission.
    Cover(ModelCollectionCoverError),
}

impl fmt::Display for ModelCollectionTeamDiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(source) => write!(formatter, "read candidate model collections: {source}"),
            Self::Cover(source) => {
                write!(formatter, "admit a candidate model collection: {source}")
            }
        }
    }
}

impl Error for ModelCollectionTeamDiscoveryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read(source) => Some(source),
            Self::Cover(source) => Some(source),
        }
    }
}

/// Failure while freezing the locally admitted model commits from an already
/// open pile.
#[derive(Debug)]
pub enum SnapshotLocalModelCollectionError {
    /// One coherent pile observation could not be frozen.
    Observation(ReadError),
    /// The exact authorized local cover could not be discovered.
    Cover(ModelCollectionCoverError),
    /// The frozen exact cover could not be materialized.
    Materialize(ModelCollectionMaterializationError),
}

/// Failure while choosing the sole model-graph team and freezing its cover.
#[derive(Debug)]
pub enum SnapshotSoleModelGraphError {
    /// The coherent pile observation could not be opened or closed.
    Observation(ReadError),
    /// Team discovery found no unique model-graph collection in the prefix.
    Team(SoleModelGraphTeamError),
    /// The exact authorized local cover could not be discovered.
    Cover(ModelCollectionCoverError),
    /// The fixed cover could not be materialized exactly.
    Materialize(ModelCollectionMaterializationError),
}

impl fmt::Display for SnapshotSoleModelGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation(source) => source.fmt(f),
            Self::Team(source) => source.fmt(f),
            Self::Cover(source) => source.fmt(f),
            Self::Materialize(source) => source.fmt(f),
        }
    }
}

impl Error for SnapshotSoleModelGraphError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Observation(source) => Some(source),
            Self::Team(source) => Some(source),
            Self::Cover(source) => Some(source),
            Self::Materialize(source) => Some(source),
        }
    }
}

impl fmt::Display for SnapshotLocalModelCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation(source) => {
                write!(f, "failed to freeze a model pile observation: {source}")
            }
            Self::Cover(source) => write!(f, "failed to discover the local model cover: {source}"),
            Self::Materialize(source) => {
                write!(f, "failed to materialize the model collection: {source}")
            }
        }
    }
}

impl Error for SnapshotLocalModelCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Observation(source) => Some(source),
            Self::Cover(source) => Some(source),
            Self::Materialize(source) => Some(source),
        }
    }
}

/// Failure while choosing the sole bundle team and freezing its cover.
#[derive(Debug)]
pub enum SnapshotSoleModelBundleError {
    /// The coherent pile observation could not be opened or closed.
    Observation(ReadError),
    /// Team discovery found no unique bundle collection in the frozen prefix.
    Team(SoleModelBundleTeamError),
    /// The exact authorized local cover could not be discovered.
    Cover(ModelCollectionCoverError),
    /// The fixed cover could not be materialized exactly.
    Materialize(ModelCollectionMaterializationError),
}

impl fmt::Display for SnapshotSoleModelBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation(source) => source.fmt(f),
            Self::Team(source) => source.fmt(f),
            Self::Cover(source) => source.fmt(f),
            Self::Materialize(source) => source.fmt(f),
        }
    }
}

impl Error for SnapshotSoleModelBundleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Observation(source) => Some(source),
            Self::Team(source) => Some(source),
            Self::Cover(source) => Some(source),
            Self::Materialize(source) => Some(source),
        }
    }
}

/// Failure while publishing one model fragment.
#[derive(Debug)]
pub enum PublishModelFragmentError {
    /// The caller's open pile could not refresh its observed prefix before any
    /// publication work began.
    Refresh(ReadError),
    /// Canonical fragment preparation or dependency/record publication failed.
    Publication(ModelFragmentPublicationError),
}

/// Failure while preparing one self-contained model bundle.
#[derive(Debug)]
pub enum PrepareModelBundleError {
    /// The asserted root has no facts in the model archive.
    RootAbsent(Id),
    /// Canonical token preparation rejected the descriptor or archives.
    Preparation(PreparationError),
}

impl fmt::Display for PrepareModelBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootAbsent(root) => {
                write!(
                    f,
                    "model bundle root {root} is absent from its model archive"
                )
            }
            Self::Preparation(source) => write!(f, "prepare model bundle token: {source}"),
        }
    }
}

impl Error for PrepareModelBundleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Preparation(source) => Some(source),
            Self::RootAbsent(_) => None,
        }
    }
}

/// Failure while publishing one signed model-bundle token.
#[derive(Debug)]
pub enum PublishModelBundleError {
    /// The caller's open pile could not refresh before any publication work.
    Refresh(ReadError),
    /// The candidate could not be turned into a canonical one-row bundle.
    Prepare(PrepareModelBundleError),
    /// Bundle dependencies or its sole signed COMMIT could not be appended.
    Publication(ModelFragmentPublicationError),
}

impl fmt::Display for PublishModelBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refresh(source) => write!(f, "refresh before model bundle publication: {source}"),
            Self::Prepare(source) => source.fmt(f),
            Self::Publication(source) => write!(f, "publish model bundle: {source}"),
        }
    }
}

impl Error for PublishModelBundleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Refresh(source) => Some(source),
            Self::Prepare(source) => Some(source),
            Self::Publication(source) => Some(source),
        }
    }
}

/// A canonical one-row bundle ready to be staged and signed.
///
/// `model_archive_data` is `H`, the identity of the complete model-fact archive, not
/// the identity of the one-row token archive which the eventual COMMIT signs.
#[derive(Clone, Debug)]
pub struct PreparedModelBundle {
    model_root: Id,
    model_archive_data: triblespace::core::collection::CollectionData,
    prepared: PreparedCollectionCommit,
}

impl PreparedModelBundle {
    /// Content-derived model root asserted by the token.
    pub fn model_root(&self) -> Id {
        self.model_root
    }

    /// Canonical model-fact archive identity `H` asserted by the token.
    pub fn model_archive_data(&self) -> triblespace::core::collection::CollectionData {
        self.model_archive_data
    }

    /// Consume this bundle into TribleSpace's commit-last staging value.
    pub fn into_prepared_commit(self) -> PreparedCollectionCommit {
        self.prepared
    }
}

impl fmt::Display for PublishModelFragmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refresh(source) => {
                write!(
                    f,
                    "failed to refresh model pile before publication: {source}"
                )
            }
            Self::Publication(source) => source.fmt(f),
        }
    }
}

impl Error for PublishModelFragmentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Refresh(source) => Some(source),
            Self::Publication(source) => Some(source),
        }
    }
}

/// Failure while opening and loading a model collection snapshot from a pile
/// path.
#[derive(Debug)]
pub enum LoadModelCollectionError {
    /// The supplied pile path could not be opened.
    Open(ReadError),
    /// The newly opened pile could not replay one observed prefix.
    Refresh(ReadError),
    /// The exact authorized local cover could not be discovered.
    Cover(ModelCollectionCoverError),
    /// The frozen exact cover could not be verified or materialized.
    Materialize(ModelCollectionMaterializationError),
    /// The read-only pile handle could not be closed after snapshot creation.
    Close(FlushError),
}

/// Failure while opening a pile and atomically choosing and materializing its
/// sole model-graph collection from one observed prefix.
#[derive(Debug)]
pub enum LoadSoleModelCollectionError {
    /// The supplied pile path could not be opened.
    Open(ReadError),
    /// Sole-team discovery or exact materialization failed in the frozen
    /// prefix.
    Snapshot(SnapshotSoleModelGraphError),
    /// The read-only pile handle could not be closed after snapshot creation.
    Close(FlushError),
}

impl fmt::Display for LoadSoleModelCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(source) => write!(f, "failed to open model pile: {source}"),
            Self::Snapshot(source) => source.fmt(f),
            Self::Close(source) => write!(f, "failed to close model pile: {source}"),
        }
    }
}

impl Error for LoadSoleModelCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Open(source) => Some(source),
            Self::Snapshot(source) => Some(source),
            Self::Close(source) => Some(source),
        }
    }
}

impl fmt::Display for LoadModelCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(source) => write!(f, "failed to open model pile: {source}"),
            Self::Refresh(source) => write!(f, "failed to refresh model pile: {source}"),
            Self::Cover(source) => write!(f, "failed to discover the local model cover: {source}"),
            Self::Materialize(source) => {
                write!(f, "failed to materialize the model collection: {source}")
            }
            Self::Close(source) => write!(f, "failed to close model pile: {source}"),
        }
    }
}

impl Error for LoadModelCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Open(source) => Some(source),
            Self::Refresh(source) => Some(source),
            Self::Cover(source) => Some(source),
            Self::Materialize(source) => Some(source),
            Self::Close(source) => Some(source),
        }
    }
}

// Name, mandatory authority, and reach all participate in descriptor identity.
// These constructors describe only the current epoch; retired namespace/open
// descriptors are inputs to the one-shot collection migration, not a runtime
// compatibility path.
fn model_graph_collection(team: VerifyingKey) -> SimpleArchiveCollection {
    SimpleArchiveCollection::new(mary_model_graph_name(), team, reach::private())
}

fn model_bundle_collection(team: VerifyingKey) -> SimpleArchiveCollection {
    SimpleArchiveCollection::new(mary_model_bundle_name(), team, reach::private())
}

/// One coherent observation of a model collection: its facts, the exact cover
/// they were materialized from, and the store observation that supplied both.
///
/// This replaces the retired `FactSnapshot<PileSnapshot>`, and the difference is
/// the whole point of the port. That type carried a value, a cover, and a
/// *separately acquired* blob reader; keeping the record scan that produced
/// the cover coherent with the reader that materialized it was the caller's
/// problem, and Mary solved it with a seqlock — `observe_stable_pile` sampled
/// `Pile::store_revision` before and after every read and retried the whole
/// observation whenever an external append landed in between. A
/// [`PileSnapshot`] now freezes blobs, collection records, capability proofs,
/// and peer evidence at one validated pile prefix, so the three fields below
/// are coherent by construction. The retry loop is gone, not moved.
///
/// `S` stays generic because a caller may materialize the same facts against
/// any store observation (a `MemoryRepo` snapshot in a test, a `PileSnapshot`
/// in production); it is no longer a *reader* parameter, because the store
/// observation is the reader.
#[derive(Clone, Debug)]
pub struct ModelSnapshot<S> {
    facts: TribleSet,
    cover: FactCover,
    store: S,
}

impl<S> ModelSnapshot<S> {
    /// Bind materialized facts to the exact cover and observation behind them.
    pub fn new(facts: TribleSet, cover: FactCover, store: S) -> Self {
        Self {
            facts,
            cover,
            store,
        }
    }

    /// Materialized fact union named by this observation's exact cover.
    pub fn facts(&self) -> &TribleSet {
        &self.facts
    }

    /// Exact collection cover the facts were materialized from.
    pub fn cover(&self) -> &FactCover {
        &self.cover
    }

    /// The frozen store observation the facts and every attachment come from.
    ///
    /// Named `store`, not `reader`: it is the same immutable observation that
    /// supplied the collection records and capability proofs, not a second
    /// lease taken afterwards.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Consume the observation and return only its materialized facts.
    pub fn into_facts(self) -> TribleSet {
        self.facts
    }

    /// Consume the observation into facts, exact cover, and store observation.
    pub fn into_parts(self) -> (TribleSet, FactCover, S) {
        (self.facts, self.cover, self.store)
    }
}

/// A model observation taken from a pile.
pub type ModelPileSnapshot = ModelSnapshot<PileSnapshot>;

/// Content identity of one team's PersonaPlex bundle source collection.
pub fn model_bundle_collection_handle(
    team: VerifyingKey,
) -> triblespace::core::collection::CollectionHandle {
    model_bundle_collection(team).collection().handle()
}

/// Prepare one canonical model commit entirely in memory.
///
/// Preparation validates the descriptor, archives, and every blob embedded in
/// `fragment`, but touches no destination storage and needs no key: the commit
/// is signed by `stage`, over the handles the store itself returns. A
/// commit-last importer can stage the returned value's dependencies, validate
/// them through the destination's own reader, and expose authority only by
/// calling `finalize` after those gates succeed.
pub fn prepare_model_fragment(
    team: VerifyingKey,
    fragment: Fragment,
) -> Result<PreparedCollectionCommit, PreparationError> {
    simplearchive_union::prepare_fragment_commit(
        &model_graph_collection(team).descriptor(),
        fragment,
    )
}

/// Publish one self-contained model fragment under the supplied signer.
///
/// The pile is refreshed before publication. Facts, metafacts, and their
/// shared embedded attachments are then passed directly to
/// [`simplearchive_union::publish_fragment_commit`]. The caller retains the
/// pile and chooses any later durability boundary; this function neither
/// flushes nor closes it.
pub fn publish_model_fragment(
    pile: &mut Pile,
    team: VerifyingKey,
    signing_key: &SigningKey,
    fragment: Fragment,
) -> Result<CollectionCommit, PublishModelFragmentError> {
    pile.refresh().map_err(PublishModelFragmentError::Refresh)?;
    simplearchive_union::publish_fragment_commit(
        pile,
        &model_graph_collection(team).descriptor(),
        fragment,
        signing_key,
    )
    .map_err(PublishModelFragmentError::Publication)
}

/// Prepare a complete model fragment as one atomic bundle membership.
///
/// The source fragment's facts are archived as `H`; its attachments and `H`
/// itself are embedded dependencies of a new fragment whose data is exactly
/// `(model_root, metadata::archive, H)`. Source metafacts become the token
/// COMMIT metadata. Preparation is pure: no pile is read or written and no
/// signature exists until the returned value is staged and finalized.
pub fn prepare_model_bundle_fragment(
    team: VerifyingKey,
    model_root: Id,
    fragment: Fragment,
) -> Result<PreparedModelBundle, PrepareModelBundleError> {
    if !fragment.facts().iter().any(|fact| fact.e() == &model_root) {
        return Err(PrepareModelBundleError::RootAbsent(model_root));
    }

    let (_, facts, metafacts, blobs) = fragment.into_parts();
    let model_archive: Blob<SimpleArchive> = facts.to_blob();
    let mut token = Fragment::empty();
    token.blobs_mut().union(blobs);
    let source = token.put::<SimpleArchive, _>(model_archive);
    let model_archive_data = inlineencodings::Handle::<SimpleArchive>::to_hash(source);
    token += entity! {
        ExclusiveId::force_ref(&model_root) @ metadata::archive: source
    };
    *token.metafacts_mut() += metafacts;
    debug_assert_eq!(token.facts().len(), 1);

    let prepared = simplearchive_union::prepare_fragment_commit(
        &model_bundle_collection(team).descriptor(),
        token,
    )
    .map_err(PrepareModelBundleError::Preparation)?;
    Ok(PreparedModelBundle {
        model_root,
        model_archive_data,
        prepared,
    })
}

/// Publish one self-contained model bundle under the supplied signer.
///
/// The pile receives descriptor dependencies, candidate attachments, `H`, the
/// one-row token archive, and metadata before the sole bundle COMMIT is
/// appended. No model-graph COMMIT is created. The caller keeps the pile and
/// chooses the durability boundary; exact retries are content-idempotent.
pub fn publish_model_bundle_fragment(
    pile: &mut Pile,
    team: VerifyingKey,
    signing_key: &SigningKey,
    model_root: Id,
    fragment: Fragment,
) -> Result<CollectionCommit, PublishModelBundleError> {
    pile.refresh().map_err(PublishModelBundleError::Refresh)?;
    let prepared = prepare_model_bundle_fragment(team, model_root, fragment)
        .map_err(PublishModelBundleError::Prepare)?;
    prepared
        .into_prepared_commit()
        .stage(pile, signing_key)
        .map_err(PublishModelBundleError::Publication)?
        .finalize()
        .map_err(PublishModelBundleError::Publication)
}

/// Materialize exactly the supplied opaque model cover from one frozen store
/// observation. Authorization produces the cover; exact replay consumes only
/// its payload identity. Other commits and later provenance remain inert.
///
/// This takes the observation rather than `&mut Pile` on purpose: the cover
/// was discovered in some prefix, and replaying it against a *later* prefix is
/// exactly the incoherence the snapshot epoch exists to make unsayable.
pub fn snapshot_model_collection_exact(
    store: &PileSnapshot,
    team: VerifyingKey,
    cover: &FactCover,
) -> Result<ModelPileSnapshot, ModelCollectionMaterializationError> {
    let facts = model_graph_collection(team).attach_exact(store, cover)?;
    Ok(ModelSnapshot::new(facts, cover.clone(), store.clone()))
}

/// Materialize exactly the supplied opaque model-bundle cover.
///
/// No local-latest widening, flush, close, or reopen occurs.
pub fn snapshot_model_bundle_collection_exact(
    store: &PileSnapshot,
    team: VerifyingKey,
    cover: &FactCover,
) -> Result<ModelPileSnapshot, ModelCollectionMaterializationError> {
    let facts = model_bundle_collection(team).attach_exact(store, cover)?;
    Ok(ModelSnapshot::new(facts, cover.clone(), store.clone()))
}

fn close_after_snapshot(
    pile: Pile,
    snapshot: Result<ModelPileSnapshot, ModelCollectionMaterializationError>,
) -> Result<ModelPileSnapshot, LoadModelCollectionError> {
    match snapshot {
        Ok(snapshot) => {
            pile.close().map_err(LoadModelCollectionError::Close)?;
            Ok(snapshot)
        }
        Err(source) => {
            // A path loader always consumes its pile handle, including on
            // validation failure. This read-only handle is not dirty, so close
            // performs no durability flush.
            let _ = pile.close();
            Err(LoadModelCollectionError::Materialize(source))
        }
    }
}

fn open_and_refresh_model_pile(path: &Path) -> Result<Pile, LoadModelCollectionError> {
    let mut pile = Pile::open(path).map_err(LoadModelCollectionError::Open)?;
    if let Err(source) = pile.refresh() {
        let _ = pile.close();
        return Err(LoadModelCollectionError::Refresh(source));
    }
    Ok(pile)
}

/// Open `path`, materialize the caller-supplied exact cover, and close the
/// pile while returning the owned store observation.
///
/// Opening and the initial replay are explicit failure stages. No missing
/// file is created, no damaged tail is amputated, and no alternate storage or
/// runtime path is consulted. The returned [`PileSnapshot`] owns its immutable
/// mapping and remains usable after the mutable [`Pile`] is closed.
pub fn load_model_collection_from_cover(
    path: impl AsRef<Path>,
    team: VerifyingKey,
    cover: &FactCover,
) -> Result<ModelPileSnapshot, LoadModelCollectionError> {
    let mut pile = open_and_refresh_model_pile(path.as_ref())?;
    let snapshot = match pile.snapshot() {
        Ok(snapshot) => snapshot,
        Err(source) => {
            let _ = pile.close();
            return Err(LoadModelCollectionError::Refresh(source));
        }
    };
    let materialized = snapshot_model_collection_exact(&snapshot, team, cover);
    close_after_snapshot(pile, materialized)
}

/// Which authorities have at least one admitted member in a
/// `mary-model-graph` collection in this pile.
///
/// A reader does not have to be told whose collection it wants. The pile is
/// self-describing: every commit names its collection by descriptor handle,
/// the descriptor blob is in the pile, and the descriptor states both the name
/// and the team. So "load the model graph" is a lookup by name, not a fact the
/// caller has to carry in from somewhere.
///
/// Returns every distinct team, not the first one found. One team is the
/// ordinary case and callers can take it; more than one is a real ambiguity
/// -- two parties publishing under the same name -- and defaulting to whichever
/// the record scan happened to reach first would resolve it by accident. The
/// caller is made to decide because there is no answer here to give it.
fn collection_teams_in(
    store: &PileSnapshot,
    wanted: &str,
) -> Result<Vec<VerifyingKey>, ModelCollectionTeamDiscoveryError> {
    let mut seen = BTreeSet::new();
    let mut teams = Vec::new();
    let mut descriptors = BTreeSet::new();
    // Records, descriptor blobs, and the capability proofs consulted below all
    // come out of this one observation. Before the snapshot epoch these were
    // three independently refreshing reads and the caller had to fence them.
    for record in store
        .records()
        .map_err(ModelCollectionTeamDiscoveryError::Read)?
    {
        if let CollectionRecord::Commit(commit) =
            record.map_err(ModelCollectionTeamDiscoveryError::Read)?
        {
            descriptors.insert(commit.collection());
        }
    }
    let reader = store;
    for handle in descriptors {
        let Ok(blob) = reader.get::<Blob<SimpleArchive>, _>(handle.transmute()) else {
            // A commit naming a descriptor this pile does not hold is a
            // phantom collection. It is not this function's business to
            // report, but it must not be mistaken for a match either.
            continue;
        };
        let Ok(facts) = <TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(blob) else {
            continue;
        };
        let Ok(Some(name_handle)) = descriptor::name(&facts) else {
            continue;
        };
        let Ok(team) = descriptor::authority(&facts) else {
            continue;
        };
        let name: anybytes::View<str> = match reader.get(name_handle) {
            Ok(name) => name,
            Err(_) => continue,
        };
        if &*name != wanted || seen.contains(&team.to_bytes()) {
            continue;
        }

        // A descriptor-shaped blob and a structural COMMIT are only candidate
        // discovery coordinates. Bind the handle to Mary's exact typed
        // descriptor, then let the collection boundary verify signatures and
        // capability admission. Otherwise an unauthorized signer could mint a
        // second apparent authority and poison sole-team discovery.
        let collection = SimpleArchiveCollection::new(wanted, team, reach::private()).collection();
        if collection.handle() != handle {
            continue;
        }
        let cover = collection
            .admitted(store)
            .map_err(ModelCollectionTeamDiscoveryError::Cover)?;
        if !cover.is_empty() && seen.insert(team.to_bytes()) {
            teams.push(team);
        }
    }
    Ok(teams)
}

fn collection_teams(
    pile: &mut Pile,
    wanted: &'static str,
) -> Result<Vec<VerifyingKey>, ModelCollectionTeamDiscoveryError> {
    let store = pile
        .snapshot()
        .map_err(ModelCollectionTeamDiscoveryError::Read)?;
    collection_teams_in(&store, wanted)
}

pub fn model_graph_teams(
    pile: &mut Pile,
) -> Result<Vec<VerifyingKey>, ModelCollectionTeamDiscoveryError> {
    collection_teams(pile, mary_model_graph_name())
}

/// Which teams publish a `mary-model-bundles` collection in this pile.
pub fn model_bundle_teams(
    pile: &mut Pile,
) -> Result<Vec<VerifyingKey>, ModelCollectionTeamDiscoveryError> {
    collection_teams(pile, mary_model_bundle_name())
}

/// The single team publishing a model graph here, or an error naming the
/// ambiguity.
///
/// The convenience form of [`model_graph_teams`] for the ordinary pile, which
/// has exactly one. It refuses rather than guesses in both directions: no
/// model graph at all, and more than one, are different failures and say so.
pub fn sole_model_graph_team(pile: &mut Pile) -> Result<VerifyingKey, SoleModelGraphTeamError> {
    let teams = model_graph_teams(pile).map_err(SoleModelGraphTeamError::Read)?;
    sole_model_graph_team_from(teams)
}

fn sole_model_graph_team_from(
    teams: Vec<VerifyingKey>,
) -> Result<VerifyingKey, SoleModelGraphTeamError> {
    match teams.len() {
        1 => Ok(teams[0]),
        0 => Err(SoleModelGraphTeamError::None),
        _ => Err(SoleModelGraphTeamError::Several {
            teams: teams.iter().map(|team| team.to_bytes()).collect(),
        }),
    }
}

/// The team already publishing a model graph here, or the writer's own.
///
/// A publisher holds a signing key and a pile, and those two do not settle the
/// question by themselves: the team owns the collection, the key only signs one
/// commit into it. So join the collection that is already here, and found one
/// under your own identity only when there is none. Ambiguity is still refused
/// — picking a team out of several is exactly the guess `sole_model_graph_team`
/// exists to refuse.
pub fn model_graph_team_or_own(
    pile: &mut Pile,
    signing_key: &SigningKey,
) -> Result<VerifyingKey, ModelCollectionWriterSelectionError> {
    collection_team_or_own(
        pile,
        signing_key,
        mary_model_graph_name(),
        model_graph_collection,
    )
}

/// The single team publishing model bundles here, or an explicit ambiguity.
pub fn sole_model_bundle_team(pile: &mut Pile) -> Result<VerifyingKey, SoleModelBundleTeamError> {
    let teams = model_bundle_teams(pile).map_err(SoleModelBundleTeamError::Read)?;
    sole_model_bundle_team_from(teams)
}

fn sole_model_bundle_team_from(
    teams: Vec<VerifyingKey>,
) -> Result<VerifyingKey, SoleModelBundleTeamError> {
    match teams.len() {
        1 => Ok(teams[0]),
        0 => Err(SoleModelBundleTeamError::None),
        _ => Err(SoleModelBundleTeamError::Several {
            teams: teams.iter().map(|team| team.to_bytes()).collect(),
        }),
    }
}

/// Join the bundle collection already present, or found it under this signer.
pub fn model_bundle_team_or_own(
    pile: &mut Pile,
    signing_key: &SigningKey,
) -> Result<VerifyingKey, ModelCollectionWriterSelectionError> {
    collection_team_or_own(
        pile,
        signing_key,
        mary_model_bundle_name(),
        model_bundle_collection,
    )
}

fn collection_team_or_own(
    pile: &mut Pile,
    signing_key: &SigningKey,
    name: &'static str,
    collection_for: fn(VerifyingKey) -> SimpleArchiveCollection,
) -> Result<VerifyingKey, ModelCollectionWriterSelectionError> {
    let signer = signing_key.verifying_key();
    // Discovering the authority and then deciding whether this signer may
    // write under it are two questions about the same prefix. Asking them of
    // one frozen observation is what makes the answer a decision rather than a
    // race: a proof appended between the two calls can no longer make Mary
    // publish under an authority whose admission it never actually observed.
    let store = pile.snapshot().map_err(|source| {
        ModelCollectionWriterSelectionError::Discovery(ModelCollectionTeamDiscoveryError::Read(
            source,
        ))
    })?;
    let teams = collection_teams_in(&store, name)
        .map_err(ModelCollectionWriterSelectionError::Discovery)?;
    let authority = match teams.len() {
        0 => return Ok(signer),
        1 => teams[0],
        _ => {
            return Err(ModelCollectionWriterSelectionError::Several {
                collection: name,
                teams: teams.iter().map(VerifyingKey::to_bytes).collect(),
            });
        }
    };
    let admitted = collection_for(authority)
        .collection()
        .writer_is_admitted(&store, signer)
        .map_err(ModelCollectionWriterSelectionError::Admission)?;
    if !admitted {
        return Err(ModelCollectionWriterSelectionError::NotAdmitted {
            collection: name,
            authority: authority.to_bytes(),
            signer: signer.to_bytes(),
        });
    }
    Ok(authority)
}

/// Why a pile does not have exactly one model-graph team.
#[derive(Debug)]
pub enum SoleModelGraphTeamError {
    /// The pile could not be read.
    Read(ModelCollectionTeamDiscoveryError),
    /// No collection in this pile is named `mary-model-graph`.
    None,
    /// Several teams publish that name here; the caller must choose.
    Several {
        /// Every team found, in discovery order.
        teams: Vec<[u8; 32]>,
    },
}

impl fmt::Display for SoleModelGraphTeamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(source) => write!(f, "read the pile's records: {source}"),
            Self::None => write!(f, "no collection named `mary-model-graph` in this pile"),
            Self::Several { teams } => write!(
                f,
                "{} teams publish `mary-model-graph` here; name the one you mean",
                teams.len()
            ),
        }
    }
}

impl Error for SoleModelGraphTeamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read(source) => Some(source),
            Self::None | Self::Several { .. } => None,
        }
    }
}

/// Why a pile does not have exactly one model-bundle team.
#[derive(Debug)]
pub enum SoleModelBundleTeamError {
    /// The pile could not be read.
    Read(ModelCollectionTeamDiscoveryError),
    /// No collection in this pile is named `mary-model-bundles`.
    None,
    /// Several teams publish that name; the caller must choose.
    Several {
        /// Every team found, in discovery order.
        teams: Vec<[u8; 32]>,
    },
}

impl fmt::Display for SoleModelBundleTeamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(source) => write!(f, "read the pile's records: {source}"),
            Self::None => write!(f, "no collection named `mary-model-bundles` in this pile"),
            Self::Several { teams } => write!(
                f,
                "{} teams publish `mary-model-bundles` here; name the one you mean",
                teams.len()
            ),
        }
    }
}

impl Error for SoleModelBundleTeamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read(source) => Some(source),
            Self::None | Self::Several { .. } => None,
        }
    }
}

/// Discover the exact authorized cover of this team's model-graph collection
/// in one frozen store observation.
///
/// The bound collapsed from four traits to one. `BlobStore<Reader = _,
/// ReaderError = _> + CollectionStore<RecordsError = _> + ArtifactOfferStore +
/// CapabilityProofStore<ProofsError = _>` was the price of a *mutable* store
/// having to hand out four separately refreshed read capabilities; a snapshot
/// already is all four, coherently, so `&PileSnapshot` says the whole thing.
fn local_model_cover(
    store: &PileSnapshot,
    team: VerifyingKey,
) -> Result<FactCover, ModelCollectionCoverError> {
    model_graph_collection(team).collection().admitted(store)
}

/// Discover the exact authorized cover of this team's model-bundle collection.
pub fn local_model_bundle_cover(
    store: &PileSnapshot,
    team: VerifyingKey,
) -> Result<FactCover, ModelCollectionCoverError> {
    model_bundle_collection(team).collection().admitted(store)
}

/// Freeze and materialize every locally admitted model commit from an already
/// open pile.
///
/// This is the in-place form of [`load_model_collection_local_latest`]. The
/// native-record scan defines one observed prefix and the returned snapshot
/// owns its immutable reader; the caller keeps responsibility for closing or
/// further appending to `pile`. No flush, close, reopen, or repair occurs.
pub fn snapshot_model_collection_local_latest(
    pile: &mut Pile,
    team: VerifyingKey,
) -> Result<ModelPileSnapshot, SnapshotLocalModelCollectionError> {
    let store = pile
        .snapshot()
        .map_err(|source| SnapshotLocalModelCollectionError::Observation(source))?;
    snapshot_model_collection_in(&store, team)
}

/// Discover and materialize this team's admitted model cover inside one
/// already-frozen store observation.
///
/// Cover discovery and materialization used to straddle two refreshes of the
/// same mutable pile: the cover named payloads found in one prefix and
/// `snapshot_exact` fetched their bytes from whatever prefix the reader
/// happened to land on. Both halves now read `store`.
pub fn snapshot_model_collection_in(
    store: &PileSnapshot,
    team: VerifyingKey,
) -> Result<ModelPileSnapshot, SnapshotLocalModelCollectionError> {
    let cover = local_model_cover(store, team).map_err(SnapshotLocalModelCollectionError::Cover)?;
    snapshot_model_collection_exact(store, team, &cover)
        .map_err(SnapshotLocalModelCollectionError::Materialize)
}

/// Freeze and exactly materialize every locally admitted model-bundle token
/// commit from an already-open pile.
pub fn snapshot_model_bundle_collection_local_latest(
    pile: &mut Pile,
    team: VerifyingKey,
) -> Result<ModelPileSnapshot, SnapshotLocalModelCollectionError> {
    let store = pile
        .snapshot()
        .map_err(SnapshotLocalModelCollectionError::Observation)?;
    snapshot_model_bundle_collection_in(&store, team)
}

/// Discover and materialize this team's admitted bundle cover inside one
/// already-frozen store observation.
pub fn snapshot_model_bundle_collection_in(
    store: &PileSnapshot,
    team: VerifyingKey,
) -> Result<ModelPileSnapshot, SnapshotLocalModelCollectionError> {
    let cover =
        local_model_bundle_cover(store, team).map_err(SnapshotLocalModelCollectionError::Cover)?;
    snapshot_model_bundle_collection_exact(store, team, &cover)
        .map_err(SnapshotLocalModelCollectionError::Materialize)
}

/// Freeze one model-bundle snapshot together with only the COMMITs admitted by
/// that exact capability observation.
///
/// Normal model reads retain the much smaller payload cover alone. This
/// opt-in form exists for migration retries that must return the original
/// signed root rather than a later or unauthorized duplicate claim over the
/// same payload.
pub fn snapshot_model_bundle_collection_local_latest_with_admission(
    pile: &mut Pile,
    team: VerifyingKey,
) -> Result<(ModelPileSnapshot, Vec<CollectionCommit>), ModelCollectionSnapshotError> {
    // `Pile::snapshot_with_admission` is gone, and what replaces it is more
    // honest about what "that exact capability observation" means: the cover,
    // the COMMITs, and the payload bytes all come out of ONE frozen prefix.
    // Previously the roots and the bytes could be sampled either side of an
    // append, so a migration retry could return a root whose payload it had
    // never actually read.
    let store = pile.snapshot().map_err(|source| {
        ModelCollectionSnapshotError::Discovery(
            triblespace::core::collection::CollectionDiscoveryError::Records(source),
        )
    })?;
    let collection = model_bundle_collection(team).collection();
    let (cover, commits) = collection.admitted_with_claims(&store)?;
    // `read` re-runs admission against the same immutable observation, so it
    // returns exactly `cover`'s facts. It is used rather than `attach_exact`
    // because only the read path is typed in the evidence error this function
    // reports; see the blocker note on widening `EvidenceError = Infallible`.
    let facts = collection.read::<TribleSet, _>(&store)?;
    Ok((ModelSnapshot::new(facts, cover, store), commits))
}

/// Choose the sole model-graph authority and materialize its admitted cover.
pub fn snapshot_sole_model_collection_local_latest(
    pile: &mut Pile,
) -> Result<(VerifyingKey, ModelPileSnapshot), SnapshotSoleModelGraphError> {
    let store = pile
        .snapshot()
        .map_err(SnapshotSoleModelGraphError::Observation)?;
    sole_model_collection_in(&store)
}

/// Choose the sole model-graph authority inside one already-frozen store
/// observation and materialize its admitted cover.
///
/// Three questions — which authorities exist, which cover one of them
/// admits, and what bytes that cover names — used to be three independently
/// refreshing reads wrapped in a retry. They are now three queries against one
/// value, so "sole" is a property of an observation instead of a hope about a
/// moving pile.
pub fn sole_model_collection_in(
    store: &PileSnapshot,
) -> Result<(VerifyingKey, ModelPileSnapshot), SnapshotSoleModelGraphError> {
    let teams = collection_teams_in(store, mary_model_graph_name())
        .map_err(SoleModelGraphTeamError::Read)
        .map_err(SnapshotSoleModelGraphError::Team)?;
    let team = sole_model_graph_team_from(teams).map_err(SnapshotSoleModelGraphError::Team)?;
    let cover = local_model_cover(store, team).map_err(SnapshotSoleModelGraphError::Cover)?;
    let snapshot = snapshot_model_collection_exact(store, team, &cover)
        .map_err(SnapshotSoleModelGraphError::Materialize)?;
    Ok((team, snapshot))
}

/// Choose the sole model-bundle authority and materialize its admitted cover.
pub fn snapshot_sole_model_bundle_collection_local_latest(
    pile: &mut Pile,
) -> Result<(VerifyingKey, ModelPileSnapshot), SnapshotSoleModelBundleError> {
    let store = pile
        .snapshot()
        .map_err(SnapshotSoleModelBundleError::Observation)?;
    sole_model_bundle_collection_in(&store)
}

/// Choose the sole model-bundle authority inside one already-frozen store
/// observation and materialize its admitted cover.
pub fn sole_model_bundle_collection_in(
    store: &PileSnapshot,
) -> Result<(VerifyingKey, ModelPileSnapshot), SnapshotSoleModelBundleError> {
    let teams = collection_teams_in(store, mary_model_bundle_name())
        .map_err(SoleModelBundleTeamError::Read)
        .map_err(SnapshotSoleModelBundleError::Team)?;
    let team = sole_model_bundle_team_from(teams).map_err(SnapshotSoleModelBundleError::Team)?;
    let cover =
        local_model_bundle_cover(store, team).map_err(SnapshotSoleModelBundleError::Cover)?;
    let snapshot = snapshot_model_bundle_collection_exact(store, team, &cover)
        .map_err(SnapshotSoleModelBundleError::Materialize)?;
    Ok((team, snapshot))
}

/// Load the exact payload cover admitted by the collection authority.
///
/// Invalid or unauthorized claims remain inert provenance. Once discovery
/// returns, the opaque cover fixes the model value consumed by materialization.
/// The pile is never repaired, reopened, or implicitly flushed.
pub fn load_model_collection_local_latest(
    path: impl AsRef<Path>,
    team: VerifyingKey,
) -> Result<ModelPileSnapshot, LoadModelCollectionError> {
    let mut pile = Pile::open(path.as_ref()).map_err(LoadModelCollectionError::Open)?;
    let snapshot = match snapshot_model_collection_local_latest(&mut pile, team) {
        Ok(snapshot) => Ok(snapshot),
        Err(SnapshotLocalModelCollectionError::Observation(source)) => {
            let _ = pile.close();
            return Err(LoadModelCollectionError::Refresh(source));
        }
        Err(SnapshotLocalModelCollectionError::Cover(source)) => {
            let _ = pile.close();
            return Err(LoadModelCollectionError::Cover(source));
        }
        Err(SnapshotLocalModelCollectionError::Materialize(source)) => Err(source),
    };
    close_after_snapshot(pile, snapshot)
}

/// Open `path`, choose the sole model-graph authority, and materialize its
/// admitted cover. The returned reader owns its immutable mapping after the
/// read-only pile handle is closed.
pub fn load_sole_model_collection_local_latest(
    path: impl AsRef<Path>,
) -> Result<(VerifyingKey, ModelPileSnapshot), LoadSoleModelCollectionError> {
    let mut pile = Pile::open(path.as_ref()).map_err(LoadSoleModelCollectionError::Open)?;
    let result = snapshot_sole_model_collection_local_latest(&mut pile);
    match result {
        Ok((team, snapshot)) => {
            pile.close().map_err(LoadSoleModelCollectionError::Close)?;
            Ok((team, snapshot))
        }
        Err(source) => {
            let _ = pile.close();
            Err(LoadSoleModelCollectionError::Snapshot(source))
        }
    }
}

/// Number of unique pre-epoch model-graph attributes projected by this module.
///
/// The historical declarations occupied more source sites because `index`,
/// `member`, and `model_name` were shared by the format and tokenizer schemas.
pub const LEGACY_MODEL_ATTRIBUTE_ALIAS_COUNT: usize = 45;

/// One historical-literal to canonical-anchored attribute mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelAttributeAlias {
    /// Stable diagnostic label naming the schema and field.
    pub label: &'static str,
    /// Literal id present in model piles written before the attribute epoch.
    pub historical: Id,
    /// Encoding-aware anchored id used by current runtime declarations.
    pub canonical: Id,
}

/// Per-attribute counts produced while projecting one fact set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelAttributeAliasCounts {
    /// Mapping these counts describe.
    pub alias: ModelAttributeAlias,
    /// Historical facts encountered under this mapping.
    pub historical_facts: usize,
    /// Canonical aliases newly added to the returned union.
    pub aliases_added: usize,
    /// Historical facts whose exact canonical alias was already present.
    pub aliases_already_present: usize,
}

/// Additive projection result and diagnostics.
#[derive(Clone, Debug)]
pub struct ModelAttributeProjection {
    /// The complete input fact set unioned with missing canonical aliases.
    pub facts: TribleSet,
    /// Number of input facts scanned.
    pub input_facts: usize,
    /// Number of facts whose attribute was from the historical model schema.
    pub historical_facts: usize,
    /// Total number of canonical aliases added to [`Self::facts`].
    pub aliases_added: usize,
    /// Exhaustive mapping table with per-attribute counts, including zeroes.
    pub mappings: Vec<ModelAttributeAliasCounts>,
}

// These declarations exist only to name bytes already found in pre-epoch
// piles. `unsafe as` is intentional: unlike the runtime schemas, this side of
// the mapping must denote the historical literal id exactly. No post-epoch
// Inkling or dataset/training attributes belong in this module.
mod historical {
    use crate::f16enc::F16Array;
    use crate::format::{F32Array, U32Array, U64Array};
    use triblespace::prelude::blobencodings::{RawBytes, UTF8String};
    use triblespace::prelude::inlineencodings::{Boolean, F64, GenId, Handle, ShortString, U256BE};
    use triblespace::prelude::*;

    attributes! {
        // Model format and graph, all present before 6b65f278.
        "572B45D52A47608F283D0F778597137A" unsafe as data: Handle<F32Array>;
        "467CCF3FDCCCCE599F6C1B933EACD933" unsafe as data_f16: Handle<F16Array>;
        "D09A91FC3F04C40AE4A42CD6628A9E38" unsafe as shape: Handle<U64Array>;
        "2ADC6462A7F70E230558C5D681E38768" unsafe as data_q4: Handle<U32Array>;
        "23178058559C762BB4B1FEAA36B3566D" unsafe as data_q8: Handle<U32Array>;
        "F9EA2FB90DC094D42A4845B013950032" unsafe as q_scales: Handle<F16Array>;
        "2CC4D16369C4980BCB512937DA204FF5" unsafe as format_marker: GenId;
        "4629D277AD6B52B50DA78DEF63440AF1" unsafe as weight: GenId;
        "18E898172078C843A0351C3D880CC238" unsafe as bias: GenId;
        "52C4A211D2A08BA25C27FFD79FF24C93" unsafe as kind: ShortString;
        "09EA2F7BCF9B0C9714EE39CF269DF2D5" unsafe as safetensor_path: Handle<UTF8String>;
        "33CE12B1B940B13E48D8E5B0ADFD2421" unsafe as index: U256BE;
        "3F46CDE630964D78D62DA32F4A8558C1" unsafe as model_root: GenId;
        "B4B6EC08A0CD70DE63A690168EE78F0F" unsafe as member: GenId;
        "4C1CD1611863E7854C59C7DC706DF77A" unsafe as model_name: Handle<UTF8String>;
        "D20B8E3556C35FF6D18D104C3443D6CF" unsafe as source: Handle<UTF8String>;
        "7AF87320C144AA29C29FE2A5EE7C7EB2" unsafe as quantization: ShortString;

        // Gemma LoRA, also present before 6b65f278. `model_name` is shared
        // with the format schema above and therefore has only one mapping.
        "1A682F45CE40171DD5C6FDB4F086AD69" unsafe as lora_rank: U256BE;
        "198B03AF556B7505CCC9ABD4A1D6E724" unsafe as lora_alpha: F64;
        "B93C4E66F4B9553BF0E8B5DBAD116ECF" unsafe as lora_adapter: GenId;
        "FF8335C187823A267E26B4E33EF157E9" unsafe as lora_projection: ShortString;
        "7CD7F0DC8BDA328735A22DF02B4B8828" unsafe as lora_a: Handle<F32Array>;
        "1F21DAE68652A4D8CAD973400F04124D" unsafe as lora_b: Handle<F32Array>;

        // Tokenizer graph. `piece_bytes` was introduced by d6dcbd3a on
        // 2026-08-05, four days before 6b65f278, so it belongs to this epoch.
        // The shared index/member/model_name attributes are already above.
        "E7014108A8F9512B19E3E8272E8A71F9" unsafe as tokenizer: GenId;
        "E839AA8F549C0D608FB86476A1EF3416" unsafe as vocab: GenId;
        "E229769197BB035A2D6F61BC6A7D44BC" unsafe as merge: GenId;
        "B2553118F4CAAF1D028619956DE7F145" unsafe as added: GenId;
        "53BAF87A0E7F1410F8212B3EDF2A498C" unsafe as normalizer: GenId;
        "6EEBF39CADD11B7CFBB624019AE21585" unsafe as pre_tokenizer: GenId;
        "98EC58B28F4D0BB43965DF7C5FF22713" unsafe as post_processor: GenId;
        "F3AAA4CD8EE04E5592059564A21FE953" unsafe as decoder: GenId;
        "AE7FE29F2F38153F58C542D5CA4A9356" unsafe as piece: Handle<UTF8String>;
        "F0E2E782F7BB62F52B1186DDE0EB5388" unsafe as token_id: U256BE;
        "714AE13F801202EB27C83E3AB2290669" unsafe as piece_bytes: Handle<RawBytes>;
        "5723ECE1FF426C58879B79D5669A7CF1" unsafe as merge_left: Handle<UTF8String>;
        "5C78FEB151F35A2C5D07BEC92E860752" unsafe as merge_right: Handle<UTF8String>;
        "68F1A9E6ED735E7C3ADCCA076AFF1742" unsafe as unk_token: ShortString;
        "11F76A2C0856C16CB030C4327D5A3B93" unsafe as continuing_subword_prefix: ShortString;
        "6FB969E8A3EDD1A657C721DD5A7D42EA" unsafe as end_of_word_suffix: ShortString;
        "DF3F88DBFA2B44A7783169C9640014AF" unsafe as max_input_chars: U256BE;
        "3BCB70478942DB710ED2A4FB023F3457" unsafe as piece_score: F64;
        "EE4C6647619A836326196F0DBF84FA98" unsafe as byte_fallback: Boolean;
        "C8262D5668B8A1F541B3C35D54201BEC" unsafe as pattern: Handle<UTF8String>;
        "3AC7574C07D02D389B4E7AD3B3B084D9" unsafe as replace_content: ShortString;
        "964B4FCF7477E7E4436F0325F89B7CB5" unsafe as behavior: ShortString;
    }
}

/// Return the complete audited mapping used by
/// [`project_legacy_model_attributes`].
///
/// Gemma is feature-gated, so its LoRA attributes are derived here from the
/// same anchors and encodings as `models::gemma::lora::attrs`; the remaining
/// canonical ids come directly from the unconditional runtime declarations.
/// Canonical targets are disjoint from all historical sources, so the table
/// cannot contain an `A -> B -> C` chain that would need a second pass.
pub fn legacy_model_attribute_aliases() -> [ModelAttributeAlias; LEGACY_MODEL_ATTRIBUTE_ALIAS_COUNT]
{
    use crate::format::attrs as current_format;
    use crate::tokenizer::attrs as current_tokenizer;

    let alias = |label, historical, canonical| ModelAttributeAlias {
        label,
        historical,
        canonical,
    };

    [
        alias(
            "format.data",
            historical::data.id(),
            current_format::data.id(),
        ),
        alias(
            "format.data_f16",
            historical::data_f16.id(),
            current_format::data_f16.id(),
        ),
        alias(
            "format.shape",
            historical::shape.id(),
            current_format::shape.id(),
        ),
        alias(
            "format.data_q4",
            historical::data_q4.id(),
            current_format::data_q4.id(),
        ),
        alias(
            "format.data_q8",
            historical::data_q8.id(),
            current_format::data_q8.id(),
        ),
        alias(
            "format.q_scales",
            historical::q_scales.id(),
            current_format::q_scales.id(),
        ),
        alias(
            "format.format_marker",
            historical::format_marker.id(),
            current_format::format_marker.id(),
        ),
        alias(
            "format.weight",
            historical::weight.id(),
            current_format::weight.id(),
        ),
        alias(
            "format.bias",
            historical::bias.id(),
            current_format::bias.id(),
        ),
        alias(
            "format.kind",
            historical::kind.id(),
            current_format::kind.id(),
        ),
        alias(
            "format.safetensor_path",
            historical::safetensor_path.id(),
            current_format::safetensor_path.id(),
        ),
        alias(
            "format.index",
            historical::index.id(),
            current_format::index.id(),
        ),
        alias(
            "format.model_root",
            historical::model_root.id(),
            current_format::model_root.id(),
        ),
        alias(
            "format.member",
            historical::member.id(),
            current_format::member.id(),
        ),
        alias(
            "format.model_name",
            historical::model_name.id(),
            current_format::model_name.id(),
        ),
        alias(
            "format.source",
            historical::source.id(),
            current_format::source.id(),
        ),
        alias(
            "format.quantization",
            historical::quantization.id(),
            current_format::quantization.id(),
        ),
        alias(
            "gemma_lora.lora_rank",
            historical::lora_rank.id(),
            Attribute::<U256BE>::anchored(historical::lora_rank.id()).id(),
        ),
        alias(
            "gemma_lora.lora_alpha",
            historical::lora_alpha.id(),
            Attribute::<F64>::anchored(historical::lora_alpha.id()).id(),
        ),
        alias(
            "gemma_lora.lora_adapter",
            historical::lora_adapter.id(),
            Attribute::<inlineencodings::GenId>::anchored(historical::lora_adapter.id()).id(),
        ),
        alias(
            "gemma_lora.lora_projection",
            historical::lora_projection.id(),
            Attribute::<inlineencodings::ShortString>::anchored(historical::lora_projection.id())
                .id(),
        ),
        alias(
            "gemma_lora.lora_a",
            historical::lora_a.id(),
            Attribute::<inlineencodings::Handle<crate::format::F32Array>>::anchored(
                historical::lora_a.id(),
            )
            .id(),
        ),
        alias(
            "gemma_lora.lora_b",
            historical::lora_b.id(),
            Attribute::<inlineencodings::Handle<crate::format::F32Array>>::anchored(
                historical::lora_b.id(),
            )
            .id(),
        ),
        alias(
            "tokenizer.tokenizer",
            historical::tokenizer.id(),
            current_tokenizer::tokenizer.id(),
        ),
        alias(
            "tokenizer.vocab",
            historical::vocab.id(),
            current_tokenizer::vocab.id(),
        ),
        alias(
            "tokenizer.merge",
            historical::merge.id(),
            current_tokenizer::merge.id(),
        ),
        alias(
            "tokenizer.added",
            historical::added.id(),
            current_tokenizer::added.id(),
        ),
        alias(
            "tokenizer.normalizer",
            historical::normalizer.id(),
            current_tokenizer::normalizer.id(),
        ),
        alias(
            "tokenizer.pre_tokenizer",
            historical::pre_tokenizer.id(),
            current_tokenizer::pre_tokenizer.id(),
        ),
        alias(
            "tokenizer.post_processor",
            historical::post_processor.id(),
            current_tokenizer::post_processor.id(),
        ),
        alias(
            "tokenizer.decoder",
            historical::decoder.id(),
            current_tokenizer::decoder.id(),
        ),
        alias(
            "tokenizer.piece",
            historical::piece.id(),
            current_tokenizer::piece.id(),
        ),
        alias(
            "tokenizer.token_id",
            historical::token_id.id(),
            current_tokenizer::token_id.id(),
        ),
        alias(
            "tokenizer.piece_bytes",
            historical::piece_bytes.id(),
            current_tokenizer::piece_bytes.id(),
        ),
        alias(
            "tokenizer.merge_left",
            historical::merge_left.id(),
            current_tokenizer::merge_left.id(),
        ),
        alias(
            "tokenizer.merge_right",
            historical::merge_right.id(),
            current_tokenizer::merge_right.id(),
        ),
        alias(
            "tokenizer.unk_token",
            historical::unk_token.id(),
            current_tokenizer::unk_token.id(),
        ),
        alias(
            "tokenizer.continuing_subword_prefix",
            historical::continuing_subword_prefix.id(),
            current_tokenizer::continuing_subword_prefix.id(),
        ),
        alias(
            "tokenizer.end_of_word_suffix",
            historical::end_of_word_suffix.id(),
            current_tokenizer::end_of_word_suffix.id(),
        ),
        alias(
            "tokenizer.max_input_chars",
            historical::max_input_chars.id(),
            current_tokenizer::max_input_chars.id(),
        ),
        alias(
            "tokenizer.piece_score",
            historical::piece_score.id(),
            current_tokenizer::piece_score.id(),
        ),
        alias(
            "tokenizer.byte_fallback",
            historical::byte_fallback.id(),
            current_tokenizer::byte_fallback.id(),
        ),
        alias(
            "tokenizer.pattern",
            historical::pattern.id(),
            current_tokenizer::pattern.id(),
        ),
        alias(
            "tokenizer.replace_content",
            historical::replace_content.id(),
            current_tokenizer::replace_content.id(),
        ),
        alias(
            "tokenizer.behavior",
            historical::behavior.id(),
            current_tokenizer::behavior.id(),
        ),
    ]
}

/// Add canonical attribute aliases for every matching historical model fact.
///
/// The result is strictly additive: every input trible is retained, unknown
/// attributes are untouched, and an alias is inserted only when the exact
/// `(entity, canonical attribute, value)` trible is missing. Entity and value
/// bytes are copied unchanged. Running the projection over its own result is
/// therefore idempotent and reports zero additions.
pub fn project_legacy_model_attributes(facts: &TribleSet) -> ModelAttributeProjection {
    let aliases = legacy_model_attribute_aliases();
    let by_historical: HashMap<Id, usize> = aliases
        .iter()
        .enumerate()
        .map(|(index, alias)| (alias.historical, index))
        .collect();
    let mut mappings: Vec<_> = aliases
        .into_iter()
        .map(|alias| ModelAttributeAliasCounts {
            alias,
            historical_facts: 0,
            aliases_added: 0,
            aliases_already_present: 0,
        })
        .collect();
    let mut projected = facts.clone();
    let mut historical_facts = 0;
    let mut aliases_added = 0;

    for fact in facts {
        let Some(&mapping_index) = by_historical.get(fact.a()) else {
            continue;
        };
        let mapping = &mut mappings[mapping_index];
        mapping.historical_facts += 1;
        historical_facts += 1;

        let alias = Trible::force(
            fact.e(),
            &mapping.alias.canonical,
            fact.v::<UnknownInline>(),
        );
        if projected.contains(&alias) {
            mapping.aliases_already_present += 1;
        } else {
            projected.insert(&alias);
            mapping.aliases_added += 1;
            aliases_added += 1;
        }
    }

    ModelAttributeProjection {
        facts: projected,
        input_facts: facts.len(),
        historical_facts,
        aliases_added,
        mappings,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use anybytes::{Bytes, View};
    use triblespace::core::capability::{
        CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
        CapabilityResource,
    };
    use triblespace::core::collection::ACTION_WRITE;
    use triblespace::core::repo::pile::WantRewritePolicy;
    use triblespace::core::repo::{BlobStoreGet, RetentionRoots};
    use triblespace::macros::id_hex;
    use triblespace::prelude::blobencodings::{RawBytes, SimpleArchive, UTF8String};
    use triblespace::prelude::inlineencodings::Handle;

    static NEXT_TEMP_PILE: AtomicU64 = AtomicU64::new(0);

    struct TempPilePath {
        path: PathBuf,
    }

    impl TempPilePath {
        fn new(label: &str) -> Self {
            let ordinal = NEXT_TEMP_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-{label}-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create isolated test pile");
            Self { path }
        }

        fn as_path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempPilePath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn fragment_fixture(
        label: &str,
    ) -> (
        Fragment,
        Inline<Handle<UTF8String>>,
        Inline<Handle<RawBytes>>,
    ) {
        let text: Blob<UTF8String> = format!("model attachment {label}").to_blob();
        let text_handle = text.get_handle();
        let mut fragment = entity! { crate::format::attrs::model_name: text };

        let payload: Blob<RawBytes> = label.as_bytes().to_vec().to_blob();
        let payload_handle = payload.get_handle();
        let description = entity! { crate::tokenizer::attrs::piece_bytes: payload };
        fragment.describe_with(description);

        (fragment, text_handle, payload_handle)
    }

    /// One team for every test in this module.
    ///
    /// Authors vary between tests on purpose -- mixed-author admission is a
    /// property worth testing -- but they all publish into the same
    /// collection, which is exactly what having a team distinct from the
    /// signer is for.
    fn test_team() -> VerifyingKey {
        test_authority().verifying_key()
    }

    fn test_authority() -> SigningKey {
        SigningKey::from_bytes(&[0x11; 32])
    }

    fn grant_writer(
        pile: &mut Pile,
        collection: triblespace::core::collection::CollectionHandle,
        writer: &SigningKey,
    ) {
        if writer.verifying_key() == test_team() {
            return;
        }
        let claim = CapabilityClaim::root(
            CapabilityAtom::new(
                CapabilityAction::new(ACTION_WRITE),
                CapabilityResource::from(collection),
            ),
            CapabilityMode::Invoke,
            None,
        );
        let bundle =
            CapabilityProofBundle::issue_root(&test_authority(), claim, writer.verifying_key())
                .expect("issue test writer grant");
        for claim in bundle.claims() {
            pile.put::<SimpleArchive, _>(claim.clone())
                .expect("store test grant claim");
        }
        triblespace::core::repo::CapabilityProofStore::insert_proof(pile, bundle.proof().clone())
            .expect("store test writer grant");
    }

    fn open_test_pile(path: &Path) -> Pile {
        let mut pile = Pile::open(path).expect("open test pile");
        pile.refresh().expect("refresh test pile");
        pile
    }

    fn claims_for(pile: &mut Pile, cover: &FactCover) -> Vec<CollectionCommit> {
        let store = pile.snapshot().expect("freeze test observation");
        let mut claims = cover.claims(&store).expect("query cover provenance");
        claims.sort_unstable_by_key(CollectionCommit::id);
        claims
    }

    /// The identity of the model-graph collection must not drift silently.
    ///
    /// It is now a function of the team as well as the name, so the pinned
    /// bytes are pinned *for a stated team*. A fixed test key makes that
    /// explicit rather than leaving the reader to wonder whose collection the
    /// constant describes.
    #[test]
    fn model_graph_descriptor_is_stable() {
        let team = SigningKey::from_bytes(&[0x11; 32]).verifying_key();
        let collection = model_graph_collection(team);
        let descriptor = collection.descriptor();

        assert_eq!(mary_model_graph_name(), "mary-model-graph");
        assert_eq!(collection.name(), mary_model_graph_name());
        assert_eq!(collection.authority(), team);

        let handle = IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle();
        assert_eq!(
            handle.raw,
            [
                0x17, 0x52, 0xC1, 0xAE, 0xC0, 0xE5, 0x39, 0x12, 0xA8, 0xE3, 0x77, 0x07, 0xC0, 0x63,
                0xEC, 0xDA, 0xA1, 0xB6, 0x9B, 0x08, 0xBC, 0x40, 0x5D, 0x82, 0xD9, 0x8A, 0x67, 0xAF,
                0xCD, 0xDE, 0xEA, 0x18,
            ]
        );
    }

    #[test]
    fn a_bundle_only_pile_is_readable_through_the_model_loader() {
        let file = TempPilePath::new("bundle-only-loader");
        let signer = SigningKey::from_bytes(&[0x71; 32]);
        let (model, _text, _payload) = fragment_fixture("bundle-only");
        let model_root = model.root().expect("fixture model root");
        let model_facts = model.facts().len();

        let mut pile = open_test_pile(file.as_path());
        grant_writer(
            &mut pile,
            model_bundle_collection(test_team()).collection().handle(),
            &signer,
        );
        publish_model_bundle_fragment(&mut pile, test_team(), &signer, model_root, model)
            .expect("publish the sole bundle");
        pile.close().expect("close after publishing the bundle");

        // No `mary-model-graph` is ever published here, which is exactly the
        // shape the 2026-08-21 bundle migration leaves a pile in. Reading only
        // that one name reported "no collection named `mary-model-graph` in
        // this pile" about a pile whose complete model archive is right there,
        // and cost two lanes a night before the loader was taught the second
        // shape.
        let source = crate::persist::read_model_pile(file.as_path())
            .expect("a bundle-only pile is a readable model pile");
        assert!(
            source.facts.iter().any(|row| row.e() == &model_root),
            "the bundle's model root is missing from the facts the loader returned"
        );
        assert!(
            source.facts.len() >= model_facts,
            "loader returned {} facts, fewer than the {model_facts} the bundle archive states",
            source.facts.len()
        );
        assert_eq!(
            source.collections.len(),
            1,
            "a bundle-only pile must report exactly its bundle collection"
        );
        assert_eq!(
            source.collections[0].shape,
            crate::persist::ModelPileCollectionShape::Bundle,
        );
        assert_eq!(
            source.collections[0].authority,
            test_team(),
            "a bundle-only pile must report the bundle's own team as its authority"
        );
        assert_eq!(
            source.collections[0].collection.handle(),
            model_bundle_collection(test_team()).collection().handle(),
        );
    }

    #[test]
    fn mixed_graph_and_bundle_shapes_share_one_complete_reader() {
        let file = TempPilePath::new("mixed-shape-loader");
        let graph_writer = SigningKey::from_bytes(&[0x72; 32]);
        let bundle_authority = SigningKey::from_bytes(&[0x73; 32]);
        let bundle_team = bundle_authority.verifying_key();
        let (graph, graph_text, _) = fragment_fixture("mixed-graph");
        let graph_root = graph.root().expect("graph root");
        let (bundle, bundle_text, _) = fragment_fixture("mixed-bundle");
        let bundle_root = bundle.root().expect("bundle root");

        let mut pile = open_test_pile(file.as_path());
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &graph_writer,
        );
        publish_model_fragment(&mut pile, test_team(), &graph_writer, graph)
            .expect("publish graph shape");
        publish_model_bundle_fragment(
            &mut pile,
            bundle_team,
            &bundle_authority,
            bundle_root,
            bundle,
        )
        .expect("publish bundle shape");
        pile.close().expect("close mixed-shape pile");

        let source = crate::persist::read_model_pile(file.as_path())
            .expect("both independently authorized shapes are readable together");
        assert!(source.facts.iter().any(|row| row.e() == &graph_root));
        assert!(source.facts.iter().any(|row| row.e() == &bundle_root));
        assert_eq!(
            source
                .collections
                .iter()
                .map(|source| (source.shape, source.authority))
                .collect::<Vec<_>>(),
            vec![
                (crate::persist::ModelPileCollectionShape::Graph, test_team(),),
                (
                    crate::persist::ModelPileCollectionShape::Bundle,
                    bundle_team,
                ),
            ],
            "the read must retain both independently authorized collection identities"
        );
        assert_eq!(
            source.collections[0].collection.handle(),
            model_graph_collection(test_team()).collection().handle(),
        );
        assert_eq!(
            source.collections[1].collection.handle(),
            model_bundle_collection(bundle_team).collection().handle(),
        );

        let graph_label: View<str> = source.store.get(graph_text).expect("graph attachment");
        let bundle_label: View<str> = source
            .store
            .get(bundle_text)
            .expect("bundle attachment through the same observation");
        assert_eq!(&*graph_label, "model attachment mixed-graph");
        assert_eq!(&*bundle_label, "model attachment mixed-bundle");
    }

    #[test]
    fn one_valid_shape_does_not_mask_ambiguity_in_the_other() {
        let file = TempPilePath::new("mixed-shape-ambiguity");
        let graph_writer = SigningKey::from_bytes(&[0x74; 32]);
        let bundle_authority = SigningKey::from_bytes(&[0x75; 32]);
        let other_bundle_authority = SigningKey::from_bytes(&[0x76; 32]);
        let (graph, _, _) = fragment_fixture("unambiguous-graph");
        let (first_bundle, _, _) = fragment_fixture("first-bundle-team");
        let first_root = first_bundle.root().expect("first bundle root");
        let (second_bundle, _, _) = fragment_fixture("second-bundle-team");
        let second_root = second_bundle.root().expect("second bundle root");

        let mut pile = open_test_pile(file.as_path());
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &graph_writer,
        );
        publish_model_fragment(&mut pile, test_team(), &graph_writer, graph)
            .expect("publish unambiguous graph");
        publish_model_bundle_fragment(
            &mut pile,
            bundle_authority.verifying_key(),
            &bundle_authority,
            first_root,
            first_bundle,
        )
        .expect("publish first bundle authority");
        publish_model_bundle_fragment(
            &mut pile,
            other_bundle_authority.verifying_key(),
            &other_bundle_authority,
            second_root,
            second_bundle,
        )
        .expect("publish second bundle authority");
        pile.close().expect("close ambiguous mixed-shape pile");

        let error = crate::persist::read_model_pile(file.as_path())
            .err()
            .expect("bundle ambiguity must not be hidden by the valid graph");
        assert!(
            error
                .to_string()
                .contains("2 teams publish `mary-model-bundles`"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn model_bundle_is_one_signed_row_and_recursively_holds_the_model() {
        let file = TempPilePath::new("model-bundle-token");
        let signer = SigningKey::from_bytes(&[0x52; 32]);
        let other_signer = SigningKey::from_bytes(&[0x53; 32]);
        let (model, text_handle, payload_handle) = fragment_fixture("bundle");
        let model_root = model.root().expect("fixture model root");
        let model_facts = model.facts().clone();
        let model_metafacts = model.metafacts().clone();
        let prepared =
            prepare_model_bundle_fragment(test_team(), model_root, model.clone()).unwrap();
        assert_eq!(prepared.model_root(), model_root);
        let model_archive_data = prepared.model_archive_data();

        let mut pile = open_test_pile(file.as_path());
        let bundle_collection = model_bundle_collection(test_team()).collection().handle();
        grant_writer(&mut pile, bundle_collection, &signer);
        grant_writer(&mut pile, bundle_collection, &other_signer);
        let first = publish_model_bundle_fragment(
            &mut pile,
            test_team(),
            &signer,
            model_root,
            model.clone(),
        )
        .unwrap();
        let len_after_first = std::fs::metadata(file.as_path()).unwrap().len();
        let repeated = publish_model_bundle_fragment(
            &mut pile,
            test_team(),
            &signer,
            model_root,
            model.clone(),
        )
        .unwrap();
        assert_eq!(first, repeated);
        assert_eq!(
            std::fs::metadata(file.as_path()).unwrap().len(),
            len_after_first,
            "exact retry must append nothing"
        );
        let other_author =
            publish_model_bundle_fragment(&mut pile, test_team(), &other_signer, model_root, model)
                .unwrap();
        assert_ne!(first.id(), other_author.id());
        assert_eq!(first.data(), other_author.data());
        assert!(model_graph_teams(&mut pile).unwrap().is_empty());

        let store = pile.snapshot().expect("freeze test observation");
        let cover = local_model_bundle_cover(&store, test_team()).unwrap();
        let snapshot = snapshot_model_bundle_collection_exact(&store, test_team(), &cover).unwrap();
        assert_eq!(snapshot.facts().len(), 1);
        let mut expected_commits = vec![first, other_author];
        expected_commits.sort_unstable_by_key(CollectionCommit::id);
        assert_eq!(snapshot.cover(), &cover);
        assert_eq!(snapshot.cover().len(), 1);
        assert_eq!(claims_for(&mut pile, &cover), expected_commits);
        let fact = snapshot.facts().iter().next().unwrap();
        assert_eq!(fact.e(), &model_root);
        assert_eq!(fact.a(), &metadata::archive.id());
        assert_eq!(
            inlineencodings::Handle::<SimpleArchive>::to_hash(
                *fact.v::<inlineencodings::Handle<SimpleArchive>>()
            ),
            model_archive_data
        );

        let bundle_token_blob: Blob<SimpleArchive> = snapshot
            .store()
            .get(inlineencodings::Handle::<SimpleArchive>::from_hash(
                first.data(),
            ))
            .unwrap();
        assert_eq!(
            TribleSet::try_from_blob(bundle_token_blob).unwrap(),
            *snapshot.facts()
        );
        let source: Blob<SimpleArchive> = snapshot
            .store()
            .get(inlineencodings::Handle::<SimpleArchive>::from_hash(
                model_archive_data,
            ))
            .unwrap();
        assert_eq!(TribleSet::try_from_blob(source).unwrap(), model_facts);
        let metadata: Blob<SimpleArchive> = snapshot.store().get(first.metadata()).unwrap();
        let metadata = TribleSet::try_from_blob(metadata).unwrap();
        assert!(model_metafacts.iter().all(|fact| metadata.contains(fact)));
        snapshot.store().get::<View<str>, _>(text_handle).unwrap();
        snapshot.store().get::<Bytes, _>(payload_handle).unwrap();

        assert_eq!(
            local_model_bundle_cover(&pile.snapshot().unwrap(), test_team()).unwrap(),
            cover
        );
        let (observed_team, observed) =
            snapshot_sole_model_bundle_collection_local_latest(&mut pile).unwrap();
        assert_eq!(observed_team, test_team());
        assert_eq!(observed.cover(), &cover);
        drop(observed);
        drop(snapshot);
        pile.close().unwrap();
    }

    #[test]
    fn sole_bundle_snapshot_rejects_multiple_teams_in_its_frozen_prefix() {
        let file = TempPilePath::new("model-bundle-team-ambiguity");
        let signer = SigningKey::from_bytes(&[0x54; 32]);
        let other_authority = SigningKey::from_bytes(&[0x55; 32]);
        let other_team = other_authority.verifying_key();
        let (model, _, _) = fragment_fixture("team-ambiguity");
        let model_root = model.root().unwrap();
        let mut pile = open_test_pile(file.as_path());
        grant_writer(
            &mut pile,
            model_bundle_collection(test_team()).collection().handle(),
            &signer,
        );
        publish_model_bundle_fragment(&mut pile, test_team(), &signer, model_root, model.clone())
            .unwrap();
        publish_model_bundle_fragment(&mut pile, other_team, &other_authority, model_root, model)
            .unwrap();

        let error = match snapshot_sole_model_bundle_collection_local_latest(&mut pile) {
            Ok(_) => panic!("two bundle teams must be ambiguous"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                SnapshotSoleModelBundleError::Team(SoleModelBundleTeamError::Several { .. })
            ),
            "{error}"
        );
        pile.close().unwrap();
    }

    #[test]
    fn unauthorized_foreign_bundle_claim_does_not_poison_team_discovery() {
        let file = TempPilePath::new("model-bundle-unauthorized-team");
        let signer = SigningKey::from_bytes(&[0x56; 32]);
        let foreign_authority = SigningKey::from_bytes(&[0x57; 32]).verifying_key();
        let (model, _, _) = fragment_fixture("unauthorized-team");
        let model_root = model.root().unwrap();
        let mut pile = open_test_pile(file.as_path());

        grant_writer(
            &mut pile,
            model_bundle_collection(test_team()).collection().handle(),
            &signer,
        );
        publish_model_bundle_fragment(&mut pile, test_team(), &signer, model_root, model.clone())
            .unwrap();

        // Local publication is deliberately unconditional. This validly
        // signed claim names another authority's collection, but without a
        // capability from that authority it is inert at the read boundary.
        publish_model_bundle_fragment(&mut pile, foreign_authority, &signer, model_root, model)
            .unwrap();

        assert_eq!(model_bundle_teams(&mut pile).unwrap(), vec![test_team()]);
        assert!(
            local_model_bundle_cover(&pile.snapshot().unwrap(), foreign_authority)
                .unwrap()
                .is_empty()
        );
        let (team, snapshot) =
            snapshot_sole_model_bundle_collection_local_latest(&mut pile).unwrap();
        assert_eq!(team, test_team());
        assert_eq!(snapshot.cover().len(), 1);
        drop(snapshot);
        pile.close().unwrap();
    }

    #[test]
    fn existing_collection_requires_an_admitted_writer_before_selection() {
        let file = TempPilePath::new("model-writer-preflight");
        let authority = test_authority();
        let delegate = SigningKey::from_bytes(&[0x58; 32]);
        let outsider = SigningKey::from_bytes(&[0x59; 32]);
        let (model, _, _) = fragment_fixture("writer-preflight");
        let mut pile = open_test_pile(file.as_path());
        publish_model_fragment(&mut pile, test_team(), &authority, model).unwrap();

        let len_before = std::fs::metadata(file.as_path()).unwrap().len();
        let error = model_graph_team_or_own(&mut pile, &outsider).unwrap_err();
        assert!(matches!(
            error,
            ModelCollectionWriterSelectionError::NotAdmitted {
                authority,
                signer,
                ..
            } if authority == test_team().to_bytes()
                && signer == outsider.verifying_key().to_bytes()
        ));
        assert_eq!(
            std::fs::metadata(file.as_path()).unwrap().len(),
            len_before,
            "writer selection must not publish an inert claim"
        );

        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &delegate,
        );
        assert_eq!(
            model_graph_team_or_own(&mut pile, &delegate).unwrap(),
            test_team()
        );
        pile.close().unwrap();
    }

    /// The seqlock this replaced (`observe_stable_pile`) existed to detect an
    /// append that landed mid-observation and retry. A frozen snapshot cannot
    /// see one at all, so the retry has nothing to retry: this asserts the
    /// stronger property directly — an interleaved append is invisible to the
    /// already-frozen observation, and a later snapshot both sees it and
    /// reports it through `changes_since`.
    #[test]
    fn a_frozen_observation_does_not_see_an_interleaved_append() {
        use triblespace::core::repo::{StoreChanges, StoreSnapshot};

        let file = TempPilePath::new("stable-observation");
        let mut pile = open_test_pile(file.as_path());
        let mut appender = open_test_pile(file.as_path());

        let before = pile.snapshot().expect("freeze first observation");
        let handle = appender
            .put::<RawBytes, _>(b"interleaved".to_vec().to_blob())
            .unwrap();
        appender.flush().unwrap();

        assert!(
            before.get::<Blob<RawBytes>, _>(handle).is_err(),
            "a frozen observation must not acquire blobs appended after it"
        );

        let after = pile.snapshot().expect("freeze second observation");
        after
            .get::<Blob<RawBytes>, _>(handle)
            .expect("a later observation sees the append");
        assert!(
            after.changes_since(&before).contains(StoreChanges::BLOBS),
            "the append must be reported as a blob change"
        );
        assert!(
            after.changes_since(&after).is_empty(),
            "an observation is unchanged against itself"
        );

        appender.close().unwrap();
        pile.close().unwrap();
    }

    #[test]
    fn retained_rewrite_follows_bundle_token_into_model_archive_and_attachments() {
        let source_file = TempPilePath::new("model-bundle-rewrite-source");
        let destination_file = TempPilePath::new("model-bundle-rewrite-destination");
        let signer = SigningKey::from_bytes(&[0x5A; 32]);
        let (model, text_handle, payload_handle) = fragment_fixture("retained");
        let model_root = model.root().expect("fixture model root");
        let prepared =
            prepare_model_bundle_fragment(test_team(), model_root, model.clone()).unwrap();
        let model_archive_data = prepared.model_archive_data();

        let mut source = open_test_pile(source_file.as_path());
        grant_writer(
            &mut source,
            model_bundle_collection(test_team()).collection().handle(),
            &signer,
        );
        publish_model_bundle_fragment(&mut source, test_team(), &signer, model_root, model)
            .unwrap();
        let orphan: Blob<RawBytes> = b"deliberate orphan".to_vec().to_blob();
        let orphan_handle = source.put::<RawBytes, _>(orphan).unwrap();

        let mut destination = open_test_pile(destination_file.as_path());
        source
            .rewrite_retained_into(
                &mut destination,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .unwrap();

        // The signed COMMIT recursively owns T, T names H, and H's facts name
        // both fixture attachments. None is an explicit rewrite root.
        // Two piles, so deliberately two observations: the cover is discovered
        // in the SOURCE and replayed against the DESTINATION. Making both
        // explicit is the point — the exact cover is the only thing that
        // crosses, and it now names which observation each half came from.
        let source_store = source.snapshot().expect("freeze source observation");
        let cover = local_model_bundle_cover(&source_store, test_team()).unwrap();
        let destination_store = destination
            .snapshot()
            .expect("freeze destination observation");
        let snapshot =
            snapshot_model_bundle_collection_exact(&destination_store, test_team(), &cover)
                .unwrap();
        assert_eq!(snapshot.cover(), &cover);
        assert_eq!(snapshot.facts().len(), 1);
        let token = snapshot.facts().iter().next().unwrap();
        assert_eq!(token.e(), &model_root);
        assert_eq!(token.a(), &metadata::archive.id());
        assert_eq!(
            inlineencodings::Handle::<SimpleArchive>::to_hash(
                *token.v::<inlineencodings::Handle<SimpleArchive>>()
            ),
            model_archive_data
        );
        snapshot
            .store()
            .get::<Blob<SimpleArchive>, _>(inlineencodings::Handle::<SimpleArchive>::from_hash(
                model_archive_data,
            ))
            .unwrap();
        snapshot.store().get::<View<str>, _>(text_handle).unwrap();
        snapshot.store().get::<Bytes, _>(payload_handle).unwrap();
        assert!(snapshot.store().get::<Bytes, _>(orphan_handle).is_err());

        drop(snapshot);
        destination.close().unwrap();
        source.close().unwrap();
    }

    #[test]
    fn model_bundle_preparation_is_pure_and_rejects_a_false_root() {
        let file = TempPilePath::new("model-bundle-prepare");
        let mut pile = open_test_pile(file.as_path());
        let len_before = std::fs::metadata(file.as_path()).unwrap().len();
        let actual = entity! { crate::format::attrs::kind: "present-root" };
        let false_root = *fucid();
        let fragment = Fragment::rooted(false_root, actual.into_facts());
        let error = prepare_model_bundle_fragment(test_team(), false_root, fragment).unwrap_err();
        assert!(matches!(error, PrepareModelBundleError::RootAbsent(root) if root == false_root));
        assert_eq!(std::fs::metadata(file.as_path()).unwrap().len(), len_before);
        assert!(model_bundle_teams(&mut pile).unwrap().is_empty());
        pile.close().unwrap();
    }

    #[test]
    fn staged_model_bundle_is_not_authority_until_finalize() {
        let file = TempPilePath::new("model-bundle-staged");
        let signer = SigningKey::from_bytes(&[0x54; 32]);
        let (model, text_handle, payload_handle) = fragment_fixture("staged");
        let model_root = model.root().unwrap();
        let prepared = prepare_model_bundle_fragment(test_team(), model_root, model).unwrap();
        let model_archive_data = prepared.model_archive_data();
        let mut pile = open_test_pile(file.as_path());
        let staged = prepared
            .into_prepared_commit()
            .stage(&mut pile, &signer)
            .unwrap();
        let withheld = *staged.commit();
        drop(staged);

        assert!(model_bundle_teams(&mut pile).unwrap().is_empty());
        let reader = pile.snapshot().unwrap();
        reader
            .get::<Blob<SimpleArchive>, _>(inlineencodings::Handle::<SimpleArchive>::from_hash(
                withheld.data(),
            ))
            .unwrap();
        reader
            .get::<Blob<SimpleArchive>, _>(inlineencodings::Handle::<SimpleArchive>::from_hash(
                model_archive_data,
            ))
            .unwrap();
        reader.get::<View<str>, _>(text_handle).unwrap();
        reader.get::<Bytes, _>(payload_handle).unwrap();
        pile.close().unwrap();
    }

    #[test]
    fn fragment_publication_roundtrips_every_channel_and_is_idempotent() {
        let file = TempPilePath::new("fragment-roundtrip");
        let signing_key = SigningKey::from_bytes(&[0x17; 32]);
        let (fragment, text_handle, payload_handle) = fragment_fixture("roundtrip");
        let expected_facts = fragment.facts().clone();
        let expected_metafacts = fragment.metafacts().clone();

        let mut pile = open_test_pile(file.as_path());
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &signing_key,
        );
        let first =
            publish_model_fragment(&mut pile, test_team(), &signing_key, fragment.clone()).unwrap();
        let repeated =
            publish_model_fragment(&mut pile, test_team(), &signing_key, fragment).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(
            first.public_key().raw,
            signing_key.verifying_key().to_bytes()
        );
        first.verify_strict().unwrap();

        // Snapshot directly from the same still-open pile. Duplicate claims
        // over one payload collapse to one cover member.
        let store = pile.snapshot().expect("freeze test observation");
        let cover = local_model_cover(&store, test_team()).unwrap();
        let snapshot = snapshot_model_collection_exact(&store, test_team(), &cover).unwrap();
        assert_eq!(snapshot.facts(), &expected_facts);
        assert_eq!(snapshot.cover().len(), 1);
        assert_eq!(claims_for(&mut pile, &cover), vec![first]);

        // The owned PileSnapshot mapping must outlive the mutable pile handle.
        pile.close().unwrap();
        let metadata: TribleSet = snapshot.store().get(first.metadata()).unwrap();
        assert_eq!(metadata, expected_metafacts);
        let text: View<str> = snapshot.store().get(text_handle).unwrap();
        let payload: Bytes = snapshot.store().get(payload_handle).unwrap();
        assert_eq!(&*text, "model attachment roundtrip");
        assert_eq!(&*payload, b"roundtrip");

        let loaded = load_model_collection_from_cover(file.as_path(), test_team(), &cover).unwrap();
        assert_eq!(loaded.facts(), &expected_facts);
        let text_after_path_close: View<str> = loaded.store().get(text_handle).unwrap();
        assert_eq!(&*text_after_path_close, "model attachment roundtrip");
    }

    #[test]
    fn selected_model_index_owns_the_reader_after_snapshot_consumption() {
        let file = TempPilePath::new("selected-model-index");
        let signing_key = SigningKey::from_bytes(&[0x19; 32]);
        let mut pile = open_test_pile(file.as_path());
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &signing_key,
        );

        let leaf = crate::format::put_raw(&mut pile, &[1.25], &[1]).unwrap();
        let leaf_id = leaf.root().expect("tensor leaf root");
        let mut facts = leaf.into_facts();
        let name = pile
            .put::<UTF8String, _>("encoder.weight".to_owned())
            .unwrap();
        let member = entity! { _ @
            crate::format::attrs::safetensor_path: name,
            crate::format::attrs::weight: leaf_id,
        };
        let member_id = member.root().expect("model member root");
        facts += member.into_facts();
        let source = pile
            .put::<UTF8String, _>("example/owned-index".to_owned())
            .unwrap();
        let model = entity! { _ @
            crate::format::attrs::source: source,
            crate::format::attrs::quantization: "native",
            crate::format::attrs::member: member_id,
        };
        let model_root = model.root().expect("model root");
        facts += model.into_facts();

        publish_model_fragment(
            &mut pile,
            test_team(),
            &signing_key,
            Fragment::rooted(model_root, facts),
        )
        .unwrap();
        let store = pile.snapshot().expect("freeze test observation");
        let cover = local_model_cover(&store, test_team()).unwrap();
        let snapshot = snapshot_model_collection_exact(&store, test_team(), &cover).unwrap();
        pile.close().unwrap();

        let selected = crate::selection::SelectedModelIndex::from_snapshot(
            snapshot,
            crate::selection::ModelSelector::Source {
                source: "example/owned-index",
                quantization: "native",
            },
        )
        .unwrap();
        assert_eq!(selected.single_root(), Some(model_root));
        let leaf = &selected.handles()["encoder.weight"];
        assert_eq!(leaf.elem(), crate::leaf::Elem::F32);
        let data: View<[f32]> = leaf.view_f32().expect("f32 leaves serve a view");
        assert_eq!(&*data, &[1.25]);
        assert_eq!(leaf.dims(), &[1]);
    }

    #[test]
    fn exact_cover_accepts_mixed_authors_and_keeps_later_members_inert() {
        let file = TempPilePath::new("exact-mixed-authors");
        let mut pile = open_test_pile(file.as_path());
        let (first_fragment, _, _) = fragment_fixture("first");
        let first_facts = first_fragment.facts().clone();
        let first_signer = SigningKey::from_bytes(&[0x21; 32]);
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &first_signer,
        );
        let first =
            publish_model_fragment(&mut pile, test_team(), &first_signer, first_fragment).unwrap();
        let (second_fragment, _, _) = fragment_fixture("second");
        let second_facts = second_fragment.facts().clone();
        let second_signer = SigningKey::from_bytes(&[0x22; 32]);
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &second_signer,
        );
        let second =
            publish_model_fragment(&mut pile, test_team(), &second_signer, second_fragment)
                .unwrap();
        let cover = local_model_cover(
            &pile.snapshot().expect("freeze pre-append observation"),
            test_team(),
        )
        .unwrap();
        let (unselected, _, _) = fragment_fixture("unselected");
        let unselected_signer = SigningKey::from_bytes(&[0x23; 32]);
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &unselected_signer,
        );
        publish_model_fragment(&mut pile, test_team(), &unselected_signer, unselected).unwrap();

        // Deliberately a LATER observation than the one the cover came from:
        // the unselected commit is resident in it, and materializing the
        // earlier cover against it must still ignore that commit. Under the
        // old API this mixture was accidental; here it is stated.
        let snapshot = snapshot_model_collection_exact(
            &pile.snapshot().expect("freeze post-append observation"),
            test_team(),
            &cover,
        )
        .unwrap();
        let mut expected = first_facts;
        expected += second_facts;
        let mut expected_commits = vec![first, second];
        expected_commits.sort_unstable_by_key(CollectionCommit::id);
        assert_eq!(snapshot.facts(), &expected);
        assert_eq!(snapshot.cover(), &cover);
        assert_eq!(snapshot.cover().len(), 2);
        assert_eq!(claims_for(&mut pile, &cover), expected_commits);
        assert_ne!(first.public_key(), second.public_key());
        pile.close().unwrap();
    }

    #[test]
    fn local_latest_admits_all_matching_authors_and_ignores_foreign_records() {
        let file = TempPilePath::new("local-latest");
        let mut pile = open_test_pile(file.as_path());
        let (first_fragment, _, _) = fragment_fixture("local-first");
        let first_facts = first_fragment.facts().clone();
        let first_signer = SigningKey::from_bytes(&[0x31; 32]);
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &first_signer,
        );
        let first =
            publish_model_fragment(&mut pile, test_team(), &first_signer, first_fragment).unwrap();
        let (second_fragment, _, _) = fragment_fixture("local-second");
        let second_facts = second_fragment.facts().clone();
        let second_signer = SigningKey::from_bytes(&[0x32; 32]);
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &second_signer,
        );
        let second =
            publish_model_fragment(&mut pile, test_team(), &second_signer, second_fragment)
                .unwrap();

        // "Foreign" now means a different name under the same team, which is
        // the shape a real unrelated collection takes.
        let foreign_name = "not-the-model-graph";
        // Private, like both production descriptors: reach participates in a
        // descriptor's identity, so a public foreign one would differ from the
        // model graph by two things and stop isolating the one under test.
        let foreign_descriptor =
            simplearchive_union::descriptor(foreign_name, test_team(), reach::private());
        let (foreign_fragment, _, _) = fragment_fixture("foreign");
        let foreign = simplearchive_union::publish_fragment_commit(
            &mut pile,
            &foreign_descriptor,
            foreign_fragment,
            &SigningKey::from_bytes(&[0x33; 32]),
        )
        .unwrap();
        let mut invalid_foreign_bytes = foreign.to_bytes();
        *invalid_foreign_bytes.last_mut().unwrap() ^= 1;
        let invalid_foreign = CollectionCommit::from_bytes(invalid_foreign_bytes);
        assert!(invalid_foreign.verify_strict().is_err());
        pile.insert(CollectionRecord::Commit(invalid_foreign))
            .unwrap();

        let in_place = snapshot_model_collection_local_latest(&mut pile, test_team()).unwrap();
        let mut expected = first_facts.clone();
        expected += second_facts.clone();
        let mut expected_commits = vec![first, second];
        expected_commits.sort_unstable_by_key(CollectionCommit::id);
        assert_eq!(in_place.facts(), &expected);
        assert_eq!(claims_for(&mut pile, in_place.cover()), expected_commits);
        assert_eq!(in_place.cover().collection().handle(), first.collection());
        assert_ne!(in_place.cover().collection().handle(), foreign.collection());
        let expected_cover = in_place.cover().clone();
        pile.close().unwrap();

        let snapshot = load_model_collection_local_latest(file.as_path(), test_team()).unwrap();
        assert_eq!(snapshot.facts(), &expected);
        assert_eq!(snapshot.cover(), &expected_cover);

        let (team, sole_snapshot) =
            load_sole_model_collection_local_latest(file.as_path()).unwrap();
        assert_eq!(team, test_team());
        assert_eq!(sole_snapshot.facts(), &expected);
        assert_eq!(sole_snapshot.cover(), &expected_cover);
    }

    #[test]
    fn sole_model_snapshot_rejects_multiple_teams_in_its_frozen_prefix() {
        let file = TempPilePath::new("model-graph-team-ambiguity");
        let signer = SigningKey::from_bytes(&[0x34; 32]);
        let other_authority = SigningKey::from_bytes(&[0x35; 32]);
        let other_team = other_authority.verifying_key();
        let (first, _, _) = fragment_fixture("first-team");
        let (second, _, _) = fragment_fixture("second-team");
        let mut pile = open_test_pile(file.as_path());
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &signer,
        );
        publish_model_fragment(&mut pile, test_team(), &signer, first).unwrap();
        publish_model_fragment(&mut pile, other_team, &other_authority, second).unwrap();

        let error = match snapshot_sole_model_collection_local_latest(&mut pile) {
            Ok(_) => panic!("two model-graph teams must be ambiguous"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                SnapshotSoleModelGraphError::Team(SoleModelGraphTeamError::Several { .. })
            ),
            "{error}"
        );
        pile.close().unwrap();
    }

    #[test]
    fn local_cover_ignores_invalid_matching_claims() {
        let file = TempPilePath::new("local-invalid-matching");
        let mut pile = open_test_pile(file.as_path());
        let (fragment, _, _) = fragment_fixture("invalid-matching");
        let signer = SigningKey::from_bytes(&[0x41; 32]);
        grant_writer(
            &mut pile,
            model_graph_collection(test_team()).collection().handle(),
            &signer,
        );
        let valid = publish_model_fragment(&mut pile, test_team(), &signer, fragment).unwrap();
        let mut invalid_bytes = valid.to_bytes();
        *invalid_bytes.last_mut().unwrap() ^= 1;
        let invalid = CollectionCommit::from_bytes(invalid_bytes);
        assert_eq!(invalid.collection(), valid.collection());
        assert!(invalid.verify_strict().is_err());
        pile.insert(CollectionRecord::Commit(invalid)).unwrap();
        pile.close().unwrap();

        let snapshot = load_model_collection_local_latest(file.as_path(), test_team()).unwrap();
        assert_eq!(snapshot.cover().len(), 1);
        let mut pile = open_test_pile(file.as_path());
        assert_eq!(claims_for(&mut pile, snapshot.cover()), vec![valid]);
        pile.close().unwrap();
    }

    #[test]
    fn corrupt_tail_is_reported_without_mutating_the_pile() {
        let file = TempPilePath::new("corrupt-tail");
        open_test_pile(file.as_path()).close().unwrap();
        OpenOptions::new()
            .append(true)
            .open(file.as_path())
            .unwrap()
            .write_all(&[0xAA; 7])
            .unwrap();
        let before = std::fs::metadata(file.as_path()).unwrap().len();

        let error = match load_model_collection_local_latest(file.as_path(), test_team()) {
            Ok(_) => panic!("corrupt pile unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(matches!(error, LoadModelCollectionError::Cover(_)));
        assert_eq!(std::fs::metadata(file.as_path()).unwrap().len(), before);
    }

    fn raw_fact(entity: Id, attribute: Id, value: u8) -> Trible {
        Trible::force(
            &entity,
            &attribute,
            &Inline::<UnknownInline>::new([value; 32]),
        )
    }

    fn mapping(label: &str) -> ModelAttributeAlias {
        legacy_model_attribute_aliases()
            .into_iter()
            .find(|mapping| mapping.label == label)
            .unwrap_or_else(|| panic!("missing mapping {label}"))
    }

    #[test]
    fn mapping_table_is_exhaustive_unique_and_encoding_aware() {
        let mappings = legacy_model_attribute_aliases();
        assert_eq!(mappings.len(), LEGACY_MODEL_ATTRIBUTE_ALIAS_COUNT);

        let labels: HashSet<_> = mappings.iter().map(|mapping| mapping.label).collect();
        let historical: HashSet<_> = mappings.iter().map(|mapping| mapping.historical).collect();
        let canonical: HashSet<_> = mappings.iter().map(|mapping| mapping.canonical).collect();
        assert_eq!(labels.len(), mappings.len(), "duplicate diagnostic label");
        assert_eq!(historical.len(), mappings.len(), "duplicate historical id");
        assert_eq!(canonical.len(), mappings.len(), "duplicate canonical id");
        assert!(
            historical.is_disjoint(&canonical),
            "canonical target is also a historical source"
        );
        assert!(
            mappings
                .iter()
                .all(|mapping| mapping.historical != mapping.canonical)
        );

        // Exhaustively pin the audited historical side. Shared declarations
        // occur once; piece_bytes is included because d6dcbd3a predates the
        // TribleSpace epoch transition.
        let expected_historical = [
            id_hex!("572B45D52A47608F283D0F778597137A"),
            id_hex!("467CCF3FDCCCCE599F6C1B933EACD933"),
            id_hex!("D09A91FC3F04C40AE4A42CD6628A9E38"),
            id_hex!("2ADC6462A7F70E230558C5D681E38768"),
            id_hex!("23178058559C762BB4B1FEAA36B3566D"),
            id_hex!("F9EA2FB90DC094D42A4845B013950032"),
            id_hex!("2CC4D16369C4980BCB512937DA204FF5"),
            id_hex!("4629D277AD6B52B50DA78DEF63440AF1"),
            id_hex!("18E898172078C843A0351C3D880CC238"),
            id_hex!("52C4A211D2A08BA25C27FFD79FF24C93"),
            id_hex!("09EA2F7BCF9B0C9714EE39CF269DF2D5"),
            id_hex!("33CE12B1B940B13E48D8E5B0ADFD2421"),
            id_hex!("3F46CDE630964D78D62DA32F4A8558C1"),
            id_hex!("B4B6EC08A0CD70DE63A690168EE78F0F"),
            id_hex!("4C1CD1611863E7854C59C7DC706DF77A"),
            id_hex!("D20B8E3556C35FF6D18D104C3443D6CF"),
            id_hex!("7AF87320C144AA29C29FE2A5EE7C7EB2"),
            id_hex!("1A682F45CE40171DD5C6FDB4F086AD69"),
            id_hex!("198B03AF556B7505CCC9ABD4A1D6E724"),
            id_hex!("B93C4E66F4B9553BF0E8B5DBAD116ECF"),
            id_hex!("FF8335C187823A267E26B4E33EF157E9"),
            id_hex!("7CD7F0DC8BDA328735A22DF02B4B8828"),
            id_hex!("1F21DAE68652A4D8CAD973400F04124D"),
            id_hex!("E7014108A8F9512B19E3E8272E8A71F9"),
            id_hex!("E839AA8F549C0D608FB86476A1EF3416"),
            id_hex!("E229769197BB035A2D6F61BC6A7D44BC"),
            id_hex!("B2553118F4CAAF1D028619956DE7F145"),
            id_hex!("53BAF87A0E7F1410F8212B3EDF2A498C"),
            id_hex!("6EEBF39CADD11B7CFBB624019AE21585"),
            id_hex!("98EC58B28F4D0BB43965DF7C5FF22713"),
            id_hex!("F3AAA4CD8EE04E5592059564A21FE953"),
            id_hex!("AE7FE29F2F38153F58C542D5CA4A9356"),
            id_hex!("F0E2E782F7BB62F52B1186DDE0EB5388"),
            id_hex!("714AE13F801202EB27C83E3AB2290669"),
            id_hex!("5723ECE1FF426C58879B79D5669A7CF1"),
            id_hex!("5C78FEB151F35A2C5D07BEC92E860752"),
            id_hex!("68F1A9E6ED735E7C3ADCCA076AFF1742"),
            id_hex!("11F76A2C0856C16CB030C4327D5A3B93"),
            id_hex!("6FB969E8A3EDD1A657C721DD5A7D42EA"),
            id_hex!("DF3F88DBFA2B44A7783169C9640014AF"),
            id_hex!("3BCB70478942DB710ED2A4FB023F3457"),
            id_hex!("EE4C6647619A836326196F0DBF84FA98"),
            id_hex!("C8262D5668B8A1F541B3C35D54201BEC"),
            id_hex!("3AC7574C07D02D389B4E7AD3B3B084D9"),
            id_hex!("964B4FCF7477E7E4436F0325F89B7CB5"),
        ];
        assert_eq!(
            mappings.map(|mapping| mapping.historical),
            expected_historical
        );

        // The unconditional runtime declarations are the authority for their
        // canonical side, including the shared format/tokenizer attributes.
        assert_eq!(
            mapping("format.data").canonical,
            crate::format::attrs::data.id()
        );
        assert_eq!(
            mapping("format.model_name").canonical,
            crate::tokenizer::attrs::model_name.id()
        );
        assert_eq!(
            mapping("tokenizer.piece_bytes").canonical,
            crate::tokenizer::attrs::piece_bytes.id()
        );
    }

    #[test]
    fn nomic_graph_projection_is_additive_byte_exact_and_idempotent() {
        let model = id_hex!("B509CC5B379B109D0EBAFA3549ABCD90");
        let leaf = id_hex!("BA90876DC53D2EBE37EBD9E98FC35C26");
        let tokenizer = id_hex!("7CD60DF13D297894E10257058114895A");
        let vocab_entry = id_hex!("00EF7BF679E27BFB7CA6AB4B78001A3C");

        // These are the schema ids observed in the Nomic model/tokenizer pile:
        // model membership and weight leaf metadata plus vocab piece/id facts.
        let legacy_facts = [
            raw_fact(model, mapping("format.member").historical, 0x11),
            raw_fact(model, mapping("format.model_name").historical, 0x12),
            raw_fact(leaf, mapping("format.data").historical, 0x21),
            raw_fact(leaf, mapping("format.shape").historical, 0x22),
            raw_fact(leaf, mapping("format.weight").historical, 0x23),
            raw_fact(model, mapping("tokenizer.tokenizer").historical, 0x31),
            raw_fact(tokenizer, mapping("tokenizer.vocab").historical, 0x32),
            raw_fact(vocab_entry, mapping("tokenizer.piece").historical, 0x33),
            raw_fact(vocab_entry, mapping("tokenizer.token_id").historical, 0x34),
        ];
        let mut input: TribleSet = legacy_facts.into_iter().collect();

        // One canonical alias already exists and must not be duplicated or
        // counted as an addition.
        let shape = legacy_facts[3];
        input.insert(&Trible::force(
            shape.e(),
            &mapping("format.shape").canonical,
            shape.v::<UnknownInline>(),
        ));

        let projected = project_legacy_model_attributes(&input);
        assert_eq!(projected.input_facts, input.len());
        assert_eq!(projected.historical_facts, legacy_facts.len());
        assert_eq!(projected.aliases_added, legacy_facts.len() - 1);
        assert_eq!(projected.facts.len(), input.len() + projected.aliases_added);
        assert!(input.iter().all(|fact| projected.facts.contains(fact)));

        for source in legacy_facts {
            let alias_mapping = legacy_model_attribute_aliases()
                .into_iter()
                .find(|mapping| mapping.historical == *source.a())
                .expect("Nomic fact must have an audited mapping");
            let alias = Trible::force(
                source.e(),
                &alias_mapping.canonical,
                source.v::<UnknownInline>(),
            );
            assert!(projected.facts.contains(&alias));
            assert_eq!(source.e(), alias.e());
            assert_eq!(&source.data[..16], &alias.data[..16]);
            assert_eq!(&source.data[32..], &alias.data[32..]);
        }

        let shape_counts = projected
            .mappings
            .iter()
            .find(|counts| counts.alias.label == "format.shape")
            .expect("shape diagnostics");
        assert_eq!(shape_counts.historical_facts, 1);
        assert_eq!(shape_counts.aliases_added, 0);
        assert_eq!(shape_counts.aliases_already_present, 1);

        let repeated = project_legacy_model_attributes(&projected.facts);
        assert_eq!(repeated.aliases_added, 0);
        assert_eq!(repeated.facts, projected.facts);
        assert_eq!(repeated.historical_facts, legacy_facts.len());
        assert_eq!(
            repeated
                .mappings
                .iter()
                .map(|counts| counts.aliases_already_present)
                .sum::<usize>(),
            legacy_facts.len()
        );
    }

    #[test]
    fn post_transition_inkling_and_dataset_attributes_are_not_remapped() {
        let entity = id_hex!("69070B055FB712EE517E716BFC3CA728");
        let excluded = [
            // Inkling attributes introduced 2026-08-10, after 6b65f278.
            id_hex!("0B51DA3E67216213871743E045590DBC"),
            id_hex!("A6ED6DBA4BE63E4E34F2787DA84AD860"),
            id_hex!("BCDDFBCFF89F67EE0B1E527C4872CED7"),
            // Dataset/training compatibility is a separate schema migration.
            id_hex!("8644CC9146EA9348DB5CF401CD183724"),
            id_hex!("806AF895E3D21D3147908D36D542F367"),
        ];
        let input: TribleSet = excluded
            .into_iter()
            .enumerate()
            .map(|(index, attribute)| raw_fact(entity, attribute, index as u8 + 1))
            .collect();

        let projected = project_legacy_model_attributes(&input);
        assert_eq!(projected.historical_facts, 0);
        assert_eq!(projected.aliases_added, 0);
        assert_eq!(projected.facts, input);
        assert!(projected.mappings.iter().all(|counts| {
            counts.historical_facts == 0
                && counts.aliases_added == 0
                && counts.aliases_already_present == 0
        }));
    }
}
