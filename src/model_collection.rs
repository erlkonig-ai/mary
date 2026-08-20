//! Native collection persistence and compatibility projection for Mary models.
//!
//! New model fragments live in one fixed, append-only `SimpleArchive` union.
//! Publication takes an already-open [`Pile`] and a caller-supplied signing
//! key; exact reads take complete signed commit records as their authority.
//! The local-latest convenience deliberately makes a different, explicit
//! admission choice: every structurally present commit for this exact model
//! descriptor in one locally observed pile prefix is admitted to its frozen
//! ticket, regardless of author, and then subjected to the same strict exact
//! verification. There is no repository, mutable head, key discovery,
//! fallback store, repair, reopen, or implicit durability flush in this
//! surface.
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
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::{Blob, IntoBlob, TryFromBlob};
use triblespace::core::collection::records::{collection_name, collection_team};
use triblespace::core::inline::encodings::ed25519::ED25519PublicKey;
use triblespace::core::inline::encodings::shortstring::ShortString;
use triblespace::core::trible::TribleSet;
use triblespace::core::attribute::Attribute;
use triblespace::core::collection::simplearchive_union::{
    self, PreparationError, PreparedCollectionCommit, PublicationError,
};
use triblespace::core::collection::{
    CollectionCommit, CollectionMaterializationError, CollectionRecord, CollectionStore,
    SimpleArchiveCollection,
};
use triblespace::core::inline::encodings::UnknownInline;
use triblespace::core::repo::pile::{
    CollectionInsertError, FlushError, GetBlobError, InsertError as PileInsertError, PileReader,
    ReadError,
};
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
pub fn mary_model_graph_name() -> CollectionName {
    CollectionName::new("mary-model-graph")
        .expect("`mary-model-graph` is a legal collection name")
}

/// Concrete failure produced by publishing one model fragment to a pile.
pub type ModelFragmentPublicationError = PublicationError<PileInsertError, CollectionInsertError>;

/// Concrete failure produced while exactly materializing Mary's collection
/// from a pile.
pub type ModelCollectionMaterializationError =
    CollectionMaterializationError<ReadError, ReadError, Infallible, GetBlobError<Infallible>>;

/// Failure while freezing the locally admitted model commits from an already
/// open pile.
#[derive(Debug)]
pub enum SnapshotLocalModelCollectionError {
    /// Full native-record enumeration for the local ticket failed.
    LocalTicket(ReadError),
    /// The frozen exact ticket could not be verified or materialized.
    Materialize(ModelCollectionMaterializationError),
}

impl fmt::Display for SnapshotLocalModelCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalTicket(source) => {
                write!(
                    f,
                    "failed to freeze the local model commit ticket: {source}"
                )
            }
            Self::Materialize(source) => {
                write!(f, "failed to materialize the model collection: {source}")
            }
        }
    }
}

impl Error for SnapshotLocalModelCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LocalTicket(source) => Some(source),
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
    /// Full native-record enumeration for a local-admission ticket failed.
    LocalTicket(ReadError),
    /// The frozen exact ticket could not be verified or materialized.
    Materialize(ModelCollectionMaterializationError),
    /// The read-only pile handle could not be closed after snapshot creation.
    Close(FlushError),
}

impl fmt::Display for LoadModelCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(source) => write!(f, "failed to open model pile: {source}"),
            Self::Refresh(source) => write!(f, "failed to refresh model pile: {source}"),
            Self::LocalTicket(source) => {
                write!(
                    f,
                    "failed to freeze the local model commit ticket: {source}"
                )
            }
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
            Self::LocalTicket(source) => Some(source),
            Self::Materialize(source) => Some(source),
            Self::Close(source) => Some(source),
        }
    }
}

fn model_graph_collection(team: VerifyingKey) -> SimpleArchiveCollection {
    SimpleArchiveCollection::new(mary_model_graph_name(), team)
}

/// Prepare one canonical model commit entirely in memory.
///
/// Preparation validates the descriptor, archives, signature, and every blob
/// embedded in `fragment`, but touches no destination storage. A commit-last
/// importer can stage the returned value's dependencies, validate them through
/// the destination's own reader, and expose authority only by calling
/// `finalize` after those gates succeed.
pub fn prepare_model_fragment(
    team: VerifyingKey,
    signing_key: &SigningKey,
    fragment: Fragment,
) -> Result<PreparedCollectionCommit, PreparationError> {
    simplearchive_union::prepare_fragment_commit(
        &model_graph_collection(team).descriptor(),
        fragment,
        signing_key,
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

/// Materialize exactly the supplied set of complete signed model commits from
/// an already-open pile.
///
/// Ticket members may have different authors. Commits not named by the ticket
/// remain inert, while every selected record and dependency must pass the
/// strict `SimpleArchiveCollection` exact-ticket checks. This function does
/// not flush or close the caller's pile.
pub fn snapshot_model_collection_exact(
    pile: &mut Pile,
    team: VerifyingKey,
    ticket: &[CollectionCommit],
) -> Result<CollectionSnapshot<PileReader>, ModelCollectionMaterializationError> {
    model_graph_collection(team).snapshot_exact(pile, ticket)
}

fn close_after_snapshot(
    pile: Pile,
    snapshot: Result<CollectionSnapshot<PileReader>, ModelCollectionMaterializationError>,
) -> Result<CollectionSnapshot<PileReader>, LoadModelCollectionError> {
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

/// Open `path`, materialize the caller-supplied exact ticket, and close the
/// pile while returning the owned reader snapshot.
///
/// Opening and the initial replay are explicit failure stages. No missing
/// file is created, no damaged tail is amputated, and no alternate storage or
/// runtime path is consulted. The returned [`PileReader`] owns its immutable
/// mapping snapshot and remains usable after the mutable [`Pile`] is closed.
pub fn load_model_collection_from_ticket(
    path: impl AsRef<Path>,
    team: VerifyingKey,
    ticket: &[CollectionCommit],
) -> Result<CollectionSnapshot<PileReader>, LoadModelCollectionError> {
    let mut pile = open_and_refresh_model_pile(path.as_ref())?;
    let snapshot = snapshot_model_collection_exact(&mut pile, team, ticket);
    close_after_snapshot(pile, snapshot)
}

/// Which teams publish a `mary-model-graph` collection in this pile.
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
pub fn model_graph_teams(pile: &mut Pile) -> Result<Vec<VerifyingKey>, ReadError> {
    let wanted = mary_model_graph_name();
    let mut seen = BTreeSet::new();
    let mut teams = Vec::new();
    let mut descriptors = BTreeSet::new();
    for record in pile.records()? {
        if let CollectionRecord::Commit(commit) = record? {
            descriptors.insert(commit.collection());
        }
    }
    let reader = pile.reader()?;
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
        let mut name = None;
        let mut team = None;
        for fact in facts.iter() {
            if *fact.a() == collection_name.id() {
                name = fact.v::<ShortString>().try_from_inline::<String>().ok();
            } else if *fact.a() == collection_team.id() {
                team = VerifyingKey::from_bytes(&fact.v::<ED25519PublicKey>().raw).ok();
            }
        }
        if name.as_deref() == Some(wanted.as_str()) {
            if let Some(team) = team {
                if seen.insert(team.to_bytes()) {
                    teams.push(team);
                }
            }
        }
    }
    Ok(teams)
}

/// The single team publishing a model graph here, or an error naming the
/// ambiguity.
///
/// The convenience form of [`model_graph_teams`] for the ordinary pile, which
/// has exactly one. It refuses rather than guesses in both directions: no
/// model graph at all, and more than one, are different failures and say so.
pub fn sole_model_graph_team(pile: &mut Pile) -> Result<VerifyingKey, SoleModelGraphTeamError> {
    let teams = model_graph_teams(pile).map_err(SoleModelGraphTeamError::Read)?;
    match teams.len() {
        1 => Ok(teams[0]),
        0 => Err(SoleModelGraphTeamError::None),
        _ => Err(SoleModelGraphTeamError::Several {
            teams: teams.iter().map(|team| team.to_bytes()).collect(),
        }),
    }
}

/// The single team publishing a model graph in the pile at `path`.
///
/// The path-taking form, for the common caller that holds a path and nothing
/// else. It opens, asks, and closes; nothing is left open on either outcome.
pub fn model_graph_team_at(path: impl AsRef<Path>) -> Result<VerifyingKey, SoleModelGraphTeamError> {
    let mut pile = open_and_refresh_model_pile(path.as_ref())
        .map_err(|source| SoleModelGraphTeamError::Open(Box::new(source)))?;
    let team = sole_model_graph_team(&mut pile);
    let _ = pile.close();
    team
}

/// Why a pile does not have exactly one model-graph team.
#[derive(Debug)]
pub enum SoleModelGraphTeamError {
    /// The pile could not be opened.
    Open(Box<LoadModelCollectionError>),
    /// The pile could not be read.
    Read(ReadError),
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
            Self::Open(source) => write!(f, "open the pile: {source}"),
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
            Self::Open(source) => Some(source.as_ref()),
            Self::Read(source) => Some(source),
            Self::None | Self::Several { .. } => None,
        }
    }
}

fn local_model_ticket(pile: &mut Pile, team: VerifyingKey) -> Result<Vec<CollectionCommit>, ReadError> {
    // Hashing a descriptor without storing it is only safe on a read path.
    // A write that did this could name a collection whose descriptor is not
    // in the pile -- records referencing something nothing can decode -- so
    // there is deliberately no helper for it and the write paths take the
    // handle that `put` hands back instead.
    let collection = IntoBlob::<SimpleArchive>::to_blob(
        model_graph_collection(team).descriptor().facts().clone(),
    )
    .get_handle();
    let mut ticket = Vec::new();
    for record in pile.records()? {
        if let CollectionRecord::Commit(commit) = record? {
            if commit.collection() == collection {
                // Deliberately retain structurally decoded but
                // cryptographically invalid matching commits. The exact
                // boundary below must reject them instead of silently
                // producing a partial local view.
                ticket.push(commit);
            }
        }
    }
    ticket.sort_unstable_by_key(CollectionCommit::id);
    Ok(ticket)
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
) -> Result<CollectionSnapshot<PileReader>, SnapshotLocalModelCollectionError> {
    let ticket =
        local_model_ticket(pile, team).map_err(SnapshotLocalModelCollectionError::LocalTicket)?;
    snapshot_model_collection_exact(pile, team, &ticket)
        .map_err(SnapshotLocalModelCollectionError::Materialize)
}

/// Load the union of every locally admitted model commit in one observed pile
/// prefix.
///
/// This is an explicit *local pile admission policy*: placing a structurally
/// valid native commit record in this pile admits it to the next frozen
/// ticket, regardless of signer. Records naming other descriptors are inert.
/// Matching records are not pre-filtered by signature, so one invalid matching
/// record makes exact verification fail closed rather than disappearing from a
/// partial result. After the full deterministic record scan, the ticket is
/// frozen and materialized exactly; concurrent later appends cannot widen it.
/// The pile is never repaired, reopened, or implicitly flushed.
pub fn load_model_collection_local_latest(
    path: impl AsRef<Path>,
    team: VerifyingKey,
) -> Result<CollectionSnapshot<PileReader>, LoadModelCollectionError> {
    // `Pile::records` performs the one bounded replay that defines the local
    // admission prefix. Do not refresh separately here: a second pre-ticket
    // replay would move that boundary for no semantic benefit.
    let mut pile = Pile::open(path.as_ref()).map_err(LoadModelCollectionError::Open)?;
    let snapshot = match snapshot_model_collection_local_latest(&mut pile, team) {
        Ok(snapshot) => Ok(snapshot),
        Err(SnapshotLocalModelCollectionError::LocalTicket(source)) => {
            let _ = pile.close();
            return Err(LoadModelCollectionError::LocalTicket(source));
        }
        Err(SnapshotLocalModelCollectionError::Materialize(source)) => Err(source),
    };
    close_after_snapshot(pile, snapshot)
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
    use triblespace::prelude::blobencodings::{LongString, RawBytes};
    use triblespace::prelude::inlineencodings::{Boolean, GenId, Handle, ShortString, F64, U256BE};
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
        "09EA2F7BCF9B0C9714EE39CF269DF2D5" unsafe as safetensor_path: Handle<LongString>;
        "33CE12B1B940B13E48D8E5B0ADFD2421" unsafe as index: U256BE;
        "3F46CDE630964D78D62DA32F4A8558C1" unsafe as model_root: GenId;
        "B4B6EC08A0CD70DE63A690168EE78F0F" unsafe as member: GenId;
        "4C1CD1611863E7854C59C7DC706DF77A" unsafe as model_name: Handle<LongString>;
        "D20B8E3556C35FF6D18D104C3443D6CF" unsafe as source: Handle<LongString>;
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
        "AE7FE29F2F38153F58C542D5CA4A9356" unsafe as piece: Handle<LongString>;
        "F0E2E782F7BB62F52B1186DDE0EB5388" unsafe as token_id: U256BE;
        "714AE13F801202EB27C83E3AB2290669" unsafe as piece_bytes: Handle<RawBytes>;
        "5723ECE1FF426C58879B79D5669A7CF1" unsafe as merge_left: Handle<LongString>;
        "5C78FEB151F35A2C5D07BEC92E860752" unsafe as merge_right: Handle<LongString>;
        "68F1A9E6ED735E7C3ADCCA076AFF1742" unsafe as unk_token: ShortString;
        "11F76A2C0856C16CB030C4327D5A3B93" unsafe as continuing_subword_prefix: ShortString;
        "6FB969E8A3EDD1A657C721DD5A7D42EA" unsafe as end_of_word_suffix: ShortString;
        "DF3F88DBFA2B44A7783169C9640014AF" unsafe as max_input_chars: U256BE;
        "3BCB70478942DB710ED2A4FB023F3457" unsafe as piece_score: F64;
        "EE4C6647619A836326196F0DBF84FA98" unsafe as byte_fallback: Boolean;
        "C8262D5668B8A1F541B3C35D54201BEC" unsafe as pattern: Handle<LongString>;
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
    use triblespace::core::collection::ExactTicketError;
    use triblespace::core::repo::BlobStoreGet;
    use triblespace::macros::id_hex;
    use triblespace::prelude::blobencodings::{LongString, RawBytes, SimpleArchive};
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
        Inline<Handle<LongString>>,
        Inline<Handle<RawBytes>>,
    ) {
        let text: Blob<LongString> = format!("model attachment {label}").to_blob();
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
        SigningKey::from_bytes(&[0x11; 32]).verifying_key()
    }

    fn open_test_pile(path: &Path) -> Pile {
        let mut pile = Pile::open(path).expect("open test pile");
        pile.refresh().expect("refresh test pile");
        pile
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

        assert_eq!(mary_model_graph_name().as_str(), "mary-model-graph");
        assert_eq!(collection.name(), &mary_model_graph_name());
        assert_eq!(collection.team(), team);

        let handle = IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle();
        assert_eq!(
            handle.raw,
            [
                0x75, 0x8B, 0xBA, 0x4A, 0x8C, 0x01, 0x2B, 0x3B,
                0xD9, 0xFB, 0xAF, 0xCA, 0x42, 0x0B, 0xA2, 0xD0,
                0x04, 0xED, 0x7D, 0x7B, 0xAC, 0x3C, 0x23, 0x93,
                0xBF, 0x32, 0x32, 0x7E, 0x32, 0x95, 0xFC, 0xDF,
            ]
        );
    }

    #[test]
    fn fragment_publication_roundtrips_every_channel_and_is_idempotent() {
        let file = TempPilePath::new("fragment-roundtrip");
        let signing_key = SigningKey::from_bytes(&[0x17; 32]);
        let (fragment, text_handle, payload_handle) = fragment_fixture("roundtrip");
        let expected_facts = fragment.facts().clone();
        let expected_metafacts = fragment.metafacts().clone();

        let mut pile = open_test_pile(file.as_path());
        let first = publish_model_fragment(&mut pile, test_team(), &signing_key, fragment.clone()).unwrap();
        let repeated = publish_model_fragment(&mut pile, test_team(), &signing_key, fragment).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(
            first.public_key().raw,
            signing_key.verifying_key().to_bytes()
        );
        first.verify_strict().unwrap();

        // Snapshot directly from the same still-open pile. A duplicate ticket
        // is a mathematical set and therefore returns one canonical commit.
        let snapshot = snapshot_model_collection_exact(&mut pile, test_team(), &[repeated, first]).unwrap();
        assert_eq!(snapshot.facts(), &expected_facts);
        assert_eq!(snapshot.commits(), &[first]);

        // The owned PileReader mapping must outlive the mutable pile handle.
        pile.close().unwrap();
        let metadata: TribleSet = snapshot.reader().get(first.metadata()).unwrap();
        assert_eq!(metadata, expected_metafacts);
        let text: View<str> = snapshot.reader().get(text_handle).unwrap();
        let payload: Bytes = snapshot.reader().get(payload_handle).unwrap();
        assert_eq!(&*text, "model attachment roundtrip");
        assert_eq!(&*payload, b"roundtrip");

        let loaded = load_model_collection_from_ticket(file.as_path(), test_team(), &[first]).unwrap();
        assert_eq!(loaded.facts(), &expected_facts);
        let text_after_path_close: View<str> = loaded.reader().get(text_handle).unwrap();
        assert_eq!(&*text_after_path_close, "model attachment roundtrip");
    }

    #[test]
    fn selected_model_index_owns_the_reader_after_snapshot_consumption() {
        let file = TempPilePath::new("selected-model-index");
        let signing_key = SigningKey::from_bytes(&[0x19; 32]);
        let mut pile = open_test_pile(file.as_path());

        let leaf = crate::format::put_raw(&mut pile, &[1.25], &[1]).unwrap();
        let leaf_id = leaf.root().expect("tensor leaf root");
        let mut facts = leaf.into_facts();
        let name = pile
            .put::<LongString, _>("encoder.weight".to_owned())
            .unwrap();
        let member = entity! { _ @
            crate::format::attrs::safetensor_path: name,
            crate::format::attrs::weight: leaf_id,
        };
        let member_id = member.root().expect("model member root");
        facts += member.into_facts();
        let source = pile
            .put::<LongString, _>("example/owned-index".to_owned())
            .unwrap();
        let model = entity! { _ @
            crate::format::attrs::source: source,
            crate::format::attrs::quantization: "native",
            crate::format::attrs::member: member_id,
        };
        let model_root = model.root().expect("model root");
        facts += model.into_facts();

        let commit =
            publish_model_fragment(&mut pile, test_team(), &signing_key, Fragment::rooted(model_root, facts))
                .unwrap();
        let snapshot = snapshot_model_collection_exact(&mut pile, test_team(), &[commit]).unwrap();
        pile.close().unwrap();

        let selected = crate::selection::SelectedModelIndex::from_snapshot(
            snapshot,
            crate::selection::ModelSelector::Source {
                source: "example/owned-index",
                quantization: "native",
            },
        )
        .unwrap();
        assert_eq!(selected.root(), model_root);
        let handles = selected.handles()["encoder.weight"];
        let crate::ingest::LeafHandles::F32(data, shape) = handles else {
            panic!("selected model did not preserve the f32 leaf");
        };
        let data: View<[f32]> = selected.reader().get(data).unwrap();
        let shape: View<[u64]> = selected.reader().get(shape).unwrap();
        assert_eq!(&*data, &[1.25]);
        assert_eq!(&*shape, &[1]);
    }

    #[test]
    fn exact_ticket_accepts_mixed_authors_and_keeps_unselected_commits_inert() {
        let file = TempPilePath::new("exact-mixed-authors");
        let mut pile = open_test_pile(file.as_path());
        let (first_fragment, _, _) = fragment_fixture("first");
        let first_facts = first_fragment.facts().clone();
        let first = publish_model_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x21; 32]),
            first_fragment,
        )
        .unwrap();
        let (second_fragment, _, _) = fragment_fixture("second");
        let second_facts = second_fragment.facts().clone();
        let second = publish_model_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x22; 32]),
            second_fragment,
        )
        .unwrap();
        let (unselected, _, _) = fragment_fixture("unselected");
        publish_model_fragment(&mut pile, test_team(), &SigningKey::from_bytes(&[0x23; 32]), unselected)
            .unwrap();

        let snapshot = snapshot_model_collection_exact(&mut pile, test_team(), &[second, first]).unwrap();
        let mut expected = first_facts;
        expected += second_facts;
        let mut expected_commits = vec![first, second];
        expected_commits.sort_unstable_by_key(CollectionCommit::id);
        assert_eq!(snapshot.facts(), &expected);
        assert_eq!(snapshot.commits(), expected_commits);
        assert_ne!(first.public_key(), second.public_key());
        pile.close().unwrap();
    }

    #[test]
    fn local_latest_admits_all_matching_authors_and_ignores_foreign_records() {
        let file = TempPilePath::new("local-latest");
        let mut pile = open_test_pile(file.as_path());
        let (first_fragment, _, _) = fragment_fixture("local-first");
        let first_facts = first_fragment.facts().clone();
        let first = publish_model_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x31; 32]),
            first_fragment,
        )
        .unwrap();
        let (second_fragment, _, _) = fragment_fixture("local-second");
        let second_facts = second_fragment.facts().clone();
        let second = publish_model_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x32; 32]),
            second_fragment,
        )
        .unwrap();

        // "Foreign" now means a different name under the same team, which is
        // the shape a real unrelated collection takes.
        let foreign_name = CollectionName::new("not-the-model-graph").unwrap();
        let foreign_descriptor = simplearchive_union::descriptor(&foreign_name, test_team());
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
        assert_eq!(in_place.commits(), expected_commits);
        pile.close().unwrap();

        let snapshot = load_model_collection_local_latest(file.as_path(), test_team()).unwrap();
        assert_eq!(snapshot.facts(), &expected);
        assert_eq!(snapshot.commits(), expected_commits);
        assert!(snapshot
            .commits()
            .iter()
            .all(|commit| commit.collection() != foreign.collection()));
    }

    #[test]
    fn local_latest_retains_invalid_matching_commits_so_exact_read_fails() {
        let file = TempPilePath::new("local-invalid-matching");
        let mut pile = open_test_pile(file.as_path());
        let (fragment, _, _) = fragment_fixture("invalid-matching");
        let valid =
            publish_model_fragment(&mut pile, test_team(), &SigningKey::from_bytes(&[0x41; 32]), fragment)
                .unwrap();
        let mut invalid_bytes = valid.to_bytes();
        *invalid_bytes.last_mut().unwrap() ^= 1;
        let invalid = CollectionCommit::from_bytes(invalid_bytes);
        assert_eq!(invalid.collection(), valid.collection());
        assert!(invalid.verify_strict().is_err());
        pile.insert(CollectionRecord::Commit(invalid)).unwrap();
        pile.close().unwrap();

        let error = match load_model_collection_local_latest(file.as_path(), test_team()) {
            Ok(_) => panic!("invalid matching commit unexpectedly materialized"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            LoadModelCollectionError::Materialize(
                CollectionMaterializationError::ExactTicket(
                    ExactTicketError::MissingOrInvalidCommit { commit }
                )
            ) if commit == invalid.id()
        ));
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
        assert!(matches!(
            error,
            LoadModelCollectionError::LocalTicket(ReadError::CorruptPile { valid_length: 0 })
        ));
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
        assert!(mappings
            .iter()
            .all(|mapping| mapping.historical != mapping.canonical));

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
