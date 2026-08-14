//! Compatibility projection for Mary's pre-anchored model graphs.
//!
//! TribleSpace commit `6b65f278` changed `"hex" as attribute: Encoding` from a
//! literal attribute id to an encoding-aware id derived from `(hex, Encoding)`.
//! Model piles written before that epoch contain the literal ids. This module
//! preserves those facts and adds their canonical anchored-attribute aliases;
//! it does not choose a repository/collection migration policy or alter runtime
//! query declarations.

use std::collections::HashMap;

use triblespace::core::attribute::Attribute;
use triblespace::core::inline::encodings::UnknownInline;
use triblespace::prelude::inlineencodings::{F64, U256BE};
use triblespace::prelude::*;

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

    use super::*;
    use triblespace::macros::id_hex;

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
