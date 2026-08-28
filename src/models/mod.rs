//! Model ports, indexed by what the model *is* (its lineage), not by the role
//! it plays in a deployment: `f5` (TTS), `gemma` (LLM + audio), `flux` (image
//! generation). Each is a graph of `mary::format` modules built with `mary::nn`.

pub mod f5;

// Qwen2.5-VL text backbone (BiQwen2_5 / nomic-embed-multimodal-7b). Reuses
// gemma's RoPE table, so it rides the `gemma` feature.
#[cfg(feature = "gemma")]
pub mod qwen2_5_vl;

#[cfg(feature = "flux")]
pub mod flux;

#[cfg(feature = "gemma")]
pub mod gemma;

#[cfg(feature = "smolvla")]
pub mod smolvla;

#[cfg(feature = "qwen3tts")]
pub mod qwen3tts;

// Mimi neural audio codec (PersonaPlex Phase 1). Reuses the qwen3tts CPU
// primitives (Accelerate sgemm + libm gelu), so it rides the same feature.
#[cfg(feature = "qwen3tts")]
pub mod personaplex;

#[cfg(feature = "voxtral")]
pub mod voxtral;

// Kimi-K3 (`kimi_linear`, 2.78 T MoE). Config + checkpoint-name->module-slot
// layout, the ported operators (SiTU, KDA, MLA, AttnRes, router, latent MoE)
// and the whole decoder layer that composes them.
#[cfg(feature = "k3")]
pub mod k3;

// Inkling (Thinking Machines, 975 B / 276 B MoE, natively multimodal). One
// feature, and it names CUDA: every lane below is a Blackwell tensor-core lane.
#[cfg(feature = "inkling-cuda")]
pub mod inkling;

// Inkling's speculation algebra — tree topology, top-b, the drafting schedule,
// the ancestor masks, the accept walk — is backend-free by construction, and
// the whole point of building it that way is that it can be TESTED where there
// is no GPU. `models::inkling` is gated on the CUDA lane, which no laptop can
// compile, so without that lane the module is re-formed around the one file
// that does not need it. Same path either way:
// `mary::models::inkling::spectree`.
#[cfg(not(feature = "inkling-cuda"))]
pub mod inkling {
    #[path = "spectree.rs"]
    pub mod spectree;

    #[path = "target.rs"]
    pub(crate) mod target;

    // The SERVING PROTOCOL is backend-free for the same reason and to greater
    // effect: the client of a serving process needs no GPU, because the GPU is
    // in the other process. That is the whole shape of the thing — a model in a
    // process that can only run where the model runs, reached from a loop that
    // must run anywhere.
    #[cfg(feature = "serve")]
    #[path = "serve.rs"]
    pub mod serve;

    // Native READY/TurnEnd evidence for the GPU-free Drive client. Keep this in
    // the same reconstructed module as `serve`, since `drive-mind` deliberately
    // does not pull the CUDA model implementation onto the host.
    #[cfg(feature = "drive-mind")]
    #[path = "telemetry.rs"]
    pub mod telemetry;
}
